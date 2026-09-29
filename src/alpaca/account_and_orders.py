import asyncio
from datetime import datetime

from mcp import ClientSession

from src.alpaca.mcp_client import MCPToolError, call_tool, snapshot_for_symbol

# Order states Alpaca will not move out of on its own.
_TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "done_for_day"}

_FILL_POLL_SECONDS = 1.0
_FILL_TIMEOUT_SECONDS = 30.0


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def minutes_to_close(clock: dict) -> float:
    """Minutes until the session ends. 0 if closed or the clock is unusable (the safe side)."""
    if not clock.get("is_open", False):
        return 0.0
    try:
        now = datetime.fromisoformat(clock["timestamp"])
        next_close = datetime.fromisoformat(clock["next_close"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    return max((next_close - now).total_seconds() / 60, 0.0)


def _daily_pnl_pct(account: dict) -> float:
    """Today's account return: equity now vs yesterday's close (Alpaca has no ready-made field)."""
    last_equity = _as_float(account.get("last_equity"))
    if last_equity <= 0:
        return 0.0
    return (_as_float(account.get("equity")) - last_equity) / last_equity


async def get_account_state(client: ClientSession, symbols: list[str]) -> dict:
    """
    Cash, holdings, open orders and prices straight from Alpaca (the source of
    truth). Prices cover the requested symbols plus everything held, so a
    leftover position is never priced at zero.
    """
    account = await call_tool(client, "get_account_info", {})
    positions_raw = await call_tool(client, "get_all_positions", {})
    orders_raw = await call_tool(client, "get_orders", {"status": "open"})
    clock = await call_tool(client, "get_clock", {})

    # Both list-shaped tools return their list under "result".
    positions = {
        p["symbol"]: {
            "qty": _as_float(p.get("qty")),
            "avg_price": _as_float(p.get("avg_entry_price")),
        }
        for p in positions_raw.get("result", [])
    }

    price_symbols = sorted(set(symbols) | set(positions))
    last_prices = await _fetch_last_prices(client, price_symbols)

    return {
        "cash": _as_float(account.get("cash")),
        "equity": _as_float(account.get("equity")),
        "daily_pnl_pct": _daily_pnl_pct(account),
        "positions": positions,
        "open_orders": orders_raw.get("result", []),
        "last_prices": last_prices,
        "clock": clock,
        "minutes_to_close": minutes_to_close(clock),
    }


async def _fetch_last_prices(client: ClientSession, symbols: list[str]) -> dict:
    if not symbols:
        return {}
    # The param is "symbols" (plural) even for one ticker.
    snapshots = await asyncio.gather(
        *(call_tool(client, "get_stock_snapshot", {"symbols": s}) for s in symbols),
        return_exceptions=True,
    )
    prices = {
        symbol: _price_of(snapshot, symbol)
        for symbol, snapshot in zip(symbols, snapshots, strict=True)
        if not isinstance(snapshot, Exception)
    }
    return {symbol: price for symbol, price in prices.items() if price > 0}


def _price_of(snapshot: dict, symbol: str, default: float = 0.0) -> float:
    """The latest trade price from a stock snapshot."""
    return _as_float(snapshot_for_symbol(snapshot, symbol).get("latestTrade", {}).get("p"), default)


_SIDE_BY_ACTION = {"BUY": "buy", "SELL": "sell"}


async def _place_market_order(client: ClientSession, symbol: str, side: str, qty) -> dict:
    return await call_tool(
        client,
        "place_stock_order",
        {"symbol": symbol, "side": side, "qty": str(qty), "type": "market", "time_in_force": "day"},
    )


async def execute_order(client: ClientSession, decision: dict) -> dict:
    """Places a stock-only market order and waits for it to reach a final state."""
    side = _SIDE_BY_ACTION.get(decision["action"])
    if side is None:
        raise ValueError(f"execute_order got non-executable action {decision['action']!r}")

    qty = decision.get("target_qty")
    if not isinstance(qty, int) or qty <= 0:
        raise ValueError(f"execute_order got invalid target_qty {qty!r} for {decision['symbol']}")

    result = await _place_market_order(client, decision["symbol"], side, qty)
    fill = await wait_for_fill(client, _order_id(result))
    return {"decision": decision, "order_result": result, "fill": fill}


def _order_id(order_result: dict) -> str | None:
    for key in ("id", "order_id"):
        value = order_result.get(key)
        if value:
            return str(value)
    nested = order_result.get("order")
    if isinstance(nested, dict):
        return _order_id(nested)
    return None


async def wait_for_fill(client: ClientSession, order_id: str | None) -> dict:
    """
    Polls until the order reaches a final state or times out. A timeout is
    reported, not raised: the order still exists at the broker.
    """
    if not order_id:
        return {"status": "unknown", "reason": "order id missing from place_stock_order response"}

    deadline = asyncio.get_running_loop().time() + _FILL_TIMEOUT_SECONDS
    last_status = "unknown"
    while asyncio.get_running_loop().time() < deadline:
        try:
            order = await call_tool(client, "get_order_by_id", {"order_id": order_id})
        except MCPToolError as exc:
            return {"status": "unknown", "reason": f"could not poll order status: {exc}"}

        last_status = str(order.get("status", "unknown")).lower()
        if last_status in _TERMINAL_STATUSES:
            return {
                "status": last_status,
                "filled_qty": _as_float(order.get("filled_qty")),
                "filled_avg_price": _as_float(order.get("filled_avg_price")),
            }
        await asyncio.sleep(_FILL_POLL_SECONDS)

    return {"status": "timeout", "last_status": last_status}


def reconcile(expected_positions: dict, account: dict) -> list[dict]:
    """Compares the positions we expect after this cycle's fills with Alpaca's actual ones."""
    actual = account.get("positions", {})
    divergences = []
    for symbol in sorted(set(expected_positions) | set(actual)):
        want = _as_float(expected_positions.get(symbol, {}).get("qty"))
        have = _as_float(actual.get(symbol, {}).get("qty"))
        if abs(want - have) > 1e-6:
            divergences.append({"symbol": symbol, "expected_qty": want, "actual_qty": have})
    return divergences


async def _last_price(client: ClientSession, symbol: str, fallback: float) -> float:
    try:
        snapshot = await call_tool(client, "get_stock_snapshot", {"symbols": symbol})
    except MCPToolError:
        return fallback
    return _price_of(snapshot, symbol, fallback)


async def liquidate_all(client: ClientSession) -> dict:
    """Closes every open position regardless of signals, and records each one's P&L."""
    positions_raw = await call_tool(client, "get_all_positions", {})
    results, closed_trades, failures = [], [], []

    for p in positions_raw.get("result", []):
        signed_qty = _as_float(p.get("qty"))
        if signed_qty == 0:
            continue

        symbol = p["symbol"]
        avg_price = _as_float(p.get("avg_entry_price"))
        side = "sell" if signed_qty > 0 else "buy"

        last_price = await _last_price(client, symbol, fallback=avg_price)

        try:
            result = await _place_market_order(client, symbol, side, abs(signed_qty))
        except MCPToolError as exc:
            # One symbol failing to close must not stop the rest.
            failures.append({"symbol": symbol, "error": str(exc)})
            continue

        fill = await wait_for_fill(client, _order_id(result))
        results.append({"symbol": symbol, "order_result": result, "fill": fill})
        fill_price = fill.get("filled_avg_price") or last_price
        closed_trades.append(
            {
                "symbol": symbol,
                "action": side.upper(),
                "qty": abs(signed_qty),
                "pnl": (fill_price - avg_price) * signed_qty,
            }
        )

    return {"liquidation_orders": results, "closed_trades": closed_trades, "failures": failures}
