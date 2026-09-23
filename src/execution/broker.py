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
    """
    Alpaca clock: {is_open, timestamp, next_open, next_close}. Returns minutes
    until the session ends; 0 if the market is already closed or the clock is
    unusable (the safe direction - it stops new BUYs and triggers flattening).
    """
    if not clock.get("is_open", False):
        return 0.0
    try:
        now = datetime.fromisoformat(str(clock["timestamp"]).replace("Z", "+00:00"))
        next_close = datetime.fromisoformat(str(clock["next_close"]).replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return 0.0
    return max((next_close - now).total_seconds() / 60, 0.0)


def _daily_pnl_pct(account: dict) -> float:
    """
    Alpaca's account object has no `daily_pnl_pct` field. Reading one meant the
    value was always 0.0, so the daily-loss limit could never fire. Derive it
    from equity against the previous close's equity, which Alpaca does provide.
    """
    if "daily_pnl_pct" in account:
        return _as_float(account.get("daily_pnl_pct"))
    equity = _as_float(account.get("equity"), _as_float(account.get("portfolio_value")))
    last_equity = _as_float(account.get("last_equity"))
    if last_equity <= 0:
        return 0.0
    return (equity - last_equity) / last_equity


async def get_account_state(client: ClientSession, symbols: list[str]) -> dict:
    """
    Reconciles cash/holdings/orders against Alpaca's own record - the
    source of truth, never locally-tracked state.

    Prices are fetched for the union of the requested symbols and everything
    currently held. Scoping them to the current candidate list alone meant a
    position left over from a previous strategy had no price, so its P&L
    silently evaluated to exactly zero.
    """
    # Real tool names differ from the obvious guess: get_account_info (not
    # get_account) and get_all_positions (not get_positions), confirmed
    # against the live server and https://docs.alpaca.markets/us/docs/alpaca-mcp-server.
    account = await call_tool(client, "get_account_info", {})
    positions_raw = await call_tool(client, "get_all_positions", {})
    orders_raw = await call_tool(client, "get_orders", {"status": "open"})
    clock = await call_tool(client, "get_clock", {})

    # Both list-shaped tools wrap their list under "result", not "positions"/
    # "orders".
    positions = {
        p["symbol"]: {
            "qty": _as_float(p.get("qty")),
            "avg_price": _as_float(p.get("avg_entry_price")),
        }
        for p in positions_raw.get("result", [])
    }

    price_symbols = sorted(set(symbols) | set(positions))
    last_prices = await _fetch_last_prices(client, price_symbols)

    cash = _as_float(account.get("cash"))
    return {
        "cash": cash,
        "portfolio_value": _as_float(account.get("portfolio_value"), cash),
        "equity": _as_float(account.get("equity"), _as_float(account.get("portfolio_value"), cash)),
        "daily_pnl_pct": _daily_pnl_pct(account),
        "positions": positions,
        "open_orders": orders_raw.get("result", []),
        "last_prices": last_prices,
        "clock": clock,
        # Carried on the snapshot so every consumer reads one value derived at
        # fetch time, rather than passing a separately-computed (and possibly
        # unset) copy around.
        "minutes_to_close": minutes_to_close(clock),
    }


async def _fetch_last_prices(client: ClientSession, symbols: list[str]) -> dict:
    if not symbols:
        return {}
    # The param is "symbols" (plural) even for a single ticker; "symbol"
    # gets a 400 from the underlying Alpaca API.
    snapshots = await asyncio.gather(
        *(call_tool(client, "get_stock_snapshot", {"symbols": s}) for s in symbols),
        return_exceptions=True,
    )
    prices = {}
    for symbol, snapshot in zip(symbols, snapshots, strict=True):
        if isinstance(snapshot, Exception):
            continue
        price = _as_float(snapshot_for_symbol(snapshot, symbol).get("latestTrade", {}).get("p"))
        if price > 0:
            prices[symbol] = price
    return prices


_SIDE_BY_ACTION = {"BUY": "buy", "SELL": "sell"}


async def execute_order(client: ClientSession, decision: dict) -> dict:
    """
    Places a stock-only market order and waits for it to reach a terminal
    state. No options, no multi-leg, no margin.

    The side is looked up explicitly: the old `"buy" if action == "BUY" else
    "sell"` turned any unexpected action string (including a lowercase "buy")
    into a SELL, silently inverting the trade.
    """
    action = decision["action"]
    side = _SIDE_BY_ACTION.get(action)
    if side is None:
        raise ValueError(f"execute_order got non-executable action {action!r}")

    qty = decision.get("target_qty")
    if not isinstance(qty, int) or qty <= 0:
        raise ValueError(f"execute_order got invalid target_qty {qty!r} for {decision['symbol']}")

    result = await call_tool(
        client,
        "place_stock_order",
        {
            "symbol": decision["symbol"],
            "side": side,
            "qty": str(qty),
            "type": "market",
            "time_in_force": "day",
        },
    )

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
    Polls until the order is terminal or the timeout elapses.

    Without this, the monitor read positions immediately after submitting a
    market order and could easily record pre-fill state. A timeout is reported,
    never raised: an unconfirmed order still exists at the broker and the
    caller needs the account refresh to find out what happened.
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

        payload = order.get("order") if isinstance(order.get("order"), dict) else order
        last_status = str(payload.get("status", "unknown")).lower()
        if last_status in _TERMINAL_STATUSES:
            return {
                "status": last_status,
                "filled_qty": _as_float(payload.get("filled_qty")),
                "filled_avg_price": _as_float(payload.get("filled_avg_price")),
            }
        await asyncio.sleep(_FILL_POLL_SECONDS)

    return {"status": "timeout", "last_status": last_status}


def reconcile(expected_positions: dict, account: dict) -> list[dict]:
    """
    Compares the positions we expect after this cycle's fills against what
    Alpaca actually reports, and returns the divergences.

    The monitor previously just re-fetched state and called that
    reconciliation; nothing ever compared expected to actual, so an order that
    silently failed to fill looked identical to one that worked.
    """
    actual = account.get("positions", {})
    divergences = []
    for symbol in sorted(set(expected_positions) | set(actual)):
        want = _as_float(expected_positions.get(symbol, {}).get("qty"))
        have = _as_float(actual.get(symbol, {}).get("qty"))
        # Fractional dust and rounding shouldn't read as a divergence.
        if abs(want - have) > 1e-6:
            divergences.append({"symbol": symbol, "expected_qty": want, "actual_qty": have})
    return divergences


async def liquidate_all(client: ClientSession) -> dict:
    """Finalization: close every open stock position regardless of signal
    state, guaranteeing zero exposure at horizon end. Also records realized
    P&L for each closed position, same as a normal SELL during the session,
    so the final report's trade history isn't missing the closing trades."""
    positions_raw = await call_tool(client, "get_all_positions", {})
    results = []
    closed_trades = []
    failures = []

    for p in positions_raw.get("result", []):
        signed_qty = _as_float(p.get("qty"))
        qty = abs(signed_qty)
        if qty == 0:
            continue

        symbol = p["symbol"]
        avg_price = _as_float(p.get("avg_entry_price"))
        side = "sell" if signed_qty > 0 else "buy"

        try:
            snapshot = await call_tool(client, "get_stock_snapshot", {"symbols": symbol})
            last_price = _as_float(
                snapshot_for_symbol(snapshot, symbol).get("latestTrade", {}).get("p"), avg_price
            )
        except MCPToolError:
            last_price = avg_price

        try:
            result = await call_tool(
                client,
                "place_stock_order",
                {
                    "symbol": symbol,
                    "side": side,
                    "qty": str(qty),
                    "type": "market",
                    "time_in_force": "day",
                },
            )
        except MCPToolError as exc:
            # One symbol failing to flatten must not abandon the rest.
            failures.append({"symbol": symbol, "error": str(exc)})
            continue

        fill = await wait_for_fill(client, _order_id(result))
        results.append({"symbol": symbol, "order_result": result, "fill": fill})

        fill_price = fill.get("filled_avg_price") or last_price
        closed_trades.append(
            {
                "symbol": symbol,
                "action": side.upper(),
                "qty": qty,
                "pnl": (fill_price - avg_price) * signed_qty,
            }
        )

    return {
        "liquidation_orders": results,
        "closed_trades": closed_trades,
        "failures": failures,
    }
