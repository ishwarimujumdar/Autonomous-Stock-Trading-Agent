import asyncio

from mcp import ClientSession

from src.alpaca.account_and_orders import (
    execute_order,
    get_account_state,
    liquidate_all,
    reconcile,
)
from src.alpaca.mcp_client import MCPToolError
from src.analysis.technical import compute_technical, has_usable_evidence
from src.concurrency import gather_limited
from src.config import CYCLE_MINUTES, REPICK_MINUTES
from src.decision.decision import decide
from src.evaluation.pnl import compute_performance
from src.graph.state import TradingState
from src.persistence.journal import log_event
from src.picker.stock_picker import pick_stocks
from src.picker.universe import universe_table
from src.risk.risk_gate import evaluate_batch

# --- 1. Pick Stocks ----------------------------------------------------------


async def n_pick_stocks(client: ClientSession, state: TradingState) -> dict:
    """LLM chooses which stocks to watch. Runs at the start, then again every hour."""
    # Refreshed here, not read from state: on an hourly re-pick this node runs
    # straight after Log + Wait's sleep, and state["account"] still holds the
    # reading from before that sleep - stale by up to one cycle.
    account = await get_account_state(client, state.get("watchlist", []))
    table = await universe_table(client)
    watchlist, error = await pick_stocks(table, state["run_config"].get("universe_hint", ""))
    minutes_left = account.get("minutes_to_close", 0.0)

    if watchlist is None:
        keeping = bool(state.get("watchlist"))
        log_event(state["run_id"], "pick_failed", {"error": error, "kept_previous": keeping})
        # A failed hourly re-pick keeps the current list; with no list at all we must stop.
        if keeping:
            return {"account": account, "picked_at_minutes_left": minutes_left}
        return {"account": account, "pick_failed": True, "session_done": True}

    log_event(
        state["run_id"],
        "stocks_picked",
        {"symbols": watchlist.symbols, "reason": watchlist.reason},
    )
    return {
        "account": account,
        "watchlist": list(watchlist.symbols),
        "picked_at_minutes_left": minutes_left,
    }


# --- 2. Market Data ----------------------------------------------------------


async def _measure(client: ClientSession, symbol: str) -> tuple[str, dict]:
    try:
        return symbol, await compute_technical(client, symbol)
    except MCPToolError as exc:
        return symbol, {"symbol": symbol, "error": str(exc)}


async def n_market_data(client: ClientSession, state: TradingState) -> dict:
    """Fresh account, prices and measurements for the watchlist plus anything we hold."""
    account = await get_account_state(client, state["watchlist"])
    # A held stock stays under review even if it fell off the watchlist.
    candidates = list(dict.fromkeys([*state["watchlist"], *account["positions"]]))
    measurements = dict(await gather_limited(_measure(client, s) for s in candidates))
    log_event(
        state["run_id"],
        "market_data",
        {"candidates": candidates, "prices": account["last_prices"]},
    )
    return {"account": account, "candidates": candidates, "measurements": measurements}


# --- 3. Trading Agent --------------------------------------------------------


async def n_trading_agent(state: TradingState) -> dict:
    """The LLM decides BUY / SELL / HOLD for each stock, one call per stock."""
    performance = state.get("performance", {})
    account = state["account"]
    recent_trades = performance.get("recent_trades", [])
    measurements = state["measurements"]

    # Always review what we hold; skip others with no usable measurements.
    to_decide = [
        s for s, m in measurements.items() if s in account["positions"] or has_usable_evidence(m)
    ]
    skipped = [s for s in measurements if s not in to_decide]

    decisions = await gather_limited(
        decide(
            symbol=symbol,
            technical=measurements[symbol],
            previous_outcomes=[t for t in recent_trades if t.get("symbol") == symbol],
            performance=performance,
            account=account,
            objective_constraints=state["objective_constraints"],
        )
        for symbol in to_decide
    )
    log_event(
        state["run_id"], "decisions_made", {"decisions": decisions, "skipped_no_evidence": skipped}
    )
    return {"decisions": decisions}


# --- 4. Risk Gate ------------------------------------------------------------


async def n_risk_gate(state: TradingState) -> dict:
    """Plain code approves or rejects each decision. No LLM can override it."""
    approved, rejected = evaluate_batch(
        state["decisions"],
        state["objective_constraints"],
        state["account"],
        state.get("performance", {}),
    )
    log_event(state["run_id"], "risk_gate_result", {"approved": approved, "rejected": rejected})
    return {"approved_orders": approved, "rejected_orders": rejected}


# --- 5. Execute --------------------------------------------------------------


def _realized_pnl_for_sell(decision: dict, account: dict, fill: dict) -> float | None:
    """Profit/loss of a closing SELL: fill price if known, else last price, minus what we paid."""
    symbol = decision["symbol"]
    position = account.get("positions", {}).get(symbol)
    if not position:
        return None
    avg_price = position.get("avg_price", 0)
    exit_price = fill.get("filled_avg_price") or account.get("last_prices", {}).get(
        symbol, avg_price
    )
    qty = fill.get("filled_qty") or decision.get("target_qty") or 0
    return (exit_price - avg_price) * qty


def _expected_positions(account: dict, orders: list[dict]) -> dict:
    """Holdings we expect once this cycle's orders fill."""
    expected = {s: dict(p) for s, p in account.get("positions", {}).items()}
    for order in orders:
        delta = order["target_qty"] * (1 if order["action"] == "BUY" else -1)
        current = expected.setdefault(order["symbol"], {"qty": 0.0, "avg_price": 0.0})
        current["qty"] = current.get("qty", 0.0) + delta
    return {s: p for s, p in expected.items() if abs(p.get("qty", 0.0)) > 1e-6}


async def n_execute(client: ClientSession, state: TradingState) -> dict:
    """Places the approved orders, then checks Alpaca's holdings match what we expect."""
    account = state["account"]
    orders = state["approved_orders"]
    results, closed_trades, failures = [], [], []

    for decision in orders:
        try:
            result = await execute_order(client, decision)
        except (MCPToolError, ValueError) as exc:
            # One bad order must not stop the rest.
            failures.append({"decision": decision, "error": str(exc)})
            log_event(state["run_id"], "order_failed", {"decision": decision, "error": str(exc)})
            continue

        results.append(result)
        fill = result.get("fill", {})
        if decision["action"] == "SELL" and fill.get("status") == "filled":
            pnl = _realized_pnl_for_sell(decision, account, fill)
            if pnl is not None:
                closed_trades.append(
                    {
                        "symbol": decision["symbol"],
                        "action": "SELL",
                        "qty": fill.get("filled_qty") or decision.get("target_qty"),
                        "pnl": pnl,
                    }
                )

    log_event(state["run_id"], "orders_executed", {"results": results, "failures": failures})

    if orders:  # holdings changed (or should have): re-read them from Alpaca
        refreshed = await get_account_state(client, state["candidates"])
        divergences = reconcile(_expected_positions(account, orders), refreshed)
        if divergences:
            log_event(state["run_id"], "reconciliation_divergence", {"divergences": divergences})
        account = refreshed
    return {"account": account, "closed_trades_this_cycle": closed_trades}


# --- 6. Log + Wait -----------------------------------------------------------


def _sleep_seconds(minutes_left: float, close_buffer: int, cadence_minutes: int) -> float:
    """Wait one cycle, but never past the start of the close buffer."""
    return min(cadence_minutes, max(minutes_left - close_buffer, 0.0)) * 60


async def n_log_wait(state: TradingState) -> dict:
    """Updates and logs profit/loss, decides what comes next, and waits."""
    account = state["account"]
    close_buffer = state["objective_constraints"]["constraints"]["market_close_buffer_minutes"]

    performance = compute_performance(
        account=account,
        session_capital=state["objective_constraints"]["session_capital"],
        starting_equity=state["starting_equity"],
        new_trades=state.get("closed_trades_this_cycle", []),
        prior=state.get("performance"),
    )
    log_event(state["run_id"], "performance", performance)

    minutes_left = account.get("minutes_to_close", 0.0)
    session_done = minutes_left <= close_buffer
    sleep_seconds = 0.0 if session_done else _sleep_seconds(minutes_left, close_buffer, CYCLE_MINUTES)

    # Re-pick once an hour has passed since the last pick (counting the wait about to happen).
    minutes_since_pick = state.get("picked_at_minutes_left", minutes_left) - (
        minutes_left - sleep_seconds / 60
    )
    repick_due = not session_done and minutes_since_pick >= REPICK_MINUTES

    cycle_index = state.get("cycle_index", 0) + 1
    log_event(
        state["run_id"],
        "cycle_end",
        {
            "cycle_index": cycle_index,
            "minutes_to_close": minutes_left,
            "session_done": session_done,
            "repick_due": repick_due,
            "sleep_seconds": sleep_seconds,
        },
    )

    if sleep_seconds > 0:
        await asyncio.sleep(sleep_seconds)

    return {
        "performance": performance,
        "cycle_index": cycle_index,
        "session_done": session_done,
        "repick_due": repick_due,
    }


# --- 7. Close ----------------------------------------------------------------


async def n_close(client: ClientSession, state: TradingState) -> dict:
    """Sells every open position and reports the final result. Nothing is held overnight."""
    liquidation = await liquidate_all(client)
    if liquidation.get("failures"):
        log_event(state["run_id"], "liquidation_incomplete", {"failures": liquidation["failures"]})

    account = await get_account_state(client, state.get("candidates", []))
    final_performance = compute_performance(
        account=account,
        session_capital=state["objective_constraints"]["session_capital"],
        starting_equity=state["starting_equity"],
        new_trades=liquidation.get("closed_trades", []),
        prior=state.get("performance"),
    )
    remaining = {s: p for s, p in account["positions"].items() if p.get("qty")}
    log_event(
        state["run_id"],
        "finalized",
        {
            "liquidation": liquidation,
            "final_performance": final_performance,
            "positions_still_open": remaining,
        },
    )
    return {
        "account": account,
        "performance": final_performance,
        "positions_still_open": remaining,
    }
