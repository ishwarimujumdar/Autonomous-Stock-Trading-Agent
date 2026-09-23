import asyncio

from mcp import ClientSession

from src.alpaca.mcp_client import MCPToolError
from src.analysis.market_news import analyze_news
from src.analysis.technical import compute_technical, has_usable_evidence
from src.config import INTRADAY_TIMEFRAME, MAX_CONCURRENCY, MAX_STRATEGY_ATTEMPTS
from src.decision.decision import decide
from src.evaluation.pnl import compute_performance, should_restrategize
from src.execution.broker import execute_order, get_account_state, liquidate_all, reconcile
from src.graph.state import TradingState
from src.persistence.journal import log_event
from src.risk.risk_gate import evaluate_batch
from src.scanner.scanner import scan
from src.strategy.schema import validate_strategy_proposal
from src.strategy.strategy_agent import propose_strategy


async def n_strategy_agent(state: TradingState) -> dict:
    attempts = state.get("strategy_attempts", 0) + 1
    raw = await propose_strategy(
        objective_constraints=state["objective_constraints"],
        run_config=state["run_config"],
        strategy_history=state.get("strategy_history", []),
    )
    log_event(state["run_id"], "strategy_proposed", {"attempt": attempts, "proposal": raw})
    return {
        "strategy": raw,
        "strategy_attempts": attempts,
        "restrategize": False,
        "restrategize_reason": None,
    }


async def n_schema_validator(state: TradingState) -> dict:
    validated, error = validate_strategy_proposal(state["strategy"])
    history = state.get("strategy_history", [])

    if error:
        entry = {
            "strategy": state["strategy"],
            "strategy_valid": False,
            "strategy_validation_error": error,
        }
        log_event(
            state["run_id"],
            "strategy_rejected",
            {"error": error, "attempt": state.get("strategy_attempts", 0)},
        )
        return {
            "strategy_valid": False,
            "strategy_validation_error": error,
            "strategy_history": history + [entry],
        }

    log_event(state["run_id"], "strategy_accepted", validated)
    # Reset both counters: strategy_attempts so a later reassessment gets a
    # fresh validation-retry budget, and rejected_cycle_streak so a brand new
    # strategy starts clean. Without this reset, a streak built up under the
    # OLD strategy carried straight into the new one - confirmed live: a
    # strategy was replaced after 3 quiet cycles, and its replacement got
    # discarded after just 1 quiet cycle because the streak was already at 4,
    # not 1. The counter is meant to measure how long the CURRENT strategy
    # has been failing, not strategies in aggregate.
    return {
        "strategy": validated,
        "strategy_valid": True,
        "strategy_validation_error": None,
        "strategy_attempts": 0,
        "rejected_cycle_streak": 0,
    }


async def n_scanner(client: ClientSession, state: TradingState) -> dict:
    strategy = state["strategy"]
    account = state.get("account", {})
    candidates, exclusions, prices = await scan(
        client,
        strategy.get("universe_criteria", {}),
        strategy.get("universe_symbols"),
    )
    # Anything currently held must stay under review even if it no longer
    # passes the filter - otherwise an open position stops being managed the
    # moment it falls out of the scan.
    held = [s for s in account.get("positions", {}) if s not in candidates]
    log_event(
        state["run_id"],
        "scan_result",
        {"candidates": candidates, "held_carried_forward": held, "exclusions": exclusions},
    )
    # Fold this cycle's quotes into the account so the risk gate, which runs
    # before the monitor refreshes, sizes trades on current prices.
    refreshed = {**account, "last_prices": {**account.get("last_prices", {}), **prices}}
    return {"candidates": candidates + held, "account": refreshed}


async def _research_symbol(
    client: ClientSession, symbol: str, strategy: dict, semaphore: asyncio.Semaphore
) -> tuple[str, dict]:
    async with semaphore:
        try:
            technical = await compute_technical(client, symbol, strategy["signals"])
        except MCPToolError as exc:
            return symbol, {"technical": {"symbol": symbol, "error": str(exc)}, "narrative": None}
        try:
            news = await analyze_news(client, symbol, strategy["name"], strategy["signals"])
        except MCPToolError as exc:
            news = {"narrative": f"news unavailable: {exc}"}
        return symbol, {"technical": technical, "narrative": news.get("narrative")}


async def n_research(client: ClientSession, state: TradingState) -> dict:
    strategy = state["strategy"]
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    # Fanned out rather than looped: this was three sequential round trips per
    # symbol, which for a 15-symbol universe could outlast a 5-minute cadence.
    pairs = await asyncio.gather(
        *(
            _research_symbol(client, symbol, strategy, semaphore)
            for symbol in state["candidates"]
        )
    )
    research = dict(pairs)
    log_event(state["run_id"], "research_complete", {"symbols": list(research)})
    return {"research": research}


async def n_trade_decision(state: TradingState) -> dict:
    strategy = state["strategy"]
    performance = state.get("performance", {})
    account = state.get("account", {})
    recent_trades = performance.get("recent_trades", [])
    positions = account.get("positions", {})
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def decide_one(symbol: str, evidence: dict) -> dict:
        async with semaphore:
            return await decide(
                symbol=symbol,
                strategy=strategy,
                technical=evidence["technical"],
                narrative=evidence["narrative"],
                previous_outcomes=[t for t in recent_trades if t.get("symbol") == symbol],
                performance=performance,
                account=account,
                timeframe=INTRADAY_TIMEFRAME,
                objective_constraints=state["objective_constraints"],
            )

    asked, skipped = [], []
    for symbol, evidence in state["research"].items():
        # Always reason about something we hold - we may need to exit it. For
        # everything else, skip the call when no requested indicator computed.
        if symbol in positions or has_usable_evidence(evidence["technical"], strategy["signals"]):
            asked.append((symbol, evidence))
        else:
            skipped.append(symbol)

    decisions = list(await asyncio.gather(*(decide_one(s, e) for s, e in asked)))

    log_event(
        state["run_id"],
        "decisions_made",
        {"decisions": decisions, "skipped_no_evidence": skipped},
    )
    return {"decisions": decisions}


async def n_risk_gate(state: TradingState) -> dict:
    approved, rejected = evaluate_batch(
        state["decisions"],
        state["objective_constraints"],
        state.get("account", {}),
        state.get("performance", {}),
    )
    log_event(state["run_id"], "risk_gate_result", {"approved": approved, "rejected": rejected})
    return {"approved_orders": approved, "rejected_orders": rejected}


def _realized_pnl_for_sell(decision: dict, account: dict, fill: dict) -> float | None:
    """
    Realized P&L for a closing SELL, priced at the actual fill where Alpaca
    reported one and falling back to the last known price otherwise. Only
    SELLs realize P&L - a BUY just opens/adds to a position.
    """
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
    """Positions we expect once this cycle's orders fill, for reconciliation."""
    expected = {s: dict(p) for s, p in account.get("positions", {}).items()}
    for order in orders:
        symbol = order["symbol"]
        delta = order["target_qty"] * (1 if order["action"] == "BUY" else -1)
        current = expected.setdefault(symbol, {"qty": 0.0, "avg_price": 0.0})
        current["qty"] = current.get("qty", 0.0) + delta
    return {s: p for s, p in expected.items() if abs(p.get("qty", 0.0)) > 1e-6}


async def n_execution(client: ClientSession, state: TradingState) -> dict:
    account = state.get("account", {})
    results, closed_trades, failures = [], [], []

    for decision in state["approved_orders"]:
        try:
            result = await execute_order(client, decision)
        except (MCPToolError, ValueError) as exc:
            # One bad order must not abandon the remaining orders, and must not
            # skip the flatten path at the end of the session.
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
    return {
        "closed_trades_this_cycle": closed_trades,
        "expected_positions": _expected_positions(account, state["approved_orders"]),
    }


async def n_monitor(client: ClientSession, state: TradingState) -> dict:
    account = await get_account_state(client, state.get("candidates", []))
    divergences = reconcile(state.get("expected_positions", {}), account)
    if divergences:
        log_event(state["run_id"], "reconciliation_divergence", {"divergences": divergences})
    log_event(state["run_id"], "account_state", account)
    return {"account": account, "reconciliation_divergences": divergences}


async def n_performance_tracker(state: TradingState) -> dict:
    performance = compute_performance(
        account=state["account"],
        session_capital=state["objective_constraints"]["session_capital"],
        starting_equity=state["starting_equity"],
        new_trades=state.get("closed_trades_this_cycle", []),
        prior=state.get("performance"),
    )
    log_event(state["run_id"], "performance", performance)
    # Clear the transient per-cycle buffers now they're folded in.
    return {
        "performance": performance,
        "closed_trades_this_cycle": [],
        "expected_positions": {},
    }


async def n_cycle_controller(state: TradingState) -> dict:
    cycle_index = state.get("cycle_index", 0) + 1
    constraints = state["objective_constraints"]["constraints"]
    close_buffer = constraints["market_close_buffer_minutes"]
    minutes_left = state["account"].get("minutes_to_close", 0.0)

    session_done = minutes_left <= close_buffer

    # A streak is consecutive *cycles* where nothing got through, not the count
    # of rejections inside one busy cycle.
    had_decisions = bool(state.get("decisions"))
    all_rejected = had_decisions and not state.get("approved_orders")
    streak = state.get("rejected_cycle_streak", 0) + 1 if all_rejected else 0

    restrategize, reason = should_restrategize(state["performance"], streak)

    history = state.get("strategy_history", [])
    if restrategize:
        # Record the outgoing strategy as "used" (with why it's being
        # replaced) before the Strategy Agent proposes a new one - otherwise
        # it has no way to know this strategy was already tried and dropped.
        history = history + [
            {
                "strategy": state["strategy"],
                "strategy_valid": True,
                "reason": reason,
                "performance_at_switch": state["performance"],
            }
        ]

    sleep_seconds = 0.0
    if not session_done and not restrategize:
        # Never sleep past the flatten window: a 30-minute cadence with 25
        # minutes left used to wake up after the close, leaving day orders to
        # be rejected and positions unflattened.
        #
        # A restrategize also skips the sleep entirely: the current
        # strategy's cadence is exactly what's being abandoned, so waiting it
        # out first before reassessing just adds a pointless delay between
        # deciding to replace the strategy and actually doing it.
        budget_minutes = max(minutes_left - close_buffer, 0.0)
        sleep_seconds = min(state["strategy"]["cadence_minutes"], budget_minutes) * 60

    log_event(
        state["run_id"],
        "cycle_end",
        {
            "cycle_index": cycle_index,
            "minutes_to_close": minutes_left,
            "session_done": session_done,
            "restrategize": restrategize,
            "reason": reason,
            "rejected_cycle_streak": streak,
            "sleep_seconds": sleep_seconds,
        },
    )

    if sleep_seconds > 0:
        await asyncio.sleep(sleep_seconds)

    return {
        "cycle_index": cycle_index,
        "minutes_to_close": minutes_left,
        "session_done": session_done,
        "restrategize": restrategize,
        "restrategize_reason": reason,
        "rejected_cycle_streak": streak,
        "strategy_history": history,
    }


async def n_finalize(client: ClientSession, state: TradingState) -> dict:
    liquidation = await liquidate_all(client)
    if liquidation.get("failures"):
        log_event(
            state["run_id"],
            "liquidation_incomplete",
            {"failures": liquidation["failures"]},
        )

    account = await get_account_state(client, state.get("candidates", []))
    final_performance = compute_performance(
        account=account,
        session_capital=state["objective_constraints"]["session_capital"],
        starting_equity=state["starting_equity"],
        new_trades=liquidation.get("closed_trades", []),
        prior=state.get("performance"),
    )
    remaining = {s: p for s, p in account.get("positions", {}).items() if p.get("qty")}
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


async def n_strategy_failed(state: TradingState) -> dict:
    """Terminal node for an agent that cannot produce a usable strategy."""
    error = state.get("strategy_validation_error")
    log_event(
        state["run_id"],
        "strategy_attempts_exhausted",
        {"attempts": MAX_STRATEGY_ATTEMPTS, "last_error": error},
    )
    return {"session_done": True, "strategy_failed": True}
