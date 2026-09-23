import asyncio
import uuid
from datetime import datetime, timezone

from src.alpaca.mcp_client import alpaca_mcp_session
from src.config import load_objective_and_constraints
from src.execution.broker import get_account_state, liquidate_all
from src.graph.build_graph import build_graph
from src.persistence.journal import log_event


def _parse_capital(raw: str) -> float:
    """Accepts plain numbers as well as what the prompt itself prints back,
    e.g. "10,000.00" or "$10,000" - typing the suggested default verbatim
    used to crash on the comma."""
    cleaned = raw.replace(",", "").replace("$", "").strip()
    return float(cleaned)


def _prompt_run_config(available_equity: float) -> dict:
    default = round(available_equity, 2)
    while True:
        raw = input(
            f"Capital to allocate for today's session (blank = all {default:,.2f}): "
        ).strip()
        if not raw:
            session_capital = default
            break
        try:
            session_capital = _parse_capital(raw)
            break
        except ValueError:
            print(f"Couldn't parse {raw!r} as a number - try again (e.g. 10000 or 10000.50).")

    if session_capital <= 0:
        print(f"{session_capital} is not a usable allocation - defaulting to {default:,.2f}.")
        session_capital = default

    if session_capital > available_equity:
        print(
            f"Allocation {session_capital:,.2f} exceeds account equity "
            f"{available_equity:,.2f} - capping at account equity."
        )
        session_capital = available_equity

    universe_hint = input("Universe hint (optional, e.g. 'liquid US tech stocks'): ").strip()
    return {
        "session_capital": session_capital,
        "universe_hint": universe_hint,
        "session_date": datetime.now(timezone.utc).date().isoformat(),
    }


async def _emergency_flatten(client, run_id: str, error: BaseException) -> None:
    """
    Last-resort flatten when the graph didn't reach `finalize`.

    An unhandled error used to propagate straight out of `ainvoke`, so a
    mid-session MCP hiccup left open positions to sit overnight - the one
    outcome this agent exists to prevent.
    """
    print(f"\n!! Session aborted: {type(error).__name__}: {error}")
    print("!! Force-flattening all open positions...")
    try:
        result = await liquidate_all(client)
        log_event(run_id, "emergency_liquidation", result)
        if result.get("failures"):
            print(f"!! Some positions could NOT be closed: {result['failures']}")
            print("!! Check the Alpaca dashboard manually.")
        else:
            print(f"!! Flattened {len(result.get('closed_trades', []))} position(s).")
    except Exception as flatten_error:  # noqa: BLE001 - nothing above us to retry
        log_event(run_id, "emergency_liquidation_failed", {"error": str(flatten_error)})
        print(f"!! FLATTEN FAILED: {flatten_error}")
        print("!! POSITIONS MAY STILL BE OPEN - close them manually on Alpaca.")


async def main() -> int:
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    async with alpaca_mcp_session() as client:
        account = await get_account_state(client, symbols=[])

        if not account.get("clock", {}).get("is_open", False):
            print("Market is currently closed. This is a day-trading agent - "
                  "run it again once the session opens.")
            return 0

        run_config = _prompt_run_config(account.get("equity", account.get("portfolio_value", 0)))
        objective_constraints = load_objective_and_constraints(
            session_capital=run_config["session_capital"],
        )
        log_event(
            run_id,
            "run_started",
            {"run_config": run_config, "objective_constraints": objective_constraints},
        )

        # Re-read the account after the prompts: the user may have sat at the
        # input for a while, and minutes_to_close must not start out stale.
        account = await get_account_state(client, symbols=[])

        initial_state = {
            "run_id": run_id,
            "run_config": run_config,
            "objective_constraints": objective_constraints,
            "starting_equity": account.get("equity", account.get("portfolio_value", 0.0)),
            "account": account,
            "performance": {
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "return_pct": 0.0,
                "wins": 0,
                "losses": 0,
                "scratches": 0,
                "trade_count": 0,
                "recent_trades": [],
            },
            "cycle_index": 0,
            "strategy_attempts": 0,
            "rejected_cycle_streak": 0,
            "minutes_to_close": account["minutes_to_close"],
            "session_done": False,
        }

        graph = build_graph(client)

        try:
            final_state = await graph.ainvoke(
                initial_state, config={"recursion_limit": 10_000}
            )
        except BaseException as error:  # includes KeyboardInterrupt / CancelledError
            await _emergency_flatten(client, run_id, error)
            log_event(run_id, "run_aborted", {"error": f"{type(error).__name__}: {error}"})
            raise

        print("\n=== SESSION COMPLETE ===")
        print(f"Run ID: {run_id}")
        if final_state.get("strategy_failed"):
            print("Ended early: the Strategy Agent could not produce a valid strategy.")
        print(f"Final performance: {final_state['performance']}")
        if final_state.get("positions_still_open"):
            print(f"WARNING - positions still open: {final_state['positions_still_open']}")
        log_event(run_id, "run_complete", {"final_performance": final_state["performance"]})
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
