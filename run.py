import asyncio
import uuid
from datetime import datetime, timezone

from src.alpaca.account_and_orders import get_account_state, liquidate_all
from src.alpaca.mcp_client import alpaca_mcp_session
from src.config import load_objective_and_constraints
from src.graph.build_graph import build_graph
from src.persistence.journal import log_event


def _ask_capital(available_equity: float) -> float:
    """Asks how much to allocate; blank means everything, and it's capped at account equity."""
    default = round(available_equity, 2)
    while True:
        raw = input(f"Capital to allocate for today's session (blank = all {default:,.2f}): ").strip()
        if not raw:
            return default
        try:
            capital = float(raw.replace(",", "").replace("$", ""))
        except ValueError:
            print(f"Couldn't parse {raw!r} as a number - try again (e.g. 10000 or 10000.50).")
            continue
        if capital <= 0:
            print(f"{capital} is not a usable allocation - defaulting to {default:,.2f}.")
            return default
        if capital > available_equity:
            print(f"Allocation exceeds account equity {available_equity:,.2f} - capping it.")
            return available_equity
        return capital


def _prompt_run_config(available_equity: float) -> dict:
    return {
        "session_capital": _ask_capital(available_equity),
        "universe_hint": input("Universe hint (optional, e.g. 'liquid US tech stocks'): ").strip(),
        "session_date": datetime.now(timezone.utc).date().isoformat(),
    }


async def _emergency_flatten(client, run_id: str, error: BaseException) -> None:
    """Last-resort sell-off when the graph crashed before reaching `finalize`."""
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


def _initial_state(run_id: str, run_config: dict, objective_constraints: dict, account: dict) -> dict:
    return {
        "run_id": run_id,
        "run_config": run_config,
        "objective_constraints": objective_constraints,
        "starting_equity": account["equity"],
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
        "session_done": False,
    }


def _print_summary(run_id: str, final_state: dict) -> None:
    print("\n=== SESSION COMPLETE ===")
    print(f"Run ID: {run_id}")
    if final_state.get("pick_failed"):
        print("Ended early: the AI could not produce a valid list of stocks to watch.")
    print(f"Final performance: {final_state['performance']}")
    if final_state.get("positions_still_open"):
        print(f"WARNING - positions still open: {final_state['positions_still_open']}")


async def main() -> int:
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    async with alpaca_mcp_session() as client:
        account = await get_account_state(client, symbols=[])

        if not account.get("clock", {}).get("is_open", False):
            print("Market is currently closed. This is a day-trading agent - "
                  "run it again once the session opens.")
            return 0

        run_config = _prompt_run_config(account["equity"])
        objective_constraints = load_objective_and_constraints(
            session_capital=run_config["session_capital"],
        )
        log_event(
            run_id,
            "run_started",
            {"run_config": run_config, "objective_constraints": objective_constraints},
        )

        # Re-read after the prompts so minutes_to_close isn't stale.
        account = await get_account_state(client, symbols=[])

        initial_state = _initial_state(run_id, run_config, objective_constraints, account)

        graph = build_graph(client)

        try:
            final_state = await graph.ainvoke(
                initial_state, config={"recursion_limit": 10_000}
            )
        except BaseException as error:  # includes KeyboardInterrupt / CancelledError
            await _emergency_flatten(client, run_id, error)
            log_event(run_id, "run_aborted", {"error": f"{type(error).__name__}: {error}"})
            raise

        _print_summary(run_id, final_state)
        log_event(run_id, "run_complete", {"final_performance": final_state["performance"]})
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
