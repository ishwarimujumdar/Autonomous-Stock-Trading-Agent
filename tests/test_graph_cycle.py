"""
End-to-end graph tests with the MCP server and the LLM stubbed out.

The previous suite only covered the two pure-logic nodes, so nothing exercised
the wiring: node signatures, state keys flowing between nodes, the routing
predicates, or the flatten-on-exit guarantee.
"""

import pytest

import src.analysis.market_news as market_news
import src.analysis.technical as technical
import src.execution.broker as broker
import src.graph.nodes as nodes
import src.scanner.scanner as scanner
import src.strategy.strategy_agent as strategy_agent
from src.config import load_objective_and_constraints
from src.graph.build_graph import build_graph

CLOCK = {
    "is_open": True,
    "timestamp": "2026-09-22T15:00:00Z",
    "next_close": "2026-09-22T20:00:00Z",
}

STRATEGY = {
    "name": "momentum",
    "signals": ["price_momentum", "rsi"],
    "universe_symbols": ["AAPL", "MSFT"],
    "universe_criteria": {"min_price": 5},
    "cadence_minutes": 5,
    "entry_logic": "intraday momentum",
    "exit_logic": "momentum fades",
    "rationale": "test",
}


class FakeBroker:
    """Stands in for the Alpaca MCP server, recording the orders it receives."""

    def __init__(self, positions=None, clock=None):
        self.positions = positions or {}
        self.clock = clock or CLOCK
        self.orders = []

    async def call_tool(self, client, name, arguments):
        # Shapes below are what call_tool() returns AFTER its envelope unwrap
        # (see src/alpaca/mcp_client.py) - this fake replaces call_tool
        # entirely, so it must hand back already-unwrapped payloads, not the
        # server's raw {"_alpaca_mcp_security": ..., "data": ...} envelope.
        if name == "get_account_info":
            return {"cash": "10000", "portfolio_value": "10000", "equity": "10000",
                    "last_equity": "10000"}
        if name == "get_all_positions":
            return {"result": [
                {"symbol": s, "qty": str(p["qty"]), "avg_entry_price": str(p["avg_price"])}
                for s, p in self.positions.items()
            ]}
        if name == "get_orders":
            return {"result": []}
        if name == "get_clock":
            return self.clock
        if name == "get_stock_snapshot":
            symbol = arguments["symbols"]
            return {symbol: {"latestTrade": {"p": 100.0}, "dailyBar": {"o": 99.0, "v": 5e6}}}
        if name == "get_stock_bars":
            symbol = arguments["symbols"]
            return {"bars": {
                symbol: [{"c": 100 + i * 0.1, "v": 10_000} for i in range(60)]
            }}
        if name == "place_stock_order":
            self.orders.append(arguments)
            return {"id": f"order-{len(self.orders)}"}
        if name == "get_order_by_id":
            return {"status": "filled", "filled_qty": "1", "filled_avg_price": "100.0"}
        if name == "get_news":
            return {"news": []}
        raise AssertionError(f"unexpected tool {name}")


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Patches every outbound edge: MCP tools, both LLM nodes, and the journal."""
    fake = FakeBroker()

    for module in (broker, scanner, technical, market_news):
        monkeypatch.setattr(module, "call_tool", fake.call_tool)

    async def fake_propose(**kwargs):
        return dict(STRATEGY)

    monkeypatch.setattr(strategy_agent, "propose_strategy", fake_propose)
    monkeypatch.setattr(nodes, "propose_strategy", fake_propose)

    # No sleeping between cycles in tests.
    async def no_sleep(seconds):
        return None

    monkeypatch.setattr(nodes.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(nodes, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(broker, "_FILL_POLL_SECONDS", 0)
    scanner.clear_cache()
    market_news.clear_cache()
    return fake


def near_close_clock(minutes: float) -> dict:
    """A clock that reports `minutes` left in the session."""
    close_hour, close_minute = divmod(int(minutes), 60)
    return {
        "is_open": True,
        "timestamp": "2026-09-22T15:00:00Z",
        "next_close": f"2026-09-22T{15 + close_hour:02d}:{close_minute:02d}:00Z",
    }


def _initial_state(minutes_left_clock=None):
    return {
        "run_id": "test_run",
        "run_config": {
            "session_capital": 10000,
            "universe_hint": "",
            "session_date": "2026-09-22",
        },
        "objective_constraints": load_objective_and_constraints(session_capital=10000),
        "starting_equity": 10000.0,
        "account": {
            "cash": 10000.0,
            "portfolio_value": 10000.0,
            "equity": 10000.0,
            "positions": {},
            "last_prices": {},
            "open_orders": [],
            "clock": minutes_left_clock or CLOCK,
            "daily_pnl_pct": 0.0,
            "minutes_to_close": 300.0,
        },
        "performance": {
            "realized_pnl": 0.0, "unrealized_pnl": 0.0, "return_pct": 0.0,
            "wins": 0, "losses": 0, "scratches": 0, "trade_count": 0, "recent_trades": [],
        },
        "cycle_index": 0,
        "strategy_attempts": 0,
        "rejected_cycle_streak": 0,
        "minutes_to_close": 300.0,
        "session_done": False,
    }


def _stub_decision(monkeypatch, action, confidence=0.9, qty=1):
    async def fake_decide(symbol, **kwargs):
        return {
            "symbol": symbol, "action": action, "confidence": confidence,
            "rationale": "stub", "target_qty": qty if action != "HOLD" else None,
        }

    monkeypatch.setattr(nodes, "decide", fake_decide)


@pytest.mark.asyncio
async def test_full_cycle_places_orders_and_finalizes(wired, monkeypatch):
    """A BUY cycle runs end to end and the session still flattens at the close."""
    _stub_decision(monkeypatch, "BUY")
    # Close buffer already reached, so one cycle runs then finalize.
    wired.clock = near_close_clock(10)
    state = _initial_state()
    state["account"]["minutes_to_close"] = 10.0

    graph = build_graph(client=None)
    final = await graph.ainvoke(state, config={"recursion_limit": 100})

    assert final["session_done"] is True
    # Inside the close buffer no BUY is allowed, and finalize flattened nothing
    # because nothing was ever opened.
    assert all(r["reason"] for r in final["rejected_orders"])
    assert "market close" in final["rejected_orders"][0]["reason"]


@pytest.mark.asyncio
async def test_open_positions_are_flattened_at_the_close(wired, monkeypatch):
    _stub_decision(monkeypatch, "HOLD")
    wired.positions = {"AAPL": {"qty": 5, "avg_price": 95.0}}
    wired.clock = near_close_clock(5)

    state = _initial_state()
    state["account"]["minutes_to_close"] = 5.0

    graph = build_graph(client=None)
    final = await graph.ainvoke(state, config={"recursion_limit": 100})

    sells = [o for o in wired.orders if o["side"] == "sell"]
    assert sells, "finalize must liquidate the open position"
    assert sells[0]["symbol"] == "AAPL"
    assert sells[0]["qty"] == "5.0"
    # +5 per share on 5 shares, realized at the 100.0 fill against a 95.0 basis.
    assert final["performance"]["realized_pnl"] == pytest.approx(25.0)


@pytest.mark.asyncio
async def test_held_position_stays_a_candidate_after_falling_out_of_the_scan(wired, monkeypatch):
    """A position must keep being managed even once it fails the filter."""
    _stub_decision(monkeypatch, "HOLD")
    wired.positions = {"TSLA": {"qty": 3, "avg_price": 100.0}}

    state = _initial_state()
    state["account"]["positions"] = {"TSLA": {"qty": 3, "avg_price": 100.0}}

    result = await nodes.n_scanner(None, {**state, "strategy": STRATEGY})
    assert "TSLA" in result["candidates"], "TSLA is not in universe_symbols but is held"


@pytest.mark.asyncio
async def test_invalid_strategy_retries_then_routes_to_flatten(wired, monkeypatch):
    """
    An unusable strategy must exhaust a bounded retry budget and still reach
    the flatten path, rather than looping to the recursion limit.
    """
    attempts = {"n": 0}

    async def bad_proposal(**kwargs):
        attempts["n"] += 1
        return {"name": "not_a_real_strategy", "signals": ["price_momentum"],
                "universe_symbols": ["AAPL"], "universe_criteria": {},
                "cadence_minutes": 5, "entry_logic": "x", "exit_logic": "y", "rationale": "z"}

    monkeypatch.setattr(nodes, "propose_strategy", bad_proposal)
    wired.positions = {"AAPL": {"qty": 2, "avg_price": 90.0}}
    wired.clock = near_close_clock(5)

    graph = build_graph(client=None)
    final = await graph.ainvoke(_initial_state(), config={"recursion_limit": 100})

    from src.config import MAX_STRATEGY_ATTEMPTS

    assert attempts["n"] == MAX_STRATEGY_ATTEMPTS
    assert final["strategy_failed"] is True
    assert [o["side"] for o in wired.orders] == ["sell"], "positions flattened despite the failure"


@pytest.mark.asyncio
async def test_first_cycle_can_trade(wired, monkeypatch):
    """
    The risk gate runs before the monitor, so on cycle 1 it had no prices and
    rejected every trade for want of one. The scanner now supplies them.
    """
    _stub_decision(monkeypatch, "BUY")
    wired.clock = near_close_clock(60)

    state = _initial_state()
    assert state["account"]["last_prices"] == {}, "cycle 1 genuinely starts with no quotes"

    state = {**state, "strategy": STRATEGY}
    scanned = await nodes.n_scanner(None, state)
    assert scanned["account"]["last_prices"], "scanner must publish fresh quotes"

    state = {**state, **scanned}
    state = {**state, **await nodes.n_research(None, state)}
    state = {**state, **await nodes.n_trade_decision(state)}
    gated = await nodes.n_risk_gate(state)

    assert gated["approved_orders"], f"nothing approved: {gated['rejected_orders']}"


@pytest.mark.asyncio
async def test_cycle_sleep_never_runs_past_the_close_buffer(wired, monkeypatch):
    """
    A 30-minute cadence with 25 minutes left used to sleep straight through
    the close, waking after hours with positions still open.
    """
    slept = []

    async def record_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(nodes.asyncio, "sleep", record_sleep)

    state = _initial_state()
    state["strategy"] = {**STRATEGY, "cadence_minutes": 30}
    state["account"]["minutes_to_close"] = 25.0
    state["decisions"] = []

    await nodes.n_cycle_controller(state)

    buffer_minutes = state["objective_constraints"]["constraints"]["market_close_buffer_minutes"]
    assert slept == [(25.0 - buffer_minutes) * 60], "sleeps only up to the flatten window"


@pytest.mark.asyncio
async def test_cycle_sleeps_the_full_cadence_when_there_is_room(wired, monkeypatch):
    slept = []

    async def record_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(nodes.asyncio, "sleep", record_sleep)

    state = _initial_state()
    state["strategy"] = {**STRATEGY, "cadence_minutes": 5}
    state["account"]["minutes_to_close"] = 300.0
    state["decisions"] = []

    await nodes.n_cycle_controller(state)
    assert slept == [300.0]


@pytest.mark.asyncio
async def test_accepting_a_new_strategy_resets_the_rejection_streak(wired):
    """
    Without this reset, a streak built up under the OLD strategy carried
    straight into its replacement - confirmed live: a strategy was replaced
    after 3 quiet cycles, and the brand new one got discarded after just 1
    quiet cycle because the streak inherited was already 3, not 0.
    """
    state = {
        "run_id": "test_run",
        "strategy": STRATEGY,
        "strategy_attempts": 2,
        "rejected_cycle_streak": 4,  # carried over from the strategy just replaced
        "strategy_history": [],
    }

    result = await nodes.n_schema_validator(state)

    assert result["strategy_valid"] is True
    assert result["strategy_attempts"] == 0
    assert result["rejected_cycle_streak"] == 0


@pytest.mark.asyncio
async def test_restrategize_skips_the_sleep(wired, monkeypatch):
    """
    A restrategize means the current strategy - and its cadence - is being
    abandoned. Sleeping out that cadence before reassessing just delays the
    replacement for no reason; the new strategy should be proposed
    immediately, not ~cadence_minutes later.
    """
    slept = []

    async def record_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(nodes.asyncio, "sleep", record_sleep)

    state = _initial_state()
    state["strategy"] = {**STRATEGY, "cadence_minutes": 30}
    state["account"]["minutes_to_close"] = 300.0  # plenty of room - not close-buffer related
    state["decisions"] = [{"symbol": "AAPL"}]
    state["approved_orders"] = []
    state["rejected_cycle_streak"] = 2  # this cycle's all-rejected pushes it to 3

    result = await nodes.n_cycle_controller(state)

    assert result["restrategize"] is True
    assert slept == [], "no sleep at all when restrategizing"


@pytest.mark.asyncio
async def test_rejection_streak_counts_cycles_not_rejections(wired):
    state = _initial_state()
    state["strategy"] = STRATEGY
    state["account"]["minutes_to_close"] = 5.0  # session_done, so no sleep
    state["decisions"] = [{"symbol": "AAPL"}, {"symbol": "MSFT"}, {"symbol": "NVDA"}]
    state["approved_orders"] = []
    state["rejected_cycle_streak"] = 0

    result = await nodes.n_cycle_controller(state)
    assert result["rejected_cycle_streak"] == 1, "one bad cycle is a streak of one"

    result = await nodes.n_cycle_controller({**state, "rejected_cycle_streak": 1})
    assert result["rejected_cycle_streak"] == 2


@pytest.mark.asyncio
async def test_rejection_streak_resets_when_a_trade_gets_through(wired):
    state = _initial_state()
    state["strategy"] = STRATEGY
    state["account"]["minutes_to_close"] = 5.0
    state["decisions"] = [{"symbol": "AAPL"}]
    state["approved_orders"] = [{"symbol": "AAPL"}]
    state["rejected_cycle_streak"] = 2

    result = await nodes.n_cycle_controller(state)
    assert result["rejected_cycle_streak"] == 0


@pytest.mark.asyncio
async def test_a_failing_order_does_not_abort_the_remaining_orders(wired, monkeypatch):
    _stub_decision(monkeypatch, "BUY")

    calls = {"n": 0}
    original = wired.call_tool

    async def flaky(client, name, arguments):
        if name == "place_stock_order":
            calls["n"] += 1
            if calls["n"] == 1:
                from src.alpaca.mcp_client import MCPToolError

                raise MCPToolError("broker rejected the order")
        return await original(client, name, arguments)

    monkeypatch.setattr(broker, "call_tool", flaky)

    state = _initial_state()
    result = await nodes.n_execution(
        None,
        {
            **state,
            "approved_orders": [
                {"symbol": "AAPL", "action": "BUY", "confidence": 0.9, "target_qty": 1},
                {"symbol": "MSFT", "action": "BUY", "confidence": 0.9, "target_qty": 1},
            ],
        },
    )
    assert calls["n"] == 2, "the second order was still attempted"
    assert result["expected_positions"]["MSFT"]["qty"] == 1
