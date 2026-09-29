"""
End-to-end graph tests with the Alpaca MCP server and the LLM stubbed out.

They exercise the wiring: node signatures, the state flowing between nodes,
the routing (next cycle / hourly re-pick / close) and the sell-everything-at-
close guarantee.
"""

from datetime import datetime, timedelta, timezone

import pytest

import src.alpaca.account_and_orders as broker
import src.analysis.technical as technical
import src.graph.nodes as nodes
import src.picker.universe as universe
from src.config import REPICK_MINUTES, load_objective_and_constraints
from src.graph.build_graph import _route_after_log_wait, build_graph
from src.picker.stock_picker import Watchlist

WATCHLIST = ["AAPL", "MSFT"]


def clock_with(minutes_left: float) -> dict:
    now = datetime(2026, 9, 22, 15, 0, tzinfo=timezone.utc)
    return {
        "is_open": True,
        "timestamp": now.isoformat(),
        "next_close": (now + timedelta(minutes=minutes_left)).isoformat(),
    }


class FakeBroker:
    """
    Stands in for the Alpaca MCP server, recording orders and calls.

    `minutes_left` is what successive get_clock calls report; the last value
    repeats, so a list like [60, 10] means "plenty of time, then near the close".
    """

    def __init__(self, positions=None, minutes_left=(300,)):
        self.positions = positions or {}
        self.minutes_left = list(minutes_left)
        self.orders = []
        self.calls = []

    async def call_tool(self, client, name, arguments):
        # Shapes are what call_tool() returns AFTER its envelope unwrap (see
        # src/alpaca/mcp_client.py): this fake replaces call_tool entirely.
        self.calls.append(name)
        if name == "get_account_info":
            return {"cash": "10000", "equity": "10000", "last_equity": "10000"}
        if name == "get_all_positions":
            return {"result": [
                {"symbol": s, "qty": str(p["qty"]), "avg_entry_price": str(p["avg_price"])}
                for s, p in self.positions.items()
            ]}
        if name == "get_orders":
            return {"result": []}
        if name == "get_clock":
            minutes = self.minutes_left.pop(0) if len(self.minutes_left) > 1 else self.minutes_left[0]
            return clock_with(minutes)
        if name == "get_stock_snapshot":
            symbol = arguments["symbols"]
            return {symbol: {
                "latestTrade": {"p": 100.0},
                "dailyBar": {"o": 99.0, "v": 5e6},
                "prevDailyBar": {"c": 99.0, "v": 5e6},
            }}
        if name == "get_stock_bars":
            symbol = arguments["symbols"]
            return {"bars": {symbol: [{"c": 100 + i * 0.1, "v": 10_000} for i in range(60)]}}
        if name == "place_stock_order":
            self.orders.append(arguments)
            return {"id": f"order-{len(self.orders)}"}
        if name == "get_order_by_id":
            return {"status": "filled", "filled_qty": "1", "filled_avg_price": "100.0"}
        raise AssertionError(f"unexpected tool {name}")


@pytest.fixture
def wired(monkeypatch):
    """Patches every outbound edge: Alpaca tools, the stock picker, sleeping and the journal."""
    fake = FakeBroker()
    fake.picks = 0

    for module in (broker, universe, technical):
        monkeypatch.setattr(module, "call_tool", fake.call_tool)

    async def fake_pick(table, hint):
        fake.picks += 1
        return Watchlist(symbols=WATCHLIST, reason="test"), None

    async def no_sleep(seconds):
        return None

    monkeypatch.setattr(nodes, "pick_stocks", fake_pick)
    monkeypatch.setattr(nodes.asyncio, "sleep", no_sleep)  # no waiting between cycles in tests
    monkeypatch.setattr(nodes, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(broker, "_FILL_POLL_SECONDS", 0)
    return fake


def _initial_state(minutes_to_close=300.0):
    return {
        "run_id": "test_run",
        "run_config": {"session_capital": 10000, "universe_hint": "", "session_date": "2026-09-22"},
        "objective_constraints": load_objective_and_constraints(session_capital=10000),
        "starting_equity": 10000.0,
        "account": {
            "cash": 10000.0,
            "equity": 10000.0,
            "positions": {},
            "last_prices": {},
            "open_orders": [],
            "clock": clock_with(minutes_to_close),
            "daily_pnl_pct": 0.0,
            "minutes_to_close": minutes_to_close,
        },
        "performance": {
            "realized_pnl": 0.0, "unrealized_pnl": 0.0, "return_pct": 0.0,
            "wins": 0, "losses": 0, "scratches": 0, "trade_count": 0, "recent_trades": [],
        },
        "cycle_index": 0,
        "session_done": False,
    }


def _stub_decision(monkeypatch, action, confidence=0.9, qty=1):
    async def fake_decide(symbol, **kwargs):
        return {
            "symbol": symbol, "action": action, "confidence": confidence,
            "rationale": "stub", "target_qty": qty if action != "HOLD" else None,
        }

    monkeypatch.setattr(nodes, "decide", fake_decide)


def _record_sleeps(monkeypatch):
    slept = []

    async def record_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(nodes.asyncio, "sleep", record_sleep)
    return slept


# --- whole-graph runs -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_buy_cycle_places_orders_then_the_session_closes(wired, monkeypatch):
    """Cycle 1 has room and trades; cycle 2 is inside the close buffer, so it can't buy."""
    _stub_decision(monkeypatch, "BUY")
    # 4 values: Pick Stocks now reads a fresh clock too, ahead of Market Data's own read.
    wired.minutes_left = [60, 60, 60, 10]

    graph = build_graph(client=None)
    final = await graph.ainvoke(_initial_state(minutes_to_close=60.0), config={"recursion_limit": 100})

    buys = [o for o in wired.orders if o["side"] == "buy"]
    assert sorted(o["symbol"] for o in buys) == WATCHLIST
    assert final["session_done"] is True
    assert "market close" in final["rejected_orders"][0]["reason"]
    assert wired.picks == 1, "no re-pick inside the first hour"


@pytest.mark.asyncio
async def test_no_buy_is_allowed_inside_the_close_buffer(wired, monkeypatch):
    _stub_decision(monkeypatch, "BUY")
    wired.minutes_left = [10]

    graph = build_graph(client=None)
    final = await graph.ainvoke(_initial_state(minutes_to_close=10.0), config={"recursion_limit": 100})

    assert final["session_done"] is True
    assert wired.orders == []
    assert all(r["reason"] for r in final["rejected_orders"])


@pytest.mark.asyncio
async def test_open_positions_are_sold_at_the_close(wired, monkeypatch):
    _stub_decision(monkeypatch, "HOLD")
    wired.positions = {"AAPL": {"qty": 5, "avg_price": 95.0}}
    wired.minutes_left = [5]

    graph = build_graph(client=None)
    final = await graph.ainvoke(_initial_state(minutes_to_close=5.0), config={"recursion_limit": 100})

    sells = [o for o in wired.orders if o["side"] == "sell"]
    assert sells, "close must sell the open position"
    assert sells[0]["symbol"] == "AAPL"
    assert sells[0]["qty"] == "5.0"
    # +5 per share on 5 shares, realized at the 100.0 fill against a 95.0 basis.
    assert final["performance"]["realized_pnl"] == pytest.approx(25.0)


@pytest.mark.asyncio
async def test_stocks_are_re_picked_after_an_hour(wired, monkeypatch):
    _stub_decision(monkeypatch, "HOLD")
    wired.minutes_left = [300, 230, 10]  # 70 minutes pass between cycle 1 and cycle 2

    graph = build_graph(client=None)
    await graph.ainvoke(_initial_state(minutes_to_close=300.0), config={"recursion_limit": 100})

    assert wired.picks == 2, "picked at the start, then once more after an hour"


@pytest.mark.asyncio
async def test_no_watchlist_means_no_trading_but_still_a_clean_close(wired, monkeypatch):
    """If the AI can't produce a valid pick, positions are still sold and the run ends."""

    async def failing_pick(table, hint):
        return None, "model returned no structured output"

    monkeypatch.setattr(nodes, "pick_stocks", failing_pick)
    wired.positions = {"AAPL": {"qty": 2, "avg_price": 90.0}}

    graph = build_graph(client=None)
    final = await graph.ainvoke(_initial_state(), config={"recursion_limit": 100})

    assert final["pick_failed"] is True
    assert [o["side"] for o in wired.orders] == ["sell"], "positions sold despite the failure"


# --- individual nodes -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failed_hourly_re_pick_keeps_the_current_watchlist(wired, monkeypatch):
    async def failing_pick(table, hint):
        return None, "rate limited"

    monkeypatch.setattr(nodes, "pick_stocks", failing_pick)
    wired.minutes_left = [200]  # a fresh clock read, not the stale one from before the last wait
    state = {**_initial_state(minutes_to_close=9999.0), "watchlist": ["AAPL"]}

    result = await nodes.n_pick_stocks(None, state)

    assert "pick_failed" not in result and "watchlist" not in result
    assert result["picked_at_minutes_left"] == 200.0, "uses the fresh reading, not the stale one"


@pytest.mark.asyncio
async def test_pick_stocks_refreshes_the_account_instead_of_trusting_the_stale_one(wired):
    """
    A re-pick runs right after Log + Wait's sleep; state["account"] is still
    the reading from before that sleep. Using it would misjudge how close the
    market close is by up to one cycle.
    """
    wired.minutes_left = [42]
    stale_state = {**_initial_state(minutes_to_close=9999.0), "watchlist": ["AAPL"]}

    result = await nodes.n_pick_stocks(None, stale_state)

    assert result["account"]["minutes_to_close"] == 42.0
    assert result["picked_at_minutes_left"] == 42.0


@pytest.mark.asyncio
async def test_a_held_stock_stays_a_candidate_after_leaving_the_watchlist(wired):
    wired.positions = {"TSLA": {"qty": 3, "avg_price": 100.0}}

    result = await nodes.n_market_data(None, {**_initial_state(), "watchlist": ["AAPL"]})

    assert result["candidates"] == ["AAPL", "TSLA"]
    assert set(result["measurements"]) == {"AAPL", "TSLA"}


@pytest.mark.asyncio
async def test_first_cycle_can_trade(wired, monkeypatch):
    """The risk gate needs prices; Market Data supplies them before any decision is made."""
    _stub_decision(monkeypatch, "BUY")
    wired.minutes_left = [60]

    state = {**_initial_state(), "watchlist": WATCHLIST}
    assert state["account"]["last_prices"] == {}, "cycle 1 genuinely starts with no quotes"

    state = {**state, **await nodes.n_market_data(None, state)}
    state = {**state, **await nodes.n_trading_agent(state)}
    gated = await nodes.n_risk_gate(state)

    assert gated["approved_orders"], f"nothing approved: {gated['rejected_orders']}"


@pytest.mark.asyncio
async def test_a_failing_order_does_not_stop_the_remaining_orders(wired, monkeypatch):
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

    result = await nodes.n_execute(
        None,
        {
            **_initial_state(),
            "candidates": WATCHLIST,
            "approved_orders": [
                {"symbol": "AAPL", "action": "BUY", "confidence": 0.9, "target_qty": 1},
                {"symbol": "MSFT", "action": "BUY", "confidence": 0.9, "target_qty": 1},
            ],
        },
    )

    assert calls["n"] == 2, "the second order was still attempted"
    assert result["closed_trades_this_cycle"] == []


@pytest.mark.asyncio
async def test_holdings_are_only_re_read_when_orders_were_placed(wired):
    state = {**_initial_state(), "candidates": WATCHLIST, "approved_orders": []}

    result = await nodes.n_execute(None, state)

    assert result["account"] is state["account"]
    assert "get_account_info" not in wired.calls


# --- Log + Wait: waiting, closing and re-picking ---------------------------


@pytest.mark.asyncio
async def test_wait_never_runs_past_the_close_buffer(wired, monkeypatch):
    """A 5-minute wait with 17 minutes left must stop at the 15-minute buffer, not after the close."""
    slept = _record_sleeps(monkeypatch)
    state = _initial_state(minutes_to_close=17.0)

    result = await nodes.n_log_wait(state)

    assert slept == [2 * 60], "sleeps only up to the start of the close buffer"
    assert result["session_done"] is False


@pytest.mark.asyncio
async def test_wait_is_the_full_cycle_when_there_is_room(wired, monkeypatch):
    slept = _record_sleeps(monkeypatch)

    await nodes.n_log_wait(_initial_state(minutes_to_close=300.0))

    assert slept == [5 * 60]


@pytest.mark.asyncio
async def test_no_wait_once_the_session_is_done(wired, monkeypatch):
    slept = _record_sleeps(monkeypatch)

    result = await nodes.n_log_wait(_initial_state(minutes_to_close=5.0))

    assert result["session_done"] is True
    assert slept == []


@pytest.mark.asyncio
async def test_re_pick_is_due_only_after_an_hour(wired):
    """Counted from the last pick, including the wait that is about to happen."""
    picked = {"picked_at_minutes_left": 300.0}

    soon = await nodes.n_log_wait({**_initial_state(minutes_to_close=250.0), **picked})
    later = await nodes.n_log_wait({**_initial_state(minutes_to_close=240.0), **picked})

    assert soon["repick_due"] is False  # 55 minutes
    assert later["repick_due"] is True  # 65 minutes
    assert REPICK_MINUTES == 60, "update this test if the interval changes"


@pytest.mark.asyncio
async def test_a_profit_or_loss_update_is_logged_every_cycle(wired):
    state = {
        **_initial_state(),
        "closed_trades_this_cycle": [{"symbol": "AAPL", "action": "SELL", "qty": 1, "pnl": 3.0}],
    }

    result = await nodes.n_log_wait(state)

    assert result["performance"]["realized_pnl"] == 3.0
    assert result["performance"]["wins"] == 1
    assert result["cycle_index"] == 1


def test_routing_after_log_wait():
    assert _route_after_log_wait({"session_done": True, "repick_due": True}) == "close"
    assert _route_after_log_wait({"session_done": False, "repick_due": True}) == "pick_stocks"
    assert _route_after_log_wait({"session_done": False, "repick_due": False}) == "market_data"
