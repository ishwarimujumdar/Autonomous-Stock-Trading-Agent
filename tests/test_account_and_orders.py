
from types import SimpleNamespace

import pytest

from src.alpaca.account_and_orders import (
    _daily_pnl_pct,
    execute_order,
    minutes_to_close,
    reconcile,
)
from src.alpaca.mcp_client import MCPToolError, call_tool


def test_minutes_to_close_computes_from_clock():
    clock = {
        "is_open": True,
        "timestamp": "2026-09-22T17:00:00Z",
        "next_close": "2026-09-22T20:00:00Z",
    }
    assert minutes_to_close(clock) == 180.0


def test_minutes_to_close_is_zero_when_closed():
    assert minutes_to_close({"is_open": False}) == 0.0


def test_minutes_to_close_degrades_safely_on_a_bad_clock():
    """0 stops new BUYs and triggers flattening - the safe direction."""
    assert minutes_to_close({"is_open": True}) == 0.0
    assert minutes_to_close({"is_open": True, "timestamp": "nonsense", "next_close": "x"}) == 0.0


def test_daily_pnl_derives_from_equity_against_last_equity():
    """Alpaca has no daily_pnl_pct field; reading one gave a constant 0.0."""
    assert _daily_pnl_pct({"equity": "9500", "last_equity": "10000"}) == pytest.approx(-0.05)


def test_daily_pnl_is_zero_without_a_baseline():
    assert _daily_pnl_pct({"equity": "9500", "last_equity": "0"}) == 0.0


def test_reconcile_reports_a_divergence():
    expected = {"AAPL": {"qty": 10}}
    account = {"positions": {"AAPL": {"qty": 4}}}
    assert reconcile(expected, account) == [
        {"symbol": "AAPL", "expected_qty": 10.0, "actual_qty": 4.0}
    ]


def test_reconcile_is_quiet_when_state_matches():
    expected = {"AAPL": {"qty": 10}}
    account = {"positions": {"AAPL": {"qty": 10}}}
    assert reconcile(expected, account) == []


def test_reconcile_catches_a_position_that_should_have_closed():
    account = {"positions": {"AAPL": {"qty": 10}}}
    assert reconcile({}, account) == [
        {"symbol": "AAPL", "expected_qty": 0.0, "actual_qty": 10.0}
    ]


class _FakeSession:
    """Stands in for the MCP connection, returning one canned tool result."""

    def __init__(self, structured=None, is_error=False):
        self._result = SimpleNamespace(
            isError=is_error, structuredContent=structured, content=["raw text"]
        )

    async def call_tool(self, name, arguments):
        return self._result


@pytest.mark.asyncio
async def test_call_tool_strips_the_servers_envelope():
    wrapped = {"_alpaca_mcp_security": {"note": "x"}, "data": {"cash": "100"}}
    assert await call_tool(_FakeSession(wrapped), "get_account_info", {}) == {"cash": "100"}


@pytest.mark.asyncio
async def test_call_tool_returns_an_unwrapped_payload_as_is():
    assert await call_tool(_FakeSession({"cash": "100"}), "get_account_info", {}) == {"cash": "100"}


@pytest.mark.asyncio
async def test_call_tool_fails_loudly_when_no_data_comes_back():
    """An empty result must never read as 'price 0 / market closed' and end the session quietly."""
    with pytest.raises(MCPToolError, match="no structured data"):
        await call_tool(_FakeSession(structured=None), "get_clock", {})


@pytest.mark.asyncio
async def test_call_tool_raises_on_a_tool_error():
    with pytest.raises(MCPToolError, match="failed"):
        await call_tool(_FakeSession(is_error=True), "place_stock_order", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["buy", "hold", "HOLD", "", None])
async def test_execute_order_refuses_a_non_executable_action(action):
    """`"buy" if action == "BUY" else "sell"` silently inverted these."""
    with pytest.raises(ValueError, match="non-executable action"):
        await execute_order(None, {"symbol": "AAPL", "action": action, "target_qty": 1})


@pytest.mark.asyncio
@pytest.mark.parametrize("qty", [None, 0, -1, 2.5, "3"])
async def test_execute_order_refuses_an_invalid_qty(qty):
    """A None qty used to be submitted to Alpaca as the string "None"."""
    with pytest.raises(ValueError, match="invalid target_qty"):
        await execute_order(None, {"symbol": "AAPL", "action": "BUY", "target_qty": qty})


@pytest.mark.asyncio
async def test_wait_for_fill_reports_a_missing_order_id():
    from src.alpaca.account_and_orders import wait_for_fill

    result = await wait_for_fill(None, None)
    assert result["status"] == "unknown"


@pytest.mark.asyncio
async def test_wait_for_fill_returns_terminal_status(monkeypatch):
    import src.alpaca.account_and_orders as broker

    async def fake_call_tool(client, name, arguments):
        return {"status": "filled", "filled_qty": "5", "filled_avg_price": "101.5"}

    monkeypatch.setattr(broker, "call_tool", fake_call_tool)
    result = await broker.wait_for_fill(None, "order-1")
    assert result == {"status": "filled", "filled_qty": 5.0, "filled_avg_price": 101.5}


@pytest.mark.asyncio
async def test_wait_for_fill_degrades_when_polling_is_unsupported(monkeypatch):
    import src.alpaca.account_and_orders as broker

    async def fake_call_tool(client, name, arguments):
        raise MCPToolError("no such tool: get_order_by_id")

    monkeypatch.setattr(broker, "call_tool", fake_call_tool)
    result = await broker.wait_for_fill(None, "order-1")
    assert result["status"] == "unknown"
    assert "could not poll" in result["reason"]
