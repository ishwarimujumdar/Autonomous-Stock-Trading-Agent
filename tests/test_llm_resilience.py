"""
A raw provider failure (rate limit surviving the client's own retries, a
truncated generation Groq rejects server-side with a 400, a transient
outage) must degrade gracefully, not crash the session. `include_raw=True`
only catches a PARSING failure into `parsing_error` - it does nothing for an
exception raised by the API call itself, which is exactly what happened live:
Groq truncated a strategy proposal mid-JSON and rejected the request with a
400 `BadRequestError`, which propagated uncaught out of `propose_strategy()`
and killed the whole run.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.config import load_objective_and_constraints
from src.strategy.schema import validate_strategy_proposal


@pytest.mark.asyncio
async def test_propose_strategy_survives_a_raw_api_error():
    from src.strategy.strategy_agent import propose_strategy

    with patch("src.strategy.strategy_agent.get_llm") as mock_get_llm:
        mock_llm = mock_get_llm.return_value.with_structured_output.return_value
        mock_llm.ainvoke = AsyncMock(
            side_effect=RuntimeError("BadRequestError: tool_use_failed (truncated JSON)")
        )

        result = await propose_strategy(
            objective_constraints=load_objective_and_constraints(10000),
            run_config={"universe_hint": "test", "session_capital": 10000},
            strategy_history=[],
        )

    assert "_unparseable" in result
    # Must be something the schema validator cleanly rejects - not a crash,
    # not a dict that happens to validate by accident.
    validated, error = validate_strategy_proposal(result)
    assert validated is None
    assert error


@pytest.mark.asyncio
async def test_decide_survives_a_raw_api_error():
    from src.decision.decision import decide

    with patch("src.decision.decision.get_llm") as mock_get_llm:
        mock_llm = mock_get_llm.return_value.with_structured_output.return_value
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

        decision = await decide(
            symbol="AAPL",
            strategy={"entry_logic": "x", "exit_logic": "y"},
            technical={},
            narrative=None,
            previous_outcomes=[],
            performance={},
            account={},
            timeframe="5Min",
            objective_constraints=load_objective_and_constraints(10000),
        )

    assert decision["action"] == "HOLD"
    assert decision["target_qty"] is None
    assert decision["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_analyze_news_survives_a_raw_api_error(monkeypatch):
    import src.analysis.market_news as market_news

    async def fake_call_tool(client, name, arguments):
        return {"news": [{"headline": "Something happened"}]}

    monkeypatch.setattr(market_news, "call_tool", fake_call_tool)
    market_news.clear_cache()

    with patch("src.analysis.market_news.get_llm") as mock_get_llm:
        mock_get_llm.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

        result = await market_news.analyze_news(
            client=None, symbol="AAPL", strategy_name="momentum", signals=["news_sentiment"]
        )

    assert result["symbol"] == "AAPL"
    assert result["narrative"] is not None
    # A failure must not be cached - it should retry next cycle, not stay
    # "unavailable" for the full TTL.
    assert "AAPL" not in market_news._cache
