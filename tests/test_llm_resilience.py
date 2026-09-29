"""
A raw provider failure (rate limit surviving the client's own retries, a
truncated generation rejected server-side, a transient outage) must degrade
gracefully, not crash the session. `include_raw=True` only catches a PARSING
failure into `parsing_error` - it does nothing for an exception raised by the
API call itself.
"""

import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from groq import RateLimitError

from src.config import MAX_PICK_ATTEMPTS, load_objective_and_constraints
from src.picker.stock_picker import Watchlist, pick_stocks

TABLE = [
    {"symbol": "AAPL", "price": 100.0, "change_pct": 0.5, "range_pct": 1.0, "yesterday_volume": 1_000_000}
]


def _rate_limit_error(retry_after: str | None = "0.05") -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://api.groq.com/x"), headers=headers
    )
    return RateLimitError("rate limited", response=response, body=None)


@pytest.mark.asyncio
async def test_pick_stocks_waits_before_retrying_a_rate_limit():
    """
    Retrying inside the same token window just fails again (seen live: Groq
    said "try again in 5.145s" and an immediate retry hit the same limit).
    """
    from src.picker.stock_picker import pick_stocks

    with patch("src.picker.stock_picker.get_llm") as mock_get_llm:
        mock_llm = mock_get_llm.return_value.with_structured_output.return_value
        mock_llm.ainvoke = AsyncMock(side_effect=_rate_limit_error("0.2"))

        started = time.monotonic()
        watchlist, error = await pick_stocks(TABLE, "")
        elapsed = time.monotonic() - started

    assert watchlist is None
    assert "RateLimitError" in error
    # 2 waits between 3 attempts, at the server's stated 0.2s each.
    assert elapsed >= 0.4, f"only waited {elapsed:.2f}s - retried without backing off"


@pytest.mark.asyncio
async def test_pick_stocks_falls_back_to_a_default_wait_with_no_retry_after_header():
    from src.picker.stock_picker import pick_stocks

    with patch("src.picker.stock_picker.get_llm") as mock_get_llm:
        mock_llm = mock_get_llm.return_value.with_structured_output.return_value
        mock_llm.ainvoke = AsyncMock(side_effect=_rate_limit_error(retry_after=None))

        with patch("src.picker.stock_picker.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            await pick_stocks(TABLE, "")

    assert mock_sleep.call_args.args[0] > 0, "must still back off with no header to read"


@pytest.mark.asyncio
async def test_pick_stocks_survives_a_raw_api_error_and_retries():
    with patch("src.picker.stock_picker.get_llm") as mock_get_llm:
        mock_llm = mock_get_llm.return_value.with_structured_output.return_value
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("BadRequestError: truncated JSON"))

        watchlist, error = await pick_stocks(TABLE, "")

    assert watchlist is None
    assert "LLM call failed" in error and "truncated JSON" in error
    assert mock_llm.ainvoke.call_count == MAX_PICK_ATTEMPTS, "bounded retries, not a crash or a loop"


@pytest.mark.asyncio
async def test_pick_stocks_shows_the_model_why_its_last_answer_was_rejected():
    bad = {"parsed": None, "parsing_error": "symbols: 'GME' is not a valid stock"}
    good = {"parsed": Watchlist(symbols=["AAPL"], reason="liquid"), "parsing_error": None}

    with patch("src.picker.stock_picker.get_llm") as mock_get_llm:
        mock_llm = mock_get_llm.return_value.with_structured_output.return_value
        mock_llm.ainvoke = AsyncMock(side_effect=[bad, good])

        watchlist, error = await pick_stocks(TABLE, "")

        second_call_messages = mock_llm.ainvoke.call_args_list[1].args[0]

    assert error is None
    assert watchlist.symbols == ["AAPL"]
    assert "'GME' is not a valid stock" in second_call_messages[1]["content"]


@pytest.mark.asyncio
async def test_decide_survives_a_raw_api_error():
    from src.decision.decision import decide

    with patch("src.decision.decision.get_llm") as mock_get_llm:
        mock_llm = mock_get_llm.return_value.with_structured_output.return_value
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

        decision = await decide(
            symbol="AAPL",
            technical={},
            previous_outcomes=[],
            performance={},
            account={},
            objective_constraints=load_objective_and_constraints(10000),
        )

    assert decision["action"] == "HOLD"
    assert decision["target_qty"] is None
    assert decision["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_decide_turns_unparseable_output_into_a_hold():
    from src.decision.decision import decide

    with patch("src.decision.decision.get_llm") as mock_get_llm:
        mock_llm = mock_get_llm.return_value.with_structured_output.return_value
        mock_llm.ainvoke = AsyncMock(
            return_value={"parsed": None, "parsing_error": "bad output"}
        )
        decision = await decide("AAPL", {}, [], {}, {}, load_objective_and_constraints(10000))

    assert decision["action"] == "HOLD"
