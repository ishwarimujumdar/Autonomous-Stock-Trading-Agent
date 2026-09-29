from src.config import load_objective_and_constraints
from src.decision.decision import _max_buy_qty

OC = load_objective_and_constraints(session_capital=10000)  # 20% trade cap, 30% position cap


def _account(**overrides):
    base = {"last_prices": {"AMD": 165.0}, "positions": {}}
    base.update(overrides)
    return base


def test_decide_asks_for_enough_tokens_to_finish_reasoning():
    """
    Groq 400'd with "Tool choice is required, but model did not call a tool"
    at max_tokens=300 - the model ran out mid-reasoning, before the output
    tool call. Confirmed live: 5 straight decisions on a held stock fell back
    to HOLD this way while its stop-loss should have been checked.
    """
    import asyncio
    from unittest.mock import patch

    from src.config import load_objective_and_constraints
    from src.decision.decision import decide

    with patch("src.decision.decision.get_llm") as mock_get_llm:
        asyncio.run(
            decide("AAPL", {}, [], {}, {}, load_objective_and_constraints(10000))
        )
        _, kwargs = mock_get_llm.call_args
        assert kwargs["max_tokens"] >= 1200, "too tight a budget starved a reasoning model before"


def test_max_buy_qty_matches_the_trade_size_cap():
    """
    Confirmed live: the model proposed 16 AMD shares ($9,890, ~5x the $2,000
    trade-size cap) on a genuine bullish signal and lost the trade entirely to
    an outright Risk Gate rejection. This is what it should have been told.
    """
    qty = _max_buy_qty("AMD", _account(), OC)
    # $2,000 cap / $165 = 12 shares
    assert qty == 12


def test_max_buy_qty_shrinks_with_an_existing_holding():
    account = _account(positions={"AMD": {"qty": 15, "avg_price": 160.0}})
    # $3,000 position cap - $2,400 already held = $600 of room / $165 = 3 shares,
    # tighter than the $2,000 trade-size cap here.
    qty = _max_buy_qty("AMD", account, OC)
    assert qty == 3


def test_max_buy_qty_is_zero_once_position_cap_is_full():
    account = _account(positions={"AMD": {"qty": 20, "avg_price": 165.0}})
    assert _max_buy_qty("AMD", account, OC) == 0


def test_max_buy_qty_is_zero_without_a_price():
    assert _max_buy_qty("ZZZZ", _account(), OC) == 0
