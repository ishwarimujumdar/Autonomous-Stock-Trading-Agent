from src.strategy.schema import StrategyProposal, validate_strategy_proposal


def _valid_proposal(**overrides):
    base = {
        "name": "momentum",
        "signals": ["price_momentum", "volume"],
        "universe_symbols": ["AAPL", "MSFT"],
        "universe_criteria": {"min_avg_daily_volume": 1_000_000, "min_price": 5},
        "cadence_minutes": 15,
        "entry_logic": "strong intraday momentum + volume confirmation",
        "exit_logic": "momentum deterioration",
        "rationale": "test",
    }
    base.update(overrides)
    return base


def test_accepts_valid_proposal():
    validated, error = validate_strategy_proposal(_valid_proposal())
    assert error is None
    assert validated["name"] == "momentum"


def test_rejects_unsupported_strategy():
    validated, error = validate_strategy_proposal(_valid_proposal(name="scalp_the_house"))
    assert validated is None
    assert "scalp_the_house" in error


def test_rejects_unsupported_signal():
    validated, error = validate_strategy_proposal(_valid_proposal(signals=["astrology"]))
    assert validated is None
    assert "astrology" in error


def test_rejects_empty_signals():
    validated, error = validate_strategy_proposal(_valid_proposal(signals=[]))
    assert validated is None
    assert error


def test_rejects_unsupported_universe_criteria_key():
    """These used to be dropped silently, so the filter never ran."""
    validated, error = validate_strategy_proposal(
        _valid_proposal(universe_criteria={"min_short_interest": 5})
    )
    assert validated is None
    assert "Unsupported universe_criteria keys" in error


def test_rejects_non_numeric_criteria_value():
    validated, error = validate_strategy_proposal(
        _valid_proposal(universe_criteria={"min_price": "cheap"})
    )
    assert validated is None
    assert "must be numeric" in error


def test_rejects_symbol_outside_tradable_universe():
    validated, error = validate_strategy_proposal(
        _valid_proposal(universe_symbols=["AAPL", "GME"])
    )
    assert validated is None
    assert "GME" in error


def test_rejects_empty_universe():
    validated, error = validate_strategy_proposal(_valid_proposal(universe_symbols=[]))
    assert validated is None
    assert error


def test_rejects_universe_over_the_concurrency_budget():
    """
    Each universe symbol costs a decide() call every cycle against a
    tokens-per-minute-limited API - unbounded here is how a cycle blew the
    Groq TPM limit in practice.
    """
    from src.config import MAX_UNIVERSE_SYMBOLS, TRADABLE_UNIVERSE

    too_many = TRADABLE_UNIVERSE[: MAX_UNIVERSE_SYMBOLS + 1]
    validated, error = validate_strategy_proposal(_valid_proposal(universe_symbols=too_many))
    assert validated is None
    assert error


def test_rejects_unparseable_proposal_without_raising():
    """The retry loop needs a rejection, not an exception."""
    validated, error = validate_strategy_proposal({"_unparseable": "model returned prose"})
    assert validated is None
    assert error


def test_schema_carries_enums_for_constrained_decoding():
    """
    The vocabularies must reach the model as real JSON-Schema enums, not just
    as prose in a description, or constrained decoding can't steer on them.
    """
    from src.config import SUPPORTED_SIGNALS, SUPPORTED_STRATEGIES, TRADABLE_UNIVERSE

    properties = StrategyProposal.model_json_schema()["properties"]

    assert properties["name"]["enum"] == SUPPORTED_STRATEGIES
    assert properties["signals"]["items"]["enum"] == SUPPORTED_SIGNALS
    assert properties["universe_symbols"]["items"]["enum"] == TRADABLE_UNIVERSE
