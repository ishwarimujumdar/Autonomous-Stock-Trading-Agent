from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.config import (
    MAX_UNIVERSE_SYMBOLS,
    SUPPORTED_SIGNALS,
    SUPPORTED_STRATEGIES,
    SUPPORTED_UNIVERSE_CRITERIA,
    TRADABLE_UNIVERSE,
)

# Literal aliases built from the config lists, so the JSON schema handed to the
# model carries the enums directly (constrained decoding) and pydantic itself
# rejects an out-of-list value - no hand-written membership check needed.
StrategyName = Literal[tuple(SUPPORTED_STRATEGIES)]  # type: ignore[valid-type]
SignalName = Literal[tuple(SUPPORTED_SIGNALS)]  # type: ignore[valid-type]
UniverseSymbol = Literal[tuple(TRADABLE_UNIVERSE)]  # type: ignore[valid-type]


class StrategyProposal(BaseModel):
    """
    What the Strategy Agent must produce.

    Used directly as the `with_structured_output(..., include_raw=True)`
    schema: on a bad proposal, `include_raw` catches the validation error
    (whether from a Literal mismatch or a `field_validator` below) instead of
    raising, so a routed retry - not a crash - is what a bad proposal gets.
    """

    name: StrategyName = Field(description="Strategy name")
    signals: list[SignalName] = Field(
        min_length=1, description="Signals this strategy relies on"
    )
    universe_symbols: list[UniverseSymbol] = Field(
        min_length=1,
        max_length=MAX_UNIVERSE_SYMBOLS,
        description=(
            "Tickers the scanner should consider this session, chosen from the "
            "tradable universe and guided by the user's universe hint. "
            f"At most {MAX_UNIVERSE_SYMBOLS} - each one costs a decision-LLM call "
            "every cycle, and this system runs on a tokens-per-minute-limited API."
        ),
    )
    universe_criteria: dict = Field(
        default_factory=dict,
        description=(
            "Numeric filter criteria for the scanner. Only these keys are "
            f"implemented and any other key is a validation error: {SUPPORTED_UNIVERSE_CRITERIA}. "
            "Example: {'min_price': 5, 'min_avg_daily_volume': 1000000, "
            "'min_intraday_return_pct': 0.5}"
        ),
    )
    cadence_minutes: int = Field(gt=0, le=120, description="Intraday re-evaluation cadence")
    entry_logic: str
    exit_logic: str
    rationale: str

    @field_validator("universe_criteria")
    @classmethod
    def criteria_supported(cls, v: dict) -> dict:
        # Dict keys can't be expressed as a Literal, so this is the one check
        # that still needs to be hand-written.
        unsupported = [k for k in v if k not in SUPPORTED_UNIVERSE_CRITERIA]
        if unsupported:
            raise ValueError(
                f"Unsupported universe_criteria keys {unsupported}. The scanner only "
                f"implements: {SUPPORTED_UNIVERSE_CRITERIA}"
            )
        non_numeric = [k for k, val in v.items() if not isinstance(val, (int, float))]
        if non_numeric:
            raise ValueError(f"universe_criteria values must be numeric; got text for {non_numeric}")
        return v


def validate_strategy_proposal(raw: dict) -> tuple[dict | None, str | None]:
    """Returns (validated_dict, None) on success, or (None, error_message) on failure."""
    if not isinstance(raw, dict):
        return None, f"strategy proposal must be an object, got {type(raw).__name__}"
    try:
        proposal = StrategyProposal(**raw)
        return proposal.model_dump(), None
    except Exception as exc:  # pydantic.ValidationError or bad shape
        return None, str(exc)
