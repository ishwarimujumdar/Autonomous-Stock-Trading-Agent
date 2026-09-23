from typing import Literal, Optional, TypedDict


class RunConfig(TypedDict):
    session_capital: float  # capital allocated to this session
    universe_hint: str  # free-text user hint, e.g. "liquid US tech stocks"
    session_date: str  # ISO date of today's trading session


class StrategySpec(TypedDict):
    """Strategy Agent's output, after schema validation."""

    name: str  # must be in SUPPORTED_STRATEGIES
    signals: list[str]  # must all be in SUPPORTED_SIGNALS
    universe_symbols: list[str]  # must all be in TRADABLE_UNIVERSE
    universe_criteria: dict  # keys must be in SUPPORTED_UNIVERSE_CRITERIA
    cadence_minutes: int
    entry_logic: str
    exit_logic: str
    rationale: str


class StrategyHistory(TypedDict, total=False):
    strategy: StrategySpec
    strategy_valid: bool
    strategy_validation_error: str  # set when strategy_valid is False (rejected)
    reason: str  # set when strategy_valid is True (why it was replaced)
    performance_at_switch: dict  # set when strategy_valid is True


class TradeDecision(TypedDict):
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    rationale: str
    target_qty: Optional[int]


class PerformanceState(TypedDict):
    realized_pnl: float
    unrealized_pnl: float
    return_pct: float
    wins: int
    losses: int
    scratches: int
    trade_count: int
    recent_trades: list[dict]


class TradingState(TypedDict, total=False):
    run_id: str
    run_config: RunConfig
    objective_constraints: dict
    # Account equity when the session started, so return is measured against
    # this session's activity rather than the account's whole history.
    starting_equity: float

    strategy: StrategySpec
    strategy_valid: bool
    strategy_history: list[StrategyHistory]
    strategy_validation_error: str
    strategy_attempts: int  # bounded retry counter for the validator loop
    strategy_failed: bool

    candidates: list[str]
    research: dict  # symbol -> {"technical": {...}, "narrative": {...}}
    decisions: list[TradeDecision]
    approved_orders: list[TradeDecision]
    rejected_orders: list[dict]  # decision + rejection reason

    account: dict  # cash, positions, last_prices, clock, minutes_to_close
    performance: PerformanceState
    closed_trades_this_cycle: list[dict]  # transient: SELLs filled this cycle
    expected_positions: dict  # transient: post-fill expectation, for reconciliation
    reconciliation_divergences: list[dict]
    positions_still_open: dict  # set by finalize if a flatten didn't complete

    cycle_index: int
    minutes_to_close: float
    rejected_cycle_streak: int  # consecutive cycles where nothing was approved
    session_done: bool
    restrategize: bool
    restrategize_reason: str
