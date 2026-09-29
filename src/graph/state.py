from typing import Literal, TypedDict


class RunConfig(TypedDict):
    session_capital: float  # capital allocated to this session
    universe_hint: str  # free-text user hint, e.g. "liquid US tech stocks"
    session_date: str  # ISO date of today's trading session


class TradeDecision(TypedDict):
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    rationale: str
    target_qty: int | None


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
    # Account equity at session start, so return measures this session only.
    starting_equity: float

    # Pick Stocks
    watchlist: list[str]  # the AI's chosen stocks
    picked_at_minutes_left: float  # minutes to close when the list was last picked
    pick_failed: bool  # no valid watchlist could be made: go straight to Close

    # Market Data
    account: dict  # cash, positions, last_prices, clock, minutes_to_close
    candidates: list[str]  # watchlist plus anything currently held
    measurements: dict  # symbol -> price change and volume figures

    # Trading Agent, Risk Gate, Execute
    decisions: list[TradeDecision]
    approved_orders: list[TradeDecision]
    rejected_orders: list[dict]  # decision + rejection reason
    closed_trades_this_cycle: list[dict]  # SELLs filled this cycle, for the profit/loss tally

    # Log + Wait
    performance: PerformanceState
    cycle_index: int
    session_done: bool
    repick_due: bool

    # Close
    positions_still_open: dict  # set if the final sell-off didn't complete
