_RECENT_TRADES_KEPT = 10
_LOSS_WINDOW = 4
_LOSSES_IN_WINDOW = 3
_REJECTION_CYCLE_STREAK = 3


def position_market_value(account: dict) -> float:
    prices = account.get("last_prices", {})
    return sum(
        pos.get("qty", 0) * prices.get(symbol, pos.get("avg_price", 0))
        for symbol, pos in account.get("positions", {}).items()
    )


def position_cost_basis(account: dict) -> float:
    return sum(
        pos.get("qty", 0) * pos.get("avg_price", 0)
        for pos in account.get("positions", {}).values()
    )


def compute_performance(
    account: dict,
    session_capital: float,
    starting_equity: float,
    new_trades: list[dict],
    prior: dict | None = None,
) -> dict:
    """
    Folds this cycle's closed trades into the running performance state.

    Totals are accumulated, not recomputed from `recent_trades`. That list is
    truncated for display and for prompt size, and re-deriving realized P&L and
    the win/loss counts from it meant they silently *shrank* once more than ten
    trades had been made - which also fed the restrategize trigger.
    """
    prior = prior or {}
    new_trades = new_trades or []

    realized_pnl = prior.get("realized_pnl", 0.0) + sum(t.get("pnl", 0.0) for t in new_trades)
    wins = prior.get("wins", 0) + sum(1 for t in new_trades if t.get("pnl", 0.0) > 0)
    losses = prior.get("losses", 0) + sum(1 for t in new_trades if t.get("pnl", 0.0) < 0)
    scratches = prior.get("scratches", 0) + sum(1 for t in new_trades if t.get("pnl", 0.0) == 0)

    unrealized_pnl = position_market_value(account) - position_cost_basis(account)

    # Return is measured on the capital allocated to this session, against the
    # equity the account started the session with - so it doesn't report a
    # nine-fold return just because the account holds more than was allocated.
    equity = account.get("equity", account.get("portfolio_value", starting_equity))
    return_pct = (
        (equity - starting_equity) / session_capital * 100 if session_capital > 0 else 0.0
    )

    return {
        "realized_pnl": round(realized_pnl, 4),
        "unrealized_pnl": round(unrealized_pnl, 4),
        "return_pct": round(return_pct, 4),
        "wins": wins,
        "losses": losses,
        # A flat trade is neither a win nor a loss; counting it as a loss
        # skewed both the win rate and the restrategize trigger.
        "scratches": scratches,
        "trade_count": prior.get("trade_count", 0) + len(new_trades),
        "recent_trades": (prior.get("recent_trades", []) + new_trades)[-_RECENT_TRADES_KEPT:],
    }


def should_restrategize(
    performance: dict,
    rejected_cycle_streak: int,
) -> tuple[bool, str | None]:
    """
    Slow-loop trigger conditions, within today's session only. No fabricated
    'target pace' - only observable facts: repeated losses or repeated
    rejections.

    `rejected_cycle_streak` counts *consecutive cycles* in which every
    decision was rejected. The previous trigger counted rejections within a
    single cycle, so one busy cycle with a few rejections looked identical to
    a strategy that had been failing the gate all session.
    """
    trades = performance.get("recent_trades", [])
    if len(trades) >= _LOSS_WINDOW:
        window = trades[-_LOSS_WINDOW:]
        if sum(1 for t in window if t.get("pnl", 0) < 0) >= _LOSSES_IN_WINDOW:
            return True, (
                f"{_LOSSES_IN_WINDOW} of last {_LOSS_WINDOW} trades were losses - "
                "strategy performance deteriorating"
            )

    if rejected_cycle_streak >= _REJECTION_CYCLE_STREAK:
        return True, (
            f"{rejected_cycle_streak} consecutive cycles with every trade rejected "
            "by the risk gate"
        )

    return False, None
