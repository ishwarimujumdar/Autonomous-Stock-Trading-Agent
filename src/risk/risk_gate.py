from dataclasses import dataclass


@dataclass
class CycleBudget:
    """
    Running tally of what a single cycle has already committed.

    Each decision used to be checked against the same pre-trade snapshot, so
    five independent 20%-of-capital BUYs all passed a 20% cap and deployed
    100% at once. Approvals now draw down a shared budget.

    Doesn't track per-symbol quantity: `n_trade_decision` produces at most one
    decision per symbol per cycle, so by the time a symbol is evaluated here,
    no earlier decision in the same batch could have touched it.
    """

    remaining_cash: float
    gross_exposure: float

    def commit(self, trade_value: float, action: str) -> None:
        if action == "BUY":
            self.remaining_cash -= trade_value
            self.gross_exposure += trade_value
        else:
            self.gross_exposure = max(self.gross_exposure - trade_value, 0.0)


def _is_within_market_hours(clock: dict) -> bool:
    return bool(clock.get("is_open", False))


def _session_loss_pct(performance: dict, session_capital: float) -> float:
    """Session drawdown as a fraction of allocated capital (negative = loss)."""
    if session_capital <= 0:
        return 0.0
    pnl = performance.get("realized_pnl", 0.0) + performance.get("unrealized_pnl", 0.0)
    return pnl / session_capital


def evaluate(
    decision: dict,
    objective_constraints: dict,
    account: dict,
    performance: dict | None = None,
    budget: CycleBudget | None = None,
) -> tuple[bool, str | None]:
    """
    Deterministic enforcement of the fixed constraint set. Returns
    (approved, rejection_reason). Nothing here is negotiable by any LLM node -
    this is the boundary the Strategy/Decision agents cannot cross.

    The clock and minutes-to-close are read off the account snapshot rather
    than passed separately: they were previously two parameters describing the
    same thing, and the one that was passed could be unset on the first cycle.
    """
    constraints = objective_constraints["constraints"]
    performance = performance or {}
    clock = account.get("clock", {})

    if constraints.get("market_hours_only", True) and not _is_within_market_hours(clock):
        return False, "market is closed"

    action = decision.get("action")
    if action == "HOLD":
        return True, None
    if action not in ("BUY", "SELL"):
        return False, f"unrecognised action {action!r} - expected BUY, SELL or HOLD"

    confidence = decision.get("confidence")
    if not isinstance(confidence, (int, float)):
        return False, f"missing or non-numeric confidence {confidence!r}"
    if confidence < constraints["minimum_confidence"]:
        return False, (
            f"confidence {confidence:.2f} below fixed minimum "
            f"{constraints['minimum_confidence']:.2f}"
        )

    # A missing quantity used to price the trade at 0, sail through every size
    # check, and reach Alpaca as the literal string "None".
    qty = decision.get("target_qty")
    if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
        return False, f"{action} requires a positive whole target_qty, got {qty!r}"

    symbol = decision["symbol"]
    price = account.get("last_prices", {}).get(symbol, 0)
    if not price or price <= 0:
        return False, f"no current price available for {symbol} - cannot size or check the trade"

    minutes_left = account.get("minutes_to_close")
    if minutes_left is None:
        return False, "minutes_to_close unavailable - refusing to trade blind to the close"

    # Day-trading rule: no new exposure once we're inside the close buffer -
    # everything must be flat before today's session ends.
    close_buffer = constraints.get("market_close_buffer_minutes", 15)
    if minutes_left <= close_buffer and action == "BUY":
        return False, (
            f"within {close_buffer} min of market close - only SELL/HOLD permitted"
        )

    session_capital = constraints["session_capital"]
    trade_value = qty * price
    held_qty = account.get("positions", {}).get(symbol, {}).get("qty", 0)

    if action == "SELL":
        # The gate only ever checked BUYs, so a hallucinated SELL opened a
        # short - which this project does not do.
        if held_qty <= 0:
            return False, f"no long position in {symbol} to sell (held {held_qty:g})"
        if qty > held_qty + 1e-6:
            return False, (
                f"SELL {qty} exceeds position of {held_qty:g} in {symbol} - "
                "shorting is not permitted"
            )
        return True, None

    # --- BUY-only checks below ---
    if trade_value > session_capital * constraints["max_trade_size_pct"]:
        return False, (
            f"trade value {trade_value:.2f} exceeds max_trade_size_pct "
            f"({constraints['max_trade_size_pct']:.0%} of session capital)"
        )

    available_cash = budget.remaining_cash if budget else account.get("cash", 0)
    if trade_value > available_cash:
        return False, (
            f"insufficient buying power: {trade_value:.2f} needed, "
            f"{available_cash:.2f} uncommitted this cycle"
        )

    if (held_qty + qty) * price > session_capital * constraints["max_position_pct"]:
        return False, (
            f"resulting position exceeds max_position_pct "
            f"({constraints['max_position_pct']:.0%} of session capital)"
        )

    max_gross = constraints.get("max_gross_exposure_pct", 1.0)
    current_gross = budget.gross_exposure if budget else _held_value(account)
    if current_gross + trade_value > session_capital * max_gross:
        return False, (
            f"gross exposure {current_gross + trade_value:.2f} exceeds "
            f"max_gross_exposure_pct ({max_gross:.0%} of session capital)"
        )

    session_loss_pct = _session_loss_pct(performance, session_capital)
    if session_loss_pct <= -constraints["max_daily_loss_pct"]:
        return False, (
            f"daily loss limit reached ({session_loss_pct:.2%} <= "
            f"-{constraints['max_daily_loss_pct']:.0%}), no new BUYs today"
        )

    account_daily_pnl_pct = account.get("daily_pnl_pct", 0)
    if account_daily_pnl_pct <= -constraints["max_daily_loss_pct"]:
        return False, (
            f"account-level daily loss limit reached ({account_daily_pnl_pct:.2%} <= "
            f"-{constraints['max_daily_loss_pct']:.0%}), no new BUYs today"
        )

    return True, None


def _held_value(account: dict) -> float:
    prices = account.get("last_prices", {})
    return sum(
        abs(pos.get("qty", 0)) * prices.get(symbol, pos.get("avg_price", 0))
        for symbol, pos in account.get("positions", {}).items()
    )


def evaluate_batch(
    decisions: list[dict],
    objective_constraints: dict,
    account: dict,
    performance: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Evaluates a whole cycle's decisions against one shared budget, so approvals
    earlier in the list constrain the ones after them.

    Returns (approved, rejected). Highest-confidence decisions are considered
    first - when the budget can't fund everything, it should fund the trades
    the decision node believed in most, not whichever symbol sorted first.
    """
    budget = CycleBudget(
        remaining_cash=account.get("cash", 0),
        gross_exposure=_held_value(account),
    )

    approved, rejected = [], []
    ordered = sorted(decisions, key=lambda d: d.get("confidence") or 0, reverse=True)

    for decision in ordered:
        ok, reason = evaluate(decision, objective_constraints, account, performance, budget)
        if not ok:
            rejected.append({"decision": decision, "reason": reason})
            continue
        if decision.get("action") == "HOLD":
            continue
        price = account.get("last_prices", {}).get(decision["symbol"], 0)
        budget.commit(decision["target_qty"] * price, decision["action"])
        approved.append(decision)

    return approved, rejected
