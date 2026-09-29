from dataclasses import dataclass


@dataclass
class CycleBudget:
    """
    What this cycle's approved trades have already used up: cash spent, and
    "gross exposure" (total money currently invested in shares). Without it,
    five separate 20%-of-capital BUYs each passed the 20% cap and put 100% at risk.
    """

    remaining_cash: float
    gross_exposure: float

    def commit(self, trade_value: float, action: str) -> None:
        if action == "BUY":
            self.remaining_cash -= trade_value
            self.gross_exposure += trade_value
        else:
            self.gross_exposure = max(self.gross_exposure - trade_value, 0.0)


def _held_value(account: dict) -> float:
    prices = account.get("last_prices", {})
    return sum(
        abs(pos.get("qty", 0)) * prices.get(symbol, pos.get("avg_price", 0))
        for symbol, pos in account.get("positions", {}).items()
    )


def _session_loss_pct(performance: dict, session_capital: float) -> float:
    """Session profit/loss as a fraction of allocated capital (negative = loss)."""
    if session_capital <= 0:
        return 0.0
    pnl = performance.get("realized_pnl", 0.0) + performance.get("unrealized_pnl", 0.0)
    return pnl / session_capital


def _check_basics(decision: dict, constraints: dict, account: dict) -> str | None:
    """Checks that apply to every BUY and SELL. Returns a rejection reason, or None."""
    confidence = decision.get("confidence")
    if not isinstance(confidence, (int, float)):
        return f"missing or non-numeric confidence {confidence!r}"
    if confidence < constraints["minimum_confidence"]:
        return (
            f"confidence {confidence:.2f} below fixed minimum "
            f"{constraints['minimum_confidence']:.2f}"
        )

    qty = decision.get("target_qty")
    if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
        return f"{decision['action']} requires a positive whole target_qty, got {qty!r}"

    symbol = decision["symbol"]
    price = account.get("last_prices", {}).get(symbol)
    if not price or price <= 0:
        return f"no current price available for {symbol} - cannot size or check the trade"

    minutes_left = account.get("minutes_to_close")
    if minutes_left is None:
        return "minutes_to_close unavailable - refusing to trade blind to the close"
    close_buffer = constraints.get("market_close_buffer_minutes", 15)
    if decision["action"] == "BUY" and minutes_left <= close_buffer:
        return f"within {close_buffer} min of market close - only SELL/HOLD permitted"
    return None


def _check_sell(symbol: str, qty: int, held_qty: float) -> str | None:
    """A SELL must close something we own. Anything else would be a short."""
    if held_qty <= 0:
        return f"no long position in {symbol} to sell (held {held_qty:g})"
    if qty > held_qty + 1e-6:
        return f"SELL {qty} exceeds position of {held_qty:g} in {symbol} - shorting is not permitted"
    return None


def _check_buy(
    decision: dict,
    constraints: dict,
    account: dict,
    performance: dict,
    budget: CycleBudget | None,
    held_qty: float,
) -> str | None:
    """
    The limits a BUY must respect, in plain words:
    1. one trade can't be too big (max_trade_size_pct)
    2. we must have the cash for it
    3. we can't hold too much of one stock (max_position_pct)
    4. total money invested can't be too high (gross exposure)
    5. stop buying once today's losses pass the daily limit
    """
    qty = decision["target_qty"]
    price = account["last_prices"][decision["symbol"]]
    trade_value = qty * price
    capital = constraints["session_capital"]

    if trade_value > capital * constraints["max_trade_size_pct"]:
        return (
            f"trade value {trade_value:.2f} exceeds max_trade_size_pct "
            f"({constraints['max_trade_size_pct']:.0%} of session capital)"
        )

    cash = budget.remaining_cash if budget else account.get("cash", 0)
    if trade_value > cash:
        return f"insufficient buying power: {trade_value:.2f} needed, {cash:.2f} uncommitted this cycle"

    if (held_qty + qty) * price > capital * constraints["max_position_pct"]:
        return (
            f"resulting position exceeds max_position_pct "
            f"({constraints['max_position_pct']:.0%} of session capital)"
        )

    max_gross = constraints.get("max_gross_exposure_pct", 1.0)
    gross = (budget.gross_exposure if budget else _held_value(account)) + trade_value
    if gross > capital * max_gross:
        return f"gross exposure {gross:.2f} exceeds max_gross_exposure_pct ({max_gross:.0%} of session capital)"

    return _check_daily_loss(performance, account, capital, constraints["max_daily_loss_pct"])


def _check_daily_loss(performance: dict, account: dict, capital: float, max_loss: float) -> str | None:
    """Stop buying once today's losses reach the limit (this session's, or the whole account's)."""
    session_loss = _session_loss_pct(performance, capital)
    if session_loss <= -max_loss:
        return f"daily loss limit reached ({session_loss:.2%} <= -{max_loss:.0%}), no new BUYs today"
    account_loss = account.get("daily_pnl_pct", 0)
    if account_loss <= -max_loss:
        return (
            f"account-level daily loss limit reached ({account_loss:.2%} <= "
            f"-{max_loss:.0%}), no new BUYs today"
        )
    return None


def evaluate(
    decision: dict,
    objective_constraints: dict,
    account: dict,
    performance: dict | None = None,
    budget: CycleBudget | None = None,
) -> tuple[bool, str | None]:
    """
    Returns (approved, rejection_reason). Plain code, not negotiable by any
    LLM node: this is the boundary the decision agent cannot cross.
    """
    constraints = objective_constraints["constraints"]

    if constraints.get("market_hours_only", True) and not account.get("clock", {}).get("is_open"):
        return False, "market is closed"

    action = decision.get("action")
    if action == "HOLD":
        return True, None
    if action not in ("BUY", "SELL"):
        return False, f"unrecognised action {action!r} - expected BUY, SELL or HOLD"

    reason = _check_basics(decision, constraints, account)
    if reason is None:
        held_qty = account.get("positions", {}).get(decision["symbol"], {}).get("qty", 0)
        if action == "SELL":
            reason = _check_sell(decision["symbol"], decision["target_qty"], held_qty)
        else:
            reason = _check_buy(decision, constraints, account, performance or {}, budget, held_qty)
    return reason is None, reason


def evaluate_batch(
    decisions: list[dict],
    objective_constraints: dict,
    account: dict,
    performance: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Checks a whole cycle's decisions against one shared budget, most confident
    first, so limited cash goes to the trades the model believed in most.
    Returns (approved, rejected). HOLDs are neither.
    """
    budget = CycleBudget(remaining_cash=account.get("cash", 0), gross_exposure=_held_value(account))
    approved, rejected = [], []

    for decision in sorted(decisions, key=lambda d: d.get("confidence") or 0, reverse=True):
        ok, reason = evaluate(decision, objective_constraints, account, performance, budget)
        if not ok:
            rejected.append({"decision": decision, "reason": reason})
        elif decision.get("action") != "HOLD":
            price = account.get("last_prices", {}).get(decision["symbol"], 0)
            budget.commit(decision["target_qty"] * price, decision["action"])
            approved.append(decision)

    return approved, rejected
