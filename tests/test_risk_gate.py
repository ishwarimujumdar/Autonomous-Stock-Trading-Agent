import pytest

from src.config import load_objective_and_constraints
from src.risk.risk_gate import CycleBudget, evaluate, evaluate_batch

OC = load_objective_and_constraints(session_capital=10000)


def _account(**overrides):
    base = {
        "cash": 10000,
        "portfolio_value": 10000,
        "equity": 10000,
        "positions": {},
        "last_prices": {"AAPL": 100},
        "daily_pnl_pct": 0,
        "clock": {"is_open": True},
        "minutes_to_close": 120,
    }
    base.update(overrides)
    return base


def _buy(symbol="AAPL", confidence=0.9, qty=1):
    return {"symbol": symbol, "action": "BUY", "confidence": confidence, "target_qty": qty}


def _sell(symbol="AAPL", confidence=0.9, qty=1):
    return {"symbol": symbol, "action": "SELL", "confidence": confidence, "target_qty": qty}


def test_rejects_low_confidence():
    ok, reason = evaluate(_buy(confidence=0.5), OC, _account())
    assert not ok
    assert "confidence" in reason


def test_approves_high_confidence_within_limits():
    ok, reason = evaluate(_buy(), OC, _account())
    assert ok
    assert reason is None


def test_rejects_buy_within_close_buffer():
    ok, reason = evaluate(_buy(), OC, _account(minutes_to_close=10))
    assert not ok
    assert "market close" in reason


def test_allows_sell_within_close_buffer():
    account = _account(minutes_to_close=10, positions={"AAPL": {"qty": 5, "avg_price": 90}})
    ok, _ = evaluate(_sell(), OC, account)
    assert ok


def test_rejects_when_market_closed():
    ok, reason = evaluate(_buy(), OC, _account(clock={"is_open": False}))
    assert not ok
    assert "market is closed" in reason


def test_rejects_oversized_trade():
    ok, reason = evaluate(_buy(qty=30), OC, _account())
    assert not ok
    assert "max_trade_size_pct" in reason


# --- regressions for bugs found in review ---


def test_missing_minutes_to_close_is_rejected_not_a_crash():
    """Used to raise TypeError comparing None to int on the very first cycle."""
    ok, reason = evaluate(_buy(), OC, _account(minutes_to_close=None))
    assert not ok
    assert "minutes_to_close" in reason


@pytest.mark.parametrize("qty", [None, 0, -5, 1.5, "3", True])
def test_rejects_non_positive_or_non_integer_qty(qty):
    """A None qty used to price the trade at 0 and reach Alpaca as "None"."""
    ok, reason = evaluate(_buy(qty=qty), OC, _account())
    assert not ok
    assert "target_qty" in reason


@pytest.mark.parametrize("action", ["buy", "Buy", "sell", "", None, "LIQUIDATE"])
def test_rejects_unrecognised_action(action):
    """A lowercase 'buy' used to pass the gate and be submitted as a SELL."""
    decision = {"symbol": "AAPL", "action": action, "confidence": 0.9, "target_qty": 1}
    ok, reason = evaluate(decision, OC, _account())
    assert not ok
    assert "unrecognised action" in reason


def test_rejects_sell_with_no_position():
    ok, reason = evaluate(_sell(), OC, _account())
    assert not ok
    assert "no long position" in reason


def test_rejects_sell_larger_than_holding():
    account = _account(positions={"AAPL": {"qty": 3, "avg_price": 90}})
    ok, reason = evaluate(_sell(qty=10), OC, account)
    assert not ok
    assert "shorting is not permitted" in reason


def test_rejects_when_price_unknown():
    ok, reason = evaluate(_buy(symbol="MSFT"), OC, _account())
    assert not ok
    assert "no current price" in reason


def test_batch_enforces_a_shared_budget():
    """
    Five independent 20%-of-capital BUYs each passed the 20% cap against the
    same snapshot, deploying 100% in one cycle.
    """
    account = _account(last_prices={s: 100 for s in "ABCDEF"})
    decisions = [_buy(symbol=s, qty=20) for s in "ABCDEF"]
    approved, rejected = evaluate_batch(decisions, OC, account)

    # 6 x $2000 = $12k against $10k of cash / 100% gross cap. Each trade is
    # individually legal at exactly the 20% per-trade cap; only the shared
    # budget can stop the sixth.
    assert len(approved) == 5
    assert len(rejected) == 1
    assert "buying power" in rejected[0]["reason"] or "gross exposure" in rejected[0]["reason"]


def test_batch_respects_a_tighter_gross_exposure_cap():
    oc = load_objective_and_constraints(session_capital=10000, max_gross_exposure_pct=0.50)
    account = _account(last_prices={s: 100 for s in "ABCDEF"})
    decisions = [_buy(symbol=s, qty=20) for s in "ABCDEF"]
    approved, rejected = evaluate_batch(decisions, oc, account)

    assert len(approved) == 2, "$5000 of gross exposure funds two $2000 trades"
    assert all("gross exposure" in r["reason"] for r in rejected)


def test_batch_funds_highest_confidence_first():
    account = _account(cash=2500, last_prices={"A": 100, "B": 100})
    decisions = [
        {"symbol": "A", "action": "BUY", "confidence": 0.75, "target_qty": 20},
        {"symbol": "B", "action": "BUY", "confidence": 0.95, "target_qty": 20},
    ]
    approved, _ = evaluate_batch(decisions, OC, account)
    assert [d["symbol"] for d in approved] == ["B"]


def test_budget_tracks_cash_drawdown():
    budget = CycleBudget(remaining_cash=2500, gross_exposure=0)
    account = _account(last_prices={"A": 100})
    ok, _ = evaluate(_buy(symbol="A", qty=20), OC, account, budget=budget)
    assert ok
    budget.commit(2000, "BUY")
    ok, reason = evaluate(_buy(symbol="A", qty=20), OC, account, budget=budget)
    assert not ok
    assert "buying power" in reason or "max_position_pct" in reason


def test_session_loss_limit_blocks_new_buys():
    """
    Previously read a `daily_pnl_pct` field Alpaca never returns, so it was
    always 0.0 and the limit could not fire.
    """
    performance = {"realized_pnl": -600.0, "unrealized_pnl": 0.0}
    ok, reason = evaluate(_buy(), OC, _account(), performance)
    assert not ok
    assert "daily loss limit" in reason


def test_account_level_loss_limit_blocks_new_buys():
    ok, reason = evaluate(_buy(), OC, _account(daily_pnl_pct=-0.08))
    assert not ok
    assert "account-level daily loss limit" in reason


def test_loss_limit_does_not_block_exits():
    account = _account(positions={"AAPL": {"qty": 5, "avg_price": 120}})
    performance = {"realized_pnl": -900.0, "unrealized_pnl": 0.0}
    ok, _ = evaluate(_sell(), OC, account, performance)
    assert ok, "closing a position must stay possible after the loss limit trips"


def test_gross_exposure_cap():
    oc = load_objective_and_constraints(session_capital=10000, max_gross_exposure_pct=0.30)
    account = _account(
        cash=10000,
        positions={"MSFT": {"qty": 25, "avg_price": 100}},
        last_prices={"AAPL": 100, "MSFT": 100},
    )
    ok, reason = evaluate(_buy(qty=10), oc, account)
    assert not ok
    assert "gross exposure" in reason
