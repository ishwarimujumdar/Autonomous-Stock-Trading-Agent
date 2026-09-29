from src.evaluation.pnl import compute_performance

ACCOUNT = {"equity": 10500, "portfolio_value": 10500, "cash": 10500, "positions": {}}


def test_realized_pnl_accumulates_past_the_recent_trades_window():
    """
    Totals used to be re-derived from `recent_trades`, which is truncated to
    10 - so realized P&L and the win count silently shrank as trading went on.
    """
    performance = None
    for _ in range(12):
        performance = compute_performance(
            account=ACCOUNT,
            session_capital=10000,
            starting_equity=10000,
            new_trades=[{"symbol": "X", "pnl": 100.0}],
            prior=performance,
        )

    assert performance["realized_pnl"] == 1200.0
    assert performance["wins"] == 12
    assert performance["trade_count"] == 12
    assert len(performance["recent_trades"]) == 10, "display list stays bounded"


def test_flat_trade_is_neither_win_nor_loss():
    performance = compute_performance(
        account=ACCOUNT,
        session_capital=10000,
        starting_equity=10000,
        new_trades=[{"symbol": "X", "pnl": 0.0}],
    )
    assert performance["wins"] == 0
    assert performance["losses"] == 0
    assert performance["scratches"] == 1


def test_return_is_measured_on_allocated_capital_not_whole_account():
    """
    A $10k allocation on a $100k account used to report a ~900% return,
    because return was (portfolio_value - allocation) / allocation.
    """
    account = {"equity": 100_500, "portfolio_value": 100_500, "cash": 100_500, "positions": {}}
    performance = compute_performance(
        account=account,
        session_capital=10_000,
        starting_equity=100_000,
        new_trades=[],
    )
    assert performance["return_pct"] == 5.0


def test_unrealized_pnl_is_market_value_less_cost_basis():
    account = {
        "equity": 10000,
        "cash": 5000,
        "positions": {"AAPL": {"qty": 50, "avg_price": 100}},
        "last_prices": {"AAPL": 110},
    }
    performance = compute_performance(
        account=account, session_capital=10000, starting_equity=10000, new_trades=[]
    )
    assert performance["unrealized_pnl"] == 500.0
