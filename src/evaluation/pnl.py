_RECENT_TRADES_KEPT = 10


def unrealized_pnl(account: dict) -> float:
    """
    Paper profit or loss on shares still held: what they're worth now minus
    what we paid. It only becomes real ("realized") when we sell.
    """
    prices = account.get("last_prices", {})
    total = 0.0
    for symbol, pos in account.get("positions", {}).items():
        paid = pos.get("avg_price", 0)
        now = prices.get(symbol, paid)
        total += pos.get("qty", 0) * (now - paid)
    return total


def compute_performance(
    account: dict,
    session_capital: float,
    starting_equity: float,
    new_trades: list[dict],
    prior: dict | None = None,
) -> dict:
    """
    The running scoreboard. Each cycle adds the trades that just closed to
    the totals so far (they aren't rebuilt from `recent_trades`, which is cut off).

    - realized_pnl: profit/loss locked in by selling
    - unrealized_pnl: profit/loss on what we still hold
    - return_pct: account value now vs. session start, as % of the capital allocated
    - wins / losses / scratches: sold for more / less / the same as we paid
    """
    prior = prior or {}
    new_trades = new_trades or []

    profits = [t.get("pnl", 0.0) for t in new_trades]

    equity = account.get("equity", starting_equity)
    return_pct = (equity - starting_equity) / session_capital * 100 if session_capital > 0 else 0.0

    return {
        "realized_pnl": round(prior.get("realized_pnl", 0.0) + sum(profits), 4),
        "unrealized_pnl": round(unrealized_pnl(account), 4),
        "return_pct": round(return_pct, 4),
        "wins": prior.get("wins", 0) + sum(p > 0 for p in profits),
        "losses": prior.get("losses", 0) + sum(p < 0 for p in profits),
        "scratches": prior.get("scratches", 0) + sum(p == 0 for p in profits),
        "trade_count": prior.get("trade_count", 0) + len(new_trades),
        "recent_trades": (prior.get("recent_trades", []) + new_trades)[-_RECENT_TRADES_KEPT:],
    }
