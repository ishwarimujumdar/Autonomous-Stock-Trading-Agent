import asyncio

from mcp import ClientSession

from src.alpaca.mcp_client import MCPToolError, bars_for_symbol, call_tool, snapshot_for_symbol
from src.config import MAX_CONCURRENCY, TRADABLE_UNIVERSE

# Average daily volume doesn't change during a session, so it's computed once
# per symbol per run rather than on every cycle.
_avg_daily_volume_cache: dict[str, float] = {}

_AVG_VOLUME_DAYS = 20


def clear_cache() -> None:
    _avg_daily_volume_cache.clear()


async def _avg_daily_volume(client: ClientSession, symbol: str) -> float:
    """
    A genuine N-day average. The scanner used to compare a single day's volume
    from the snapshot's `daily_bar` against a criterion named
    `min_avg_volume` - a different quantity than the name promised.
    """
    if symbol in _avg_daily_volume_cache:
        return _avg_daily_volume_cache[symbol]

    # sort=desc: without it, "limit" bars come from the OLDEST end of the
    # server's default lookback window, not the most recent N days - same
    # issue as compute_technical(). The average itself is order-independent,
    # but which 20 days get averaged isn't, and it should be the recent ones.
    bars = await call_tool(
        client,
        "get_stock_bars",
        {"symbols": symbol, "timeframe": "1Day", "limit": _AVG_VOLUME_DAYS, "sort": "desc"},
    )
    # Raw Alpaca bar keys: "v" is volume, not "volume".
    volumes = [b["v"] for b in bars_for_symbol(bars, symbol) if b.get("v") is not None]
    average = sum(volumes) / len(volumes) if volumes else 0.0
    _avg_daily_volume_cache[symbol] = average
    return average


async def _evaluate_symbol(
    client: ClientSession, symbol: str, criteria: dict, semaphore: asyncio.Semaphore
) -> tuple[str, bool, str, float]:
    async with semaphore:
        try:
            snapshot = await call_tool(client, "get_stock_snapshot", {"symbols": symbol})
        except MCPToolError as exc:
            return symbol, False, f"snapshot unavailable: {exc}", 0.0

        # Raw Alpaca field letters: latestTrade.p (price), dailyBar.o (open).
        info = snapshot_for_symbol(snapshot, symbol)
        latest_price = info.get("latestTrade", {}).get("p") or 0
        daily_bar = info.get("dailyBar", {}) or {}
        open_price = daily_bar.get("o") or 0

        if latest_price <= 0:
            return symbol, False, "no current price", 0.0

        if "min_price" in criteria and latest_price < criteria["min_price"]:
            return symbol, False, (
                f"price {latest_price} below min_price {criteria['min_price']}"
            ), latest_price
        if "max_price" in criteria and latest_price > criteria["max_price"]:
            return symbol, False, (
                f"price {latest_price} above max_price {criteria['max_price']}"
            ), latest_price

        if "min_avg_daily_volume" in criteria:
            try:
                average = await _avg_daily_volume(client, symbol)
            except MCPToolError as exc:
                return symbol, False, f"volume history unavailable: {exc}", latest_price
            if average < criteria["min_avg_daily_volume"]:
                return symbol, False, (
                    f"{_AVG_VOLUME_DAYS}d avg volume {average:.0f} below "
                    f"{criteria['min_avg_daily_volume']}"
                ), latest_price

        needs_intraday = (
            "min_intraday_return_pct" in criteria or "max_intraday_return_pct" in criteria
        )
        if needs_intraday:
            if open_price <= 0:
                return symbol, False, "no session open price for intraday return", latest_price
            intraday_return = (latest_price - open_price) / open_price * 100
            if intraday_return < criteria.get("min_intraday_return_pct", float("-inf")):
                return symbol, False, (
                    f"intraday return {intraday_return:.2f}% below "
                    f"{criteria['min_intraday_return_pct']}%"
                ), latest_price
            if intraday_return > criteria.get("max_intraday_return_pct", float("inf")):
                return symbol, False, (
                    f"intraday return {intraday_return:.2f}% above "
                    f"{criteria['max_intraday_return_pct']}%"
                ), latest_price

        return symbol, True, "passed", latest_price


async def scan(
    client: ClientSession,
    universe_criteria: dict,
    universe_symbols: list[str] | None = None,
) -> tuple[list[str], list[dict], dict[str, float]]:
    """
    Deterministic filter: applies the Strategy Agent's numeric criteria
    (min_price, min_avg_daily_volume, min_intraday_return_pct, ...) against the
    symbols the strategy selected. No LLM calls here - this is plain filtering,
    done in code because it's cheaper and faster than spending LLM calls per
    candidate.

    Returns (candidates, exclusions, prices):
    - `exclusions` so a cycle that finds nothing says why, rather than
      returning an empty list with no explanation.
    - `prices` because the risk gate runs before the monitor refreshes the
      account. Without these it would size this cycle's trades on the previous
      cycle's quotes - and on the first cycle it had no prices at all, so every
      trade was rejected for want of one.
    """
    base_universe = [s for s in (universe_symbols or TRADABLE_UNIVERSE) if s in TRADABLE_UNIVERSE]
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    results = await asyncio.gather(
        *(_evaluate_symbol(client, s, universe_criteria, semaphore) for s in base_universe)
    )

    candidates = [symbol for symbol, passed, _, _ in results if passed]
    exclusions = [
        {"symbol": symbol, "reason": reason}
        for symbol, passed, reason, _ in results
        if not passed
    ]
    prices = {symbol: price for symbol, _, _, price in results if price > 0}
    return candidates, exclusions, prices
