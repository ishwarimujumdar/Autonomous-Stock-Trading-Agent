from mcp import ClientSession

from src.alpaca.mcp_client import bars_for_symbol, call_tool
from src.config import INTRADAY_BAR_LIMIT, INTRADAY_TIMEFRAME


def pct_change(before: float, after: float) -> float:
    """How much a price moved, in percent. 100 -> 101 is +1.0."""
    return (after - before) / before * 100


def _change_over(closes: list[float], bars_back: int) -> float | None:
    """% change from `bars_back` bars ago to now, or None if there aren't enough bars."""
    if len(closes) <= bars_back:
        return None
    return pct_change(closes[-1 - bars_back], closes[-1])


def compute_indicators(closes: list[float], volumes: list[float]) -> dict:
    """
    Two simple measurements from recent 5-minute bars (fetching is separate,
    so this can be tested without a server):

    - price change: how far the price moved, as a % over the last 3 bars
      (~15 min) and 12 bars (~60 min).
    - volume ratio: how busy the latest bar is compared with the 6 bars before
      it (~30 min). 1.0 = normal, 2.0 = twice as many shares changing hands.
    """
    out: dict = {
        "bars_available": len(closes),
        "return_last_3_bars_pct": _change_over(closes, 3),
        "return_last_12_bars_pct": _change_over(closes, 12),
    }
    if len(volumes) >= 7:
        # Compare against the 6 bars BEFORE the latest, so a spike can't inflate its own baseline.
        earlier = volumes[-7:-1]
        average = sum(earlier) / len(earlier)
        out["volume_ratio_6bar"] = round(volumes[-1] / average, 4) if average else None
    return out


def has_usable_evidence(technical: dict) -> bool:
    """True if at least one measurement computed (else skip the LLM call)."""
    measurements = [k for k in technical if k not in ("symbol", "bars_available", "error")]
    return any(technical.get(k) is not None for k in measurements)


async def compute_technical(client: ClientSession, symbol: str) -> dict:
    """Fetches recent 5-minute bars and measures them."""
    # sort=desc gets the NEWEST bars (the default returns the oldest); reversed to oldest-first below.
    bars = await call_tool(
        client,
        "get_stock_bars",
        {
            "symbols": symbol,
            "timeframe": INTRADAY_TIMEFRAME,
            "limit": INTRADAY_BAR_LIMIT,
            "sort": "desc",
        },
    )
    rows = list(reversed(bars_for_symbol(bars, symbol)))
    closes = [b["c"] for b in rows if b.get("c") is not None]
    volumes = [b["v"] for b in rows if b.get("v") is not None]

    return {
        "symbol": symbol,
        "timeframe": INTRADAY_TIMEFRAME,
        **compute_indicators(closes, volumes),
    }
