from mcp import ClientSession

from src.alpaca.mcp_client import bars_for_symbol, call_tool
from src.config import INTRADAY_BAR_LIMIT, INTRADAY_TIMEFRAME


def _sma(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def _ema_series(values: list[float], window: int) -> list[float] | None:
    """Standard EMA, seeded with the SMA of the first `window` values."""
    if len(values) < window:
        return None
    multiplier = 2 / (window + 1)
    ema = sum(values[:window]) / window
    series = [ema]
    for value in values[window:]:
        ema = (value - ema) * multiplier + ema
        series.append(ema)
    return series


def _rsi(closes: list[float], window: int = 14) -> float | None:
    """
    Wilder's RSI. The previous simple-average version diverged from what every
    charting tool (and the LLM reading the field name "rsi_14") assumes.
    """
    if len(closes) < window + 1:
        return None

    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(c, 0.0) for c in changes]
    losses = [max(-c, 0.0) for c in changes]

    avg_gain = sum(gains[:window]) / window
    avg_loss = sum(losses[:window]) / window
    for i in range(window, len(changes)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(closes: list[float]) -> dict:
    """True EMA-based MACD (12/26/9), replacing the SMA-difference proxy."""
    fast = _ema_series(closes, 12)
    slow = _ema_series(closes, 26)
    if not fast or not slow:
        return {"macd": None, "macd_signal": None, "macd_histogram": None}

    # Align: the slow EMA starts 14 bars later than the fast one.
    offset = len(fast) - len(slow)
    # strict=True pins the alignment: fast[offset:] and slow must be equal length.
    macd_line = [f - s for f, s in zip(fast[offset:], slow, strict=True)]

    signal_series = _ema_series(macd_line, 9)
    if not signal_series:
        return {"macd": round(macd_line[-1], 6), "macd_signal": None, "macd_histogram": None}

    return {
        "macd": round(macd_line[-1], 6),
        "macd_signal": round(signal_series[-1], 6),
        "macd_histogram": round(macd_line[-1] - signal_series[-1], 6),
    }


def compute_indicators(closes: list[float], volumes: list[float], signals: list[str]) -> dict:
    """
    Pure indicator math over already-fetched bars. Split out from the fetch so
    it can be unit tested without an MCP server.
    """
    out: dict = {"bars_available": len(closes)}

    if "price_momentum" in signals and len(closes) >= 2:
        # Two windows, not one: strategies routinely phrase entry logic as
        # "the last three bars" or "the last 30 minutes" - a single fixed
        # 12-bar (60-min) figure was often a much coarser measure than what
        # the strategy actually asked for, understating a real short-term
        # move (confirmed live: a stock at 2x its 30-min volume baseline
        # measured under 1.1x on the 100-min baseline this code used to be
        # the only option).
        out["return_last_3_bars_pct"] = (
            (closes[-1] - closes[-4]) / closes[-4] * 100 if len(closes) >= 4 else None
        )
        out["return_last_12_bars_pct"] = (
            (closes[-1] - closes[-13]) / closes[-13] * 100 if len(closes) >= 13 else None
        )
        out["return_session_pct"] = (closes[-1] - closes[0]) / closes[0] * 100

    if "volume" in signals and volumes:
        # Same two-window reasoning as price_momentum, and both ratios
        # compare the latest bar against the PRECEDING bars only (not
        # including itself) - including it in its own baseline damps a real
        # spike, since the spike inflates the very average it's compared to.
        out["latest_bar_volume"] = volumes[-1]
        prior = volumes[:-1]
        if len(prior) >= 6:
            avg_6 = sum(prior[-6:]) / 6
            out["avg_bar_volume_6bar"] = avg_6
            out["volume_ratio_6bar"] = round(volumes[-1] / avg_6, 4) if avg_6 else None
        if len(prior) >= 20:
            avg_20 = sum(prior[-20:]) / 20
            out["avg_bar_volume_20bar"] = avg_20
            out["volume_ratio_20bar"] = round(volumes[-1] / avg_20, 4) if avg_20 else None

    if "volatility" in signals and len(closes) >= 2:
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        out["bar_return_stdev"] = round(variance**0.5, 6)

    if "moving_average_crossover" in signals:
        fast, slow = _sma(closes, 10), _sma(closes, 30)
        out["sma_10"] = fast
        out["sma_30"] = slow
        out["sma_spread_pct"] = (
            round((fast - slow) / slow * 100, 4) if fast and slow else None
        )

    if "rsi" in signals:
        rsi = _rsi(closes)
        out["rsi_14"] = round(rsi, 4) if rsi is not None else None

    if "macd" in signals:
        out.update(_macd(closes))

    return out


def has_usable_evidence(technical: dict, signals: list[str]) -> bool:
    """
    True when at least one requested indicator actually computed.

    Early in a session, or for a thinly-traded name, every indicator can come
    back None. Sending that to the decision LLM buys nothing but a token bill,
    so those symbols are held without a call.
    """
    indicator_keys = [
        k for k in technical if k not in ("symbol", "bars_available", "error")
    ]
    return any(technical.get(k) is not None for k in indicator_keys) and bool(signals)


async def compute_technical(client: ClientSession, symbol: str, signals: list[str]) -> dict:
    """
    Deterministic indicator calculation on INTRADAY bars, scoped to only the
    signals the Strategy Agent selected - we don't compute (or pay for)
    signals it didn't ask for.

    These were previously computed on 1Day bars, which for a day-trading agent
    re-evaluating every 5-30 minutes meant the "evidence" was byte-identical
    from one cycle to the next for the entire session.
    """
    # sort defaults to "asc" with no explicit start time, which returns the
    # OLDEST bars in the server's default lookback window, not the most
    # recent ones - confirmed live: without sort=desc this silently returned
    # bars several days stale while every other tool (snapshot, quote, clock)
    # was current. desc + reverse gets the latest bars back in chronological
    # order, which the rest of this function assumes.
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
    # Raw Alpaca bar keys: "c" is close, "v" is volume.
    rows = list(reversed(bars_for_symbol(bars, symbol)))
    closes = [b["c"] for b in rows if b.get("c") is not None]
    volumes = [b["v"] for b in rows if b.get("v") is not None]

    return {
        "symbol": symbol,
        "timeframe": INTRADAY_TIMEFRAME,
        **compute_indicators(closes, volumes, signals),
    }
