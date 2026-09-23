import time

from mcp import ClientSession

from src.alpaca.mcp_client import call_tool
from src.config import NEWS_CACHE_TTL_SECONDS
from src.llm import get_llm

_SYSTEM_PROMPT = """You are a market/news analyst for a stock trading system.
Given recent news headlines for a symbol, give sentiment (positive/neutral/
negative) and any catalyst that matters to a {strategy_name} strategy watching
these signals: {signals}. ONE short sentence, under 30 words. Do not make a
trade recommendation; that is a different node's job."""

# symbol -> (expires_at_monotonic, narrative). Headlines don't turn over every
# five minutes, so re-fetching and re-summarising each cycle spent an LLM call
# per symbol per cycle to regenerate the same paragraph.
_cache: dict[str, tuple[float, str]] = {}


def clear_cache() -> None:
    _cache.clear()


async def analyze_news(
    client: ClientSession, symbol: str, strategy_name: str, signals: list[str]
) -> dict:
    if "news_sentiment" not in signals:
        return {"symbol": symbol, "narrative": None}

    cached = _cache.get(symbol)
    if cached and cached[0] > time.monotonic():
        return {"symbol": symbol, "narrative": cached[1], "cached": True}

    news = await call_tool(client, "get_news", {"symbols": symbol, "limit": 5})
    headlines = [item.get("headline", "") for item in news.get("news", []) if item.get("headline")]

    if not headlines:
        narrative = "No recent news found."
    else:
        prompt = _SYSTEM_PROMPT.format(strategy_name=strategy_name, signals=signals)
        try:
            response = await get_llm(temperature=0.2, max_tokens=100).ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "\n".join(headlines)},
                ]
            )
        except Exception as exc:
            # A raw provider failure here must not end the session over a
            # narrative that's optional context, not a decision. Not cached -
            # this retries next cycle instead of being stuck "unavailable"
            # for the full TTL.
            return {"symbol": symbol, "narrative": f"narrative unavailable: {exc}", "cached": False}
        narrative = response.content

    _cache[symbol] = (time.monotonic() + NEWS_CACHE_TTL_SECONDS, narrative)
    return {"symbol": symbol, "narrative": narrative, "cached": False}
