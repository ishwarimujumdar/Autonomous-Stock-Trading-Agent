from mcp import ClientSession

from src.alpaca.mcp_client import MCPToolError, call_tool, snapshot_for_symbol
from src.analysis.technical import pct_change
from src.concurrency import gather_limited
from src.config import TRADABLE_UNIVERSE


async def _row(client: ClientSession, symbol: str) -> dict | None:
    """One table line: price, today's move and range, yesterday's volume."""
    try:
        snapshot = await call_tool(client, "get_stock_snapshot", {"symbols": symbol})
    except MCPToolError:
        return None
    info = snapshot_for_symbol(snapshot, symbol)
    price = info.get("latestTrade", {}).get("p") or 0
    yesterday = info.get("prevDailyBar") or {}
    today = info.get("dailyBar") or {}
    if price <= 0 or not yesterday.get("c"):
        return None
    return {
        "symbol": symbol,
        "price": round(price, 2),
        "change_pct": round(pct_change(yesterday["c"], price), 2),
        # How far the price has travelled today, low to high: a busy stock has a wide range.
        "range_pct": round(pct_change(today["l"], today["h"]), 2) if today.get("l") else 0.0,
        "yesterday_volume": yesterday.get("v", 0),
    }


async def universe_table(client: ClientSession) -> list[dict]:
    """What the LLM reads to choose its watchlist"""
    # These are Alpaca calls, not Groq calls, so more can run at once than the default.
    rows = await gather_limited((_row(client, s) for s in TRADABLE_UNIVERSE), limit=6)
    return [row for row in rows if row]
