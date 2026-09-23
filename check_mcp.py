"""
Live connectivity and contract check against the Alpaca MCP server.

Run this before the first real session:

    python check_mcp.py

The agent reads specific fields out of specific tool responses. This asserts
those fields are actually present and shows what the server really returns, so
a schema mismatch surfaces here rather than as an agent that silently trades on
zeros. (The old `test_mcp.py` imported `mcp.Client`, which does not exist in
the installed SDK, so it could never run.)
"""

import asyncio
import sys
from datetime import datetime, timezone

from src.alpaca.mcp_client import MCPToolError, alpaca_mcp_session, bars_for_symbol, call_tool

# get_stock_bars defaults to sort="asc" with no explicit start time, which
# silently returns the OLDEST bars in the server's default lookback window -
# confirmed live to be several days stale - rather than the most recent ones.
# Always pass sort="desc" (then reverse locally back to chronological order).
_BARS_FRESHNESS_LIMIT_MINUTES = 20

# tool -> (arguments, dotted field paths the agent depends on).
# Paths are checked against the payload AFTER call_tool's envelope unwrap
# (see src/alpaca/mcp_client.py) - not the raw MCP response.
CONTRACT = {
    "get_clock": ({}, ["is_open", "timestamp", "next_close"]),
    "get_account_info": ({}, ["cash", "portfolio_value", "equity", "last_equity"]),
    "get_all_positions": ({}, ["result"]),
    "get_orders": ({"status": "open"}, ["result"]),
    "get_stock_snapshot": (
        {"symbols": "AAPL"},
        ["AAPL.latestTrade.p", "AAPL.dailyBar.o"],
    ),
    "get_stock_bars": (
        {"symbols": "AAPL", "timeframe": "5Min", "limit": 5, "sort": "desc"},
        ["bars.AAPL"],
    ),
    "get_news": ({"symbols": "AAPL", "limit": 2}, ["news"]),
}


def _lookup(payload: dict, path: str):
    node = payload
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None, False
        node = node[part]
    return node, True


async def main() -> int:
    failures = []

    async with alpaca_mcp_session() as session:
        listing = await session.list_tools()
        available = {t.name for t in listing.tools}
        print(f"Connected. Server exposes {len(available)} tools.\n")

        for tool, (arguments, fields) in CONTRACT.items():
            if tool not in available:
                hint = "needs 'assets' in" if tool == "get_clock" else "check"
                print(f"[MISSING] {tool} - not exposed ({hint} ALPACA_TOOLSETS)")
                failures.append(tool)
                continue

            try:
                payload = await call_tool(session, tool, arguments)
            except MCPToolError as exc:
                print(f"[ERROR]   {tool}: {exc}")
                failures.append(tool)
                continue

            missing = [f for f in fields if not _lookup(payload, f)[1]]
            status = "OK      " if not missing else "MISMATCH"
            print(f"[{status}] {tool}")
            print(f"           keys: {sorted(payload)[:12]}")
            if missing:
                print(f"           expected fields not found: {missing}")
                failures.append(tool)

        # Field presence alone wouldn't have caught the sort-order bug above -
        # the response had the right shape, just old data. Compare the latest
        # bar against the market clock directly.
        if "get_stock_bars" in available and "get_clock" in available:
            bars_payload, _ = CONTRACT["get_stock_bars"]
            bars = await call_tool(session, "get_stock_bars", bars_payload)
            latest_bar = bars_for_symbol(bars, "AAPL")
            clock = await call_tool(session, "get_clock", {})
            if latest_bar and clock.get("is_open"):
                bar_time = datetime.fromisoformat(latest_bar[0]["t"].replace("Z", "+00:00"))
                now = datetime.fromisoformat(str(clock["timestamp"]).replace("Z", "+00:00"))
                age_minutes = (now - bar_time).total_seconds() / 60
                if age_minutes > _BARS_FRESHNESS_LIMIT_MINUTES:
                    print(
                        f"[STALE   ] get_stock_bars - latest bar is {age_minutes:.0f} min old "
                        f"while the market is open (>{_BARS_FRESHNESS_LIMIT_MINUTES} min is "
                        "suspicious - check sort='desc' is being passed)"
                    )
                    failures.append("get_stock_bars (stale)")
                else:
                    print(f"[OK      ] get_stock_bars freshness - latest bar {age_minutes:.1f} min old")

        # Order placement is the one thing worth checking by hand, since it
        # moves real (paper) money. Left deliberately un-run.
        if "place_stock_order" in available:
            schema = next(t for t in listing.tools if t.name == "place_stock_order")
            print("\nplace_stock_order input schema (verify before the first run):")
            print(f"  {schema.inputSchema}")

    if failures:
        print(f"\n{len(failures)} tool(s) did not match what the agent expects: {failures}")
        print("Adjust src/execution/broker.py, src/scanner/scanner.py or")
        print("src/analysis/technical.py to the real field names before trading.")
        return 1

    print("\nAll tool contracts match. The agent's field assumptions hold.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
