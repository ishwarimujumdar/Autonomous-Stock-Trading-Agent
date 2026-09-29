from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.config import ALPACA_API_KEY, ALPACA_PAPER_TRADE, ALPACA_SECRET_KEY, ALPACA_TOOLSETS


class MCPToolError(RuntimeError):
    """An Alpaca MCP tool call failed or returned something unusable."""


@asynccontextmanager
async def alpaca_mcp_session():
    """One Alpaca MCP connection for the whole run, shared by every node."""
    server = StdioServerParameters(
        command="uvx",
        args=["alpaca-mcp-server"],
        env={
            "ALPACA_API_KEY": ALPACA_API_KEY,
            "ALPACA_SECRET_KEY": ALPACA_SECRET_KEY,
            "ALPACA_PAPER_TRADE": ALPACA_PAPER_TRADE,
            "ALPACA_TOOLSETS": ALPACA_TOOLSETS,
        },
    )
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


def bars_for_symbol(payload: dict, symbol: str) -> list[dict]:
    """Bars for one symbol. Fields: o/h/l/c/v (open/high/low/close/volume) and t (time)."""
    return payload.get("bars", {}).get(symbol, [])


def snapshot_for_symbol(payload: dict, symbol: str) -> dict:
    """One symbol's snapshot: dailyBar, latestTrade (price is "p") and latestQuote."""
    return payload.get(symbol, {})


async def call_tool(session: ClientSession, name: str, arguments: dict) -> dict:
    """Calls an Alpaca tool and returns its data, or raises MCPToolError."""
    result = await session.call_tool(name, arguments)
    if result.isError:
        raise MCPToolError(f"MCP tool '{name}' failed: {result.content}")
    if not result.structuredContent:
        raise MCPToolError(f"MCP tool '{name}' returned no structured data: {result.content!r}")

    payload = result.structuredContent
    # This server wraps every result as {"_alpaca_mcp_security": ..., "data": <the real result>}.
    return payload["data"] if "_alpaca_mcp_security" in payload else payload
