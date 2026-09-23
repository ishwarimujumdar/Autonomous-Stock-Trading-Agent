import json
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.config import ALPACA_API_KEY, ALPACA_PAPER_TRADE, ALPACA_SECRET_KEY, ALPACA_TOOLSETS


class MCPToolError(RuntimeError):
    """An Alpaca MCP tool call failed or returned something unusable."""


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="uvx",
        args=["alpaca-mcp-server"],
        env={
            "ALPACA_API_KEY": ALPACA_API_KEY,
            "ALPACA_SECRET_KEY": ALPACA_SECRET_KEY,
            "ALPACA_PAPER_TRADE": ALPACA_PAPER_TRADE,
            "ALPACA_TOOLSETS": ALPACA_TOOLSETS,
        },
    )


@asynccontextmanager
async def alpaca_mcp_session():
    """
    One Alpaca MCP session for the lifetime of a run. Nodes reuse this
    connection rather than each spawning their own MCP server subprocess.
    """
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _payload_from_content(content) -> dict | None:
    """
    Fall back to the text content block when the server doesn't populate
    `structuredContent`. Some MCP servers only return text, and returning an
    empty dict for those turned every tool into a silent no-op: prices read as
    0, the clock read as closed, and the agent would end its session
    immediately with no error anywhere.
    """
    for block in content or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
    return None


def _unwrap(payload: dict) -> dict:
    """
    This Alpaca MCP server wraps every tool result in a security envelope:
    {"_alpaca_mcp_security": {...}, "data": <actual payload>}. Confirmed live
    against get_account_info, get_clock, get_all_positions, get_orders,
    get_stock_snapshot, get_stock_bars and get_news. Strip it here, once, so
    every caller works with the real payload instead of the wrapper.
    """
    if isinstance(payload, dict) and "_alpaca_mcp_security" in payload and "data" in payload:
        return payload["data"]
    return payload


def bars_for_symbol(payload: dict, symbol: str) -> list[dict]:
    """
    get_stock_bars' unwrapped payload is {"bars": {symbol: [bar, ...]}} -
    multi-symbol shaped even for a single-symbol request. Each bar uses
    Alpaca's raw field letters: o/h/l/c/v/vw/n/t (open/high/low/close/volume/
    vwap/trade-count/timestamp).
    """
    return payload.get("bars", {}).get(symbol, [])


def snapshot_for_symbol(payload: dict, symbol: str) -> dict:
    """
    get_stock_snapshot's unwrapped payload is {symbol: {"dailyBar": {...},
    "latestTrade": {...}, "latestQuote": {...}}} - keyed by symbol even for a
    single-symbol request. latestTrade's price field is "p", not "price".
    """
    return payload.get(symbol, {})


async def call_tool(session: ClientSession, name: str, arguments: dict) -> dict:
    result = await session.call_tool(name, arguments)
    if result.isError:
        raise MCPToolError(f"MCP tool '{name}' failed: {result.content}")

    if result.structuredContent:
        return _unwrap(result.structuredContent)

    payload = _payload_from_content(result.content)
    if payload is not None:
        return _unwrap(payload)

    raise MCPToolError(
        f"MCP tool '{name}' returned no structured content and no JSON text block. "
        f"Raw content: {result.content!r}"
    )
