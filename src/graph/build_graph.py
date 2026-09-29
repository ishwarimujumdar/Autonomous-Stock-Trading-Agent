from functools import partial

from langgraph.graph import END, StateGraph
from mcp import ClientSession

from src.graph import nodes
from src.graph.state import TradingState


def _route_after_pick(state: TradingState) -> str:
    # No watchlist at all: skip trading, but still run Close so nothing is left open.
    return "close" if state.get("pick_failed") else "market_data"


def _route_after_log_wait(state: TradingState) -> str:
    if state.get("session_done"):
        return "close"
    return "pick_stocks" if state.get("repick_due") else "market_data"


def build_graph(client: ClientSession):
    graph = StateGraph(TradingState)

    # Nodes that talk to Alpaca take the MCP client as their first argument.
    graph.add_node("pick_stocks", partial(nodes.n_pick_stocks, client))
    graph.add_node("market_data", partial(nodes.n_market_data, client))
    graph.add_node("trading_agent", nodes.n_trading_agent)
    graph.add_node("risk_gate", nodes.n_risk_gate)
    graph.add_node("execute", partial(nodes.n_execute, client))
    graph.add_node("log_wait", nodes.n_log_wait)
    graph.add_node("close", partial(nodes.n_close, client))

    graph.set_entry_point("pick_stocks")
    graph.add_conditional_edges("pick_stocks", _route_after_pick, ["market_data", "close"])
    graph.add_edge("market_data", "trading_agent")
    graph.add_edge("trading_agent", "risk_gate")
    graph.add_edge("risk_gate", "execute")
    graph.add_edge("execute", "log_wait")
    graph.add_conditional_edges(
        "log_wait", _route_after_log_wait, ["close", "pick_stocks", "market_data"]
    )
    graph.add_edge("close", END)

    return graph.compile()
