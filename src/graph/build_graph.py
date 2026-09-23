from functools import partial

from langgraph.graph import END, StateGraph
from mcp import ClientSession

from src.config import MAX_STRATEGY_ATTEMPTS
from src.graph import nodes
from src.graph.state import TradingState


def _route_after_schema_validator(state: TradingState) -> str:
    if state.get("strategy_valid"):
        return "scanner"
    if state.get("strategy_attempts", 0) >= MAX_STRATEGY_ATTEMPTS:
        # Bounded rather than looping until the recursion limit. It routes to
        # the flatten path, not to END, because a failed *reassessment*
        # mid-session can leave open positions that still must be closed.
        return "strategy_failed"
    return "strategy_agent"


def _route_after_cycle_controller(state: TradingState) -> str:
    if state.get("session_done"):
        return "finalize"
    if state.get("restrategize"):
        return "strategy_agent"
    return "scanner"


def build_graph(client: ClientSession):
    graph = StateGraph(TradingState)

    # Nodes that talk to Alpaca take the MCP client as their first argument;
    # the rest take `state` alone.
    graph.add_node("strategy_agent", nodes.n_strategy_agent)
    graph.add_node("schema_validator", nodes.n_schema_validator)
    graph.add_node("strategy_failed", nodes.n_strategy_failed)
    graph.add_node("scanner", partial(nodes.n_scanner, client))
    graph.add_node("research", partial(nodes.n_research, client))
    graph.add_node("trade_decision", nodes.n_trade_decision)
    graph.add_node("risk_gate", nodes.n_risk_gate)
    graph.add_node("execution", partial(nodes.n_execution, client))
    graph.add_node("monitor", partial(nodes.n_monitor, client))
    graph.add_node("performance_tracker", nodes.n_performance_tracker)
    graph.add_node("cycle_controller", nodes.n_cycle_controller)
    graph.add_node("finalize", partial(nodes.n_finalize, client))

    graph.set_entry_point("strategy_agent")

    graph.add_edge("strategy_agent", "schema_validator")
    graph.add_conditional_edges(
        "schema_validator",
        _route_after_schema_validator,
        ["scanner", "strategy_agent", "strategy_failed"],
    )
    graph.add_edge("strategy_failed", "finalize")
    graph.add_edge("scanner", "research")
    graph.add_edge("research", "trade_decision")
    graph.add_edge("trade_decision", "risk_gate")
    graph.add_edge("risk_gate", "execution")
    graph.add_edge("execution", "monitor")
    graph.add_edge("monitor", "performance_tracker")
    graph.add_edge("performance_tracker", "cycle_controller")
    graph.add_conditional_edges(
        "cycle_controller",
        _route_after_cycle_controller,
        ["finalize", "strategy_agent", "scanner"],
    )
    graph.add_edge("finalize", END)

    return graph.compile()
