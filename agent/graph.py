from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent import nodes


def build_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("load_memory", nodes.load_memory_node)
    graph.add_node("classify_intent", nodes.classify_intent_node)
    graph.add_node("faq", nodes.faq_node)
    graph.add_node("order_status", nodes.order_status_node)
    graph.add_node("collect_refund_info", nodes.collect_refund_info_node)
    graph.add_node("dispatch_refund", nodes.dispatch_refund_node)
    graph.add_node("fetch_order_details", nodes.fetch_order_details_node)
    graph.add_node("fetch_refund_policy", nodes.fetch_refund_policy_node)
    graph.add_node("fetch_customer_history", nodes.fetch_customer_history_node)
    graph.add_node("refund_decision", nodes.refund_decision_node)
    graph.add_node("human_review", nodes.human_review_node)
    graph.add_node("create_ticket", nodes.create_ticket_node)
    graph.add_node("unclear_intent", nodes.unclear_intent_node)
    graph.add_node("save_memory", nodes.save_memory_node)


    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "classify_intent")

    # --- conditional workflow: route based on intent ---
    graph.add_conditional_edges(
        "classify_intent",
        nodes.route_after_classification,
        {
            "faq": "faq",
            "order_status": "order_status",
            "refund": "collect_refund_info",
            "ticket": "create_ticket",
            "unclear": "unclear_intent",
        },
    )

    # --- linear workflows end straight at save_memory ---
    graph.add_edge("faq", "save_memory")
    graph.add_edge("order_status", "save_memory")
    graph.add_edge("create_ticket", "save_memory")
    graph.add_edge("unclear_intent", "save_memory")

    # --- iterative workflow: keep collecting info until it's complete ---
    graph.add_conditional_edges(
        "collect_refund_info",
        nodes.route_after_collect_info,
        {
            "ask_more": "save_memory",     # turn ends here, waiting on the customer's reply
            "ready": "dispatch_refund",
        },
    )

    # --- parallel workflow: fan out to three independent lookups ---
    graph.add_edge("dispatch_refund", "fetch_order_details")
    graph.add_edge("dispatch_refund", "fetch_refund_policy")
    graph.add_edge("dispatch_refund", "fetch_customer_history")

    # --- fan back in once all three have finished ---
    graph.add_edge("fetch_order_details", "refund_decision")
    graph.add_edge("fetch_refund_policy", "refund_decision")
    graph.add_edge("fetch_customer_history", "refund_decision")

    # --- conditional workflow: auto-resolve or escalate to a human ---
    graph.add_conditional_edges(
        "refund_decision",
        nodes.route_after_refund_decision,
        {
            "needs_approval": "human_review",
            "auto_resolved": "save_memory",
        },
    )

    # --- human-in-the-loop: the graph pauses right before this node ---
    graph.add_edge("human_review", "save_memory")

    graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["human_review"])



