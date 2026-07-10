from typing import TypedDict, Annotated, Optional, List
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ---- core conversation ----
    messages: Annotated[list, add_messages]   # full chat history, auto-accumulated
    user_id: str

    # ---- routing ----
    intent: Optional[str]            # faq | order_status | refund | ticket | unclear
    pending_action: Optional[str]    # set while we're in the middle of a multi-turn flow

    # ---- order / refund flow ----
    order_id: Optional[str]
    refund_reason: Optional[str]
    missing_fields: Optional[List[str]]

    # filled in by the three parallel lookups during refund evaluation
    order_details: Optional[str]
    policy_context: Optional[str]
    customer_history: Optional[str]

    refund_eligible: Optional[str]       # raw result from the eligibility tool
    needs_human_approval: bool
    human_decision: Optional[str]        # "approved" | "rejected", set by the admin

    # ---- support tickets ----
    ticket_id: Optional[str]

    # ---- long-term memory, loaded fresh at the start of every turn ----
    long_term_memory: Optional[List[str]]

    # ---- what the agent actually says back this turn ----
    final_response: Optional[str]
