import re
from langchain_core.messages import AIMessage

import database
from agent.llm import get_llm
from agent.rag import retrieve_policy_context
from agent.tools import (
    get_order_status,
    check_refund_eligibility,
    create_support_ticket,
    get_customer_history,
    save_customer_memory,
)

ORDER_ID_PATTERN = re.compile(r"ORD\d+", re.IGNORECASE)


def _latest_user_text(state):
    for message in reversed(state["messages"]):
        if message.type == "human":
            return message.content
    return ""


def _extract_order_id(text):
    match = ORDER_ID_PATTERN.search(text)
    return match.group(0).upper() if match else None


def _memory_summary(state):
    facts = state.get("long_term_memory") or []
    return "\n".join(f"- {fact}" for fact in facts) if facts else "No prior history with this customer."


# ---------------------------------------------------------------------
# 1. Memory loader - always runs first (long-term memory)
# ---------------------------------------------------------------------
def load_memory_node(state):
    facts = database.load_memory_facts(state["user_id"])
    return {"long_term_memory": facts}


# ---------------------------------------------------------------------
# 2. Intent router - conditional workflow starts here
# ---------------------------------------------------------------------
def classify_intent_node(state):
    # if we're already in the middle of collecting refund details, stay
    # on that flow instead of re-classifying every short reply
    if state.get("pending_action") == "awaiting_refund_info":
        return {"intent": "refund"}

    user_text = _latest_user_text(state)
    llm = get_llm(temperature=0)

    prompt = (
        "Classify the customer's message into exactly one category:\n"
        "faq - general questions about policy, shipping, returns, or payments\n"
        "order_status - asking where an order is or what its delivery status is\n"
        "refund - wants a refund or return, or has a damaged/wrong/missing item\n"
        "ticket - a complaint, or explicitly wants to talk to a human agent\n"
        "unclear - greetings, small talk, or anything that doesn't fit above\n\n"
        f'Customer message: "{user_text}"\n\n'
        "Reply with only the category word, nothing else."
    )
    raw = llm.invoke(prompt).content.strip().lower()

    valid_intents = ["faq", "order_status", "refund", "ticket", "unclear"]
    intent = next((i for i in valid_intents if i in raw), "unclear")
    return {"intent": intent}


def route_after_classification(state):
    return state.get("intent", "unclear")


# ---------------------------------------------------------------------
# 3. FAQ - linear workflow + RAG
# ---------------------------------------------------------------------
def faq_node(state):
    user_text = _latest_user_text(state)
    context = retrieve_policy_context(user_text)

    llm = get_llm(temperature=0.3)
    prompt = (
        "You are a friendly customer support assistant for an online store called ShopEasy.\n"
        "Answer the customer's question using ONLY the policy information below. If the answer "
        "isn't in there, say you're not fully sure and offer to raise a support ticket instead.\n\n"
        f"Policy information:\n{context}\n\n"
        f"Customer question: {user_text}\n\n"
        "Reply in 2-4 short, natural sentences."
    )
    answer = llm.invoke(prompt).content
    return {"final_response": answer, "messages": [AIMessage(content=answer)]}


# ---------------------------------------------------------------------
# 4. Order status - linear workflow + tool calling
# ---------------------------------------------------------------------
def order_status_node(state):
    user_text = _latest_user_text(state)
    order_id = _extract_order_id(user_text) or state.get("order_id")

    if not order_id:
        answer = "Sure, I can check that. Could you share your order ID? It usually looks like ORD101."
        return {"final_response": answer, "messages": [AIMessage(content=answer)]}

    order_info = get_order_status.invoke({"order_id": order_id})

    llm = get_llm(temperature=0.3)
    prompt = (
        "Rewrite the following order information as a short, friendly reply to the customer. "
        "Keep every fact exactly the same, just make it sound natural and conversational.\n\n"
        f"{order_info}"
    )
    answer = llm.invoke(prompt).content
    return {
        "order_id": order_id,
        "order_details": order_info,
        "final_response": answer,
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------
# 5. Collect refund info - iterative workflow, loops across turns
# ---------------------------------------------------------------------
def collect_refund_info_node(state):
    user_text = _latest_user_text(state)

    order_id = state.get("order_id")
    reason = state.get("refund_reason")

    found_order_id = _extract_order_id(user_text)
    if found_order_id:
        order_id = found_order_id
    elif order_id and not reason:
        # we already know the order, so whatever they just said is the reason
        reason = user_text.strip()

    missing = []
    if not order_id:
        missing.append("order_id")
    if not reason:
        missing.append("reason")

    if missing:
        if "order_id" in missing:
            question = "I'm sorry to hear that. Could you share the order ID for the item you'd like refunded?"
        else:
            question = "Got it. What's the reason for the refund - for example damaged, wrong item, or no longer needed?"
        return {
            "order_id": order_id,
            "refund_reason": reason,
            "missing_fields": missing,
            "pending_action": "awaiting_refund_info",
            "final_response": question,
            "messages": [AIMessage(content=question)],
        }

    # everything we need is here - clear the pending flag and move on
    return {
        "order_id": order_id,
        "refund_reason": reason,
        "missing_fields": [],
        "pending_action": None,
    }


def route_after_collect_info(state):
    return "ask_more" if state.get("missing_fields") else "ready"


# ---------------------------------------------------------------------
# 6. Parallel workflow - fan out to three independent lookups at once
# ---------------------------------------------------------------------
def dispatch_refund_node(state):
    # this node has no work of its own - it exists purely so the graph
    # can fan out from one place into the three lookups below, which
    # LangGraph then runs in the same step (i.e. in parallel)
    return {}


def fetch_order_details_node(state):
    return {"order_details": get_order_status.invoke({"order_id": state["order_id"]})}


def fetch_refund_policy_node(state):
    query = f"refund eligibility {state.get('refund_reason', '')}"
    return {"policy_context": retrieve_policy_context(query)}


def fetch_customer_history_node(state):
    return {"customer_history": get_customer_history.invoke({"user_id": state["user_id"]})}


# ---------------------------------------------------------------------
# 7. Refund decision - conditional workflow, merges the parallel results
# ---------------------------------------------------------------------
def refund_decision_node(state):
    eligibility = check_refund_eligibility.invoke({"order_id": state["order_id"]})
    needs_approval = eligibility.startswith("eligible_needs_approval")

    # every refund request that concludes without escalation still gets a
    # ticket reference - whether approved or turned down - so the customer
    # always has something to quote if they follow up later
    ticket_id = None
    outcome = eligibility.split(":")[0]
    if not needs_approval:
        ticket_id = create_support_ticket.invoke(
            {
                "user_id": state["user_id"],
                "order_id": state["order_id"],
                "issue_type": f"refund_{outcome}",
                "description": state.get("refund_reason", ""),
            }
        )

    llm = get_llm(temperature=0.3)
    instruction = (
        "Tell them their refund needs manager approval and you'll get back to them shortly."
        if needs_approval
        else "If eligible, confirm the refund clearly and mention the ticket reference. If not "
             "eligible, explain why politely and mention any next step (such as returning the "
             "item or waiting for delivery)."
    )
    prompt = (
        "You are a customer support assistant. Write a short, clear reply to the customer about "
        "their refund request, using only the information below - don't invent anything.\n\n"
        f"Order details:\n{state.get('order_details')}\n\n"
        f"Refund eligibility check:\n{eligibility}\n\n"
        f"Ticket reference: {ticket_id or 'will be created once a manager reviews this'}\n\n"
        f"Relevant policy:\n{state.get('policy_context')}\n\n"
        f"What we know about this customer:\n{_memory_summary(state)}\n\n"
        f"Customer's stated reason: {state.get('refund_reason')}\n\n"
        f"{instruction}"
    )
    answer = llm.invoke(prompt).content

    if not needs_approval:
        # the flow ends right here, so this is the right moment to remember the outcome
        fact = (
            f"Customer requested a refund for order {state['order_id']} "
            f"(reason: {state.get('refund_reason')}). Outcome: {outcome}. Ticket: {ticket_id}."
        )
        save_customer_memory.invoke({"user_id": state["user_id"], "memory_text": fact})

    return {
        "refund_eligible": eligibility,
        "needs_human_approval": needs_approval,
        "ticket_id": ticket_id,
        "final_response": answer,
        "messages": [AIMessage(content=answer)],
    }


def route_after_refund_decision(state):
    return "needs_approval" if state.get("needs_human_approval") else "auto_resolved"


# ---------------------------------------------------------------------
# 8. Human review (HITL) - the graph pauses BEFORE this node runs
# ---------------------------------------------------------------------
def human_review_node(state):
    decision = state.get("human_decision") or "approved"

    if decision == "approved":
        ticket_id = create_support_ticket.invoke(
            {
                "user_id": state["user_id"],
                "order_id": state["order_id"],
                "issue_type": "refund_approved",
                "description": state.get("refund_reason", ""),
            }
        )
        answer = (
            f"Good news - your refund has been approved by our support manager. "
            f"Reference ticket: {ticket_id}. The amount will be credited within 3-5 business days."
        )
    else:
        ticket_id = create_support_ticket.invoke(
            {
                "user_id": state["user_id"],
                "order_id": state["order_id"],
                "issue_type": "refund_rejected",
                "description": state.get("refund_reason", ""),
            }
        )
        answer = (
            f"After review, we're unable to approve this refund. "
            f"Reference ticket: {ticket_id}. Feel free to reply here if you'd like to add more details."
        )

    save_customer_memory.invoke(
        {
            "user_id": state["user_id"],
            "memory_text": f"Refund request for order {state['order_id']} was {decision} by a human reviewer (ticket {ticket_id}).",
        }
    )

    return {
        "ticket_id": ticket_id,
        "final_response": answer,
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------
# 9. Ticket / escalation - for general complaints and "talk to a human"
# ---------------------------------------------------------------------
def create_ticket_node(state):
    user_text = _latest_user_text(state)
    order_id = state.get("order_id") or _extract_order_id(user_text)

    ticket_id = create_support_ticket.invoke(
        {
            "user_id": state["user_id"],
            "order_id": order_id,
            "issue_type": "general_complaint",
            "description": user_text,
        }
    )
    save_customer_memory.invoke(
        {
            "user_id": state["user_id"],
            "memory_text": f"Customer raised a complaint (ticket {ticket_id}): {user_text[:120]}",
        }
    )

    answer = (
        f"I've created a support ticket for you - reference {ticket_id}. "
        f"A member of our human support team will follow up with you shortly."
    )
    return {
        "ticket_id": ticket_id,
        "final_response": answer,
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------
# 10. Fallback for unclear intent
# ---------------------------------------------------------------------
def unclear_intent_node(state):
    answer = (
        "I'm not sure I caught that. I can help you track an order, answer a policy question, "
        "process a refund, or connect you with a human agent - what would you like to do?"
    )
    return {"final_response": answer, "messages": [AIMessage(content=answer)]}


# ---------------------------------------------------------------------
# 11. Save memory - runs at the end of every turn, logs the transcript
# ---------------------------------------------------------------------
def save_memory_node(state):
    user_text = _latest_user_text(state)
    database.log_conversation(state["user_id"], state["user_id"], "user", user_text)
    if state.get("final_response"):
        database.log_conversation(state["user_id"], state["user_id"], "assistant", state["final_response"])
    return {}
