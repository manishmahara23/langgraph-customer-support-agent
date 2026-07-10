from typing import Optional

from datetime import datetime
from langchain_core.tools import tool

import database
from config import REFUND_WINDOW_DAYS, REFUND_AUTO_APPROVE_LIMIT


@tool
def get_order_status(order_id: str) -> str:
    """Look up an order in the database and return its status as readable text."""
    order = database.get_order(order_id)
    if not order:
        return f"No order found with ID {order_id}."

    lines = [
        f"Order {order['order_id']}: {order['product_name']}",
        f"Status: {order['status']}",
        f"Order date: {order['order_date']}",
    ]
    if order["status"] == "delivered":
        lines.append(f"Delivered on: {order['delivery_date']}")
    lines.append(f"Amount: Rs. {order['amount']:.0f}")
    lines.append(f"Payment: {order['payment_status']}")
    return "\n".join(lines)


@tool
def check_refund_eligibility(order_id: str) -> str:
    """
    Decide whether an order qualifies for a refund, based on delivery
    status, the refund window, and the order amount. Returns a result
    that starts with one of: not_found, not_delivered, window_expired,
    eligible_auto, eligible_needs_approval - followed by an explanation.
    """
    order = database.get_order(order_id)
    if not order:
        return "not_found: No order found with that ID."

    if order["status"] != "delivered":
        return (
            f"not_delivered: Order is currently '{order['status']}'. "
            f"Refunds are only available once an order has been delivered."
        )

    delivered_on = datetime.strptime(order["delivery_date"], "%Y-%m-%d")
    days_since_delivery = (datetime.now() - delivered_on).days

    if days_since_delivery > REFUND_WINDOW_DAYS:
        return (
            f"window_expired: Order was delivered {days_since_delivery} day(s) ago, "
            f"which is beyond the {REFUND_WINDOW_DAYS}-day refund window."
        )

    if order["amount"] > REFUND_AUTO_APPROVE_LIMIT:
        return (
            f"eligible_needs_approval: Order is eligible for a refund of Rs. {order['amount']:.0f}, "
            f"but amounts above Rs. {REFUND_AUTO_APPROVE_LIMIT:.0f} need manager approval."
        )

    return f"eligible_auto: Order is eligible for an automatic refund of Rs. {order['amount']:.0f}."


@tool
def create_support_ticket(user_id: str, order_id: Optional[str], issue_type: str, description: str) -> str:
    """Create a support ticket in the database and return its ticket ID."""
    return database.create_ticket(user_id, order_id, issue_type, description)


@tool
def get_customer_history(user_id: str) -> str:
    """Summarize a customer's past orders and tickets, for context during a refund decision."""
    orders = database.list_orders_for_user(user_id)
    tickets = database.list_tickets_for_user(user_id)

    if not orders:
        return "No past orders found for this customer."

    lines = [f"Customer has {len(orders)} order(s) on record:"]
    for o in orders:
        lines.append(f"- {o['order_id']}: {o['product_name']} ({o['status']}, Rs. {o['amount']:.0f})")

    if tickets:
        lines.append(f"Customer has raised {len(tickets)} support ticket(s) before:")
        for t in tickets:
            lines.append(f"- {t['ticket_id']}: {t['issue_type']} ({t['status']})")

    return "\n".join(lines)


@tool
def save_customer_memory(user_id: str, memory_text: str) -> str:
    """Save a short, useful fact about the customer so future conversations can recall it."""
    database.save_memory_fact(user_id, memory_text)
    return "saved"
