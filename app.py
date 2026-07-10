import sqlite3
import time
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

import database
import build_knowledge_base
from agent.graph import build_graph
from config import CHECKPOINT_DB_PATH, VECTOR_STORE_DIR, REFUND_WINDOW_DAYS, REFUND_AUTO_APPROVE_LIMIT

st.set_page_config(page_title="ShopEasy Support", page_icon="🛒", layout="wide")

NODE_STATUS_LABELS = {
    "load_memory": "📚 Loading your history...",
    "classify_intent": "🤔 Understanding your request...",
    "faq": "🔍 Checking our policies...",
    "order_status": "📦 Looking up your order...",
    "collect_refund_info": "📝 Reviewing the details...",
    "dispatch_refund": "⚡ Starting refund review...",
    "fetch_order_details": "📦 Fetching order details...",
    "fetch_refund_policy": "📜 Checking refund policy...",
    "fetch_customer_history": "🕘 Reviewing your history...",
    "refund_decision": "⚖️ Making a decision...",
    "create_ticket": "🎫 Creating a support ticket...",
    "unclear_intent": "🤷 Thinking...",
    "save_memory": "💾 Saving this conversation...",
}


@st.cache_resource
def ensure_setup():
    database.init_db()
    database.seed_demo_data()
    if not VECTOR_STORE_DIR.exists():
        build_knowledge_base.main()
    return True


@st.cache_resource
def get_graph():
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return build_graph(checkpointer)


def stream_text(text: str):
    """Fake a real-time, word-by-word reveal of the final answer - this
    is what 'streaming' looks like in the UI, even though the LLM call
    itself already finished before we start revealing the text."""
    placeholder = st.empty()
    shown = ""
    for word in text.split(" "):
        shown += word + " "
        placeholder.markdown(shown)
        time.sleep(0.02)


def handle_user_message(graph, config, user_id, user_text):
    with st.chat_message("user"):
        st.write(user_text)

    with st.chat_message("assistant"):
        status = st.empty()
        with st.spinner("Working on it..."):
            for update in graph.stream(
                {
                    "messages": [HumanMessage(content=user_text)],
                    "user_id": user_id,
                    "needs_human_approval": False,
                    "human_decision": None,
                },
                config,
                stream_mode="updates",
            ):
                node_name = next(iter(update))
                status.caption(NODE_STATUS_LABELS.get(node_name, f"Running {node_name}..."))

        status.empty()
        final_state = graph.get_state(config).values
        response_text = final_state.get("final_response") or "Sorry, something went wrong on my end."
        stream_text(response_text)


def render_human_review_panel(graph, config, state):
    st.warning("⏸️ This refund needs manager approval before the agent can reply.")
    with st.container(border=True):
        st.markdown(f"**Order:** {state.get('order_id')}")
        st.markdown(f"**Reason given:** {state.get('refund_reason')}")
        st.markdown(f"**Eligibility check:** {state.get('refund_eligible')}")

        col1, col2 = st.columns(2)
        if col1.button("✅ Approve refund", use_container_width=True):
            graph.update_state(config, {"human_decision": "approved"})
            with st.spinner("Finalizing..."):
                for _ in graph.stream(None, config, stream_mode="updates"):
                    pass
            st.rerun()

        if col2.button("❌ Reject refund", use_container_width=True):
            graph.update_state(config, {"human_decision": "rejected"})
            with st.spinner("Finalizing..."):
                for _ in graph.stream(None, config, stream_mode="updates"):
                    pass
            st.rerun()



# app start

ensure_setup()
graph = get_graph()

# ---- sidebar ----
st.sidebar.title("🛒 ShopEasy Support")
st.sidebar.caption("AI customer support agent - built with LangGraph + Groq")

users = database.list_users()
user_labels = {f"{u['user_id']} - {u['name']}": u["user_id"] for u in users}
selected_label = st.sidebar.selectbox("You're chatting as:", list(user_labels.keys()))
user_id = user_labels[selected_label]

if "thread_suffix" not in st.session_state:
    st.session_state.thread_suffix = "main"

if st.sidebar.button("🔄 Start a new conversation"):
    st.session_state.thread_suffix = uuid.uuid4().hex[:8]
    st.rerun()

thread_id = f"{user_id}-{st.session_state.thread_suffix}"

st.sidebar.divider()
st.sidebar.subheader("Recent orders")
for order in database.list_orders_for_user(user_id):
    st.sidebar.markdown(
        f"**{order['order_id']}** - {order['product_name']}  \n"
        f"_{order['status']}, Rs. {order['amount']:.0f}_"
    )

memory_facts = database.load_memory_facts(user_id)
if memory_facts:
    st.sidebar.divider()
    st.sidebar.subheader("🧠 What I remember about you")
    for fact in memory_facts:
        st.sidebar.caption(f"- {fact}")

st.sidebar.divider()
st.sidebar.caption(
    f"Refund window: {REFUND_WINDOW_DAYS} days  \n"
    f"Auto-approve limit: Rs. {REFUND_AUTO_APPROVE_LIMIT:.0f}"
)

# ---- main chat area ----
st.title("Chat with ShopEasy Support")
st.caption(f"Thread: `{thread_id}` - try asking about an order, a refund, or a policy question.")

config = {"configurable": {"thread_id": thread_id}}
snapshot = graph.get_state(config)
existing_messages = snapshot.values.get("messages", []) if snapshot.values else []

if not existing_messages:
    with st.chat_message("assistant"):
        st.write("Hi! I'm the ShopEasy support assistant. How can I help you today?")

for message in existing_messages:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(message.content)

awaiting_review = bool(snapshot.next) and "human_review" in snapshot.next
if awaiting_review:
    render_human_review_panel(graph, config, snapshot.values)

user_text = st.chat_input("Type your message...")
if user_text:
    handle_user_message(graph, config, user_id, user_text)
    st.rerun()
