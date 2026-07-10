import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

from config import DATABASE_PATH


@contextmanager
def get_connection():
    """Open a connection, commit on success, always close - so callers
    never have to remember to do that themselves."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create every table the agent needs. Safe to call on every app
    startup - CREATE TABLE IF NOT EXISTS won't touch existing data."""
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT
            );

            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                product_name TEXT NOT NULL,
                status TEXT NOT NULL,           -- processing | shipped | delivered
                order_date TEXT NOT NULL,
                delivery_date TEXT,              -- NULL until status = delivered
                amount REAL NOT NULL,
                payment_status TEXT NOT NULL,    -- paid | pending
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            );

            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                order_id TEXT,
                issue_type TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS customer_memory (
                memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                memory_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,             -- user | assistant
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            """
        )


def seed_demo_data():
    """Insert a handful of fake users/orders so the agent has something
    real to look up during a demo. Skips itself if data already exists.
    Returns True if it actually seeded anything."""
    with get_connection() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if existing > 0:
            return False

        today = datetime.now()

        def days_ago(n):
            return (today - timedelta(days=n)).strftime("%Y-%m-%d")

        users = [
            ("U001", "Aarav Sharma", "aarav.sharma@example.com", "9876500001"),
            ("U002", "Priya Singh", "priya.singh@example.com", "9876500002"),
            ("U003", "Rohan Verma", "rohan.verma@example.com", "9876500003"),
        ]
        conn.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", users)

        # a mix of statuses/amounts so every branch of the refund logic
        # (auto-approve, needs approval, window expired, not delivered)
        # has a real order to demo against
        orders = [
            ("ORD101", "U001", "Wireless Headphones", "delivered", days_ago(10), days_ago(3), 4999.0, "paid"),
            ("ORD102", "U001", "Bluetooth Speaker", "shipped", days_ago(2), None, 2499.0, "paid"),
            ("ORD103", "U002", "Running Shoes", "delivered", days_ago(15), days_ago(10), 3499.0, "paid"),
            ("ORD104", "U002", "Smartwatch", "delivered", days_ago(5), days_ago(2), 12999.0, "paid"),
            ("ORD105", "U003", "Laptop Bag", "processing", days_ago(1), None, 1299.0, "paid"),
            ("ORD106", "U003", "Yoga Mat", "delivered", days_ago(4), days_ago(1), 899.0, "paid"),
        ]
        conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)", orders
        )

        # one fact planted up front so long-term memory is visible from the
        # very first conversation, without needing two separate sessions
        conn.execute(
            "INSERT INTO customer_memory (user_id, memory_text, created_at) VALUES (?, ?, ?)",
            (
                "U002",
                "Customer previously asked about a delayed delivery for order ORD103.",
                today.isoformat(timespec="seconds"),
            ),
        )
        return True


# ---------------------------------------------------------------------
# query helpers - the rest of the app talks to the database through
# these functions instead of writing raw SQL everywhere
# ---------------------------------------------------------------------

def list_users():
    with get_connection() as conn:
        return conn.execute("SELECT * FROM users ORDER BY user_id").fetchall()


def get_user(user_id):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()


def get_order(order_id):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()


def list_orders_for_user(user_id):
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM orders WHERE user_id = ? ORDER BY order_date DESC", (user_id,)
        ).fetchall()


def create_ticket(user_id, order_id, issue_type, description):
    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        ticket_id = f"TCK{count + 1:04d}"
        conn.execute(
            """INSERT INTO tickets (ticket_id, user_id, order_id, issue_type, description, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'open', ?)""",
            (ticket_id, user_id, order_id, issue_type, description, datetime.now().isoformat(timespec="seconds")),
        )
        return ticket_id


def list_tickets_for_user(user_id):
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM tickets WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        ).fetchall()


def save_memory_fact(user_id, memory_text):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO customer_memory (user_id, memory_text, created_at) VALUES (?, ?, ?)",
            (user_id, memory_text, datetime.now().isoformat(timespec="seconds")),
        )


def load_memory_facts(user_id):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT memory_text FROM customer_memory WHERE user_id = ? ORDER BY created_at DESC LIMIT 10",
            (user_id,),
        ).fetchall()
        return [row["memory_text"] for row in rows]


def log_conversation(user_id, session_id, role, message):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (user_id, session_id, role, message, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, session_id, role, message, datetime.now().isoformat(timespec="seconds")),
        )
