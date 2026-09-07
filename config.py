import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# all generated/runtime data (db files, vector index) lives here so it's
# easy to wipe and start fresh - just delete this one folder
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_PATH = DATA_DIR / "support_agent.db"
CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.db"
VECTOR_STORE_DIR = DATA_DIR / "faiss_index"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"

# ---- LLM (Groq is free - https://console.groq.com) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b"

# ---- embeddings used for RAG, runs locally, no API key needed ----
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME") or "sentence-transformers/all-MiniLM-L6-v2"

# ---- refund business rules ----
REFUND_WINDOW_DAYS = 7          # refunds only allowed within this many days of delivery
REFUND_AUTO_APPROVE_LIMIT = 5000  # orders above this amount (Rs.) need human approval
