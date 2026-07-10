from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME

_vectorstore = None  # loaded once and cached for the lifetime of the process


def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        if not VECTOR_STORE_DIR.exists():
            raise FileNotFoundError(
                "Knowledge base index not found. Run `python build_knowledge_base.py` first."
            )
        _vectorstore = FAISS.load_local(
            str(VECTOR_STORE_DIR),
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def retrieve_policy_context(query: str, k: int = 3) -> str:
    """Return the top-k most relevant policy chunks as one text block,
    each tagged with the file it came from."""
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    if not results:
        return "No relevant policy information found."
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'policy')}] {doc.page_content}" for doc in results
    )
