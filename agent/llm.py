from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL


def get_llm(temperature: float = 0.2) -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError(
            "LLM_API_KEY is not set."
        )
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=temperature,
    )
