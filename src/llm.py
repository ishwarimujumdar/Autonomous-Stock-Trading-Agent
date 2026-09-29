from functools import lru_cache

from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, GROQ_MODEL


@lru_cache(maxsize=None)
def get_llm(temperature: float, max_tokens: int | None = None) -> ChatGroq:
    """One reusable client per (temperature, max_tokens); max_tokens caps each call's cost."""
    return ChatGroq(
        api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=temperature, max_tokens=max_tokens
    )
