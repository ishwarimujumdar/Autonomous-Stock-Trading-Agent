from functools import lru_cache

from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, GROQ_MODEL


@lru_cache(maxsize=None)
def get_llm(temperature: float, max_tokens: int | None = None) -> ChatGroq:
    """
    One client per (temperature, max_tokens) pair, reused for the whole run.

    Previously every decide()/analyze_news() call constructed a fresh ChatGroq -
    once per symbol per cycle, which over a session is thousands of throwaway
    HTTP clients. max_tokens bounds completion length per call site - decisions
    and news narratives don't need the same output budget as a full strategy
    proposal, and an unbounded completion is wasted tokens against Groq's
    tokens-per-minute rate limit.
    """
    return ChatGroq(
        api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=temperature, max_tokens=max_tokens
    )
