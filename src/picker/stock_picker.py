import asyncio
from typing import TYPE_CHECKING, Literal

from groq import RateLimitError
from pydantic import BaseModel, Field, field_validator

from src.config import ENTRY_RULE, MAX_PICK_ATTEMPTS, MAX_WATCHLIST, TRADABLE_UNIVERSE
from src.llm import get_llm

# At runtime a Literal built from our list, so the AI can only pick from it and the
# schema it is shown carries the list as a real enum. Editors can't evaluate a
# Literal built from a variable, so they are told it's a plain string.
if TYPE_CHECKING:
    UniverseSymbol = str
else:
    UniverseSymbol = Literal[tuple(TRADABLE_UNIVERSE)]

_SYSTEM_PROMPT = """You choose which stocks a day-trading agent watches for the next hour.

The agent buys a stock when this rule fires: {rule}
It only buys stocks that are rising, and sells everything before the close.
So pick stocks that move enough for the rule to fire: a wide range today
(range %) with real trading volume. Skip quiet stocks. Follow the user's hint.

Choose 1 to {max_stocks} symbols from the table, then give one short sentence of reasoning.
Columns: symbol, price, % change today vs yesterday's close, today's high-to-low
range %, yesterday's volume.
"""


class Watchlist(BaseModel):
    """What the LLM must produce."""

    symbols: list[UniverseSymbol] = Field(
        min_length=1,
        max_length=MAX_WATCHLIST,
        description=f"1 to {MAX_WATCHLIST} distinct stocks to watch, chosen from the table.",
    )
    reason: str = Field(description="One short sentence on why these stocks.")

    @field_validator("symbols")
    @classmethod
    def no_duplicates(cls, symbols: list[str]) -> list[str]:
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be distinct")
        return symbols


def _user_message(table: list[dict], hint: str, error: str | None) -> str:
    rows = "\n".join(
        f"{r['symbol']} {r['price']} {r['change_pct']:+.2f}% {r['range_pct']:.2f}% {r['yesterday_volume']}"
        for r in table
    )
    message = f"User's hint: {hint or 'none'}\n\nStocks:\n{rows}"
    if error:
        message += f"\n\nYour previous answer was rejected, fix it: {error}"
    return message


async def pick_stocks(table: list[dict], hint: str) -> tuple[Watchlist | None, str | None]:
    """
    Asks the LLM for a watchlist, retrying (with the error shown to it) up to
    MAX_PICK_ATTEMPTS times. Returns (watchlist, None), or (None, last error) so the
    caller can decide what to do: a failed pick never raises.
    """
    # 1200: this is a reasoning model; at 300 it ran out of tokens before answering (Groq 400s).
    llm = get_llm(temperature=0.3, max_tokens=1200).with_structured_output(
        Watchlist, include_raw=True
    )
    prompt = _SYSTEM_PROMPT.format(rule=ENTRY_RULE, max_stocks=MAX_WATCHLIST)

    error = None
    for attempt in range(MAX_PICK_ATTEMPTS):
        try:
            result = await llm.ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": _user_message(table, hint, error)},
                ]
            )
        except RateLimitError as exc:
            # Retrying immediately just fails again inside the same 1-minute
            # token window (seen live) - wait for it to clear, or fall back to
            # a fixed pause. Skip the wait after the last attempt.
            error = f"LLM call failed: {type(exc).__name__}: {exc}"
            if attempt < MAX_PICK_ATTEMPTS - 1:
                await asyncio.sleep(_retry_after_seconds(exc))
            continue
        except Exception as exc:  # a provider error still raises past include_raw
            error = f"LLM call failed: {type(exc).__name__}: {exc}"
            continue
        if result.get("parsed") is not None:
            return result["parsed"], None
        error = str(result.get("parsing_error") or "model returned no structured output")
    return None, error


def _retry_after_seconds(exc: RateLimitError, default: float = 6.0) -> float:
    """How long Groq says to wait before trying again, from the response header."""
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    try:
        return float(header) if header is not None else default
    except ValueError:
        return default
