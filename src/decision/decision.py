from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.config import ENTRY_RULE, EXIT_RULE, INTRADAY_TIMEFRAME
from src.llm import get_llm

_SYSTEM_PROMPT = """You are the Trade Decision node for a stock trading system.

For the given symbol, decide BUY, SELL, or HOLD using:
- Current market evidence (intraday measurements below)
- Previous trade outcomes for this symbol
- Overall performance state so far this run
- Current Alpaca account/order state (cash, existing holdings, open orders)

This is intraday day trading. All indicators below are computed on {timeframe}
bars, and any position opened will be force-closed before today's market
close, so judge the next minutes-to-hours, not the next days.

Entry rule: {entry_rule}
Exit rule: {exit_rule}
The entry rule is your main trigger: when it holds and max_buy_qty is above 0,
BUY unless something clearly argues against it. Otherwise HOLD.

Rules for your output:
- action must be exactly BUY, SELL or HOLD.
- If action is BUY or SELL, target_qty must be a positive whole number of
  shares. If action is HOLD, leave target_qty null.
- For a BUY, target_qty must not exceed max_buy_qty below. The Risk Gate
  rejects an oversized BUY outright rather than resizing it, so a good signal
  is wasted. If max_buy_qty is 0 there is no room left: HOLD.
- You may only SELL shares that are currently held - never propose a SELL
  larger than the holding shown below, and never propose a SELL with no
  holding. Shorting is not permitted.
- rationale must be ONE short sentence (under 20 words). This runs once per
  candidate per cycle - be terse, not exhaustive.

You do not enforce risk limits yourself - a separate Risk Gate will reject
your decision if it violates hard constraints (min confidence, position
size, buying power, loss limits). Give your honest confidence; do not
inflate it and do not shade it to just clear a threshold you don't know.
"""


class DecisionOutput(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    target_qty: int | None = Field(
        default=None, description="Positive whole share count when action is BUY or SELL"
    )

    @model_validator(mode="after")
    def qty_matches_action(self):
        if self.action == "HOLD":
            self.target_qty = None
        elif not self.target_qty or self.target_qty <= 0:
            raise ValueError(f"{self.action} requires a positive target_qty")
        return self


def _evidence_summary(technical: dict, performance: dict, open_orders: list[dict]) -> tuple:
    """
    Only what the model can use, to save tokens (this runs once per symbol per
    cycle): indicators that computed, headline P&L figures, and open order basics.
    """
    indicators = {
        k: v
        for k, v in technical.items()
        if k not in ("symbol", "timeframe", "bars_available") and v is not None
    }
    headline = {
        k: v
        for k, v in performance.items()
        if k in ("realized_pnl", "unrealized_pnl", "return_pct", "wins", "losses")
    }
    orders = [{"symbol": o.get("symbol"), "side": o.get("side"), "qty": o.get("qty")} for o in open_orders]
    return indicators, headline, orders


def _max_buy_qty(symbol: str, account: dict, objective_constraints: dict) -> int:
    """
    The largest BUY the Risk Gate's trade-size and position-size caps would
    allow right now. Shown to the model so it sizes trades instead of guessing.
    Cash and total-exposure caps are left to the Risk Gate.
    """
    price = account.get("last_prices", {}).get(symbol)
    if not price or price <= 0:
        return 0

    c = objective_constraints["constraints"]
    capital = c["session_capital"]
    held_qty = account.get("positions", {}).get(symbol, {}).get("qty", 0)

    # Two ceilings on the money for this trade: one trade's size, and how
    # much of this stock we may hold in total (minus what we already hold).
    max_trade_value = c["max_trade_size_pct"] * capital
    room_in_position = c["max_position_pct"] * capital - held_qty * price
    money = max(min(max_trade_value, room_in_position), 0)
    return int(money // price)


def _fallback_hold(symbol: str, error: Exception) -> dict:
    """A decision we couldn't parse becomes a HOLD, not a crash."""
    return {
        "symbol": symbol,
        "action": "HOLD",
        "confidence": 0.0,
        "rationale": f"decision unavailable - model output could not be parsed: {error}",
        "target_qty": None,
    }


def _user_message(
    symbol: str,
    technical: dict,
    previous_outcomes: list[dict],
    performance: dict,
    account: dict,
    objective_constraints: dict,
) -> str:
    """Everything the model is told about this one stock, this cycle."""
    indicators, headline, orders = _evidence_summary(
        technical, performance, account.get("open_orders", [])
    )
    return (
        f"Symbol: {symbol}\n"
        f"Current price: {account.get('last_prices', {}).get(symbol)}\n"
        f"Intraday technical evidence: {indicators}\n"
        f"Previous trade outcomes for this symbol: {previous_outcomes}\n"
        f"Performance so far this run: {headline}\n"
        f"Account state: cash={account.get('cash')}, "
        f"holdings={account.get('positions', {}).get(symbol) or 'none'}, "
        f"minutes_to_close={account.get('minutes_to_close')}, "
        f"open_orders={orders}\n"
        f"max_buy_qty: {_max_buy_qty(symbol, account, objective_constraints)} shares "
        "(see rules above - a BUY over this is rejected, not resized)\n"
    )


async def decide(
    symbol: str,
    technical: dict,
    previous_outcomes: list[dict],
    performance: dict,
    account: dict,
    objective_constraints: dict,
) -> dict:
    # 1200: this is a reasoning model; at 300 it ran out of tokens before
    # calling the output tool, and Groq 400'd with "Tool choice is required,
    # but model did not call a tool" (seen live - 5 straight HOLDs on one
    # held stock while its stop-loss should have been checked).
    llm = get_llm(temperature=0.1, max_tokens=1200).with_structured_output(
        DecisionOutput, include_raw=True
    )

    prompt = _SYSTEM_PROMPT.format(
        entry_rule=ENTRY_RULE, exit_rule=EXIT_RULE, timeframe=INTRADAY_TIMEFRAME
    )
    human_message = _user_message(
        symbol, technical, previous_outcomes, performance, account, objective_constraints
    )

    try:
        result = await llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": human_message},
            ]
        )
    except Exception as exc:
        # include_raw only catches parse failures; a provider error still raises.
        return _fallback_hold(symbol, exc)

    parsed = result.get("parsed")
    if parsed is None:
        return _fallback_hold(symbol, result.get("parsing_error") or ValueError("no output"))
    return {"symbol": symbol, **parsed.model_dump()}
