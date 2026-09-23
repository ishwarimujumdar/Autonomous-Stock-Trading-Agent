from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.llm import get_llm

_SYSTEM_PROMPT = """You are the Trade Decision node for a stock trading system.

For the given symbol, decide BUY, SELL, or HOLD using:
- Current market evidence (intraday technical + narrative below)
- Previous trade outcomes for this symbol/strategy
- Overall performance state so far this run
- Current Alpaca account/order state (cash, existing holdings, open orders)

This is intraday day trading. All indicators below are computed on {timeframe}
bars, and any position opened will be force-closed before today's market
close, so judge the next minutes-to-hours, not the next days.

Strategy entry logic: {entry_logic}
Strategy exit logic: {exit_logic}

Rules for your output:
- action must be exactly BUY, SELL or HOLD.
- If action is BUY or SELL, target_qty must be a positive whole number of
  shares. If action is HOLD, leave target_qty null.
- For a BUY, target_qty must not exceed max_buy_qty given below - that number
  already accounts for the trade-size and position-size limits, computed from
  the current price. A BUY over it is rejected outright by the Risk Gate, not
  resized, so asking for more than max_buy_qty wastes a real signal for
  nothing - if max_buy_qty is 0, there is no room left and HOLD is correct
  even on a good setup.
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
    # A bare `str` here let a lowercase "buy" through the gate as a non-HOLD
    # action, which the executor then submitted as a SELL.
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


def _compact_technical(technical: dict) -> dict:
    """
    Drops fields with no decision value: None entries (a signal that didn't
    compute yet), and symbol/timeframe/bars_available, which are already
    stated elsewhere in the prompt or are pure diagnostics. This is called
    once per candidate per cycle, so every dropped key is real, repeated
    savings against the tokens-per-minute budget.
    """
    skip = {"symbol", "timeframe", "bars_available"}
    return {k: v for k, v in technical.items() if k not in skip and v is not None}


def _compact_performance(performance: dict) -> dict:
    """
    Only the aggregate figures - not `recent_trades`, which is already sent
    separately (and more relevantly) as `previous_outcomes` for this symbol.
    Sending the full trade history in every one of a cycle's N decide() calls
    was pure duplication.
    """
    return {
        k: v
        for k, v in performance.items()
        if k in ("realized_pnl", "unrealized_pnl", "return_pct", "wins", "losses")
    }


def _compact_orders(open_orders: list[dict]) -> list[dict]:
    """Just enough to know what's already working, not Alpaca's full order object."""
    return [
        {"symbol": o.get("symbol"), "side": o.get("side"), "qty": o.get("qty")}
        for o in open_orders
    ]


def _max_buy_qty(symbol: str, account: dict, objective_constraints: dict) -> int:
    """
    The largest BUY the Risk Gate could actually approve for this symbol right
    now, from the trade-size and position-size caps.

    Without this, the model was guessing a quantity with no sense of scale -
    confirmed live: a genuine bullish signal on AMD proposed 16 shares
    ($9,890, five times the $2,000 trade-size cap) and was rejected outright,
    losing the trade entirely rather than getting sized down. The Risk Gate
    rejects an oversized BUY rather than resizing it, so a bad guess here
    wastes a real signal. Cash and gross-exposure caps aren't included - those
    depend on what else this cycle's other decisions do, which only the
    Risk Gate's shared CycleBudget can see; it still enforces those
    independently of this estimate.
    """
    price = account.get("last_prices", {}).get(symbol)
    if not price or price <= 0:
        return 0

    constraints = objective_constraints["constraints"]
    session_capital = constraints["session_capital"]
    max_trade_value = constraints["max_trade_size_pct"] * session_capital

    held_qty = account.get("positions", {}).get(symbol, {}).get("qty", 0)
    held_value = held_qty * price
    max_position_value = constraints["max_position_pct"] * session_capital
    room_by_position = max(max_position_value - held_value, 0)

    max_value = min(max_trade_value, room_by_position)
    return max(int(max_value // price), 0)


def _fallback_hold(symbol: str, error: Exception) -> dict:
    """
    A malformed decision is a HOLD, not a crash. One symbol's unparseable
    response shouldn't end a trading session that has open positions.
    """
    return {
        "symbol": symbol,
        "action": "HOLD",
        "confidence": 0.0,
        "rationale": f"decision unavailable - model output could not be parsed: {error}",
        "target_qty": None,
    }


async def decide(
    symbol: str,
    strategy: dict,
    technical: dict,
    narrative: str | None,
    previous_outcomes: list[dict],
    performance: dict,
    account: dict,
    timeframe: str,
    objective_constraints: dict,
) -> dict:
    # max_tokens=300: the whole output is one short rationale plus three
    # small fields - no call site here should ever need more.
    llm = get_llm(temperature=0.1, max_tokens=300).with_structured_output(
        DecisionOutput, include_raw=True
    )

    prompt = _SYSTEM_PROMPT.format(
        entry_logic=strategy["entry_logic"],
        exit_logic=strategy["exit_logic"],
        timeframe=timeframe,
    )

    holding = account.get("positions", {}).get(symbol)
    max_buy_qty = _max_buy_qty(symbol, account, objective_constraints)
    human_message = (
        f"Symbol: {symbol}\n"
        f"Current price: {account.get('last_prices', {}).get(symbol)}\n"
        f"Intraday technical evidence: {_compact_technical(technical)}\n"
        f"Narrative: {narrative or 'none'}\n"
        f"Previous trade outcomes for this symbol: {previous_outcomes}\n"
        f"Performance so far this run: {_compact_performance(performance)}\n"
        f"Account state: cash={account.get('cash')}, "
        f"holdings={holding or 'none'}, "
        f"minutes_to_close={account.get('minutes_to_close')}, "
        f"open_orders={_compact_orders(account.get('open_orders', []))}\n"
        f"max_buy_qty: {max_buy_qty} shares (see rules above - a BUY over this is rejected, not resized)\n"
    )

    try:
        result = await llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": human_message},
            ]
        )
    except Exception as exc:
        # include_raw=True only catches a PARSING failure - a raw provider
        # error (rate limit surviving the client's own retries, a truncated
        # generation rejected server-side, a transient outage) still raises
        # here. This runs once per candidate per cycle; one bad call must
        # not end a session that may have open positions to manage.
        return _fallback_hold(symbol, exc)

    parsed = result.get("parsed")
    if parsed is None:
        return _fallback_hold(symbol, result.get("parsing_error") or ValueError("no output"))
    return {"symbol": symbol, **parsed.model_dump()}
