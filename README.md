# Stock Trading Agent

A LangGraph-orchestrated day-trading agent that trades US stocks through a
live Alpaca MCP server on paper trading. An LLM proposes *what* to trade and
*why*; deterministic code decides whether any of it is actually allowed to
happen. The LLM can never place an order, size a position, or override a risk
limit directly — every dollar-moving decision passes through code that has no
opinion and cannot be argued with.

BUY/SELL/HOLD on stocks only — no options, no margin, no shorting. Runs for a
single session: start it any time the market is open, it re-evaluates on a
cadence it chooses itself, and it force-flattens every position before the
close. No overnight exposure, ever.

## Design philosophy

The core bet this project makes: an LLM is good at *reading* a messy, ambiguous
situation and proposing a plausible course of action, and bad at being trusted
with the actual execution of anything irreversible. So the architecture puts a
hard boundary between the two:

| | LLM decides | Code decides |
|---|---|---|
| What strategy to run, which signals to watch | ✅ | |
| Whether that strategy is schema-valid | | ✅ (rejects and forces a retry) |
| Which symbols pass a numeric filter | | ✅ (no LLM calls per candidate) |
| BUY / SELL / HOLD and confidence, per symbol | ✅ | |
| Whether that decision violates a risk limit | | ✅ (confidence floor, position size, exposure, daily loss, market hours, no shorting) |
| Order placement, fill confirmation, reconciliation | | ✅ |
| Whether to end the session and force-flatten | | ✅ (a fixed clock deadline, not a vibe) |

The LLM can propose being *more* conservative than the fixed limits. It can
never propose being less. If it tries, the proposal is rejected and it gets
told exactly why, so it can retry — the retry loop is itself bounded, so a
model that can't produce anything valid fails safely instead of looping
forever.

## Architecture

```
Strategy Agent (LLM) → Schema Validator (code) → Scanner (code)
                                                       ↓
                                                   Research (code + LLM)
                                                       ↓
                                              Trade Decision (LLM, per symbol)
                                                       ↓
                                                 Risk Gate (code)
                                                       ↓
                                                  Execution (code)
                                                       ↓
                                                   Monitor (code)
                                                       ↓
                                            Performance Tracker (code)
                                                       ↓
                                              Cycle Controller (code) ──→ loop, reassess, or finalize
```

- **Strategy Agent** (LLM) chooses a strategy, signals, up to 8 symbols from a
  45-symbol universe, universe filter criteria, cadence, and entry/exit logic
  — reassessed when the current one keeps failing, never trades directly.
- **Schema Validator** (deterministic) rejects a proposal that invents an
  unsupported strategy, signal, symbol, or filter key, or exceeds the symbol
  cap — bounded retries, not an infinite loop.
- **Scanner** (deterministic) filters the strategy's symbols against its
  criteria via live Alpaca market data, concurrently, with no LLM calls.
- **Research** (code + LLM) computes intraday technical indicators on real
  bars — each one on *two* time windows (e.g. a 3-bar and a 12-bar momentum
  reading), because a strategy's own stated entry logic ("the last 30
  minutes") doesn't always match a single fixed window a naive implementation
  would pick. News sentiment is LLM-summarized and cached.
- **Trade Decision** (LLM) outputs BUY/SELL/HOLD + confidence + a bounded
  quantity per candidate, given market evidence, past outcomes for that
  symbol, current performance, and a computed `max_buy_qty` so it isn't
  guessing at position sizing blind.
- **Risk Gate** (deterministic) enforces the fixed constraint set against a
  **shared per-cycle budget** — so five individually-legal trades can't
  collectively blow past the session's exposure limit. Confidence floor, trade
  size, position size, gross exposure, daily loss limit, market hours, a
  close-buffer window, no shorting. Every check is a plain function with a
  unit test, not a prompt.
- **Execution** (deterministic) places orders and polls for a terminal fill
  status before reporting anything as done.
- **Monitor** (deterministic) reconciles Alpaca's actual positions against
  what the cycle's fills should have produced, and logs any divergence.
- **Performance Tracker** (deterministic) accumulates realized/unrealized P&L
  as running totals, not a recomputation from a truncated recent-trades list.
- **Cycle Controller** (deterministic) sleeps for the strategy's own cadence,
  clamped so it can never sleep past the close buffer; triggers a strategy
  reassessment on a losing streak or consecutive rejected cycles (never a
  fabricated target return); ends the session at the close.
- **Finalize** (deterministic) force-liquidates every open position and
  reports final P&L, regardless of what the last cycle's signals said.

## Engineering notes

A few things that were true only after they were checked against the *real*
Alpaca MCP server, not assumed:

- **The MCP server's actual tool names, response shapes, and parameter names
  differ from the obvious guess** (`get_account_info` not `get_account`,
  `symbols` not `symbol`, every response wrapped in a `{"data": ...}`
  envelope). `check_mcp.py` verifies all of this live, including a
  **freshness check** on returned bars — the historical-bars endpoint
  defaults to `sort="asc"` with no explicit start time, which silently
  returns days-old data unless `sort="desc"` is passed explicitly. Confirmed
  live: this alone was quietly feeding the decision layer stale data for an
  entire session.
- **A real position-sizing bug was found from an actual rejected order in a
  live session**: a genuine bullish signal proposed a share count worth 5x
  the trade-size cap and lost the trade to an outright rejection. The fix
  computes `max_buy_qty` deterministically and hands it to the model, rather
  than asking it to do the arithmetic.
- **LLM calls are wrapped for failure, not just for parsing.**
  `with_structured_output(..., include_raw=True)` only catches a parsing
  failure — a raw provider error (a truncated generation the server rejects
  outright, a rate limit surviving the client's own retries) still raises
  past it. Every LLM call site catches this and degrades to a safe default
  (HOLD, a rejected strategy proposal, an "unavailable" narrative) instead of
  crashing a session that may have open positions.
- **Token budget is a real constraint, not an afterthought.** Groq's
  tokens-per-minute limit was hit in practice; prompts are trimmed to only
  the fields with decision value, `max_tokens` is capped per call site, and
  the Strategy Agent is hard-limited (schema-enforced, not just asked nicely)
  to 8 symbols, since each one costs an LLM call every cycle.
- **`check_mcp.py`** verifies the live tool contract before trading.
  **`manual_test_trade.py`** is a one-off smoke test that buys and
  immediately sells 1 share through the exact same execution code the real
  graph uses — proof the fill/reconciliation pipeline works, independent of
  whether any strategy's entry criteria happen to fire that day.

## Setup

```bash
source myvenv/bin/activate
pip install -r requirements.txt
```

Set `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER_TRADE=true`,
`ALPACA_TOOLSETS` (must include `assets`, for the market clock), and
`GROQ_API_KEY` in `.env`.

`ALPACA_PAPER_TRADE` must stay `true` — `src/config.py` refuses to start
otherwise. This project only ever trades on paper.

Before the first session, verify the live server matches what the code
expects:

```bash
python check_mcp.py
```

## Run

```bash
python run.py
```

Only runs while the market is open. You'll be prompted for session capital
(blank = full account equity) and an optional universe hint (e.g. "liquid
high-beta tech and semiconductors") that steers which symbols the Strategy
Agent picks. It then trades intraday on its own cadence until the close
buffer, force-liquidates everything, and reports final P&L.

Every decision, strategy change, rejection, order, fill, and reconciliation
divergence is journaled to `journal/<run_id>.jsonl` — one JSON object per
line, so a running session can be watched live with:

```bash
tail -f journal/run_<id>.jsonl | jq -c '{event: .event_type, payload}'
```

For a quick, low-stakes proof the execution pipeline works without waiting on
a strategy to find a real setup:

```bash
python manual_test_trade.py [SYMBOL]
```

## Tests

```bash
pytest
```

100+ tests: the deterministic nodes (risk gate, schema validator, P&L,
restrategize policy, technical indicators) as pure-logic unit tests, plus
end-to-end graph cycles with the MCP server and both LLM call sites stubbed —
covering the flatten-at-close guarantee, bounded strategy retries, per-cycle
risk budgeting, and the LLM-failure fallbacks. `check_mcp.py` covers the live
server contract, which unit tests deliberately can't reach.

## Known gaps

- **No Pattern Day Trader (PDT) check.** Accounts under $25k are limited to 3
  day trades per rolling 5 business days on Alpaca; the Risk Gate doesn't
  enforce this yet, so Alpaca itself may reject an order the gate approved.
- **No crash-resilient checkpointing.** The graph runs in-process with no
  checkpointer, so a process death loses graph state (the journal keeps the
  audit trail, and the emergency handler still flattens). A `SqliteSaver`
  would make sessions resumable.
- **The decision-skip filter is conservative** — it only skips a symbol when
  *no* requested indicator computed at all, not when the computed values are
  clearly nowhere near the strategy's entry bar. A real prefilter would cut
  LLM calls further but needs per-strategy thresholds that don't exist yet.
- **Partial fills aren't re-reconciled within the same cycle** — realized
  P&L uses whatever `filled_avg_price` Alpaca reports at the time it's read.
