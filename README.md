# Stock Trading Agent

An autonomous intraday trading agent that selects a stock watchlist, generates BUY/SELL/HOLD decisions, and executes paper trades on US equities through the Alpaca MCP Server.

The system separates LLM decision-making from deterministic risk and execution: the LLM proposes trades, while code enforces position sizing, exposure, daily-loss, market-hours, and no-shorting constraints.

Runs as a single intraday session, reassessing trades every 5 minutes, refreshing the watchlist hourly, and closing all holdings before market close. Stocks only — no options, margin, shorting, or overnight holdings.

## Architecture

A 7-node LangGraph workflow. Only Pick Stocks and Trading Agent invoke the LLM; market data, risk validation, execution, reconciliation, and session control are deterministic.

```
                                    START
                                      |
                                      v
                        +----------------------------+
  +-------------------->|  Pick Stocks (code + LLM)  |-- no valid pick -------+
  |                     +----------------------------+                        |
  |                                   |                                       |
  |                                   v                                       |
  |                     +----------------------------+                        |
  |                     |     Market Data (code)     |<-----+                 |
  |                     +----------------------------+      |                 |
  |                                   | once per stock      |                 |
  ^                                   v                     |                 |
  |                     +----------------------------+      |                 |
  |                     |    Trading Agent (LLM)     |      ^                 |
  |                     +----------------------------+      |                 |
  |                                   |                     |                 |
  |                                   v                     |                 |
  |                     +----------------------------+      | every 5         |
  |                     |      Risk Gate (code)      |      | minutes         |
  ^                     +----------------------------+      |                 |
  |                                   |                     |                 |
  |                                   v                     ^                 |
  |                     +----------------------------+      |                 |
  |                     |       Execute (code)       |      |                 |
  |                     +----------------------------+      |                 |
  |                                   |                     |                 |
  |                                   v                     |                 |
  |                     +----------------------------+      |                 |
  +<- every hour ----   |     Log + Wait (code)      |------+                 |
                        +----------------------------+                        |
                                      | near the close                        |
                                      |                                       |
                                      v                                       |
                        +----------------------------+                        |
                        |        Close (code)        |<-----------------------+
                        +----------------------------+
```

- **Pick Stocks** — selects 1–5 symbols from a fixed 45-stock universe with
  schema-validated LLM output and retry/backoff handling.
- **Market Data** — refreshes account state and computes short-term return and
  volume metrics for watched and held stocks.
- **Trading Agent** — evaluates each candidate using live market data, previous outcomes, and a deterministic maximum buy quantity.
- **Risk Gate** — validates every proposed trade against position size, exposure, daily-loss, market-hours, confidence, and no-shorting limits using a shared per-cycle capital budget.
- **Execute** — places approved market orders, verifies fills, and reconciles expected holdings with Alpaca's actual account state.
- **Log + Wait** — records decisions and P&L, then triggers the next evaluation or watchlist refresh.
- **Close** — liquidates all remaining holdings before the session ends.

The risk gate enforces hard risk limits, but does not mechanically verify the LLM's interpretation of the entry/exit strategy. That distinction is intentional: strategy judgment remains with the model, while financial constraints remain deterministic.

## Layout

```
run.py                          entry point: prompts for capital, runs the graph
src/
├── config.py                   env vars, risk limits, fixed entry/exit rule
├── llm.py                      shared Groq client
├── concurrency.py              bounded-concurrency helper for LLM/MCP calls
├── alpaca/
│   ├── mcp_client.py           Alpaca MCP session + tool-call wrapper
│   └── account_and_orders.py   account state, order execution, fill polling,
│                                reconciliation, end-of-session liquidation
├── picker/
│   ├── universe.py             builds the tradable-universe snapshot table
│   └── stock_picker.py         LLM watchlist selection (schema-validated, retried)
├── analysis/
│   └── technical.py            intraday return/volume indicators
├── decision/
│   └── decision.py             LLM BUY/SELL/HOLD decision (schema-constrained)
├── risk/
│   └── risk_gate.py            deterministic risk gate, per-cycle capital budget
├── evaluation/
│   └── pnl.py                  running P&L / win-loss tracking
├── persistence/
│   └── journal.py              JSONL event logging
└── graph/
    ├── nodes.py                the 7 node implementations
    ├── build_graph.py          wires the nodes into the LangGraph pipeline
    └── state.py                shared TradingState schema
tests/                          mirrors src/, one test file per module
journal/                        per-run JSONL logs (run_<id>.jsonl)
```

## Results

One complete live paper-trading session on $5,000 session capital:

1. **48 cycles** over ~4 hours with **3 hourly watchlist refreshes** (4 picks total)
2. **16 proposed trades**, with **15 approved** by the risk gate
3. **7 closed positions:** 5 wins, 2 losses
4. **+$9.24 realized P&L (+0.18%)**
5. Session ended through the normal automated liquidation path with **no manual intervention**

The results are from a single paper-trading session and are not a profitability claim. The main purpose of the run was to validate autonomous decision-making, deterministic risk enforcement, order execution, and state reconciliation.

Detailed execution records are stored in `journal/run_99b069b3.jsonl`.

**Test suite:** 105 tests (pure-logic unit tests for the deterministic nodes,
plus end-to-end graph cycles with the MCP server and both LLM call sites
mocked) — `pytest`, ~1.5s.

## Tech stack

Python · LangGraph · Groq (LLM) · MCP (Alpaca MCP Server)· Alpaca API (market data + paper
execution) · pytest

## Setup

Needs Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) (the Alpaca MCP
server runs via `uvx alpaca-mcp-server`).

```bash
source myvenv/bin/activate
pip install -r requirements.txt
```

Set in `.env`: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`,
`ALPACA_PAPER_TRADE=true`, `ALPACA_TOOLSETS`, `GROQ_API_KEY`.
`ALPACA_PAPER_TRADE` must stay `true` — this project only ever trades on
paper, enforced in code.

## Run

```bash
python run.py
```

Only runs while the market is open. Prompts for session capital and an
optional watchlist hint, then runs on its own until the close buffer,
force-liquidates everything, and reports final P&L. Stop with `Ctrl+C`.