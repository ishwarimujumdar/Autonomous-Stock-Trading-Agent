import os

from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_PAPER_TRADE = os.environ.get("ALPACA_PAPER_TRADE", "true")
# "assets" is required for get_clock - without it the agent can't tell
# whether the market is open or how close it is to the session close.
ALPACA_TOOLSETS = os.environ.get("ALPACA_TOOLSETS", "account,trading,assets,stock-data,news")

# This project only ever trades on paper. The README and .env both say so;
# this makes it an enforced invariant rather than a convention, because the
# cost of the flag silently flipping is real money.
if ALPACA_PAPER_TRADE.strip().lower() != "true":
    raise RuntimeError(
        f"ALPACA_PAPER_TRADE must be 'true' (got {ALPACA_PAPER_TRADE!r}). "
        "This agent is paper-trading only."
    )

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

JOURNAL_DIR = os.environ.get("JOURNAL_DIR", "journal")

# Tuning knobs below are plain constants, not env vars: nothing here needs to
# change without touching the code anyway, so an .env indirection would just
# be one more place to look. Edit these directly if you want to change them.

# Bars the technical layer reads. This is a DAY-trading agent: indicators are
# computed on intraday bars so they actually move between cycles. Daily bars
# would be byte-identical from one 5-minute cycle to the next.
INTRADAY_TIMEFRAME = "5Min"
INTRADAY_BAR_LIMIT = 200

# Max concurrent in-flight MCP/LLM calls when fanning out over symbols. This
# only paces how many calls are in flight at once - Groq's limit is tokens
# PER MINUTE, and Groq responds in well under a second, so this alone doesn't
# stop a cycle's calls from landing in the same 60-second window. Raising it
# makes bursts finish faster, which is the wrong direction for a TPM limit.
# The real cap on tokens-per-cycle is MAX_UNIVERSE_SYMBOLS below.
MAX_CONCURRENCY = 2

# Hard cap on how many symbols the Strategy Agent may put in universe_symbols
# (enforced in src/strategy/schema.py, not just suggested in its prompt).
# Each one costs a decide() call every cycle - at ~800 tokens/call after
# prompt trimming, 8 symbols is ~6,500 tokens/cycle against Groq's 8,000 TPM
# limit; the previous "8-15 is fine" guidance let it pick enough symbols to
# blow that budget in a single cycle (confirmed: 8 symbols + untrimmed
# prompts hit 7,920/8,000 in one burst).
MAX_UNIVERSE_SYMBOLS = 8

# How long a symbol's news narrative stays usable before we re-fetch and
# re-summarise it. Headlines do not turn over every cycle.
NEWS_CACHE_TTL_SECONDS = 900

# Give up rather than loop forever if the Strategy Agent cannot produce a
# schema-valid proposal.
MAX_STRATEGY_ATTEMPTS = 3

# Supported strategy/signal vocabulary the Strategy Agent is allowed to choose from.
# The schema validator rejects any Strategy Agent output that references anything
# outside these lists - this is the enforcement boundary, not a suggestion.
SUPPORTED_STRATEGIES = [
    "momentum",
    "momentum_news",
    "mean_reversion",
    "breakout",
    "trend_following",
]

SUPPORTED_SIGNALS = [
    "price_momentum",
    "volume",
    "volatility",
    "moving_average_crossover",
    "rsi",
    "macd",
    "news_sentiment",
]

# Universe-filter keys the Scanner actually implements. Anything outside this
# set used to be silently dropped, so the Strategy Agent could believe it had
# set a filter that never ran. Same enforcement boundary as strategies/signals.
SUPPORTED_UNIVERSE_CRITERIA = [
    "min_price",
    "max_price",
    "min_avg_daily_volume",
    "min_intraday_return_pct",
    "max_intraday_return_pct",
]

# The symbols the Strategy Agent may select from. It picks a subset (informed
# by the user's universe hint); the validator rejects anything off this list,
# so the LLM cannot route the agent into an illiquid or untradeable name.
TRADABLE_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "AVGO", "NFLX", "JPM", "V", "COST", "PEP", "LIN", "ADBE", "CRM",
    "INTC", "QCOM", "TXN", "MU", "AMAT", "ORCL", "CSCO", "BAC", "WFC",
    "GS", "MS", "XOM", "CVX", "UNH", "JNJ", "PFE", "LLY", "HD", "WMT",
    "DIS", "NKE", "MCD", "SBUX", "UBER", "ABNB", "PLTR", "COIN", "SHOP",
]


def load_objective_and_constraints(
    session_capital: float,
    max_trade_size_pct: float = 0.20,
    max_daily_loss_pct: float = 0.05,
    max_position_pct: float = 0.30,
    minimum_confidence: float = 0.70,
    market_close_buffer_minutes: int = 15,
    max_gross_exposure_pct: float = 1.00,
) -> dict:
    """
    Fixed, non-LLM-editable objective + hard constraints for a single
    day-trading session (start now, flatten and stop at market close - no
    multi-day horizon).

    All percentage limits are measured against `session_capital` - the capital
    the user allocated to this session - not against total account equity.
    Sizing and reporting therefore share one basis.

    The Strategy Agent reads this as context (it can be MORE conservative,
    never less). The Risk Gate enforces it against every proposed trade.
    """
    return {
        "objective": "maximize trading return within today's session",
        "session_capital": session_capital,
        "constraints": {
            "session_capital": session_capital,
            "max_trade_size_pct": max_trade_size_pct,
            "max_position_pct": max_position_pct,
            "max_daily_loss_pct": max_daily_loss_pct,
            # Ceiling on the combined value of everything held at once, so a
            # cycle cannot approve many individually-legal trades that add up
            # to far more than the session's capital.
            "max_gross_exposure_pct": max_gross_exposure_pct,
            "minimum_confidence": minimum_confidence,
            "supported_strategies": SUPPORTED_STRATEGIES,
            "supported_signals": SUPPORTED_SIGNALS,
            "supported_universe_criteria": SUPPORTED_UNIVERSE_CRITERIA,
            "market_hours_only": True,
            # No new BUYs once we're this close to the close; existing
            # positions get force-flattened inside this window too.
            "market_close_buffer_minutes": market_close_buffer_minutes,
        },
    }
