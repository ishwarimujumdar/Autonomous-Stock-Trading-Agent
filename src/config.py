import os

from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_PAPER_TRADE = os.environ.get("ALPACA_PAPER_TRADE", "true")
# "assets" is required for get_clock: without it the agent can't tell how close the market close is.
ALPACA_TOOLSETS = os.environ.get("ALPACA_TOOLSETS", "account,trading,assets,stock-data")

# Paper trading only. Enforced here rather than left as a convention, because
# the cost of this flag silently flipping is real money.
if ALPACA_PAPER_TRADE.strip().lower() != "true":
    raise RuntimeError(
        f"ALPACA_PAPER_TRADE must be 'true' (got {ALPACA_PAPER_TRADE!r}). "
        "This agent is paper-trading only."
    )

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

JOURNAL_DIR = os.environ.get("JOURNAL_DIR", "journal")

# Tuning knobs are plain constants: edit them here.

# --- Timing -----------------------------------------------------------------
CYCLE_MINUTES = 5  # wait between decision cycles
REPICK_MINUTES = 60  # how often the AI re-chooses which stocks to watch
INTRADAY_TIMEFRAME = "5Min"  # indicators use 5-minute bars, so they change every cycle
INTRADAY_BAR_LIMIT = 200

# --- AI stock picking -------------------------------------------------------
# Max stocks on the watchlist. Each costs one decision call per cycle; 5 stocks
# is ~4,000 tokens against Groq's 8,000 tokens/minute limit.
MAX_WATCHLIST = 5
MAX_PICK_ATTEMPTS = 3  # tries before giving up on a valid pick

# Max simultaneous MCP/LLM calls (Groq limits tokens per minute).
MAX_CONCURRENCY = 2

# --- The fixed trading rule -------------------------------------------------
# The AI chooses WHICH stocks to watch; this rule defines what counts as a
# chance. When the AI was left to choose these numbers it picked ~0.5% / 1.5x
# every time. On a week of 5-minute bars that pair held on 2.4% of checks and
# produced zero trades in 6 runs; 0.2% / 1.1x holds on roughly 3-12% of checks
# depending on the stock. That shows the rule CAN fire, not that the trades pay.
ENTRY_MIN_MOVE_PCT = 0.2  # price up at least this % over the last 3 bars (~15 min)
ENTRY_MIN_VOLUME_RATIO = 1.1  # latest bar at least this many times busier than the prior 6
ENTRY_RULE = (
    f"BUY when return_last_3_bars_pct >= +{ENTRY_MIN_MOVE_PCT}% and "
    f"volume_ratio_6bar >= {ENTRY_MIN_VOLUME_RATIO}."
)
EXIT_RULE = (
    "SELL a held stock when return_last_12_bars_pct turns negative, or when its "
    "price is 0.5% below your entry price."
)

# The only stocks the LLM may pick from (liquid, tradeable names).
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
    The fixed limits for one day-trading session. No LLM can change them; the
    Risk Gate enforces them. Percentages are of `session_capital`, not the
    whole account.
    """
    return {
        "objective": "maximize trading return within today's session",
        "session_capital": session_capital,
        "constraints": {
            "session_capital": session_capital,
            "max_trade_size_pct": max_trade_size_pct,
            "max_position_pct": max_position_pct,
            "max_daily_loss_pct": max_daily_loss_pct,
            # Cap on the total value held at once.
            "max_gross_exposure_pct": max_gross_exposure_pct,
            "minimum_confidence": minimum_confidence,
            "market_hours_only": True,
            # No new BUYs this close to the close; positions are sold off in this window.
            "market_close_buffer_minutes": market_close_buffer_minutes,
        },
    }
