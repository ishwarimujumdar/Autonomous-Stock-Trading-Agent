from src.config import INTRADAY_TIMEFRAME, MAX_UNIVERSE_SYMBOLS, TRADABLE_UNIVERSE
from src.llm import get_llm
from src.strategy.schema import StrategyProposal

_SYSTEM_PROMPT = """You are the Strategy Agent for a day-trading system.

This is intraday day trading: every position opened must be closed before
today's market close - there is no holding overnight and no multi-day
horizon. You are choosing a strategy for THIS session only.

All signals are computed on INTRADAY bars ({timeframe}), not daily bars, so
your entry/exit logic should be phrased in intraday terms (e.g. "5-minute
momentum", "intraday VWAP-style mean reversion"), never in terms of multi-day
moves.

You choose HOW to trade, not whether risk limits apply - those are fixed and
non-negotiable, enforced outside your control:
- minimum_confidence, max_trade_size_pct, max_position_pct, max_daily_loss_pct,
  max_gross_exposure_pct, market_close_buffer_minutes are hard constraints you
  cannot loosen. You may only be MORE conservative than them, never less.
- You may only choose from the supported strategies and signals listed below.
- universe_symbols must be a subset of the tradable universe listed below,
  and at most {max_symbols} symbols - each one runs an LLM call every cycle
  against a tokens-per-minute-limited API, so more symbols is not better.
  Pick the {max_symbols} (or fewer) that best fit the user's universe hint
  and your strategy.
- universe_criteria may only use these keys - any other key is rejected:
  {criteria}
- Prefer a cadence_minutes suited to intraday re-evaluation (e.g. 5-30 min),
  not a daily/weekly cadence.

Each signal only ever computes these exact fields - phrase entry/exit logic
using THESE windows, not an arbitrary one (e.g. say "30-minute volume", not
"15-minute volume", which nothing computes and will just be misread against
the nearest available window):
- price_momentum: return_last_3_bars_pct (~15 min), return_last_12_bars_pct
  (~60 min), return_session_pct
- volume: volume_ratio_6bar (latest bar vs prior 6 bars, ~30 min),
  volume_ratio_20bar (vs prior 20 bars, ~100 min)
- moving_average_crossover: sma_10, sma_30, sma_spread_pct
- rsi: rsi_14 (Wilder, 14-period)
- macd: macd, macd_signal, macd_histogram (12/26/9 EMA)
- news_sentiment: a short qualitative narrative, no numeric field

Given the objective, constraints, run configuration, and (on reassessment)
recent performance so far today, propose a strategy: which approach to use,
which signals to watch, which symbols and filter criteria the scanner should
apply, how often to re-evaluate (cadence_minutes), and your entry/exit logic
in plain terms. Check the strategy history below before proposing - avoid
repeating a strategy that was already rejected for a schema error, and avoid
re-proposing a strategy that already underperformed this session unless
you're changing it meaningfully. entry_logic, exit_logic and rationale should
each be one plain sentence - this is a strategy spec, not an essay.

Supported strategies: {strategies}
Supported signals: {signals}
Tradable universe: {universe}
"""


def _format_strategy_history(strategy_history: list[dict]) -> str:
    if not strategy_history:
        return "none yet - this is the first proposal this session."

    lines = []
    for entry in strategy_history:
        strategy = entry.get("strategy") or {}
        name = strategy.get("name", "unknown")
        signals = ", ".join(strategy.get("signals") or [])
        if entry.get("strategy_valid") is False:
            error = entry.get("strategy_validation_error", "unknown error")
            lines.append(f"- REJECTED: '{name}' (signals: {signals}) - error: {error}")
        else:
            perf = entry.get("performance_at_switch") or {}
            reason = entry.get("reason", "no reason recorded")
            lines.append(
                f"- USED: '{name}' (signals: {signals}) - replaced because: {reason}. "
                f"Performance when replaced: return {perf.get('return_pct')}%, "
                f"wins/losses {perf.get('wins')}/{perf.get('losses')}."
            )
    return "\n".join(lines)


async def propose_strategy(
    objective_constraints: dict,
    run_config: dict,
    strategy_history: list[dict] | None = None,
) -> dict:
    """
    Calls the LLM to produce a raw strategy proposal (unvalidated dict).

    Returns the raw proposal even when the model produced something the schema
    would reject: enforcement is the schema-validator node's job, and it needs
    the bad proposal in hand to route a retry. `include_raw=True` is what makes
    that possible - it catches a validation failure (from a bad enum value or
    a failed field_validator) into `parsing_error` instead of raising it, so a
    bad proposal comes back here as data rather than killing the run.

    Current performance isn't passed separately - on a reassessment,
    strategy_history's most recent "USED" entry already carries the
    performance snapshot at the moment the prior strategy was replaced,
    which is the same figure current performance would show at this point.
    """
    # max_tokens=1200: 600 was too tight - Groq truncated the JSON mid-field
    # and rejected the whole request server-side (a 400, not a parse error
    # include_raw could catch). This call is infrequent (once per session,
    # occasionally on reassessment), so there's little to save by cutting it
    # closer than this.
    llm = get_llm(temperature=0.3, max_tokens=1200).with_structured_output(
        StrategyProposal, include_raw=True
    )

    constraints = objective_constraints["constraints"]
    prompt = _SYSTEM_PROMPT.format(
        strategies=constraints["supported_strategies"],
        signals=constraints["supported_signals"],
        criteria=constraints["supported_universe_criteria"],
        universe=TRADABLE_UNIVERSE,
        timeframe=INTRADAY_TIMEFRAME,
        max_symbols=MAX_UNIVERSE_SYMBOLS,
    )

    # Just the numeric limits here - supported_strategies/signals/criteria are
    # already spelled out in the system prompt above, and session_capital is
    # already in the line below; repeating the whole constraints dict duplicated
    # both.
    numeric_limits = {
        k: v
        for k, v in constraints.items()
        if k
        in (
            "max_trade_size_pct",
            "max_position_pct",
            "max_daily_loss_pct",
            "max_gross_exposure_pct",
            "minimum_confidence",
            "market_close_buffer_minutes",
        )
    }
    context_lines = [
        f"Objective: {objective_constraints['objective']}",
        f"Capital allocated to this session: {objective_constraints['session_capital']}",
        f"Universe hint from user: {run_config.get('universe_hint') or 'none given'}",
        f"Fixed numeric limits (cannot be loosened): {numeric_limits}",
        f"Strategy history this session:\n{_format_strategy_history(strategy_history or [])}",
    ]

    try:
        result = await llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "\n".join(context_lines)},
            ]
        )
    except Exception as exc:
        # include_raw=True only catches a PARSING failure into parsing_error -
        # it can't catch the provider itself rejecting the request (e.g. Groq
        # returning a 400 because generation got cut off mid-JSON, or a rate
        # limit surviving the client's own retries). Either way this must not
        # crash the whole session: hand the validator something it will
        # cleanly reject, so it becomes a routed retry like any other bad
        # proposal, bounded by MAX_STRATEGY_ATTEMPTS.
        return {"_unparseable": f"LLM call failed: {type(exc).__name__}: {exc}"}

    parsed = result.get("parsed")
    if parsed is not None:
        return parsed.model_dump()

    # Parsing failed - hand the validator whatever the model actually emitted so
    # it can produce a precise rejection reason and trigger a retry.
    return _raw_arguments(result)


def _raw_arguments(result: dict) -> dict:
    """Best-effort extraction of the model's raw tool-call arguments."""
    raw = result.get("raw")
    tool_calls = getattr(raw, "tool_calls", None) or []
    if tool_calls:
        args = tool_calls[0].get("args")
        if isinstance(args, dict):
            return args
    error = result.get("parsing_error")
    return {"_unparseable": str(error) if error else "model returned no structured output"}
