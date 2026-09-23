import pytest

from src.analysis.technical import _ema_series, _macd, _rsi, compute_indicators, has_usable_evidence

# Standard published RSI worked example. Full precision matters: rounding
# these to 2dp shifts the result by ~0.07.
WILDER_CLOSES = [
    44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245,
    45.8433, 46.0826, 45.8931, 46.0328, 45.6140, 46.2820, 46.2820, 46.0033,
    46.4116, 46.2222, 45.6439, 46.2122, 46.2521, 45.7137, 46.4515, 45.7835,
]


def test_rsi_matches_wilder_reference():
    """The old simple-average RSI diverged from every charting tool."""
    assert round(_rsi(WILDER_CLOSES[:15]), 2) == 70.53
    assert round(_rsi(WILDER_CLOSES[:16]), 2) == 66.33


def test_rsi_uses_wilder_smoothing_after_the_seed():
    """
    Wilder's RSI smooths with alpha = 1/window after seeding on the SMA. A
    plain rolling mean agrees at the seed and drifts immediately after, so the
    seed value alone doesn't pin the implementation down.
    """

    def wilder_reference(closes, window=14):
        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(c, 0) for c in changes]
        losses = [max(-c, 0) for c in changes]
        avg_gain = sum(gains[:window]) / window
        avg_loss = sum(losses[:window]) / window
        for i in range(window, len(changes)):
            avg_gain += (gains[i] - avg_gain) / window
            avg_loss += (losses[i] - avg_loss) / window
        return 100 - 100 / (1 + avg_gain / avg_loss)

    for n in range(15, len(WILDER_CLOSES) + 1):
        assert _rsi(WILDER_CLOSES[:n]) == pytest.approx(wilder_reference(WILDER_CLOSES[:n]))


def test_rsi_needs_enough_bars():
    assert _rsi([1.0, 2.0, 3.0]) is None


def test_rsi_flat_series_is_neutral_not_overbought():
    assert _rsi([100.0] * 30) == 50.0


def test_ema_seeds_from_sma_and_decays():
    series = _ema_series([1.0] * 10 + [2.0] * 10, 10)
    assert series[0] == 1.0
    assert 1.0 < series[-1] < 2.0


def test_macd_returns_signal_and_histogram():
    """The old version was an SMA difference with no signal line at all."""
    closes = [100 + i * 0.5 for i in range(60)]
    macd = _macd(closes)
    assert macd["macd"] is not None
    assert macd["macd_signal"] is not None
    assert round(macd["macd_histogram"], 6) == round(macd["macd"] - macd["macd_signal"], 6)


def test_macd_on_a_steady_uptrend_is_positive():
    macd = _macd([100 + i for i in range(60)])
    assert macd["macd"] > 0


def test_macd_insufficient_history_is_none_not_a_crash():
    assert _macd([100.0, 101.0])["macd"] is None


def test_indicators_are_scoped_to_requested_signals():
    closes = [100 + i * 0.1 for i in range(60)]
    volumes = [1000] * 60
    out = compute_indicators(closes, volumes, ["rsi"])
    assert "rsi_14" in out
    assert "macd" not in out
    assert "sma_10" not in out


def test_has_usable_evidence_false_when_everything_is_none():
    technical = {"symbol": "AAPL", "bars_available": 3, "rsi_14": None, "macd": None}
    assert not has_usable_evidence(technical, ["rsi", "macd"])


def test_has_usable_evidence_true_when_something_computed():
    technical = {"symbol": "AAPL", "bars_available": 60, "rsi_14": 55.0, "macd": None}
    assert has_usable_evidence(technical, ["rsi", "macd"])
