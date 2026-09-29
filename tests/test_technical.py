from src.analysis.technical import compute_indicators, has_usable_evidence


def test_price_change_is_percent_over_each_window():
    closes = [100.0] * 10 + [101.0, 102.0, 103.0]  # 13 bars, last three rising
    out = compute_indicators(closes, [1000] * 13)
    assert out["return_last_3_bars_pct"] == 3.0  # 100 -> 103
    assert out["return_last_12_bars_pct"] == 3.0


def test_price_change_is_none_until_there_are_enough_bars():
    out = compute_indicators([100.0, 101.0, 102.0], [1000] * 3)
    assert out["return_last_3_bars_pct"] is None  # needs 4 bars
    assert out["return_last_12_bars_pct"] is None


def test_volume_ratio_compares_latest_bar_with_the_bars_before_it():
    out = compute_indicators([100.0] * 7, [1000] * 6 + [2000])
    assert out["volume_ratio_6bar"] == 2.0  # twice the usual


def test_volume_spike_does_not_inflate_its_own_baseline():
    """If the latest bar were in its own average, a 10x spike would read as ~3.3x."""
    out = compute_indicators([100.0] * 7, [1000] * 6 + [10000])
    assert out["volume_ratio_6bar"] == 10.0


def test_volume_ratio_is_missing_without_enough_history():
    out = compute_indicators([100.0] * 3, [1000, 1000, 1000])
    assert "volume_ratio_6bar" not in out


def test_has_usable_evidence_false_when_nothing_computed():
    technical = {"symbol": "AAPL", "bars_available": 3, "return_last_3_bars_pct": None}
    assert not has_usable_evidence(technical)


def test_has_usable_evidence_true_when_something_computed():
    technical = {"symbol": "AAPL", "bars_available": 60, "return_last_3_bars_pct": 0.4}
    assert has_usable_evidence(technical)


def test_has_usable_evidence_false_after_a_fetch_error():
    assert not has_usable_evidence({"symbol": "AAPL", "error": "boom"})
