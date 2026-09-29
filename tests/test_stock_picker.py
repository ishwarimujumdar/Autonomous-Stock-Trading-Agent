import pytest

from src.config import MAX_WATCHLIST, TRADABLE_UNIVERSE
from src.picker import universe
from src.picker.stock_picker import Watchlist, _user_message


def _valid(**overrides):
    base = {"symbols": ["AAPL", "MSFT"], "reason": "liquid and active"}
    base.update(overrides)
    return base


def test_accepts_a_valid_watchlist():
    assert Watchlist(**_valid()).symbols == ["AAPL", "MSFT"]


def test_rejects_a_stock_outside_the_tradable_universe():
    with pytest.raises(ValueError, match="GME"):
        Watchlist(**_valid(symbols=["AAPL", "GME"]))


def test_rejects_an_empty_watchlist():
    with pytest.raises(ValueError):
        Watchlist(**_valid(symbols=[]))


def test_rejects_more_than_the_maximum_number_of_stocks():
    """Each stock costs one AI call per cycle against a tokens-per-minute limit."""
    with pytest.raises(ValueError):
        Watchlist(**_valid(symbols=TRADABLE_UNIVERSE[: MAX_WATCHLIST + 1]))


def test_rejects_duplicates():
    with pytest.raises(ValueError, match="distinct"):
        Watchlist(**_valid(symbols=["AAPL", "AAPL"]))


def test_schema_carries_the_universe_as_a_real_enum():
    """So constrained decoding can steer on it, not just prose in a description."""
    items = Watchlist.model_json_schema()["properties"]["symbols"]["items"]
    assert items["enum"] == TRADABLE_UNIVERSE


def test_prompt_message_carries_the_hint_the_table_and_any_earlier_error():
    table = [
        {"symbol": "AAPL", "price": 100.0, "change_pct": -1.234, "range_pct": 2.5, "yesterday_volume": 5000}
    ]
    message = _user_message(table, "volatile tech", "symbols: bad")
    assert "volatile tech" in message
    assert "AAPL 100.0 -1.23% 2.50% 5000" in message
    assert "symbols: bad" in message
    assert "rejected" not in _user_message(table, "", None)


@pytest.mark.asyncio
async def test_universe_table_builds_rows_and_drops_stocks_without_data(monkeypatch):
    async def fake_call_tool(client, name, arguments):
        symbol = arguments["symbols"]
        if symbol == "AAPL":
            return {"AAPL": {
                "latestTrade": {"p": 102.0},
                "prevDailyBar": {"c": 100.0, "v": 7000},
                "dailyBar": {"l": 100.0, "h": 103.0},
            }}
        return {symbol: {}}  # no price, no previous bar

    monkeypatch.setattr(universe, "call_tool", fake_call_tool)

    table = await universe.universe_table(None)

    assert table == [
        {"symbol": "AAPL", "price": 102.0, "change_pct": 2.0, "range_pct": 3.0, "yesterday_volume": 7000}
    ]
