from datetime import date
from types import SimpleNamespace

import polars as pl

from app.services import strategy_win_rate as win_rate_module
from app.services.strategy_win_rate import StrategyWinRateService, _merge_board_filter, board_for_symbol


def _prices() -> pl.DataFrame:
    return pl.DataFrame({
        "symbol": ["600001.SH"] * 3,
        "date": [date(2026, 7, 30), date(2026, 7, 31), date(2026, 8, 3)],
        "open": [9.0, 10.0, 12.0],
        "close": [9.5, 11.0, 13.0],
    })


class _Repo:
    def __init__(self, tmp_path):
        self.store = SimpleNamespace(data_dir=tmp_path)

    def get_enriched_range(self, start, end, columns=None):
        return _prices()


class _Engine:
    def __init__(self):
        self.strategy = SimpleNamespace(basic_filter={})

    def list_strategies(self):
        return [{
            "id": "s1",
            "name": "测试策略",
            "asset_types": ["stock"],
            "timeframes": ["1d"],
        }]

    def get(self, strategy_id):
        assert strategy_id == "s1"
        return self.strategy

    def run_all(self, context, **kwargs):
        return {"s1": SimpleNamespace(rows=[{"symbol": "600001.SH", "name": "测试股"}])}


class _Screener:
    def __init__(self, repo, asset_type="stock"):
        assert asset_type == "stock"

    def latest_date(self):
        return date(2026, 8, 3)

    def build_strategy_context(self, *args, **kwargs):
        return object()


def test_calculate_maps_signal_to_next_two_trading_days_and_four_combinations(monkeypatch, tmp_path):
    monkeypatch.setattr(win_rate_module, "ScreenerService", _Screener)
    result = StrategyWinRateService(_Repo(tmp_path), _Engine()).calculate(
        ["s1"],
        date(2026, 7, 30),
        date(2026, 7, 30),
        ["沪主板"],
    )

    detail = result["details"][0]
    assert detail["signal_date"] == "2026-07-30"
    assert detail["buy_date"] == "2026-07-31"
    assert detail["sell_date"] == "2026-08-03"
    assert detail["returns"] == {
        "open_open": 0.2,
        "open_close": 0.3,
        "close_open": round(12 / 11 - 1, 6),
        "close_close": round(13 / 11 - 1, 6),
    }
    assert result["summary"]["combinations"]["open_open"]["win_rate"] == 1.0


def test_missing_open_only_skips_open_combinations(monkeypatch, tmp_path):
    monkeypatch.setattr(win_rate_module, "ScreenerService", _Screener)
    prices = _prices().with_columns(
        pl.when(pl.col("date") == date(2026, 7, 31))
        .then(None)
        .otherwise(pl.col("open"))
        .alias("open")
    )
    monkeypatch.setattr(_Repo, "get_enriched_range", lambda self, *args, **kwargs: prices)

    result = StrategyWinRateService(_Repo(tmp_path), _Engine()).calculate(
        ["s1"], date(2026, 7, 30), date(2026, 7, 30), ["沪主板"],
    )
    stats = result["summary"]["combinations"]
    assert stats["open_open"]["valid"] == 0
    assert stats["open_close"]["skipped"] == 1
    assert stats["close_open"]["valid"] == 1
    assert stats["close_close"]["valid"] == 1


def test_uses_each_symbol_trading_days_when_symbol_is_suspended(monkeypatch, tmp_path):
    monkeypatch.setattr(win_rate_module, "ScreenerService", _Screener)
    prices = pl.DataFrame({
        "symbol": ["600001.SH", "600002.SH", "600001.SH", "600001.SH"],
        "date": [
            date(2026, 7, 30),
            date(2026, 7, 31),
            date(2026, 8, 3),
            date(2026, 8, 4),
        ],
        "open": [9.0, 20.0, 12.0, 14.0],
        "close": [9.5, 21.0, 13.0, 15.0],
    })
    monkeypatch.setattr(_Repo, "get_enriched_range", lambda self, *args, **kwargs: prices)

    result = StrategyWinRateService(_Repo(tmp_path), _Engine()).calculate(
        ["s1"], date(2026, 7, 30), date(2026, 7, 30), ["沪主板"],
    )

    detail = result["details"][0]
    assert detail["buy_date"] == "2026-08-03"
    assert detail["sell_date"] == "2026-08-04"
    assert detail["returns"]["open_open"] == round(14 / 12 - 1, 6)


def test_tail_signal_is_reported_as_skipped_without_entering_denominator(monkeypatch, tmp_path):
    monkeypatch.setattr(win_rate_module, "ScreenerService", _Screener)
    prices = pl.DataFrame({
        "symbol": ["600001.SH"],
        "date": [date(2026, 8, 3)],
        "open": [12.0],
        "close": [13.0],
    })
    monkeypatch.setattr(_Repo, "get_enriched_range", lambda self, *args, **kwargs: prices)

    result = StrategyWinRateService(_Repo(tmp_path), _Engine()).calculate(
        ["s1"], date(2026, 8, 3), date(2026, 8, 3), ["沪主板"],
    )

    stats = result["summary"]["combinations"]["open_open"]
    assert result["summary"]["signal_count"] == 1
    assert stats["valid"] == 0
    assert stats["skipped"] == 1
    assert stats["skip_reasons"] == {"缺少后续交易日": 1}


def test_board_mapping_includes_689_and_beijing():
    assert board_for_symbol("689001.SH") == "科创板"
    assert board_for_symbol("830001.BJ") == "北交所"


def test_market_selection_is_applied_when_strategy_basic_filter_is_disabled():
    strategy = SimpleNamespace(basic_filter={"enabled": False})
    merged = _merge_board_filter(strategy, {}, ["科创板"])
    assert merged["basic_filter"] == {"enabled": True, "boards": ["科创板"]}


def test_summary_counts_flat_and_duplicate_strategy_signals():
    details = [
        {
            "strategy_id": "s1",
            "strategy_name": "策略一",
            "returns": {key: 0.01 for key in ("open_open", "open_close", "close_open", "close_close")},
        },
        {
            "strategy_id": "s2",
            "strategy_name": "策略二",
            "returns": {key: 0.0 for key in ("open_open", "open_close", "close_open", "close_close")},
        },
    ]
    result = StrategyWinRateService._build_response(
        details,
        ["s1", "s2"],
        {"s1": "策略一", "s2": "策略二"},
        date(2026, 7, 1),
        date(2026, 7, 31),
        ["沪主板"],
    )
    stats = result["summary"]["combinations"]["open_open"]
    assert stats["valid"] == 2
    assert stats["wins"] == 1
    assert stats["flats"] == 1
    assert stats["win_rate"] == 0.5
