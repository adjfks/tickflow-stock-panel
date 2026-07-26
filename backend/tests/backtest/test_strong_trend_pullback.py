from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from app.backtest.engine import (
    BacktestEngine,
    MatcherConfig,
    _attach_pit_financial_fields,
    _attach_technical_data_quality,
    _with_listing_day,
)
from app.backtest.matrix import (
    build_market_data_matrix,
    build_market_matrix_from_signals,
    make_signal_matrix,
)
from app.backtest.strategy import StrategyBacktestConfig, StrategyBacktestService
from app.strategy.builtin import strong_trend_pullback_confirm as strategy_module
from app.strategy.builtin.strong_trend_pullback_confirm import MATRIX_STRATEGY, META


def _panel(
    closes: list[float],
    *,
    start: date = date(2024, 1, 1),
    patches: dict[int, dict] | None = None,
) -> pl.DataFrame:
    patches = patches or {}
    rows = []
    for offset, close in enumerate(closes):
        patch = patches.get(offset, {})
        rows.append({
            "symbol": "600000.SH",
            "name": "浦发银行",
            "date": start + timedelta(days=offset),
            "open": patch.get("open", close),
            "high": patch.get("high", close),
            "low": patch.get("low", close),
            "close": patch.get("close", close),
            "volume": patch.get("volume", 100_000.0),
            "signal_limit_up": patch.get("signal_limit_up", False),
            "signal_limit_down": patch.get("signal_limit_down", False),
            "macd_dif": patch.get("macd_dif", 1.0),
            "macd_dea": patch.get("macd_dea", 0.5),
            "macd_hist": patch.get("macd_hist", 0.5),
        })
    return pl.DataFrame(rows)


def _execution_matrix(
    closes: list[float],
    targets: dict[int, float],
    *,
    patches: dict[int, dict] | None = None,
    risk_ma5: np.ndarray | None = None,
    risk_ma10: np.ndarray | None = None,
    risk_ma20: np.ndarray | None = None,
):
    market = build_market_data_matrix(_panel(closes, patches=patches))
    entry = np.zeros(market.shape, dtype=np.uint8)
    entry[0, 0] = 1
    target = np.full(market.shape, np.nan, dtype=np.float32)
    exit_code = np.full(market.shape, -1, dtype=np.int16)
    for time_id, ratio in targets.items():
        target[time_id, 0] = ratio
        exit_code[time_id, 0] = 0
    signals = make_signal_matrix(
        market.shape,
        entry=entry,
        exit=np.isfinite(target).astype(np.uint8),
        target_position_ratio=target,
        exit_signal_code=exit_code,
        exit_signal_ids=("risk_reduce",),
        risk_ma5=risk_ma5,
        risk_ma10=risk_ma10,
        risk_ma20=risk_ma20,
    )
    return build_market_matrix_from_signals(market, signals)


def _matcher(**kwargs) -> MatcherConfig:
    return MatcherConfig(
        matching="close_t",
        entry_fill="close_t",
        exit_fill="close_t",
        fees_pct=0,
        commission_pct=0,
        stamp_tax_pct=0,
        slippage_bps=0,
        initial_capital=100_000,
        max_positions=1,
        **kwargs,
    )


def test_partial_exit_targets_are_idempotent_and_recorded_in_order():
    matrix = _execution_matrix(
        [10, 11, 12, 13, 14, 15],
        {1: 0.5, 2: 0.5, 3: 0.25, 4: 0.0},
    )

    result = BacktestEngine(repo=None).simulate_market_matrix(matrix, _matcher())

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert [leg["target_position_ratio"] for leg in trade.exit_legs] == [0.5, 0.25, 0.0]
    assert [leg["remaining_position_ratio"] for leg in trade.exit_legs] == [0.5, 0.25, 0.0]
    assert trade.exit_signal_id == "risk_reduce"


def test_non_default_partial_targets_are_respected():
    matrix = _execution_matrix([10, 11, 12, 13], {1: 0.6, 2: 0.2, 3: 0.0})

    result = BacktestEngine(repo=None).simulate_market_matrix(matrix, _matcher())

    assert [leg["target_position_ratio"] for leg in result.trades[0].exit_legs] == [0.6, 0.2, 0.0]


def test_limit_down_delays_partial_exit_and_preserves_original_signal():
    matrix = _execution_matrix(
        [10, 9, 9.5, 10],
        {1: 0.5},
        patches={1: {
            "open": 9,
            "high": 9,
            "low": 9,
            "close": 9,
            "signal_limit_down": True,
        }},
    )

    result = BacktestEngine(repo=None).simulate_market_matrix(matrix, _matcher())

    first_leg = result.trades[0].exit_legs[0]
    assert first_leg["date"] == "2024-01-03"
    assert first_leg["signal_date"] == "2024-01-02"
    assert first_leg["signal_id"] == "risk_reduce"
    assert result.stats["execution"]["sell_limit_down"] == 1


def test_dynamic_ma_stop_moves_up_but_never_back_down():
    shape = (5, 1)
    matrix = _execution_matrix(
        [10, 12, 11.5, 10.6, 10.7],
        {},
        patches={
            1: {"open": 10.5, "high": 12, "low": 10.4, "close": 12},
            2: {"open": 11.7, "high": 12, "low": 11.4, "close": 11.5},
            3: {"open": 11.0, "high": 11.1, "low": 10.5, "close": 10.6},
        },
        risk_ma5=np.full(shape, 9.0, dtype=np.float32),
        risk_ma10=np.asarray([[9.0], [11.0], [9.0], [9.0], [9.0]], dtype=np.float32),
        risk_ma20=np.full(shape, 9.0, dtype=np.float32),
    )

    result = BacktestEngine(repo=None).simulate_market_matrix(
        matrix,
        _matcher(
            dynamic_ma_stop=True,
            initial_ma20_buffer_pct=0.025,
            profit_activation_1_pct=0.10,
            trailing_ma10_buffer_pct=0.02,
        ),
    )

    trade = result.trades[0]
    assert trade.exit_reason == "dynamic_ma_stop"
    assert trade.exit_date == "2024-01-04"
    assert trade.exit_price == pytest.approx(10.78)


def test_financial_record_starts_after_announcement_day():
    market = build_market_data_matrix(
        _panel([10, 10], start=date(2024, 4, 15)),
    )
    financials = pl.DataFrame({
        "symbol": ["600000.SH"],
        "period_end": ["2023-12-31"],
        "announce_date": ["2024-04-15"],
        "revenue_yoy": [0.12],
        "net_income_yoy": [0.10],
        "roe": [0.09],
        "debt_to_asset_ratio": [0.55],
    })

    attached = _attach_pit_financial_fields(
        market,
        financials,
        {"revenue_yoy", "net_income_yoy", "roe", "debt_to_asset_ratio"},
        min_coverage=0.5,
    )

    assert np.isnan(attached.field("roe")[0, 0])
    assert attached.field("roe")[1, 0] == pytest.approx(0.09)
    assert attached.data_quality["financial_point_in_time"] is True


def test_financial_missing_announcement_and_low_coverage_are_rejected():
    market = build_market_data_matrix(_panel([10] * 10))
    base = {
        "symbol": ["600000.SH"],
        "period_end": ["2023-12-31"],
        "revenue_yoy": [0.12],
        "net_income_yoy": [0.10],
        "roe": [0.09],
        "debt_to_asset_ratio": [0.55],
    }
    with pytest.raises(ValueError, match="缺少 announce_date"):
        _attach_pit_financial_fields(
            market,
            pl.DataFrame({**base, "announce_date": [None]}),
            {"roe"},
        )
    with pytest.raises(ValueError, match="覆盖率不足"):
        _attach_pit_financial_fields(
            market,
            pl.DataFrame({**base, "announce_date": ["2024-01-09"]}),
            {"revenue_yoy", "net_income_yoy", "roe", "debt_to_asset_ratio"},
        )


@pytest.mark.parametrize("column", ["list_date", "listing_date"])
def test_listing_date_aliases_are_converted_to_epoch_days(column: str):
    instruments = pl.DataFrame({"symbol": ["600000.SH"], column: ["2020-01-02"]})

    converted = _with_listing_day(instruments)

    assert converted is not None
    assert converted["listing_day"][0] == (date(2020, 1, 2) - date(1970, 1, 1)).days


def test_technical_coverage_below_95_percent_blocks_backtest():
    market = build_market_data_matrix(
        _panel([10] * 20).with_columns(
            pl.when(pl.col("date").is_in([date(2024, 1, 1), date(2024, 1, 2)]))
            .then(None)
            .otherwise(pl.col("open"))
            .alias("open")
        )
    )

    with pytest.raises(ValueError, match="技术数据覆盖率不足 95%"):
        _attach_technical_data_quality(market)


def test_active_swing_defaults_preserve_explicit_strict_overrides():
    defaults = {item["id"]: item["default"] for item in META["params"]}
    active_swing = {
        "momentum_min": 0.08,
        "pullback_window": 5,
        "ma_pullback_distance": 0.03,
        "ma20_break_tolerance": 0.03,
        "shrink_volume_ratio": 0.90,
        "confirm_volume_ratio": 1.0,
        "use_financial_filter": False,
    }
    strict_overrides = {
        "momentum_min": 0.10,
        "pullback_window": 3,
        "ma_pullback_distance": 0.02,
        "ma20_break_tolerance": 0.02,
        "shrink_volume_ratio": 0.80,
        "confirm_volume_ratio": 1.2,
        "use_financial_filter": True,
    }

    assert {key: defaults[key] for key in active_swing} == active_swing

    strategy = type("Strategy", (), {"meta": META})()
    resolved = StrategyBacktestService._normalize_params(strict_overrides, strategy)
    assert {key: resolved[key] for key in strict_overrides} == strict_overrides

    financial_fields = {
        "revenue_yoy",
        "net_income_yoy",
        "roe",
        "debt_to_asset_ratio",
    }
    assert financial_fields.isdisjoint(MATRIX_STRATEGY.required_fields_for_params({}))
    assert financial_fields <= MATRIX_STRATEGY.required_fields_for_params({
        "use_financial_filter": True,
    })


def test_active_swing_defaults_accept_relaxed_confirmation_setup(monkeypatch):
    closes = [25.0] * 10
    closes[4] = 17.94  # MA20 下方 2.5%, 且发生在确认日前第 5 天。
    closes[8] = 20.8
    closes[9] = 21.0
    patches = {
        6: {"volume": 85_000.0},
        7: {"volume": 85_000.0},
        8: {"volume": 85_000.0, "macd_hist": -0.2},
        9: {"volume": 100_000.0, "macd_hist": -0.1},
    }
    listing_day = (date(2020, 1, 1) - date(1970, 1, 1)).days
    panel = _panel(closes, patches=patches).with_columns(
        pl.lit(float(listing_day)).alias("listing_day")
    )
    market = build_market_data_matrix(
        panel,
        field_columns={"macd_dif", "macd_dea", "macd_hist", "listing_day"},
    )

    steps = np.arange(10, dtype=np.float32)[:, None] * np.float32(0.1)
    features = {
        "ma5": 20.0 + steps,
        "ma10": 19.0 + steps,
        "ma20": 18.0 + steps,
        "ma30": 17.0 + steps,
        "ma60": 16.0 + steps,
        "momentum_20d": np.full(market.shape, 0.085, dtype=np.float32),
        "obv": np.arange(10, dtype=np.float32)[:, None],
    }
    monkeypatch.setattr(
        strategy_module,
        "matrix_feature",
        lambda _market, feature: features[feature],
    )

    active_signals = MATRIX_STRATEGY.compute_signals(market, {})
    strict_signals = MATRIX_STRATEGY.compute_signals(market, {
        "momentum_min": 0.10,
        "pullback_window": 3,
        "ma_pullback_distance": 0.02,
        "ma20_break_tolerance": 0.02,
        "shrink_volume_ratio": 0.80,
        "confirm_volume_ratio": 1.2,
        "use_financial_filter": False,
    })
    missing_financial_signals = MATRIX_STRATEGY.compute_signals(market, {
        "use_financial_filter": True,
    })

    assert active_signals.entry[9, 0] == 1
    assert strict_signals.entry[9, 0] == 0
    assert missing_financial_signals.entry[9, 0] == 0


def test_strategy_target_parameter_changes_risk_signal_and_defaults_are_stable():
    closes = [10 + index * 0.1 for index in range(70)] + [17.0, 15.0, 14.0, 13.5]
    panel = _panel(closes)
    market = build_market_data_matrix(
        panel,
        field_columns={"macd_dif", "macd_dea", "macd_hist"},
    )
    defaults = {item["id"]: item["default"] for item in META["params"]}
    defaults.update({
        "use_financial_filter": False,
        "minimum_listing_days": 0,
        "enable_high_wick_exit": False,
        "enable_divergence_exit": False,
        "enable_ma20_reduction": False,
        "enable_trend_break_exit": False,
    })

    default_signals = MATRIX_STRATEGY.compute_signals(market, defaults)
    repeated = MATRIX_STRATEGY.compute_signals(market, dict(defaults))
    changed = MATRIX_STRATEGY.compute_signals(
        market,
        {**defaults, "first_target_ratio": 0.7},
    )
    short_weakness = default_signals.exit_signal_code == 0

    assert short_weakness.any()
    np.testing.assert_array_equal(
        default_signals.target_position_ratio,
        repeated.target_position_ratio,
    )
    assert np.all(default_signals.target_position_ratio[short_weakness] == pytest.approx(0.5))
    assert np.all(changed.target_position_ratio[short_weakness] == pytest.approx(0.7))


def test_effective_parameter_snapshot_contains_defaults_and_execution_settings():
    strategy = type("Strategy", (), {"meta": META})()
    params = StrategyBacktestService._normalize_params(
        {"first_target_ratio": 0.6},
        strategy,
    )
    config = StrategyBacktestConfig(
        strategy_id="strong_trend_pullback_confirm",
        symbols=None,
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        params=params,
        initial_capital=2_000_000,
        benchmark_symbol="000300.SH",
    )

    snapshot = StrategyBacktestService._config_to_dict(config)

    assert len(params) == len(META["params"])
    assert snapshot["params"]["first_target_ratio"] == 0.6
    assert snapshot["params"]["ma_long_period"] == 60
    assert snapshot["initial_capital"] == 2_000_000
    assert snapshot["benchmark_symbol"] == "000300.SH"
