"""强趋势回踩确认: 趋势、缩量回踩、放量确认、PIT 财务与分批退出。"""

from __future__ import annotations

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    matrix_feature,
    valid_ewm_adjust_false,
    valid_rolling_max,
    valid_rolling_mean,
    valid_shift,
)


def _pct_param(
    param_id: str,
    label: str,
    default: float,
    minimum: float,
    maximum: float,
    step: float,
    group: str,
    *,
    depends_on: str | None = None,
) -> dict:
    result = {
        "id": param_id,
        "label": label,
        "type": "float",
        "default": default,
        "min": minimum,
        "max": maximum,
        "step": step,
        "unit": "percent",
        "group": group,
        "optimizable": True,
    }
    if depends_on:
        result["depends_on"] = depends_on
    return result


META = {
    "id": "strong_trend_pullback_confirm",
    "name": "强趋势回踩确认",
    "description": "趋势多头、缩量回踩、放量确认、公告日财务过滤与参数化分批退出",
    "tags": ["趋势", "回踩", "基本面", "分批退出"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "params": [
        {"id": "ma_fast_period", "label": "快速均线", "type": "int", "default": 5, "min": 2, "max": 20, "step": 1, "group": "strategy", "optimizable": True},
        {"id": "ma_short_period", "label": "短期均线", "type": "int", "default": 10, "min": 3, "max": 40, "step": 1, "group": "strategy", "optimizable": True},
        {"id": "ma_mid_period", "label": "中期均线", "type": "int", "default": 20, "min": 5, "max": 80, "step": 1, "group": "strategy", "optimizable": True},
        {"id": "ma_trend_period", "label": "趋势均线", "type": "int", "default": 30, "min": 10, "max": 120, "step": 1, "group": "strategy", "optimizable": True},
        {"id": "ma_long_period", "label": "长期均线", "type": "int", "default": 60, "min": 20, "max": 250, "step": 1, "group": "strategy", "optimizable": True},
        {"id": "slope_lookback_days", "label": "均线斜率回看天数", "type": "int", "default": 5, "min": 1, "max": 30, "step": 1, "group": "strategy", "optimizable": True},
        {"id": "momentum_period", "label": "动量周期", "type": "int", "default": 20, "min": 5, "max": 120, "step": 1, "group": "strategy", "optimizable": True},
        _pct_param("momentum_min", "最低涨幅", 0.08, 0.0, 1.0, 0.01, "strategy"),
        {"id": "pullback_window", "label": "回踩观察窗口", "type": "int", "default": 5, "min": 1, "max": 20, "step": 1, "group": "strategy", "optimizable": True},
        _pct_param("ma_pullback_distance", "MA10/20 距离", 0.03, 0.0, 0.15, 0.005, "strategy"),
        _pct_param("ma20_break_tolerance", "MA20 容忍跌破", 0.03, 0.0, 0.10, 0.005, "strategy"),
        {"id": "volume_shrink_window", "label": "缩量窗口", "type": "int", "default": 3, "min": 1, "max": 20, "step": 1, "group": "strategy", "optimizable": True},
        {"id": "volume_compare_window", "label": "成交量对比窗口", "type": "int", "default": 5, "min": 2, "max": 30, "step": 1, "group": "strategy", "optimizable": True},
        _pct_param("shrink_volume_ratio", "最大缩量比", 0.90, 0.10, 1.50, 0.05, "strategy"),
        {"id": "confirm_volume_ratio", "label": "确认量比", "type": "float", "default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1, "group": "strategy", "optimizable": True},
        {"id": "require_ma_recross", "label": "要求重新站上 MA5/10", "type": "bool", "default": True, "group": "strategy"},
        {"id": "accept_macd_green_shrinking", "label": "接受 MACD 绿柱缩短", "type": "bool", "default": True, "group": "strategy"},
        {"id": "use_financial_filter", "label": "启用公告日财务过滤", "type": "bool", "default": False, "group": "strategy"},
        _pct_param("revenue_yoy_min", "营收同比下限", 0.0, -1.0, 5.0, 0.01, "strategy", depends_on="use_financial_filter"),
        _pct_param("net_income_yoy_min", "净利润同比下限", 0.0, -1.0, 10.0, 0.01, "strategy", depends_on="use_financial_filter"),
        _pct_param("roe_min", "ROE 下限", 0.08, -1.0, 1.0, 0.01, "strategy", depends_on="use_financial_filter"),
        _pct_param("debt_to_asset_ratio_max", "负债率上限", 0.60, 0.0, 1.5, 0.01, "strategy", depends_on="use_financial_filter"),
        {"id": "exclude_st", "label": "排除 ST/退市整理", "type": "bool", "default": True, "group": "strategy"},
        {"id": "minimum_listing_days", "label": "上市最少天数", "type": "int", "default": 120, "min": 0, "max": 2000, "step": 10, "group": "strategy", "optimizable": True},
        {"id": "board_scope", "label": "板块范围", "type": "select", "default": "all", "options": ["all", "main", "gem", "star", "beijing"], "option_labels": {"all": "全部 A 股", "main": "主板", "gem": "创业板", "star": "科创板", "beijing": "北交所"}, "group": "strategy"},
        {"id": "enable_short_weakness", "label": "启用 MA10 次日未收回减仓", "type": "bool", "default": True, "group": "risk"},
        {"id": "enable_ma20_reduction", "label": "启用连续跌破 MA20 降仓", "type": "bool", "default": True, "group": "risk"},
        {"id": "consecutive_below_ma20_days", "label": "MA20 下方连续天数", "type": "int", "default": 3, "min": 2, "max": 10, "step": 1, "group": "risk", "optimizable": True, "depends_on": "enable_ma20_reduction"},
        {"id": "enable_trend_break_exit", "label": "启用 MA30/60 放量破位清仓", "type": "bool", "default": True, "group": "risk"},
        {"id": "trend_break_volume_ratio", "label": "趋势破位量比", "type": "float", "default": 1.5, "min": 0.5, "max": 5.0, "step": 0.1, "group": "risk", "optimizable": True, "depends_on": "enable_trend_break_exit"},
        {"id": "enable_high_wick_exit", "label": "启用高位长上影减仓", "type": "bool", "default": True, "group": "risk"},
        _pct_param("high_distance_ma10", "远离 MA10 比例", 0.15, 0.02, 1.0, 0.01, "risk", depends_on="enable_high_wick_exit"),
        {"id": "high_volume_ratio", "label": "高位量比", "type": "float", "default": 1.5, "min": 0.5, "max": 5.0, "step": 0.1, "group": "risk", "optimizable": True, "depends_on": "enable_high_wick_exit"},
        _pct_param("upper_wick_ratio", "上影占比", 0.50, 0.10, 1.0, 0.05, "risk", depends_on="enable_high_wick_exit"),
        {"id": "enable_divergence_exit", "label": "启用 MACD/OBV 背离减仓", "type": "bool", "default": True, "group": "risk"},
        {"id": "divergence_window", "label": "背离窗口", "type": "int", "default": 20, "min": 5, "max": 120, "step": 1, "group": "risk", "optimizable": True, "depends_on": "enable_divergence_exit"},
        _pct_param("first_target_ratio", "一级目标仓位", 0.50, 0.0, 1.0, 0.05, "risk"),
        _pct_param("second_target_ratio", "二级目标仓位", 0.25, 0.0, 1.0, 0.05, "risk"),
        {"id": "enable_dynamic_ma_stop", "label": "启用动态均线止损", "type": "bool", "default": True, "group": "risk"},
        _pct_param("initial_ma20_buffer", "初始 MA20 缓冲", 0.025, 0.0, 0.20, 0.005, "risk", depends_on="enable_dynamic_ma_stop"),
        _pct_param("profit_activation_1", "一级盈利激活点", 0.10, 0.0, 2.0, 0.01, "risk", depends_on="enable_dynamic_ma_stop"),
        _pct_param("profit_activation_2", "二级盈利激活点", 0.20, 0.0, 3.0, 0.01, "risk", depends_on="enable_dynamic_ma_stop"),
        _pct_param("trailing_ma10_buffer", "MA10 止损缓冲", 0.02, 0.0, 0.20, 0.005, "risk", depends_on="enable_dynamic_ma_stop"),
        _pct_param("trailing_ma5_buffer", "MA5 止损缓冲", 0.02, 0.0, 0.20, 0.005, "risk", depends_on="enable_dynamic_ma_stop"),
    ],
    "basic_filter": {"enabled": False},
    "scoring": {"momentum_20d": 1.0},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["strong_trend_pullback_confirm"]
EXIT_SIGNALS = [
    "ma10_not_recovered",
    "high_wick",
    "momentum_divergence",
    "ma20_consecutive_break",
    "ma30_60_volume_break",
]
STOP_LOSS = None
MAX_HOLD_DAYS = None
ALERTS = []

_PARAM_DEFAULTS = {
    str(param["id"]): param.get("default")
    for param in META["params"]
    if param.get("id")
}


def _shift(values: np.ndarray, periods: int) -> np.ndarray:
    return valid_shift(values, periods, np.isfinite(values))


def _optional_field(market: MarketDataMatrix, name: str) -> np.ndarray:
    values = market.fields.get(name)
    if values is not None:
        return values
    return np.full(market.shape, np.nan, dtype=np.float32)


def _asset_board_mask(symbols: tuple[str, ...], scope: str) -> np.ndarray:
    if scope == "all":
        return np.ones(len(symbols), dtype=bool)
    result = np.zeros(len(symbols), dtype=bool)
    for index, symbol in enumerate(symbols):
        code = symbol.split(".", 1)[0]
        if scope == "gem":
            result[index] = code.startswith(("300", "301"))
        elif scope == "star":
            result[index] = code.startswith(("688", "689"))
        elif scope == "beijing":
            result[index] = symbol.endswith(".BJ") or code.startswith(("4", "8"))
        elif scope == "main":
            result[index] = not (
                code.startswith(("300", "301", "688", "689", "4", "8"))
                or symbol.endswith(".BJ")
            )
    return result


class StrongTrendPullbackConfirmMatrixStrategy:
    _TECHNICAL_FIELDS = frozenset({
        "close",
        "high",
        "low",
        "volume",
        "macd_dif",
        "macd_dea",
        "macd_hist",
        "listing_day",
        "name",
    })
    _FINANCIAL_FIELDS = frozenset({
        "revenue_yoy",
        "net_income_yoy",
        "roe",
        "debt_to_asset_ratio",
    })

    def required_fields(self) -> frozenset[str]:
        return self._TECHNICAL_FIELDS | self._FINANCIAL_FIELDS

    def required_fields_for_params(self, params: dict) -> frozenset[str]:
        fields = set(self._TECHNICAL_FIELDS)
        if params.get(
            "use_financial_filter",
            _PARAM_DEFAULTS["use_financial_filter"],
        ):
            fields.update(self._FINANCIAL_FIELDS)
        return frozenset(fields)

    def required_warmup_bars(self, params: dict) -> int:
        return max(
            int(params.get("ma_long_period", 60)),
            int(params.get("momentum_period", 20)),
            int(params.get("divergence_window", 20)),
        ) + max(
            int(params.get("slope_lookback_days", 5)),
            int(params.get("volume_shrink_window", 3))
            + int(params.get("volume_compare_window", 5)),
            int(params.get("pullback_window", _PARAM_DEFAULTS["pullback_window"])),
        ) + 5

    @staticmethod
    def validate_params(params: dict) -> None:
        periods = [
            int(params[name])
            for name in (
                "ma_fast_period",
                "ma_short_period",
                "ma_mid_period",
                "ma_trend_period",
                "ma_long_period",
            )
        ]
        if periods != sorted(periods) or len(set(periods)) != len(periods):
            raise ValueError("均线周期必须满足 快速 < 短期 < 中期 < 趋势 < 长期")
        first = float(params["first_target_ratio"])
        second = float(params["second_target_ratio"])
        if not 0.0 <= second <= first <= 1.0:
            raise ValueError("仓位目标必须满足 0 <= 二级目标仓位 <= 一级目标仓位 <= 100%")
        activation_1 = float(params["profit_activation_1"])
        activation_2 = float(params["profit_activation_2"])
        if activation_2 < activation_1:
            raise ValueError("二级盈利激活点必须大于或等于一级盈利激活点")

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        params = {**_PARAM_DEFAULTS, **params}
        self.validate_params(params)
        close = market.close
        volume = market.volume
        valid = np.isfinite(close)
        ma5 = matrix_feature(market, f"ma{int(params['ma_fast_period'])}")
        ma10 = matrix_feature(market, f"ma{int(params['ma_short_period'])}")
        ma20 = matrix_feature(market, f"ma{int(params['ma_mid_period'])}")
        ma30 = matrix_feature(market, f"ma{int(params['ma_trend_period'])}")
        ma60 = matrix_feature(market, f"ma{int(params['ma_long_period'])}")

        slope_lookback = int(params["slope_lookback_days"])
        trend = (
            (ma5 > ma10)
            & (ma10 > ma20)
            & (ma20 > ma30)
            & (ma30 > ma60)
            & (ma30 > _shift(ma30, slope_lookback))
            & (ma60 > _shift(ma60, slope_lookback))
        )
        momentum = matrix_feature(
            market,
            f"momentum_{int(params['momentum_period'])}d",
        )
        strength = momentum >= float(params["momentum_min"])

        pullback_distance = float(params["ma_pullback_distance"])
        ma20_tolerance = float(params["ma20_break_tolerance"])
        near_ma = (
            (np.abs(close / ma10 - 1.0) <= pullback_distance)
            | (np.abs(close / ma20 - 1.0) <= pullback_distance)
        )
        pullback_bar = near_ma & (close >= ma20 * (1.0 - ma20_tolerance))
        pullback_seen = np.zeros(market.shape, dtype=bool)
        for offset in range(1, int(params["pullback_window"]) + 1):
            pullback_seen |= _shift(pullback_bar.astype(np.float32), offset) > 0.5

        shrink_window = int(params["volume_shrink_window"])
        compare_window = int(params["volume_compare_window"])
        previous_volume = _shift(volume, 1)
        recent_mean = valid_rolling_mean(
            previous_volume,
            np.isfinite(previous_volume),
            shrink_window,
        )
        comparison_source = _shift(volume, shrink_window + 1)
        comparison_mean = valid_rolling_mean(
            comparison_source,
            np.isfinite(comparison_source),
            compare_window,
        )
        shrink_confirmed = recent_mean <= comparison_mean * float(params["shrink_volume_ratio"])
        confirmation_base = valid_rolling_mean(
            previous_volume,
            np.isfinite(previous_volume),
            compare_window,
        )
        volume_confirmed = volume >= confirmation_base * float(params["confirm_volume_ratio"])

        previous_close = _shift(close, 1)
        recross = (
            ((close > ma5) & (previous_close <= _shift(ma5, 1)))
            | ((close > ma10) & (previous_close <= _shift(ma10, 1)))
        )
        if not params.get("require_ma_recross", True):
            recross = valid

        dif = market.fields.get("macd_dif")
        dea = market.fields.get("macd_dea")
        hist = market.fields.get("macd_hist")
        if dif is None or dea is None or hist is None:
            ema12 = valid_ewm_adjust_false(close, valid, span=12)
            ema26 = valid_ewm_adjust_false(close, valid, span=26)
            dif = ema12 - ema26
            dea = valid_ewm_adjust_false(dif, np.isfinite(dif), span=9)
            hist = (dif - dea) * np.float32(2.0)
        macd_golden = (dif > dea) & (_shift(dif, 1) <= _shift(dea, 1))
        macd_confirmed = macd_golden
        if params.get("accept_macd_green_shrinking", True):
            macd_confirmed |= (hist < 0) & (hist > _shift(hist, 1))

        fundamentals = valid.copy()
        if params.get("use_financial_filter", True):
            revenue = _optional_field(market, "revenue_yoy")
            profit = _optional_field(market, "net_income_yoy")
            roe = _optional_field(market, "roe")
            debt = _optional_field(market, "debt_to_asset_ratio")
            fundamentals = (
                np.isfinite(revenue)
                & np.isfinite(profit)
                & np.isfinite(roe)
                & np.isfinite(debt)
                & (revenue >= float(params["revenue_yoy_min"]))
                & (profit >= float(params["net_income_yoy_min"]))
                & (roe >= float(params["roe_min"]))
                & (debt <= float(params["debt_to_asset_ratio_max"]))
            )

        universe = np.broadcast_to(
            _asset_board_mask(market.symbols, str(params["board_scope"]))[None, :],
            market.shape,
        ).copy()
        if params.get("exclude_st", True):
            allowed_names = np.asarray([
                not any(token in name.upper() for token in ("ST", "*ST", "退"))
                for name in market.names
            ], dtype=bool)
            universe &= allowed_names[None, :]
        minimum_listing_days = int(params["minimum_listing_days"])
        if minimum_listing_days > 0:
            listing_day = _optional_field(market, "listing_day")
            trading_day = (market.timestamps // 86_400_000).astype(np.float32)
            listing_age = trading_day[:, None] - listing_day
            universe &= np.isfinite(listing_age) & (listing_age >= minimum_listing_days)

        entry = (
            valid
            & trend
            & strength
            & pullback_seen
            & shrink_confirmed
            & volume_confirmed
            & recross
            & macd_confirmed
            & fundamentals
            & universe
        )

        first_target = float(params["first_target_ratio"])
        second_target = float(params["second_target_ratio"])
        target = np.full(market.shape, np.nan, dtype=np.float32)
        exit_code = np.full(market.shape, -1, dtype=np.int16)

        below_ma10 = close < ma10
        short_weakness = below_ma10 & (_shift(close, 1) < _shift(ma10, 1))
        if params.get("enable_short_weakness", True):
            target[short_weakness] = first_target
            exit_code[short_weakness] = 0

        previous_mean_volume = valid_rolling_mean(
            previous_volume,
            np.isfinite(previous_volume),
            compare_window,
        )
        daily_range = market.high - market.low
        upper_wick = np.divide(
            market.high - np.maximum(close, market.open),
            daily_range,
            out=np.zeros(market.shape, dtype=np.float32),
            where=np.isfinite(daily_range) & (daily_range > 0),
        )
        high_wick = (
            (close >= ma10 * (1.0 + float(params["high_distance_ma10"])))
            & (volume >= previous_mean_volume * float(params["high_volume_ratio"]))
            & (upper_wick >= float(params["upper_wick_ratio"]))
        )
        if params.get("enable_high_wick_exit", True):
            update = high_wick & ~np.isfinite(target)
            target[update] = first_target
            exit_code[update] = 1

        divergence_window = int(params["divergence_window"])
        obv = matrix_feature(market, "obv")
        prior_price_high = _shift(valid_rolling_max(close, valid, divergence_window), 1)
        prior_dif_high = _shift(
            valid_rolling_max(dif, np.isfinite(dif), divergence_window),
            1,
        )
        prior_obv_high = _shift(
            valid_rolling_max(obv, np.isfinite(obv), divergence_window),
            1,
        )
        divergence = (close >= prior_price_high) & (
            (dif < prior_dif_high) | (obv < prior_obv_high)
        )
        if params.get("enable_divergence_exit", True):
            update = divergence & ~np.isfinite(target)
            target[update] = first_target
            exit_code[update] = 2

        below_ma20 = close < ma20
        consecutive_break = below_ma20.copy()
        for offset in range(1, int(params["consecutive_below_ma20_days"])):
            consecutive_break &= _shift(below_ma20.astype(np.float32), offset) > 0.5
        if params.get("enable_ma20_reduction", True):
            target[consecutive_break] = second_target
            exit_code[consecutive_break] = 3

        volume_break = volume >= previous_mean_volume * float(params["trend_break_volume_ratio"])
        trend_break = volume_break & (
            ((close < ma30) & (previous_close >= _shift(ma30, 1)))
            | ((close < ma60) & (previous_close >= _shift(ma60, 1)))
        )
        if params.get("enable_trend_break_exit", True):
            target[trend_break] = 0.0
            exit_code[trend_break] = 4

        exit_ = np.isfinite(target)
        score = np.nan_to_num(momentum * 100.0, nan=0.0).astype(np.float32)
        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            exit=exit_.astype(np.uint8),
            score=score,
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=exit_code,
            entry_signal_ids=("strong_trend_pullback_confirm",),
            exit_signal_ids=(
                "ma10_not_recovered",
                "high_wick",
                "momentum_divergence",
                "ma20_consecutive_break",
                "ma30_60_volume_break",
            ),
            target_position_ratio=target,
            risk_ma5=ma5,
            risk_ma10=ma10,
            risk_ma20=ma20,
        )


MATRIX_STRATEGY = StrongTrendPullbackConfirmMatrixStrategy()
