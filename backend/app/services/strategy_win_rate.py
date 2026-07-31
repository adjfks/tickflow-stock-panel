"""Historical win-rate calculation for screener strategy selections."""
from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from datetime import date
from typing import Any

import polars as pl

from app.services.screener import ScreenerService
from app.strategy import config as strategy_config

BOARD_LABELS = ("沪主板", "深主板", "创业板", "科创板", "北交所")

COMBINATIONS = (
    ("open_open", "开盘买 / 开盘卖", "open", "open"),
    ("open_close", "开盘买 / 收盘卖", "open", "close"),
    ("close_open", "收盘买 / 开盘卖", "close", "open"),
    ("close_close", "收盘买 / 收盘卖", "close", "close"),
)


def board_for_symbol(symbol: str) -> str:
    if symbol.endswith(".BJ"):
        return "北交所"
    if symbol.startswith(("300", "301")):
        return "创业板"
    if symbol.startswith(("688", "689")):
        return "科创板"
    if symbol.endswith(".SH"):
        return "沪主板"
    if symbol.endswith(".SZ"):
        return "深主板"
    return "其他"


def _finite_price(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _empty_stats(signal_count: int = 0) -> dict[str, Any]:
    return {
        "signal_count": signal_count,
        "valid": 0,
        "wins": 0,
        "losses": 0,
        "flats": 0,
        "skipped": 0,
        "skip_reasons": {},
        "win_rate": None,
        "avg_return": None,
        "median_return": None,
    }


def _finish_stats(stats: dict[str, Any], returns: list[float]) -> dict[str, Any]:
    result = dict(stats)
    result["valid"] = result["wins"] + result["losses"] + result["flats"]
    result["win_rate"] = (
        round(result["wins"] / result["valid"], 6) if result["valid"] else None
    )
    result["avg_return"] = round(statistics.fmean(returns), 6) if returns else None
    result["median_return"] = round(statistics.median(returns), 6) if returns else None
    return result


def _merge_board_filter(strategy: Any, overrides: dict[str, Any], boards: list[str]) -> dict[str, Any]:
    basic_filter = dict(getattr(strategy, "basic_filter", {}) or {})
    basic_filter.update(dict(overrides.get("basic_filter") or {}))
    strategy_boards = basic_filter.get("boards")
    if isinstance(strategy_boards, list) and strategy_boards:
        selected = [board for board in boards if board in strategy_boards]
        selected = selected or ["__no_matching_board__"]
    else:
        selected = list(boards)
    # A disabled basic filter means its other constraints are intentionally
    # bypassed, but the win-rate market selection must still be enforced.
    if basic_filter.get("enabled") is False:
        basic_filter = {"enabled": True}
    basic_filter["boards"] = selected
    merged = deepcopy(overrides)
    merged["basic_filter"] = basic_filter
    return merged


class StrategyWinRateService:
    """Calculate independent two-day returns for all selected strategy signals."""

    def __init__(self, repo: Any, strategy_engine: Any) -> None:
        self.repo = repo
        self.strategy_engine = strategy_engine

    def _load_prices(self, start_date: date, latest_date: date) -> pl.DataFrame:
        loader = getattr(self.repo, "get_enriched_range", None)
        if not callable(loader):
            raise ValueError("本地 enriched 历史数据不可用")
        prices = loader(
            start_date,
            latest_date,
            columns=["symbol", "date", "open", "close"],
        )
        if prices is None or prices.is_empty():
            raise ValueError("指定日期范围内没有 enriched 历史数据")
        required = {"symbol", "date", "open", "close"}
        missing = sorted(required - set(prices.columns))
        if missing:
            raise ValueError(f"enriched 历史数据缺少字段: {', '.join(missing)}")
        return prices

    @staticmethod
    def _price_map(
        prices: pl.DataFrame,
    ) -> tuple[list[date], dict[str, list[date]], dict[tuple[str, date], dict[str, Any]]]:
        dates = [value for value in prices["date"].unique().sort().to_list() if isinstance(value, date)]
        dates_by_symbol: dict[str, list[date]] = defaultdict(list)
        rows: dict[tuple[str, date], dict[str, Any]] = {}
        for row in prices.iter_rows(named=True):
            symbol = str(row.get("symbol") or "")
            trade_date = row.get("date")
            if symbol and isinstance(trade_date, date):
                dates_by_symbol[symbol].append(trade_date)
                rows[(symbol, trade_date)] = row
        for symbol_dates in dates_by_symbol.values():
            symbol_dates.sort()
        return dates, dates_by_symbol, rows

    @staticmethod
    def _return_for(
        detail: dict[str, Any],
        buy_field: str,
        sell_field: str,
    ) -> tuple[float | None, str | None]:
        buy_price = _finite_price(detail.get(f"buy_{buy_field}"))
        sell_price = _finite_price(detail.get(f"sell_{sell_field}"))
        if buy_price is None or sell_price is None:
            return None, "缺少买入或卖出价格"
        return round(sell_price / buy_price - 1.0, 6), None

    def calculate(
        self,
        strategy_ids: list[str],
        start_date: date,
        end_date: date,
        boards: list[str],
    ) -> dict[str, Any]:
        if not strategy_ids:
            raise ValueError("至少选择一个策略")
        if start_date > end_date:
            raise ValueError("开始日期不能晚于结束日期")
        invalid_boards = [board for board in boards if board not in BOARD_LABELS]
        if not boards or invalid_boards:
            raise ValueError("市场选择无效")

        unique_ids = list(dict.fromkeys(str(strategy_id) for strategy_id in strategy_ids))
        strategies = {meta["id"]: meta for meta in self.strategy_engine.list_strategies()}
        unknown = [strategy_id for strategy_id in unique_ids if strategy_id not in strategies]
        if unknown:
            raise ValueError(f"unknown strategies: {unknown}")
        unsupported = [
            strategy_id
            for strategy_id in unique_ids
            if "stock" not in strategies[strategy_id].get("asset_types", ["stock"])
            or "1d" not in strategies[strategy_id].get("timeframes", ["1d"])
        ]
        if unsupported:
            raise ValueError(f"策略不支持股票日线: {unsupported}")

        screener = ScreenerService(self.repo, asset_type="stock")
        latest_date = screener.latest_date()
        if latest_date is None:
            raise ValueError("没有可用的 enriched 数据")
        if end_date > latest_date:
            raise ValueError(f"结束日期不能晚于最新数据日期 {latest_date}")

        prices = self._load_prices(start_date, latest_date)
        trading_dates, dates_by_symbol, price_map = self._price_map(prices)
        signal_dates = [value for value in trading_dates if start_date <= value <= end_date]
        if not signal_dates:
            raise ValueError("指定日期范围内没有交易日数据")

        data_dir = self.repo.store.data_dir
        all_overrides = strategy_config.list_overrides(data_dir)
        params_map = {
            strategy_id: dict((all_overrides.get(strategy_id) or {}).get("params") or {})
            for strategy_id in unique_ids
        }
        overrides_map = {
            strategy_id: _merge_board_filter(
                self.strategy_engine.get(strategy_id),
                all_overrides.get(strategy_id, {}),
                boards,
            )
            for strategy_id in unique_ids
        }
        strategy_names = {
            strategy_id: (all_overrides.get(strategy_id) or {}).get("name")
            or strategies[strategy_id].get("name", strategy_id)
            for strategy_id in unique_ids
        }

        details: list[dict[str, Any]] = []
        for signal_date in signal_dates:
            context = screener.build_strategy_context(
                self.strategy_engine,
                signal_date,
                unique_ids,
                timeframe="1d",
                params_map=params_map,
                overrides_map=overrides_map,
            )
            results = self.strategy_engine.run_all(
                context,
                params_map=params_map,
                overrides_map=overrides_map,
                strategy_ids=unique_ids,
            )
            for strategy_id in unique_ids:
                strategy_result = results.get(strategy_id)
                if strategy_result is None:
                    continue
                for selected in strategy_result.rows:
                    symbol = str(selected.get("symbol") or "")
                    if not symbol:
                        continue
                    symbol_dates = dates_by_symbol.get(symbol, [])
                    next_index = bisect_right(symbol_dates, signal_date)
                    next_dates = symbol_dates[next_index:next_index + 2]
                    buy_date = next_dates[0] if next_dates else None
                    sell_date = next_dates[1] if len(next_dates) > 1 else None
                    buy = price_map.get((symbol, buy_date), {})
                    sell = price_map.get((symbol, sell_date), {})
                    detail: dict[str, Any] = {
                        "strategy_id": strategy_id,
                        "strategy_name": strategy_names[strategy_id],
                        "symbol": symbol,
                        "name": selected.get("name"),
                        "board": board_for_symbol(symbol),
                        "signal_date": signal_date.isoformat(),
                        "buy_date": buy_date.isoformat() if buy_date else None,
                        "sell_date": sell_date.isoformat() if sell_date else None,
                        "buy_open": buy.get("open"),
                        "buy_close": buy.get("close"),
                        "sell_open": sell.get("open"),
                        "sell_close": sell.get("close"),
                        "returns": {},
                        "statuses": {},
                    }
                    for key, _, buy_field, sell_field in COMBINATIONS:
                        if len(next_dates) < 2:
                            return_value, reason = None, "缺少后续交易日"
                        else:
                            return_value, reason = self._return_for(detail, buy_field, sell_field)
                        detail["returns"][key] = return_value
                        detail["statuses"][key] = "valid" if reason is None else reason
                    details.append(detail)

        details.sort(
            key=lambda row: (row["signal_date"], row["strategy_name"], row["symbol"]),
            reverse=True,
        )
        return self._build_response(
            details, unique_ids, strategy_names, start_date, end_date, boards,
        )

    @staticmethod
    def _build_response(
        details: list[dict[str, Any]],
        strategy_ids: list[str],
        strategy_names: dict[str, str],
        start_date: date,
        end_date: date,
        boards: list[str],
    ) -> dict[str, Any]:
        strategy_details: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for detail in details:
            strategy_details[detail["strategy_id"]].append(detail)

        def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
            combinations: dict[str, Any] = {}
            for key, *_ in COMBINATIONS:
                stats = _empty_stats(len(rows))
                returns: list[float] = []
                for row in rows:
                    value = row["returns"].get(key)
                    if value is None:
                        stats["skipped"] += 1
                        reason = row["statuses"].get(key, "未知原因")
                        skip_reasons = stats["skip_reasons"]
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                        continue
                    returns.append(float(value))
                    if value > 0:
                        stats["wins"] += 1
                    elif value < 0:
                        stats["losses"] += 1
                    else:
                        stats["flats"] += 1
                combinations[key] = _finish_stats(stats, returns)
            return combinations

        by_strategy = []
        for strategy_id in strategy_ids:
            rows = strategy_details.get(strategy_id, [])
            by_strategy.append({
                "strategy_id": strategy_id,
                "strategy_name": strategy_names[strategy_id],
                "signal_count": len(rows),
                "combinations": summarize(rows),
            })

        return {
            "config": {
                "strategy_ids": strategy_ids,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "boards": boards,
            },
            "summary": {
                "signal_count": len(details),
                "strategy_count": len(strategy_ids),
                "combinations": summarize(details),
                "strategies": by_strategy,
            },
            "combination_labels": {key: label for key, label, *_ in COMBINATIONS},
            "details": details,
        }
