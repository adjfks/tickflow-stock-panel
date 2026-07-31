from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from app.tickflow.repository import KlineRepository


def test_enriched_history_window_is_anchored_to_target_date():
    start = date(2026, 1, 1)
    dates = [start + timedelta(days=offset) for offset in range(220)]
    repo = object.__new__(KlineRepository)
    repo._enriched_history_cache = pl.DataFrame({
        "symbol": ["000001.SZ"] * len(dates),
        "date": dates,
        "close": [10.0] * len(dates),
    })

    history = repo.get_enriched_history(date(2026, 7, 27), lookback_days=3)

    assert history["date"].min() == date(2026, 7, 24)
    assert history["date"].max() == date(2026, 7, 27)
    assert date(2026, 7, 28) not in history["date"].to_list()
