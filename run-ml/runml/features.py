"""Turn a run log into training features.

For each hard effort (race, time trial, tempo) we look only at the training
*before* that day and describe it with numbers: how much you ran, how often,
how hard, and how fresh you were. The model learns how those numbers relate
to the 5K time that effort was worth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import riegel_5k

# How hard each kind of run is on the body, per minute.
INTENSITY = {"easy": 1.0, "long": 1.1, "tempo": 1.6, "intervals": 1.8, "race": 2.0}

FEATURES = [
    "fitness",          # 42-day training load (chronic, "CTL")
    "fatigue",          # 7-day training load (acute, "ATL")
    "freshness",        # fitness - fatigue ("TSB")
    "km_per_week",      # average over the last 4 weeks
    "km_last_7d",
    "runs_per_week",
    "hard_per_week",    # tempo / intervals / races per week
    "longest_run_km",   # in the last 4 weeks
    "days_since_run",
    "recent_best_5k",   # best 5K-equivalent in the 60 days before
    "is_race",          # was the effort a true all-out race?
]


def daily_load(runs: pd.DataFrame, until: pd.Timestamp | None = None) -> pd.Series:
    load = runs["duration_min"] * runs["type"].map(INTENSITY).fillna(1.0)
    daily = load.groupby(runs["date"]).sum()
    end = until if until is not None else runs["date"].max()
    days = pd.date_range(runs["date"].min(), end, freq="D")
    return daily.reindex(days, fill_value=0.0)


def _ewma(series: pd.Series, days: int) -> pd.Series:
    return series.ewm(alpha=1 - np.exp(-1 / days), adjust=False).mean()


def features_on(runs: pd.DataFrame, day: pd.Timestamp, is_race: int = 1,
                recent_best: float | None = None) -> dict:
    """Features describing training strictly before `day`."""
    before = runs[runs["date"] < day]
    if before.empty:
        raise ValueError("No runs before this date")
    load = daily_load(before, until=day - pd.Timedelta(days=1))
    fitness = _ewma(load, 42).iloc[-1]
    fatigue = _ewma(load, 7).iloc[-1]

    last28 = before[before["date"] >= day - pd.Timedelta(days=28)]
    last7 = before[before["date"] >= day - pd.Timedelta(days=7)]
    if recent_best is None:
        hard = before[(before["date"] >= day - pd.Timedelta(days=60))
                      & before["type"].isin(["race", "tempo"])]
        recent_best = riegel_5k(hard["duration_min"], hard["distance_km"]).min() if len(hard) else np.nan

    return {
        "fitness": fitness,
        "fatigue": fatigue,
        "freshness": fitness - fatigue,
        "km_per_week": last28["distance_km"].sum() / 4,
        "km_last_7d": last7["distance_km"].sum(),
        "runs_per_week": len(last28) / 4,
        "hard_per_week": last28["type"].isin(["tempo", "intervals", "race"]).sum() / 4,
        "longest_run_km": last28["distance_km"].max() if len(last28) else 0.0,
        "days_since_run": (day - before["date"].max()).days,
        "recent_best_5k": recent_best,
        "is_race": is_race,
    }


def build_dataset(runs: pd.DataFrame, min_history_days: int = 28) -> pd.DataFrame:
    """One row per hard effort with enough history before it."""
    start = runs["date"].min() + pd.Timedelta(days=min_history_days)
    efforts = runs[runs["type"].isin(["race", "tempo"]) & (runs["date"] >= start)]
    rows = []
    for _, run in efforts.iterrows():
        f = features_on(runs, run["date"], is_race=int(run["type"] == "race"))
        f["date"] = run["date"]
        f["target_5k_min"] = riegel_5k(run["duration_min"], run["distance_km"])
        rows.append(f)
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    # First efforts have no earlier best: use the runner's typical value.
    data["recent_best_5k"] = data["recent_best_5k"].fillna(data["target_5k_min"].median())
    return data.reset_index(drop=True)
