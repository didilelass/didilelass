"""Try many 4-week training plans and ask the model which one helps most.

Each plan is turned into a pretend run log for the next 4 weeks, added to
your real history, and the model predicts your 5K on race day. Plans that
grow your mileage faster than about 10% a week are not considered, since
that is the classic way runners get injured.
"""
from __future__ import annotations

import itertools

import pandas as pd

from .data import riegel_5k
from .features import features_on
from .model import predict

# Pace for each kind of run, relative to 5K race pace.
PACE_FACTOR = {"easy": 1.25, "long": 1.30, "tempo": 1.08, "intervals": 1.12}
# Which weekdays to run on (0 = Monday), by number of runs per week.
SCHEDULE = {3: [1, 3, 6], 4: [1, 3, 5, 6], 5: [0, 1, 3, 5, 6]}


def _week_runs(start, km, runs_per_week, hard, long_share, pace5k):
    days = SCHEDULE[runs_per_week]
    long_km = km * long_share
    hard_km = 7.0  # warm-up + session + cool-down
    easy_count = runs_per_week - 1 - hard
    easy_km = max(3.0, (km - long_km - hard * hard_km) / max(easy_count, 1))
    kinds = (["intervals", "tempo"][:hard] if hard <= 2 else ["intervals"] * hard)
    kinds = kinds + ["easy"] * easy_count + ["long"]
    runs = []
    for day, kind in zip(days, kinds):
        dist = {"long": long_km, "easy": easy_km}.get(kind, hard_km)
        runs.append({"date": start + pd.Timedelta(days=day), "distance_km": dist,
                     "duration_min": dist * pace5k / 5 * PACE_FACTOR[kind], "type": kind})
    return runs


def simulate(history, bundle, weekly_km, runs_per_week, hard, long_share, taper):
    start = history["date"].max() + pd.Timedelta(days=1)
    start = start + pd.Timedelta(days=(7 - start.weekday()) % 7)  # next Monday
    current = current_prediction(history, bundle)
    future = []
    for week, km in enumerate(weekly_km):
        if taper and week == len(weekly_km) - 1:
            km *= 0.6
        future += _week_runs(start + pd.Timedelta(weeks=week), km, runs_per_week,
                             hard, long_share, current)
    log = pd.concat([history, pd.DataFrame(future)], ignore_index=True)
    race_day = start + pd.Timedelta(weeks=len(weekly_km))
    feats = features_on(log, race_day, is_race=1,
                        recent_best=recent_best(history))
    return float(predict(bundle, feats)[0]), race_day


def recent_best(history):
    hard = history[history["type"].isin(["race", "tempo"])].tail(8)
    return float(riegel_5k(hard["duration_min"], hard["distance_km"]).min())


def current_prediction(history, bundle) -> float:
    day = history["date"].max() + pd.Timedelta(days=1)
    return float(predict(bundle, features_on(history, day, is_race=1,
                                             recent_best=recent_best(history)))[0])


def best_plans(history, bundle, top=5) -> pd.DataFrame:
    recent = history[history["date"] > history["date"].max() - pd.Timedelta(days=28)]
    base_km = max(recent["distance_km"].sum() / 4, 10.0)
    rows = []
    for growth, rpw, hard, long_share, taper in itertools.product(
            [0.0, 0.05, 0.10], [3, 4, 5], [1, 2], [0.25, 0.33], [False, True]):
        if hard >= rpw - 1:
            continue
        weekly = [base_km * (1 + growth) ** (w + 1) for w in range(4)]
        if weekly[0] / rpw < 3:
            continue
        pred, race_day = simulate(history, bundle, weekly, rpw, hard, long_share, taper)
        rows.append({
            "predicted_5k": pred,
            "weekly_km": " → ".join(f"{k:.0f}" for k in weekly),
            "runs_per_week": rpw,
            "hard_sessions": hard,
            "long_run_share": f"{long_share:.0%}",
            "taper_last_week": taper,
            "race_day": race_day.date(),
        })
    return pd.DataFrame(rows).sort_values("predicted_5k").head(top).reset_index(drop=True)
