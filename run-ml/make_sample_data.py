"""Make a realistic pretend run log so you can try the tool before using your own.

The pretend runner gets fitter with steady training, slower when tired or
after a break, and races a 5K or does a tempo run most weeks.
Writes data/sample_runs.csv.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
days = pd.date_range("2026-03-02", periods=36 * 7, freq="D")
fitness = fatigue = 0.0
base_5k = 31.0  # minutes, when untrained
rows = []

for i, day in enumerate(days):
    week = i // 7
    weekday = day.weekday()
    # Training gets bigger over time, with an easy week every 4th week and a
    # 2-week break (holiday) in the middle.
    volume = (14 + 0.9 * week) * (0.6 if week % 4 == 3 else 1.0)
    if 17 <= week < 19:
        volume = 0
    plan = {1: ("intervals", 0.22), 3: ("tempo" if week % 2 else "race", 0.20),
            5: ("easy", 0.2), 6: ("long", 0.38)}
    if week >= 10:
        plan[0] = ("easy", 0.15)
    load = 0.0
    if volume and weekday in plan and rng.random() > 0.08:  # sometimes skip
        kind, share = plan[weekday]
        current_5k = base_5k - 0.11 * fitness + 0.05 * fatigue
        pace5k = current_5k / 5
        if kind == "race":
            km = 5.0
            minutes = current_5k + rng.normal(0, 0.35)
        else:
            km = max(3.0, volume * share + rng.normal(0, 0.6))
            factor = {"intervals": 1.12, "tempo": 1.08, "easy": 1.27, "long": 1.32}[kind]
            minutes = km * pace5k * factor * (1 + rng.normal(0, 0.02))
        hr = {"race": 178, "tempo": 168, "intervals": 165, "easy": 142, "long": 146}[kind]
        rows.append({"date": day.date(), "distance_km": round(km, 2),
                     "duration_min": round(minutes, 2), "avg_hr": hr + int(rng.normal(0, 4)),
                     "type": kind})
        load = minutes * {"easy": 1.0, "long": 1.1, "tempo": 1.6, "intervals": 1.8, "race": 2.0}[kind]
    fitness += (load - fitness) * (1 - np.exp(-1 / 42))
    fatigue += (load - fatigue) * (1 - np.exp(-1 / 7))

pd.DataFrame(rows).to_csv("data/sample_runs.csv", index=False)
print(f"Wrote {len(rows)} runs to data/sample_runs.csv")
