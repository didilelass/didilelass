"""Load a run log from CSV (your own format, or an export from a running app or watch)."""
from __future__ import annotations

import numpy as np
import pandas as pd

RUN_TYPES = {"easy", "long", "tempo", "intervals", "race"}
HARD_TYPES = {"tempo", "intervals", "race"}

# Column names we recognise, mapped to our own names.
ALIASES = {
    "date": "date", "activity date": "date", "start time": "date", "start_date": "date",
    "distance_km": "distance_km", "distance": "distance_km", "distance (km)": "distance_km",
    "duration_min": "duration_min", "moving time": "moving_s", "elapsed time": "moving_s",
    "time": "time_str", "moving_time": "moving_s",
    "avg_hr": "avg_hr", "average heart rate": "avg_hr", "avg hr": "avg_hr",
    "type": "type", "run_type": "type", "workout type": "type",
    "activity type": "activity_type",
}


def _parse_hms(value) -> float:
    """'25:30' or '1:02:03' -> minutes."""
    parts = [float(p) for p in str(value).split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs / 60


def load_runs(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    # Some exports have both; moving time leaves out stops at traffic lights.
    if "moving time" in raw.columns and "elapsed time" in raw.columns:
        raw = raw.drop(columns=["elapsed time"])
    df = raw.rename(columns={c: ALIASES[c] for c in raw.columns if c in ALIASES})
    df = df.loc[:, ~df.columns.duplicated()]

    # App exports also contain rides, swims... keep runs only.
    if "activity_type" in df:
        df = df[df["activity_type"].astype(str).str.contains("run", case=False)]

    if "duration_min" not in df:
        if "moving_s" in df:
            df["duration_min"] = pd.to_numeric(df["moving_s"], errors="coerce") / 60
        elif "time_str" in df:
            df["duration_min"] = df["time_str"].map(_parse_hms)
        else:
            raise ValueError("Need a duration_min (or Moving Time / Time) column")

    missing = {"date", "distance_km", "duration_min"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed").dt.normalize()
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    # Some exports give metres.
    if df["distance_km"].median() > 200:
        df["distance_km"] /= 1000
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    df = df.dropna(subset=["date", "distance_km", "duration_min"])
    df = df[(df["distance_km"] >= 1) & (df["duration_min"] > 3)]
    df["pace"] = df["duration_min"] / df["distance_km"]  # min/km
    df = df[(df["pace"] > 2.5) & (df["pace"] < 12)]  # drop GPS junk and walks

    if "type" in df:
        df["type"] = df["type"].astype(str).str.lower().str.strip()
        df.loc[df["type"].isin(["time trial", "time_trial", "tt", "1"]), "type"] = "race"
        df.loc[~df["type"].isin(RUN_TYPES), "type"] = np.nan
    else:
        df["type"] = np.nan
    df["type"] = _infer_types(df)

    if "avg_hr" in df:
        df["avg_hr"] = pd.to_numeric(df["avg_hr"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)[
        ["date", "distance_km", "duration_min", "pace", "type"]
        + (["avg_hr"] if "avg_hr" in df else [])
    ]


def _infer_types(df: pd.DataFrame) -> pd.Series:
    """Fill in missing run types from pace and distance."""
    types = df["type"].astype(object)
    unknown = types.isna()
    if not unknown.any():
        return types
    # 5K-equivalent pace removes the effect of distance, so it shows effort.
    equiv = riegel_5k(df["duration_min"], df["distance_km"]) / 5
    q10, q30 = equiv.quantile([0.10, 0.30])
    long_km = df["distance_km"].quantile(0.85)
    inferred = np.where(equiv <= q10, "race",
               np.where(equiv <= q30, "tempo",
               np.where(df["distance_km"] >= long_km, "long", "easy")))
    types[unknown] = inferred[unknown.to_numpy()]
    return types


def riegel_5k(duration_min, distance_km):
    """Riegel's formula: the 5K time this effort is worth, in minutes."""
    return duration_min * (5.0 / distance_km) ** 1.06
