"""Train, compare and save models that predict your 5K time."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


def candidates() -> dict:
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
    }


def evaluate(data: pd.DataFrame) -> dict:
    """Time-ordered cross-validation: always train on the past, test on the future.

    Returns the mean absolute error in seconds for each model, plus the naive
    baseline "your next 5K = your recent best".
    """
    X, y = data[FEATURES], data["target_5k_min"]
    splits = TimeSeriesSplit(n_splits=min(5, max(2, len(data) // 8)))
    errors = {name: [] for name in candidates()}
    errors["baseline_recent_best"] = []
    for train, test in splits.split(X):
        for name, model in candidates().items():
            model.fit(X.iloc[train], y.iloc[train])
            errors[name].append(np.abs(model.predict(X.iloc[test]) - y.iloc[test]).mean())
        base = X.iloc[test]["recent_best_5k"]
        errors["baseline_recent_best"].append(np.abs(base - y.iloc[test]).mean())
    return {name: float(np.mean(e) * 60) for name, e in errors.items()}


def train(data: pd.DataFrame) -> dict:
    if len(data) < 12:
        raise ValueError(
            f"Only {len(data)} hard efforts (races/tempo runs) found; need at least 12. "
            "Log a few more, or mark them with a 'type' column.")
    scores = evaluate(data)
    best = min((n for n in scores if n != "baseline_recent_best"), key=scores.get)
    model = candidates()[best].fit(data[FEATURES], data["target_5k_min"])
    imp = permutation_importance(model, data[FEATURES], data["target_5k_min"],
                                 n_repeats=20, random_state=0)
    importance = pd.Series(imp.importances_mean * 60, index=FEATURES).sort_values(ascending=False)
    return {
        "model": model,
        "name": best,
        "scores": scores,
        "importance": importance,
        "feature_ranges": data[FEATURES].agg(["min", "max"]),
        "n_efforts": len(data),
    }


def save(bundle: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load(path: str) -> dict:
    # Only load model files you made yourself: pickle can run code.
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(bundle: dict, features: dict | pd.DataFrame) -> np.ndarray:
    X = pd.DataFrame([features]) if isinstance(features, dict) else features
    # Keep inputs inside what the model has seen; it cannot extrapolate safely.
    lo, hi = bundle["feature_ranges"].loc["min"], bundle["feature_ranges"].loc["max"]
    X = X[FEATURES].clip(lower=lo, upper=hi, axis=1)
    return bundle["model"].predict(X)
