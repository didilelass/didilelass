"""Command line: python -m runml <train|predict|plan> runs.csv"""
from __future__ import annotations

import argparse

from .data import load_runs
from .features import build_dataset
from .model import load, save, train
from .plan import best_plans, current_prediction

EXPLAIN = {
    "fitness": "long-term training load (last ~6 weeks)",
    "fatigue": "short-term training load (last ~week)",
    "freshness": "fitness minus fatigue (rested vs tired)",
    "km_per_week": "weekly distance",
    "km_last_7d": "distance in the last 7 days",
    "runs_per_week": "how often you run",
    "hard_per_week": "hard sessions per week",
    "longest_run_km": "your long run",
    "days_since_run": "days of rest before",
    "recent_best_5k": "your recent best",
    "is_race": "race vs training effort",
}


def fmt(minutes: float) -> str:
    total = round(minutes * 60)
    return f"{total // 60}:{total % 60:02d}"


def cmd_train(args):
    runs = load_runs(args.csv)
    data = build_dataset(runs)
    print(f"Loaded {len(runs)} runs, {len(data)} hard efforts to learn from.\n")
    bundle = train(data)
    print("How far off each model is, on runs it never saw (average error):")
    for name, secs in sorted(bundle["scores"].items(), key=lambda kv: kv[1]):
        mark = "  <- chosen" if name == bundle["name"] else ""
        print(f"  {name:22s} {secs:5.0f} s{mark}")
    print("\nWhat matters most for your 5K (seconds of error if the model loses it):")
    for feat, secs in bundle["importance"].head(6).items():
        print(f"  {EXPLAIN[feat]:42s} {secs:5.0f} s")
    save(bundle, args.model)
    print(f"\nSaved model to {args.model}")


def cmd_predict(args):
    bundle = load(args.model)
    runs = load_runs(args.csv)
    pred = current_prediction(runs, bundle)
    print(f"Predicted 5K if you raced tomorrow: {fmt(pred)}  ({fmt(pred / 5)}/km)")
    print(f"Typical error of this model: ±{bundle['scores'][bundle['name']]:.0f} s")


def cmd_plan(args):
    bundle = load(args.model)
    runs = load_runs(args.csv)
    now = current_prediction(runs, bundle)
    plans = best_plans(runs, bundle, top=args.top)
    print(f"Predicted 5K today: {fmt(now)}\n")
    print("Best 4-week plans (mileage never grows more than 10% a week):\n")
    for i, p in plans.iterrows():
        gain = (now - p["predicted_5k"]) * 60
        print(f"{i + 1}. {fmt(p['predicted_5k'])} on {p['race_day']}  ({gain:+.0f} s vs today)")
        print(f"   km per week: {p['weekly_km']}   runs/week: {p['runs_per_week']}   "
              f"hard sessions/week: {p['hard_sessions']}   long run: {p['long_run_share']} of the week"
              f"{'   taper: easy last week' if p['taper_last_week'] else ''}")
    print("\nThe model only knows what your own training has shown it. Treat this as a guide,"
          "\nnot a promise, and back off if anything hurts.")


def main():
    parser = argparse.ArgumentParser(prog="runml", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in [("train", cmd_train, "learn from your run log"),
                            ("predict", cmd_predict, "predict your 5K today"),
                            ("plan", cmd_plan, "find the best 4-week plan")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("csv", help="your run log (your own CSV, or an export from a running app or watch)")
        p.add_argument("--model", default="model.pkl", help="where the model is saved")
        if name == "plan":
            p.add_argument("--top", type=int, default=5)
        p.set_defaults(fn=fn)
    args = parser.parse_args()
    args.fn(args)
