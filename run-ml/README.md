# 🧠 Run ML: a machine learning coach for your 5K

It learns from **your own run log** how your training affects your 5K time, then:

1. **predicts** your 5K if you raced tomorrow
2. **tells you what matters most** for you (weekly distance, rest, long runs, hard sessions...)
3. **tests ~70 different 4-week plans** and shows the ones it expects to make you fastest, never growing mileage more than 10% a week

**👉 Try it in your browser, no install: https://didilelass.github.io/didilelass/run-ml/**
Upload an export from your running app or watch (or try the sample data). It runs this same Python code in your browser with [Pyodide](https://pyodide.org/), so your file never leaves your computer.

It goes with the [Run Coach](../running-coach) app: use Run Coach to do the sessions, and this to decide what to train next.

## ▶️ Run the code in Google Colab (nothing to install)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/didilelass/didilelass/blob/main/run-ml/run_ml_colab.ipynb) **[`run_ml_colab.ipynb`](run_ml_colab.ipynb)**: the full pipeline: load runs, build features, the three models compete, predicted 5K, what matters most, best plans. Click **Runtime → Run all**, with sample data or your own file.

## 🎓 Learn the math behind it
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/didilelass/didilelass/blob/main/run-ml/learn_bayes_ridge_lasso.ipynb) [`learn_bayes_ridge_lasso.ipynb`](learn_bayes_ridge_lasso.ipynb): a hands-on refresher on Bayes, Bayesian regression, ridge and lasso, using this run data, with exercises

## Try it

```bash
cd run-ml
pip install -r requirements.txt
python make_sample_data.py                 # pretend runner, so you can try it now
python -m runml train data/sample_runs.csv
python -m runml predict data/sample_runs.csv
python -m runml plan data/sample_runs.csv
```

## Use your own runs

**From your running app or watch:** most let you download your activities as a CSV (look for "export data" or "download your data" in the settings). Use it as is; non-running activities are skipped:

```bash
python -m runml train my_activities.csv
python -m runml plan my_activities.csv
```

**Or your own CSV** with these columns (`avg_hr` and `type` are optional):

```
date,distance_km,duration_min,avg_hr,type
2026-03-05,5.0,30.78,178,race
2026-03-08,5.38,43.67,146,long
```

`type` is one of `easy`, `long`, `tempo`, `intervals`, `race` (a 5K time trial counts as `race`). If you leave it out, it is guessed from your pace. Marking your races and tempo runs makes the model much better.

You need at least **12 races or tempo runs** and about 2 months of history. More is better.

## How it works

| Step | What happens |
| --- | --- |
| Target | Every race or tempo run is turned into a "5K-equivalent" time with Riegel's formula (`T₅ₖ = T × (5 / km)^1.06`) |
| Features | For each of those runs, only the training **before** it: fitness (42-day load), fatigue (7-day load), freshness, km/week, km in the last 7 days, runs/week, hard sessions/week, longest run, rest days, recent best |
| Models | Ridge regression, random forest and gradient boosting are compared with time-ordered cross-validation (always train on the past, test on the future) against a naive "you'll run your recent best" baseline. The most accurate one is kept |
| Plans | Each candidate plan becomes a pretend run log for the next 4 weeks; the model predicts your 5K on race day |

Training load = minutes × how hard the run was (easy 1.0, long 1.1, tempo 1.6, intervals 1.8, race 2.0).

## Be realistic

- With one person's data the model is small: expect an error of about ±1 minute at first. It gets better as you log more.
- It cannot predict beyond what you have done before. If you've never run 50 km a week, it won't promise what 50 km does.
- If the "baseline" beats every model, you don't have enough varied data yet. Keep logging.
