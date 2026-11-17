# 🏃 Run Coach

A guided running coach for a 4-week 5K plan, made to install on an iPhone home screen.

**Open it:** https://didilelass.github.io/didilelass/running-coach/

## The plan (each week)
1. **Intervals** – 10 min warm-up, 6–8 × 400 m at 5:15–5:25/km with 90 s walk/jog between, 10 min cool-down
2. **Easy** – 30–40 min at 6:45–7:15/km, conversational
3. **Tempo** – 10 min warm-up, 20 min at 6:05–6:15/km, 10 min cool-down
4. **Long run** – 7–10 km, all easy
5. **Optional** – easy 20–30 min, or skip if your legs are tired

After four weeks, the **5K time trial** appears in week 4.

## During a run
- Big countdown, target pace, and the screen colour changes with each step
- Voice and beeps tell you when each step starts and ends
- 400 m repeats and the long run end on distance: when the time is up, finish the distance and tap **Finish**
- The screen stays awake. Keep the app open and your ringer on to hear the coach

## 🔮 5K prediction (machine learning)
The home screen predicts your 5K time and updates after every run.

- **What it learns from:** the time of each 400 m repeat (tap Finish at 400 m), your long-run pace, how many key runs you've done, and every 5K result: the time trial, or any race or parkrun you add with **+ Add a 5K result**
- **How:** Bayesian linear regression. It starts from what's true for most runners (5K pace is about 8% slower than 400 m repeat pace and 18% faster than easy long-run pace), then each 5K result teaches it how *your* sessions turn into *your* race time. With few results it stays close to the general rule; the more you log, the more it trusts your own data
- It shows a likely range, a trend line, and tips (speed vs endurance, pacing your repeats evenly, consistency)
- Everything runs on your phone, nothing is sent anywhere

## Install on iPhone
Open the link in **Safari** → Share → **Add to Home Screen**.

Your progress is saved on your phone only.
