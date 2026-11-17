# 🏃 Run Coach

A guided running coach for a 4-week 5K plan, made to install on an iPhone home screen.

**Open it:** https://didilelass.github.io/didilelass/running-coach/

## The plan (each week)
1. **Intervals** – 10 min warm-up, 6–8 × 400 m at interval pace with 90 s walk/jog between, 10 min cool-down
2. **Easy** – 30–40 min at easy pace, conversational
3. **Tempo** – 10 min warm-up, 20 min at tempo pace, 10 min cool-down
4. **Long run** – 7–10 km, all easy
5. **Optional** – easy 20–30 min, or skip if your legs are tired

After four weeks, the **5K time trial** appears in week 4.

## 🎯 Your paces
Paces are worked out from a 5K time, so the plan fits any level. In **🎯 Your paces**, choose:
- **My prediction:** paces follow your predicted 5K and get faster as you get fitter
- **A time I choose:** type a recent 5K time

Intervals are about 8% faster than 5K pace, tempo about 7% slower, and easy runs about 22% slower. Until you set anything, paces are based on a 28:45 5K (intervals 5:15–5:25/km, tempo 6:05–6:15/km, easy 6:45–7:15/km).

## During a run
- Big countdown, target pace, and the screen colour changes with each step
- Voice and beeps tell you when each step starts and ends
- 400 m repeats and the long run end on distance: when the time is up, finish the distance and tap **Finish**
- The screen stays awake. Keep the app open and your ringer on to hear the coach

## 📒 Log tab
Tap **Log** at the bottom to add runs you did before, or without the app: 5K races and time trials, tempo runs, long runs, easy runs and intervals. Add the date, distance and time.

## 🧠 Mood, stress and sleep
When you log a run, and after every run in the app, you can tap how you felt: **mood**, **stress**, **sleep quality** and **hours of sleep**. It's optional. The coach uses it to:
- give tips when your sleep is short, your stress is high or your mood is low
- learn, after 3 logged 5Ks with a check-in, how much sleep and mood change *your* 5K time

## 🔮 5K prediction (machine learning)
The home screen predicts your 5K time and updates after every run.

- **What it learns from:** the time of each 400 m repeat (tap Finish at 400 m), your tempo and long-run pace, how many key runs you've done, your mood, stress and sleep, and every 5K result: the time trial, or any race or parkrun you add in the **Log** tab
- **How:** Bayesian linear regression. It starts from what's true for most runners (5K pace is about 8% slower than 400 m repeat pace, 7% faster than tempo pace and 18% faster than easy long-run pace; sleeping well and feeling good help a little), then each 5K result teaches it how *your* sessions turn into *your* race time. With few results it stays close to the general rule; the more you log, the more it trusts your own data
- It shows a likely range, a trend line, and tips (speed vs endurance, pacing your repeats evenly, consistency)
- Everything runs on your phone, nothing is sent anywhere

## Install on iPhone
Open the link in **Safari** → Share → **Add to Home Screen**.

Your progress is saved on your phone only.
