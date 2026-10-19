# 💸 Spending Tracker

A simple, private spending tracker you can install on your phone's home screen.
It works offline, and your data stays on your phone (nothing goes to a server).

## Features
- Add an expense quickly: amount, category, note, date
- Monthly total, with an optional budget bar that turns orange near the limit and red when you go over
- History you can search, covering every month
- Insights: spending by category, a 6-month trend, daily average, biggest expense
- Tap any expense to edit or delete it
- Export to CSV (opens in Excel or Google Sheets), plus backup and restore
- Light and dark mode follow your phone's setting

## Put it on your phone

The app needs to be hosted somewhere. GitHub Pages is free:

1. Merge this branch into `main`.
2. On GitHub, open the repo's **Settings → Pages**. Under "Build and deployment", pick
   **Deploy from a branch**, choose `main` and `/ (root)`, then click **Save**.
3. After a minute, the app is live at
   `https://didilelass.github.io/didilelass/spending-tracker/`

Then open that link on your phone:

- **iPhone (Safari):** tap the Share button, then **Add to Home Screen**.
- **Android (Chrome):** tap the ⋮ menu, then **Install app** (or **Add to Home screen**).

It opens full screen like a normal app and still works without internet.

## Your data
Expenses are saved in the browser's storage on your phone. If you clear Safari or Chrome
website data, or uninstall the app, they will be deleted. Use **Settings → Download backup**
now and then; you can load that file back with **Restore from backup**.

## Run it on your computer
```sh
cd spending-tracker
python3 -m http.server 8000
# open http://localhost:8000
```
