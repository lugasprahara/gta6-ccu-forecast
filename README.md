# gta6-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/lugasprahara/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/lugasprahara/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An hour-by-hour forecaster of a big game launch day's Twitch audience: given everything recorded up to the end of hour 1 (that hour's viewer count included) and the hourly histories of dozens of past big launches, it writes five quantiles of the mean concurrent viewers in the game's Twitch category for each of hours 2 to 24. Its core observation, fitted per hour by least squares over the comparable launches, is that a launch day's level regresses on the log of its hour-1 print with a slope well under 1 — a huge first hour means the audience arrived at once and the day grows little, a small one means the rolling midnight unlock is still unfolding and the day multiplies. The slope is shrunk halfway back to the flat-ratio model, the spread comes from the fit's residuals with each training launch weighted by how close its first hour was to the target's, and simultaneous-global-unlock launches are excluded as a different shape. The method was calibrated by leave-one-out over the comparable launches themselves, each forecast from the others with the contest's pinball-on-log loss, where it scores about 0.42 against the persistence baseline's 0.93 (about 0.23 vs 0.36 on rolling-midnight launches like this one). Deterministic, offline, Python 3.13 standard library only.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

(with a case file, `*.in`, in `input/`; the grader runs it with `/input` and `/output` instead)

---

<sub>Entered on [make.fun](https://make.fun/?ref=aes5np1n&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
