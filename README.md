# gta6-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/lugasprahara/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/lugasprahara/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An hour-by-hour forecaster of a big game launch day's Twitch audience: given everything recorded up to the end of hour 1 (that hour's viewer count included) and the hourly histories of dozens of past big launches, it writes five quantiles of the mean concurrent viewers in the game's Twitch category for each of hours 2 to 24. Its core observation, fitted per hour by least squares over the comparable launches, is that a launch day's level regresses on the log of its hour-1 print with a slope well under 1 — a huge first hour means the audience arrived at once and the day grows little, a small one means the rolling midnight unlock is still unfolding and the day multiplies. Training launches are weighted by unlock shape: a comparable whose unlock matches the target's — rolling local-midnight wave, one simultaneous global moment, or another shape, read from the case notes — counts in full, and a mismatched one counts at a quarter weight, in both the regression and the residual pool; and the residual spread is calibrated per quantile, the quartile and tail quantiles at full scale while the already-calibrated median keeps its 0.8 scale. This third version adds one branch, for the one situation that anchor logic cannot see: a launch that unlocks at a single worldwide moment still ahead at hour 1, whose first print is only the waiting audience. When the notes state that unlock time and the hour-1 print is a trickle against the peaks of the clean single-unlock comparables, pre-unlock hours are forecast from those comparables' pooled pre-unlock behaviour and the rest of the day from their curves aligned at their unlock moments, in absolute viewers. Every other launch — a rolling-midnight one like this contest's included — takes the standard fit unchanged. Leave-one-out loss improves from 0.373 to 0.347 overall (persistence baseline 0.930) and holds at 0.227 on rolling-midnight launches like this one. Deterministic, offline, Python 3.13 standard library only.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

(with a case file, `*.in`, in `input/`; the grader runs it with `/input` and `/output` instead)

---

<sub>Entered on [make.fun](https://make.fun/?ref=aes5np1n&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
