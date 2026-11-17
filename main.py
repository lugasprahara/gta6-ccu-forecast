"""Hour-by-hour Twitch CCU forecaster for a big game launch day.

Written for make.fun's gta6-twitch-ccu-forecast-01. The program reads one case
file (schema makefun/twitch-ccu-case/v1): everything recorded up to the end of
hour 1 of the launch day, that hour's viewer count included, plus the hourly
histories of dozens of past big launches (the case's `comparables`). For each
of hours 2..24 it writes five quantiles of the hour's mean concurrent viewers
in the game's Twitch category.

Method (calibrated by leave-one-out over the comparable launches themselves,
each in turn forecast from the others with the contest's pinball-on-log loss):

  The naive anchor "every hour equals hour 1" wastes the strongest pattern in
  the data: how a launch day unfolds depends on how large hour 1 already was.
  Across past launches, ln(day level) regresses on ln(hour-1 print) with a
  slope well under 1 -- a huge first hour means much of the audience arrived
  at once and the day grows little; a small first hour (a region-by-region
  midnight unlock still rolling) means the day multiplies. So, per hour h:

    1. Training set: the case's comparables whose hourly series covers their
       own launch day (hour 1 plus at least 18 of hours 2..24) and whose
       unlock was not a single simultaneous global one (a different shape:
       their hour 1 is a pre-unlock trickle). The target category's own
       pre-release history fails the coverage test and drops out by itself.
    2. Fit by least squares, per hour, ln y_h = a_h + b_h * ln p1 over the
       training launches; shrink the slope halfway back to the flat-ratio
       model: b_h <- 1 + 0.5 * (b_h - 1). Smooth a_h and b_h over neighbouring
       hours (window +-1).
    3. Spread: the fit's residuals per hour, their five quantiles, computed
       with each training launch weighted by a Gaussian kernel on its ln p1
       around the target's (bandwidth 1.5): a launch whose first hour was a
       slow-burn trickle has a different spread regime from a front-loaded
       mega-launch, and its outliers should not set a big launch's tails.
       The quantiles are smoothed over neighbouring hours and scaled by 0.8 --
       leave-one-out showed the raw small-sample tails run wide.
    4. Forecast: q_tau(h) = exp(a_h + b_h * ln p1 + 0.8 * resid_tau(h)),
       forced non-decreasing in tau, clamped to (0, 1e9], rounded to cents.

  If a case carries too few usable comparables, the program falls back to
  persistence: every hour at hour 1's print with a lognormal spread of
  0.15 * sqrt(h - 1). Any per-case error falls back the same way, so a run
  always writes a well-formed forecast.

Deterministic: pure arithmetic on the case file, no randomness, no clock, no
network. Python 3.13 standard library only.

Run locally:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOURS = list(range(2, 25))

GAMMA = 0.5        # slope shrinkage toward the flat-ratio model
KAPPA = 0.8        # residual-spread scale
BANDWIDTH = 1.5    # ln-p1 kernel width for the local residual quantiles
MIN_TRAIN = 8      # usable comparables needed before the regression is trusted
MIN_DAY_HOURS = 18 # of hours 2..24 a comparable must report to count as a launch

SIMULTANEOUS_MARKERS = [
    "simultaneous", "single global", "single simultaneous", "one time",
    "global unlock", "same time", "10 am", "10:00", "14:00z", "02:00z",
]


def parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def is_simultaneous(unlock_text):
    """True when the unlock note describes one worldwide moment rather than a
    rolling local-midnight unlock (a rolling note wins when both appear)."""
    u = (unlock_text or "").lower()
    rolling = "midnight" in u or "rolling" in u
    marked = any(marker in u for marker in SIMULTANEOUS_MARKERS)
    return marked and not rolling


def launch_day_hours(comparable):
    """{hour: viewers} for the comparable's release day, or None when its
    series does not cover that day well enough to learn a curve from."""
    try:
        day = parse_ts(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_start = {}
    for point in comparable.get("hourly") or []:
        try:
            by_start[point["from"]] = point["viewers"]
        except (KeyError, TypeError):
            continue
    hours = {}
    for h in range(1, 25):
        start = (day + timedelta(hours=h - 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        value = by_start.get(start)
        if value is not None:
            hours[h] = value
    if hours.get(1, 0) < 1:
        return None
    if sum(1 for h in HOURS if h in hours) < MIN_DAY_HOURS:
        return None
    return hours


def training_launches(case):
    """The case's comparables as (ln p1, {hour: ln viewers}) training rows."""
    rows = []
    for comparable in case.get("comparables") or []:
        if "before release" in (comparable.get("name") or ""):
            continue  # the target category's own pre-release history
        if is_simultaneous(comparable.get("unlock")):
            continue
        hours = launch_day_hours(comparable)
        if hours is None:
            continue
        rows.append((math.log(hours[1]), {h: math.log(max(1, v)) for h, v in hours.items()}))
    return rows


def weighted_quantile(pairs, tau):
    """The tau quantile of (value, weight) pairs, linear interpolation."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    cum = 0.0
    for i, (value, weight) in enumerate(ordered):
        cum += weight
        if cum >= tau * total:
            if i + 1 < len(ordered) and weight > 0:
                frac = (tau * total - (cum - weight)) / weight
                return value + frac * (ordered[i + 1][0] - value)
            return value
    return ordered[-1][0]


def smooth(values, window=1):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window)
        hi = min(len(values), i + window + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def baseline_quantiles(p1, hour, levels):
    out = []
    for tau in levels:
        # z for the five scored levels; anything else interpolates on the table
        if tau in LEVELS:
            z = Z[LEVELS.index(tau)]
        else:
            z = _z_approx(tau)
        out.append(p1 * math.exp(z * 0.15 * math.sqrt(hour - 1)))
    return out


def _z_approx(tau):
    """Acklam's rational approximation of the standard normal quantile."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if tau < plow:
        q = math.sqrt(-2 * math.log(tau))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if tau > phigh:
        q = math.sqrt(-2 * math.log(1 - tau))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = tau - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def fit_all(train, ln_p1):
    """Per-hour (a, b) fits with shrunk slopes, smoothed; per-hour residual
    quantile curves around the target's ln p1, smoothed.
    Returns (a_list, b_list, resid_quantiles)."""
    a_list, b_list = [], []
    residuals = {}
    for h in HOURS:
        pts = [(x, ys[h]) for x, ys in train if h in ys]
        if not pts:
            # No training launch reports this hour: stay flat at hour 1 and
            # let the spread come from the pooled residuals below.
            a_list.append(0.0)
            b_list.append(1.0)
            residuals[h] = []
            continue
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        cov = sum((x - mx) * (y - my) for x, y in pts)
        var = sum((x - mx) ** 2 for x, y in pts)
        slope = cov / var if var > 0 else 1.0
        slope = 1.0 + GAMMA * (slope - 1.0)
        intercept = my - slope * mx
        a_list.append(intercept)
        b_list.append(slope)
        residuals[h] = [(y - (intercept + slope * x), x) for x, y in pts]
    a_list = smooth(a_list)
    b_list = smooth(b_list)
    pooled = [pair for h in HOURS for pair in residuals[h]]
    resid_quantiles = {}
    for tau in LEVELS:
        column = []
        for h in HOURS:
            source = residuals[h] if residuals[h] else pooled
            pairs = [(r, math.exp(-((x - ln_p1) ** 2) / (2 * BANDWIDTH ** 2)))
                     for r, x in source]
            column.append(weighted_quantile(pairs, tau))
        resid_quantiles[tau] = smooth(column)
    return a_list, b_list, resid_quantiles


def forecast_case(case):
    known = case.get("known") or []
    first = next((row for row in known if row.get("hour") == 1), None)
    p1 = max(1, first["viewers"]) if first and first.get("viewers") else 1
    levels = case.get("levels") or LEVELS
    hours_out = []
    train = training_launches(case)
    ln_p1 = math.log(p1)
    fits = fit_all(train, ln_p1) if len(train) >= MIN_TRAIN else None
    for row in case.get("forecast") or []:
        h = row["hour"]
        if fits is not None and h in HOURS:
            a_list, b_list, resid_quantiles = fits
            i = h - 2
            pred = a_list[i] + b_list[i] * ln_p1
            qs = []
            for tau in levels:
                if tau in resid_quantiles:
                    rq = resid_quantiles[tau][i]
                else:
                    column = [resid_quantiles[t][i] for t in LEVELS]
                    rq = _interp_tau(LEVELS, column, tau)
                qs.append(math.exp(pred + KAPPA * rq))
        else:
            qs = baseline_quantiles(p1, h, levels)
        # contract: positive, at most 1e9, non-decreasing
        qs = [min(1e9, max(1e-6, q)) for q in qs]
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours_out.append({"hour": h, "q": [round2(q) for q in qs]})
    return {"hours": hours_out}


def _interp_tau(levels, values, tau):
    if tau <= levels[0]:
        return values[0]
    if tau >= levels[-1]:
        return values[-1]
    for i in range(1, len(levels)):
        if tau <= levels[i]:
            frac = (tau - levels[i - 1]) / (levels[i] - levels[i - 1])
            return values[i - 1] * (1 - frac) + values[i] * frac
    return values[-1]


def baseline_case(case):
    known = case.get("known") or []
    first = next((row for row in known if row.get("hour") == 1), None)
    p1 = max(1, first["viewers"]) if first and first.get("viewers") else 1
    levels = case.get("levels") or LEVELS
    out = []
    for row in case.get("forecast") or []:
        qs = baseline_quantiles(p1, row["hour"], levels)
        qs = [min(1e9, max(1e-6, q)) for q in qs]
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        out.append({"hour": row["hour"], "q": [round2(q) for q in qs]})
    return {"hours": out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
            if len(forecast["hours"]) != 23:
                raise ValueError("forecast row count")
        except Exception:
            try:
                forecast = baseline_case(case)
            except Exception:
                # Last resort: a flat, well-formed forecast for hours 2..24.
                forecast = {"hours": [{"hour": h, "q": [1, 1, 1, 1, 1]} for h in HOURS]}
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
