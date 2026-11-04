"""AQ Twin - how much AIS noise the trust check must tolerate, and how the check scores on every official figure.

The check compares ratios on a LOG scale: gap = |ln(r_AIS / r_off)|. A halving and a doubling then count
the same. The arithmetic form |r_AIS - r_off| is kept for comparison: it is capped on a fall (at most r_off)
and uncapped on a rise, so upward months set its tolerance and it could not flag July 2024. The log form was
adopted AFTER the arithmetic form missed that fall, so it is post hoc on one shock.

1. Noise: in the two STABLE windows, the AIS year-on-year ratio r of rolling L-week container-import sums.
   Tolerance = 95th percentile of |ln r| (nominal 5% false flags if the port did not change in those windows).
   L = 4 (about a month), 13 (a quarter), 39 (nine months), 52 (a year). Both the window and its year-earlier
   twin must lie inside one stable window. The second window contains the March 2026 fall.
2. Periods: every official year-on-year figure (research-sources.md) against the AIS ratio of the same period.
3. Robustness: calendar months instead of rolling ones; first-window start date; March 2026 left out; window 1 alone.
4. Zero weeks: weeks with exactly zero container import tonnes, per period (one data vintage).

Run: python -P trust.py   (writes results/trust_noise.csv, trust_periods.csv, trust_robustness.csv,
                           zero_weeks.csv, trust-summary.json)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).parent; OUT = ROOT / "results"
STABLE = [("2021-07-01", "2023-11-12"), ("2025-04-01", "2026-10-18")]   # same as forecast.py
TAUS = [0.1, 0.25, 0.5, 0.75, 1.0]
LS = {4: "month", 13: "quarter", 39: "nine months", 52: "year"}

WK = pd.read_csv(OUT / "weekly.csv", parse_dates=["date"]).set_index("date")
W = WK["container_import_t"]


def ratios(L, stable=STABLE, drop=None):
    """Stable-window year-on-year ratios of rolling L-week sums."""
    s = W.rolling(L).sum(); r = (s / s.shift(52)).dropna()
    keep = []
    for t in r.index:
        a = t - pd.Timedelta(weeks=52 + L - 1) - pd.Timedelta(days=6)
        if not any(pd.Timestamp(x) <= a and t <= pd.Timestamp(y) for x, y in stable): continue
        w0 = t - pd.Timedelta(weeks=L - 1) - pd.Timedelta(days=6)      # first day of the current window
        if drop and w0 <= pd.Timestamp(drop[1]) and t >= pd.Timestamp(drop[0]): continue
        keep.append(t)
    return r[keep]


noise, tol_log, tol_ar = [], {}, {}
for L, name in LS.items():
    r = ratios(L)
    lg, ar = np.abs(np.log(r)), np.abs(r - 1)
    tol_log[L], tol_ar[L] = float(np.quantile(lg, 0.95)), float(np.quantile(ar, 0.95))
    n1 = sum(t <= pd.Timestamp(STABLE[0][1]) for t in r.index)
    row = dict(window_weeks=L, aggregation=name, n=len(r), n_window1=n1, n_window2=len(r) - n1,
               r_min=round(float(r.min()), 3), r_q05=round(float(r.quantile(0.05)), 3),
               r_q95=round(float(r.quantile(0.95)), 3), r_max=round(float(r.max()), 3),
               log_tol=round(tol_log[L], 2), log_max=round(float(lg.max()), 2),
               arith_tol=round(tol_ar[L], 2), arith_max=round(float(ar.max()), 2))
    row.update({f"arith_false_flag_at_{t}": round(float((ar > t).mean()), 2) for t in TAUS})
    noise.append(row)
noise = pd.DataFrame(noise); noise.to_csv(OUT / "trust_noise.csv", index=False)
print(noise.to_string(index=False))

# official year-on-year ratios: (period, published, AIS window, year-earlier window, official ratio, aggregation L)
d = pd.read_csv(ROOT / "data" / "ports.csv", parse_dates=["date"])
d = d[d.portname == "Aqaba"].set_index("date").sort_index(); d.index = d.index.tz_localize(None)
c = d["import_container"]
P = [("2023 vs 2022", "2024-10-02", "2023-01-01", "2023-12-31", "2022-01-01", "2022-12-31", 1.054, 52),
     ("Q1 2024", "2024-04-15", "2024-01-01", "2024-03-31", "2023-01-01", "2023-03-31", 87708 / 109674, 13),
     ("Jul 2024", "2024-08-21", "2024-07-01", "2024-07-31", "2023-07-01", "2023-07-31", 1.08, 4),
     ("Nov 2024", "2025-01-13", "2024-11-01", "2024-11-30", "2023-11-01", "2023-11-30", 42562 / 36790, 4),
     ("Dec 2024", "2025-01-13", "2024-12-01", "2024-12-31", "2023-12-01", "2023-12-31", 40883 / 31770, 4),
     ("2024 vs 2023", "2025-01-13", "2024-01-01", "2024-12-31", "2023-01-01", "2023-12-31", 427 / 445, 52),
     ("2025 vs 2024", "2025-12-30", "2025-01-01", "2025-12-31", "2024-01-01", "2024-12-31", 1.22, 52),
     ("9M 2026", "2026-10-21", "2026-01-01", "2026-09-30", "2025-01-01", "2025-09-30", 786948 / 738349, 39)]
rows = []
for name, pub, a, b, a0, b0, off, L in P:
    ais = c[a:b].sum() / c[a0:b0].sum()
    lg, ar = abs(np.log(ais / off)), abs(ais - off)
    rows.append(dict(period=name, published=pub, r_ais=round(ais, 3), r_official=round(off, 3),
                     log_gap=round(lg, 2), log_tol=round(tol_log[L], 2), log_flag=bool(lg > tol_log[L]),
                     arith_gap=round(ar, 2), arith_tol=round(tol_ar[L], 2), arith_flag=bool(ar > tol_ar[L]),
                     flag_at_0_25=bool(ar > 0.25)))
per = pd.DataFrame(rows); per.to_csv(OUT / "trust_periods.csv", index=False)
print(per.to_string(index=False))

# robustness of the tolerance
rob = []
m = c.resample("MS").sum(); rm = (m / m.shift(12)).dropna()
okm = [t for t in rm.index if any(pd.Timestamp(x) <= t - pd.DateOffset(years=1) and t + pd.offsets.MonthEnd(0) <= pd.Timestamp(y) for x, y in STABLE)]
rob.append(dict(test="calendar months", aggregation="month", n=len(okm),
                log_tol=round(float(np.quantile(np.abs(np.log(rm[okm])), 0.95)), 2)))
for start in ("2021-07-01", "2021-01-01", "2020-07-01"):
    st = [(start, STABLE[0][1]), STABLE[1]]
    for L in (52, 39):
        r = ratios(L, stable=st); lg, ar = np.abs(np.log(r)), np.abs(r - 1)
        rob.append(dict(test=f"first window starts {start}", aggregation=LS[L], n=len(r),
                        log_tol=round(float(np.quantile(lg, 0.95)), 2), arith_tol=round(float(np.quantile(ar, 0.95)), 2)))
for L in LS:      # current window touches March 2026; and the first stable window alone
    r = ratios(L, drop=("2026-03-01", "2026-03-31"))
    rob.append(dict(test="March 2026 left out", aggregation=LS[L], n=len(r),
                    log_tol=round(float(np.quantile(np.abs(np.log(r)), 0.95)), 2)))
    r = ratios(L, stable=STABLE[:1])
    rob.append(dict(test="first stable window only", aggregation=LS[L], n=len(r),
                    log_tol=round(float(np.quantile(np.abs(np.log(r)), 0.95)), 2)))
rob = pd.DataFrame(rob); rob.to_csv(OUT / "trust_robustness.csv", index=False)
print(rob.to_string(index=False))

# zero weeks: exactly zero AIS tonnes in a week
BLOCKS = [("before window 1", "2019-01-06", "2021-06-30"), ("stable window 1", *STABLE[0]),
          ("crisis", "2023-11-13", "2025-03-31"), ("stable window 2", *STABLE[1])]
zw = []
for name, a, b in BLOCKS:
    x = WK[a:b]
    zw.append(dict(period=name, start=a, end=b, weeks=len(x), zero_container=int((x.container_import_t == 0).sum()),
                   zero_cargo=int((x.cargo_import_t == 0).sum()), zero_calls=int((x.port_calls == 0).sum()),
                   zero_weeks=" ".join(str(t.date()) for t in x.index[x.container_import_t == 0])))
zw = pd.DataFrame(zw); zw.to_csv(OUT / "zero_weeks.csv", index=False)
print(zw.to_string(index=False))

json.dump(dict(stable_windows=STABLE, series="container_import_t (AIS container import tonnes)",
               log_tolerance_q95={LS[L]: round(v, 2) for L, v in tol_log.items()},
               arithmetic_tolerance_q95={LS[L]: round(v, 2) for L, v in tol_ar.items()}),
          open(OUT / "trust-summary.json", "w"), indent=2)
