"""AQ Twin - how much AIS noise the trust check must tolerate, and how the check scores on every official figure.

1. Noise: in the two STABLE windows (no known disruption), compare the AIS year-on-year ratio of rolling
   L-week container-import sums with 1. |ratio - 1| is pure AIS noise, because nothing happened.
   The false-flag rate at tolerance tau is the share of stable windows with |ratio - 1| > tau.
   L = 4 (about a month), 13 (a quarter), 39 (nine months), 52 (a year). Both the window and its
   year-earlier twin must lie inside one stable window.
2. Periods: for each official year-on-year figure (research-sources.md) the AIS ratio for the same calendar
   period, the gap, and whether it flags at the 0.25 tolerance used by loop.py and at the noise-calibrated
   tolerance (the 95th percentile of stable noise for that aggregation, i.e. a 5% false-flag rate).

Run: python -P trust.py   (writes results/trust_noise.csv, results/trust_periods.csv, results/trust-summary.json)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).parent; OUT = ROOT / "results"
STABLE = [("2021-07-01", "2023-11-12"), ("2025-04-01", "2026-10-18")]   # same as forecast.py
TAUS = [0.1, 0.25, 0.5, 0.75, 1.0]
LS = {4: "month", 13: "quarter", 39: "nine months", 52: "year"}

W = pd.read_csv(OUT / "weekly.csv", parse_dates=["date"]).set_index("date")["container_import_t"]


def inside(a, b):
    return any(pd.Timestamp(x) <= a and b <= pd.Timestamp(y) for x, y in STABLE)


noise, q95 = [], {}
for L, name in LS.items():
    s = W.rolling(L).sum(); r = (s / s.shift(52)).dropna()
    ok = [t for t in r.index if inside(t - pd.Timedelta(weeks=52 + L - 1) - pd.Timedelta(days=6), t)]
    e = np.abs(r[ok] - 1); q95[L] = float(np.quantile(e, 0.95))
    row = dict(window_weeks=L, aggregation=name, n=len(e), median=round(float(e.median()), 2),
               q95=round(q95[L], 2), max=round(float(e.max()), 2))
    row.update({f"false_flag_at_{t}": round(float((e > t).mean()), 2) for t in TAUS})
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
    ais = c[a:b].sum() / c[a0:b0].sum(); gap = abs(ais - off)
    rows.append(dict(period=name, published=pub, r_ais=round(ais, 3), r_official=round(off, 3), gap=round(gap, 2),
                     flag_at_0_25=bool(gap > 0.25), calibrated_tolerance=round(q95[L], 2),
                     flag_calibrated=bool(gap > q95[L])))
per = pd.DataFrame(rows); per.to_csv(OUT / "trust_periods.csv", index=False)
print(per.to_string(index=False))
json.dump(dict(stable_windows=STABLE, series="container_import_t (AIS container import tonnes)",
               calibrated_tolerance_q95={LS[L]: round(v, 2) for L, v in q95.items()}),
          open(OUT / "trust-summary.json", "w"), indent=2)
