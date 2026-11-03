"""AQ Twin - early warning, extra baselines and checks (no Chronos re-run).

1. Damped ETS (forecast.py: additive damped trend, no seasonal term) in design A (one-step band) and
   design B (band frozen K=4 weeks ahead), with exactly the band and alarm rules of forecast.py / ew4.py.
   SeasonalNaive and MA8 are recomputed the same way and checked against the published CSVs.
2. 80% band coverage and below-band rate in the stable windows, per model and design. Chronos bands are
   read from results/exp2*_bands_*.csv (not recomputed).
3. Matched false-alarm rate: for each classic model, lower the low-side error quantile (0.10 -> 0) until the
   stable low-side false-alarm rate is <= that of Chronos-2 in the same series and design; report the delay.
4. Freeze length sweep K = 1, 2, 4, 6, 8 for the classic baselines only (Chronos is run at K = 4 only).
Classic forecasts are cached once per origin up to 8 weeks ahead.

Run: python -P ew5.py   (writes results/exp2c_*.csv)
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent; OUT = ROOT / "results"
SERIES = ["container_import_t", "cargo_import_t", "port_calls"]
CRISIS = pd.Timestamp("2023-11-19")
STABLE = [("2021-07-01", "2023-11-12"), ("2025-04-01", "2026-10-18")]     # as forecast.py
YRS = sum((pd.Timestamp(b) - pd.Timestamp(a)).days for a, b in STABLE) / 365.25
HMAX = 8
W = pd.read_csv(OUT / "weekly.csv", parse_dates=["date"]).set_index("date")


def in_stable(ts): return any(pd.Timestamp(a) <= ts <= pd.Timestamp(b) for a, b in STABLE)


# same forecasters as forecast.py
def snaive(y, h): return y[-52:-52 + h]
def ma8(y, h): return np.repeat(y[-8:].mean(), h)
def ets(y, h):
    m = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=None).fit()
    return np.maximum(m.forecast(h), 0)
MODELS = {"SeasonalNaive": snaive, "MA8": ma8, "ETS": ets}


def alarms(flag, weeks):
    return weeks[np.convolve(flag.astype(int), [1, 1], "full")[:len(flag)] == 2]


def score(obs, lo, hi, weeks):
    out = {}
    for side, flag in [("down", obs < lo), ("up", obs > hi)]:
        fire = alarms(flag, weeks); after = [w for w in fire if w >= CRISIS]
        after = after if side == "down" else []                 # delay is a low-side measure, as in forecast.py
        out[side] = dict(first=str(after[0].date()) if after else "",
                         delay=round((after[0] - CRISIS).days / 7, 1) if after else np.nan,
                         in_crisis=bool(after and after[0] < pd.Timestamp(STABLE[1][0])),
                         fa=round(sum(in_stable(w) for w in fire) / YRS, 2),
                         in13=bool(after and (after[0] - CRISIS).days / 7 <= 13),
                         crisis_alarms=sum(CRISIS <= w < pd.Timestamp(STABLE[1][0]) for w in fire))
    return out


ew, cov, matched = [], [], []
for s in SERIES:
    y = W[s].values; idx = W.index; n = len(y)
    F = {m: {j: f(y[:j + 1], HMAX) for j in range(51, n)} for m, f in MODELS.items()}   # cached forecasts
    a0 = idx.get_loc(idx[idx >= "2021-01-03"][0])
    designs = {}
    for m in MODELS:
        # design A: forecast.py residual_band (one-step, last 52 one-step errors)
        org = list(range(a0, n - 1))
        errs = [np.array([y[j + 1] - F[m][j][0] for j in range(i - 52, i)]) for i in org]
        c = np.array([F[m][i][0] for i in org])
        designs[(m, "A", 1)] = (org, 1, c, errs)
        for K in (1, 2, 4, 6, 8):   # design B: ew4.py band_k
            org = list(range(a0 - K + 1, n - K))
            errs = [np.array([y[j + K] - F[m][j][K - 1] for j in range(max(i - 52 - K, 51), i - K + 1)]) for i in org]
            c = np.array([F[m][i][K - 1] for i in org])
            designs[(m, "B", K)] = (org, K, c, errs)
    for (m, dsg, K), (org, k, c, errs) in designs.items():
        obs = np.array([y[i + k] for i in org]); weeks = idx[[i + k for i in org]]
        lo = c + np.array([np.quantile(e, 0.1) for e in errs]); hi = c + np.array([np.quantile(e, 0.9) for e in errs])
        sc = score(obs, lo, hi, weeks)
        for side in sc:
            ew.append(dict(series=s, model=m, design=dsg, K=K, side=side, first_alarm_after_crisis=sc[side]["first"],
                           delay_weeks=sc[side]["delay"], false_alarms_per_year=sc[side]["fa"],
                           alarm_within_13_weeks=sc[side]["in13"], low_alarms_crisis_period=sc[side]["crisis_alarms"]))
        if dsg == "A" or K == 4:
            st = np.array([in_stable(w) for w in weeks])
            cov.append(dict(series=s, model=m, design=dsg, cov80=round(float(np.mean(((obs >= lo) & (obs <= hi))[st])), 3),
                            below=round(float(np.mean((obs < lo)[st])), 3), above=round(float(np.mean((obs > hi)[st])), 3)))
            # matched false-alarm rate against Chronos-2 (same series and design)
            ref = pd.read_csv(OUT / f"{'exp2' if dsg == 'A' else 'exp2b'}_early_warning.csv")
            target = float(ref.query("series == @s and model == 'Chronos-2' and side == 'down'").false_alarms_per_year.iloc[0])
            for q in (0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0):
                lo_q = c + np.array([np.quantile(e, q) for e in errs]); sq = score(obs, lo_q, hi, weeks)["down"]
                if sq["fa"] <= target: break
            matched.append(dict(series=s, model=m, design=dsg, chronos2_fa=target, low_quantile=q, fa=sq["fa"],
                                reached=bool(sq["fa"] <= target), delay_weeks=sq["delay"], first_alarm_after_crisis=sq["first"],
                                first_alarm_before_2025_04_01=sq["in_crisis"]))
    print("ew5", s, "done")

# Chronos coverage from the saved bands (Experiment 2 CSVs)
for dsg, pre in [("A", "exp2"), ("B", "exp2b")]:
    for s in SERIES:
        b = pd.read_csv(OUT / f"{pre}_bands_{s}.csv", parse_dates=["week"])
        st = b.week.map(in_stable).values
        for m in ["SeasonalNaive", "MA8", "Chronos-Bolt-S", "Chronos-Bolt-B", "Chronos-2"]:
            lo, hi, o = b[f"{m}_lo"].values, b[f"{m}_hi"].values, b.obs.values
            cov.append(dict(series=s, model=m + ("" if "Chronos" in m else " (saved)"), design=dsg,
                            cov80=round(float(np.mean(((o >= lo) & (o <= hi))[st])), 3),
                            below=round(float(np.mean((o < lo)[st])), 3), above=round(float(np.mean((o > hi)[st])), 3)))

ew = pd.DataFrame(ew); ew.to_csv(OUT / "exp2c_early_warning.csv", index=False)
cov = pd.DataFrame(cov); cov.to_csv(OUT / "exp2c_coverage.csv", index=False)
pd.DataFrame(matched).to_csv(OUT / "exp2c_matched.csv", index=False)

# reproduction check: SeasonalNaive and MA8 must match the published tables
for dsg, pre in [("A", "exp2"), ("B", "exp2b")]:
    ref = pd.read_csv(OUT / f"{pre}_early_warning.csv"); kk = 1 if dsg == "A" else 4
    for m in ["SeasonalNaive", "MA8"]:
        for s in SERIES:
            for side in ("down", "up"):
                a = ew.query("series == @s and model == @m and design == @dsg and K == @kk and side == @side").iloc[0]
                r = ref.query("series == @s and model == @m and side == @side").iloc[0]
                same = a.false_alarms_per_year == r.false_alarms_per_year and (
                    (pd.isna(a.delay_weeks) and pd.isna(r.delay_weeks)) or a.delay_weeks == r.delay_weeks)
                if not same: print("MISMATCH", dsg, m, s, side, a.delay_weeks, r.delay_weeks, a.false_alarms_per_year, r.false_alarms_per_year)
print("reproduction check done")
print(ew.query("side == 'down'").to_string(index=False))
print(cov.to_string(index=False))
print(pd.DataFrame(matched).to_string(index=False))
