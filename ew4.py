"""AQ Twin - early warning, design B: compare each week with the 80% band forecast 4 weeks earlier.

Design A (forecast.py) uses a one-step band, which follows a slow decline. Design B freezes the
forecast at t-4, so a fall that builds over weeks lands outside the band. Same alarm rule:
2 consecutive weeks below (or above) the band. Both designs are reported.

Run: python -P ew4.py   (writes results/exp2b_early_warning.csv, exp2b_alarm_weeks.csv, exp2b_bands_<series>.csv)
"""
from pathlib import Path
src = (Path(__file__).parent / "forecast.py").read_text(encoding="utf-8")
# reuse data, models and helpers from forecast.py (everything above Experiment 1)
exec(src.split("# ---------- Experiment 1")[0], g := {"__file__": str(Path(__file__).parent / "forecast.py"), "__name__": "fc"})
np, pd, W, Q, CHRONOS, CLASSIC, CRISIS, STABLE, OUT, SERIES, in_stable, chronos_batch = (
    g[k] for k in "np pd W Q CHRONOS CLASSIC CRISIS STABLE OUT SERIES in_stable chronos_batch".split())
K = 4


def band_k(y, i, f, win=52):
    """Classic: forecast at origin i for week i+K, with the 10/90% of its last `win` K-step errors."""
    errs = [y[j + K] - f(y[:j + 1], K)[0][K - 1] for j in range(max(i - win - K, 51), i - K + 1)]   # SeasonalNaive needs 52 weeks of history
    c = f(y[:i + 1], K)[0][K - 1]
    return c + np.quantile(errs, 0.1), c + np.quantile(errs, 0.9)


rows, alarms = [], []
yrs = sum((pd.Timestamp(b) - pd.Timestamp(a)).days for a, b in STABLE) / 365.25
for s in SERIES:
    y = W[s].values; idx = W.index
    start = idx.get_loc(idx[idx >= "2021-01-03"][0]) - K + 1      # first target week 2021-01-03, as in design A
    origins = list(range(start, len(y) - K))
    bands = {n: np.array([band_k(y, i, CLASSIC[n]) for i in origins]) for n in ["SeasonalNaive", "MA8"]}
    for n, p in CHRONOS.items():
        q = chronos_batch(p, [y[:i + 1] for i in origins], K)[:, K - 1, :]
        bands[n] = np.stack([q[:, Q.index(0.1)], q[:, Q.index(0.9)]], 1)
    obs = np.array([y[i + K] for i in origins]); weeks = idx[[i + K for i in origins]]
    pd.DataFrame(dict(week=weeks, obs=obs, **{f"{n}_lo": b[:, 0] for n, b in bands.items()},
                      **{f"{n}_hi": b[:, 1] for n, b in bands.items()})).to_csv(OUT / f"exp2b_bands_{s}.csv", index=False)
    for n, b in bands.items():
        for side, flag in [("down", obs < b[:, 0]), ("up", obs > b[:, 1])]:
            fire = weeks[np.convolve(flag.astype(int), [1, 1], "full")[:len(flag)] == 2]
            alarms += [dict(series=s, model=n, side=side, week=w.date()) for w in fire]
            after = [w for w in fire if w >= CRISIS]
            rows.append(dict(series=s, model=n, side=side,
                             first_alarm_after_crisis=str(after[0].date()) if after and side == "down" else "",
                             delay_weeks=round((after[0] - CRISIS).days / 7, 1) if after and side == "down" else np.nan,
                             false_alarms_per_year=round(sum(in_stable(w) for w in fire) / yrs, 2)))
    print("exp2b", s, "done")
ew = pd.DataFrame(rows); ew.to_csv(OUT / "exp2b_early_warning.csv", index=False)
pd.DataFrame(alarms).to_csv(OUT / "exp2b_alarm_weeks.csv", index=False)
print(ew.to_string())
