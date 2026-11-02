"""AQ Twin - AI forecasting and early-warning experiments on IMF PortWatch data for Aqaba.

Run: python -P forecast.py   (writes results/*.csv and results/forecast-summary.json)
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
torch.manual_seed(0); np.random.seed(0)
ROOT = Path(__file__).parent
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)

SERIES = {"container_import_t": "import_container", "cargo_import_t": "import", "port_calls": "portcalls"}
CRISIS = pd.Timestamp("2023-11-19")              # Galaxy Leader seized (first attack)
STABLE = [("2021-07-01", "2023-11-12"), ("2025-04-01", "2026-10-18")]  # windows used for accuracy + false alarms
H = 4                                            # forecast horizon, weeks
Q = [0.05, 0.1, 0.5, 0.9, 0.95]

d = pd.read_csv(ROOT / "data" / "ports.csv", parse_dates=["date"])
d = d[d.portname == "Aqaba"].set_index("date").sort_index()
d.index = d.index.tz_localize(None)
# weekly sums; drop the last partial week (data ends 2026-10-23, a Friday)
W = d[list(SERIES.values())].resample("W-SUN").sum()
W = W[W.index <= "2026-10-18"]
W.columns = list(SERIES.keys())
W.to_csv(OUT / "weekly.csv")


def in_stable(ts):
    return any(pd.Timestamp(a) <= ts <= pd.Timestamp(b) for a, b in STABLE)


# ---------- forecasters: f(history np.array, h) -> (point[h], quantiles[h, len(Q)] or None)
def naive(y, h): return np.repeat(y[-1], h), None
def snaive(y, h): return y[-52:-52 + h] if h < 52 else None, None
def ma8(y, h): return np.repeat(y[-8:].mean(), h), None


def ets(y, h):
    m = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=None).fit()
    return np.maximum(m.forecast(h), 0), None


def gbm(y, h, lags=(1, 2, 3, 4, 8, 13, 26, 52)):
    L = max(lags); X, T = [], []
    for t in range(L, len(y) - h + 1):
        for k in range(h):
            X.append([y[t - l] for l in lags] + [k]); T.append(y[t + k])
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0).fit(np.array(X), np.array(T))
    x = [y[len(y) - l] for l in lags]
    return np.maximum(m.predict(np.array([x + [k] for k in range(h)])), 0), None


CLASSIC = {"Naive": naive, "SeasonalNaive": snaive, "MA8": ma8, "ETS": ets, "GBM": gbm}


def load_chronos():
    from chronos import BaseChronosPipeline
    return {n: BaseChronosPipeline.from_pretrained(m, device_map="cpu", torch_dtype=torch.float32)
            for n, m in [("Chronos-Bolt-S", "amazon/chronos-bolt-small"),
                         ("Chronos-Bolt-B", "amazon/chronos-bolt-base"),
                         ("Chronos-2", "amazon/chronos-2")]}


def chronos_batch(pipe, contexts, h):
    """Zero-shot quantile forecasts for many contexts -> (n, h, len(Q))."""
    out = []
    for i in range(0, len(contexts), 64):
        chunk = [torch.tensor(c, dtype=torch.float32) for c in contexts[i:i + 64]]
        q, _ = pipe.predict_quantiles(chunk, prediction_length=h, quantile_levels=Q)
        if isinstance(q, list):            # Chronos-2 returns a list of (1, h, q)
            q = torch.stack([x[0] for x in q])
        out.append(q.float().numpy())
    return np.maximum(np.concatenate(out), 0)


CHRONOS = load_chronos()

# ---------- Experiment 1: rolling-origin accuracy, horizon 1..4 weeks, stable windows only (skipped if done)
rows = []
for s in ([] if (OUT / "exp1_accuracy.csv").exists() else SERIES):
    y = W[s].values; idx = W.index
    origins = [i for i in range(156, len(y) - H) if in_stable(idx[i]) and in_stable(idx[i + H])]
    scale = np.mean(np.abs(np.diff(y[:origins[0]])))           # MASE scale: in-sample naive-1 MAE
    truth = np.array([y[i + 1:i + 1 + H] for i in origins])
    preds = {n: np.array([f(y[:i + 1], H)[0] for i in origins]) for n, f in CLASSIC.items()}
    for n, p in CHRONOS.items():
        q = chronos_batch(p, [y[:i + 1] for i in origins], H)
        preds[n] = q[:, :, Q.index(0.5)]
        lo, hi = q[:, :, Q.index(0.1)], q[:, :, Q.index(0.9)]
        rows.append(dict(series=s, model=n, metric="cov80", value=float(np.mean((truth >= lo) & (truth <= hi)))))
    for n, p in preds.items():
        rows.append(dict(series=s, model=n, metric="MASE", value=float(np.mean(np.abs(p - truth)) / scale)))
    print("exp1", s, len(origins), "origins")
acc = pd.DataFrame(rows) if rows else pd.read_csv(OUT / "exp1_accuracy.csv")
acc.to_csv(OUT / "exp1_accuracy.csv", index=False)
print(acc.pivot_table(index=["model"], columns=["series", "metric"], values="value").round(3).to_string())

# ---------- Experiment 2: early warning. One-step-ahead 80% band (Chronos-Bolt is trained on 0.1-0.9 only, so all models use 10/90%); alarm = 2 consecutive weeks outside the band.
def residual_band(y, i, f, win=52):
    """Empirical 10/90% band from the last `win` one-step errors of a classic forecaster."""
    errs = [y[j + 1] - f(y[:j + 1], 1)[0][0] for j in range(i - win, i)]
    c = f(y[:i + 1], 1)[0][0]
    return c + np.quantile(errs, 0.1), c, c + np.quantile(errs, 0.9)


ew_rows, alarms = [], []      # start 2021-01-03: >= 52 past one-step errors for every model (SeasonalNaive needs 52 weeks of history)
start = W.index.get_loc(W.index[W.index >= "2021-01-03"][0])
for s in SERIES:
    y = W[s].values; idx = W.index
    origins = list(range(start, len(y) - 1))
    bands = {}
    for n in ["SeasonalNaive", "MA8"]:
        bands[n] = np.array([residual_band(y, i, CLASSIC[n]) for i in origins])
    for n, p in CHRONOS.items():
        q = chronos_batch(p, [y[:i + 1] for i in origins], 1)[:, 0, :]
        bands[n] = np.stack([q[:, Q.index(0.1)], q[:, Q.index(0.5)], q[:, Q.index(0.9)]], 1)
    obs = np.array([y[i + 1] for i in origins]); weeks = idx[[i + 1 for i in origins]]
    for n, b in bands.items():
        low, high = obs < b[:, 0], obs > b[:, 2]
        for side, flag in [("down", low), ("up", high)]:
            run = np.convolve(flag.astype(int), [1, 1], "full")[:len(flag)] == 2   # 2nd consecutive week
            fire = weeks[run]
            for wk in fire: alarms.append(dict(series=s, model=n, side=side, week=wk.date()))
            after = [w for w in fire if w >= CRISIS]
            stable_fires = [w for w in fire if in_stable(w)]
            yrs = sum((pd.Timestamp(b_) - pd.Timestamp(a_)).days for a_, b_ in STABLE) / 365.25
            ew_rows.append(dict(series=s, model=n, side=side,
                                first_alarm_after_crisis=str(after[0].date()) if after and side == "down" else "",
                                delay_weeks=round((after[0] - CRISIS).days / 7, 1) if after and side == "down" else np.nan,
                                false_alarms_per_year=round(len(stable_fires) / yrs, 2)))
    pd.DataFrame(dict(week=weeks, obs=obs, **{f"{n}_lo": b[:, 0] for n, b in bands.items()},
                      **{f"{n}_hi": b[:, 2] for n, b in bands.items()})).to_csv(OUT / f"exp2_bands_{s}.csv", index=False)
    print("exp2", s, "done")
ew = pd.DataFrame(ew_rows); ew.to_csv(OUT / "exp2_early_warning.csv", index=False)
pd.DataFrame(alarms).to_csv(OUT / "exp2_alarm_weeks.csv", index=False)
print(ew.to_string())

# ---------- Data-trust check: PortWatch vs official figures (ratios only; official numbers from opened sources)
q = W.resample("QE").sum()
summary = dict(
    weeks=len(W), first_week=str(W.index[0].date()), last_week=str(W.index[-1].date()),
    pw_container_import_Q1_2024_vs_Q1_2023=round(q.loc["2024-03-31", "container_import_t"] / q.loc["2023-03-31", "container_import_t"] - 1, 3),
    pw_port_calls_Q1_2024_vs_Q1_2023=round(q.loc["2024-03-31", "port_calls"] / q.loc["2023-03-31", "port_calls"] - 1, 3),
    pw_container_calls_JanAug_2024_vs_2023=round(
        d.loc["2024-01-01":"2024-08-31", "portcalls_container"].sum() / d.loc["2023-01-01":"2023-08-31", "portcalls_container"].sum() - 1, 3),
    pw_cargo_import_JanAug_2024_vs_2023=round(
        d.loc["2024-01-01":"2024-08-31", "import"].sum() / d.loc["2023-01-01":"2023-08-31", "import"].sum() - 1, 3),
    pw_total_cargo_JanAug_2024_vs_2023=round(
        (d.loc["2024-01-01":"2024-08-31", ["import", "export"]].sum().sum()) /
        (d.loc["2023-01-01":"2023-08-31", ["import", "export"]].sum().sum()) - 1, 3),
    pw_container_calls_2023=int(d.loc["2023", "portcalls_container"].sum()),
    pw_container_calls_2024=int(d.loc["2024", "portcalls_container"].sum()),
    pw_container_calls_2025=int(d.loc["2025", "portcalls_container"].sum()),
)
json.dump(summary, open(OUT / "forecast-summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
