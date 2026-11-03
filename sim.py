"""AQ Twin - discrete-event simulation of the truck side of Aqaba Container Terminal, and the automated decision rule.

Truck -> entry gate (c lanes) -> in-terminal travel -> yard handling (k units) -> exit.
Parameters adapted from published studies (not calibrated to ACT data, which is not public):
  gate service TRIA(0.5, 1, 4) min  [Bett et al. 2024, citing Huynh 2009]
  in-terminal speed 18 km/h          [Bett et al. 2024]; 1 km each way is an ASSUMPTION
  truck visits per TEU 0.42          derived from APM: ~32,000 visits / 76,000 TEU, July 2024
  mean turnaround ~30 min            APM, July 2024 -> used to calibrate the yard handling time
Assumptions swept for sensitivity: gate lanes, peak factor, yard units.

Run: python -P sim.py   (writes results/sim_*.csv and results/sim-summary.json)
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd, simpy

ROOT = Path(__file__).parent; OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
VISITS_PER_TEU = 32000 / 76000
DAYS, WARMUP, REPS = 8, 1, 20
TRAVEL_MIN = 2 * 1.0 / 18 * 60            # 1 km in + 1 km out at 18 km/h
YARD_UNITS = 24                           # ASSUMPTION (swept)
TAS_COMPLIANCE = 0.8                      # ASSUMPTION: 80% of trucks book a slot


def profile(peak):
    """24 hourly weights (mean 1). Day-heavy shape 06-20 h, scaled so max/mean == peak."""
    h = np.arange(24)
    base = np.exp(-0.5 * ((h - 12.5) / 3.5) ** 2)
    w = 1 + (peak - 1) * base / base.max()
    for _ in range(50):                    # rescale so mean is 1 and max/mean stays == peak
        w = w / w.mean(); w = 1 + (w - 1) * (peak - 1) / (w.max() - 1)
    return w / w.mean()


def arrivals(rng, per_day, peak, tas, compliance=TAS_COMPLIANCE, open_hours=24):
    """Arrival times (min) over DAYS. TAS: compliant trucks spread evenly over the open-hour slots.
    open_hours < 24: the gate opens 06:00 for open_hours; all demand arrives while it is open and the
    non-booked trucks keep the day shape of profile() inside those hours (sensitivity runs only)."""
    w = profile(peak); times = []
    hours = range(24) if open_hours == 24 else range(6, 6 + open_hours)
    if open_hours != 24:
        w = w.copy(); w[list(hours)] = w[list(hours)] / w[list(hours)].mean()
    for day in range(DAYS):
        for hr in hours:
            lam_free = per_day / open_hours * (w[hr] * (1 - compliance) + compliance) if tas else per_day / open_hours * w[hr]
            n = rng.poisson(lam_free)
            times += list((day * 24 + hr) * 60 + rng.uniform(0, 60, n))
    return np.sort(times)


def run(per_day, lanes, peak=1.8, tas=False, yard_mean=None, yard_units=YARD_UNITS, seed=0,
        compliance=TAS_COMPLIANCE, open_hours=24):
    rng = np.random.default_rng(seed)
    env = simpy.Environment(); gate = simpy.Resource(env, lanes); yard = simpy.Resource(env, yard_units)
    rec = []

    def truck(t0):
        with gate.request() as r:
            yield r; gw = env.now - t0
            yield env.timeout(rng.triangular(0.5, 1, 4))
        yield env.timeout(TRAVEL_MIN)
        with yard.request() as r:
            yield r
            sigma = 0.5; mu = math.log(yard_mean) - sigma ** 2 / 2
            yield env.timeout(rng.lognormal(mu, sigma))
        if t0 >= WARMUP * 1440: rec.append((gw, env.now - t0))

    def source():
        for t in arrivals(rng, per_day, peak, tas, compliance, open_hours):
            yield env.timeout(t - env.now); env.process(truck(t))

    env.process(source()); env.run()
    a = np.array(rec)
    return dict(gate_wait_mean=a[:, 0].mean(), gate_wait_p95=np.percentile(a[:, 0], 95),
                turn_mean=a[:, 1].mean(), turn_p95=np.percentile(a[:, 1], 95), trucks=len(a))


def reps(n=REPS, **kw):
    rs = pd.DataFrame([run(seed=s, **kw) for s in range(n)])
    m = rs.mean(); ci = 1.96 * rs.std(ddof=1) / math.sqrt(n)
    return {**{k: round(v, 2) for k, v in m.items()}, **{k + "_ci": round(v, 2) for k, v in ci.items()}}


# ---------- calibration: yard handling mean so that turnaround == 30 min at July-2024 demand, 4 lanes, peak 1.8
BASE_DAY = 32000 / 31
lo, hi = 5.0, 30.0
for _ in range(14):
    mid = (lo + hi) / 2
    t = np.mean([run(BASE_DAY, 4, yard_mean=mid, seed=s)["turn_mean"] for s in range(6)])
    lo, hi = (mid, hi) if t < 30 else (lo, mid)
YARD_MEAN = round((lo + hi) / 2, 2)
print("calibrated yard handling mean (min):", YARD_MEAN)

# ---------- demand scenarios (truck visits/day), from opened official figures
SCEN = {
    "S1 Jul-2024 (calibration)": BASE_DAY,
    "S2 2023 average": 898000 / 12 * VISITS_PER_TEU / 30.4,
    "S3 9M-2026 average": 786948 / 9 * VISITS_PER_TEU / 30.4,
    "S4 Dec-2025 record": 96060 * VISITS_PER_TEU / 31,
    "S5 stress +30% on S4": 1.3 * 96060 * VISITS_PER_TEU / 31,
}
rows = []
for name, dem in SCEN.items():
    for lanes in range(2, 9):
        for tas in (False, True):
            r = reps(per_day=dem, lanes=lanes, tas=tas, yard_mean=YARD_MEAN)
            rows.append(dict(scenario=name, per_day=round(dem), lanes=lanes, tas=tas, **r))
    print("scenario", name, "done")
grid = pd.DataFrame(rows); grid.to_csv(OUT / "sim_grid.csv", index=False)

# ---------- sensitivity (S3 demand): peak factor and yard units
sens = []
for peak in (1.4, 1.8, 2.2, 2.6):
    for lanes in (3, 4, 5, 6):
        for tas in (False, True):
            sens.append(dict(kind="peak", value=peak, lanes=lanes, tas=tas,
                             **reps(per_day=SCEN["S3 9M-2026 average"], lanes=lanes, peak=peak, tas=tas, yard_mean=YARD_MEAN)))
for yu in (20, 24, 28):
    for tas in (False, True):
        sens.append(dict(kind="yard_units", value=yu, lanes=4, tas=tas,
                         **reps(per_day=SCEN["S3 9M-2026 average"], lanes=4, tas=tas, yard_mean=YARD_MEAN, yard_units=yu)))
pd.DataFrame(sens).to_csv(OUT / "sim_sensitivity.csv", index=False)
print("sensitivity done")

json.dump(dict(yard_mean_min=YARD_MEAN, visits_per_teu=round(VISITS_PER_TEU, 3), travel_min=round(TRAVEL_MIN, 2),
               yard_units=YARD_UNITS, tas_compliance=TAS_COMPLIANCE, reps=REPS, days=DAYS - WARMUP,
               scenarios={k: round(v) for k, v in SCEN.items()}),
          open(OUT / "sim-summary.json", "w"), indent=2)
print("sim done")
