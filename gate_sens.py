"""AQ Twin - does the TAS benefit survive a shorter gate day and lower booking compliance?

For each demand scenario S1-S5 (sim-summary.json) and each gate regime, find the fewest lanes that meet
the loop.py rule (p95 gate wait + 95% CI half-width <= 10 min) without TAS and with TAS.
Regimes: 24 h gate with compliance 0.8 (base), 0.65, 0.5; 16 h gate (06-22 h) with compliance 0.8, 0.5.
16 h gate: all daily demand arrives while the gate is open (sim.arrivals, open_hours=16).
Uses N_REPS = 10 replications per point (the main grid uses 20) to keep the run time short.

Run after sim.py: python -P gate_sens.py   (writes results/gate_sensitivity.csv)
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent; OUT = ROOT / "results"
src = (ROOT / "sim.py").read_text(encoding="utf-8").split("# ---------- calibration")[0]
g = {"__file__": str(ROOT / "sim.py"), "__name__": "simcore"}; exec(src, g)
reps = g["reps"]
S = json.load(open(OUT / "sim-summary.json")); YARD_MEAN = S["yard_mean_min"]
P95_LIMIT, N_REPS = 10.0, 10
REGIMES = [(24, 0.8), (24, 0.65), (24, 0.5), (16, 0.8), (16, 0.5)]


def fewest(per_day, tas, open_hours, compliance):
    for lanes in range(2, 13):
        r = reps(n=N_REPS, per_day=per_day, lanes=lanes, tas=tas, yard_mean=YARD_MEAN,
                 open_hours=open_hours, compliance=compliance)
        if r["gate_wait_p95"] + r["gate_wait_p95_ci"] <= P95_LIMIT:
            return lanes, r
    return None, r


rows, no_tas = [], {}
for name, dem in S["scenarios"].items():
    for oh, comp in REGIMES:
        if oh not in no_tas:
            no_tas[oh] = fewest(dem, False, oh, comp)
        l0, r0 = no_tas[oh]; l1, r1 = fewest(dem, True, oh, comp)
        rows.append(dict(scenario=name, per_day=dem, open_hours=oh, compliance=comp, reps=N_REPS,
                         lanes_no_tas=l0, p95_no_tas=r0["gate_wait_p95"], lanes_tas=l1, p95_tas=r1["gate_wait_p95"],
                         lanes_saved=(l0 - l1) if l0 and l1 else None))
        print(rows[-1])
    no_tas = {}
pd.DataFrame(rows).to_csv(OUT / "gate_sensitivity.csv", index=False)
print("gate sensitivity done")
