"""AQ Twin - where does the 2-lane, no-TAS plan stop meeting the 10-min limit?

Runs 2 lanes without TAS at demands between the causal plan (765 trucks/day, Q1-2024 official ratio)
and the July 2024 real demand (1,032), with N_REPS = 10 replications per point.
Run after sim.py: python -P cliff.py   (writes results/loop_cliff.csv)
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent; OUT = ROOT / "results"
src = (ROOT / "sim.py").read_text(encoding="utf-8").split("# ---------- calibration")[0]
g = {"__file__": str(ROOT / "sim.py"), "__name__": "simcore"}; exec(src, g)
YARD_MEAN = json.load(open(OUT / "sim-summary.json"))["yard_mean_min"]
N_REPS = 10
rows = []
for d in [765, 800, 850, 900, 956, 1032]:
    r = g["reps"](n=N_REPS, per_day=d, lanes=2, tas=False, yard_mean=YARD_MEAN)
    rows.append(dict(per_day=d, lanes=2, tas=False, reps=N_REPS, gate_wait_p95=r["gate_wait_p95"],
                     gate_wait_p95_ci=r["gate_wait_p95_ci"], meets_limit=r["gate_wait_p95"] + r["gate_wait_p95_ci"] <= 10))
    print(rows[-1])
pd.DataFrame(rows).to_csv(OUT / "loop_cliff.csv", index=False)
