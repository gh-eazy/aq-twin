"""AQ Twin - how much planning margin flips the July 2024 plan, and how far each plan holds.

1. Margin sweep: the full loop.py rule (p95 + 95% CI half-width <= 10 min, 20 replications) applied to the causal
   plan demand (765 trucks/day) times 1 + margin; each selected plan is then run at the real July 2024 demand.
2. Flip point: 2 lanes without TAS on a fine demand grid, with seeds 0-19 and with independent seeds 100-119,
   by the rule (upper CI bound) and by the mean alone.
3. Reach: the highest demand on a grid at which each fallback plan still meets the rule.

Run after sim.py: python -P cliff.py   (writes results/margin_sweep.csv, loop_cliff.csv, plan_reach.csv)
"""
import json, math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent; OUT = ROOT / "results"
src = (ROOT / "sim.py").read_text(encoding="utf-8").split("# ---------- calibration")[0]
g = {"__file__": str(ROOT / "sim.py"), "__name__": "simcore"}; exec(src, g)
reps, run, BASE_DAY = g["reps"], g["run"], g["BASE_DAY"]
YARD_MEAN = json.load(open(OUT / "sim-summary.json"))["yard_mean_min"]
LIMIT, N = 10.0, 20
CAUSAL = BASE_DAY / 1.08 * 0.80


def decide(per_day):
    for lanes in range(2, 9):
        for tas in (False, True):
            r = reps(n=N, per_day=per_day, lanes=lanes, tas=tas, yard_mean=YARD_MEAN)
            if r["gate_wait_p95"] + r["gate_wait_p95_ci"] <= LIMIT:
                return lanes, tas, r


def seeded(per_day, lanes, tas, first):
    x = pd.Series([run(per_day, lanes, tas=tas, yard_mean=YARD_MEAN, seed=first + s)["gate_wait_p95"] for s in range(N)])
    return round(x.mean(), 2), round(1.96 * x.std(ddof=1) / math.sqrt(N), 2)


rows = []
for m in (0, 0.03, 0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.15, 0.20, 0.25, 0.35):
    d = CAUSAL * (1 + m); lanes, tas, r = decide(d)
    real = reps(n=N, per_day=BASE_DAY, lanes=lanes, tas=tas, yard_mean=YARD_MEAN)
    rows.append(dict(margin=m, plan_per_day=round(d), lanes=lanes, tas=tas, plan_p95=r["gate_wait_p95"], plan_ci=r["gate_wait_p95_ci"],
                     real_p95=real["gate_wait_p95"], real_ci=real["gate_wait_p95_ci"]))
    print(rows[-1])
pd.DataFrame(rows).to_csv(OUT / "margin_sweep.csv", index=False)

rows = []
for d in (765, 780, 790, 800, 805, 810, 815, 820, 825, 830, 840, 850, 860, 880, 900, 956, 1032):
    for first in (0, 100):
        p, ci = seeded(d, 2, False, first)
        rows.append(dict(per_day=d, margin_on_765=round(d / CAUSAL - 1, 3), seeds=f"{first}-{first + N - 1}", gate_wait_p95=p,
                         gate_wait_p95_ci=ci, meets_rule=p + ci <= LIMIT, mean_below_limit=p <= LIMIT))
        print(rows[-1])
pd.DataFrame(rows).to_csv(OUT / "loop_cliff.csv", index=False)

rows = []
for lanes, tas in [(2, True), (3, False), (3, True)]:
    for d in (1032, 1100, 1150, 1200, 1211, 1230, 1250, 1275, 1305, 1350, 1400, 1500, 1600, 1696):
        r = reps(n=N, per_day=d, lanes=lanes, tas=tas, yard_mean=YARD_MEAN)
        rows.append(dict(lanes=lanes, tas=tas, per_day=d, gate_wait_p95=r["gate_wait_p95"], gate_wait_p95_ci=r["gate_wait_p95_ci"],
                         turn_mean=r["turn_mean"], meets_rule=r["gate_wait_p95"] + r["gate_wait_p95_ci"] <= LIMIT))
pd.DataFrame(rows).to_csv(OUT / "plan_reach.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
