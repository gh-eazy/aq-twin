"""AQ Twin - does a hand formula predict the simulated lane count?

  load = D x PMR / (24 x capacity),  capacity = 60 / mean service time = 60 / 1.833 = 32.7 trucks per lane-hour
  lanes = ceil(load)

D = trucks per day, PMR = peak-to-mean arrival ratio (1.8; under TAS at compliance c: (1 - c) x 1.8 + c).
Compared with the fewest lanes that meet the rule in sim_grid.csv (S1-S5, with and without TAS, 20 replications),
sim_sensitivity.csv (S3 at four peak ratios; its grid starts at 3 lanes) and gate_sensitivity.csv (24 h gate, TAS at
65% and 50% compliance, 10 replications). This is a check of the model against queueing arithmetic, not a validation.

Run after sim.py and gate_sens.py: python -P gate_formula.py   (writes results/gate_formula.csv)
"""
import json, math
from pathlib import Path
import pandas as pd

OUT = Path(__file__).parent / "results"
SERVICE = (0.5 + 1 + 4) / 3; CAP = 60 / SERVICE; LIMIT = 10.0
S = json.load(open(OUT / "sim-summary.json"))["scenarios"]
SHORT = lambda name: name.split()[0]


def fewest(df):
    ok = df[df.gate_wait_p95 + df.gate_wait_p95_ci <= LIMIT]
    return int(ok.lanes.min()) if len(ok) else None


rows = []
def add(case, source, d, pmr, tas, sim):
    load = d * pmr / (24 * CAP)
    rows.append(dict(case=case, source=source, per_day=round(d), pmr=round(pmr, 2), tas=tas, load=round(load, 2),
                     formula_lanes=math.ceil(load), sim_lanes=sim, match=math.ceil(load) == sim))


g = pd.read_csv(OUT / "sim_grid.csv")
for name, d in S.items():
    for tas, pmr in [(False, 1.8), (True, 0.2 * 1.8 + 0.8)]:
        add(SHORT(name), "grid", d, pmr, tas, fewest(g[(g.scenario == name) & (g.tas == tas)]))
sn = pd.read_csv(OUT / "sim_sensitivity.csv"); sn = sn[sn.kind == "peak"]
d3 = S["S3 9M-2026 average"]
for peak in (1.4, 1.8, 2.2, 2.6):
    x = sn[(sn.value == peak) & (~sn.tas)]
    add("S3", "peak sweep", d3, peak, False, fewest(x))
gs = pd.read_csv(OUT / "gate_sensitivity.csv")
for comp in (0.65, 0.5):
    for name, d in S.items():
        r = gs[(gs.scenario == name) & (gs.open_hours == 24) & (gs.compliance == comp)].iloc[0]
        add(SHORT(name), f"compliance {comp}", d, (1 - comp) * 1.8 + comp, True, int(r.lanes_tas))
out = pd.DataFrame(rows); out.to_csv(OUT / "gate_formula.csv", index=False)
print(f"capacity {CAP:.1f} trucks per lane-hour")
print(out.to_string(index=False))
print("matches:", int(out.match.sum()), "of", len(out))
