"""AQ Twin - the closed loop: data-trust check -> demand -> automated gate decision -> simulated outcome.

Decision rule (automated): the fewest gate lanes, without TAS first and then with TAS, such that the
p95 gate wait <= 10 min. Lanes searched 2..8.

Trust experiment (July 2024):
  official   ACT July 2024 = 76,000 TEU, +8% on July 2023 (APM release) -> July 2023 = 76,000 / 1.08 TEU
  PortWatch  July 2024 / July 2023 ratio for container import tonnes and container port calls (AIS-derived)
  blind twin   believes the AIS ratio: plans for July-2023 demand x AIS ratio
  trusted twin cross-checks the AIS change against the official change; the gap is far above the
               tolerance, so it overrides AIS with the official level: plans for July-2024 demand
  both plans are then run under the real July-2024 demand.

Run after sim.py: python -P loop.py   (writes results/loop_*.csv and results/loop-summary.json)
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent; OUT = ROOT / "results"
# reuse the simulation model (everything above the calibration block in sim.py)
src = (ROOT / "sim.py").read_text(encoding="utf-8").split("# ---------- calibration")[0]
g = {"__file__": str(ROOT / "sim.py"), "__name__": "simcore"}; exec(src, g)
reps, VISITS_PER_TEU = g["reps"], g["VISITS_PER_TEU"]
S = json.load(open(OUT / "sim-summary.json")); YARD_MEAN = S["yard_mean_min"]
P95_LIMIT, TOLERANCE = 10.0, 0.25          # minutes; max allowed gap between AIS and official YoY change


def decide(per_day):
    """Automated decision: return (lanes, tas, result) for the cheapest plan that meets the limit."""
    for lanes in range(2, 9):
        for tas in (False, True):
            r = reps(per_day=per_day, lanes=lanes, tas=tas, yard_mean=YARD_MEAN)
            if r["gate_wait_p95"] <= P95_LIMIT:
                return lanes, tas, r
    return 8, True, r


# ---------- 1. decision per demand scenario (from sim.py)
rows = []
for name, dem in S["scenarios"].items():
    lanes, tas, r = decide(dem)
    rows.append(dict(scenario=name, per_day=dem, lanes=lanes, tas=tas, **r))
    print("decide", name, lanes, tas)
pd.DataFrame(rows).to_csv(OUT / "loop_decisions.csv", index=False)

# ---------- 2. trust experiment, July 2024
d = pd.read_csv(ROOT / "data" / "ports.csv", parse_dates=["date"])
d = d[d.portname == "Aqaba"].set_index("date").sort_index(); d.index = d.index.tz_localize(None)
m = d[["portcalls_container", "import_container"]].resample("MS").sum()
ais = {c: m.loc["2024-07-01", c] / m.loc["2023-07-01", c] for c in m.columns}
ais_ratio = ais["import_container"]                    # tonnes: the series the twin forecasts
official_ratio = 1.08
gap = abs(ais_ratio - official_ratio)
real_day = 76000 * VISITS_PER_TEU / 31                 # July 2024 truck visits/day (= sim BASE_DAY)
jul23_day = 76000 / official_ratio * VISITS_PER_TEU / 31
blind_day = jul23_day * ais_ratio
trusted_day = real_day if gap > TOLERANCE else blind_day

trust = []
for twin, plan_day in [("blind (AIS only)", blind_day), ("trust-checked", trusted_day)]:
    lanes, tas, planned = decide(plan_day)
    actual = reps(per_day=real_day, lanes=lanes, tas=tas, yard_mean=YARD_MEAN)
    trust.append(dict(twin=twin, planned_per_day=round(plan_day), lanes=lanes, tas=tas,
                      planned_p95=planned["gate_wait_p95"], actual_per_day=round(real_day),
                      **{"actual_" + k: v for k, v in actual.items()}))
    print("trust", twin, lanes, tas, actual["gate_wait_p95"])
pd.DataFrame(trust).to_csv(OUT / "loop_trust.csv", index=False)

json.dump(dict(p95_limit_min=P95_LIMIT, tolerance=TOLERANCE,
               ais_ratio_jul24_vs_jul23={k: round(v, 3) for k, v in ais.items()},
               official_ratio_jul24_vs_jul23=official_ratio, gap=round(gap, 3), flagged=bool(gap > TOLERANCE),
               jul23_per_day=round(jul23_day), blind_per_day=round(blind_day), real_per_day=round(real_day)),
          open(OUT / "loop-summary.json", "w"), indent=2)
print("loop done")
