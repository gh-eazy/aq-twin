"""AQ Twin - paper figures, drawn for a 3.3 in IEEE column at 300 dpi (text >= 8 pt).

fig1-warning.png  weekly container imports (PortWatch), Chronos-2 80% band frozen 4 weeks ahead (design B), alarms, crisis line
fig2-gate.png     p95 gate wait vs gate lanes, with and without TAS, for three demand scenarios
fig3-trust.png    July 2024 plans (AIS only, causal Q1-2024 official, oracle July official, 3 lanes no TAS): p95 gate wait at real demand

Run after forecast.py, ew4.py, sim.py and loop.py: python -P figures.py
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path(__file__).parent / "results"; OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, LIGHT, ORANGE, GREY = "#111827", "#1D4ED8", "#BFDBFE", "#EA580C", "#6B7280"
CRISIS = pd.Timestamp("2023-11-19")

# ---------- fig 1
b = pd.read_csv(R / "exp2b_bands_container_import_t.csv", parse_dates=["week"])
al = pd.read_csv(R / "exp2b_alarm_weeks.csv", parse_dates=["week"])
al = al[(al.series == "container_import_t") & (al.model == "Chronos-2") & (al.side == "down")]
fig, ax = plt.subplots(figsize=(3.3, 2.15), dpi=300)
ax.fill_between(b.week, b["Chronos-2_lo"] / 1e3, b["Chronos-2_hi"] / 1e3, color=LIGHT, lw=0, label="Chronos-2 80% band, 4 weeks ahead")
ax.plot(b.week, b.obs / 1e3, color=INK, lw=0.7, label="observed (AIS estimate)")
ax.scatter(al.week, [0] * len(al), marker="|", s=18, color=ORANGE, lw=1, label="alarm (low)", zorder=3)
ax.axvline(CRISIS, color=GREY, ls="--", lw=0.7); ax.text(CRISIS, 128, "19 Nov 2023 ", color=GREY, ha="right", fontsize=7)
ax.set_ylabel("container imports (kt/week)")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7, ncol=2, handlelength=1.2,
          columnspacing=0.8, borderaxespad=0.1)
ax.set_ylim(-3, 140)
fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig1-warning.png"); plt.close(fig)

# ---------- fig 2
g = pd.read_csv(R / "sim_grid.csv")
fig, ax = plt.subplots(figsize=(3.3, 2.45), dpi=300)
for sc, c in [("S1 Jul-2024 (calibration)", GREY), ("S4 Dec-2025 record", BLUE), ("S5 stress +30% on S4", ORANGE)]:
    for tas, ls in [(False, "-"), (True, "--")]:
        s = g[(g.scenario == sc) & (g.tas == tas) & (g.lanes <= 6) & (g.gate_wait_p95 >= 0.1)]   # < 0.1 min = no queue
        ax.plot(s.lanes, s.gate_wait_p95, color=c, ls=ls, marker="o" if tas else "s", ms=2.5,
                label=f"{sc.split(' (')[0]}{' + TAS' if tas else ''}")
ax.axhline(10, color=INK, lw=0.5, ls=":"); ax.text(6, 10, "10 min limit", ha="right", va="bottom", fontsize=7)
ax.set_yscale("log"); ax.set_ylim(0.1, 2000); ax.set_xticks(range(2, 7))
ax.set_xlabel("gate lanes"); ax.set_ylabel("p95 gate wait (min)")
ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.27), handlelength=2.4,
          columnspacing=1.0)
fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig2-gate.png", bbox_inches="tight", pad_inches=0.03); plt.close(fig)

# ---------- fig 3
t = pd.read_csv(R / "loop_trust.csv")
t = t[t.twin.isin(["blind (AIS only)", "causal (Q1-2024 official)", "oracle (Jul-2024 official)", "fixed 3 lanes"])].reset_index(drop=True)
name = {"blind (AIS only)": "AIS only", "causal (Q1-2024 official)": "Q1-2024 official",
        "oracle (Jul-2024 official)": "Jul-2024 official (oracle)", "fixed 3 lanes": "fixed plan"}
lab = [f"{name[r.twin]}" + (f": {int(r.planned_per_day):,}/day" if pd.notna(r.planned_per_day) else "")
       + f"\n→ {r.lanes} lanes{' + TAS' if r.tas else ', no TAS'}" for r in t.itertuples()]
fig, ax = plt.subplots(figsize=(3.3, 1.7), dpi=300)
ax.barh(lab, t.actual_gate_wait_p95, color=[ORANGE, ORANGE, BLUE, GREY], xerr=t.actual_gate_wait_p95_ci, error_kw={"lw": 0.6})
ax.axvline(10, color=INK, lw=0.5, ls=":")
for i, v in enumerate(t.actual_gate_wait_p95): ax.text(max(v + t.actual_gate_wait_p95_ci[i] + 2, 12), i, f"{v:.1f} min" if v >= 10 else f"{v:.2f} min", va="center")
ax.text(10.5, -0.6, "10 min limit", fontsize=7, va="center")
ax.set_xlim(0, 80); ax.set_xlabel(f"p95 gate wait (min) at {t.actual_per_day[0]:,} trucks/day")
ax.invert_yaxis()
fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig3-trust.png", bbox_inches="tight", pad_inches=0.03); plt.close(fig)
print("figures done")
