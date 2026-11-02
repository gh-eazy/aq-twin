# AQ Twin

Code, data snapshot and results for *AQ Twin: a public-data digital twin with foundation-model early warning, data-trust checking and gate simulation for the Port of Aqaba* (StudenTek 2026).

The twin is a closed loop on public data only:

1. IMF PortWatch AIS data for Aqaba (weekly, 2019-01-06 to 2026-10-18)
2. zero-shot Chronos forecasts (Chronos-Bolt-S, Chronos-Bolt-B, Chronos-2) with 80% bands, against seasonal-naive, MA8, damped ETS and gradient-boosting baselines
3. an early-warning alarm (one-step band and a band frozen 4 weeks ahead)
4. a data-trust check of the AIS signal against official throughput figures
5. a SimPy discrete-event simulation of the truck gate
6. a rule that picks the fewest gate lanes that keep the p95 wait under 10 minutes, with and without a truck appointment system (TAS)

## Data

`data/ports.csv` is a snapshot (metadata read 2026-11-01, daily rows to 2026-10-23) of the HDX dataset *Jordan: Daily Port Activity Data and Shipment Estimates* (UN Global Platform; IMF PortWatch):
https://data.humdata.org/dataset/eeb31578-9bef-4923-bc5a-77c13a06c3ad

`data/pkg.json` is the HDX package metadata for that snapshot. The official figure used by the trust check (ACT, July 2024: 76,000 TEU, +8% on July 2023, APM Terminals release) is written in `loop.py`.

## Run

Python 3.13. Install the dependencies, then run the scripts in this order from the repo folder:

```bash
pip install -r requirements.txt
python -P forecast.py   # accuracy (exp1) and one-step early warning (exp2)
python -P ew4.py        # early warning with a band frozen 4 weeks ahead (exp2b)
python -P sim.py        # gate simulation grid and sensitivity
python -P loop.py       # trust check and lane decisions
python -P figures.py    # figures/fig1-3
```

Each script writes to `results/` (CSV and JSON) or `figures/`. The committed `results/` are the values reported in the paper. Chronos weights download from Hugging Face on first run (`amazon/chronos-bolt-small`, `amazon/chronos-bolt-base`, `amazon/chronos-2`). Seeds are fixed; Chronos on CPU can differ in the last digit across machines.

## Files

| File | What it does |
|---|---|
| `forecast.py` | weekly series, rolling-origin back-test, MASE and 80% coverage, one-step alarms |
| `ew4.py` | alarms from a band frozen 4 weeks ahead |
| `sim.py` | SimPy gate model (lanes, service time, yard time, TAS compliance), 20 replications per case |
| `loop.py` | AIS vs official ratio check, then the lane rule for each demand scenario |
| `figures.py` | the three figures |
