# Data Center Carbon Footprint: NREL ESIF Analysis

Code and data for *Carbon Footprint of Cloud Computing: The Hidden Environmental Cost of Digital Infrastructure* (A. Sastry).

The analysis uses the NREL Energy Systems Integration Facility (ESIF) high-performance computing data center dataset (Clark & Strelka, OEDI submission 8616, CC BY 4.0).

## Reproducing the results

```bash
pip install pandas pyarrow numpy scipy scikit-learn matplotlib
python run_all.py
```

### Raw data

Download the two Parquet files from https://data.openei.org/submissions/8616 and save them as:

| File on OEDI | Save as |
|---|---|
| `esif.influx.buildingData.PUE.combined.parquet` | `data/esif_pue_raw.parquet` |
| `esif.influx.buildingData.outside.combined_2.parquet` | `data/esif_weather_raw.parquet` |

The raw files (~170 MB) are not committed. `data/world_grid_carbon_intensity_ember.csv` is the world lifecycle carbon intensity of electricity (Ember, via Our World in Data, https://ourworldindata.org/grapher/carbon-intensity-electricity), used as the historical benchmark for the Business as Usual scenario. The cleaned hourly file `esif_hourly_merged.csv` is committed, so `02`–`04` can run without them.

## Pipeline

| Script | What it does |
|---|---|
| `01_prepare_data.py` | Filters invalid one-minute readings, aggregates **all** readings to hourly means, and keeps hours with at least 50% of their readings. Then it joins power and weather data and recomputes hourly PUE. It also rebuilds the original every-10th-row version for a sensitivity check. |
| `02_analysis.py` | Descriptive statistics and Pearson correlations with Fisher and moving-block bootstrap 95% CIs. Fits Models A/B/C with Linear Regression, Random Forest and Gradient Boosting under three validation schemes, and checks feature-importance stability, sampling sensitivity and the April 2024 subsystem change. |
| `03_projections.py` | 2024–2030 scenarios built from one baseline (415 TWh, 0.445 kg CO₂/kWh), with a year-by-year emission-factor table and a sensitivity table. |
| `04_generate_figures.py` | All figures. The ESIF figures are drawn from saved results so they match the reported numbers. |

Outputs: `output/results.json`, `output/tables/*.csv`, `output/figures/*.png`.

## Changes in the NHSJS revision

| Reviewer comment | Change |
|---|---|
| Two 2024 baselines (415 vs 460 TWh) | One baseline: 415 TWh × 0.445 kg/kWh ≈ 185 Mt. The old code used 0.49 kg/kWh even though the paper stated 0.445. |
| Emission-factor path not given | `EF(t) = 0.445 × (1 − d)^(t−2024)` with *d* = 1.5% (Business as Usual, equal to the 2015–2025 decline in world grid carbon intensity from Ember data in `data/world_grid_carbon_intensity_ember.csv`), 1% (Accelerated AI) and 11%/yr (Green Transition). Year-by-year values are in `projections_by_year.csv`; scenario assumptions in `scenario_assumptions.csv`. |
| Meaning of "55%" | The reduction in grid intensity is now calculated and reported against 2024 (`ef_2030_reduction_vs_2024_pct`). |
| Random 80/20 split on time series | Adds a chronological 80/20 split and rolling-origin validation by year (2019–2025). The random split is kept only for comparison. |
| Feature-importance robustness | Impurity and permutation importance, mean ± SD across rolling folds and 10 repeated random splits. |
| Gradient Boosting missing for Model A | All three algorithms are now run for Models A, B and C. |
| Correlation uncertainty | Fisher and one-week moving-block bootstrap CIs, r², and approximate effective sample size. |
| Every-10th-row sampling | Removed. All readings are aggregated directly; the old method is kept only as a sensitivity check. |
| PUE outlier range 0.95–1.50 | Lower bound is now 1.00, applied to one-minute readings before aggregation. |
| Overhead definition | Plug and light load is now included in overhead, matching how NREL computes PUE. |
| Figure 3 vs ">50% embodied" | Figure 3 now adds an embodied-share vs grid-intensity panel with break-even intensities. Server power assumptions are in `config.py`. |
| PUE rise in mid-2024 | Figure 5b breaks overhead down by subsystem. The rise follows a step increase in HVAC power in April 2024 (`regime_change` in results). |
| Figures 7, 8, 10 wording | Titles now say *power* (kW), not *energy*. Figure 10 shows both random and chronological test sets. |
| Figure 2 values | The ranges now reflect published estimates (0.3–2.9 Wh per LLM query; ~2.9 Wh per image). |

The raw timestamps are UTC. Month and hour of day are converted to Golden, CO local time (`America/Denver`).
