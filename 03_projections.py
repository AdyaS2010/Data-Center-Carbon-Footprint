"""
03_projections.py - Carbon footprint projections through 2030.

Combines:
  - NREL facility historical energy and carbon trends (2016-2025)
  - IEA/EPRI published growth rates for global data center electricity
  - Three scenarios: business-as-usual, accelerated AI, green transition

Adds projection data to output/results.json.
"""

import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import MERGED_CSV, RESULTS_FILE, US_GRID_EMISSION_FACTOR, GLOBAL_GRID_EMISSION_FACTOR


def main():
    print("=" * 60)
    print("  STEP 3: Carbon Footprint Projections")
    print("=" * 60)

    df = pd.read_csv(MERGED_CSV, index_col=0, parse_dates=True)

    # ── Facility-level annual energy and carbon ──────────────────────────
    df["total_power_kw"] = df["it_power_kw"] + df["cooling_kw"] + df["hvac_kw"] + df["pump_kw"]
    df["year"] = df.index.year

    yearly = df.groupby("year").agg(
        mean_total_kw=("total_power_kw", "mean"),
        mean_it_kw=("it_power_kw", "mean"),
        mean_cooling_kw=("cooling_kw", "mean"),
        mean_pue=("pue", "mean"),
        hours=("total_power_kw", "count"),
    )
    # Keep only years with at least 6 months of data
    yearly = yearly[yearly["hours"] >= 4000]
    yearly["annual_energy_mwh"] = yearly["mean_total_kw"] * yearly["hours"] / 1000
    # CO2 tonnes = MWh * (kg CO2/kWh) * (1000 kWh/MWh) / (1000 kg/tonne) = MWh * ef
    yearly["annual_co2_tonnes"] = yearly["annual_energy_mwh"] * US_GRID_EMISSION_FACTOR

    print("\nFacility Historical Data:")
    print(yearly[["mean_total_kw", "annual_energy_mwh", "annual_co2_tonnes", "mean_pue"]].round(1).to_string())

    # ── Global projections (three scenarios) ─────────────────────────────
    # Base: IEA reports global DC electricity at ~415 TWh in 2024
    projection_years = list(range(2024, 2031))
    base_twh = 415

    # Growth rates per year
    growth = {
        "business_as_usual": 0.12,    # IEA baseline
        "accelerated_ai":    0.15,    # High AI demand scenario
        "green_transition":  0.08,    # Strong efficiency + renewables
    }

    # Grid emission factors (kg CO2/kWh) declining as renewables grow
    ef = {
        "business_as_usual": [0.49, 0.47, 0.45, 0.43, 0.41, 0.39, 0.37],
        "accelerated_ai":    [0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43],
        "green_transition":  [0.49, 0.44, 0.39, 0.34, 0.30, 0.26, 0.22],
    }

    projections = {}
    for scenario in growth:
        twh_list = []
        co2_list = []
        current = base_twh
        for i, year in enumerate(projection_years):
            if i > 0:
                current *= (1 + growth[scenario])
            twh_list.append(round(current, 1))
            # 1 TWh = 10^9 kWh; Mt = 10^9 kg; so Mt CO2 = TWh * ef(kg/kWh)
            co2_mt = current * ef[scenario][i]
            co2_list.append(round(co2_mt, 1))

        projections[scenario] = {
            "years": projection_years,
            "twh": twh_list,
            "co2_mt": co2_list,
        }
        print(f"\n  {scenario}:")
        for y, t, c in zip(projection_years, twh_list, co2_list):
            print(f"    {y}: {t:>7.0f} TWh  ->  {c:>6.1f} Mt CO2")

    # ── Save to results.json ─────────────────────────────────────────────
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    results["projections"] = projections
    results["facility_yearly"] = {
        str(int(year)): {
            "mean_total_kw": round(float(row["mean_total_kw"]), 1),
            "annual_energy_mwh": round(float(row["annual_energy_mwh"]), 1),
            "annual_co2_tonnes": round(float(row["annual_co2_tonnes"]), 1),
            "mean_pue": round(float(row["mean_pue"]), 3),
        }
        for year, row in yearly.iterrows()
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nProjections added to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
