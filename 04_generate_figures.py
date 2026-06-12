"""
04_generate_figures.py - Generate ALL figures for the paper.

Figures 1-3:  Literature review charts (from published data)
Figures 4-8:  Original NREL data analysis
Figure 9:     Carbon footprint projections
Figure 10:    AI energy comparison

Total: 10 figures, numbered sequentially as they appear in the paper.
"""

import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import MERGED_CSV, OUTPUT_DIR, RESULTS_FILE, COLORS, TEST_SIZE, RANDOM_STATE, setup_plot_style

plt = setup_plot_style()

BLUE    = COLORS["blue"]
RED     = COLORS["red"]
GREEN   = COLORS["green"]
ORANGE  = COLORS["orange"]
GRAY    = COLORS["gray"]
PURPLE  = COLORS["purple"]
TEAL    = COLORS["teal"]
LT_BLUE = COLORS["light_blue"]


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    kb = os.path.getsize(path) / 1024
    print(f"  Saved {name} ({kb:.0f} KB)")


def main():
    print("=" * 60)
    print("  STEP 4: Generate All Figures")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load analysis data
    df = pd.read_csv(MERGED_CSV, index_col=0, parse_dates=True)
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    print(f"Loaded {len(df):,} observations\n")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 1: Global DC Electricity Consumption (2015-2030)
    # ═══════════════════════════════════════════════════════════════════
    print("Figure 1: Global DC Electricity Trend...")
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    # Historical data points (IEA, various reports)
    hist_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    hist_twh =   [200,  215,  225,  235,  260,  260,  290,  340,  375,  415]

    # IEA projections
    proj_years = [2025, 2026, 2027, 2028, 2029, 2030]
    proj_low =   [465,  520,  585,  655,  735,  820]
    proj_high =  [500,  580,  680,  790,  910,  1050]
    proj_mid =   [(l + h) / 2 for l, h in zip(proj_low, proj_high)]

    ax.bar(hist_years, hist_twh, color=BLUE, width=0.7, label="Historical (IEA)", zorder=3)
    ax.bar(proj_years, proj_mid, color=LT_BLUE, width=0.7, label="Projected (IEA mid-range)", zorder=3)
    ax.fill_between(proj_years,
                    [l - 20 for l in proj_low],
                    [h + 20 for h in proj_high],
                    alpha=0.15, color=BLUE, label="Projection range")

    ax.axhline(y=945, color=RED, linestyle="--", lw=1.5, alpha=0.7)
    ax.text(2030.3, 945, "Japan's total\nelectricity (~945 TWh)", fontsize=7,
            color=RED, va="center", ha="left")

    ax.set_xlabel("Year")
    ax.set_ylabel("Electricity Consumption (TWh)")
    ax.set_title("Global Data Center Electricity Consumption (2015\u20132030)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(2014.2, 2031.5)
    ax.set_ylim(0, 1200)
    fig.tight_layout()
    save_fig(fig, "fig01_global_dc_electricity.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 2: AI Energy Amplification
    # ═══════════════════════════════════════════════════════════════════
    print("Figure 2: AI Energy Comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))

    # (a) Energy per query
    categories = ["Google\nSearch", "ChatGPT\nQuery", "AI Image\nGeneration"]
    wh_per_query = [0.3, 3.0, 15.0]  # Watt-hours (IEA, De Vries estimates)
    colors_a = [GREEN, ORANGE, RED]
    bars = ax1.bar(categories, wh_per_query, color=colors_a, width=0.5, edgecolor="white", lw=1.5)
    for bar, v in zip(bars, wh_per_query):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{v} Wh", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Energy per Query (Wh)")
    ax1.set_title("(a) Energy Cost per Query", fontweight="bold", fontsize=10)
    ax1.set_ylim(0, 20)

    # (b) GPU rack vs traditional rack power
    rack_types = ["Traditional\nServer Rack", "GPU Training\nRack (AI)"]
    rack_kw = [7, 70]
    colors_b = [BLUE, RED]
    bars2 = ax1_b = ax2.bar(rack_types, rack_kw, color=colors_b, width=0.4, edgecolor="white", lw=1.5)
    for bar, v in zip(bars2, rack_kw):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{v} kW", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Power per Rack (kW)")
    ax2.set_title("(b) Server Rack Power Density", fontweight="bold", fontsize=10)
    ax2.set_ylim(0, 90)

    fig.suptitle("The AI Energy Amplification Effect",
                 fontweight="bold", fontsize=11, y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig02_ai_energy_comparison.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 3: Embodied vs Operational Carbon
    # ═══════════════════════════════════════════════════════════════════
    print("Figure 3: Embodied vs Operational Carbon...")
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    servers = ["2015 Intel\nServer", "2024 ARM\nServer"]
    embodied = [345, 1024]
    # Operational over 5 years at US avg (0.39 kg/kWh), server ~500W avg
    # 2015: ~500W * 8760h * 5yr * 0.39 / 1000 = ~8,541 kg
    # 2024: ~600W * 8760h * 5yr * 0.39 / 1000 = ~10,249 kg
    operational = [8541, 10249]

    x = np.arange(len(servers))
    w = 0.3
    b1 = ax.bar(x - w/2, embodied, w, label="Embodied Carbon", color=ORANGE, edgecolor="white", lw=1.5)
    b2 = ax.bar(x + w/2, operational, w, label="Operational Carbon (5 yr)", color=BLUE, edgecolor="white", lw=1.5)

    for bar, v in zip(b1, embodied):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"{v:,}", ha="center", fontsize=8, fontweight="bold")
    for bar, v in zip(b2, operational):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"{v:,}", ha="center", fontsize=8, fontweight="bold")

    # Percentage labels
    for i, (e, o) in enumerate(zip(embodied, operational)):
        total = e + o
        pct = e / total * 100
        ax.text(i, -800, f"Embodied = {pct:.0f}%\nof total", ha="center", fontsize=7, style="italic")

    ax.set_ylabel("Carbon Emissions (kg CO₂e)")
    ax.set_xticks(x)
    ax.set_xticklabels(servers)
    ax.set_title("Embodied vs. Operational Carbon Over 5-Year Server Lifespan",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(-1200, 13000)
    fig.tight_layout()
    save_fig(fig, "fig03_embodied_vs_operational.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 4: PUE Timeseries
    # ═══════════════════════════════════════════════════════════════════
    print("Figure 4: PUE Timeseries...")
    fig, ax = plt.subplots(figsize=(6.5, 3))
    daily = df["pue"].resample("D").mean().dropna()
    ax.plot(daily.index, daily.values, lw=0.5, color=BLUE, alpha=0.5, label="Daily average")
    roll = daily.rolling(30, center=True).mean()
    ax.plot(roll.index, roll.values, lw=2, color=RED, label="30-day rolling mean")
    ax.set_ylabel("PUE")
    ax.set_xlabel("Date")
    ax.set_title("Data Center PUE Over Time (NREL ESIF, 2016\u20132025)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.tight_layout()
    save_fig(fig, "fig04_pue_timeseries.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 5: Temperature vs. Cooling Scatter
    # ═══════════════════════════════════════════════════════════════════
    print("Figure 5: Temperature vs. Cooling...")
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    feat = df[["outdoor_air_temp", "outdoor_air_humidity", "it_power_kw", "cooling_kw"]].dropna()
    X_all = feat[["outdoor_air_temp", "outdoor_air_humidity", "it_power_kw"]]
    y_all = feat["cooling_kw"]

    # Train models for this figure
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    lr = LinearRegression().fit(X_train, y_train)
    r2_lr = r2_score(y_test, lr.predict(X_test))

    sample = feat.sample(min(5000, len(feat)), random_state=RANDOM_STATE)
    sc = ax.scatter(
        sample["outdoor_air_temp"], sample["cooling_kw"],
        c=sample["it_power_kw"], alpha=0.3, s=10, cmap="viridis", edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="IT Power (kW)", shrink=0.8)

    temp_range = np.linspace(sample["outdoor_air_temp"].min(), sample["outdoor_air_temp"].max(), 100)
    hum_med = sample["outdoor_air_humidity"].median()
    it_med = sample["it_power_kw"].median()
    pred = lr.predict(np.column_stack([temp_range, np.full_like(temp_range, hum_med),
                                       np.full_like(temp_range, it_med)]))
    ax.plot(temp_range, pred, color=RED, lw=2.5, label=f"Linear fit (R\u00B2={r2_lr:.3f})")

    ax.set_xlabel("Outside Air Temperature (\u00B0F)")
    ax.set_ylabel("Cooling Power (kW)")
    ax.set_title("Ambient Temperature vs. Cooling Energy Demand",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "fig05_temp_vs_cooling.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 6: Seasonal Boxplots
    # ═══════════════════════════════════════════════════════════════════
    print("Figure 6: Seasonal Patterns...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))
    df["month"] = df.index.month

    season_colors = [
        "#4575B4", "#4575B4", "#74ADD1",
        "#ABD9E9", "#ABD9E9", "#F46D43",
        "#D73027", "#D73027", "#F46D43",
        "#ABD9E9", "#ABD9E9", "#4575B4",
    ]

    bp1 = ax1.boxplot(
        [df[df["month"] == m]["cooling_kw"].dropna() for m in range(1, 13)],
        labels=["J","F","M","A","M","J","J","A","S","O","N","D"],
        patch_artist=True, widths=0.6, showfliers=False,
    )
    for patch, color in zip(bp1["boxes"], season_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel("Cooling Power (kW)")
    ax1.set_xlabel("Month")
    ax1.set_title("(a) Monthly Cooling Energy", fontweight="bold", fontsize=10)

    bp2 = ax2.boxplot(
        [df[df["month"] == m]["pue"].dropna() for m in range(1, 13)],
        labels=["J","F","M","A","M","J","J","A","S","O","N","D"],
        patch_artist=True, widths=0.6, showfliers=False,
    )
    for patch, color in zip(bp2["boxes"], season_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("PUE")
    ax2.set_xlabel("Month")
    ax2.set_title("(b) Monthly PUE Distribution", fontweight="bold", fontsize=10)

    fig.suptitle("Seasonal Variation in Cooling Energy and PUE",
                 fontweight="bold", fontsize=11, y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig06_seasonal_patterns.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 7: Model Performance + Feature Importance
    # ═══════════════════════════════════════════════════════════════════
    print("Figure 7: Model Performance...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.5))

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
    ).fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)

    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, random_state=RANDOM_STATE
    ).fit(X_train, y_train)
    r2_gb = r2_score(y_test, gb.predict(X_test))

    model_names = ["Linear\nRegression", "Random\nForest", "Gradient\nBoosting"]
    r2_vals = [r2_lr, r2_rf, r2_gb]
    bars = ax1.bar(model_names, r2_vals, color=[BLUE, GREEN, ORANGE],
                   width=0.5, edgecolor="white", lw=1.5)
    for bar, v in zip(bars, r2_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("R\u00B2 Score")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("(a) Model Accuracy\n(Predicting Cooling kW)", fontweight="bold", fontsize=10)

    feat_labels = ["Temperature", "Humidity", "IT Power"]
    imps = list(rf.feature_importances_)
    bars2 = ax2.barh(feat_labels, imps, color=[RED, BLUE, GREEN],
                     height=0.5, edgecolor="white", lw=1.5)
    for bar, v in zip(bars2, imps):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1%}", va="center", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Feature Importance")
    ax2.set_title("(b) Random Forest\nFeature Importance", fontweight="bold", fontsize=10)
    ax2.set_xlim(0, 0.75)

    fig.suptitle("ML Model Performance: Predicting Data Center Cooling Energy",
                 fontweight="bold", fontsize=11, y=1.05)
    fig.tight_layout()
    save_fig(fig, "fig07_model_performance.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 8: Actual vs. Predicted
    # ═══════════════════════════════════════════════════════════════════
    print("Figure 8: Actual vs. Predicted...")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, y_pred_rf, alpha=0.15, s=6, color=GREEN, edgecolors="none")
    mn = min(y_test.min(), y_pred_rf.min())
    mx = max(y_test.max(), y_pred_rf.max())
    ax.plot([mn, mx], [mn, mx], "--", color=RED, lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Cooling Power (kW)")
    ax.set_ylabel("Predicted Cooling Power (kW)")
    ax.set_title(
        f"Random Forest: Actual vs. Predicted\n"
        f"(R\u00B2 = {r2_rf:.3f}, MAE = {mae_rf:.2f} kW)",
        fontweight="bold", fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    save_fig(fig, "fig08_actual_vs_predicted.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 9: Carbon Footprint Projections
    # ═══════════════════════════════════════════════════════════════════
    print("Figure 9: Carbon Projections...")
    proj = results.get("projections", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.5))

    if proj:
        styles = {
            "business_as_usual": (BLUE,   "-",  "Business as Usual"),
            "accelerated_ai":    (RED,    "--", "Accelerated AI"),
            "green_transition":  (GREEN,  "-.", "Green Transition"),
        }
        for scenario, (color, ls, label) in styles.items():
            data = proj[scenario]
            ax1.plot(data["years"], data["twh"], color=color, ls=ls, lw=2,
                     marker="o", markersize=4, label=label)
            ax2.plot(data["years"], data["co2_mt"], color=color, ls=ls, lw=2,
                     marker="o", markersize=4, label=label)

        ax1.set_ylabel("Electricity (TWh)")
        ax1.set_title("(a) Data Center Electricity", fontweight="bold", fontsize=10)
        ax1.legend(fontsize=7, loc="upper left")

        ax2.set_ylabel("CO₂ Emissions (Mt)")
        ax2.set_title("(b) Data Center Carbon Emissions", fontweight="bold", fontsize=10)
        ax2.legend(fontsize=7, loc="upper left")

        for a in [ax1, ax2]:
            a.set_xlabel("Year")
            a.grid(True, alpha=0.3)

    fig.suptitle("Global Data Center Carbon Footprint Projections (2024\u20132030)",
                 fontweight="bold", fontsize=11, y=1.05)
    fig.tight_layout()
    save_fig(fig, "fig09_carbon_projections.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 10: US Water Demand
    # ═══════════════════════════════════════════════════════════════════
    print("Figure 10: Water Demand...")
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    water_cats = ["2024\nCurrent", "2030\nLow Estimate", "NYC Daily\nSupply", "2030\nHigh Estimate"]
    water_mgd = [300, 697, 1000, 1451]
    water_colors = [BLUE, ORANGE, GRAY, RED]

    bars = ax.bar(water_cats, water_mgd, color=water_colors, width=0.5, edgecolor="white", lw=1.5)
    for bar, v in zip(bars, water_mgd):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 25,
                f"{v:,}\nMGD", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Water Demand (Million Gallons/Day)")
    ax.set_title("Projected U.S. Data Center Water Demand vs. NYC Daily Supply",
                 fontweight="bold", fontsize=10)
    ax.set_ylim(0, 1800)
    fig.tight_layout()
    save_fig(fig, "fig10_water_demand.png")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"All {len(os.listdir(OUTPUT_DIR))} figures saved to: {OUTPUT_DIR}")
    print("=" * 60)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".png"):
            kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
            print(f"  {f}: {kb:.0f} KB")


if __name__ == "__main__":
    main()
