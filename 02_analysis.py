"""
02_analysis.py - Correlation analysis and machine learning models.

Performs:
  - Pearson correlation with p-values
  - Model A: PUE from temperature + humidity
  - Model B: Cooling power from temperature + humidity + IT power
  - Model C: Total overhead from temperature + humidity + IT power

All results are saved to output/results.json so the paper can reference
exact, consistent numbers.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(__file__))
from config import MERGED_CSV, RESULTS_FILE, TEST_SIZE, RANDOM_STATE


def evaluate(model, X_train, X_test, y_train, y_test, name):
    """Train, predict, print metrics, return dict of results."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print(f"  {name}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    result = {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2)}

    if hasattr(model, "coef_"):
        result["coefficients"] = {
            fname: round(float(c), 6)
            for fname, c in zip(X_train.columns, model.coef_)
        }
        result["intercept"] = round(float(model.intercept_), 6)

    if hasattr(model, "feature_importances_"):
        result["feature_importance"] = {
            fname: round(float(imp), 4)
            for fname, imp in zip(X_train.columns, model.feature_importances_)
        }
        print(f"    Features: {result['feature_importance']}")

    return result


def main():
    print("=" * 60)
    print("  STEP 2: Machine Learning Analysis")
    print("=" * 60)

    df = pd.read_csv(MERGED_CSV, index_col=0, parse_dates=True)
    print(f"Loaded {len(df):,} hourly observations")
    print(f"Date range: {df.index.min()} -> {df.index.max()}")

    results = {
        "n_observations": len(df),
        "date_range": [str(df.index.min()), str(df.index.max())],
    }

    # ── Descriptive statistics ───────────────────────────────────────────
    desc = df.describe().round(3)
    results["descriptive"] = {}
    for col in ["pue", "it_power_kw", "cooling_kw", "hvac_kw", "pump_kw",
                "outdoor_air_temp", "outdoor_air_humidity"]:
        if col in desc.columns:
            results["descriptive"][col] = {
                "mean": round(float(desc.loc["mean", col]), 3),
                "std": round(float(desc.loc["std", col]), 3),
                "min": round(float(desc.loc["min", col]), 3),
                "max": round(float(desc.loc["max", col]), 3),
            }

    # ── Correlation analysis ─────────────────────────────────────────────
    print("\nPearson Correlation Analysis:")
    results["correlations"] = {}
    for target in ["pue", "cooling_kw", "hvac_kw", "pump_kw"]:
        if target in df.columns:
            valid = df[["outdoor_air_temp", target]].dropna()
            if len(valid) >= 10:
                r, p = stats.pearsonr(valid["outdoor_air_temp"], valid[target])
                results["correlations"][f"temp_vs_{target}"] = {
                    "r": round(r, 4), "p": float(f"{p:.2e}"), "n": len(valid),
                }
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                print(f"  Temp vs {target:15s}: r={r:+.4f}, p={p:.2e} {sig}")

    # ── Model A: Predict PUE ─────────────────────────────────────────────
    print("\nModel A: PUE from Temperature + Humidity")
    feat_a = df[["outdoor_air_temp", "outdoor_air_humidity", "pue"]].dropna()
    X, y = feat_a[["outdoor_air_temp", "outdoor_air_humidity"]], feat_a["pue"]
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(Xa_tr):,}  Test: {len(Xa_te):,}")

    results["model_a"] = {
        "n_train": len(Xa_tr), "n_test": len(Xa_te),
        "linear": evaluate(LinearRegression(), Xa_tr, Xa_te, ya_tr, ya_te, "Linear"),
        "random_forest": evaluate(
            RandomForestRegressor(n_estimators=100, max_depth=12,
                                  random_state=RANDOM_STATE, n_jobs=-1),
            Xa_tr, Xa_te, ya_tr, ya_te, "Random Forest"),
    }

    # ── Model B: Predict Cooling ─────────────────────────────────────────
    print("\nModel B: Cooling from Temperature + Humidity + IT Power")
    feat_b = df[["outdoor_air_temp", "outdoor_air_humidity", "it_power_kw", "cooling_kw"]].dropna()
    Xc, yc = feat_b[["outdoor_air_temp", "outdoor_air_humidity", "it_power_kw"]], feat_b["cooling_kw"]
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        Xc, yc, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(Xb_tr):,}  Test: {len(Xb_te):,}")

    results["model_b"] = {
        "n_train": len(Xb_tr), "n_test": len(Xb_te),
        "linear": evaluate(LinearRegression(), Xb_tr, Xb_te, yb_tr, yb_te, "Linear"),
        "random_forest": evaluate(
            RandomForestRegressor(n_estimators=200, max_depth=15,
                                  random_state=RANDOM_STATE, n_jobs=-1),
            Xb_tr, Xb_te, yb_tr, yb_te, "Random Forest"),
        "gradient_boosting": evaluate(
            GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                      random_state=RANDOM_STATE),
            Xb_tr, Xb_te, yb_tr, yb_te, "Gradient Boosting"),
    }

    # ── Model C: Predict Total Overhead ──────────────────────────────────
    print("\nModel C: Total Overhead from Temperature + Humidity + IT Power")
    df["overhead_kw"] = df["cooling_kw"] + df["hvac_kw"] + df["pump_kw"]
    feat_c = df[["outdoor_air_temp", "outdoor_air_humidity", "it_power_kw", "overhead_kw"]].dropna()
    Xo, yo = feat_c[["outdoor_air_temp", "outdoor_air_humidity", "it_power_kw"]], feat_c["overhead_kw"]
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        Xo, yo, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(Xc_tr):,}  Test: {len(Xc_te):,}")

    results["model_c"] = {
        "n_train": len(Xc_tr), "n_test": len(Xc_te),
        "random_forest": evaluate(
            RandomForestRegressor(n_estimators=200, max_depth=15,
                                  random_state=RANDOM_STATE, n_jobs=-1),
            Xc_tr, Xc_te, yc_tr, yc_te, "Random Forest"),
    }

    # ── Save results ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
