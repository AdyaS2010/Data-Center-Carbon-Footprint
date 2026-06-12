"""
config.py - Shared configuration for the carbon footprint analysis.
All scripts import from here so paths and settings live in one place.
"""

import os
import matplotlib

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "figures")
RESULTS_FILE = os.path.join(PROJECT_ROOT, "output", "results.json")

# Raw data lives in the original download folder (avoids re-downloading ~900 MB)
RAW_DATA_DIR = os.path.join(DATA_DIR)  # data already merged in this project
POWER_CSV = os.path.join(RAW_DATA_DIR, "esif.influx.buildingData.PUE.combined.csv")
WEATHER_CSV = os.path.join(RAW_DATA_DIR, "esif.influx.buildingData.outside.combined.csv")

# Cleaned merged output
MERGED_CSV = os.path.join(DATA_DIR, "esif_hourly_merged.csv")

# ── Processing Constants ─────────────────────────────────────────────────
SAMPLE_STEP = 10          # Deterministic: take every 10th row by position
PUE_MIN = 0.95
PUE_MAX = 1.50
TEMP_MIN_F = -50
TEMP_MAX_F = 130
TEST_SIZE = 0.20
RANDOM_STATE = 42

# US average grid emission factor (kg CO2 per kWh) — EPA eGRID 2023
US_GRID_EMISSION_FACTOR = 0.39

# Global average grid emission factor (kg CO2 per kWh) — IEA 2024
GLOBAL_GRID_EMISSION_FACTOR = 0.49

# ── Plot Style ───────────────────────────────────────────────────────────
COLORS = {
    "blue":       "#2166AC",
    "red":        "#B2182B",
    "green":      "#1B7837",
    "orange":     "#E08214",
    "gray":       "#666666",
    "light_blue": "#92C5DE",
    "purple":     "#762A83",
    "teal":       "#01665E",
}

def setup_plot_style():
    """Apply consistent matplotlib styling for publication figures."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt
