from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ERA5_PATH = Path(
    "Datasets/processed/ERA5/"
    "era5_monthly_indian_ocean_1960_2026.nc"
)

TARGET_PATH = Path(
    "Datasets/processed/training/weather_risk/"
    "weather_risk_dataset.parquet"
)

OUT_DIR = Path("Datasets/processed/training/weather_risk")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "weather_risk_features.parquet"

BASE_VARS = [
    "u10",
    "v10",
    "wind_speed_10m",
    "t2m",
    "msl",
    "sst",
    "tp",
]


print("Loading ERA5...")
ds = xr.open_dataset(ERA5_PATH)

# Remove non-feature auxiliary coordinates that can create duplicate
# dataframe columns during conversion.
drop_coords = [
    c for c in ["number", "expver"]
    if c in ds.coords
]

if drop_coords:
    ds = ds.drop_vars(drop_coords)

# Ensure chronological ordering.
ds = ds.sortby("valid_time")

print("ERA5:", dict(ds.sizes))
print(
    "Coverage:",
    str(ds.valid_time.values[0]),
    "->",
    str(ds.valid_time.values[-1]),
)

# ---------------------------------------------------------------------
# Historical climatology: 1960-2019.
# This period is used ONLY to calculate normal monthly conditions.
# ---------------------------------------------------------------------
clim = ds[BASE_VARS].sel(
    valid_time=slice("1960-01-01", "2019-12-01")
)

monthly_clim = clim.groupby("valid_time.month").mean(
    "valid_time",
    skipna=True,
)

# ---------------------------------------------------------------------
# Supervised period: 2020-2025.
# ---------------------------------------------------------------------
current = ds[BASE_VARS].sel(
    valid_time=slice("2020-01-01", "2025-12-01")
)

# Anomaly = current monthly value - 1960-2019 monthly climatology.
anomaly = current.groupby("valid_time.month") - monthly_clim

# Month-to-month changes.
delta_vars = {}
for var in BASE_VARS:
    delta_vars[f"{var}_delta_1m"] = current[var].diff(
        "valid_time"
    )

deltas = xr.Dataset(delta_vars)

# Explicitly align first-month deltas to the full supervised timeline.
# The first month has no previous month, so NaN there is intentional.
deltas = deltas.reindex(valid_time=current.valid_time)

# ---------------------------------------------------------------------
# Add cyclic season features.
# ---------------------------------------------------------------------
month_number = current.valid_time.dt.month

season = xr.Dataset({
    "month_number": month_number,
    "sin_month": np.sin(
        2 * np.pi * month_number / 12.0
    ),
    "cos_month": np.cos(
        2 * np.pi * month_number / 12.0
    ),
})

# ---------------------------------------------------------------------
# Construct one clean feature dataset.
# ---------------------------------------------------------------------
feature_ds = current.copy()

for var in BASE_VARS:
    feature_ds[f"{var}_anomaly"] = anomaly[var]

feature_ds = xr.merge(
    [
        feature_ds,
        deltas,
        season,
    ],
    join="exact",
    compat="override",
)

# Convert once, after all feature engineering.
features = (
    feature_ds
    .to_dataframe()
    .reset_index()
)

# Remove any remaining auxiliary columns defensively.
for col in ["number", "expver"]:
    if col in features.columns:
        features = features.drop(columns=col)

features["month"] = (
    pd.to_datetime(features["valid_time"])
    .dt.to_period("M")
)

features = features.drop(
    columns=["valid_time"]
)

# ---------------------------------------------------------------------
# Attach the previously constructed real targets.
# ---------------------------------------------------------------------
target = pd.read_parquet(TARGET_PATH)

required_target = [
    "month",
    "latitude",
    "longitude",
    "target",
    "target_name",
    "split",
]

missing_target = [
    c for c in required_target
    if c not in target.columns
]

if missing_target:
    raise RuntimeError(
        f"Target dataset missing columns: {missing_target}"
    )

target = target[required_target].copy()

result = target.merge(
    features,
    on=["month", "latitude", "longitude"],
    how="left",
    validate="one_to_one",
)

# Stable ordering.
result = result.sort_values(
    ["month", "latitude", "longitude"]
).reset_index(drop=True)

# ---------------------------------------------------------------------
# Sanity checks.
# ---------------------------------------------------------------------
if len(result) != len(target):
    raise RuntimeError(
        f"Row count changed after feature join: "
        f"{len(target)} -> {len(result)}"
    )

feature_columns = [
    c for c in result.columns
    if c not in [
        "month",
        "latitude",
        "longitude",
        "target",
        "target_name",
        "split",
    ]
]

print("\n=== WEATHER FEATURE DATASET ===")
print("Rows:", len(result))
print("Feature count:", len(feature_columns))
print("Columns:", len(result.columns))

print("\nClass distribution:")
print(
    result.groupby(["split", "target_name"])
    .size()
    .unstack(fill_value=0)
    .to_string()
)

print("\nFeature missingness (%):")
print(
    result[feature_columns]
    .isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .to_string()
)

print("\nFeature columns:")
for col in feature_columns:
    print(" ", col)

result.to_parquet(
    OUT_PATH,
    index=False,
)

print("\nSaved:", OUT_PATH)

ds.close()
