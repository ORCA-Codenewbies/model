from pathlib import Path
import json

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE = Path(".")

ERA5 = BASE / "Datasets/processed/ERA5/era5_monthly_indian_ocean_1960_2026.nc"
CATCH = BASE / "Datasets/processed/IOTC/iotc_catch_clean.parquet"
EFFORT = BASE / "Datasets/processed/IOTC/iotc_effort_clean.parquet"

OUT = BASE / "Models/ocean_suitability/training/hierarchical"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. IOTC spatial CPUE
# ============================================================

catch = pd.read_parquet(CATCH)
effort = pd.read_parquet(EFFORT)

catch = catch[
    (catch["FISHERY_GROUP"] == "Longline") &
    (catch["GEAR"] == "Longline (deep-freezing)") &
    (catch["CATCH_UNIT_CODE"] == "MT")
].copy()

effort = effort[
    (effort["FISHERY_GROUP"] == "Longline") &
    (effort["GEAR"] == "Longline (deep-freezing)") &
    (effort["EFFORT_UNIT_CODE"] == "HOOKS")
].copy()


def decode_5deg(code):
    s = str(code).zfill(7)

    if int(s[0]) != 6:
        return None

    q = int(s[1])
    lat_code = int(s[2:4])
    lon_code = int(s[4:7])

    if q == 1:
        lat_min = float(lat_code)
        lon_min = float(lon_code)
    elif q == 2:
        lat_min = -float(lat_code) - 5
        lon_min = float(lon_code)
    elif q == 3:
        lat_min = -float(lat_code) - 5
        lon_min = -float(lon_code) - 5
    elif q == 4:
        lat_min = float(lat_code)
        lon_min = -float(lon_code) - 5
    else:
        return None

    return {
        "lat_min": lat_min,
        "lat_max": lat_min + 5,
        "lon_min": lon_min,
        "lon_max": lon_min + 5,
        "lat_center": lat_min + 2.5,
        "lon_center": lon_min + 2.5,
    }


catch_cell = (
    catch
    .groupby(
        ["MONTH_DATE", "FISHING_GROUND_CODE"],
        as_index=False
    )["CATCH"]
    .sum()
    .rename(columns={"CATCH": "catch_mt"})
)

effort_cell = (
    effort
    .groupby(
        ["MONTH_DATE", "FISHING_GROUND_CODE"],
        as_index=False
    )["EFFORT"]
    .sum()
    .rename(columns={"EFFORT": "effort_hooks"})
)

catch_cell["FISHING_GROUND_CODE"] = (
    catch_cell["FISHING_GROUND_CODE"].astype(str)
)

effort_cell["FISHING_GROUND_CODE"] = (
    effort_cell["FISHING_GROUND_CODE"].astype(str)
)

target = catch_cell.merge(
    effort_cell,
    on=["MONTH_DATE", "FISHING_GROUND_CODE"],
    how="inner"
)

target = target[target["effort_hooks"] > 0].copy()

target["cpue"] = (
    target["catch_mt"] /
    target["effort_hooks"]
)

geo_rows = []

for code in target["FISHING_GROUND_CODE"].unique():
    g = decode_5deg(code)
    if g is not None:
        geo_rows.append({
            "FISHING_GROUND_CODE": code,
            **g
        })

geo = pd.DataFrame(geo_rows)

target = target.merge(
    geo,
    on="FISHING_GROUND_CODE",
    how="inner"
)

target = target[
    (target["lat_min"] >= 5) &
    (target["lat_max"] <= 25) &
    (target["lon_min"] >= 65) &
    (target["lon_max"] <= 95)
].copy()

target["MONTH_DATE"] = pd.to_datetime(
    target["MONTH_DATE"]
)

target["log_cpue"] = np.log1p(
    target["cpue"]
)


# ============================================================
# 2. ERA5 cell-level environmental data
# ============================================================

ds = xr.open_dataset(
    ERA5,
    engine="netcdf4"
)

ERA_VARS = [
    "sst",
    "u10",
    "v10",
    "wind_speed_10m",
    "t2m",
    "msl",
    "tp",
]

rows = []

for code, g in target.groupby(
    "FISHING_GROUND_CODE",
    sort=True
):

    first = g.iloc[0]

    sub = ds.sel(
        valid_time=g["MONTH_DATE"].values,
        latitude=slice(
            float(first["lat_min"]),
            float(first["lat_max"])
        ),
        longitude=slice(
            float(first["lon_min"]),
            float(first["lon_max"])
        )
    )

    out = pd.DataFrame({
        "MONTH_DATE": pd.to_datetime(
            sub["valid_time"].values
        ),
        "FISHING_GROUND_CODE": code
    })

    for var in ERA_VARS:

        da = sub[var]

        spatial_dims = [
            d for d in da.dims
            if d != "valid_time"
        ]

        out[f"{var}_mean"] = np.asarray(
            da.mean(
                dim=spatial_dims,
                skipna=True
            ).values,
            dtype=float
        )

        out[f"{var}_std"] = np.asarray(
            da.std(
                dim=spatial_dims,
                skipna=True
            ).values,
            dtype=float
        )

        out[f"{var}_valid_fraction"] = np.asarray(
            da.notnull()
            .mean(dim=spatial_dims)
            .values,
            dtype=float
        )

    rows.append(out)

env = pd.concat(
    rows,
    ignore_index=True
)

df = target.merge(
    env,
    on=[
        "MONTH_DATE",
        "FISHING_GROUND_CODE"
    ],
    how="inner"
)

df = df.sort_values(
    ["MONTH_DATE", "FISHING_GROUND_CODE"]
).reset_index(drop=True)


# ============================================================
# 3. Physical transformations
# ============================================================

df["sst_c"] = (
    df["sst_mean"] - 273.15
)

df["t2m_c"] = (
    df["t2m_mean"] - 273.15
)

df["wind_direction"] = (
    np.degrees(
        np.arctan2(
            df["v10_mean"],
            df["u10_mean"]
        )
    ) % 360
)

month = df["MONTH_DATE"].dt.month.astype(float)

df["month_sin"] = np.sin(
    2 * np.pi * (month - 1) / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * (month - 1) / 12
)


# ============================================================
# 4. Chronological split
# ============================================================

TRAIN_END = pd.Timestamp("2021-01-01")

train = df[
    df["MONTH_DATE"] < TRAIN_END
].copy()

test = df[
    df["MONTH_DATE"] >= TRAIN_END
].copy()

print("=" * 100)
print("HIERARCHICAL OCEAN SUITABILITY")
print("=" * 100)

print("Total rows :", len(df))
print("Cells      :", df["FISHING_GROUND_CODE"].nunique())

print("\nTrain:", len(train))
print("Test :", len(test))


# ============================================================
# 5. TRAINING-ONLY CELL CLIMATOLOGY
# ============================================================

cell_stats = (
    train
    .groupby("FISHING_GROUND_CODE")["log_cpue"]
    .agg(
        cell_log_mean="mean",
        cell_log_std="std",
        cell_log_median="median",
        cell_count="count",
    )
    .reset_index()
)

global_log_mean = float(
    train["log_cpue"].mean()
)

# Shrink sparse cell means toward global mean.
# This is empirical-Bayes-style smoothing without future leakage.
K = 12.0

cell_stats["cell_baseline"] = (
    (
        cell_stats["cell_count"] *
        cell_stats["cell_log_mean"]
    )
    +
    (
        K *
        global_log_mean
    )
) / (
    cell_stats["cell_count"] + K
)

cell_stats["cell_baseline_strength"] = (
    cell_stats["cell_count"] /
    (
        cell_stats["cell_count"] + K
    )
)

train = train.merge(
    cell_stats[
        [
            "FISHING_GROUND_CODE",
            "cell_baseline",
            "cell_baseline_strength"
        ]
    ],
    on="FISHING_GROUND_CODE",
    how="left"
)

test = test.merge(
    cell_stats[
        [
            "FISHING_GROUND_CODE",
            "cell_baseline",
            "cell_baseline_strength"
        ]
    ],
    on="FISHING_GROUND_CODE",
    how="left"
)

# For a completely unseen cell, use the training global mean.
train["cell_baseline"] = train[
    "cell_baseline"
].fillna(global_log_mean)

test["cell_baseline"] = test[
    "cell_baseline"
].fillna(global_log_mean)

train["cell_baseline_strength"] = train[
    "cell_baseline_strength"
].fillna(0.0)

test["cell_baseline_strength"] = test[
    "cell_baseline_strength"
].fillna(0.0)


# ============================================================
# 6. Environmental anomalies
#
# Climatology is computed from TRAIN only.
# ============================================================

ENV_BASE = [
    "sst_c",
    "u10_mean",
    "v10_mean",
    "wind_speed_10m_mean",
    "t2m_c",
    "msl_mean",
    "tp_mean",
]

# Training climatology by month.
monthly_clim = (
    train
    .groupby(
        train["MONTH_DATE"].dt.month
    )[ENV_BASE]
    .mean()
)

for col in ENV_BASE:

    train_month = train["MONTH_DATE"].dt.month

    test_month = test["MONTH_DATE"].dt.month

    train[f"{col}_anom"] = (
        train[col]
        - train_month.map(
            monthly_clim[col]
        )
    )

    test[f"{col}_anom"] = (
        test[col]
        - test_month.map(
            monthly_clim[col]
        )
    )


# Cell-specific environmental anomaly.
cell_env_means = (
    train
    .groupby("FISHING_GROUND_CODE")[
        ENV_BASE
    ]
    .mean()
)

for col in ENV_BASE:

    train[f"{col}_cell_anom"] = (
        train[col]
        - train["FISHING_GROUND_CODE"]
        .map(cell_env_means[col])
    )

    test[f"{col}_cell_anom"] = (
        test[col]
        - test["FISHING_GROUND_CODE"]
        .map(cell_env_means[col])
    )

    # Unknown cell fallback.
    train[f"{col}_cell_anom"] = (
        train[f"{col}_cell_anom"]
        .fillna(train[col] - train[col].mean())
    )

    test[f"{col}_cell_anom"] = (
        test[f"{col}_cell_anom"]
        .fillna(test[col] - train[col].mean())
    )


# ============================================================
# 7. Target anomaly
# ============================================================

train["target_anomaly"] = (
    train["log_cpue"] -
    train["cell_baseline"]
)

test["target_anomaly"] = (
    test["log_cpue"] -
    test["cell_baseline"]
)


# ============================================================
# 8. Feature set
# ============================================================

FEATURES = [
    "sst_c",
    "sst_std",
    "sst_valid_fraction",

    "u10_mean",
    "u10_std",
    "u10_valid_fraction",

    "v10_mean",
    "v10_std",
    "v10_valid_fraction",

    "wind_speed_10m_mean",
    "wind_speed_10m_std",

    "t2m_c",
    "t2m_std",
    "t2m_valid_fraction",

    "msl_mean",
    "msl_std",
    "msl_valid_fraction",

    "tp_mean",
    "tp_std",
    "tp_valid_fraction",

    "wind_direction",

    "lat_center",
    "lon_center",

    "month_sin",
    "month_cos",

    "cell_baseline",
    "cell_baseline_strength",
]

FEATURES += [
    f"{c}_anom"
    for c in ENV_BASE
]

FEATURES += [
    f"{c}_cell_anom"
    for c in ENV_BASE
]


# ============================================================
# 9. Train anomaly model
# ============================================================

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=600,
    learning_rate=0.025,
    max_depth=3,
    min_child_weight=10,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_alpha=0.25,
    reg_lambda=10.0,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
)

model.fit(
    train[FEATURES],
    train["target_anomaly"],
    verbose=False
)


# ============================================================
# 10. Reconstruct log CPUE
# ============================================================

pred_anomaly = model.predict(
    test[FEATURES]
)

pred_log_cpue = (
    test["cell_baseline"].values
    + pred_anomaly
)

pred_log_cpue = np.maximum(
    pred_log_cpue,
    0.0
)

pred_cpue = np.expm1(
    pred_log_cpue
)

true_log_cpue = test[
    "log_cpue"
].values

# ============================================================
# 11. Metrics
# ============================================================

log_mae = mean_absolute_error(
    true_log_cpue,
    pred_log_cpue
)

log_rmse = np.sqrt(
    mean_squared_error(
        true_log_cpue,
        pred_log_cpue
    )
)

log_r2 = r2_score(
    true_log_cpue,
    pred_log_cpue
)

cpue_mae = mean_absolute_error(
    test["cpue"],
    pred_cpue
)

cpue_rmse = np.sqrt(
    mean_squared_error(
        test["cpue"],
        pred_cpue
    )
)

cpue_r2 = r2_score(
    test["cpue"],
    pred_cpue
)


# ============================================================
# 12. Baseline: training cell baseline
# ============================================================

baseline_log = (
    test["cell_baseline"].values
)

baseline_cpue = np.expm1(
    baseline_log
)

baseline_log_mae = mean_absolute_error(
    true_log_cpue,
    baseline_log
)

baseline_log_rmse = np.sqrt(
    mean_squared_error(
        true_log_cpue,
        baseline_log
    )
)

baseline_log_r2 = r2_score(
    true_log_cpue,
    baseline_log
)

baseline_cpue_mae = mean_absolute_error(
    test["cpue"],
    baseline_cpue
)

baseline_cpue_rmse = np.sqrt(
    mean_squared_error(
        test["cpue"],
        baseline_cpue
    )
)

baseline_cpue_r2 = r2_score(
    test["cpue"],
    baseline_cpue
)


# ============================================================
# 13. Suitability score
#
# Use training target range only.
# ============================================================

train_min = float(
    train["log_cpue"].min()
)

train_max = float(
    train["log_cpue"].max()
)

def score(x):
    return np.clip(
        (x - train_min) /
        (train_max - train_min),
        0.0,
        1.0
    )

test_score_true = score(
    test["log_cpue"].values
)

test_score_pred = score(
    pred_log_cpue
)

score_mae = mean_absolute_error(
    test_score_true,
    test_score_pred
)

score_rmse = np.sqrt(
    mean_squared_error(
        test_score_true,
        test_score_pred
    )
)

score_r2 = r2_score(
    test_score_true,
    test_score_pred
)


# ============================================================
# 14. Spatial diagnostics
# ============================================================

predictions = test[
    [
        "MONTH_DATE",
        "FISHING_GROUND_CODE",
        "lat_center",
        "lon_center",
        "cpue",
    ]
].copy()

predictions["predicted_cpue"] = pred_cpue
predictions["true_score"] = test_score_true
predictions["predicted_score"] = test_score_pred


cell_rows = []

for code, g in predictions.groupby(
    "FISHING_GROUND_CODE"
):

    if len(g) < 2:
        continue

    cell_rows.append({
        "FISHING_GROUND_CODE": code,
        "rows": len(g),
        "cpue_mae": mean_absolute_error(
            g["cpue"],
            g["predicted_cpue"]
        ),
        "baseline_cpue_mae": mean_absolute_error(
            g["cpue"],
            np.full(
                len(g),
                g["cpue"].mean()
            )
        ),
    })

cell_metrics = pd.DataFrame(
    cell_rows
)


# ============================================================
# 15. Feature importance
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values(
    "importance",
    ascending=False
)


# ============================================================
# 16. Save candidate model
# ============================================================

model_path = (
    BASE /
    "Models/ocean_suitability/model/"
    "ocean_suitability_hierarchical_xgboost.json"
)

model.save_model(
    model_path
)

predictions.to_csv(
    OUT /
    "ocean_suitability_hierarchical_predictions.csv",
    index=False
)

cell_metrics.to_csv(
    OUT /
    "ocean_suitability_hierarchical_cell_metrics.csv",
    index=False
)

importance.to_csv(
    OUT /
    "ocean_suitability_hierarchical_feature_importance.csv",
    index=False
)

train.to_csv(
    OUT /
    "ocean_suitability_hierarchical_training.csv",
    index=False
)


metrics = {
    "model": "Hierarchical XGBoost",
    "status": "experiment",
    "rows": int(len(df)),
    "cells": int(
        df["FISHING_GROUND_CODE"].nunique()
    ),
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "features": FEATURES,

    "xgboost": {
        "log_cpue_mae": float(log_mae),
        "log_cpue_rmse": float(log_rmse),
        "log_cpue_r2": float(log_r2),
        "cpue_mae": float(cpue_mae),
        "cpue_rmse": float(cpue_rmse),
        "cpue_r2": float(cpue_r2),
        "suitability_mae": float(score_mae),
        "suitability_rmse": float(score_rmse),
        "suitability_r2": float(score_r2),
    },

    "cell_baseline": {
        "log_cpue_mae": float(
            baseline_log_mae
        ),
        "log_cpue_rmse": float(
            baseline_log_rmse
        ),
        "log_cpue_r2": float(
            baseline_log_r2
        ),
        "cpue_mae": float(
            baseline_cpue_mae
        ),
        "cpue_rmse": float(
            baseline_cpue_rmse
        ),
        "cpue_r2": float(
            baseline_cpue_r2
        ),
    },

    "target": {
        "source": "IOTC catch and effort",
        "catch_unit": "MT",
        "effort_unit": "HOOKS",
        "fishery": "Longline",
        "gear": "Longline (deep-freezing)",
        "spatial_resolution": "5 degree x 5 degree",
        "target": "log1p(catch_mt / effort_hooks)",
    },

    "environment": {
        "source": "ERA5 monthly",
        "variables": ERA_VARS,
        "cell_aggregation": "spatial mean and standard deviation",
        "anomaly_features": True,
    },

    "evaluation": {
        "chronological_cutoff": "2021-01-01",
        "cell_climatology_fit_on_training_only": True,
        "no_future_target_leakage": True,
    },

    "limitations": [
        "CPUE is a fishery-dependent proxy rather than direct ecological suitability ground truth.",
        "Spatial IOTC coverage is uneven.",
        "Historical CHL, MLD, D20 and INCOIS currents are unavailable over the complete target period.",
        "The model should not be interpreted as causal evidence that environmental conditions determine CPUE.",
    ],
}

with open(
    OUT /
    "ocean_suitability_hierarchical_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=2
    )


# ============================================================
# 17. Report
# ============================================================

print("\n" + "=" * 100)
print("HIERARCHICAL OCEAN SUITABILITY RESULT")
print("=" * 100)

print("Rows :", len(df))
print("Cells:", df["FISHING_GROUND_CODE"].nunique())

print("\nChronological test")
print("Test rows :", len(test))

print("\nXGBoost")
print(f"Log CPUE MAE : {log_mae:.6f}")
print(f"Log CPUE RMSE: {log_rmse:.6f}")
print(f"Log CPUE R2  : {log_r2:.6f}")

print(f"CPUE MAE     : {cpue_mae:.9f}")
print(f"CPUE RMSE    : {cpue_rmse:.9f}")
print(f"CPUE R2      : {cpue_r2:.6f}")

print("\nSuitability")
print(f"MAE          : {score_mae:.6f}")
print(f"RMSE         : {score_rmse:.6f}")
print(f"R2           : {score_r2:.6f}")

print("\nCell baseline")
print(f"Log CPUE MAE : {baseline_log_mae:.6f}")
print(f"Log CPUE RMSE: {baseline_log_rmse:.6f}")
print(f"Log CPUE R2  : {baseline_log_r2:.6f}")

print(f"CPUE MAE     : {baseline_cpue_mae:.9f}")
print(f"CPUE RMSE    : {baseline_cpue_rmse:.9f}")
print(f"CPUE R2      : {baseline_cpue_r2:.6f}")

print("\nTop features")
print(
    importance.head(20).to_string(
        index=False
    )
)

print("\nModel:")
print(model_path)

print("\n" + "=" * 100)

ds.close()
