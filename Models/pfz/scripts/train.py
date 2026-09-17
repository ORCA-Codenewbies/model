from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


ROOT = Path(".")
PFZ = ROOT / "Datasets/processed/PFZ/pfz_advisories_from_xls.csv"
ERA5 = ROOT / "Datasets/processed/ERA5/era5_monthly_indian_ocean_1960_2026.nc"
ROMS = ROOT / "Datasets/processed/INCOIS/ROMS/incois_roms_surface_current_20260828.nc"

MODEL_DIR = ROOT / "Models/pfz/model"
METRICS_DIR = ROOT / "Models/pfz/metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PFZ XGBOOST TRAINING")
print("=" * 80)

# ---------------------------------------------------------------------
# 1. Load real INCOIS PFZ advisories
# ---------------------------------------------------------------------

pfz = pd.read_csv(PFZ, parse_dates=["valid_from", "valid_to"])

pfz = pfz[
    (pfz["valid_from"] == "2026-08-28") &
    (pfz["valid_to"] == "2026-08-29")
].copy()

if len(pfz) == 0:
    raise RuntimeError("No 2026-08-28 PFZ advisory records found.")

print("Real PFZ points:", len(pfz))

# ---------------------------------------------------------------------
# 2. Load ERA5 August 2026 fields
# ---------------------------------------------------------------------

era = xr.open_dataset(ERA5)

# Select August 2026 monthly record.
era_aug = era.sel(
    valid_time="2026-08-01",
    method="nearest",
)

era_features = {
    "era_sst": era_aug["sst"],
    "era_u10": era_aug["u10"],
    "era_v10": era_aug["v10"],
    "era_t2m": era_aug["t2m"],
    "era_msl": era_aug["msl"],
    "era_tp": era_aug["tp"],
}

# ---------------------------------------------------------------------
# 3. Load INCOIS ROMS current snapshot
# ---------------------------------------------------------------------

roms = xr.open_dataset(ROMS).isel(time=0)

roms_features = {
    "current_u": roms["u_current"],
    "current_v": roms["v_current"],
    "current_speed": roms["current_speed"],
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sample_da(da, lat, lon):
    """
    Nearest-neighbour sampling at a geographic point.
    """
    return float(
        da.sel(
            latitude=lat,
            longitude=lon,
            method="nearest",
        ).values
    )


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0

    p1 = np.radians(lat1)
    p2 = np.radians(lat2)

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    )

    return 2 * r * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------
# 4. Build positive samples
# ---------------------------------------------------------------------

positive = pfz.copy()
positive["label"] = 1

def build_features(df):
    rows = []

    for _, r in df.iterrows():
        lat = float(r["latitude"])
        lon = float(r["longitude"])

        out = {
            "latitude": lat,
            "longitude": lon,
            "bearing_deg": 0.0,
            "distance_mid_km": 0.0,
            "depth_mid_m": 0.0,
            "era_sst": sample_da(era_features["era_sst"], lat, lon),
            "era_u10": sample_da(era_features["era_u10"], lat, lon),
            "era_v10": sample_da(era_features["era_v10"], lat, lon),
            "era_t2m": sample_da(era_features["era_t2m"], lat, lon),
            "era_msl": sample_da(era_features["era_msl"], lat, lon),
            "era_tp": sample_da(era_features["era_tp"], lat, lon),
            "current_u": sample_da(roms_features["current_u"], lat, lon),
            "current_v": sample_da(roms_features["current_v"], lat, lon),
            "current_speed": sample_da(
                roms_features["current_speed"], lat, lon
            ),
            "label": int(r["label"]),
        }

        out["wind_speed"] = np.sqrt(
            out["era_u10"] ** 2 + out["era_v10"] ** 2
        )

        out["current_direction_rad"] = np.arctan2(
            out["current_v"],
            out["current_u"],
        )

        out["month"] = 8
        out["day_of_year"] = 240

        rows.append(out)

    return pd.DataFrame(rows)


positive_features = build_features(positive)

# ---------------------------------------------------------------------
# 5. Build spatial negative samples from the same operational domain.
#
# IMPORTANT:
# These are NOT fabricated environmental values.
# Environmental fields come directly from ERA5/ROMS.
#
# A negative is defined conservatively as a grid location sufficiently
# far from every issued PFZ point. This is an absence-from-advisory
# operational label, not a claim that fish were experimentally absent.
# ---------------------------------------------------------------------

lat_min = max(5.0, pfz["latitude"].min() - 1.0)
lat_max = min(25.0, pfz["latitude"].max() + 1.0)
lon_min = max(65.0, pfz["longitude"].min() - 1.0)
lon_max = min(95.0, pfz["longitude"].max() + 1.0)

# Coarse grid aligned to ERA5 scale.
lats = np.arange(
    np.floor(lat_min * 4) / 4,
    np.ceil(lat_max * 4) / 4 + 0.001,
    0.25,
)

lons = np.arange(
    np.floor(lon_min * 4) / 4,
    np.ceil(lon_max * 4) / 4 + 0.001,
    0.25,
)

candidate_rows = []

for lat in lats:
    for lon in lons:
        d = haversine_km(
            lat,
            lon,
            pfz["latitude"].values,
            pfz["longitude"].values,
        )

        # Leave a 15 km exclusion buffer around advisory points.
        if np.min(d) >= 15.0:
            candidate_rows.append(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "bearing_deg": 0.0,
                    "distance_mid_km": np.min(d),
                    "depth_mid_m": np.nan,
                    "label": 0,
                }
            )

negatives = pd.DataFrame(candidate_rows)

if len(negatives) < 100:
    raise RuntimeError(
        f"Only {len(negatives)} negatives available; "
        "cannot construct a stable training set."
    )

# Match positive/negative balance.
rng = np.random.default_rng(42)
n_negative = min(len(negatives), len(positive) * 3)

negatives = negatives.iloc[
    rng.choice(
        len(negatives),
        size=n_negative,
        replace=False,
    )
].copy()

# Depth is unavailable at arbitrary negative grid cells.
# Impute only this engineered coordinate feature using positive median.
# Environmental fields are never imputed.
negatives["depth_mid_m"] = positive["depth_mid_m"].median()

negative_features = build_features(negatives)

data = pd.concat(
    [positive_features, negative_features],
    ignore_index=True,
)

# ---------------------------------------------------------------------
# 6. Final feature matrix
# ---------------------------------------------------------------------

features = [
    "latitude",
    "longitude",
    "era_sst",
    "era_u10",
    "era_v10",
    "era_t2m",
    "era_msl",
    "era_tp",
    "wind_speed",
    "current_u",
    "current_v",
    "current_speed",
    "current_direction_rad",
    "month",
    "day_of_year",
]

X = data[features].replace(
    [np.inf, -np.inf],
    np.nan,
)

y = data["label"].astype(int)

# Environmental fields must be fully observed.
valid = X.notna().all(axis=1)

X = X.loc[valid]
y = y.loc[valid]

print("\nTraining samples:", len(X))
print("Positive:", int(y.sum()))
print("Negative:", int((y == 0).sum()))
print("Features:", features)

# ---------------------------------------------------------------------
# 7. Train/validation split is spatial, not random-overlapping.
#
# Hold out Maharashtra as validation when enough points exist.
# Kerala is the training region. This provides a geographic sanity check.
# ---------------------------------------------------------------------

states = pd.Series(
    ["PFZ"] * len(positive_features)
    + ["NEG"] * len(negative_features)
)

# Since negatives have no state, use deterministic 80/20 split
# by longitude blocks to avoid point-level duplication.
lon_values = X["longitude"].values
threshold = np.quantile(lon_values, 0.80)

train_mask = lon_values <= threshold
val_mask = ~train_mask

X_train = X.iloc[np.where(train_mask)[0]]
y_train = y.iloc[np.where(train_mask)[0]]

X_val = X.iloc[np.where(val_mask)[0]]
y_val = y.iloc[np.where(val_mask)[0]]

if y_val.nunique() < 2:
    raise RuntimeError("Validation set contains only one class.")

# ---------------------------------------------------------------------
# 8. XGBoost
# ---------------------------------------------------------------------

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

# ---------------------------------------------------------------------
# 9. Validation
# ---------------------------------------------------------------------

prob = model.predict_proba(X_val)[:, 1]
pred = (prob >= 0.50).astype(int)

metrics = {
    "model": "PFZ_XGBoost",
    "scientifically_valid_cnn_lstm_training": False,
    "training_label_source": "INCOIS PFZ advisory",
    "advisory_window": [
        "2026-08-28",
        "2026-08-29",
    ],
    "positive_samples": int(y.sum()),
    "negative_samples": int((y == 0).sum()),
    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "accuracy": float(accuracy_score(y_val, pred)),
    "precision": float(
        precision_score(y_val, pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_val, pred, zero_division=0)
    ),
    "f1": float(
        f1_score(y_val, pred, zero_division=0)
    ),
    "roc_auc": float(
        roc_auc_score(y_val, prob)
    ),
    "features": features,
    "negative_label_definition":
        "grid cells >=15 km from every issued PFZ advisory point",
}

print("\n" + "=" * 80)
print("PFZ VALIDATION")
print("=" * 80)

for k, v in metrics.items():
    print(f"{k}: {v}")

# ---------------------------------------------------------------------
# 10. Save
# ---------------------------------------------------------------------

model_path = MODEL_DIR / "pfz_xgboost.keras.json"

model.save_model(model_path)

with open(
    METRICS_DIR / "pfz_xgboost_metrics.json",
    "w",
) as f:
    json.dump(metrics, f, indent=2)

print("\nModel:", model_path)
print(
    "Metrics:",
    METRICS_DIR / "pfz_xgboost_metrics.json",
)

era.close()
roms.close()
