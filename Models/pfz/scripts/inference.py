from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb


PFZ_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PFZ_ROOT / "model" / "pfz_xgboost.json"
CONFIG_PATH = PFZ_ROOT / "config.json"

REPO_ROOT = PFZ_ROOT.parents[1]

ERA5_PATH = (
    REPO_ROOT
    / "Datasets"
    / "processed"
    / "ERA5"
    / "era5_monthly_indian_ocean_1960_2026.nc"
)

ROMS_PATH = (
    REPO_ROOT
    / "Datasets"
    / "processed"
    / "INCOIS"
    / "ROMS"
    / "incois_roms_surface_current_20260828.nc"
)


with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

FEATURES = CONFIG["features"]
THRESHOLD = float(CONFIG["decision_threshold"])


def sample(da, lat, lon):
    value = da.sel(
        latitude=lat,
        longitude=lon,
        method="nearest",
    ).values

    value = np.asarray(value).squeeze()

    if not np.isfinite(value):
        return np.nan

    return float(value)


def build_features(lat, lon, timestamp):
    timestamp = pd.Timestamp(timestamp)

    with xr.open_dataset(ERA5_PATH) as era, \
         xr.open_dataset(ROMS_PATH) as roms_ds:

        roms = roms_ds.isel(time=0)

        era_month = timestamp.to_period("M").to_timestamp()

        era_t = era.sel(
            valid_time=era_month,
            method="nearest",
        )

        era_sst = sample(era_t["sst"], lat, lon)
        era_u10 = sample(era_t["u10"], lat, lon)
        era_v10 = sample(era_t["v10"], lat, lon)
        era_t2m = sample(era_t["t2m"], lat, lon)
        era_msl = sample(era_t["msl"], lat, lon)
        era_tp = sample(era_t["tp"], lat, lon)

        current_u = sample(
            roms["u_current"],
            lat,
            lon,
        )

        current_v = sample(
            roms["v_current"],
            lat,
            lon,
        )

        current_speed = sample(
            roms["current_speed"],
            lat,
            lon,
        )

    wind_speed = np.sqrt(
        era_u10 ** 2 + era_v10 ** 2
    ) if np.isfinite(era_u10) and np.isfinite(era_v10) else np.nan

    current_direction_rad = np.arctan2(
        current_v,
        current_u,
    ) if np.isfinite(current_u) and np.isfinite(current_v) else np.nan

    values = {
        "latitude": float(lat),
        "longitude": float(lon),
        "era_sst": era_sst,
        "era_u10": era_u10,
        "era_v10": era_v10,
        "era_t2m": era_t2m,
        "era_msl": era_msl,
        "era_tp": era_tp,
        "wind_speed": wind_speed,
        "current_u": current_u,
        "current_v": current_v,
        "current_speed": current_speed,
        "current_direction_rad": current_direction_rad,
        "month": int(timestamp.month),
        "day_of_year": int(timestamp.dayofyear),
    }

    return pd.DataFrame(
        [[values[f] for f in FEATURES]],
        columns=FEATURES,
    )


def predict(lat, lon, timestamp):
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    X = build_features(
        lat,
        lon,
        timestamp,
    )

    missing = [
        c for c in FEATURES
        if not np.isfinite(X.iloc[0][c])
    ]

    if missing:
        return {
            "timestamp": str(pd.Timestamp(timestamp)),
            "latitude": float(lat),
            "longitude": float(lon),
            "status": "insufficient_environmental_data",
            "missing_features": missing,
            "model": CONFIG["model"],
            "model_version": CONFIG["model_version"],
        }

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    return {
        "timestamp": str(pd.Timestamp(timestamp)),
        "latitude": float(lat),
        "longitude": float(lon),
        "status": "ok",
        "pfz_probability": probability,
        "pfz_present": probability >= THRESHOLD,
        "decision_threshold": THRESHOLD,
        "model": CONFIG["model"],
        "model_version": CONFIG["model_version"],
    }


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python inference.py "
            "<latitude> <longitude> <timestamp>"
        )
        sys.exit(1)

    lat = float(sys.argv[1])
    lon = float(sys.argv[2])
    timestamp = sys.argv[3]

    print(json.dumps(
        predict(
            lat,
            lon,
            timestamp,
        ),
        indent=2,
    ))
