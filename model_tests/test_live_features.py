"""
Live API Data & Feature Construction Test (test_live_features.py)

Validates deterministic conversion from raw real-time weather & ocean API responses
into exact 15-feature PFZ and 21-feature Weather Risk input schemas.

Pipeline:
Raw Live API Payload -> Feature Extraction & Derivations -> Schema Validation -> Model Inference
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
PFZ_MODEL_PATH = REPO_ROOT / "Models" / "pfz" / "model" / "pfz_xgboost.json"
WEATHER_MODEL_PATH = REPO_ROOT / "Models" / "weather_risk" / "model" / "weather_risk_xgboost.json"
RESULTS_DIR = REPO_ROOT / "model_tests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PFZ_EXPECTED_FEATURES = [
    "latitude", "longitude", "era_sst", "era_u10", "era_v10",
    "era_t2m", "era_msl", "era_tp", "wind_speed", "current_u",
    "current_v", "current_speed", "current_direction_rad", "month", "day_of_year"
]

WEATHER_EXPECTED_FEATURES = [
    "latitude", "longitude", "u10", "v10", "wind_speed_10m",
    "t2m", "msl", "tp", "hour", "month", "sin_hour", "cos_hour",
    "sin_month", "cos_month", "wind_gradient", "pressure_gradient",
    "temperature_gradient", "pressure_local_anomaly", "wind_local_anomaly",
    "wind_delta_3h", "pressure_delta_3h"
]


# =====================================================================
# 1. Feature Transformers (Mocking backend Real-time Data Fetcher)
# =====================================================================

def transform_raw_to_pfz_features(raw_payload: dict) -> pd.DataFrame:
    """
    Transforms raw atmospheric + ocean currents API data into exact 15 PFZ features.
    """
    lat = float(raw_payload["latitude"])
    lon = float(raw_payload["longitude"])
    ts = pd.Timestamp(raw_payload["timestamp"])

    u10 = float(raw_payload["u_wind_10m"])
    v10 = float(raw_payload["v_wind_10m"])
    curr_u = float(raw_payload["ocean_current_u"])
    curr_v = float(raw_payload["ocean_current_v"])

    wind_speed = float(np.sqrt(u10 ** 2 + v10 ** 2))
    curr_speed = float(np.sqrt(curr_u ** 2 + curr_v ** 2))
    curr_dir_rad = float(np.arctan2(curr_v, curr_u))

    features = {
        "latitude": lat,
        "longitude": lon,
        "era_sst": float(raw_payload["sea_surface_temp_k"]),
        "era_u10": u10,
        "era_v10": v10,
        "era_t2m": float(raw_payload["temp_2m_k"]),
        "era_msl": float(raw_payload["mean_sea_level_pressure_pa"]),
        "era_tp": float(raw_payload["total_precipitation_m"]),
        "wind_speed": wind_speed,
        "current_u": curr_u,
        "current_v": curr_v,
        "current_speed": curr_speed,
        "current_direction_rad": curr_dir_rad,
        "month": int(ts.month),
        "day_of_year": int(ts.dayofyear)
    }

    return pd.DataFrame([[features[f] for f in PFZ_EXPECTED_FEATURES]], columns=PFZ_EXPECTED_FEATURES)


def transform_raw_to_weather_features(raw_current: dict, raw_3h_ago: dict = None) -> pd.DataFrame:
    """
    Transforms raw weather API data (current + 3h historical) into exact 21 Weather Risk features.
    """
    lat = float(raw_current["latitude"])
    lon = float(raw_current["longitude"])
    ts = pd.Timestamp(raw_current["timestamp"])

    u10 = float(raw_current["u_wind_10m"])
    v10 = float(raw_current["v_wind_10m"])
    w_speed = float(raw_current.get("wind_speed_10m", np.sqrt(u10**2 + v10**2)))
    msl = float(raw_current["mean_sea_level_pressure_pa"])
    t2m = float(raw_current["temp_2m_k"])

    hour = int(ts.hour)
    month = int(ts.month)

    sin_hour = float(np.sin(2 * np.pi * hour / 24.0))
    cos_hour = float(np.cos(2 * np.pi * hour / 24.0))
    sin_month = float(np.sin(2 * np.pi * (month - 1) / 12.0))
    cos_month = float(np.cos(2 * np.pi * (month - 1) / 12.0))

    # Derived spatial gradients and anomalies (from spatial neighborhood API or defaults)
    w_grad = float(raw_current.get("wind_gradient", 0.0))
    p_grad = float(raw_current.get("pressure_gradient", 0.0))
    t_grad = float(raw_current.get("temperature_gradient", 0.0))
    p_anomaly = float(raw_current.get("pressure_local_anomaly", 0.0))
    w_anomaly = float(raw_current.get("wind_local_anomaly", 0.0))

    # 3-hour deltas calculation
    if raw_3h_ago is not None:
        u10_prev = float(raw_3h_ago["u_wind_10m"])
        v10_prev = float(raw_3h_ago["v_wind_10m"])
        w_speed_prev = float(np.sqrt(u10_prev**2 + v10_prev**2))
        msl_prev = float(raw_3h_ago["mean_sea_level_pressure_pa"])
        w_delta_3h = float(w_speed - w_speed_prev)
        p_delta_3h = float(msl - msl_prev)
    else:
        w_delta_3h = float(raw_current.get("wind_delta_3h", 0.0))
        p_delta_3h = float(raw_current.get("pressure_delta_3h", 0.0))

    features = {
        "latitude": lat,
        "longitude": lon,
        "u10": u10,
        "v10": v10,
        "wind_speed_10m": w_speed,
        "t2m": t2m,
        "msl": msl,
        "tp": float(raw_current["total_precipitation_m"]),
        "hour": hour,
        "month": month,
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "sin_month": sin_month,
        "cos_month": cos_month,
        "wind_gradient": w_grad,
        "pressure_gradient": p_grad,
        "temperature_gradient": t_grad,
        "pressure_local_anomaly": p_anomaly,
        "wind_local_anomaly": w_anomaly,
        "wind_delta_3h": w_delta_3h,
        "pressure_delta_3h": p_delta_3h
    }

    return pd.DataFrame([[features[f] for f in WEATHER_EXPECTED_FEATURES]], columns=WEATHER_EXPECTED_FEATURES)


# =====================================================================
# 2. Test Execution
# =====================================================================

def run_live_feature_tests():
    print("=" * 80)
    print("RUNNING LIVE DATA & FEATURE CONSTRUCTION VALIDATION")
    print("=" * 80)

    pfz_model = xgb.XGBClassifier()
    pfz_model.load_model(str(PFZ_MODEL_PATH))

    weather_model = xgb.XGBClassifier()
    weather_model.load_model(str(WEATHER_MODEL_PATH))

    test_results = {
        "pfz_live_transformation": {},
        "weather_live_transformation": {},
    }

    # Mock Live API Payloads
    mock_pfz_raw_api = {
        "latitude": 11.808,
        "longitude": 74.883,
        "timestamp": "2026-08-28T06:00:00Z",
        "sea_surface_temp_k": 302.15,
        "temp_2m_k": 299.85,
        "u_wind_10m": -3.2,
        "v_wind_10m": 5.4,
        "mean_sea_level_pressure_pa": 101150.0,
        "total_precipitation_m": 0.0025,
        "ocean_current_u": 0.22,
        "ocean_current_v": -0.15
    }

    mock_weather_raw_current = {
        "latitude": 14.5,
        "longitude": 82.1,
        "timestamp": "2026-09-09T18:00:00Z",
        "temp_2m_k": 301.2,
        "u_wind_10m": 15.4,
        "v_wind_10m": 22.1,
        "mean_sea_level_pressure_pa": 99400.0,
        "total_precipitation_m": 0.028,
        "wind_gradient": 4.8,
        "pressure_gradient": -6.5,
        "temperature_gradient": -1.2,
        "pressure_local_anomaly": -2200.0,
        "wind_local_anomaly": 14.2
    }

    mock_weather_raw_3h_ago = {
        "latitude": 14.5,
        "longitude": 82.1,
        "timestamp": "2026-09-09T15:00:00Z",
        "temp_2m_k": 301.8,
        "u_wind_10m": 10.2,
        "v_wind_10m": 14.5,
        "mean_sea_level_pressure_pa": 99950.0,
        "total_precipitation_m": 0.010
    }

    # 1. Test PFZ Transformation
    df_pfz = transform_raw_to_pfz_features(mock_pfz_raw_api)
    pfz_cols_match = (list(df_pfz.columns) == PFZ_EXPECTED_FEATURES)
    pfz_no_nulls = not df_pfz.isna().values.any()
    
    pfz_prob = float(pfz_model.predict_proba(df_pfz)[0, 1])
    pfz_pred_class = int(pfz_model.predict(df_pfz)[0])

    test_results["pfz_live_transformation"] = {
        "schema_match": pfz_cols_match,
        "no_nulls": pfz_no_nulls,
        "num_features": len(df_pfz.columns),
        "constructed_features": df_pfz.to_dict(orient="records")[0],
        "model_inference": {
            "predicted_class": pfz_pred_class,
            "pfz_probability": pfz_prob,
            "decision_threshold": 0.85,
            "pfz_present": pfz_prob >= 0.85
        }
    }
    print(f"✓ PFZ Live Feature Transformation: Schema Match={pfz_cols_match}, Zero NaNs={pfz_no_nulls}")
    print(f"  Inference Output: PFZ Prob={pfz_prob:.4f}, Present={pfz_prob >= 0.85}")

    # 2. Test Weather Risk Transformation
    df_weather = transform_raw_to_weather_features(mock_weather_raw_current, mock_weather_raw_3h_ago)
    weather_cols_match = (list(df_weather.columns) == WEATHER_EXPECTED_FEATURES)
    weather_no_nulls = not df_weather.isna().values.any()

    weather_probs = weather_model.predict_proba(df_weather)[0].tolist()
    weather_pred_class = int(weather_model.predict(df_weather)[0])
    class_map = {0: "NORMAL", 1: "CAUTION", 2: "DANGEROUS"}

    test_results["weather_live_transformation"] = {
        "schema_match": weather_cols_match,
        "no_nulls": weather_no_nulls,
        "num_features": len(df_weather.columns),
        "derived_features_check": {
            "sin_hour": df_weather.iloc[0]["sin_hour"],
            "cos_hour": df_weather.iloc[0]["cos_hour"],
            "wind_delta_3h": df_weather.iloc[0]["wind_delta_3h"],
            "pressure_delta_3h": df_weather.iloc[0]["pressure_delta_3h"]
        },
        "constructed_features": df_weather.to_dict(orient="records")[0],
        "model_inference": {
            "predicted_class": weather_pred_class,
            "class_label": class_map[weather_pred_class],
            "class_probabilities": weather_probs
        }
    }
    print(f"✓ Weather Live Feature Transformation: Schema Match={weather_cols_match}, Zero NaNs={weather_no_nulls}")
    print(f"  Derived Deltas: wind_delta_3h={df_weather.iloc[0]['wind_delta_3h']:.2f} m/s, pressure_delta_3h={df_weather.iloc[0]['pressure_delta_3h']:.1f} Pa")
    print(f"  Inference Output: Class={weather_pred_class} ({class_map[weather_pred_class]}), Probabilities={[round(p,4) for p in weather_probs]}")

    # 3. Test Feature Construction on REAL Project Datasets (ERA5 + ROMS NetCDF)
    print("  Testing Feature Extraction on REAL Project NetCDF Datasets...")
    era5_path = REPO_ROOT / "Datasets" / "processed" / "ERA5" / "era5_monthly_indian_ocean_1960_2026.nc"
    roms_path = REPO_ROOT / "Datasets" / "processed" / "INCOIS" / "ROMS" / "incois_roms_surface_current_20260828.nc"

    if era5_path.exists() and roms_path.exists():
        import xarray as xr
        with xr.open_dataset(era5_path) as era_ds, xr.open_dataset(roms_path) as roms_ds:
            lat_test, lon_test = 12.48, 74.40
            era_t = era_ds.sel(valid_time="2026-08-01", method="nearest")
            roms_t = roms_ds.isel(time=0)

            real_pfz_raw = {
                "latitude": lat_test,
                "longitude": lon_test,
                "timestamp": "2026-08-28T00:00:00Z",
                "sea_surface_temp_k": float(era_t["sst"].sel(latitude=lat_test, longitude=lon_test, method="nearest").values),
                "temp_2m_k": float(era_t["t2m"].sel(latitude=lat_test, longitude=lon_test, method="nearest").values),
                "u_wind_10m": float(era_t["u10"].sel(latitude=lat_test, longitude=lon_test, method="nearest").values),
                "v_wind_10m": float(era_t["v10"].sel(latitude=lat_test, longitude=lon_test, method="nearest").values),
                "mean_sea_level_pressure_pa": float(era_t["msl"].sel(latitude=lat_test, longitude=lon_test, method="nearest").values),
                "total_precipitation_m": float(era_t["tp"].sel(latitude=lat_test, longitude=lon_test, method="nearest").values),
                "ocean_current_u": float(roms_t["u_current"].sel(latitude=lat_test, longitude=lon_test, method="nearest").values),
                "ocean_current_v": float(roms_t["v_current"].sel(latitude=lat_test, longitude=lon_test, method="nearest").values),
            }

        df_real_pfz = transform_raw_to_pfz_features(real_pfz_raw)
        pfz_real_prob = float(pfz_model.predict_proba(df_real_pfz)[0, 1])

        test_results["pfz_real_dataset_pipeline"] = {
            "dataset_sources": [str(era5_path.name), str(roms_path.name)],
            "sample_lat_lon": [lat_test, lon_test],
            "schema_match": list(df_real_pfz.columns) == PFZ_EXPECTED_FEATURES,
            "no_nulls": not df_real_pfz.isna().values.any(),
            "pfz_probability": pfz_real_prob,
        }
        print(f"✓ Real Dataset Pipeline (ERA5 + ROMS): PFZ Prob={pfz_real_prob:.4f} at ({lat_test}, {lon_test})")

    out_file = RESULTS_DIR / "live_features_results.json"
    with open(out_file, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\n✓ Saved live feature validation results to {out_file}")
    print("=" * 80)
    return test_results


if __name__ == "__main__":
    run_live_feature_tests()
