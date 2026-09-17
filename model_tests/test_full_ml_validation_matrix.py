# model_tests/test_full_ml_validation_matrix.py

import pytest
import numpy as np
from Proto.agents.ocean.model import predict as predict_ocean
from Proto.agents.risk.model import predict as predict_marine_risk
from Proto.agents.pfz.model import predict as predict_pfz
from Proto.agents.weather.model import predict as predict_weather
from Proto.agents.productivity.model import predict_productivity


def test_ocean_suitability_model_standalone():
    """Validates Ocean Suitability Hierarchical XGBoost model execution and outputs."""
    res = predict_ocean(12.48, 74.40)
    assert isinstance(res, dict)
    assert "ocean_suitability_score" in res
    assert 0.0 <= res["ocean_suitability_score"] <= 100.0
    assert "predicted_log_cpue" in res
    assert "predicted_anomaly" in res
    assert res["model_version"] == "v1_hierarchical_xgboost"
    assert res["confidence_score"] in [0.95, 0.50]


def test_marine_risk_v2_model_standalone():
    """Validates Marine Risk v2 XGBoost 6-hour-ahead hazard prediction."""
    # Normal ocean conditions
    normal_env = {
        "latitude": 12.48,
        "longitude": 74.40,
        "u10": -2.5,
        "v10": 4.1,
        "wind_speed_10m": 4.8,
        "msl": 101200.0,
        "t2m": 298.2,
        "cyclone_distance_km": 9999.0
    }
    res = predict_marine_risk(normal_env)
    assert isinstance(res, dict)
    assert res["risk_level"] in ["SAFE", "CAUTION", "DANGER"]
    assert res["prediction_horizon_hours"] == 6
    assert "class_probabilities" in res
    probs = res["class_probabilities"]
    assert pytest.approx(sum(probs.values()), 1e-3) == 1.0
    assert res["model_version"] == "v2_xgboost"

    # Extreme Cyclone Proximity Edge Case
    cyclone_env = {**normal_env, "cyclone_distance_km": 100.0, "active_cyclone_flag": 1.0}
    res_cyclone = predict_marine_risk(cyclone_env)
    assert res_cyclone["risk_level"] == "DANGER"
    assert res_cyclone["allowed"] is False


def test_pfz_model_standalone():
    """Validates Potential Fishing Zone (PFZ) XGBoost classifier."""
    res = predict_pfz(12.48, 74.40)
    assert isinstance(res, dict)
    assert "pfz_probability" in res
    assert 0.0 <= res["pfz_probability"] <= 1.0
    assert isinstance(res["pfz_present"], bool)
    assert res["decision_threshold"] == 0.85
    assert res["model_version"] == "v1_xgboost"


def test_weather_risk_model_standalone():
    """Validates Weather Risk XGBoost classifier with safety thresholds."""
    # Normal calm weather
    calm_env = {"u10": 1.0, "v10": 1.0, "wind_speed_10m": 1.4, "msl": 101300.0, "t2m": 300.0, "tp": 0.0}
    res_calm = predict_weather(12.48, 74.40, raw_current=calm_env)
    assert isinstance(res_calm, dict)
    assert res_calm["risk_level"] in ["NORMAL", "CAUTION", "DANGEROUS"]
    assert "class_probabilities" in res_calm

    # Storm weather edge case
    storm_env = {"u10": 25.0, "v10": 20.0, "wind_speed_10m": 32.0, "msl": 98000.0, "tp": 0.05}
    res_storm = predict_weather(12.48, 74.40, raw_current=storm_env)
    assert res_storm["risk_level"] in ["CAUTION", "DANGEROUS"]
    assert res_storm["class_probabilities"]["DANGEROUS"] > res_calm["class_probabilities"]["DANGEROUS"]


def test_fish_productivity_lstm_standalone():
    """Validates 12-month LSTM Fish Productivity model execution and scaling."""
    env = {
        "wind_speed_10m": 4.5,
        "t2m": 299.5,
        "msl": 101100.0,
        "sst": 301.2,
        "tp": 0.002,
        "month": 8
    }
    res = predict_productivity(env)
    assert isinstance(res, dict)
    assert "productivity_score" in res
    assert 0.0 <= res["productivity_score"] <= 1.0
    assert "estimated_cpue" in res
    assert res["estimated_cpue"] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
