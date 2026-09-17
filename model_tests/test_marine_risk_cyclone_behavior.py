# model_tests/test_marine_risk_cyclone_behavior.py

import pytest
from Proto.agents.risk.model import predict as predict_marine_risk


def test_marine_risk_cyclone_distance_boundaries():
    """Black-box behavioral boundary test verifying Marine Risk v2 response across cyclone distance thresholds."""
    base_env = {
        "latitude": 12.80,
        "longitude": 74.70,
        "u10": -2.5,
        "v10": 4.1,
        "wind_speed_10m": 4.8,
        "msl": 101200.0,
        "t2m": 298.2
    }

    # 1. No Active Cyclone (9999 km) -> SAFE
    res_no_cyclone = predict_marine_risk({**base_env, "cyclone_distance_km": 9999.0})
    assert res_no_cyclone["risk_level"] == "SAFE"
    assert res_no_cyclone["allowed"] is True

    # 2. Moderate Distant Cyclone (200 km) -> CAUTION (elevated risk, but not forced DANGER)
    res_200km = predict_marine_risk({**base_env, "cyclone_distance_km": 200.0, "active_cyclone_flag": 1.0})
    assert res_200km["risk_level"] == "CAUTION"
    assert res_200km["allowed"] is True

    # 3. Proximity Threshold Boundary (149 km) -> DANGER
    res_149km = predict_marine_risk({**base_env, "cyclone_distance_km": 149.0, "active_cyclone_flag": 1.0})
    assert res_149km["risk_level"] == "DANGER"
    assert res_149km["allowed"] is False

    # 4. Severe Cyclone Proximity (100 km) -> DANGER
    res_100km = predict_marine_risk({**base_env, "cyclone_distance_km": 100.0, "active_cyclone_flag": 1.0})
    assert res_100km["risk_level"] == "DANGER"
    assert res_100km["allowed"] is False


def test_marine_risk_active_cyclone_flag_sensitivity():
    """Validates model/feature sensitivity between active vs inactive cyclone flag states."""
    base_env = {
        "latitude": 12.80,
        "longitude": 74.70,
        "cyclone_distance_km": 300.0,
        "cyclone_wind_kt": 50.0,
        "u10": -2.5,
        "v10": 4.1
    }

    env_inactive = {**base_env, "active_cyclone_flag": 0.0}
    env_active = {**base_env, "active_cyclone_flag": 1.0}

    res_inactive = predict_marine_risk(env_inactive)
    res_active = predict_marine_risk(env_active)

    # Active cyclone flag must increase hazard probability
    p_hazard_inactive = res_inactive["class_probabilities"]["CAUTION"] + res_inactive["class_probabilities"]["DANGER"]
    p_hazard_active = res_active["class_probabilities"]["CAUTION"] + res_active["class_probabilities"]["DANGER"]

    assert p_hazard_active >= p_hazard_inactive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
