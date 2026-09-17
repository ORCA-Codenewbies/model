# model_tests/test_real_data_multi_location.py

import pytest
import numpy as np
from Proto.agents.ocean.model import predict as predict_ocean
from Proto.agents.risk.model import predict as predict_marine_risk
from Proto.agents.pfz.model import predict as predict_pfz
from Proto.agents.weather.model import predict as predict_weather
from Proto.agents.productivity.model import predict_productivity


LOCATIONS = [
    {"name": "Mangalore Offshore (Arabian Sea)", "lat": 12.80, "lon": 74.70},
    {"name": "Vizag Offshore (Bay of Bengal)", "lat": 17.60, "lon": 83.35},
    {"name": "Port Blair Offshore (Andaman Sea)", "lat": 11.60, "lon": 92.80},
    {"name": "Deep Offshore (High Seas)", "lat": 10.00, "lon": 80.00}
]


def test_real_data_multi_location_execution():
    """Validates real NetCDF dataset feature pipeline and dynamic output variance across 4 distinct locations."""
    ocean_scores = []
    pfz_probs = []
    sst_values = []
    wind_speeds = []

    for loc in LOCATIONS:
        lat, lon = loc["lat"], loc["lon"]
        print(f"\n--- Testing Location: {loc['name']} ({lat}N, {lon}E) ---")

        # 1. Ocean Agent
        ocean_res = predict_ocean(lat, lon)
        assert ocean_res["source_type"] == "REAL_NETCDF"
        assert "sst" in ocean_res
        assert not np.isnan(ocean_res["sst"])
        assert "ocean_suitability_score" in ocean_res
        ocean_scores.append(ocean_res["ocean_suitability_score"])
        sst_values.append(ocean_res["sst"])

        # 2. Weather Agent
        weather_res = predict_weather(lat, lon, raw_current=ocean_res)
        assert "risk_level" in weather_res
        wind_speeds.append(weather_res["features"]["wind_speed_10m"])

        # 3. Marine Risk Agent
        marine_res = predict_marine_risk(ocean_res)
        assert marine_res["prediction_horizon_hours"] == 6
        assert marine_res["model_version"] == "v2_xgboost"

        # 4. PFZ Agent
        pfz_res = predict_pfz(lat, lon, env_data=ocean_res)
        assert "pfz_probability" in pfz_res
        pfz_probs.append(pfz_res["pfz_probability"])

        # 5. Productivity Agent
        prod_res = predict_productivity(ocean_res)
        assert "productivity_score" in prod_res

    # ASSERT DYNAMIC VARIANCE ACROSS GEOGRAPHICAL LOCATIONS
    print("\nVariance checks:")
    print("SST Values across locations:", sst_values)
    print("Ocean Suitability Scores:", ocean_scores)
    print("PFZ Probabilities:", pfz_probs)
    print("Wind Speeds:", wind_speeds)

    # Environmental inputs must vary by location
    assert np.var(sst_values) > 1e-4, "SST should vary across geographically distinct locations"
    assert np.var(wind_speeds) > 1e-4, "Wind speeds should vary across locations"
    # Model predictions must vary dynamically
    assert len(set(pfz_probs)) > 1 and np.var(pfz_probs) > 1e-4, "PFZ probability must vary dynamically by location"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
