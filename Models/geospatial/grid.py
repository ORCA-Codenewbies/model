# Proto/agents/geospatial/grid.py

import math
from typing import List, Dict, Any
from .distance import haversine_distance, bearing, compass_direction

EARTH_RADIUS_KM = 6371.0

def offset_coordinate(lat: float, lon: float, distance_km: float, bearing_deg: float) -> tuple[float, float]:
    """
    Computes destination (lat, lon) given start point, distance (km), and bearing (degrees).
    """
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)
    brng = math.radians(bearing_deg)
    d_r = distance_km / EARTH_RADIUS_KM

    phi2 = math.asin(
        math.sin(phi1) * math.cos(d_r) +
        math.cos(phi1) * math.sin(d_r) * math.cos(brng)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(brng) * math.sin(d_r) * math.cos(phi1),
        math.cos(d_r) - math.sin(phi1) * math.sin(phi2)
    )

    return math.degrees(phi2), math.degrees(lambda2)


def generate_candidate_grid_points(
    origin_lat: float,
    origin_lon: float,
    radii_km: List[float] = None,
    bearings_deg: List[float] = None
) -> List[Dict[str, Any]]:
    """
    Generates structured marine candidate grid points around an origin location (e.g. Digha, Kochi).
    """
    if radii_km is None:
        radii_km = [5.0, 10.0, 15.0, 20.0, 25.0]
    if bearings_deg is None:
        bearings_deg = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

    candidates = []
    # Always include origin point
    candidates.append({
        "id": "origin",
        "latitude": round(origin_lat, 4),
        "longitude": round(origin_lon, 4),
        "distance_km": 0.0,
        "bearing_deg": 0.0,
        "compass_direction": "CENTER"
    })

    # Generate grid points
    idx = 1
    for r in radii_km:
        for b in bearings_deg:
            plat, plon = offset_coordinate(origin_lat, origin_lon, r, b)
            candidates.append({
                "id": f"cand_{idx}",
                "latitude": round(plat, 4),
                "longitude": round(plon, 4),
                "distance_km": r,
                "bearing_deg": b,
                "compass_direction": compass_direction(b)
            })
            idx += 1

    return candidates
