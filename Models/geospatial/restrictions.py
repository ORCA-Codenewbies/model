# Proto/agents/geospatial/restrictions.py

from typing import Dict, Any, Tuple

# Domain bounds for Indian Ocean Operational Region
MIN_LAT, MAX_LAT = 5.0, 25.0
MIN_LON, MAX_LON = 65.0, 95.0

# Known EEZ / Restricted Polygons / Bounding Boxes (e.g. Naval Exercise / Protected Sanctuary)
RESTRICTED_BOUNDARIES = [
    # Polygon 1: Offshore Security Restriction Zone
    {"name": "Naval Defense Restricted Zone", "min_lat": 9.90, "max_lat": 10.10, "min_lon": 84.80, "max_lon": 85.20},
    # Polygon 2: Marine Biosphere Core Sanctuary
    {"name": "Gulf of Mannar Marine Sanctuary", "min_lat": 8.80, "max_lat": 9.20, "min_lon": 78.80, "max_lon": 79.30}
]

def is_in_domain(lat: float, lon: float) -> bool:
    """
    Checks if coordinates fall within the supported Indian Ocean domain (5°N–25°N, 65°E–95°E).
    """
    return MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON


def is_eez_restricted(lat: float, lon: float) -> Tuple[bool, str]:
    """
    Checks if coordinates fall inside any restricted EEZ or security zone.
    Returns (is_restricted, zone_name).
    """
    for zone in RESTRICTED_BOUNDARIES:
        if zone["min_lat"] <= lat <= zone["max_lat"] and zone["min_lon"] <= lon <= zone["max_lon"]:
            return True, zone["name"]
    return False, "AUTHORIZED_FISHING_ZONE"
