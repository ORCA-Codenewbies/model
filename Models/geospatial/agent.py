# Proto/agents/geospatial/agent.py

from datetime import datetime, timezone
from typing import Dict, Any
from schemas.contracts import BaseAgent, QueryPlan, AgentResult, GeoLocation, AgentAudit
from .distance import haversine_distance, bearing, compass_direction
from .grid import generate_candidate_grid_points
from .restrictions import is_in_domain, is_eez_restricted
from .astar import find_optimal_fishing_route_astar

class GeospatialAgent(BaseAgent):
    """
    Algorithmic Geospatial Agent
    
    Generates marine candidate grid points, checks EEZ boundary restrictions,
    and performs utility-aware spatial path planning.
    """
    @property
    def name(self) -> str:
        return "geospatial"

    def run(self, plan: QueryPlan, context: Dict[str, AgentResult]) -> AgentResult:
        origin_loc = plan.target_location or plan.location or plan.reference_location
        target_loc = plan.location or plan.target_location

        if not origin_loc:
            return AgentResult(
                agent=self.name,
                status="ERROR",
                location=None,
                timestamp=datetime.now(timezone.utc),
                data={"error": "LOCATION_REQUIRED", "allowed": False, "candidate_grid_points": []},
                confidence=1.0,
                sources=["ORCA_GEOSPATIAL_ENGINE_v1"],
                warnings=["LOCATION_REQUIRED: No valid spatial location provided in QueryPlan."]
            )

        is_fishing_search = plan.operation in ("SELECT_BEST_FISHING_OPTION", "NEAREST_PFZ_SEARCH", "ROUTE_TO_FISHING_AREA", "FIND_FISHING_SPOTS")
        if is_fishing_search:
            if not target_loc:
                from schemas.contracts import AgentAudit
                return AgentResult(
                    agent=self.name,
                    status="ERROR",
                    location=origin_loc,
                    timestamp=datetime.now(timezone.utc),
                    data={"error": "NO_MARINE_CANDIDATES", "allowed": False, "candidate_grid_points": []},
                    confidence=1.0,
                    sources=["ORCA_GEOSPATIAL_ENGINE_v1"],
                    warnings=["NO_MARINE_CANDIDATES: Inland target without coastal access requested."],
                    audit=AgentAudit(
                        inputs_used={"lat": origin_loc.latitude, "lon": origin_loc.longitude},
                        outputs={"allowed": False, "candidate_grid_points_count": 0},
                        output_reason="Inland target without coastal access requested.",
                        sources=["ORCA_GEOSPATIAL_ENGINE_v1"],
                        score_source=None,
                        score_reason=None
                    )
                )
            from location.location_metadata import INLAND_LOCATIONS, LOCATION_METADATA
            t_name = target_loc.name.lower() if target_loc.name else ""
            has_coastal_access = LOCATION_METADATA.get(t_name, {}).get("coastal_access", False)
            if t_name in INLAND_LOCATIONS and not has_coastal_access:
                return AgentResult(
                    agent=self.name,
                    status="ERROR",
                    location=origin_loc,
                    timestamp=datetime.now(timezone.utc),
                    data={"error": "NO_MARINE_CANDIDATES", "allowed": False, "candidate_grid_points": []},
                    confidence=1.0,
                    sources=["ORCA_GEOSPATIAL_ENGINE_v1"],
                    warnings=["NO_MARINE_CANDIDATES: Target candidate is an inland location."],
                    audit=AgentAudit(
                        inputs_used={"lat": origin_loc.latitude, "lon": origin_loc.longitude},
                        outputs={"allowed": False, "candidate_grid_points_count": 0},
                        output_reason="Target candidate is an inland location.",
                        sources=["ORCA_GEOSPATIAL_ENGINE_v1"],
                        score_source=None,
                        score_reason=None
                    )
                )

        lat = origin_loc.latitude
        lon = origin_loc.longitude
        loc_name = origin_loc.name if origin_loc.name else "Target Hub"

        # Check domain for origin point
        in_dom = is_in_domain(lat, lon)
        
        # Query marine boundaries
        is_restr = False
        restr_name = ""
        try:
            is_restr, restr_name = is_eez_restricted(lat, lon)
        except Exception as ex:
            print(f"Local EEZ fallback failed: {ex}")
            return AgentResult(
                agent=self.name,
                status="ERROR",
                location=GeoLocation(latitude=lat, longitude=lon, name=loc_name),
                timestamp=datetime.now(timezone.utc),
                data={"error": "BOUNDARY_VERIFICATION_FAILED", "allowed": "UNKNOWN", "candidate_grid_points": []},
                confidence=0.0,
                sources=[],
                warnings=["BOUNDARY_VERIFICATION_FAILED: Could not verify if location is in restricted zone."],
                audit=AgentAudit(
                    inputs_used={"lat": lat, "lon": lon},
                    outputs={"allowed": False, "candidate_grid_points_count": 0},
                    output_reason="Boundary verification failed.",
                    sources=[],
                    score_source=None,
                    score_reason=None
                )
            )

        warnings = []
        if not in_dom:
            warnings.append(f"UNSUPPORTED LOCATION: Coordinates ({lat:.2f}N, {lon:.2f}E) are outside the supported Indian Ocean domain.")
        if is_restr:
            warnings.append(f"ZONE RESTRICTION: Coordinates are inside {restr_name}.")

        # Generate spatial candidate grid points for PFZ location search
        cand_lat = target_loc.latitude if target_loc else lat
        cand_lon = target_loc.longitude if target_loc else lon
        candidates = generate_candidate_grid_points(cand_lat, cand_lon)

        # Distance to Coast calculation for geography / nearest_coast operations
        from location.gazetteer import GAZETTEER
        from location.location_metadata import LOCATION_METADATA

        meta = LOCATION_METADATA.get(loc_name.lower(), {})
        coastal_acc = meta.get("coastal_access", True)
        
        if coastal_acc or loc_name.lower() in ["kerala", "goa", "odisha", "tamil nadu", "andhra pradesh", "karnataka", "maharashtra", "gujarat", "west bengal"]:
            min_dist = 0.0
            nearest_coast_name = f"{loc_name} Coastline"
        else:
            # Find nearest coastal gazetteer point from origin coordinates
            min_dist = float("inf")
            nearest_coast_name = "Nearest Coast"
            for g_key, (g_lat, g_lon, g_name) in GAZETTEER.items():
                g_meta = LOCATION_METADATA.get(g_key, {})
                if g_meta.get("coastal_access", True) and g_key not in ["kolkata", "calcutta", "bengaluru", "pune", "ranchi"]:
                    dist = haversine_distance(lat, lon, g_lat, g_lon)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_coast_name = g_name

        payload = {
            "origin": {
                "latitude": lat,
                "longitude": lon,
                "name": loc_name,
                "in_domain": in_dom,
                "is_restricted": is_restr,
                "restriction_name": restr_name
            },
            "allowed": in_dom and not is_restr and min_dist == 0.0,
            "candidate_grid_points_count": len(candidates),
            "candidate_grid_points": candidates,
            "distance_to_coast_summary": {
                "origin_name": loc_name,
                "nearest_coast_name": nearest_coast_name,
                "distance_km": round(min_dist, 1)
            }
        }

        from schemas.contracts import AgentAudit
        
        audit_payload = {
            "allowed": payload["allowed"],
            "candidate_grid_points_count": payload["candidate_grid_points_count"]
        }
        
        return AgentResult(
            agent=self.name,
            status="SUCCESS",
            location=GeoLocation(latitude=lat, longitude=lon, name=loc_name),
            timestamp=datetime.now(timezone.utc),
            data=payload,
            confidence=0.98,
            sources=["ORCA_GEOSPATIAL_ENGINE_v1", "HAVERSINE_GRID_v1"],
            warnings=warnings,
            audit=AgentAudit(
                inputs_used={"lat": lat, "lon": lon},
                outputs=audit_payload,
                output_reason="Generated candidate marine locations and evaluated geospatial boundaries.",
                sources=["ORCA_GEOSPATIAL_ENGINE_v1", "HAVERSINE_GRID_v1"],
                score_source=None,
                score_reason=None
            )
        )