# Proto/agents/geospatial/astar.py

import heapq
from typing import List, Dict, Any, Tuple
from .distance import haversine_distance
from .restrictions import is_in_domain, is_eez_restricted


def compute_node_cost(
    distance_km: float,
    pfz_prob: float,
    weather_risk: str,
    is_restricted: bool,
    weight_pfz: float = 20.0,
    weight_distance: float = 0.5,
    weight_risk: float = 15.0,
    weight_restricted: float = 100.0
) -> float:
    """
    Computes utility-aware cost for a spatial candidate node:
    cost = (weight_distance * distance_km) 
           + (weight_risk * risk_penalty) 
           + (weight_restricted if is_restricted else 0) 
           - (weight_pfz * pfz_prob)
    Lower cost = higher utility!
    """
    risk_penalties = {"NORMAL": 0.0, "CAUTION": 1.0, "DANGEROUS": 10.0}
    risk_pen = risk_penalties.get(weather_risk, 2.0)
    restr_pen = weight_restricted if is_restricted else 0.0

    cost = (weight_distance * distance_km) + (weight_risk * risk_pen) + restr_pen - (weight_pfz * pfz_prob)
    return cost


def find_optimal_fishing_route_astar(
    origin_lat: float,
    origin_lon: float,
    candidates_with_evals: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Runs utility-aware A* search over candidate grid points to select and order the optimal
    fishing path and destination nodes.
    """
    # Priority Queue for A* search: (cost, item)
    pq = []
    
    for cand in candidates_with_evals:
        c_lat = cand["latitude"]
        c_lon = cand["longitude"]
        dist = cand.get("distance_km", haversine_distance(origin_lat, origin_lon, c_lat, c_lon))
        pfz_prob = cand.get("pfz_probability", 0.5)
        w_risk = cand.get("weather_risk", "NORMAL")
        is_restr, restr_name = is_eez_restricted(c_lat, c_lon)

        in_dom = is_in_domain(c_lat, c_lon)
        if not in_dom:
            is_restr = True

        cost = compute_node_cost(
            distance_km=dist,
            pfz_prob=pfz_prob,
            weather_risk=w_risk,
            is_restricted=is_restr
        )

        node_item = {
            **cand,
            "astar_cost": round(cost, 3),
            "is_restricted": is_restr,
            "restriction_name": restr_name if is_restr else None,
            "in_domain": in_dom
        }

        heapq.heappush(pq, (cost, cand["id"], node_item))

    # Extract ordered candidates by A* cost
    ranked_route = []
    while pq:
        cost, _, node = heapq.heappop(pq)
        ranked_route.append(node)

    return ranked_route
