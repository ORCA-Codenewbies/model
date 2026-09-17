# model_tests/benchmark_engine_breakdown.py

import sys
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

PROTO_DIR = Path(__file__).resolve().parents[1] / "Proto"
if str(PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_DIR))

from schemas.contracts import QueryPlan, GeoLocation, QueryTime
from agents.weather.agent import WeatherAgent
from agents.ocean.agent import OceanAgent
from agents.geospatial.agent import GeospatialAgent
from agents.pfz.agent import PFZAgent
from agents.productivity.agent import FishProductivityAgent
from agents.rules.safety_agent import SafetyRuleAgent
from agents.rules.recommendation_agent import RecommendationAgent

from agents.ocean.model import load_environmental_datasets
from agents.pfz.model import get_pfz_model
from agents.weather.model import get_weather_model
from agents.productivity.model import get_productivity_model
from orchestrator.engine import OrcaOrchestrator


def build_orchestrator() -> OrcaOrchestrator:
    registry = {
        "weather": WeatherAgent(),
        "ocean": OceanAgent(),
        "pfz": PFZAgent(),
        "geospatial": GeospatialAgent(),
        "productivity": FishProductivityAgent(),
        "safety_rules": SafetyRuleAgent(),
        "recommendation": RecommendationAgent(),
    }
    return OrcaOrchestrator(registry=registry)


def benchmark_workflow(engine: OrcaOrchestrator, name: str, agents: list[str], runs: int = 20) -> dict:
    plan = QueryPlan(
        query=f"Benchmark query for {name}",
        intent="marine_safety" if "productivity" not in agents else "pfz_search",
        result_type="SAFETY_ASSESSMENT",
        language="en",
        location=GeoLocation(latitude=12.48, longitude=74.40, name="Benchmark Hub"),
        agents=agents,
        time=QueryTime(exact=datetime.now(timezone.utc))
    )

    # Warmup
    _ = engine.run(plan)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = engine.run(plan)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    p50_ms = float(np.percentile(latencies, 50))
    p95_ms = float(np.percentile(latencies, 95))

    return {
        "workflow": name,
        "agents_executed": agents,
        "mean_ms": round(mean_ms, 2),
        "p50_ms": round(p50_ms, 2),
        "p95_ms": round(p95_ms, 2)
    }


def main():
    print("=" * 80)
    print("ORCA ENGINE EXECUTION LATENCY BREAKDOWN BENCHMARK")
    print("=" * 80)

    print("Pre-warming datasets and ML models...")
    load_environmental_datasets()
    get_pfz_model()
    get_weather_model()
    get_productivity_model()
    print("✓ Pre-warming complete.\n")

    engine = build_orchestrator()

    workflows = [
        ("1. Single-Point Safety Query", ["weather", "ocean", "safety_rules", "recommendation"]),
        ("2. PFZ Search (No Productivity)", ["weather", "ocean", "pfz", "safety_rules", "recommendation"]),
        ("3. PFZ Search + Geospatial (No Productivity)", ["weather", "ocean", "pfz", "geospatial", "safety_rules", "recommendation"]),
        ("4. Full Candidate Search + Productivity LSTM", ["weather", "ocean", "pfz", "geospatial", "productivity", "safety_rules", "recommendation"]),
    ]

    results = []
    for name, agents in workflows:
        res = benchmark_workflow(engine, name, agents)
        results.append(res)
        print(f"Workflow: {res['workflow']:<45} | Mean: {res['mean_ms']:6.2f} ms | P50: {res['p50_ms']:6.2f} ms | P95: {res['p95_ms']:6.2f} ms")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Workflow Description':<45} | {'P50 Latency':<12} | {'P95 Latency':<12}")
    print("-" * 75)
    for r in results:
        print(f"{r['workflow']:<45} | {r['p50_ms']:6.2f} ms    | {r['p95_ms']:6.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
