"""
Model Latency & Throughput Benchmark Suite (test_model_latency.py)

Evaluates PFZ (15 features) and Weather Risk (21 features) XGBoost models under:
- Single-sample latency (N=1)
- Micro-batch latency (N=10)
- Medium-batch latency (N=100)
- Large-batch latency (N=1000)

Metrics recorded per batch size:
- Mean, Std, Min, Max, P50, P95, P99 (ms)
- Throughput (predictions / sec)
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

PFZ_FEATURES = [
    "latitude", "longitude", "era_sst", "era_u10", "era_v10",
    "era_t2m", "era_msl", "era_tp", "wind_speed", "current_u",
    "current_v", "current_speed", "current_direction_rad", "month", "day_of_year"
]

WEATHER_FEATURES = [
    "latitude", "longitude", "u10", "v10", "wind_speed_10m",
    "t2m", "msl", "tp", "hour", "month", "sin_hour", "cos_hour",
    "sin_month", "cos_month", "wind_gradient", "pressure_gradient",
    "temperature_gradient", "pressure_local_anomaly", "wind_local_anomaly",
    "wind_delta_3h", "pressure_delta_3h"
]


def make_dummy_df(features, batch_size):
    np.random.seed(42)
    data = {col: np.random.randn(batch_size).tolist() for col in features}
    return pd.DataFrame(data, columns=features)


def benchmark_model(model, features, model_name, iterations=100):
    print(f"\n--- Benchmarking {model_name} ---")
    batch_sizes = [1, 10, 100, 1000]
    benchmarks = {}

    for batch_size in batch_sizes:
        df_batch = make_dummy_df(features, batch_size)
        
        # Warmup
        for _ in range(5):
            model.predict_proba(df_batch)

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            model.predict_proba(df_batch)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(latencies))
        p50_ms = float(np.percentile(latencies, 50))
        p95_ms = float(np.percentile(latencies, 95))
        p99_ms = float(np.percentile(latencies, 99))
        throughput_qps = float(batch_size / (mean_ms / 1000.0))

        benchmarks[f"batch_{batch_size}"] = {
            "batch_size": batch_size,
            "iterations": iterations,
            "mean_ms": mean_ms,
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
            "p99_ms": p99_ms,
            "throughput_preds_per_sec": throughput_qps,
        }
        print(f"Batch={batch_size:4d} | P50={p50_ms:6.3f}ms | P95={p95_ms:6.3f}ms | P99={p99_ms:6.3f}ms | Throughput={throughput_qps:8.1f} preds/sec")

    return benchmarks


def run_latency_benchmark():
    print("=" * 80)
    print("RUNNING ISOLATED MODEL LATENCY & THROUGHPUT BENCHMARK")
    print("=" * 80)

    pfz_model = xgb.XGBClassifier()
    pfz_model.load_model(str(PFZ_MODEL_PATH))

    weather_model = xgb.XGBClassifier()
    weather_model.load_model(str(WEATHER_MODEL_PATH))

    results = {
        "pfz_xgboost": benchmark_model(pfz_model, PFZ_FEATURES, "PFZ Model (15 features)"),
        "weather_risk_xgboost": benchmark_model(weather_model, WEATHER_FEATURES, "Weather Risk Model (21 features)")
    }

    out_file = RESULTS_DIR / "latency_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved latency benchmark results to {out_file}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_latency_benchmark()
