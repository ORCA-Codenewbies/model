# model_tests/inspect_keras_models.py

import os
import sys
import time
from pathlib import Path
import numpy as np

# Suppress TF logging clutter
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    try:
        import keras
    except ImportError:
        print("ERROR: Neither TensorFlow nor Keras library is installed in the active environment.")
        sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILES = [
    REPO_ROOT / "Models" / "pfz" / "model" / "pfz_cnn_lstm_smoketest.keras",
    REPO_ROOT / "Models" / "pfz" / "model" / "pfz_cnn_lstm_v1_development.keras",
    REPO_ROOT / "Models" / "fish_productivity" / "model" / "fish_productivity_lstm.keras",
    REPO_ROOT / "Models" / "fish_productivity" / "model" / "fish_productivity_lstm_candidate.keras",
    REPO_ROOT / "Models" / "fish_productivity" / "model" / "fish_productivity_lstm_12m_env.keras",
]

def inspect_keras_model(model_path: Path) -> dict:
    if not model_path.exists():
        return {"file_name": model_path.name, "status": "FILE_NOT_FOUND"}

    rel_path = model_path.relative_to(REPO_ROOT)
    file_size_mb = model_path.stat().st_size / (1024 * 1024)

    t0_load = time.perf_counter()
    try:
        model = keras.models.load_model(str(model_path), compile=False)
        load_time_ms = (time.perf_counter() - t0_load) * 1000.0
    except Exception as e:
        return {
            "file_name": model_path.name,
            "relative_path": str(rel_path),
            "status": "LOAD_ERROR",
            "error": str(e),
            "file_size_mb": round(file_size_mb, 2)
        }

    input_shape = model.input_shape
    output_shape = model.output_shape
    total_params = model.count_params()

    # Build dummy input array matching input signature
    try:
        if isinstance(input_shape, list):
            dummy_inputs = []
            for shape in input_shape:
                s = [1 if (dim is None or dim == -1) else dim for dim in shape]
                dummy_inputs.append(np.ones(s, dtype=np.float32))
        else:
            s = [1 if (dim is None or dim == -1) else dim for dim in input_shape]
            dummy_inputs = np.ones(s, dtype=np.float32)

        # Warm-up inference
        _ = model.predict(dummy_inputs, verbose=0)

        # Benchmark 50 inference passes
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            out = model.predict(dummy_inputs, verbose=0)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(latencies))
        p50_ms = float(np.percentile(latencies, 50))
        p95_ms = float(np.percentile(latencies, 95))

        out_array = np.array(out)
        out_min = float(np.min(out_array))
        out_max = float(np.max(out_array))
        out_sample = out_array.tolist()

        return {
            "file_name": model_path.name,
            "relative_path": str(rel_path),
            "status": "SUCCESS",
            "file_size_mb": round(file_size_mb, 2),
            "load_time_ms": round(load_time_ms, 2),
            "input_shape": str(input_shape),
            "output_shape": str(output_shape),
            "total_params": total_params,
            "mean_inference_ms": round(mean_ms, 2),
            "p50_inference_ms": round(p50_ms, 2),
            "p95_inference_ms": round(p95_ms, 2),
            "output_range": [round(out_min, 4), round(out_max, 4)],
            "sample_output": out_sample,
            "layers": [f"{l.name} ({l.__class__.__name__})" for l in model.layers[:5]]
        }
    except Exception as e:
        return {
            "file_name": model_path.name,
            "relative_path": str(rel_path),
            "status": "INFERENCE_ERROR",
            "error": str(e),
            "file_size_mb": round(file_size_mb, 2),
            "input_shape": str(input_shape),
            "output_shape": str(output_shape),
            "total_params": total_params
        }


def main():
    print("=" * 80)
    print("STANDALONE MODEL FORENSICS: Keras Deep Learning Models Inspection")
    print("=" * 80)

    results = []
    for path in MODEL_FILES:
        print(f"\nInspecting: {path.relative_to(REPO_ROOT)}...")
        res = inspect_keras_model(path)
        results.append(res)
        
        if res["status"] == "SUCCESS":
            print(f"  ✓ Loaded in {res['load_time_ms']} ms | Size: {res['file_size_mb']} MB | Params: {res['total_params']:,}")
            print(f"  ✓ Input Shape : {res['input_shape']}")
            print(f"  ✓ Output Shape: {res['output_shape']}")
            print(f"  ✓ Latency     : Mean={res['mean_inference_ms']}ms | P50={res['p50_inference_ms']}ms | P95={res['p95_inference_ms']}ms")
            print(f"  ✓ Output Range: {res['output_range']} | Sample Output: {res['sample_output']}")
            print(f"  ✓ First Layers: {', '.join(res['layers'])}")
        else:
            print(f"  ❌ Status: {res['status']} | Error: {res.get('error', 'N/A')}")

    print("\n" + "=" * 80)
    print("SUMMARY FORENSICS REPORT")
    print("=" * 80)
    print(f"{'Model File':<35} | {'Input Shape':<18} | {'Output Shape':<12} | {'P50 Latency':<10} | {'Status'}")
    print("-" * 88)
    for r in results:
        fname = r["file_name"]
        inp = r.get("input_shape", "N/A")
        outp = r.get("output_shape", "N/A")
        lat = f"{r.get('p50_inference_ms', 'N/A')} ms" if r["status"] == "SUCCESS" else "N/A"
        st = r["status"]
        print(f"{fname:<35} | {inp:<18} | {outp:<12} | {lat:<10} | {st}")
    print("=" * 80)


if __name__ == "__main__":
    main()
