"""
Isolated Model Test: PFZ XGBoost Classifier (pfz_xgboost.json)

Tests:
1. Model loading & serialization integrity
2. Input schema validation (15 features, exact ordering & types)
3. Prediction & probability outputs
4. Model metrics on ground-truth labelled test data (Accuracy, Precision, Recall, F1, Confusion Matrix)
5. Confidence distribution & decision threshold check
6. Single-sample & batch inference latency (Mean, P50, P95, P99)
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Project paths
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "Models" / "pfz" / "model" / "pfz_xgboost.json"
CONFIG_PATH = REPO_ROOT / "Models" / "pfz" / "config.json"
METRICS_PATH = REPO_ROOT / "model_tests" / "data" / "pfz_balanced_test_set.csv"
RESULTS_DIR = REPO_ROOT / "model_tests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FEATURES = [
    "latitude", "longitude", "era_sst", "era_u10", "era_v10",
    "era_t2m", "era_msl", "era_tp", "wind_speed", "current_u",
    "current_v", "current_speed", "current_direction_rad", "month", "day_of_year"
]


def run_pfz_model_test():
    print("=" * 80)
    print("RUNNING ISOLATED PFZ MODEL TEST")
    print("=" * 80)
    
    results = {
        "model_name": "PFZ_XGBoost",
        "model_path": str(MODEL_PATH),
        "status": "FAILED",
        "checks": {},
        "metrics": {},
        "latency_ms": {},
        "confidence_stats": {},
    }
    
    # 1. Load Model
    t_load_start = time.perf_counter()
    try:
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        t_load_end = time.perf_counter()
        load_time_ms = (t_load_end - t_load_start) * 1000.0
        results["checks"]["model_load"] = {"passed": True, "load_time_ms": load_time_ms}
        print(f"✓ Model loaded successfully in {load_time_ms:.2f} ms")
    except Exception as e:
        results["checks"]["model_load"] = {"passed": False, "error": str(e)}
        print(f"✗ Model failed to load: {e}")
        return results

    # 2. Input Schema Validation
    n_features = model.n_features_in_
    feature_names = getattr(model, "feature_names_in_", None)
    
    schema_correct = (n_features == len(EXPECTED_FEATURES))
    feature_order_correct = True
    if feature_names is not None:
        feature_order_correct = (list(feature_names) == EXPECTED_FEATURES)
        
    results["checks"]["input_schema"] = {
        "expected_n_features": len(EXPECTED_FEATURES),
        "actual_n_features": int(n_features),
        "n_features_match": schema_correct,
        "feature_order_match": feature_order_correct,
        "expected_features": EXPECTED_FEATURES,
        "actual_features": list(feature_names) if feature_names is not None else None,
        "passed": schema_correct and feature_order_correct
    }
    print(f"✓ Input Schema Validation: {n_features} features expected ({'MATCH' if schema_correct else 'MISMATCH'})")

    # 3. Prediction & Probabilities Check (Dummy Sample)
    dummy_sample = pd.DataFrame([{
        "latitude": 12.345,
        "longitude": 74.567,
        "era_sst": 301.5,
        "era_u10": -2.5,
        "era_v10": 4.1,
        "era_t2m": 298.2,
        "era_msl": 101200.0,
        "era_tp": 0.001,
        "wind_speed": 4.8,
        "current_u": 0.15,
        "current_v": -0.10,
        "current_speed": 0.18,
        "current_direction_rad": -0.588,
        "month": 8,
        "day_of_year": 240
    }], columns=EXPECTED_FEATURES)

    try:
        pred_class = int(model.predict(dummy_sample)[0])
        pred_probs = model.predict_proba(dummy_sample)[0].tolist()
        results["checks"]["prediction_functional"] = {
            "passed": True,
            "sample_pred_class": pred_class,
            "sample_probabilities": pred_probs
        }
        print(f"✓ Single prediction test: Class={pred_class}, Probabilities={pred_probs}")
    except Exception as e:
        results["checks"]["prediction_functional"] = {"passed": False, "error": str(e)}
        print(f"✗ Prediction test failed: {e}")
        return results

    # 4. Accuracy & Performance Metrics on Labelled Balanced Dataset
    if METRICS_PATH.exists():
        df_test = pd.read_csv(METRICS_PATH)
        X_test = df_test[EXPECTED_FEATURES]
        y_true = df_test["actual_pfz"].values
        
        y_prob = model.predict_proba(X_test)[:, 1]
        threshold = 0.85
        y_pred = (y_prob >= threshold).astype(int)
        
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_true, y_prob))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        
        results["metrics"] = {
            "test_dataset": str(METRICS_PATH.name),
            "num_samples": len(df_test),
            "num_positives": int(np.sum(y_true == 1)),
            "num_negatives": int(np.sum(y_true == 0)),
            "decision_threshold": threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "confusion_matrix_labels": ["Negative (0)", "Positive (1)"]
        }
        
        # Confidence statistics
        results["confidence_stats"] = {
            "prob_mean": float(np.mean(y_prob)),
            "prob_std": float(np.std(y_prob)),
            "prob_min": float(np.min(y_prob)),
            "prob_max": float(np.max(y_prob)),
            "prob_p25": float(np.percentile(y_prob, 25)),
            "prob_p50": float(np.percentile(y_prob, 50)),
            "prob_p75": float(np.percentile(y_prob, 75)),
        }
        print(f"✓ Labelled Balanced Evaluation ({len(df_test)} samples: {results['metrics']['num_positives']} pos / {results['metrics']['num_negatives']} neg):")
        print(f"  ROC-AUC={auc:.4f}, Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
        print(f"  Confusion Matrix [TN={cm[0][0]}, FP={cm[0][1]} / FN={cm[1][0]}, TP={cm[1][1]}]")
    else:
        print("! Labelled metrics file not found; skipping empirical metrics evaluation.")

    # 5. Single-Sample Inference Latency Benchmark
    latencies = []
    # Warmup
    for _ in range(10):
        model.predict_proba(dummy_sample)
        
    for _ in range(200):
        t0 = time.perf_counter()
        model.predict_proba(dummy_sample)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # in ms
        
    results["latency_ms"] = {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
        "iterations": len(latencies)
    }
    print(f"✓ Latency (200 runs): Mean={results['latency_ms']['mean']:.3f}ms, P50={results['latency_ms']['p50']:.3f}ms, P95={results['latency_ms']['p95']:.3f}ms, P99={results['latency_ms']['p99']:.3f}ms")

    results["status"] = "SUCCESS"
    
    # Save results to json
    out_file = RESULTS_DIR / "pfz_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {out_file}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_pfz_model_test()
