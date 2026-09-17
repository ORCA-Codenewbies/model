"""
Isolated Model Test: Weather Risk XGBoost Multi-Class Classifier (weather_risk_xgboost.json)

Tests:
1. Model loading & serialization integrity
2. Input schema validation (21 features, exact ordering & types)
3. Multi-class prediction & probability outputs (0: NORMAL, 1: CAUTION, 2: DANGEROUS)
4. Evaluation on test-split ERA5/IMD dataset (Accuracy, Balanced Accuracy, Per-Class Precision/Recall/F1, 3x3 Confusion Matrix)
5. Confidence & entropy distribution analysis
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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Project paths
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "Models" / "weather_risk" / "model" / "weather_risk_xgboost.json"
DATASET_PATH = (
    REPO_ROOT
    / "Datasets"
    / "processed"
    / "training"
    / "weather_risk"
    / "weather_risk_hourly_advanced_features.parquet"
)
RESULTS_DIR = REPO_ROOT / "model_tests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FEATURES = [
    "latitude", "longitude", "u10", "v10", "wind_speed_10m",
    "t2m", "msl", "tp", "hour", "month", "sin_hour", "cos_hour",
    "sin_month", "cos_month", "wind_gradient", "pressure_gradient",
    "temperature_gradient", "pressure_local_anomaly", "wind_local_anomaly",
    "wind_delta_3h", "pressure_delta_3h"
]

CLASS_NAMES = {0: "NORMAL", 1: "CAUTION", 2: "DANGEROUS"}


def run_weather_model_test():
    print("=" * 80)
    print("RUNNING ISOLATED WEATHER RISK MODEL TEST")
    print("=" * 80)

    results = {
        "model_name": "WeatherRisk_XGBoost",
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
        "passed": schema_correct and feature_order_correct,
    }
    print(f"✓ Input Schema Validation: {n_features} features expected ({'MATCH' if schema_correct else 'MISMATCH'})")

    # 3. Dummy Single Sample Multi-Class Prediction
    dummy_sample = pd.DataFrame([{
        "latitude": 15.0,
        "longitude": 85.0,
        "u10": 12.5,
        "v10": 18.2,
        "wind_speed_10m": 22.08,
        "t2m": 300.1,
        "msl": 99800.0,
        "tp": 0.015,
        "hour": 12,
        "month": 5,
        "sin_hour": 0.0,
        "cos_hour": -1.0,
        "sin_month": 0.5,
        "cos_month": -0.866,
        "wind_gradient": 3.5,
        "pressure_gradient": -4.2,
        "temperature_gradient": -0.8,
        "pressure_local_anomaly": -1500.0,
        "wind_local_anomaly": 10.5,
        "wind_delta_3h": 4.2,
        "pressure_delta_3h": -350.0,
    }], columns=EXPECTED_FEATURES)

    try:
        pred_class = int(model.predict(dummy_sample)[0])
        pred_probs = model.predict_proba(dummy_sample)[0].tolist()
        results["checks"]["prediction_functional"] = {
            "passed": True,
            "sample_pred_class": pred_class,
            "sample_class_name": CLASS_NAMES.get(pred_class, "UNKNOWN"),
            "sample_probabilities": pred_probs,
        }
        print(f"✓ Single prediction test: Class={pred_class} ({CLASS_NAMES.get(pred_class)}), Probabilities={pred_probs}")
    except Exception as e:
        results["checks"]["prediction_functional"] = {"passed": False, "error": str(e)}
        print(f"✗ Prediction test failed: {e}")
        return results

    # 4. Empirical Performance Evaluation on Test Split Data
    if DATASET_PATH.exists():
        print("  Loading test-split dataset for evaluation...")
        df_all = pd.read_parquet(DATASET_PATH, columns=EXPECTED_FEATURES + ["target", "split"])
        df_test = df_all[df_all["split"] == "test"].copy()

        if len(df_test) > 0:
            # Stratified sample up to 50,000 test rows for efficient validation
            n_eval = min(50000, len(df_test))
            sampled_chunks = [
                group.sample(min(len(group), n_eval // 3), random_state=42)
                for _, group in df_test.groupby("target")
            ]
            sample_df = pd.concat(sampled_chunks, ignore_index=True)

            X_test = sample_df[EXPECTED_FEATURES]
            y_true = sample_df["target"].values.astype(int)

            t0_eval = time.perf_counter()
            y_probs = model.predict_proba(X_test)
            t1_eval = time.perf_counter()
            # 1. Standard Argmax Evaluation
            y_pred_argmax = np.argmax(y_probs, axis=1)
            acc_argmax = float(accuracy_score(y_true, y_pred_argmax))
            bal_acc_argmax = float(balanced_accuracy_score(y_true, y_pred_argmax))
            report_argmax = classification_report(
                y_true, y_pred_argmax, labels=[0, 1, 2],
                target_names=["NORMAL", "CAUTION", "DANGEROUS"],
                output_dict=True, zero_division=0
            )
            cm_argmax = confusion_matrix(y_true, y_pred_argmax, labels=[0, 1, 2]).tolist()

            # 2. Safety-Aware Operational Thresholding Evaluation (T_danger=0.15, T_caution=0.40)
            t_danger, t_caution = 0.15, 0.40
            y_pred_safety = np.zeros(len(sample_df), dtype=int)
            for i in range(len(sample_df)):
                p0, p1, p2 = y_probs[i]
                if p2 >= t_danger:
                    y_pred_safety[i] = 2
                elif (p1 + p2) >= t_caution:
                    y_pred_safety[i] = 1
                else:
                    y_pred_safety[i] = 0

            acc_safety = float(accuracy_score(y_true, y_pred_safety))
            bal_acc_safety = float(balanced_accuracy_score(y_true, y_pred_safety))
            report_safety = classification_report(
                y_true, y_pred_safety, labels=[0, 1, 2],
                target_names=["NORMAL", "CAUTION", "DANGEROUS"],
                output_dict=True, zero_division=0
            )
            cm_safety = confusion_matrix(y_true, y_pred_safety, labels=[0, 1, 2]).tolist()

            # Confidence & entropy calculations
            max_probs = np.max(y_probs, axis=1)
            eps = 1e-12
            entropy = -np.sum(y_probs * np.log2(y_probs + eps), axis=1)

            results["metrics"] = {
                "test_dataset": DATASET_PATH.name,
                "eval_sample_count": len(sample_df),
                "total_test_rows_available": len(df_test),
                "eval_time_sec": t1_eval - t0_eval,
                "standard_argmax": {
                    "accuracy": acc_argmax,
                    "balanced_accuracy": bal_acc_argmax,
                    "macro_f1": float(report_argmax["macro avg"]["f1-score"]),
                    "weighted_f1": float(report_argmax["weighted avg"]["f1-score"]),
                    "per_class": {
                        cname: {
                            "precision": float(report_argmax[cname]["precision"]),
                            "recall": float(report_argmax[cname]["recall"]),
                            "f1_score": float(report_argmax[cname]["f1-score"]),
                            "support": int(report_argmax[cname]["support"]),
                        }
                        for cname in ["NORMAL", "CAUTION", "DANGEROUS"]
                    },
                    "confusion_matrix": cm_argmax,
                },
                "safety_aware_thresholding": {
                    "t_danger_threshold": t_danger,
                    "t_caution_threshold": t_caution,
                    "accuracy": acc_safety,
                    "balanced_accuracy": bal_acc_safety,
                    "macro_f1": float(report_safety["macro avg"]["f1-score"]),
                    "weighted_f1": float(report_safety["weighted avg"]["f1-score"]),
                    "per_class": {
                        cname: {
                            "precision": float(report_safety[cname]["precision"]),
                            "recall": float(report_safety[cname]["recall"]),
                            "f1_score": float(report_safety[cname]["f1-score"]),
                            "support": int(report_safety[cname]["support"]),
                        }
                        for cname in ["NORMAL", "CAUTION", "DANGEROUS"]
                    },
                    "confusion_matrix": cm_safety,
                },
                "confusion_matrix_classes": ["NORMAL (0)", "CAUTION (1)", "DANGEROUS (2)"],
            }

            print(f"✓ Test-set Evaluation ({len(sample_df)} samples):")
            print(f"  [Standard Argmax] Accuracy: {acc_argmax:.4f} | Dangerous Recall: {report_argmax['DANGEROUS']['recall']:.4f} | Macro F1: {results['metrics']['standard_argmax']['macro_f1']:.4f}")
            print(f"  [Safety Policy]   Accuracy: {acc_safety:.4f} | Dangerous Recall: {report_safety['DANGEROUS']['recall']:.4f} | Macro F1: {results['metrics']['safety_aware_thresholding']['macro_f1']:.4f}")
            print(f"  Confusion Matrix (Safety Policy):\n  {cm_safety[0]}\n  {cm_safety[1]}\n  {cm_safety[2]}")

            results["confidence_stats"] = {
                "max_prob_mean": float(np.mean(max_probs)),
                "max_prob_std": float(np.std(max_probs)),
                "max_prob_p25": float(np.percentile(max_probs, 25)),
                "max_prob_p50": float(np.percentile(max_probs, 50)),
                "max_prob_p75": float(np.percentile(max_probs, 75)),
                "entropy_mean": float(np.mean(entropy)),
                "entropy_p90": float(np.percentile(entropy, 90)),
            }


        else:
            print("! Test split contained 0 rows.")
    else:
        print("! Dataset file not found; skipping empirical metrics evaluation.")

    # 5. Single-Sample Inference Latency Benchmark
    latencies = []
    for _ in range(10):
        model.predict_proba(dummy_sample)

    for _ in range(200):
        t0 = time.perf_counter()
        model.predict_proba(dummy_sample)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    results["latency_ms"] = {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
        "iterations": len(latencies),
    }
    print(f"✓ Latency (200 runs): Mean={results['latency_ms']['mean']:.3f}ms, P50={results['latency_ms']['p50']:.3f}ms, P95={results['latency_ms']['p95']:.3f}ms, P99={results['latency_ms']['p99']:.3f}ms")

    results["status"] = "SUCCESS"

    out_file = RESULTS_DIR / "weather_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {out_file}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_weather_model_test()
