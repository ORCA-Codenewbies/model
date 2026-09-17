from pathlib import Path
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight


DATASET = Path(
    "Datasets/processed/training/weather_risk/"
    "weather_risk_hourly_advanced_features.parquet"
)

MODEL_DIR = Path("Models/weather_risk/model")
METRICS_DIR = Path("Models/weather_risk/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "latitude",
    "longitude",
    "u10",
    "v10",
    "wind_speed_10m",
    "t2m",
    "msl",
    "tp",
    "hour",
    "month",
    "sin_hour",
    "cos_hour",
    "sin_month",
    "cos_month",
    "wind_gradient",
    "pressure_gradient",
    "temperature_gradient",
    "pressure_local_anomaly",
    "wind_local_anomaly",
    "wind_delta_3h",
    "pressure_delta_3h",
]

CLASS_NAMES = {
    0: "NORMAL",
    1: "CAUTION",
    2: "DANGEROUS",
}

CLASS_IDS = [0, 1, 2]


def choose_device():
    requested = os.environ.get(
        "ORCA_XGB_DEVICE",
        "auto",
    ).lower()

    if requested in {"cpu", "cuda"}:
        return requested

    return "cpu"


def evaluate(name, y_true, probabilities):
    pred = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_true,
        pred,
        labels=CLASS_IDS,
        target_names=[
            CLASS_NAMES[i]
            for i in CLASS_IDS
        ],
        output_dict=True,
        zero_division=0,
    )

    per_class = {}

    for class_id in CLASS_IDS:
        cname = CLASS_NAMES[class_id]
        row = report[cname]

        per_class[cname] = {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1-score"]),
            "support": int(row["support"]),
        }

    return {
        "split": name,
        "n_samples": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=CLASS_IDS,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true,
            pred,
            labels=CLASS_IDS,
        ).tolist(),
    }


print("=" * 90)
print("ORCA WEATHER RISK — ADVANCED HOURLY XGBOOST")
print("=" * 90)

df = pd.read_parquet(DATASET)

print("\nDataset shape:", df.shape)

required = FEATURES + [
    "target",
    "target_name",
    "split",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )

train = df[df["split"] == "train"]
val = df[df["split"] == "validation"]
test = df[df["split"] == "test"]

X_train = train[FEATURES]
y_train = train["target"].astype(np.int32).to_numpy()

X_val = val[FEATURES]
y_val = val["target"].astype(np.int32).to_numpy()

X_test = test[FEATURES]
y_test = test["target"].astype(np.int32).to_numpy()

print("\nTrain:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nTrain classes:")
print(train["target_name"].value_counts().to_string())

print("\nValidation classes:")
print(val["target_name"].value_counts().to_string())

print("\nTest classes:")
print(test["target_name"].value_counts().to_string())

sample_weight = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

device = choose_device()

print("\nDevice:", device)

model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,

    n_estimators=1800,
    learning_rate=0.045,

    max_depth=9,
    min_child_weight=8,

    subsample=0.85,
    colsample_bytree=0.92,

    reg_alpha=0.08,
    reg_lambda=2.5,
    gamma=0.05,

    tree_method="hist",
    device=device,

    eval_metric="mlogloss",
    early_stopping_rounds=100,

    random_state=42,
    n_jobs=-1,
)

print("\nStarting training...")

t0 = time.perf_counter()

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weight,
    eval_set=[
        (X_val, y_val),
    ],
    verbose=50,
)

training_seconds = (
    time.perf_counter() - t0
)

best_iteration = getattr(
    model,
    "best_iteration",
    None,
)

print(
    f"\nTraining completed in "
    f"{training_seconds:.2f} seconds"
)

print("Best iteration:", best_iteration)

print("\nGenerating predictions...")

train_prob = model.predict_proba(X_train)
val_prob = model.predict_proba(X_val)
test_prob = model.predict_proba(X_test)

train_metrics = evaluate(
    "train",
    y_train,
    train_prob,
)

val_metrics = evaluate(
    "validation",
    y_val,
    val_prob,
)

test_metrics = evaluate(
    "test",
    y_test,
    test_prob,
)

# ------------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------------
importance = {}

for feature, value in zip(
    FEATURES,
    model.feature_importances_,
):
    importance[feature] = float(value)

importance = dict(
    sorted(
        importance.items(),
        key=lambda item: item[1],
        reverse=True,
    )
)

# ------------------------------------------------------------------
# Inference benchmark.
# ------------------------------------------------------------------
sample = X_test.iloc[[0]]

for _ in range(20):
    model.predict_proba(sample)

latencies = []

for _ in range(200):
    a = time.perf_counter()
    model.predict_proba(sample)
    b = time.perf_counter()

    latencies.append(
        (b - a) * 1000.0
    )

latencies = np.asarray(latencies)

benchmark = {
    "p50_ms": float(
        np.percentile(latencies, 50)
    ),
    "p95_ms": float(
        np.percentile(latencies, 95)
    ),
    "mean_ms": float(
        np.mean(latencies)
    ),
    "min_ms": float(
        np.min(latencies)
    ),
    "max_ms": float(
        np.max(latencies)
    ),
}

# ------------------------------------------------------------------
# Save model.
# ------------------------------------------------------------------
model_path = (
    MODEL_DIR /
    "weather_risk_xgboost.json"
)

model.save_model(model_path)

# ------------------------------------------------------------------
# Save metrics.
# ------------------------------------------------------------------
result = {
    "model": "weather_risk_xgboost",
    "dataset": "weather_risk_hourly_advanced_features",
    "device": device,
    "training_seconds": float(
        training_seconds
    ),
    "best_iteration": (
        int(best_iteration)
        if best_iteration is not None
        else None
    ),
    "features": FEATURES,
    "train": train_metrics,
    "validation": val_metrics,
    "test": test_metrics,
    "feature_importance": importance,
    "inference_benchmark": benchmark,
    "training_policy": {
        "synthetic_data": False,
        "synthetic_labels": False,
        "normal_sampling": (
            "deterministic sampling of real "
            "ERA5 grid cells"
        ),
        "class_weighting": "balanced",
        "chronological_split": True,
        "train_period": "2020-2023",
        "validation_period": "2024",
        "test_period": "2025",
    },
}

with open(
    METRICS_DIR / "test_metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        result,
        f,
        indent=2,
    )

metadata = {
    "task": "Weather Risk Classification",
    "model": "XGBoost",
    "dataset": (
        "Hourly ERA5 with spatial and "
        "temporal derived features"
    ),
    "classes": CLASS_NAMES,
    "features": FEATURES,
    "target_definition": {
        "NORMAL": (
            "No qualifying IMD system within "
            "300 km"
        ),
        "CAUTION": (
            "D/DD within 300 km"
        ),
        "DANGEROUS": (
            "CS/SCS/VSCS/ESCS/SUCS within 300 km"
        ),
    },
    "target_note": (
        "300 km is a project-defined operational "
        "labeling footprint."
    ),
    "leakage_controls": {
        "imd_grade_as_feature": False,
        "imd_distance_as_feature": False,
        "exact_imd_timestamp_only": True,
        "test_used_for_training": False,
    },
    "synthetic_data": False,
}

with open(
    METRICS_DIR / "advanced_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )

# ------------------------------------------------------------------
# Final report.
# ------------------------------------------------------------------
print("\n" + "=" * 90)
print("FINAL ADVANCED WEATHER RISK RESULTS")
print("=" * 90)

for metrics in [
    train_metrics,
    val_metrics,
    test_metrics,
]:
    print(
        f"\n{metrics['split'].upper()}"
    )

    print(
        f"  accuracy          : "
        f"{metrics['accuracy']:.5f}"
    )

    print(
        f"  balanced accuracy : "
        f"{metrics['balanced_accuracy']:.5f}"
    )

    print(
        f"  macro F1          : "
        f"{metrics['macro_f1']:.5f}"
    )

    print(
        f"  weighted F1       : "
        f"{metrics['weighted_f1']:.5f}"
    )

    print(
        f"  log loss          : "
        f"{metrics['log_loss']:.5f}"
    )

    for class_id in CLASS_IDS:
        cname = CLASS_NAMES[class_id]
        item = metrics["per_class"][cname]

        print(
            f"  {cname:10s} "
            f"P={item['precision']:.5f} "
            f"R={item['recall']:.5f} "
            f"F1={item['f1']:.5f}"
        )

print("\nTOP FEATURES")

for feature, value in list(
    importance.items()
)[:15]:
    print(
        f"  {feature:28s} "
        f"{value:.6f}"
    )

print("\nTEST CONFUSION MATRIX")
print(
    np.asarray(
        test_metrics["confusion_matrix"]
    )
)

print("\nINFERENCE")

for key, value in benchmark.items():
    print(
        f"  {key}: {value:.4f}"
    )

print("\nSaved:")
print(" ", model_path)
print(
    " ",
    METRICS_DIR / "test_metrics.json",
)
print(
    " ",
    METRICS_DIR / "advanced_metadata.json",
)
