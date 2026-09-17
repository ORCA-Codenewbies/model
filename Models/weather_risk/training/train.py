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
    "weather_risk_hourly_features.parquet"
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
]

CLASS_IDS = [0, 1, 2]

CLASS_NAMES = {
    0: "NORMAL",
    1: "CAUTION",
    2: "DANGEROUS",
}


def choose_device():
    requested = os.environ.get(
        "ORCA_XGB_DEVICE",
        "auto",
    ).lower()

    if requested in {"cpu", "cuda"}:
        return requested

    return "cpu"


def evaluate(
    split_name,
    y_true,
    probabilities,
):
    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    report = classification_report(
        y_true,
        predictions,
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
        name = CLASS_NAMES[class_id]
        row = report[name]

        per_class[name] = {
            "precision": float(
                row["precision"]
            ),
            "recall": float(
                row["recall"]
            ),
            "f1": float(
                row["f1-score"]
            ),
            "support": int(
                row["support"]
            ),
        }

    return {
        "split": split_name,
        "n_samples": int(len(y_true)),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                predictions,
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
            predictions,
            labels=CLASS_IDS,
        ).tolist(),
    }


print("=" * 90)
print("ORCA WEATHER RISK — HOURLY XGBOOST")
print("=" * 90)

print("\nLoading hourly dataset...")

df = pd.read_parquet(DATASET)

required = FEATURES + [
    "target",
    "target_name",
    "split",
]

missing = [
    col
    for col in required
    if col not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

print("Dataset shape:", df.shape)

train = df[
    df["split"] == "train"
].copy()

validation = df[
    df["split"] == "validation"
].copy()

test = df[
    df["split"] == "test"
].copy()

X_train = train[FEATURES]
y_train = train["target"].astype(
    np.int32
).to_numpy()

X_val = validation[FEATURES]
y_val = validation["target"].astype(
    np.int32
).to_numpy()

X_test = test[FEATURES]
y_test = test["target"].astype(
    np.int32
).to_numpy()

print("\nTrain:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nTRAIN CLASS DISTRIBUTION")
print(
    train["target_name"]
    .value_counts()
    .to_string()
)

print("\nVALIDATION CLASS DISTRIBUTION")
print(
    validation["target_name"]
    .value_counts()
    .to_string()
)

print("\nTEST CLASS DISTRIBUTION")
print(
    test["target_name"]
    .value_counts()
    .to_string()
)

# Real-data class weighting.
sample_weight = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

device = choose_device()

print("\nXGBoost device:", device)

model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,

    n_estimators=1400,
    learning_rate=0.05,

    max_depth=8,
    min_child_weight=6,

    subsample=0.85,
    colsample_bytree=0.90,

    reg_alpha=0.05,
    reg_lambda=2.0,
    gamma=0.05,

    tree_method="hist",
    device=device,

    eval_metric="mlogloss",
    early_stopping_rounds=80,

    random_state=42,
    n_jobs=-1,
)

print("\nStarting training...")

start = time.perf_counter()

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
    time.perf_counter() - start
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

print(
    "Best iteration:",
    best_iteration,
)

print("\nGenerating probabilities...")

train_prob = model.predict_proba(
    X_train
)

val_prob = model.predict_proba(
    X_val
)

test_prob = model.predict_proba(
    X_test
)

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

# ---------------------------------------------------------------------
# Inference benchmark.
# ---------------------------------------------------------------------
sample = X_test.iloc[[0]]

for _ in range(20):
    model.predict_proba(sample)

latencies = []

for _ in range(200):
    t0 = time.perf_counter()
    model.predict_proba(sample)
    t1 = time.perf_counter()

    latencies.append(
        (t1 - t0) * 1000.0
    )

latencies = np.asarray(
    latencies
)

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

# ---------------------------------------------------------------------
# Save model.
# ---------------------------------------------------------------------
model_path = (
    MODEL_DIR /
    "weather_risk_xgboost.json"
)

model.save_model(model_path)

# ---------------------------------------------------------------------
# Save metrics.
# ---------------------------------------------------------------------
metrics = {
    "model": "weather_risk_xgboost",
    "dataset": "weather_risk_hourly_features",
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
    "inference_benchmark": benchmark,
    "training_policy": {
        "synthetic_data": False,
        "synthetic_labels": False,
        "synthetic_oversampling": False,
        "normal_sampling": (
            "deterministic sampling of real "
            "ERA5 grid cells"
        ),
        "class_weighting": (
            "balanced sample weights"
        ),
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
        metrics,
        f,
        indent=2,
    )

metadata = {
    "task": "Weather Risk Classification",
    "model": "XGBoost",
    "dataset": (
        "Hourly ERA5 aligned to exact "
        "IMD cyclone observation timestamps"
    ),
    "classes": {
        "0": "NORMAL",
        "1": "CAUTION",
        "2": "DANGEROUS",
    },
    "features": FEATURES,
    "split": {
        "train": "2020-2023",
        "validation": "2024",
        "test": "2025",
    },
    "target_definition": {
        "NORMAL": (
            "No qualifying IMD system within "
            "the project-defined 300 km footprint"
        ),
        "CAUTION": (
            "IMD D/DD within 300 km"
        ),
        "DANGEROUS": (
            "IMD CS/SCS/VSCS/ESCS/SUCS "
            "within 300 km"
        ),
    },
    "target_note": (
        "The 300 km footprint is a project-defined "
        "operational labeling rule and is not an "
        "IMD-provided risk label."
    ),
    "leakage_controls": {
        "cyclone_grade_as_feature": False,
        "cyclone_distance_as_feature": False,
        "target_name_as_feature": False,
        "exact_imd_timestamp_only": True,
    },
    "synthetic_data": False,
    "device": device,
}

with open(
    METRICS_DIR / "metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )

# ---------------------------------------------------------------------
# Console report.
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("FINAL HOURLY WEATHER RISK RESULTS")
print("=" * 90)

for result in [
    train_metrics,
    val_metrics,
    test_metrics,
]:
    print(
        f"\n{result['split'].upper()}"
    )

    print(
        f"  accuracy          : "
        f"{result['accuracy']:.5f}"
    )

    print(
        f"  balanced accuracy : "
        f"{result['balanced_accuracy']:.5f}"
    )

    print(
        f"  macro F1          : "
        f"{result['macro_f1']:.5f}"
    )

    print(
        f"  weighted F1       : "
        f"{result['weighted_f1']:.5f}"
    )

    print(
        f"  log loss          : "
        f"{result['log_loss']:.5f}"
    )

    for class_id in CLASS_IDS:
        name = CLASS_NAMES[class_id]
        item = result["per_class"][name]

        print(
            f"  {name:10s} "
            f"P={item['precision']:.5f} "
            f"R={item['recall']:.5f} "
            f"F1={item['f1']:.5f}"
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
    METRICS_DIR / "metadata.json",
)
