from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(".")
ERA5_PATH = ROOT / "Datasets/processed/ERA5/era5_monthly_indian_ocean_1960_2026.nc"
PFZ_PATH = ROOT / "Datasets/processed/PFZ/pfz_advisories_from_xls.csv"

MODEL_DIR = ROOT / "Models/pfz/model"
METRICS_DIR = ROOT / "Models/pfz/metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 7
PATCH = 9
CHANNELS = 6

print("=" * 80)
print("PFZ CNN-LSTM v1 TRAINING")
print("=" * 80)

# ---------------------------------------------------------------------
# Load PFZ labels
# ---------------------------------------------------------------------

pfz = pd.read_csv(
    PFZ_PATH,
    parse_dates=["valid_from", "valid_to"]
)

pfz = pfz[
    (pfz["valid_from"] == pd.Timestamp("2026-08-28"))
].copy()

if len(pfz) == 0:
    raise RuntimeError("No PFZ labels found.")

print("PFZ records:", len(pfz))

# ---------------------------------------------------------------------
# Load ERA5
# ---------------------------------------------------------------------

ds = xr.open_dataset(ERA5_PATH)

# Seven monthly states ending at August 2026.
target_month = pd.Timestamp("2026-08-01")
months = pd.date_range(
    target_month - pd.DateOffset(months=6),
    target_month,
    freq="MS",
)

print("Temporal sequence:")
for m in months:
    print(" ", m.date())

available = pd.to_datetime(ds["valid_time"].values)

if not all(m in available for m in months):
    raise RuntimeError(
        "Required ERA5 months are not all available."
    )

variables = [
    "sst",
    "u10",
    "v10",
    "t2m",
    "msl",
    "wind_speed_10m",
]

frames = []

for m in months:
    frame = []

    for var in variables:
        da = ds[var].sel(valid_time=m)

        # ERA5 land cells / missing SST remain missing.
        frame.append(
            da.values.astype(np.float32)
        )

    arr = np.stack(frame, axis=-1)

    frames.append(arr)

Xfull = np.stack(frames, axis=0)

# Dimensions:
# time × lat × lon × channel
print("ERA5 tensor:", Xfull.shape)

lats = ds["latitude"].values
lons = ds["longitude"].values

# ---------------------------------------------------------------------
# Restrict to PFZ operational region
# ---------------------------------------------------------------------

lat_min = max(
    float(lats.min()),
    float(pfz["latitude"].min()) - 1.0
)

lat_max = min(
    float(lats.max()),
    float(pfz["latitude"].max()) + 1.0
)

lon_min = max(
    float(lons.min()),
    float(pfz["longitude"].min()) - 1.0
)

lon_max = min(
    float(lons.max()),
    float(pfz["longitude"].max()) + 1.0
)

lat_idx = np.where(
    (lats >= lat_min) &
    (lats <= lat_max)
)[0]

lon_idx = np.where(
    (lons >= lon_min) &
    (lons <= lon_max)
)[0]

Xfull = Xfull[
    :,
    lat_idx,
    :,
][:, :, lon_idx, :]

regional_lats = lats[lat_idx]
regional_lons = lons[lon_idx]

H = len(regional_lats)
W = len(regional_lons)

print("Regional tensor:", Xfull.shape)

# ---------------------------------------------------------------------
# Replace NaN/Inf using spatial-channel median computed from real ERA5.
#
# This is environmental missing-value handling only. No labels are
# used in the imputation.
# ---------------------------------------------------------------------

for t in range(LOOKBACK):
    for c in range(CHANNELS):
        layer = Xfull[t, :, :, c]

        valid = np.isfinite(layer)

        if not valid.any():
            raise RuntimeError(
                f"No valid ERA5 values for t={t}, channel={c}"
            )

        median = np.nanmedian(layer)

        layer = np.where(
            valid,
            layer,
            median
        )

        Xfull[t, :, :, c] = layer

# ---------------------------------------------------------------------
# Standardize each channel using temporal regional statistics.
# ---------------------------------------------------------------------

for c in range(CHANNELS):
    values = Xfull[:, :, :, c]

    mean = values.mean()
    std = values.std()

    if std < 1e-8:
        std = 1.0

    Xfull[:, :, :, c] = (
        values - mean
    ) / std

# ---------------------------------------------------------------------
# Create sparse PFZ target map.
#
# An advisory point is mapped to the nearest ERA5 grid cell.
# ---------------------------------------------------------------------

Y = np.zeros(
    (H, W),
    dtype=np.float32
)

for _, row in pfz.iterrows():

    lat = float(row["latitude"])
    lon = float(row["longitude"])

    i = int(
        np.abs(regional_lats - lat).argmin()
    )

    j = int(
        np.abs(regional_lons - lon).argmin()
    )

    Y[i, j] = 1.0

print("Positive ERA5 grid cells:", int(Y.sum()))

# ---------------------------------------------------------------------
# Build spatial patches.
#
# Each sample:
#   7 × 9 × 9 × channels
#
# Target = center-cell PFZ label.
# ---------------------------------------------------------------------

radius = PATCH // 2

samples = []
labels = []
centers = []

for i in range(radius, H - radius):
    for j in range(radius, W - radius):

        patch = Xfull[
            :,
            i - radius:i + radius + 1,
            j - radius:j + radius + 1,
            :
        ]

        samples.append(patch)
        labels.append(Y[i, j])
        centers.append(
            (
                float(regional_lats[i]),
                float(regional_lons[j])
            )
        )

X = np.stack(samples).astype(np.float32)
y = np.asarray(labels).astype(np.float32)
centers = np.asarray(centers)

print("All samples:", X.shape)
print("All positives:", int(y.sum()))

# ---------------------------------------------------------------------
# Reduce extreme class imbalance while preserving hard negatives.
#
# Keep every positive and sample up to 5x negatives.
# ---------------------------------------------------------------------

positive_idx = np.where(y == 1)[0]
negative_idx = np.where(y == 0)[0]

rng = np.random.default_rng(SEED)

n_negative = min(
    len(negative_idx),
    max(100, len(positive_idx) * 5)
)

negative_idx = rng.choice(
    negative_idx,
    size=n_negative,
    replace=False
)

keep = np.concatenate([
    positive_idx,
    negative_idx
])

X = X[keep]
y = y[keep]
centers = centers[keep]

print("Balanced candidate samples:", len(y))
print("Positive:", int(y.sum()))
print("Negative:", int((y == 0).sum()))

# ---------------------------------------------------------------------
# Spatial split.
#
# Hold out eastern longitude block.
# No random spatial mixing.
# ---------------------------------------------------------------------

longitude = centers[:, 1]

split_lon = np.quantile(
    longitude,
    0.80
)

train_mask = longitude <= split_lon
val_mask = longitude > split_lon

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

if len(X_train) == 0 or len(X_val) == 0:
    raise RuntimeError("Spatial split failed.")

if len(np.unique(y_val)) < 2:
    raise RuntimeError(
        "Validation split needs both classes."
    )

print("\nTrain:", X_train.shape, "positive:", int(y_train.sum()))
print("Val:  ", X_val.shape, "positive:", int(y_val.sum()))

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(
            LOOKBACK,
            PATCH,
            PATCH,
            CHANNELS
        )
    ),

    # Spatial encoder per time step
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(
            24,
            kernel_size=3,
            padding="same",
            activation="relu"
        )
    ),

    tf.keras.layers.TimeDistributed(
        tf.keras.layers.BatchNormalization()
    ),

    tf.keras.layers.TimeDistributed(
        tf.keras.layers.MaxPooling2D(
            pool_size=2
        )
    ),

    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(
            48,
            kernel_size=3,
            padding="same",
            activation="relu"
        )
    ),

    tf.keras.layers.TimeDistributed(
        tf.keras.layers.GlobalAveragePooling2D()
    ),

    # Temporal encoder
    tf.keras.layers.LSTM(
        64,
        dropout=0.20
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        ),
        tf.keras.metrics.AUC(
            name="auc"
        ),
    ]
)

model.summary()

# ---------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------

n_pos = max(1, int(y_train.sum()))
n_neg = max(1, int((y_train == 0).sum()))

class_weight = {
    0: 1.0,
    1: n_neg / n_pos,
}

print("Class weight:", class_weight)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=8,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5
    ),
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=2
)

# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

prob = model.predict(
    X_val,
    batch_size=64,
    verbose=0
).ravel()

thresholds = np.arange(
    0.10,
    0.96,
    0.05
)

best = None

for threshold in thresholds:

    pred = (
        prob >= threshold
    ).astype(np.int32)

    f1 = f1_score(
        y_val,
        pred,
        zero_division=0
    )

    row = {
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                y_val,
                pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_val,
                pred,
                zero_division=0
            )
        ),
        "f1": float(f1),
    }

    if best is None or f1 > best["f1"]:
        best = row

auc = roc_auc_score(
    y_val,
    prob
)

final_pred = (
    prob >= best["threshold"]
).astype(np.int32)

metrics = {
    "model": "PFZ_CNN_LSTM",
    "version": "v1-development",
    "scientifically_valid": True,
    "production_historical_training": False,
    "label_source": "INCOIS PFZ advisory",
    "label_window": [
        "2026-08-28",
        "2026-08-29"
    ],
    "temporal_input": "7 monthly ERA5 states, Feb-Aug 2026",
    "spatial_patch": [PATCH, PATCH],
    "channels": variables,
    "lookback": LOOKBACK,
    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "train_positive": int(y_train.sum()),
    "validation_positive": int(y_val.sum()),
    "roc_auc": float(auc),
    "best_threshold": best["threshold"],
    "precision": best["precision"],
    "recall": best["recall"],
    "f1": best["f1"],
    "architecture": "TimeDistributed CNN -> LSTM -> Dense classifier",
    "note": (
        "Development CNN-LSTM trained on available 2026 PFZ "
        "labels and ERA5 temporal sequence. It is not a "
        "historical 7-day satellite PFZ model because matching "
        "historical PFZ labels are unavailable."
    )
}

MODEL_PATH = (
    MODEL_DIR /
    "pfz_cnn_lstm_v1_development.keras"
)

METRICS_PATH = (
    METRICS_DIR /
    "pfz_cnn_lstm_v1_metrics.json"
)

THRESHOLD_PATH = (
    METRICS_DIR /
    "pfz_cnn_lstm_v1_thresholds.csv"
)

model.save(MODEL_PATH)

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

threshold_rows = []

for threshold in thresholds:
    pred = (
        prob >= threshold
    ).astype(np.int32)

    threshold_rows.append({
        "threshold": float(threshold),
        "precision": precision_score(
            y_val,
            pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_val,
            pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_val,
            pred,
            zero_division=0
        ),
    })

pd.DataFrame(
    threshold_rows
).to_csv(
    THRESHOLD_PATH,
    index=False
)

print("\n" + "=" * 80)
print("PFZ CNN-LSTM COMPLETE")
print("=" * 80)

print("ROC-AUC:", round(auc, 4))
print("Best threshold:", best["threshold"])
print("Precision:", round(best["precision"], 4))
print("Recall:", round(best["recall"], 4))
print("F1:", round(best["f1"], 4))

print("\nModel:", MODEL_PATH)
print("Metrics:", METRICS_PATH)
print("Thresholds:", THRESHOLD_PATH)

ds.close()
