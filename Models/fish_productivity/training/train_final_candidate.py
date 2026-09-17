from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(".")
DATA = ROOT / "Datasets/processed/training/fish_productivity"

MODEL_DIR = ROOT / "Models/fish_productivity/model"
METRIC_DIR = ROOT / "Models/fish_productivity/metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

X_train = np.load(DATA / "X_train.npy")
X_val = np.load(DATA / "X_validation.npy")
X_test = np.load(DATA / "X_test.npy")

y_train = np.load(DATA / "y_train.npy")
y_val = np.load(DATA / "y_validation.npy")
y_test = np.load(DATA / "y_test.npy")

metadata = json.loads(
    (DATA / "metadata.json").read_text()
)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X_train.shape[1:]),

    tf.keras.layers.LSTM(
        32,
        dropout=0.15,
        recurrent_dropout=0.0,
    ),

    tf.keras.layers.Dense(
        16,
        activation="relu",
    ),

    tf.keras.layers.Dropout(0.15),

    tf.keras.layers.Dense(1),
], name="fish_productivity_lstm")


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        clipnorm=1.0,
    ),
    loss=tf.keras.losses.Huber(),
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse"),
    ],
)


checkpoint = MODEL_DIR / "fish_productivity_lstm_candidate.keras"


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=25,
        min_delta=1e-5,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1,
    ),

    tf.keras.callbacks.ModelCheckpoint(
        checkpoint,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]


print("\nTraining final candidate...")
print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)

model.summary()


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=250,
    batch_size=32,
    shuffle=False,
    callbacks=callbacks,
    verbose=2,
)


model = tf.keras.models.load_model(checkpoint)


pred_scaled = model.predict(
    X_test,
    batch_size=32,
    verbose=0,
).reshape(-1)


target_mean = float(
    metadata["target_scaler"]["mean"][0]
)

target_scale = float(
    metadata["target_scaler"]["scale"][0]
)


actual = (
    y_test * target_scale
    + target_mean
)

predicted = (
    pred_scaled * target_scale
    + target_mean
)


mae = mean_absolute_error(
    actual,
    predicted,
)

rmse = mean_squared_error(
    actual,
    predicted,
) ** 0.5

r2 = r2_score(
    actual,
    predicted,
)


# Persistence baseline.
# Feature index 2 = scaled CPUE.
last_cpue_scaled = X_test[:, -1, 2]

baseline = (
    last_cpue_scaled * target_scale
    + target_mean
)

baseline_mae = mean_absolute_error(
    actual,
    baseline,
)

baseline_rmse = mean_squared_error(
    actual,
    baseline,
) ** 0.5

baseline_r2 = r2_score(
    actual,
    baseline,
)

rmse_improvement = (
    (baseline_rmse - rmse)
    / baseline_rmse
    * 100.0
)


print("\n" + "=" * 70)
print("FINAL CANDIDATE — TEST RESULTS")
print("=" * 70)

print(f"MAE :  {mae:.10f}")
print(f"RMSE:  {rmse:.10f}")
print(f"R2  :  {r2:.6f} ({r2 * 100:.2f}%)")

print("\nPersistence baseline:")
print(f"MAE :  {baseline_mae:.10f}")
print(f"RMSE:  {baseline_rmse:.10f}")
print(f"R2  :  {baseline_r2:.6f} ({baseline_r2 * 100:.2f}%)")

print(
    f"\nRMSE improvement over baseline: "
    f"{rmse_improvement:.2f}%"
)


metrics = {
    "model": "fish_productivity_lstm_candidate",
    "architecture": {
        "lstm_units": 32,
        "dense_units": 16,
        "dropout": 0.15,
        "learning_rate": 0.001,
    },
    "lookback_months": 12,
    "num_features": int(X_train.shape[-1]),
    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "epochs_trained": int(len(history.history["loss"])),
    "test": {
        "mae_mt_per_hook": float(mae),
        "rmse_mt_per_hook": float(rmse),
        "r2": float(r2),
    },
    "persistence_baseline": {
        "mae_mt_per_hook": float(baseline_mae),
        "rmse_mt_per_hook": float(baseline_rmse),
        "r2": float(baseline_r2),
    },
    "rmse_improvement_percent": float(
        rmse_improvement
    ),
}


(METRIC_DIR / "fish_productivity_lstm_candidate_metrics.json").write_text(
    json.dumps(metrics, indent=2)
)

with open(
    METRIC_DIR / "fish_productivity_lstm_candidate_history.json",
    "w",
) as f:
    json.dump(
        {
            k: [float(v) for v in values]
            for k, values in history.history.items()
        },
        f,
        indent=2,
    )


print("\nCandidate saved:")
print(checkpoint)
