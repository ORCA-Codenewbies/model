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

print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())


def load_data():
    X_train = np.load(DATA / "X_train.npy")
    X_val = np.load(DATA / "X_validation.npy")
    X_test = np.load(DATA / "X_test.npy")

    y_train = np.load(DATA / "y_train.npy")
    y_val = np.load(DATA / "y_validation.npy")
    y_test = np.load(DATA / "y_test.npy")

    metadata = json.loads(
        (DATA / "metadata.json").read_text()
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        metadata,
    )


def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.LSTM(
        64,
        return_sequences=True,
        dropout=0.15,
        recurrent_dropout=0.10,
    )(inputs)

    x = tf.keras.layers.LSTM(
        32,
        return_sequences=False,
        dropout=0.15,
        recurrent_dropout=0.10,
    )(x)

    x = tf.keras.layers.Dense(
        16,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(0.10)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="linear",
        name="cpue_forecast",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="fish_productivity_lstm",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    return model


def inverse_target(y_scaled, metadata):
    mean = float(metadata["target_scaler"]["mean"][0])
    scale = float(metadata["target_scaler"]["scale"][0])

    return y_scaled * scale + mean


def main():
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        metadata,
    ) = load_data()

    print("\nDataset:")
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_val:  ", y_val.shape)
    print("y_test: ", y_test.shape)

    model = build_model(X_train.shape[1:])

    model.summary()

    checkpoint_path = MODEL_DIR / "fish_productivity_lstm.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=35,
            min_delta=1e-5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=12,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nStarting training...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=32,
        shuffle=False,
        callbacks=callbacks,
        verbose=2,
    )

    print("\nLoading best checkpoint...")
    model = tf.keras.models.load_model(checkpoint_path)

    pred_scaled = model.predict(
        X_test,
        batch_size=32,
        verbose=0,
    ).reshape(-1)

    actual = inverse_target(y_test, metadata)
    predicted = inverse_target(pred_scaled, metadata)

    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    r2 = r2_score(actual, predicted)

    # Persistence baseline:
    # Since the target is next-month CPUE, the latest CPUE in
    # each sequence is the natural naive forecast.
    last_cpue_scaled = X_test[:, -1, 2]
    baseline = inverse_target(last_cpue_scaled, metadata)

    baseline_mae = mean_absolute_error(actual, baseline)
    baseline_rmse = mean_squared_error(actual, baseline) ** 0.5
    baseline_r2 = r2_score(actual, baseline)

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(f"LSTM MAE :  {mae:.10f}")
    print(f"LSTM RMSE:  {rmse:.10f}")
    print(f"LSTM R2  :  {r2:.6f} ({r2 * 100:.2f}%)")

    print("\nPersistence baseline:")
    print(f"MAE :  {baseline_mae:.10f}")
    print(f"RMSE:  {baseline_rmse:.10f}")
    print(f"R2  :  {baseline_r2:.6f} ({baseline_r2 * 100:.2f}%)")

    improvement = baseline_rmse - rmse

    if baseline_rmse != 0:
        improvement_pct = (
            improvement / baseline_rmse
        ) * 100.0
    else:
        improvement_pct = 0.0

    print(
        f"\nRMSE improvement over baseline: "
        f"{improvement_pct:.2f}%"
    )

    metrics = {
        "model": "fish_productivity_lstm",
        "task": "next-month CPUE forecasting",
        "test_samples": int(len(y_test)),
        "lookback_months": int(X_train.shape[1]),
        "feature_count": int(X_train.shape[2]),
        "epochs_trained": int(len(history.history["loss"])),
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        },
        "persistence_baseline": {
            "mae": float(baseline_mae),
            "rmse": float(baseline_rmse),
            "r2": float(baseline_r2),
        },
        "rmse_improvement_percent": float(improvement_pct),
        "train_end": metadata["train_end"],
        "validation_end": metadata["validation_end"],
        "target_definition": metadata["target_definition"],
    }

    (METRIC_DIR / "fish_productivity_lstm_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    with open(METRIC_DIR / "training_history.json", "w") as f:
        json.dump(
            {
                k: [float(v) for v in values]
                for k, values in history.history.items()
            },
            f,
            indent=2,
        )

    print("\nSaved:")
    print("  Model  :", checkpoint_path)
    print(
        "  Metrics:",
        METRIC_DIR / "fish_productivity_lstm_metrics.json",
    )


if __name__ == "__main__":
    main()
