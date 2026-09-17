from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(".")
DATA = ROOT / "Datasets/processed/training/fish_productivity_12m_env"
OUT = ROOT / "Models/fish_productivity"

MODEL_DIR = OUT / "model"
METRIC_DIR = OUT / "metrics"

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

    return X_train, X_val, X_test, y_train, y_val, y_test, metadata


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
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="fish_productivity_lstm_12m_env",
    )

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

    return model


def inverse_target(y, metadata):
    mean = float(metadata["target_scaler"]["mean"][0])
    scale = float(metadata["target_scaler"]["scale"][0])
    return y * scale + mean


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

    model = build_model(X_train.shape[1:])

    model.summary()

    checkpoint = MODEL_DIR / "fish_productivity_lstm_12m_env.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            min_delta=1e-5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
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

    actual = inverse_target(y_test, metadata)
    predicted = inverse_target(pred_scaled, metadata)

    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    r2 = r2_score(actual, predicted)

    # Persistence baseline:
    # feature 2 is log_cpue in the current feature ordering.
    last_cpue_scaled = X_test[:, -1, 2]
    baseline = inverse_target(
        last_cpue_scaled,
        metadata,
    )

    baseline_mae = mean_absolute_error(actual, baseline)
    baseline_rmse = mean_squared_error(
        actual,
        baseline,
    ) ** 0.5
    baseline_r2 = r2_score(actual, baseline)

    improvement = (
        (baseline_rmse - rmse)
        / baseline_rmse
        * 100
    )

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

    print(
        f"\nRMSE improvement over baseline: "
        f"{improvement:.2f}%"
    )

    metrics = {
        "model": "fish_productivity_lstm_12m_env",
        "task": "next-month CPUE forecasting",
        "lookback_months": 12,
        "features": int(X_train.shape[-1]),
        "test_samples": int(len(y_test)),
        "epochs_trained": int(len(history.history["loss"])),
        "test": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        },
        "persistence_baseline": {
            "mae": float(baseline_mae),
            "rmse": float(baseline_rmse),
            "r2": float(baseline_r2),
        },
        "rmse_improvement_percent": float(improvement),
        "train_end": metadata["train_end"],
        "validation_end": metadata["validation_end"],
    }

    (METRIC_DIR / "fish_productivity_lstm_12m_env_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    (METRIC_DIR / "fish_productivity_lstm_12m_env_history.json").write_text(
        json.dumps(
            {
                k: [float(v) for v in vals]
                for k, vals in history.history.items()
            },
            indent=2,
        )
    )

    print("\nSaved:")
    print("Model  :", checkpoint)
    print(
        "Metrics:",
        METRIC_DIR / "fish_productivity_lstm_12m_env_metrics.json",
    )


if __name__ == "__main__":
    main()
