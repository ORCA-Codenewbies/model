from pathlib import Path
import json
import numpy as np
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Development tensor only:
# 7 days × channels × spatial grid
TIMESTEPS = 7
HEIGHT = 32
WIDTH = 32
CHANNELS = 4

SAMPLES = 24
BATCH_SIZE = 4
EPOCHS = 5

MODEL_DIR = Path("Models/pfz/model")
METRICS_DIR = Path("Models/pfz/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PFZ CNN-LSTM SMOKE TEST")
print("=" * 80)

gpus = tf.config.list_physical_devices("GPU")
print("GPU:", gpus if gpus else "CPU fallback")

# Architecture validation tensors only.
# These are NOT real scientific training data.
X = np.random.randn(
    SAMPLES,
    TIMESTEPS,
    HEIGHT,
    WIDTH,
    CHANNELS
).astype(np.float32)

y = (
    np.random.rand(
        SAMPLES,
        HEIGHT,
        WIDTH,
        1
    ) > 0.88
).astype(np.float32)

# Keep output sigmoid probabilities.
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(TIMESTEPS, HEIGHT, WIDTH, CHANNELS)
    ),

    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(
            16, 3, padding="same", activation="relu"
        )
    ),
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.BatchNormalization()
    ),
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.MaxPooling2D(2)
    ),

    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(
            32, 3, padding="same", activation="relu"
        )
    ),
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.BatchNormalization()
    ),

    # Spatial → feature vector at each day
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.GlobalAveragePooling2D()
    ),

    # Temporal encoder
    tf.keras.layers.LSTM(
        64,
        return_sequences=False,
        dropout=0.2
    ),

    # Reconstruct spatial probability surface
    tf.keras.layers.Dense(
        HEIGHT * WIDTH,
        activation="sigmoid"
    ),
    tf.keras.layers.Reshape(
        (HEIGHT, WIDTH, 1)
    ),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=2,
        restore_best_weights=True,
    ),
]

history = model.fit(
    X,
    y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    callbacks=callbacks,
)

loss, accuracy, precision, recall = model.evaluate(
    X, y, verbose=0
)

model_path = MODEL_DIR / "pfz_cnn_lstm_smoketest.keras"
model.save(model_path)

metrics = {
    "type": "architecture_smoketest",
    "scientific_training_valid": False,
    "timesteps": TIMESTEPS,
    "height": HEIGHT,
    "width": WIDTH,
    "channels": CHANNELS,
    "samples": SAMPLES,
    "epochs": len(history.history["loss"]),
    "loss": float(loss),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "gpu_available": bool(gpus),
}

with open(
    METRICS_DIR / "pfz_cnn_lstm_smoketest_metrics.json",
    "w"
) as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 80)
print("SMOKE TEST COMPLETE")
print("=" * 80)
print("Model:", model_path)
print("Metrics:", METRICS_DIR / "pfz_cnn_lstm_smoketest_metrics.json")
print("NOTE: This is architecture validation, NOT production PFZ training.")
