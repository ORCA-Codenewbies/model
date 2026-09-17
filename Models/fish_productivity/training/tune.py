from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf

ROOT = Path(".")
DATA = ROOT / "Datasets/processed/training/fish_productivity"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

X_train = np.load(DATA / "X_train.npy")
X_val = np.load(DATA / "X_validation.npy")
y_train = np.load(DATA / "y_train.npy")
y_val = np.load(DATA / "y_validation.npy")


def build_model(units, dense_units, dropout, lr):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=X_train.shape[1:]),
        tf.keras.layers.LSTM(
            units,
            dropout=dropout,
            recurrent_dropout=0.0,
        ),
        tf.keras.layers.Dense(
            dense_units,
            activation="relu",
        ),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr,
            clipnorm=1.0,
        ),
        loss=tf.keras.losses.Huber(),
    )

    return model


configs = [
    {"name": "baseline64", "units": 64, "dense": 32, "dropout": 0.20, "lr": 1e-3},
    {"name": "lstm32", "units": 32, "dense": 16, "dropout": 0.15, "lr": 1e-3},
    {"name": "lstm64_lowdrop", "units": 64, "dense": 32, "dropout": 0.10, "lr": 1e-3},
    {"name": "lstm96", "units": 96, "dense": 32, "dropout": 0.20, "lr": 5e-4},
    {"name": "lstm128", "units": 128, "dense": 32, "dropout": 0.20, "lr": 5e-4},
]


results = []

for cfg in configs:
    print("\n" + "=" * 70)
    print("CONFIG:", cfg["name"])
    print("=" * 70)

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    model = build_model(
        cfg["units"],
        cfg["dense"],
        cfg["dropout"],
        cfg["lr"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
    )

    best_val = float(np.min(history.history["val_loss"]))
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

    print("best_val_loss:", best_val)
    print("best_epoch:", best_epoch)

    results.append({
        **cfg,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
    })


results.sort(key=lambda x: x["best_val_loss"])

print("\n" + "=" * 70)
print("VALIDATION RANKING")
print("=" * 70)

for i, r in enumerate(results, 1):
    print(
        f"{i}. {r['name']:20s} "
        f"val_loss={r['best_val_loss']:.6f} "
        f"epoch={r['best_epoch']}"
    )

out = Path("Models/fish_productivity/metrics")
out.mkdir(parents=True, exist_ok=True)

(out / "lstm_tuning_results.json").write_text(
    json.dumps(results, indent=2)
)
