#!/usr/bin/env python3
"""
ml/lung/train_fast.py — Lung binary classifier with 64x64 resized spectrograms.

Lighter architecture, smaller input — faster training, less overfitting risk.
Target: val_accuracy >= 0.82
"""
from __future__ import annotations

import logging
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).resolve().parent.parent.parent
PROC_DIR  = ROOT / "ml" / "lung" / "processed"
LOG_DIR   = ROOT / "ml" / "logs"
MODEL_DIR = ROOT / "models"
MODEL_OUT = MODEL_DIR / "lung_model_binary.keras"
LOG_FILE  = LOG_DIR / "lung_fast_training.log"
EXP_FILE  = LOG_DIR / "experiments.md"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

TARGET    = 0.82
INPUT_SIZE = (64, 64)


def _resize(X: np.ndarray) -> np.ndarray:
    from skimage.transform import resize
    log.info("Resizing %d spectrograms from %s to %s …", len(X), X.shape[1:3], INPUT_SIZE)
    out = np.empty((len(X), INPUT_SIZE[0], INPUT_SIZE[1], 1), dtype=np.float32)
    for i, img in enumerate(X):
        out[i, :, :, 0] = resize(img[:, :, 0], INPUT_SIZE, anti_aliasing=True)
    log.info("Resize complete — shape %s", out.shape)
    return out


def _build(lr: float, extra_conv: bool = False):
    from tensorflow import keras

    inp = keras.Input(shape=(*INPUT_SIZE, 1))
    x   = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x   = keras.layers.MaxPooling2D(2)(x)
    x   = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x   = keras.layers.MaxPooling2D(2)(x)
    if extra_conv:
        x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPooling2D(2)(x)
    x   = keras.layers.GlobalAveragePooling2D()(x)
    x   = keras.layers.Dense(64, activation="relu")(x)
    x   = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out, name="lung_fast_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _train(X_tr, y_tr, X_val, y_val, lr: float, batch: int,
           epochs: int, patience: int, extra_conv: bool,
           class_weight: dict, ckpt_path: Path) -> tuple[float, float, int]:
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(42)
    np.random.seed(42)

    model = _build(lr, extra_conv)
    log.info("Params: %s  lr=%.4f  batch=%d  extra_conv=%s",
             f"{model.count_params():,}", lr, batch, extra_conv)

    cbs = [
        keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_accuracy",
            save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    h = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch,
        class_weight=class_weight,
        callbacks=cbs,
        verbose=1,
    )

    best_i  = int(np.argmax(h.history["val_accuracy"]))
    val_acc = float(h.history["val_accuracy"][best_i])
    tr_acc  = float(h.history["accuracy"][best_i])
    ep      = len(h.history["val_accuracy"])
    return val_acc, tr_acc, ep


def _log_exp(attempt: int, total: int, arch: str, lr: float,
             batch: int, epochs_run: int,
             train_acc: float, val_acc: float, next_action: str) -> None:
    met   = val_acc >= TARGET
    entry = (
        f"\n## {datetime.now().isoformat(timespec='seconds')}\n"
        f"- **Model**: lung-fast  (attempt {attempt}/{total})\n"
        f"- **Architecture**: {arch}\n"
        f"- **Optimizer**: adam  lr={lr}\n"
        f"- **Batch**: {batch}\n"
        f"- **Augmentation**: none\n"
        f"- **Epochs run**: {epochs_run}\n"
        f"- **Train accuracy**: {train_acc:.4f}\n"
        f"- **Val accuracy**: {val_acc:.4f}\n"
        f"- **Target met**: {'✓ yes' if met else '✗ no'}  (target ≥ {TARGET})\n"
        f"- **Next action**: {next_action}\n"
    )
    with open(EXP_FILE, "a", encoding="utf-8") as fh:
        fh.write(entry)
    log.info("Logged attempt %d → experiments.md", attempt)


def _print_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    from sklearn.metrics import classification_report, confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print("\n── Confusion matrix ─────────────────────────────────")
    print(f"              predicted_normal  predicted_abnormal")
    print(f"  actual_normal     {tn:>6}              {fp:>6}")
    print(f"  actual_abnormal   {fn:>6}              {tp:>6}")
    print("\n── Per-class metrics ────────────────────────────────")
    print(classification_report(y_true, y_pred,
                                labels=[0, 1],
                                target_names=["normal", "abnormal"], digits=4))


def main() -> None:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        log.info("GPU: %s", [g.name for g in gpus])
    else:
        log.warning("No GPU — training on CPU")

    if not (PROC_DIR / "X.npy").exists():
        log.error("X.npy not found")
        sys.exit(1)

    log.info("Loading lung data …")
    X_raw = np.load(PROC_DIR / "X.npy")[..., np.newaxis]
    y4    = np.load(PROC_DIR / "y.npy").astype(np.int32)

    log.info("Original 4-class dist: %s", dict(Counter(y4.tolist())))
    y = (y4 > 0).astype(np.int32)
    bin_dist = Counter(y.tolist())
    log.info("Binary dist: normal=%d  abnormal=%d", bin_dist[0], bin_dist[1])
    print(f"\nClass distribution — normal: {bin_dist[0]} | abnormal: {bin_dist[1]}")

    X = _resize(X_raw)

    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    log.info("Split — train %d  val %d  test %d", len(X_tr), len(X_val), len(X_te))

    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    cls_wt  = {0: float(weights[0]), 1: float(weights[1])}
    log.info("Class weights: normal=%.3f  abnormal=%.3f", cls_wt[0], cls_wt[1])

    # Schedule follows CLAUDE.md retraining rules:
    # 1. Adjust lr (±50%)  2. Increase depth  3. Reduce batch
    schedule = [
        dict(lr=0.001,  batch=64, epochs=20, patience=3, extra_conv=False,
             arch="Conv2D(16,32,64x64) → MaxPool → GAP → Dense(64) → Dropout(0.3) → sigmoid"),
        dict(lr=0.0015, batch=64, epochs=25, patience=4, extra_conv=False,
             arch="Conv2D(16,32,64x64) lr+50% → MaxPool → GAP → Dense(64) → Dropout(0.3) → sigmoid"),
        dict(lr=0.001,  batch=64, epochs=30, patience=5, extra_conv=True,
             arch="Conv2D(16,32,64,64x64) → MaxPool → GAP → Dense(64) → Dropout(0.3) → sigmoid"),
        dict(lr=0.001,  batch=32, epochs=30, patience=5, extra_conv=True,
             arch="Conv2D(16,32,64,64x64) batch=32 → MaxPool → GAP → Dense(64) → Dropout(0.3) → sigmoid"),
    ]

    results: list[tuple[int, float]] = []
    best_val_global = 0.0
    total = len(schedule)

    # Load current best from existing model to avoid overwriting a better one
    existing_best = 0.0
    if MODEL_OUT.exists():
        try:
            from tensorflow import keras
            existing_model = keras.models.load_model(MODEL_OUT)
            probs_ex = existing_model.predict(X_val, batch_size=64, verbose=0).squeeze()
            preds_ex = (probs_ex >= 0.5).astype(int)
            existing_best = float(np.mean(preds_ex == y_val))
            log.info("Existing model val accuracy on current split: %.4f", existing_best)
            best_val_global = existing_best
        except Exception as e:
            log.warning("Could not evaluate existing model: %s", e)

    for attempt, cfg in enumerate(schedule, start=1):
        log.info("=" * 60)
        log.info("ATTEMPT %d/%d  lr=%.4f  batch=%d  extra_conv=%s",
                 attempt, total, cfg["lr"], cfg["batch"], cfg["extra_conv"])

        ckpt = MODEL_DIR / f"_lung_fast_ckpt_{attempt}.keras"
        val_acc, tr_acc, ep = _train(
            X_tr, y_tr, X_val, y_val,
            lr=cfg["lr"], batch=cfg["batch"],
            epochs=cfg["epochs"], patience=cfg["patience"],
            extra_conv=cfg["extra_conv"],
            class_weight=cls_wt,
            ckpt_path=ckpt,
        )
        results.append((attempt, val_acc))

        if val_acc > best_val_global:
            best_val_global = val_acc
            shutil.copy2(str(ckpt), str(MODEL_OUT))
            log.info("New best (%.4f) → %s", val_acc, MODEL_OUT)
        try:
            ckpt.unlink()
        except OSError:
            pass

        met = val_acc >= TARGET
        if met:
            next_action = "target reached — done"
        elif attempt == total:
            next_action = "all attempts exhausted"
        else:
            next_cfg    = schedule[attempt]
            next_action = f"attempt {attempt+1}: {next_cfg['arch'][:80]}"

        _log_exp(attempt, total, cfg["arch"], cfg["lr"], cfg["batch"],
                 ep, tr_acc, val_acc, next_action)
        log.info("Attempt %d  val_acc=%.4f  target_met=%s", attempt, val_acc, met)

        if met:
            break

    log.info("=" * 60)
    log.info("Loading best model from %s", MODEL_OUT)
    from tensorflow import keras
    model    = keras.models.load_model(MODEL_OUT)
    probs    = model.predict(X_te, batch_size=64, verbose=0).squeeze()
    preds    = (probs >= 0.5).astype(int)
    test_acc = float(np.mean(preds == y_te))

    print(f"\n{'='*60}")
    print(f"  LUNG FAST MODEL — TRAINING COMPLETE")
    print(f"{'='*60}")
    for att, vac in results:
        print(f"    attempt {att}  val_acc={vac:.4f}  {'✓' if vac >= TARGET else '✗'}")
    print(f"  Best val accuracy : {best_val_global:.4f}")
    print(f"  Test accuracy     : {test_acc:.4f}")
    print(f"  Target (≥ {TARGET})  : {'✓ MET' if best_val_global >= TARGET else '✗ NOT MET'}")
    print(f"  Model saved to    : {MODEL_OUT}")
    _print_metrics(y_te, preds)


if __name__ == "__main__":
    main()
