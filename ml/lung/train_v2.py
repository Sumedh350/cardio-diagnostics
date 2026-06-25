#!/usr/bin/env python3
"""
ml/lung/train_v2.py — Lung binary classifier, full-resolution spectrograms.

Returns to full-size (104, 313) spectrograms.
Schedule follows CLAUDE.md retraining rules (rule 3: reduce batch size).
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
LOG_FILE  = LOG_DIR / "lung_v2_training.log"
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

TARGET = 0.82


def _build_cnn(shape: tuple, extra_conv: bool = False, l2: float = 0.0):
    from tensorflow import keras
    reg = keras.regularizers.l2(l2) if l2 else None

    inp = keras.Input(shape=shape)
    x   = inp
    for f in [32, 64, 128] + ([256] if extra_conv else []):
        x = keras.layers.Conv2D(f, 3, padding="same", activation="relu",
                                kernel_regularizer=reg)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(2)(x)
    x   = keras.layers.GlobalAveragePooling2D()(x)
    x   = keras.layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x   = keras.layers.Dropout(0.5)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="lung_v2_cnn")


def _build_mobilenet(shape_hw: tuple):
    import tensorflow as tf
    from tensorflow import keras

    h, w = shape_hw
    inp_gray = keras.Input(shape=(h, w, 1))
    inp_rgb  = keras.layers.Lambda(
        lambda x: tf.image.resize(tf.repeat(x, 3, axis=-1), (96, 96))
    )(inp_gray)

    base = keras.applications.MobileNetV2(
        input_shape=(96, 96, 3), include_top=False,
        weights="imagenet", pooling="avg",
    )
    x   = base(inp_rgb, training=False)
    x   = keras.layers.Dense(128, activation="relu")(x)
    x   = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp_gray, out, name="lung_mobilenet")
    return model, base


def _train_model(model, X_tr, y_tr, X_val, y_val,
                 lr: float, batch: int, epochs: int, patience: int,
                 class_weight: dict, ckpt_path: Path,
                 trainable_base=None) -> tuple[float, float, int]:
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(42)
    np.random.seed(42)

    # Phase 1 (head only if base provided)
    if trainable_base is not None:
        trainable_base.trainable = False
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="binary_crossentropy", metrics=["accuracy"])
        log.info("Phase 1 (frozen base): lr=%.4f  batch=%d", lr, batch)
        cbs1 = [keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience,
            restore_best_weights=True, verbose=1)]
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                  epochs=min(epochs, 10), batch_size=batch,
                  class_weight=class_weight, callbacks=cbs1, verbose=1)

        # Phase 2: fine-tune entire base
        trainable_base.trainable = True
        ft_lr = lr / 10
        log.info("Phase 2 (fine-tune): lr=%.5f", ft_lr)
    else:
        ft_lr = lr

    model.compile(
        optimizer=keras.optimizers.Adam(ft_lr),
        loss="binary_crossentropy", metrics=["accuracy"])
    log.info("Params: %s  lr=%.5f  batch=%d",
             f"{model.count_params():,}", ft_lr, batch)

    cbs = [
        keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_accuracy",
            save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]
    h = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=batch,
                  class_weight=class_weight, callbacks=cbs, verbose=1)

    best_i  = int(np.argmax(h.history["val_accuracy"]))
    val_acc = float(h.history["val_accuracy"][best_i])
    tr_acc  = float(h.history["accuracy"][best_i])
    ep      = len(h.history["val_accuracy"])
    return val_acc, tr_acc, ep


def _log_exp(attempt: int, total: int, arch: str, lr: float, batch: int,
             epochs_run: int, train_acc: float, val_acc: float,
             next_action: str) -> None:
    met = val_acc >= TARGET
    entry = (
        f"\n## {datetime.now().isoformat(timespec='seconds')}\n"
        f"- **Model**: lung-v2  (attempt {attempt}/{total})\n"
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


def _print_metrics(y_true, y_pred) -> None:
    from sklearn.metrics import classification_report, confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print("\n── Confusion matrix ─────────────────────────────────")
    print(f"              predicted_normal  predicted_abnormal")
    print(f"  actual_normal     {tn:>6}              {fp:>6}")
    print(f"  actual_abnormal   {fn:>6}              {tp:>6}")
    print("\n── Per-class metrics ────────────────────────────────")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=["normal", "abnormal"], digits=4))


def main() -> None:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    log.info("GPU: %s", [g.name for g in gpus] if gpus else "none (CPU)")

    if not (PROC_DIR / "X.npy").exists():
        log.error("X.npy not found")
        sys.exit(1)

    log.info("Loading full-size lung spectrograms …")
    X = np.load(PROC_DIR / "X.npy")[..., np.newaxis]
    y4 = np.load(PROC_DIR / "y.npy").astype(np.int32)
    y = (y4 > 0).astype(np.int32)
    INPUT_SHAPE = X.shape[1:]
    log.info("Shape: %s  binary dist: %s", INPUT_SHAPE, dict(Counter(y.tolist())))

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

    # Load existing best from models/lung_model_binary.keras (full-size, before retraining)
    best_val_global = 0.0
    if MODEL_OUT.exists():
        try:
            existing = tf.keras.models.load_model(MODEL_OUT)
            # Check if this model accepts the same input shape
            if existing.input_shape[1:] == INPUT_SHAPE:
                probs = existing.predict(X_val, batch_size=32, verbose=0).squeeze()
                preds = (probs >= 0.5).astype(int)
                best_val_global = float(np.mean(preds == y_val))
                log.info("Existing model val_acc on current split: %.4f", best_val_global)
            else:
                log.info("Existing model has different input shape %s — ignoring",
                         existing.input_shape[1:])
        except Exception as e:
            log.warning("Could not load existing model: %s", e)

    # CLAUDE.md rule 3: reduce batch size
    # Attempt 1: same winning arch, batch=16
    # Attempt 2: same arch, batch=8 (±50% from attempt 1)
    # Attempt 3: MobileNetV2 transfer learning
    # Attempt 4: CNN with L2 regularization
    schedule = [
        dict(kind="cnn",   lr=0.001,  batch=16, epochs=60, patience=8,
             extra_conv=False, l2=0.0,
             arch="Conv2D(32,64,128)+BN → GAP → Dense(256) → Dropout(0.5) batch=16"),
        dict(kind="cnn",   lr=0.001,  batch=8,  epochs=60, patience=8,
             extra_conv=False, l2=0.0,
             arch="Conv2D(32,64,128)+BN → GAP → Dense(256) → Dropout(0.5) batch=8"),
        dict(kind="mbnet", lr=0.001,  batch=32, epochs=40, patience=6,
             extra_conv=False, l2=0.0,
             arch="MobileNetV2(96x96 imagenet) → Dense(128) → Dropout(0.3) fine-tune"),
        dict(kind="cnn",   lr=0.001,  batch=16, epochs=60, patience=8,
             extra_conv=False, l2=1e-4,
             arch="Conv2D(32,64,128)+BN+L2(1e-4) → GAP → Dense(256) → Dropout(0.5) batch=16"),
    ]

    results: list[tuple[int, float]] = []
    total = len(schedule)

    for attempt, cfg in enumerate(schedule, start=1):
        log.info("=" * 60)
        log.info("ATTEMPT %d/%d  kind=%s  lr=%.4f  batch=%d",
                 attempt, total, cfg["kind"], cfg["lr"], cfg["batch"])

        ckpt = MODEL_DIR / f"_lung_v2_ckpt_{attempt}.keras"

        if cfg["kind"] == "mbnet":
            model, base = _build_mobilenet(INPUT_SHAPE[:2])
            val_acc, tr_acc, ep = _train_model(
                model, X_tr, y_tr, X_val, y_val,
                lr=cfg["lr"], batch=cfg["batch"],
                epochs=cfg["epochs"], patience=cfg["patience"],
                class_weight=cls_wt, ckpt_path=ckpt,
                trainable_base=base,
            )
        else:
            model = _build_cnn(INPUT_SHAPE, cfg["extra_conv"], cfg["l2"])
            val_acc, tr_acc, ep = _train_model(
                model, X_tr, y_tr, X_val, y_val,
                lr=cfg["lr"], batch=cfg["batch"],
                epochs=cfg["epochs"], patience=cfg["patience"],
                class_weight=cls_wt, ckpt_path=ckpt,
                trainable_base=None,
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
            next_action = f"attempt {attempt+1}: {schedule[attempt]['arch'][:80]}"

        _log_exp(attempt, total, cfg["arch"], cfg["lr"], cfg["batch"],
                 ep, tr_acc, val_acc, next_action)
        log.info("Attempt %d  val_acc=%.4f  target_met=%s", attempt, val_acc, met)

        if met:
            break

    log.info("=" * 60)
    log.info("Loading best model: %s", MODEL_OUT)
    from tensorflow import keras
    best_model = keras.models.load_model(MODEL_OUT)
    probs    = best_model.predict(X_te, batch_size=32, verbose=0).squeeze()
    preds    = (probs >= 0.5).astype(int)
    test_acc = float(np.mean(preds == y_te))

    print(f"\n{'='*60}")
    print(f"  LUNG V2 — TRAINING COMPLETE")
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
