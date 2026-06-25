#!/usr/bin/env python3
"""
train_quick.py — Fast lung binary classifier targeting ≥75% in ~30 minutes.

Speed trick: crop time axis from 313 → 104 frames (3.3 s of audio instead of 10 s).
This makes each training step ~3× faster while keeping full frequency resolution.
Same winning architecture: Conv2D(32,64,128)+BN → GAP → Dense(256) → Dropout(0.5).
"""
from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).resolve().parent.parent.parent
PROC_DIR  = ROOT / "ml" / "lung" / "processed"
LOG_DIR   = ROOT / "ml" / "logs"
MODEL_DIR = ROOT / "models"
MODEL_OUT = MODEL_DIR / "lung_model_binary.keras"
LOG_FILE  = LOG_DIR / "lung_quick_training.log"
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

TARGET    = 0.75
T_CROP    = 104   # crop time axis: 104 frames × 128 hop / 4000 Hz ≈ 3.3 s
EPOCHS    = 45
BATCH     = 32
PATIENCE  = 10


def _build(shape: tuple, dropout: float, l2_reg: float = 0.0):
    from tensorflow import keras
    from tensorflow.keras import regularizers

    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    inp = keras.Input(shape=shape)
    x   = inp
    for f in [32, 64, 128]:
        x = keras.layers.Conv2D(f, 3, padding="same", activation="relu",
                                kernel_regularizer=reg)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(2)(x)
    x   = keras.layers.GlobalAveragePooling2D()(x)
    x   = keras.layers.Dense(256, activation="relu")(x)
    x   = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="lung_quick_cnn")


def _train(X_tr, y_tr, X_val, y_val, shape, lr, dropout, l2_reg,
           ckpt_path, class_weight) -> tuple[float, float, int]:
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(42)
    np.random.seed(42)

    model = _build(shape, dropout, l2_reg)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    log.info("Params: %s  input=%s  lr=%.5f  drop=%.2f  l2=%.0e",
             f"{model.count_params():,}", shape, lr, dropout, l2_reg)

    cbs = [
        keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_accuracy",
            save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            min_lr=1e-6, verbose=1),
    ]

    h = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        class_weight=class_weight,
        callbacks=cbs,
        verbose=1,
    )

    ep      = len(h.history["val_accuracy"])
    best_i  = int(np.argmax(h.history["val_accuracy"]))
    val_acc = float(h.history["val_accuracy"][best_i])
    tr_acc  = float(h.history["accuracy"][best_i])
    return val_acc, tr_acc, ep


def _log_exp(attempt, arch, lr, dropout, l2_reg, t_crop,
             epochs_run, train_acc, val_acc, next_action):
    met   = val_acc >= TARGET
    entry = (
        f"\n## {datetime.now().isoformat(timespec='seconds')}\n"
        f"- **Model**: lung-quick  (attempt {attempt})\n"
        f"- **Architecture**: {arch}\n"
        f"- **Optimizer**: adam  lr={lr}  ReduceLROnPlateau\n"
        f"- **Dropout**: {dropout}  L2={l2_reg:.0e}  T_crop={t_crop}\n"
        f"- **Epochs run**: {epochs_run}\n"
        f"- **Train accuracy**: {train_acc:.4f}\n"
        f"- **Val accuracy**: {val_acc:.4f}\n"
        f"- **Target met**: {'✓ yes' if met else '✗ no'}  (target ≥ {TARGET})\n"
        f"- **Next action**: {next_action}\n"
    )
    with open(EXP_FILE, "a", encoding="utf-8") as fh:
        fh.write(entry)
    log.info("Logged attempt %d → experiments.md  val_acc=%.4f", attempt, val_acc)


def main():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    log.info("GPU: %s", [g.name for g in gpus] if gpus else "none (CPU)")

    if not (PROC_DIR / "X.npy").exists():
        log.error("X.npy not found — run ml/setup_datasets.py first")
        sys.exit(1)

    log.info("Loading lung data …")
    X_full = np.load(PROC_DIR / "X.npy")          # (N, 104, 313)
    y4     = np.load(PROC_DIR / "y.npy").astype(np.int32)
    y      = (y4 > 0).astype(np.int32)

    # Crop time axis to T_CROP for 3× faster training
    X = X_full[:, :, :T_CROP][..., np.newaxis]    # (N, 104, 104, 1)
    log.info("Input shape after crop: %s  (T: 313 → %d)", X.shape[1:], T_CROP)
    log.info("Dist: normal=%d  abnormal=%d", (y == 0).sum(), (y == 1).sum())

    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    log.info("Split — train %d  val %d  test %d", len(X_tr), len(X_val), len(X_te))

    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    cls_wt  = {int(c): float(w) for c, w in zip(classes, weights)}
    log.info("Class weights: normal=%.3f  abnormal=%.3f", cls_wt[0], cls_wt[1])

    INPUT_SHAPE = X.shape[1:]
    BASE_ARCH   = f"Conv2D(32,64,128)+BN → GAP → Dense(256) → Dropout → sigmoid  [T={T_CROP}]"

    schedule = [
        dict(lr=0.001, dropout=0.50, l2_reg=0.0),
        dict(lr=0.001, dropout=0.40, l2_reg=1e-4),
    ]

    best_val_global = 0.0

    for attempt, cfg in enumerate(schedule, start=1):
        log.info("=" * 60)
        log.info("QUICK ATTEMPT %d  lr=%.5f  drop=%.2f  l2=%.0e",
                 attempt, cfg["lr"], cfg["dropout"], cfg["l2_reg"])

        ckpt_path = MODEL_DIR / f"_lung_quick_ckpt_{attempt}.keras"

        val_acc, tr_acc, ep = _train(
            X_tr, y_tr, X_val, y_val,
            shape=INPUT_SHAPE,
            lr=cfg["lr"],
            dropout=cfg["dropout"],
            l2_reg=cfg["l2_reg"],
            ckpt_path=ckpt_path,
            class_weight=cls_wt,
        )

        if val_acc > best_val_global:
            best_val_global = val_acc
            shutil.copy2(str(ckpt_path), str(MODEL_OUT))
            log.info("New best %.4f → saved to %s", val_acc, MODEL_OUT)

        try:
            ckpt_path.unlink()
        except OSError:
            pass

        met         = val_acc >= TARGET
        next_action = ("target reached" if met
                       else f"attempt {attempt+1}" if attempt < len(schedule)
                       else "all attempts exhausted")

        _log_exp(attempt, BASE_ARCH, cfg["lr"], cfg["dropout"],
                 cfg["l2_reg"], T_CROP, ep, tr_acc, val_acc, next_action)

        if met:
            log.info("TARGET %.2f REACHED — stopping", TARGET)
            break

    # ── Final evaluation ──────────────────────────────────────────────────────
    log.info("=" * 60)
    from tensorflow import keras
    model    = keras.models.load_model(MODEL_OUT)
    probs    = model.predict(X_te, batch_size=BATCH, verbose=0).squeeze()
    preds    = (probs >= 0.5).astype(int)
    test_acc = float(np.mean(preds == y_te))

    print(f"\n{'='*60}")
    print(f"  LUNG QUICK — COMPLETE")
    print(f"  Input shape   : {INPUT_SHAPE}  (T_crop={T_CROP})")
    print(f"  Best val acc  : {best_val_global:.4f}")
    print(f"  Test accuracy : {test_acc:.4f}")
    print(f"  Target ≥{TARGET} : {'✓ MET' if best_val_global >= TARGET else '✗ NOT MET'}")
    print(f"  Model saved   : {MODEL_OUT}")

    from sklearn.metrics import classification_report, confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_te, preds, labels=[0, 1]).ravel()
    print(f"\n  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(classification_report(y_te, preds,
                                target_names=["normal", "abnormal"], digits=4))


if __name__ == "__main__":
    main()
