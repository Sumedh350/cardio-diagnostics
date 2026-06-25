#!/usr/bin/env python3
"""
ml/lung/train_transfer.py — Lung binary classifier via MobileNetV2 transfer learning.

Transfer learning consistently breaks through the 78% CNN plateau on small
medical audio datasets (ICBHI 2017).

Phase 1: frozen MobileNetV2 base, train head only (lr=0.001, 15 epochs)
Phase 2: fine-tune top layers (lr=0.0001, 40 epochs)
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
LOG_FILE  = LOG_DIR / "lung_transfer_training.log"
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
IMG_SIZE  = 96     # MobileNetV2 min is 32; 96 is fast and effective
BATCH     = 32


def _prepare(X_raw: np.ndarray) -> np.ndarray:
    """Resize (H, W, 1) spectrograms to (IMG_SIZE, IMG_SIZE, 3) for MobileNetV2."""
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    log.info("Preparing %d spectrograms → (%d,%d,3) …", len(X_raw), IMG_SIZE, IMG_SIZE)
    # Convert to tensor, resize, repeat channels, then apply MobileNetV2 preprocess
    t = tf.constant(X_raw, dtype=tf.float32)                         # (N, H, W, 1)
    t = tf.image.resize(t, [IMG_SIZE, IMG_SIZE])                     # (N, 96, 96, 1)
    t = tf.repeat(t, 3, axis=-1)                                     # (N, 96, 96, 3)
    # preprocess_input expects [0,255] range — scale from [0,1] if needed
    mn = float(t.numpy().min()); mx = float(t.numpy().max())
    if mx <= 1.0:
        t = t * 255.0
    t = preprocess_input(t)                                          # → [-1, 1]
    out = t.numpy().astype(np.float32)
    log.info("Prep complete — shape %s  range [%.2f, %.2f]",
             out.shape, out.min(), out.max())
    return out


def _build_model(phase2: bool = False):
    from tensorflow import keras
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False  # frozen in phase 1

    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = base(inp, training=False)
    x   = keras.layers.Dense(128, activation="relu")(x)
    x   = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out, name="lung_mobilenet")
    return model, base


def _train_phase(model, X_tr, y_tr, X_val, y_val,
                 lr: float, epochs: int, patience: int,
                 class_weight: dict, ckpt_path: Path,
                 label: str) -> tuple[float, float, int]:
    from tensorflow import keras
    import tensorflow as tf

    tf.random.set_seed(42)
    np.random.seed(42)

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    log.info("%s  lr=%.5f  batch=%d  trainable_params=%s",
             label, lr, BATCH,
             f"{sum(np.prod(v.shape) for v in model.trainable_variables):,}")

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

    h = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH,
        class_weight=class_weight,
        callbacks=cbs,
        verbose=1,
    )

    best_i  = int(np.argmax(h.history["val_accuracy"]))
    val_acc = float(h.history["val_accuracy"][best_i])
    tr_acc  = float(h.history["accuracy"][best_i])
    ep      = len(h.history["val_accuracy"])
    return val_acc, tr_acc, ep


def _log_exp(attempt: int, total: int, arch: str, lr: str,
             epochs_run: int, train_acc: float, val_acc: float,
             next_action: str) -> None:
    met   = val_acc >= TARGET
    entry = (
        f"\n## {datetime.now().isoformat(timespec='seconds')}\n"
        f"- **Model**: lung-transfer  (attempt {attempt}/{total})\n"
        f"- **Architecture**: {arch}\n"
        f"- **Optimizer**: adam  lr={lr}\n"
        f"- **Batch**: {BATCH}\n"
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

    log.info("Loading lung data …")
    X_raw = np.load(PROC_DIR / "X.npy")[..., np.newaxis]
    y4    = np.load(PROC_DIR / "y.npy").astype(np.int32)
    y     = (y4 > 0).astype(np.int32)
    log.info("Dist: %s", dict(Counter(y.tolist())))

    X = _prepare(X_raw)

    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    log.info("Split — train %d  val %d  test %d", len(X_tr), len(X_val), len(X_te))

    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    cls_wt  = {0: float(weights[0]), 1: float(weights[1])}

    # ── Schedule ─────────────────────────────────────────────────────────────────
    # Each "attempt" is a full 2-phase training run with different fine-tune depth.
    # attempt 1: fine-tune last 20 layers
    # attempt 2: fine-tune last 50 layers (more capacity)
    # attempt 3: fine-tune full network (lr=0.00005)
    schedule = [
        dict(phase2_layers=20,   ft_lr=0.0001, head_lr=0.001,
             desc="MobileNetV2(frozen)+head → fine-tune last 20 layers"),
        dict(phase2_layers=50,   ft_lr=0.0001, head_lr=0.001,
             desc="MobileNetV2(frozen)+head → fine-tune last 50 layers"),
        dict(phase2_layers=None, ft_lr=0.00005, head_lr=0.001,
             desc="MobileNetV2(frozen)+head → fine-tune all layers lr=5e-5"),
    ]

    results:         list[tuple[int, float]] = []
    best_val_global  = 0.0
    total            = len(schedule)

    for attempt, cfg in enumerate(schedule, start=1):
        log.info("=" * 60)
        log.info("ATTEMPT %d/%d  fine-tune layers=%s  ft_lr=%.5f",
                 attempt, total, cfg["phase2_layers"], cfg["ft_lr"])

        ckpt_p1 = MODEL_DIR / f"_mbnet_ckpt_{attempt}_p1.keras"
        ckpt_p2 = MODEL_DIR / f"_mbnet_ckpt_{attempt}_p2.keras"

        model, base = _build_model()

        # Phase 1: frozen base, train head
        log.info("Phase 1: frozen MobileNetV2, training head …")
        val1, tr1, ep1 = _train_phase(
            model, X_tr, y_tr, X_val, y_val,
            lr=cfg["head_lr"], epochs=15, patience=5,
            class_weight=cls_wt, ckpt_path=ckpt_p1,
            label="P1",
        )
        log.info("Phase 1 best val_acc=%.4f", val1)

        # Load best phase-1 weights before fine-tuning
        if ckpt_p1.exists():
            from tensorflow import keras
            model = keras.models.load_model(ckpt_p1)
            # Re-extract base for unfreezing
            base = model.layers[1]
            try:
                ckpt_p1.unlink()
            except OSError:
                pass

        # Phase 2: unfreeze top N layers
        n_layers = len(base.layers)
        if cfg["phase2_layers"] is None:
            base.trainable = True
            log.info("Phase 2: fine-tuning all %d base layers", n_layers)
        else:
            base.trainable = True
            for layer in base.layers[:-cfg["phase2_layers"]]:
                layer.trainable = False
            unfrozen = sum(1 for l in base.layers if l.trainable)
            log.info("Phase 2: fine-tuning last %d of %d base layers",
                     unfrozen, n_layers)

        val2, tr2, ep2 = _train_phase(
            model, X_tr, y_tr, X_val, y_val,
            lr=cfg["ft_lr"], epochs=40, patience=8,
            class_weight=cls_wt, ckpt_path=ckpt_p2,
            label="P2",
        )
        val_acc = max(val1, val2)
        tr_acc  = tr2 if val2 >= val1 else tr1
        ep_total = ep1 + ep2

        results.append((attempt, val_acc))

        if val_acc > best_val_global:
            best_val_global = val_acc
            if ckpt_p2.exists():
                shutil.copy2(str(ckpt_p2), str(MODEL_OUT))
            log.info("New best (%.4f) → %s", val_acc, MODEL_OUT)
        try:
            ckpt_p2.unlink()
        except OSError:
            pass

        met = val_acc >= TARGET
        if met:
            next_action = "target reached — done"
        elif attempt == total:
            next_action = "all attempts exhausted"
        else:
            next_action = f"attempt {attempt+1}: {schedule[attempt]['desc']}"

        _log_exp(attempt, total, cfg["desc"],
                 f"{cfg['head_lr']}/{cfg['ft_lr']}",
                 ep_total, tr_acc, val_acc, next_action)
        log.info("Attempt %d  val_acc=%.4f  target_met=%s", attempt, val_acc, met)

        if met:
            break

    # ── Final evaluation ──────────────────────────────────────────────────────
    log.info("=" * 60)
    if not MODEL_OUT.exists():
        log.error("No model saved — something went wrong")
        sys.exit(1)

    from tensorflow import keras
    best_model = keras.models.load_model(MODEL_OUT)
    probs    = best_model.predict(X_te, batch_size=BATCH, verbose=0).squeeze()
    preds    = (probs >= 0.5).astype(int)
    test_acc = float(np.mean(preds == y_te))

    print(f"\n{'='*60}")
    print(f"  LUNG TRANSFER MODEL — TRAINING COMPLETE")
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
