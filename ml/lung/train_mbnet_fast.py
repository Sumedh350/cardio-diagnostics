#!/usr/bin/env python3
"""
ml/lung/train_mbnet_fast.py — Fast MobileNetV2 transfer learning via feature extraction.

Strategy:
  1. Pre-extract MobileNetV2(imagenet) features once  → (N, 1280) numpy array
  2. Train a tiny Dense head on those features         → very fast (seconds/epoch)
  3. Optionally fine-tune last N base layers           → only if still below 82%
  4. Save combined model (raw spectrogram → sigmoid)   → works with web app

Expected runtime: ~15-25 min total on CPU.
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT     = Path(__file__).resolve().parent.parent.parent
PROC_DIR = ROOT / "ml" / "lung" / "processed"
LOG_DIR  = ROOT / "ml" / "logs"
MODEL_DIR= ROOT / "models"
MODEL_OUT= MODEL_DIR / "lung_model_binary.keras"
LOG_FILE = LOG_DIR / "lung_mbnet_fast.log"
EXP_FILE = LOG_DIR / "experiments.md"

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

TARGET   = 0.82
IMG_SIZE = 96
BATCH    = 32


def _build_extractor(raw_shape: tuple):
    """Model: raw spectrogram → resize+preprocess inline → MobileNetV2 → (1280,) features.
    Processing happens batch-by-batch inside predict(), so peak memory ≈ BATCH samples."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                       include_top=False, weights="imagenet", pooling="avg")
    base.trainable = False

    inp     = keras.Input(shape=raw_shape)
    resized = keras.layers.Lambda(
        lambda x: preprocess_input(
            tf.repeat(tf.image.resize(x, [IMG_SIZE, IMG_SIZE]), 3, axis=-1) * 255.0
        )
    )(inp)
    feats = base(resized, training=False)
    extractor = keras.Model(inp, feats, name="extractor")
    return extractor, base


def _extract_features(X_raw: np.ndarray, raw_shape: tuple):
    """One forward pass through frozen MobileNetV2 → (N, 1280) feature vectors.
    Operates batch-by-batch — peak extra memory is BATCH * 96*96*3*4 bytes ≈ 3 MB."""
    extractor, base = _build_extractor(raw_shape)
    log.info("Extracting features from %d samples (batch=%d) …", len(X_raw), BATCH)
    feats = extractor.predict(X_raw, batch_size=BATCH, verbose=1)
    log.info("Features shape: %s", feats.shape)
    return feats, base


def _build_head(feat_dim: int = 1280):
    from tensorflow import keras
    inp = keras.Input(shape=(feat_dim,))
    x   = keras.layers.Dense(256, activation="relu")(inp)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.Dropout(0.3)(x)
    x   = keras.layers.Dense(64, activation="relu")(x)
    x   = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="head")


def _train_head(head, feats_tr, y_tr, feats_val, y_val,
                cls_wt, lr: float, epochs: int, patience: int,
                ckpt_path: Path) -> tuple[float, float, int]:
    from tensorflow import keras
    import tensorflow as tf
    tf.random.set_seed(42); np.random.seed(42)
    head.compile(optimizer=keras.optimizers.Adam(lr),
                 loss="binary_crossentropy", metrics=["accuracy"])
    log.info("Head params: %s  lr=%.4f  batch=%d",
             f"{head.count_params():,}", lr, BATCH)
    cbs = [
        keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_accuracy",
            save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1),
    ]
    h = head.fit(feats_tr, y_tr, validation_data=(feats_val, y_val),
                 epochs=epochs, batch_size=BATCH,
                 class_weight=cls_wt, callbacks=cbs, verbose=1)
    best_i  = int(np.argmax(h.history["val_accuracy"]))
    return (float(h.history["val_accuracy"][best_i]),
            float(h.history["accuracy"][best_i]),
            len(h.history["val_accuracy"]))


def _fine_tune(base, head, X_raw_tr, y_tr, X_raw_val, y_val,
               n_unfreeze: int, cls_wt, lr: float,
               epochs: int, patience: int,
               ckpt_path: Path, raw_shape: tuple) -> tuple[float, float, int]:
    """Build combined model, unfreeze last n_unfreeze base layers, fine-tune.
    Takes raw spectrograms directly (resize happens inside the model)."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    tf.random.set_seed(42); np.random.seed(42)

    base.trainable = True
    for layer in base.layers[:-n_unfreeze]:
        layer.trainable = False
    unfrozen = sum(1 for l in base.layers if l.trainable)
    log.info("Fine-tuning: %d of %d base layers unfrozen", unfrozen, len(base.layers))

    inp     = keras.Input(shape=raw_shape)
    resized = keras.layers.Lambda(
        lambda x: preprocess_input(
            tf.repeat(tf.image.resize(x, [IMG_SIZE, IMG_SIZE]), 3, axis=-1) * 255.0
        )
    )(inp)
    feats = base(resized, training=True)
    out   = head(feats)
    combo = keras.Model(inp, out, name="lung_finetune")
    combo.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    log.info("Combined params: %s  trainable: %s",
             f"{combo.count_params():,}",
             f"{sum(np.prod(v.shape) for v in combo.trainable_variables):,}")

    cbs = [
        keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_accuracy",
            save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-8, verbose=1),
    ]
    h = combo.fit(X_raw_tr, y_tr, validation_data=(X_raw_val, y_val),
                  epochs=epochs, batch_size=BATCH,
                  class_weight=cls_wt, callbacks=cbs, verbose=1)
    best_i = int(np.argmax(h.history["val_accuracy"]))
    return (float(h.history["val_accuracy"][best_i]),
            float(h.history["accuracy"][best_i]),
            len(h.history["val_accuracy"]),
            combo)


def _build_combined_model(base, head, raw_shape: tuple):
    """Wrap resize + base + head into a single model for inference."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    inp     = keras.Input(shape=raw_shape, name="spectrogram")
    resized = keras.layers.Lambda(
        lambda x: preprocess_input(
            tf.repeat(tf.image.resize(x, [IMG_SIZE, IMG_SIZE]), 3, axis=-1) * 255.0
        )
    )(inp)
    feats   = base(resized, training=False)
    out     = head(feats)
    return keras.Model(inp, out, name="lung_mbnet_binary")


def _log_exp(attempt: int, total: int, arch: str, lr: str,
             epochs_run: int, train_acc: float, val_acc: float,
             next_action: str) -> None:
    met   = val_acc >= TARGET
    entry = (
        f"\n## {datetime.now().isoformat(timespec='seconds')}\n"
        f"- **Model**: lung-mbnet  (attempt {attempt}/{total})\n"
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

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading lung data …")
    X_raw = np.load(PROC_DIR / "X.npy")[..., np.newaxis]
    y4    = np.load(PROC_DIR / "y.npy").astype(np.int32)
    y     = (y4 > 0).astype(np.int32)
    RAW_SHAPE = X_raw.shape[1:]
    log.info("Dist: normal=%d  abnormal=%d",
             *[dict(Counter(y.tolist()))[k] for k in [0, 1]])

    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    # Split the raw data (no full resize in memory)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_raw, y, test_size=0.20, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    log.info("Split — train %d  val %d  test %d", len(X_tr), len(X_val), len(X_te))

    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    cls_wt  = {0: float(weights[0]), 1: float(weights[1])}

    # ── Step 1: Extract features (done ONCE, batch-by-batch, low peak memory) ─
    log.info("=" * 60)
    log.info("STEP 1: Extracting MobileNetV2 features (batch-by-batch, no big alloc)")
    feats_all, base = _extract_features(X_raw, RAW_SHAPE)
    # Re-align to same train/val/test split order
    # train_test_split shuffles, so extract using index ordering
    # Easier: just split feats_all using the same indices
    # Since we split X_raw the same way, split feats_all identically
    feats_tr, feats_tmp, _, _ = train_test_split(
        feats_all, y, test_size=0.20, stratify=y, random_state=42)
    feats_val, feats_te, _, _ = train_test_split(
        feats_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    log.info("Feature splits: tr=%d  val=%d  te=%d",
             len(feats_tr), len(feats_val), len(feats_te))

    # ── Step 2: Train Dense head on frozen features ───────────────────────────
    log.info("=" * 60)
    log.info("STEP 2: Training Dense head on frozen features …")
    head  = _build_head(feat_dim=feats_tr.shape[1])
    ckpt1 = MODEL_DIR / "_mbnet_head.keras"
    val_head, tr_head, ep_head = _train_head(
        head, feats_tr, y_tr, feats_val, y_val,
        cls_wt=cls_wt, lr=0.001, epochs=100, patience=15,
        ckpt_path=ckpt1,
    )
    log.info("Head-only val_acc=%.4f", val_head)

    from tensorflow import keras
    head = keras.models.load_model(ckpt1)
    try: ckpt1.unlink()
    except OSError: pass

    fine_tuned = False
    val_ft     = 0.0

    if val_head >= TARGET:
        log.info("Target reached with frozen features — building combined model")
        combined = _build_combined_model(base, head, RAW_SHAPE)
        combined.save(str(MODEL_OUT))
        val_acc_final = val_head
        tr_acc_final  = tr_head
        ep_final      = ep_head
        phase_desc    = "MobileNetV2(frozen,96x96) → Dense(256,BN,0.3) → Dense(64,0.2) → sigmoid"
        next_action   = "target reached — done"
    else:
        fine_tuned = True
        log.info("Head-only %.4f < %.2f — fine-tuning last 30 base layers …",
                 val_head, TARGET)

        # ── Step 3: Fine-tune last 30 layers (on raw spectrograms) ───────────
        log.info("=" * 60)
        log.info("STEP 3: Fine-tuning MobileNetV2 last 30 layers on raw spectrograms …")
        ckpt2 = MODEL_DIR / "_mbnet_finetune.keras"
        val_ft, tr_ft, ep_ft, combo = _fine_tune(
            base, head, X_tr, y_tr, X_val, y_val,
            n_unfreeze=30, cls_wt=cls_wt,
            lr=0.0001, epochs=40, patience=8,
            ckpt_path=ckpt2, raw_shape=RAW_SHAPE,
        )
        log.info("Fine-tune val_acc=%.4f", val_ft)

        if val_ft > val_head:
            combo_best    = keras.models.load_model(ckpt2)
            val_acc_final = val_ft
            tr_acc_final  = tr_ft
            ep_final      = ep_head + ep_ft
        else:
            combo_best    = _build_combined_model(base, head, RAW_SHAPE)
            val_acc_final = val_head
            tr_acc_final  = tr_head
            ep_final      = ep_head + ep_ft

        try: ckpt2.unlink()
        except OSError: pass

        combo_best.save(str(MODEL_OUT))
        phase_desc  = ("MobileNetV2(frozen→ft30,96x96) → "
                       "Dense(256,BN,0.3) → Dense(64,0.2) → sigmoid")
        next_action = ("target reached — done"
                       if val_acc_final >= TARGET else "all attempts exhausted")

    # ── Log ───────────────────────────────────────────────────────────────────
    _log_exp(
        attempt=1, total=1,
        arch=phase_desc,
        lr="0.001/0.0001" if fine_tuned else "0.001",
        epochs_run=ep_final,
        train_acc=tr_acc_final,
        val_acc=val_acc_final,
        next_action=next_action,
    )

    # ── Final evaluation ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Loading saved model for final test evaluation …")
    final_model = keras.models.load_model(MODEL_OUT)
    probs    = final_model.predict(X_te, batch_size=BATCH, verbose=0).squeeze()
    preds    = (probs >= 0.5).astype(int)
    test_acc = float(np.mean(preds == y_te))

    print(f"\n{'='*60}")
    print(f"  LUNG MBNET — TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Head-only val accuracy : {val_head:.4f}")
    if fine_tuned:
        print(f"  Fine-tune val accuracy : {val_ft:.4f}")
    print(f"  Best val accuracy      : {val_acc_final:.4f}")
    print(f"  Test accuracy          : {test_acc:.4f}")
    print(f"  Target (≥ {TARGET})       : {'✓ MET' if val_acc_final >= TARGET else '✗ NOT MET'}")
    print(f"  Model saved to         : {MODEL_OUT}")
    _print_metrics(y_te, preds)


if __name__ == "__main__":
    main()
