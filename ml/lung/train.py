#!/usr/bin/env python3
"""
ml/lung/train.py — Lung sound 4-class classifier (Conv2D on MFCC+Mel features).

Architecture (attempt 1):
  Conv2D(32,3×3,relu) → BatchNorm → MaxPool(2×2)
  Conv2D(64,3×3,relu) → BatchNorm → MaxPool(2×2)
  Conv2D(128,3×3,relu) → BatchNorm → MaxPool(2×2)
  GlobalAveragePooling2D → Dense(256,relu) → Dropout(0.5) → Dense(4,softmax)

Classes: 0=normal  1=crackle  2=wheeze  3=both

Auto-retrain schedule (CLAUDE.md § Retraining Rules):
  Attempt 1 : base 3-block CNN, Adam lr=0.001
  Attempt 2 : +Conv2D(256), Adam lr=0.0005
  Attempt 3 : +SpecAugment (time T=20, freq F=8), Adam lr=0.0003
  Attempt 4 : SGD momentum=0.9, lr=0.001
  All attempts logged to ml/logs/experiments.md
"""
from __future__ import annotations

import logging
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent.parent
PROC_DIR  = ROOT / "ml" / "lung" / "processed"
LOG_DIR   = ROOT / "ml" / "logs"
MODEL_DIR = ROOT / "models"
MODEL_OUT = MODEL_DIR / "lung_model.keras"
LOG_FILE  = LOG_DIR / "lung_training.log"
EXP_FILE  = LOG_DIR / "experiments.md"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET    = 0.82
N_CLS     = 4
CLS_NAMES = ["normal", "crackle", "wheeze", "both"]
EPOCHS    = 50
BATCH     = 32
EARLY_PAT = 7


# ── Model ──────────────────────────────────────────────────────────────────────
def _build(shape: tuple, extra_conv: bool = False):
    from tensorflow import keras

    inp = keras.Input(shape=shape)
    x   = inp
    for f in [32, 64, 128] + ([256] if extra_conv else []):
        x = keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(2)(x)
    x   = keras.layers.GlobalAveragePooling2D()(x)
    x   = keras.layers.Dense(256, activation="relu")(x)
    x   = keras.layers.Dropout(0.5)(x)
    out = keras.layers.Dense(N_CLS, activation="softmax")(x)
    return keras.Model(inp, out, name="lung_cnn")


# ── Training run ───────────────────────────────────────────────────────────────
def _train(
    X_tr, y_tr, X_val, y_val,
    shape: tuple,
    lr: float,
    opt_name: str,
    extra_conv: bool,
    augment: bool,
    ckpt_path: Path,
    class_weight: dict,
) -> tuple[float, float, int]:
    """Returns (best_val_acc, train_acc_at_best, epochs_run)."""
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(42)
    np.random.seed(42)

    model = _build(shape, extra_conv)
    opt   = (keras.optimizers.SGD(lr, momentum=0.9)
             if opt_name == "sgd" else keras.optimizers.Adam(lr))
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    log.info("Params: %s  shape=%s  lr=%.4f  opt=%s  extra=%s  aug=%s",
             f"{model.count_params():,}", shape, lr, opt_name, extra_conv, augment)

    cbs = [
        keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_accuracy", save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=EARLY_PAT,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    if augment:
        class AugSeq(keras.utils.Sequence):
            def __init__(self, X, y, bs, T=20, F=8):
                self.X, self.y, self.bs = X, y, bs
                self.T, self.F = T, F
                self._idx = np.arange(len(X))

            def __len__(self):
                return int(np.ceil(len(self.X) / self.bs))

            def __getitem__(self, i):
                bi = self._idx[i * self.bs:(i + 1) * self.bs]
                Xb = self.X[bi].copy()
                yb = self.y[bi]
                fq, tm = Xb.shape[1], Xb.shape[2]
                for k in range(len(Xb)):
                    f = np.random.randint(0, self.F + 1)
                    if f > 0:
                        f0 = np.random.randint(0, max(1, fq - f + 1))
                        Xb[k, f0:f0 + f, :, :] = 0.0
                    t = np.random.randint(0, self.T + 1)
                    if t > 0:
                        t0 = np.random.randint(0, max(1, tm - t + 1))
                        Xb[k, :, t0:t0 + t, :] = 0.0
                return Xb, yb

            def on_epoch_end(self):
                np.random.shuffle(self._idx)

        h = model.fit(
            AugSeq(X_tr, y_tr, BATCH),
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            class_weight=class_weight,
            callbacks=cbs,
            verbose=1,
        )
    else:
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


# ── Experiment logger ──────────────────────────────────────────────────────────
def _log_exp(
    attempt: int,
    arch: str,
    lr: float,
    opt: str,
    aug: bool,
    epochs_run: int,
    train_acc: float,
    val_acc: float,
    next_action: str,
) -> None:
    met   = val_acc >= TARGET
    entry = (
        f"\n## {datetime.now().isoformat(timespec='seconds')}\n"
        f"- **Model**: lung  (attempt {attempt}/4)\n"
        f"- **Architecture**: {arch}\n"
        f"- **Optimizer**: {opt}  lr={lr}\n"
        f"- **Augmentation**: {'SpecAugment T=20 F=8' if aug else 'none'}\n"
        f"- **Epochs run**: {epochs_run}\n"
        f"- **Train accuracy**: {train_acc:.4f}\n"
        f"- **Val accuracy**: {val_acc:.4f}\n"
        f"- **Target met**: {'✓ yes' if met else '✗ no'}  (target ≥ {TARGET})\n"
        f"- **Next action**: {next_action}\n"
    )
    with open(EXP_FILE, "a", encoding="utf-8") as fh:
        fh.write(entry)
    log.info("Logged attempt %d → experiments.md", attempt)


# ── Metrics ────────────────────────────────────────────────────────────────────
def _print_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    from sklearn.metrics import classification_report, confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLS)))
    print("\n── Confusion matrix ─────────────────────────────────")
    header = "              " + "".join(f"{n:>10}" for n in CLS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {CLS_NAMES[i]:12s}" + "".join(f"{v:>10}" for v in row))

    print("\n── Per-class metrics ────────────────────────────────")
    print(classification_report(y_true, y_pred,
                                labels=list(range(N_CLS)),
                                target_names=CLS_NAMES, digits=4))


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── GPU check ─────────────────────────────────────────────────────────────
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        log.info("GPU available: %s", [g.name for g in gpus])
    else:
        log.warning("No GPU detected — training on CPU (expect ~5–15 min/epoch)")

    # ── Load data ─────────────────────────────────────────────────────────────
    if not (PROC_DIR / "X.npy").exists():
        log.error("X.npy not found — run ml/setup_datasets.py first")
        sys.exit(1)

    log.info("Loading lung data from %s", PROC_DIR)
    X = np.load(PROC_DIR / "X.npy")[..., np.newaxis]    # (N, 104, 313, 1)
    y = np.load(PROC_DIR / "y.npy").astype(np.int32)    # (N,)
    INPUT_SHAPE = X.shape[1:]
    log.info("X%s  y%s  dist=%s", X.shape, y.shape, dict(Counter(y.tolist())))

    # ── Stratified 80 / 10 / 10 split ────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    log.info("Split — train %d  val %d  test %d", len(X_tr), len(X_val), len(X_te))

    # Balanced class weights (minority classes get higher weight)
    classes  = np.arange(N_CLS, dtype=int)
    weights  = compute_class_weight("balanced", classes=classes, y=y_tr)
    cls_wt   = {int(c): float(w) for c, w in zip(classes, weights)}
    log.info("Class weights: %s", {CLS_NAMES[k]: f"{v:.3f}" for k, v in cls_wt.items()})

    # ── Auto-retrain schedule ─────────────────────────────────────────────────
    schedule = [
        dict(lr=0.001,  opt="adam", extra_conv=False, aug=False,
             arch="Conv2D(32,64,128) → GAP → Dense(256) → Dropout(0.5) → softmax(4)"),
        dict(lr=0.0005, opt="adam", extra_conv=True,  aug=False,
             arch="Conv2D(32,64,128,256) → GAP → Dense(256) → Dropout(0.5) → softmax(4)"),
        dict(lr=0.0003, opt="adam", extra_conv=True,  aug=True,
             arch="Conv2D(32,64,128,256)+SpecAugment → GAP → Dense(256) → Dropout(0.5) → softmax(4)"),
        dict(lr=0.001,  opt="sgd",  extra_conv=True,  aug=True,
             arch="Conv2D(32,64,128,256)+SpecAugment+SGD(mom=0.9) → GAP → Dense(256) → softmax(4)"),
    ]

    results: list[tuple[int, float]] = []
    best_val_global = 0.0

    for attempt, cfg in enumerate(schedule, start=1):
        log.info("=" * 60)
        log.info("ATTEMPT %d/4  lr=%.4f  opt=%s  extra=%s  aug=%s",
                 attempt, cfg["lr"], cfg["opt"], cfg["extra_conv"], cfg["aug"])

        ckpt_path = MODEL_DIR / f"_lung_ckpt_{attempt}.keras"

        val_acc, tr_acc, ep = _train(
            X_tr, y_tr, X_val, y_val,
            shape=INPUT_SHAPE,
            lr=cfg["lr"],
            opt_name=cfg["opt"],
            extra_conv=cfg["extra_conv"],
            augment=cfg["aug"],
            ckpt_path=ckpt_path,
            class_weight=cls_wt,
        )
        results.append((attempt, val_acc))

        if val_acc > best_val_global:
            best_val_global = val_acc
            shutil.copy2(str(ckpt_path), str(MODEL_OUT))
            log.info("New best model (%.4f) copied to %s", val_acc, MODEL_OUT)
        try:
            ckpt_path.unlink()
        except OSError:
            pass

        met = val_acc >= TARGET
        if met or attempt == len(schedule):
            next_action = "target reached — done" if met else "all 4 attempts exhausted"
        else:
            next_cfg    = schedule[attempt]
            next_action = f"attempt {attempt+1}: {next_cfg['arch'][:70]}"

        _log_exp(attempt, cfg["arch"], cfg["lr"], cfg["opt"], cfg["aug"],
                 ep, tr_acc, val_acc, next_action)
        log.info("Attempt %d  val_acc=%.4f  target_met=%s", attempt, val_acc, met)

        if met:
            log.info("Target %.2f reached — stopping early", TARGET)
            break

    # ── Final evaluation ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Loading best model from %s", MODEL_OUT)
    from tensorflow import keras
    model    = keras.models.load_model(MODEL_OUT)
    logits   = model.predict(X_te, batch_size=BATCH, verbose=0)
    preds    = np.argmax(logits, axis=1).astype(int)
    test_acc = float(np.mean(preds == y_te))

    print(f"\n{'='*60}")
    print(f"  LUNG MODEL — TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Attempts run      : {len(results)}")
    for att, vac in results:
        print(f"    attempt {att}  val_acc={vac:.4f}  "
              f"{'✓' if vac >= TARGET else '✗'}")
    print(f"  Best val accuracy : {best_val_global:.4f}")
    print(f"  Test accuracy     : {test_acc:.4f}")
    print(f"  Target (≥ {TARGET})   : {'✓ MET' if best_val_global >= TARGET else '✗ NOT MET'}")
    print(f"  Model saved to    : {MODEL_OUT}")
    _print_metrics(y_te, preds)


if __name__ == "__main__":
    main()
