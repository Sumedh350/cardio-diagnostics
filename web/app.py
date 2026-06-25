"""Flask web app for Cardio Diagnostics — heart & lung sound classification."""

import io
import logging
import struct
import threading
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

# ── Signal-processing constants (must match ml/setup_datasets.py) ─────────────
FS        = 4_000
N_MELS    = 64
N_MFCC    = 40
HOP_LEN   = 128
N_FFT     = 512
HEART_WIN = 5        # seconds → 20 000 samples
LUNG_WIN  = 10       # seconds → 40 000 samples
HEART_T   = 157      # expected time frames for heart spectrogram
LUNG_T    = 313      # overridden at startup from lung model's input_shape[2]

# ── Serial / live recording constants ─────────────────────────────────────────
FRAME_HDR         = bytes([0xAA, 0xBB])
SYNC_HDR          = bytes([0xFF, 0xFF])
SAMPLES_PER_BURST = 64
BAUD              = 115_200
MODE_HEART_BYTE   = 0x01
MODE_LUNG_BYTE    = 0x02

app = Flask(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR   = Path(__file__).parent.parent / "models"
_heart_model = None
_lung_model  = None

# ── Live classification state ─────────────────────────────────────────────────
_live = {"status": "idle", "progress": 0, "mode": None, "result": None, "error": None}
_live_lock   = threading.Lock()
_stop_flag   = threading.Event()


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _load_wav(file_bytes: bytes) -> tuple[np.ndarray, float]:
    import librosa
    y, _ = librosa.load(io.BytesIO(file_bytes), sr=FS, mono=True, dtype=np.float32)
    return y, float(len(y)) / FS


def _mel(y: np.ndarray) -> np.ndarray:
    import librosa
    S    = librosa.feature.melspectrogram(
        y=y, sr=FS, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS, fmax=FS // 2)
    S_db = librosa.power_to_db(S, ref=np.max)
    mu, sigma = float(S_db.mean()), float(S_db.std())
    return ((S_db - mu) / (sigma + 1e-8)).astype(np.float32)


def _mfcc(y: np.ndarray) -> np.ndarray:
    import librosa
    return librosa.feature.mfcc(
        y=y, sr=FS, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LEN, fmax=FS // 2
    ).astype(np.float32)


def _pad_or_crop(y: np.ndarray, target: int) -> np.ndarray:
    if len(y) >= target:
        return y[:target]
    return np.pad(y, (0, target - len(y)))


def _preprocess_heart(y: np.ndarray) -> np.ndarray:
    segment = _pad_or_crop(y, HEART_WIN * FS)
    mel     = _mel(segment)
    T       = mel.shape[1]
    mel     = mel[:, :HEART_T] if T >= HEART_T else np.pad(mel, ((0,0),(0,HEART_T-T)))
    return mel[np.newaxis, :, :, np.newaxis].astype(np.float32)


def _preprocess_lung(y: np.ndarray) -> np.ndarray:
    segment = _pad_or_crop(y, LUNG_WIN * FS)
    m       = _mfcc(segment)
    s       = _mel(segment)
    T       = min(m.shape[1], s.shape[1])
    feat    = np.concatenate([m[:, :T], s[:, :T]], axis=0)
    feat    = feat[:, :LUNG_T] if feat.shape[1] >= LUNG_T else np.pad(feat, ((0,0),(0,LUNG_T-feat.shape[1])))
    return feat[np.newaxis, :, :, np.newaxis].astype(np.float32)


def _classify_float(y: np.ndarray, sound_type: str, n_samples: int) -> dict:
    if sound_type == "heart":
        x    = _preprocess_heart(y)
        prob = float(_heart_model.predict(x, verbose=0)[0][0])
    else:
        x    = _preprocess_lung(y)
        prob = float(_lung_model.predict(x, verbose=0)[0][0])
    label      = "Abnormal" if prob >= 0.5 else "Normal"
    confidence = prob if prob >= 0.5 else 1.0 - prob
    return {
        "type":       sound_type,
        "label":      label,
        "confidence": round(confidence * 100, 1),
        "duration":   round(n_samples / FS, 2),
        "all_scores": {
            "normal":   round((1.0 - prob) * 100, 1),
            "abnormal": round(prob * 100, 1),
        },
    }


# ── Serial helpers (byte-level state machine, matches capture/capture_serial.py) ─

# ── Live recording thread (ASCII firmware — reads one integer per line) ──────
# Firmware sends "MODE:HEART" or "MODE:LUNG" at startup and every ~1 s,
# then streams raw ADC integers at ~2 kHz. Mirrors capture.py normalization.

LIVE_ARDUINO_SR = 2000   # firmware effective sample rate
# Samples needed: heart=4s, lung=10s at 2000 Hz
LIVE_SAMPLES = {"heart": 8000, "lung": 20000}

def _live_worker(port: str) -> None:
    import serial as pyserial
    import librosa

    def set_state(**kw):
        with _live_lock:
            _live.update(kw)

    try:
        ser = pyserial.Serial(port, BAUD, timeout=2)
    except Exception as e:
        set_state(status="error", error=f"Cannot open {port}: {e}")
        return

    import time
    time.sleep(1.5)
    ser.reset_input_buffer()

    try:
        # ── Step 1: detect mode from first MODE: line ─────────────────────────
        set_state(status="syncing", progress=0)
        sound_type = "heart"          # default if no MODE line seen quickly
        for _ in range(50):           # check up to 50 lines for MODE announcement
            if _stop_flag.is_set():
                raise InterruptedError("stopped")
            line = ser.readline().decode(errors="ignore").strip()
            if line == "MODE:HEART":
                sound_type = "heart"
                break
            elif line == "MODE:LUNG":
                sound_type = "lung"
                break

        target = LIVE_SAMPLES[sound_type]
        set_state(status="recording", mode=sound_type, progress=0)

        # ── Step 2: collect samples ───────────────────────────────────────────
        samples: list[int] = []

        while len(samples) < target and not _stop_flag.is_set():
            try:
                line = ser.readline().decode(errors="ignore").strip()
                if line.startswith("MODE:"):
                    continue              # ignore periodic mode re-broadcasts
                if line:
                    samples.append(int(line))
                    pct = min(99, int(len(samples) / target * 100))
                    set_state(progress=pct)
            except (ValueError, UnicodeDecodeError):
                pass

        ser.close()

        if _stop_flag.is_set():
            set_state(status="idle", progress=0)
            return

        set_state(status="classifying", progress=100)

        # ── Step 3: normalise exactly like capture.py ─────────────────────────
        audio = np.array(samples[:target], dtype=np.float32)
        audio -= audio.mean()
        peak = np.abs(audio).max()
        if peak > 0:
            audio /= peak

        # Resample 2000 → 4000 Hz (same as librosa.load on a 2000 Hz WAV)
        audio_4k = librosa.resample(audio, orig_sr=LIVE_ARDUINO_SR, target_sr=FS)

        result = _classify_float(audio_4k, sound_type, len(audio_4k))
        set_state(status="done", result=result, progress=100)

    except InterruptedError:
        set_state(status="idle", progress=0)
    except Exception as e:
        log.exception("Live worker error")
        set_state(status="error", error=str(e))
    finally:
        try:
            ser.close()
        except Exception:
            pass


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_models():
    global _heart_model, _lung_model
    from tensorflow import keras
    heart_path = MODELS_DIR / "heart_model.keras"
    lung_path  = MODELS_DIR / "lung_model_binary.keras"
    if heart_path.exists():
        log.info("Loading heart model from %s", heart_path)
        _heart_model = keras.models.load_model(heart_path)
        log.info("Heart model input: %s", _heart_model.input_shape)
    else:
        log.warning("heart_model.keras not found")
    if lung_path.exists():
        log.info("Loading lung model from %s", lung_path)
        _lung_model = keras.models.load_model(lung_path)
        log.info("Lung model input: %s", _lung_model.input_shape)
        global LUNG_T
        LUNG_T = int(_lung_model.input_shape[2])
        log.info("LUNG_T set to %d from model input shape", LUNG_T)
    else:
        log.warning("lung_model_binary.keras not found")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file_bytes = request.files["file"].read()
    try:
        y, duration = _load_wav(file_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not read WAV: {e}"}), 400

    sound_type = "heart" if duration <= 6.0 else "lung"
    if sound_type == "heart" and _heart_model is None:
        return jsonify({"error": "Heart model not loaded"}), 503
    if sound_type == "lung" and _lung_model is None:
        return jsonify({"error": "Lung model not loaded"}), 503
    return jsonify(_classify_float(y, sound_type, len(y)))


@app.route("/live/start")
def live_start():
    global _live_thread
    with _live_lock:
        if _live["status"] in ("recording", "syncing", "classifying"):
            return jsonify({"error": "Already running"}), 400
        _live.update(status="idle", progress=0, mode=None, result=None, error=None)

    port = request.args.get("port", "COM3")
    _stop_flag.clear()
    _live_thread = threading.Thread(target=_live_worker, args=(port,), daemon=True)
    _live_thread.start()
    return jsonify({"ok": True})


@app.route("/live/stop")
def live_stop():
    _stop_flag.set()
    with _live_lock:
        _live.update(status="idle", progress=0)
    return jsonify({"ok": True})


@app.route("/live/status")
def live_status():
    with _live_lock:
        return jsonify(dict(_live))


@app.route("/health")
def health():
    return jsonify({
        "heart_model": _heart_model is not None,
        "lung_model":  _lung_model is not None,
    })


if __name__ == "__main__":
    _load_models()
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
