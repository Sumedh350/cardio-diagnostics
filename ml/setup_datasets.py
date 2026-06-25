#!/usr/bin/env python3
"""
ml/setup_datasets.py — Download and preprocess cardio-diagnostics datasets.

STEP 1  Download PhysioNet CinC 2016 heart dataset (training-a … training-f)
STEP 2  Download ICBHI 2017 lung dataset
STEP 3  Preprocess heart  → Mel-spectrogram (64 bands), 5 s windows, 50 % overlap
STEP 4  Preprocess lung   → MFCC (40) + Mel (64) 2-channel, 10 s windows, 50 % overlap
STEP 5  Print dataset summary and training-time warnings

Signal pipeline (both datasets)
    original SR  →  resample to 4000 Hz  →  segment  →  extract features  →  .npy

Label schemes
    Heart : PhysioNet REFERENCE.csv  |  1=normal→0, -1=abnormal→1
    Lung  : ICBHI per-cycle .txt     |  (crackle,wheeze): 00→normal 10→crackle
                                     |                     01→wheeze 11→both
            ICBHI_Challenge_diagnosis.txt  read for reference (patient-level)
"""

from __future__ import annotations

import csv
import os
import sys
import urllib3
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)   # ICBHI cert is expired

try:
    import librosa
    import librosa.feature
    import soundfile  # noqa: F401 — required by librosa on Windows
except ImportError:
    sys.exit("Install deps: pip install librosa soundfile numpy scipy requests tqdm wfdb")

# ── PhysioNet credentials (set before running, or export in your shell) ───────
# PhysioNet requires a free account even for "Open Access" data.
# Register at: https://physionet.org/register/
# Then either:
#   set PHYSIONET_USER=your_username  &&  set PHYSIONET_PASSWORD=your_password
# or pass --physionet-user / --physionet-password on the CLI (see main()).
_PN_USER = os.getenv("PHYSIONET_USER", "")
_PN_PASS = os.getenv("PHYSIONET_PASSWORD", "")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
HEART_RAW  = ROOT / "ml" / "heart" / "raw"
HEART_PROC = ROOT / "ml" / "heart" / "processed"
LUNG_RAW   = ROOT / "ml" / "lung"  / "raw"
LUNG_PROC  = ROOT / "ml" / "lung"  / "processed"

# ── Dataset URLs ──────────────────────────────────────────────────────────────
HEART_URLS = [
    "https://physionet.org/files/challenge-2016/1.0.0/training-a.zip",
    "https://physionet.org/files/challenge-2016/1.0.0/training-b.zip",
    "https://physionet.org/files/challenge-2016/1.0.0/training-c.zip",
    "https://physionet.org/files/challenge-2016/1.0.0/training-d.zip",
    "https://physionet.org/files/challenge-2016/1.0.0/training-e.zip",
    "https://physionet.org/files/challenge-2016/1.0.0/training-f.zip",
]

LUNG_URL = (
    "https://bhichallenge.med.auth.gr/sites/default/files/"
    "ICBHI_challenge_data/ICBHI_final_database.zip"
)

# ── Signal-processing constants ───────────────────────────────────────────────
FS        = 4_000    # target sample rate — matches Arduino firmware exactly
HEART_WIN = 5        # heart window seconds   → 20 000 samples
LUNG_WIN  = 10       # lung  window seconds   → 40 000 samples
OVERLAP   = 0.50     # 50 % overlap
N_MELS    = 64       # Mel filter banks
N_MFCC    = 40       # MFCC coefficients
HOP_LEN   = 128      # hop length (samples)
N_FFT     = 512      # FFT window

# ── Label maps ────────────────────────────────────────────────────────────────
HEART_LABEL_MAP  = {1: 0, -1: 1}          # normal=0, abnormal=1
HEART_CLASS_NAMES = ["normal", "abnormal"]

# ICBHI per-cycle: (crackle_flag, wheeze_flag) → class int
LUNG_LABEL_MAP = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
LUNG_CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _hdr(title: str) -> None:
    print(f"\n{'─'*62}\n  {title}\n{'─'*62}")


def _download(
    url: str,
    dest: Path,
    label: str,
    *,
    auth: tuple[str, str] | None = None,
    verify_ssl: bool = True,
) -> bool:
    """
    Stream-download url → dest with a tqdm bytes bar.

    auth       : (user, password) for HTTP Basic Auth (PhysioNet).
    verify_ssl : set False for servers with expired certs (ICBHI).
    Skips the download if dest already exists.
    """
    if dest.exists():
        mb = dest.stat().st_size / 1_048_576
        print(f"  ✓ {label}: already on disk ({mb:.1f} MB)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {label}")
    try:
        r = requests.get(
            url,
            stream=True,
            timeout=180,
            verify=verify_ssl,
            auth=auth,
            headers={"User-Agent": "cardio-diagnostics/1.0"},
        )
        r.raise_for_status()

        # PhysioNet returns HTML (login page) when credentials are wrong/missing
        ct = r.headers.get("content-type", "")
        if "text/html" in ct:
            raise requests.HTTPError(
                "Server returned HTML — credentials missing or wrong.\n"
                f"    URL: {url}"
            )

        total = int(r.headers.get("content-length", 0))
        with (
            open(dest, "wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True,
                 unit_divisor=1024, ncols=70, leave=False) as bar,
        ):
            for chunk in r.iter_content(chunk_size=65_536):
                fh.write(chunk)
                bar.update(len(chunk))

        mb = dest.stat().st_size / 1_048_576
        print(f"  ✓ {label}: {mb:.1f} MB saved")
        return True

    except requests.RequestException as exc:
        print(f"  ✗ {label}: {exc}", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return False


def _extract(zip_path: Path, dest: Path, label: str) -> bool:
    """Extract zip → dest/ with per-member tqdm bar. Idempotent via sentinel."""
    sentinel = dest / f".done_{zip_path.stem}"
    if sentinel.exists():
        print(f"  ✓ {label}: already extracted")
        return True

    if not zip_path.exists():
        print(f"  ✗ {label}: zip not found, skipping extraction", file=sys.stderr)
        return False

    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            members = z.namelist()
            for m in tqdm(members, desc=f"  unzip {label}", ncols=70, leave=False):
                z.extract(m, dest)
        sentinel.touch()
        print(f"  ✓ {label}: {len(members)} files extracted")
        return True
    except zipfile.BadZipFile as exc:
        print(f"  ✗ {label}: bad zip — {exc}\n"
              f"     The file may be an HTML login page. "
              f"Download manually from:\n     {zip_path}", file=sys.stderr)
        return False


def _windows(signal: np.ndarray, win: int, step: int):
    """Yield fixed-length non-overlapping sub-arrays; drops the short tail."""
    i = 0
    while i + win <= len(signal):
        yield signal[i : i + win]
        i += step


def _load(path: Path) -> np.ndarray:
    """Load audio file and resample to FS Hz, returning float32 mono array."""
    y, _ = librosa.load(str(path), sr=FS, mono=True, dtype=np.float32)
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractors
# ─────────────────────────────────────────────────────────────────────────────

def _mel(y: np.ndarray) -> np.ndarray:
    """
    Log-Mel-spectrogram, zero-mean unit-variance normalised.
    Input : 1-D float32, length = HEART_WIN * FS = 20 000 samples
    Output: (N_MELS=64, T≈157) float32
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=FS, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS,
        fmax=FS // 2,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    mu, sigma = float(S_db.mean()), float(S_db.std())
    return ((S_db - mu) / (sigma + 1e-8)).astype(np.float32)


def _mfcc(y: np.ndarray) -> np.ndarray:
    """
    MFCC coefficients.
    Input : 1-D float32
    Output: (N_MFCC=40, T) float32
    """
    return librosa.feature.mfcc(
        y=y, sr=FS, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LEN,
        fmax=FS // 2,
    ).astype(np.float32)


def _lung_feat(y: np.ndarray) -> np.ndarray:
    """
    2-channel lung feature: MFCC stacked on top of Mel-spectrogram.
    Input : 1-D float32, length = LUNG_WIN * FS = 40 000 samples
    Output: (N_MFCC + N_MELS = 104, T≈313) float32
    """
    m = _mfcc(y)   # (40, T)
    s = _mel(y)    # (64, T)
    T = min(m.shape[1], s.shape[1])
    return np.concatenate([m[:, :T], s[:, :T]], axis=0)   # (104, T)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Download heart dataset
# ─────────────────────────────────────────────────────────────────────────────

_PHYSIONET_MANUAL = """
  ── PhysioNet manual download (takes ~5 minutes) ──────────────────────────
  PhysioNet requires a free account even for Open Access datasets.

  Option A — browser download (easiest):
    1. Register / log in at  https://physionet.org/register/
    2. Open each URL and click Download:
       https://physionet.org/content/challenge-2016/1.0.0/#files-panel
    3. Download training-a.zip … training-f.zip
    4. Place them in:  {raw_dir}

  Option B — wget with credentials (fastest):
    set PHYSIONET_USER=your_username
    set PHYSIONET_PASSWORD=your_password
    python ml/setup_datasets.py

  Option C — PhysioNet CLI:
    pip install wfdb
    python -c "
    import wfdb
    for s in 'abcdef':
        wfdb.dl_database('challenge-2016',
            dl_dir=r'{{raw_dir}}',
            records=['training-' + s + '/REFERENCE'],
            keep_subdirs=True)
    "
  ──────────────────────────────────────────────────────────────────────────
"""


def step1_download_heart() -> list[str]:
    _hdr("STEP 1 — Download PhysioNet CinC 2016 heart dataset")
    HEART_RAW.mkdir(parents=True, exist_ok=True)

    auth = (_PN_USER, _PN_PASS) if _PN_USER and _PN_PASS else None
    if not auth:
        print("  ℹ PHYSIONET_USER / PHYSIONET_PASSWORD not set — trying unauthenticated")

    # ── Handle combined zip (PhysioNet full-dataset download) ─────────────────
    # The PhysioNet files panel produces one large zip containing all training-a…f
    # inside a versioned subdirectory.  Detect and extract it before anything else.
    combined_zips = [
        z for z in HEART_RAW.glob("*.zip")
        if not z.stem.startswith("training-")
    ]
    for czip in combined_zips:
        mb = czip.stat().st_size / 1_048_576
        print(f"  Found combined zip: {czip.name}  ({mb:.0f} MB)")
        _extract(czip, HEART_RAW, "heart-combined")

    # ── Try downloading individual training-?.zip files (original flow) ───────
    downloaded: list[str] = []
    for url in HEART_URLS:
        name = url.rsplit("/", 1)[-1]
        if _download(url, HEART_RAW / name, name, auth=auth):
            downloaded.append(name)

    # Also pick up any individually placed training-?.zip files
    for z in HEART_RAW.glob("training-?.zip"):
        if z.name not in downloaded:
            downloaded.append(z.name)

    if not downloaded and not combined_zips:
        print(_PHYSIONET_MANUAL.format(raw_dir=HEART_RAW), file=sys.stderr)

    extracted: list[str] = []
    for name in downloaded:
        stem = name.replace(".zip", "")
        if _extract(HEART_RAW / name, HEART_RAW, stem):
            extracted.append(stem)

    # Count WAVs across both flat (training-a/*.wav) and nested (**/training-a/*.wav)
    wav_count = sum(1 for _ in HEART_RAW.glob("**/training-*/*.wav"))
    print(f"\n  Result: {wav_count} heart WAV files found")
    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Download lung dataset
# ─────────────────────────────────────────────────────────────────────────────

_ICBHI_MANUAL = """
  ── ICBHI manual download ─────────────────────────────────────────────────
  The ICBHI server's TLS certificate has expired; direct download succeeded
  with SSL verification disabled, but if it failed:

    1. Visit   https://bhichallenge.med.auth.gr  (free registration)
    2. Download ICBHI_final_database.zip  (~1.3 GB)
    3. Place it at:
         {dest}
    Then re-run:  python ml/setup_datasets.py
  ──────────────────────────────────────────────────────────────────────────
"""


def step2_download_lung() -> bool:
    _hdr("STEP 2 — Download ICBHI 2017 lung dataset")
    LUNG_RAW.mkdir(parents=True, exist_ok=True)

    dest = LUNG_RAW / "ICBHI_final_database.zip"

    # The ICBHI server has an expired TLS cert — bypass verification with a warning
    print("  ⚠ Note: ICBHI server has an expired TLS certificate; "
          "using verify=False (data integrity checked by ZIP CRC)")
    ok = _download(LUNG_URL, dest, "ICBHI_final_database.zip", verify_ssl=False)

    if not ok:
        # Check for manually placed zip
        if dest.exists():
            print("  Found existing ICBHI zip, attempting extraction")
            ok = True
        else:
            print(_ICBHI_MANUAL.format(dest=dest), file=sys.stderr)
            return False

    return _extract(dest, LUNG_RAW, "ICBHI_final_database")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Preprocess heart
# ─────────────────────────────────────────────────────────────────────────────

def _heart_labels() -> dict[str, int]:
    """Merge all REFERENCE.csv files → {wav_stem: class_int (0=normal, 1=abnormal)}."""
    labels: dict[str, int] = {}
    # Handles both flat extraction (training-a/REFERENCE.csv) and the PhysioNet
    # combined zip which nests everything under a long versioned subdirectory.
    refs = sorted({
        *HEART_RAW.glob("training-*/REFERENCE.csv"),
        *HEART_RAW.glob("*/training-*/REFERENCE.csv"),
    })
    for ref in refs:
        with open(ref, newline="") as fh:
            for row in csv.reader(fh):
                if len(row) < 2:
                    continue
                stem = row[0].strip()
                try:
                    raw = int(row[1].strip())
                    labels[stem] = HEART_LABEL_MAP.get(raw, 0)
                except ValueError:
                    pass   # header row or malformed line
    print(f"  Loaded {len(labels):,} heart labels from {len(refs)} REFERENCE.csv files")
    return labels


def step3_preprocess_heart() -> tuple[np.ndarray, np.ndarray] | None:
    _hdr("STEP 3 — Preprocess heart  (Mel-spectrogram | 4 kHz | 5 s | 50 % overlap)")

    HEART_PROC.mkdir(parents=True, exist_ok=True)
    x_path, y_path = HEART_PROC / "X.npy", HEART_PROC / "y.npy"
    if x_path.exists() and y_path.exists():
        X, y = np.load(x_path), np.load(y_path)
        print(f"  ✓ Already preprocessed — X{X.shape}  y{y.shape}")
        return X, y

    labels = _heart_labels()
    # Search both flat (training-a/*.wav) and the nested versioned directory from
    # the combined PhysioNet zip (*/training-a/*.wav).
    wavs = sorted({
        *HEART_RAW.glob("training-*/*.wav"),
        *HEART_RAW.glob("*/training-*/*.wav"),
    })

    if not wavs or not labels:
        print("  ✗ No WAV files or labels found — skipping", file=sys.stderr)
        return None

    win  = HEART_WIN * FS                     # 20 000 samples
    step = int(win * (1.0 - OVERLAP))         # 10 000 samples

    X_buf: list[np.ndarray] = []
    y_buf: list[int] = []
    skipped = 0

    for wav in tqdm(wavs, desc="  [heart] feature extraction", ncols=70):
        stem = wav.stem
        if stem not in labels:
            skipped += 1
            continue
        cls = labels[stem]
        try:
            sig = _load(wav)
        except Exception as exc:
            tqdm.write(f"  [skip] {wav.name}: {exc}")
            skipped += 1
            continue

        for seg in _windows(sig, win, step):
            X_buf.append(_mel(seg))    # (64, ~157)
            y_buf.append(cls)

    if not X_buf:
        print("  ✗ No segments extracted", file=sys.stderr)
        return None

    X = np.stack(X_buf)                       # (N, 64, T)
    y = np.array(y_buf, dtype=np.int32)

    np.save(x_path, X)
    np.save(y_path, y)

    counts = Counter(y_buf)
    print(f"\n  Saved → X{X.shape}  y{y.shape}")
    print(f"  Feature shape per sample: {X.shape[1:]}  "
          f"({X[0].nbytes/1024:.1f} KB each)")
    print(f"  Class distribution:")
    for i, name in enumerate(HEART_CLASS_NAMES):
        n = counts.get(i, 0)
        pct = 100 * n / len(y_buf)
        bar = "█" * int(pct / 2)
        print(f"    {name:12s} {n:5d}  ({pct:5.1f}%)  {bar}")
    if skipped:
        print(f"  [{skipped} files skipped — missing label or load error]")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Preprocess lung
# ─────────────────────────────────────────────────────────────────────────────

def _lung_annotation_roots() -> list[Path]:
    """Return directories that may contain ICBHI .wav/.txt pairs."""
    # Primary path: the pre-extracted Respiratory_Sound_Database folder
    primary = LUNG_RAW / "Respiratory_Sound_Database" / "audio_and_txt_files"
    if primary.exists():
        return [primary]

    # Fallback: try extracted zip directories and the raw root itself
    roots: list[Path] = []
    for pat in ("ICBHI_final_database*/audio_and_txt_files",
                "Respiratory_Sound_Database*/audio_and_txt_files",
                "ICBHI_final_database*",
                "ICBHI_challenge_data*"):
        roots.extend(LUNG_RAW.glob(pat))
    roots.append(LUNG_RAW)
    return roots


def _lung_cycle_labels() -> dict[str, list[tuple[float, float, int]]]:
    """
    Parse per-cycle annotation .txt files.
    Each line: start_time  end_time  crackle  wheeze   (tab-separated)
    Returns {wav_stem: [(t0, t1, class_int), ...]}
    """
    ann: dict[str, list[tuple[float, float, int]]] = {}
    for root in _lung_annotation_roots():
        for txt in root.glob("*.txt"):
            if "diagnosis" in txt.name.lower() or "readme" in txt.name.lower():
                continue
            cycles: list[tuple[float, float, int]] = []
            try:
                with open(txt) as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if len(parts) < 4:
                            continue
                        t0, t1   = float(parts[0]), float(parts[1])
                        crackle  = int(parts[2])
                        wheeze   = int(parts[3])
                        cls      = LUNG_LABEL_MAP.get((crackle, wheeze), 0)
                        cycles.append((t0, t1, cls))
            except (ValueError, OSError):
                continue
            if cycles:
                ann[txt.stem] = cycles
    return ann


def step4_preprocess_lung() -> tuple[np.ndarray, np.ndarray] | None:
    _hdr("STEP 4 — Preprocess lung  (MFCC+Mel 2-ch | 4 kHz | 10 s | 50 % overlap)")

    LUNG_PROC.mkdir(parents=True, exist_ok=True)
    x_path, y_path = LUNG_PROC / "X.npy", LUNG_PROC / "y.npy"
    if x_path.exists() and y_path.exists():
        X, y = np.load(x_path), np.load(y_path)
        print(f"  ✓ Already preprocessed — X{X.shape}  y{y.shape}")
        return X, y

    # Locate audio files
    wavs: list[Path] = []
    for root in _lung_annotation_roots():
        wavs.extend(root.glob("*.wav"))
    wavs = sorted(set(wavs))

    if not wavs:
        print("  ✗ No lung WAV files found — skipping", file=sys.stderr)
        return None

    # Patient-level diagnosis file (reference only — actual labels come from cycle .txt files)
    diag_candidates = [
        LUNG_RAW / "Respiratory_Sound_Database" / "patient_diagnosis.csv",
        LUNG_RAW / "Respiratory_Sound_Database" / "ICBHI_Challenge_diagnosis.txt",
    ]
    for diag in diag_candidates:
        if diag.exists():
            print(f"  Found {diag.name} — reference only; "
                  f"cycle-level labels come from per-file annotation .txt files")
            break

    ann     = _lung_cycle_labels()
    ann_hit = sum(1 for w in wavs if w.stem in ann)
    print(f"  {len(wavs)} WAV files found, {ann_hit} have cycle annotations")

    win  = LUNG_WIN * FS                       # 40 000 samples
    step = int(win * (1.0 - OVERLAP))          # 20 000 samples

    X_buf: list[np.ndarray] = []
    y_buf: list[int] = []
    skipped = 0

    for wav in tqdm(wavs, desc="  [lung] feature extraction", ncols=70):
        try:
            sig = _load(wav)
        except Exception as exc:
            tqdm.write(f"  [skip] {wav.name}: {exc}")
            skipped += 1
            continue

        stem = wav.stem

        if stem in ann:
            # Per-cycle extraction: each annotated breath cycle is one sample
            for t0, t1, cls in ann[stem]:
                s0 = int(t0 * FS)
                s1 = int(t1 * FS)
                seg = sig[s0:s1]

                if len(seg) < FS // 4:         # skip cycles shorter than 0.25 s
                    continue

                # Pad short cycles / trim long ones to exactly LUNG_WIN seconds
                if len(seg) < win:
                    seg = np.pad(seg, (0, win - len(seg)))
                else:
                    seg = seg[:win]

                X_buf.append(_lung_feat(seg))  # (104, ~313)
                y_buf.append(cls)
        else:
            # No annotation: fixed windows, label = normal (0)
            for seg in _windows(sig, win, step):
                X_buf.append(_lung_feat(seg))
                y_buf.append(0)

    if not X_buf:
        print("  ✗ No segments extracted", file=sys.stderr)
        return None

    X = np.stack(X_buf)
    y = np.array(y_buf, dtype=np.int32)

    np.save(x_path, X)
    np.save(y_path, y)

    counts = Counter(y_buf)
    print(f"\n  Saved → X{X.shape}  y{y.shape}")
    print(f"  Feature shape per sample: {X.shape[1:]}  "
          f"({X[0].nbytes/1024:.1f} KB each)")
    print(f"  MFCC rows : 0–{N_MFCC-1}   Mel rows : {N_MFCC}–{N_MFCC+N_MELS-1}")
    print(f"  Class distribution:")
    for i, name in enumerate(LUNG_CLASS_NAMES):
        n = counts.get(i, 0)
        pct = 100 * n / len(y_buf) if y_buf else 0
        bar = "█" * int(pct / 2)
        print(f"    {name:12s} {n:5d}  ({pct:5.1f}%)  {bar}")
    if skipped:
        print(f"  [{skipped} files skipped — load errors]")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Summary
# ─────────────────────────────────────────────────────────────────────────────

def step5_summary(
    heart: tuple[np.ndarray, np.ndarray] | None,
    lung:  tuple[np.ndarray, np.ndarray] | None,
) -> None:
    _hdr("STEP 5 — Final Summary")

    def _report(name: str, result, class_names: list[str]) -> None:
        if result is None:
            print(f"  {name}: NOT AVAILABLE")
            return
        X, y = result
        counts = Counter(y.tolist())
        n = len(y)
        disk_mb = (X.nbytes + y.nbytes) / 1_048_576
        print(f"\n  {name} dataset")
        print(f"    X shape  : {X.shape}")
        print(f"    y shape  : {y.shape}")
        print(f"    Disk     : {disk_mb:.1f} MB")
        for i, cname in enumerate(class_names):
            cnt = counts.get(i, 0)
            print(f"    {cname:12s}: {cnt:5d}  ({100*cnt/n:.1f}%)")
        if n > 10_000:
            est_min = n * 0.5 / 60       # rough: ~0.5 s per sample on CPU
            print(f"  ⚠  {n:,} samples — estimated CPU training time: "
                  f"~{est_min:.0f} min. Consider a GPU or smaller batch.")

    _report("Heart", heart, HEART_CLASS_NAMES)
    _report("Lung",  lung,  LUNG_CLASS_NAMES)

    print()
    heart_ok = heart is not None
    lung_ok  = lung  is not None
    if heart_ok and lung_ok:
        print("  ✓ Both datasets ready.")
    elif heart_ok:
        print("  ✓ Heart ready. Lung unavailable (check download).")
    elif lung_ok:
        print("  ✓ Lung ready. Heart unavailable (check download).")
    else:
        print("  ✗ No datasets available. Check network and re-run.")
        print("    PhysioNet may require free account credentials.")
        print("    ICBHI site requires registration: https://bhichallenge.med.auth.gr")

    print("\n  Next steps:")
    if heart_ok:
        print("    python ml/heart/train.py")
    if lung_ok:
        print("    python ml/lung/train.py")
    print()


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    global _PN_USER, _PN_PASS

    p = argparse.ArgumentParser(description="Download and preprocess cardio datasets")
    p.add_argument("--physionet-user",     default=_PN_USER,
                   help="PhysioNet username (or set PHYSIONET_USER env var)")
    p.add_argument("--physionet-password", default=_PN_PASS,
                   help="PhysioNet password (or set PHYSIONET_PASSWORD env var)")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip downloads; preprocess whatever is already in raw/")
    args = p.parse_args()

    _PN_USER = args.physionet_user
    _PN_PASS = args.physionet_password

    print("cardio-diagnostics dataset setup")
    print(f"Output root : {ROOT}")
    print(f"Target SR   : {FS} Hz  (matches Arduino firmware)")
    print(f"Heart win   : {HEART_WIN} s  →  {HEART_WIN * FS:,} samples")
    print(f"Lung win    : {LUNG_WIN} s  →  {LUNG_WIN * FS:,} samples")
    if _PN_USER:
        print(f"PhysioNet   : {_PN_USER}  (credentials set)")
    else:
        print("PhysioNet   : no credentials — unauthenticated download attempt")

    if not args.skip_download:
        step1_download_heart()
        step2_download_lung()

    heart = step3_preprocess_heart()
    lung  = step4_preprocess_lung()
    step5_summary(heart, lung)


if __name__ == "__main__":
    main()
