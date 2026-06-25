# Cardio Diagnostics — Project Status & Handoff Document
Generated: 2026-06-24

## Project Location
Full path: C:\Users\sumka\cardio-diagnostics

## Hardware Setup (COMPLETE)
- Arduino Uno on COM3
- MAX4466 mic amp: VCC→3.3V, OUT→A0, GND→GND
- BPL stethoscope connected to MAX4466
- D2 jumper to GND = HEART mode, D2 disconnected = LUNG mode
- Firmware: firmware/src/main.cpp — Timer1 4000Hz, binary framing, 28% bandwidth headroom
- Capture: capture/capture_serial.py — binary frame parser, saves 16-bit mono WAV

## Pipeline (COMPLETE)
- Arduino → binary serial frames → Python → WAV → ML model → Flask web app
- Heart: 5 second recordings at 4000Hz
- Lung: 10 second recordings at 4000Hz

## Datasets (COMPLETE)
- Heart: PhysioNet CinC 2016 — ml/heart/raw/ — 24,450 segments — ml/heart/processed/X.npy + y.npy
- Lung: ICBHI 2017 — ml/lung/raw/ — 6,895 segments — ml/lung/processed/X.npy + y.npy

## Models
### Heart Model — COMPLETE ✓
- File: models/heart_model.keras (61 MB)
- Architecture: Conv2D(32,64,128) → GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → sigmoid
- Val accuracy: 95.99% — Test accuracy: 95.50%
- Binary: normal vs abnormal
- Target was 85% — EXCEEDED

### Lung Model — IN PROGRESS ✗
- Current best: 78% val accuracy (attempt 1, saved as models/lung_model_binary.keras)
- Target: 82% — NOT MET YET
- Problem: original 4-class model collapsed to majority class
- Fix applied: GlobalAveragePooling2D replacing Flatten
- Next step: retrain as BINARY (normal=0, crackle+wheeze+both=1) with reduced input (64×64) for speed
- Exact retraining prompt to paste:

NEXT LUNG TRAINING PROMPT:
"""
Cancel all training. Save whatever is in models/ already.

Retrain lung model with these speed optimizations:
- Reload ml/lung/processed/X.npy and y.npy
- Remap to binary: normal=0, crackle+wheeze+both=1
- Resize spectrograms from (104, 313) to (64, 64) using skimage.transform.resize
- Architecture:
  Conv2D(16, 3x3, relu) → MaxPool(2x2)
  Conv2D(32, 3x3, relu) → MaxPool(2x2)
  GlobalAveragePooling2D
  Dense(64, relu) → Dropout(0.3)
  Dense(1, sigmoid)
- Adam lr=0.001, batch=64, epochs=20, EarlyStopping patience=3
- Class weights balanced
- Save to models/lung_model_binary.keras
Run immediately: python ml/lung/train_binary.py
"""

## Web App — NOT STARTED
- Location: web/app.py and web/templates/index.html
- Flask app with drag and drop WAV upload
- /predict endpoint returns label + confidence score
- Auto-detects heart vs lung from file duration (≤6s = heart, >6s = lung)

## After Lung Training — Web App Prompt
Once lung model hits target, paste this into Claude Code:
"""
Both models are ready:
- models/heart_model.keras — binary heart classifier 95.5% accuracy
- models/lung_model_binary.keras — binary lung classifier

Build the complete Flask web app in web/app.py:
- Load both models on startup
- Route GET / → serves templates/index.html
- Route POST /classify → accepts WAV file upload
  - Auto-detect heart vs lung from duration: <=6s = heart, >6s = lung
  - Preprocess WAV identically to training:
    Heart: resample to 4000Hz, 5s window, log-Mel (64,157)
    Lung: resample to 4000Hz, resize to (64,64)
  - Return JSON: {type, label, confidence, all_scores}
- templates/index.html: single file, no external dependencies
  - Large drag and drop zone for WAV files
  - Waveform preview using Web Audio API
  - Result card: label in large badge, confidence bar, all class scores
  - Medical style: white background, blue accents
Run with: python web/app.py
"""

## Requirements
pip install: pyserial numpy scipy librosa scikit-learn tensorflow keras flask tqdm scikit-image

## CLAUDE.md Rules
- Always read serial monitor before modifying firmware
- Check ml/logs/ before changing hyperparameters
- Heart target: >=85% val accuracy
- Lung target: >=82% val accuracy
- Log every experiment to ml/logs/experiments.md
- Submission deadline: tomorrow

## Current Priority Order
1. Restart laptop (overheating/slow)
2. Open new Claude Code session in cardio-diagnostics/ folder
3. Read this PROJECT_STATUS.md file
4. Run lung binary retraining prompt above
5. Build web app
6. Test end to end
