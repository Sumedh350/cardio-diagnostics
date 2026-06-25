#!/usr/bin/env python3
"""
record.py — Capture audio from cardio-diagnostics Arduino firmware and save as WAV.

Firmware frame format (132 bytes):
  [0xAA][0xBB][mode][xor_cs][64 × int16_t LE]
  mode: 0x01=HEART (D2 LOW), 0x02=LUNG (D2 open/HIGH)

SYNC frame (4 bytes, once per second):
  [0xFF][0xFF][mode][0xFF]

Usage:
  python record.py                        # auto-detect mode, default duration
  python record.py --port COM3            # explicit port
  python record.py --duration 5 --out heart.wav
  python record.py --duration 10 --out lung.wav
"""
import argparse
import struct
import sys
import time
import wave

BAUD        = 115200
FRAME_HDR   = bytes([0xAA, 0xBB])
SYNC_HDR    = bytes([0xFF, 0xFF])
SAMPLES_PER_BURST = 64
SAMPLE_RATE = 4000
MODE_HEART  = 0x01
MODE_LUNG   = 0x02

# Scale ADC int16 (-512..512) to full int16 range for better WAV amplitude
SCALE = 64


def _read_exact(ser, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise TimeoutError("Serial read timeout")
        buf += chunk
    return buf


def _sync_to_frame(ser) -> None:
    """Scan byte-by-byte until we see the start of a data frame header 0xAA 0xBB."""
    print("Syncing to first data frame …", end="", flush=True)
    buf = b""
    while True:
        byte = ser.read(1)
        if not byte:
            continue
        buf = (buf + byte)[-2:]
        if buf == FRAME_HDR:
            print(" OK")
            return


def record(port: str, duration: float, out_path: str) -> None:
    import serial

    ser = serial.Serial(port, BAUD, timeout=2)
    time.sleep(1.5)   # let Arduino reset after DTR toggle
    ser.reset_input_buffer()

    _sync_to_frame(ser)

    samples: list[int] = []
    target  = int(duration * SAMPLE_RATE)
    mode    = None
    t_start = time.time()

    print(f"Recording {duration}s → {out_path}")
    print("0%", end="", flush=True)

    while len(samples) < target:
        # Read remaining 2 bytes of header (mode + xor_cs)
        hdr_rest = _read_exact(ser, 2)
        frame_mode, xor_cs = hdr_rest[0], hdr_rest[1]

        if mode is None:
            mode = frame_mode
            label = "HEART" if mode == MODE_HEART else "LUNG"
            print(f"  [{label} mode detected]", end="", flush=True)

        # Read 128 bytes of sample data (64 × int16)
        data = _read_exact(ser, SAMPLES_PER_BURST * 2)

        # Verify XOR checksum
        cs = 0
        for b in data:
            cs ^= b
        if cs != xor_cs:
            # Checksum mismatch — skip burst and re-sync
            _sync_to_frame(ser)
            continue

        # Unpack int16 LE samples
        burst = struct.unpack_from(f"<{SAMPLES_PER_BURST}h", data)
        samples.extend(burst)

        # Progress bar
        pct = int(len(samples) / target * 100)
        sys.stdout.write(f"\r{pct:3d}%  ({len(samples)}/{target} samples)  "
                         f"[{int(pct/5)*'█':{20}}]")
        sys.stdout.flush()

        # Re-sync: after each burst the loop must find the next 0xAA 0xBB header.
        # Drain any SYNC frames (0xFF 0xFF ...) between bursts, then re-sync.
        hdr = _read_exact(ser, 2)
        while hdr == SYNC_HDR:
            _read_exact(ser, 2)          # consume remaining 2 bytes of SYNC frame
            hdr = _read_exact(ser, 2)
        if hdr != FRAME_HDR:
            _sync_to_frame(ser)

    elapsed = time.time() - t_start
    ser.close()

    # Trim to exact length and scale to full int16 range
    samples = samples[:target]
    scaled  = [max(-32768, min(32767, s * SCALE)) for s in samples]

    # Write WAV
    with wave.open(out_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{len(scaled)}h", *scaled))

    label = "HEART" if mode == MODE_HEART else "LUNG"
    print(f"\nDone — {label} — {len(samples)/SAMPLE_RATE:.1f}s saved to {out_path}  "
          f"(recorded in {elapsed:.1f}s)")
    if mode == MODE_HEART and duration > 6:
        print("⚠  Duration >6 s — web app will route this to the LUNG model. "
              "Use ≤6 s for heart classification.")
    if mode == MODE_LUNG and duration <= 6:
        print("⚠  Duration ≤6 s — web app will route this to the HEART model. "
              "Use >6 s for lung classification.")


def main():
    parser = argparse.ArgumentParser(description="Record WAV from cardio-diagnostics Arduino")
    parser.add_argument("--port",     default="COM3",      help="Serial port (default: COM3)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Recording duration in seconds (default: 5 for heart, 10 for lung)")
    parser.add_argument("--out",      default=None,        help="Output WAV filename")
    parser.add_argument("--mode",     choices=["heart","lung"], default=None,
                        help="Force mode (overrides D2 pin detection for filename/duration defaults)")
    args = parser.parse_args()

    # If mode given explicitly, use it for defaults
    forced = args.mode
    if forced == "heart":
        duration = args.duration or 5
        out      = args.out      or "heart.wav"
    elif forced == "lung":
        duration = args.duration or 10
        out      = args.out      or "lung.wav"
    else:
        # Will auto-detect from first frame; pick conservative defaults
        duration = args.duration or 10
        out      = args.out      or "recording.wav"

    try:
        record(args.port, duration, out)
    except KeyboardInterrupt:
        print("\nRecording cancelled.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
