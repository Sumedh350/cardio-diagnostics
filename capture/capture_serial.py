"""Binary serial capture for cardio-diagnostics v2.0 firmware.

Frame formats
─────────────
  Burst : [0xAA][0xBB][mode][xor_cs][128 bytes: 64 × int16_t LE]  (132 bytes)
  SYNC  : [0xFF][0xFF][mode][0xFF]                                   (4 bytes)

  mode    : 0x01 = HEART, 0x02 = LUNG
  xor_cs  : XOR of all 128 data bytes in the burst

Bandwidth budget (verified)
────────────────────────────
  Sample rate    4 000 Hz
  Bytes/sample   2  (int16_t LE, signed, DC-removed)
  Burst size     64 × 2 = 128 data + 4 header = 132 bytes
  Bursts/sec     4 000 / 64 = 62.5
  Burst traffic  62.5 × 132 = 8 250 bytes/sec
  SYNC overhead  1 × 4      =     4 bytes/sec
  Total                     = 8 254 bytes/sec
  115 200 baud → 11 520 bytes/sec capacity
  Headroom ≈ 28 %  ✓

Usage
─────
  python capture_serial.py --port COM3 --mode heart
  python capture_serial.py --port COM3 --mode lung --duration 10 --output data/lung_001.wav
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import serial

# ── Protocol constants ────────────────────────────────────────────────────────
SAMPLE_RATE     = 4_000
BUF_SIZE        = 64
BURST_DATA_LEN  = BUF_SIZE * 2                  # 128 bytes

MODE_NAME    = {0x01: "HEART", 0x02: "LUNG"}
RECORD_SECS  = {0x01: 5,       0x02: 10}        # defaults overridable via --duration

# Firmware DC-removed values span ≈ ±512 (10-bit ADC, midpoint 512)
NORM_SCALE = 512.0

BAR_WIDTH = 40


# ─────────────────────────────────────────────────────────────────────────────
def _read_exact(ser: serial.Serial, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise EOFError("serial port closed unexpectedly")
        buf += chunk
    return buf


def _rms_bar(samples: np.ndarray) -> str:
    """One-line ASCII amplitude bar based on per-burst RMS."""
    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) / NORM_SCALE
    rms = min(rms, 1.0)
    filled = int(rms * BAR_WIDTH)
    return f"|{'█' * filled}{' ' * (BAR_WIDTH - filled)}| {rms:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
class FrameParser:
    """Byte-level state machine for the binary frame stream.

    Handles:
    * ASCII boot banner  — skipped; bytes that don't start a valid frame are
                           dropped silently, so no pre-scan is needed
    * SYNC frames        — 0xFF 0xFF [mode] 0xFF
    * Burst frames       — 0xAA 0xBB [mode] [xor_cs] [128 bytes]
    """

    def __init__(self, ser: serial.Serial) -> None:
        self._ser = ser

    # ── Public API ────────────────────────────────────────────────────────────

    def wait_for_sync(self) -> int:
        """Scan the byte stream until a valid SYNC frame appears.

        Discards the ASCII banner and any partial bytes transparently.
        Returns the mode byte (0x01 or 0x02).
        """
        prev = 0x00
        while True:
            b = self._ser.read(1)
            if not b:
                continue
            byte = b[0]
            if prev == 0xFF and byte == 0xFF:
                tail = _read_exact(self._ser, 2)  # [mode, 0xFF]
                if tail[1] == 0xFF and tail[0] in MODE_NAME:
                    return tail[0]
            prev = byte

    def read_next(self) -> tuple[str, int | None, np.ndarray | None]:
        """Read one frame.

        Returns:
          ('sync',  mode_byte, None)            — SYNC frame
          ('burst', mode_byte, samples_int16)   — verified burst
          ('skip',  None,      None)            — unrecognised byte; caller loops
        """
        b = self._ser.read(1)
        if not b:
            return "skip", None, None

        byte = b[0]
        if byte == 0xFF:
            return self._parse_sync_tail()
        if byte == 0xAA:
            return self._parse_burst_tail()
        return "skip", None, None

    # ── Private ───────────────────────────────────────────────────────────────

    def _parse_sync_tail(self) -> tuple[str, int | None, np.ndarray | None]:
        """Consume 0xFF [mode] 0xFF after the leading 0xFF was already read."""
        tail = _read_exact(self._ser, 3)
        if tail[0] == 0xFF and tail[2] == 0xFF and tail[1] in MODE_NAME:
            return "sync", tail[1], None
        return "skip", None, None

    def _parse_burst_tail(self) -> tuple[str, int | None, np.ndarray | None]:
        """Consume 0xBB [mode] [xor_cs] [128 bytes] after 0xAA was already read."""
        header = _read_exact(self._ser, 3)       # 0xBB, mode, xor_cs
        if header[0] != 0xBB:
            return "skip", None, None

        mode_byte    = header[1]
        expected_xor = header[2]
        data         = _read_exact(self._ser, BURST_DATA_LEN)

        actual_xor = 0
        for db in data:
            actual_xor ^= db

        if actual_xor != expected_xor:
            print(
                f"\n[WARN] checksum mismatch: got {actual_xor:#04x}, "
                f"expected {expected_xor:#04x} — burst discarded",
                file=sys.stderr,
            )
            return "skip", None, None

        # int16_t little-endian, matches firmware's signed DC-removed values
        samples = np.frombuffer(data, dtype="<i2").copy()
        return "burst", mode_byte, samples


# ─────────────────────────────────────────────────────────────────────────────
def capture(
    port: str,
    baud: int,
    output: Path,
    mode_hint: str | None = None,
    duration_override: int | None = None,
) -> None:
    print(f"Opening {port} @ {baud} baud …")

    with serial.Serial(port, baud, timeout=2.0) as ser:
        parser = FrameParser(ser)

        print("Scanning for SYNC frame (boot banner skipped automatically) …")
        mode_byte = parser.wait_for_sync()
        mode_name = MODE_NAME[mode_byte]

        if mode_hint and mode_name.lower() != mode_hint.lower():
            print(
                f"[WARN] expected {mode_hint.upper()} but firmware reports {mode_name}."
                f" Check D2 wiring (LOW=HEART, open=LUNG).",
                file=sys.stderr,
            )

        target_s = duration_override if duration_override is not None else RECORD_SECS[mode_byte]
        n_needed = target_s * SAMPLE_RATE          # heart 5s→20000, lung 10s→40000

        print(f"SYNC:{mode_name} locked — recording {target_s} s ({n_needed} samples)")

        chunks: list[np.ndarray] = []
        n_got = 0

        while n_got < n_needed:
            kind, frame_mode, samples = parser.read_next()

            if kind != "burst" or samples is None:
                continue
            if frame_mode != mode_byte:
                continue                         # mode pin changed mid-capture; skip

            chunks.append(samples)
            n_got += len(samples)

            pct   = n_got / n_needed
            label = f"[{mode_name}] {n_got:>6}/{n_needed}  {pct:5.1%}"
            print(f"\r{label}  {_rms_bar(samples)}", end="", flush=True)

    print()  # newline after progress line

    raw_int16  = np.concatenate(chunks)[:n_needed]
    floats     = raw_int16.astype(np.float32)
    floats    -= floats.mean()                         # remove any residual DC
    peak       = np.abs(floats).max()
    if peak > 0:
        floats /= peak                                 # normalize to full ±1.0 range

    # WAV: 4000 Hz mono 16-bit PCM
    pcm16 = (floats * 32767.0).clip(-32768, 32767).astype(np.int16)

    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())

    dur = len(pcm16) / SAMPLE_RATE
    print(f"Saved {len(pcm16)} samples ({dur:.2f} s) → {output}")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="Binary capture from cardio-diagnostics v2.0 firmware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--port",     required=True,
                   help="Serial port (COM3 / /dev/ttyUSB0)")
    p.add_argument("--baud",     type=int, default=115200)
    p.add_argument("--output",   type=Path, default=Path("data/capture.wav"),
                   help="Output WAV file (default: data/capture.wav)")
    p.add_argument("--mode",     choices=["heart", "lung"], default=None,
                   help="Expected mode — warns if D2 pin disagrees (HEART=LOW, LUNG=open)")
    p.add_argument("--duration", type=int, default=None,
                   help="Recording duration in seconds (default: 5 for heart, 10 for lung)")
    args = p.parse_args()
    capture(args.port, args.baud, args.output, args.mode, args.duration)


if __name__ == "__main__":
    main()
