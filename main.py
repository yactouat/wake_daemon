#!/usr/bin/env python3
"""
Step-1 POC: continuous wake-word listener using PyAudio (PipeWire/PulseAudio) + openWakeWord (ONNX, CPU-only).

Audio capture runs in a PortAudio callback; PCM chunks are queued in a deque. The main thread
assembles 80 ms frames (1280 samples @ 16 kHz mono int16) and runs openWakeWord inference on CPU.
"""

from __future__ import annotations

import argparse
import collections
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Deque

import numpy as np
import pyaudio

from openwakeword.model import Model
from openwakeword.utils import download_models

# ---------------------------------------------------------------------------
# Tunables (POC defaults)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000  # openWakeWord expects 16 kHz PCM
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # int16
BYTES_PER_FRAME = CHANNELS * SAMPLE_WIDTH_BYTES

# One inference frame = 1280 samples = 80 ms at 16 kHz (recommended by openWakeWord)
INFERENCE_SAMPLES = 1280
INFERENCE_BYTES = INFERENCE_SAMPLES * BYTES_PER_FRAME

# Smaller PortAudio buffer => callback fires more often; main thread still consumes in 1280-sample steps
FRAMES_PER_BUFFER = 512

DEFAULT_WAKE_PHRASE = "yo Chrome"  # also try "alexa", "hey mycroft", etc. (see openwakeword.MODELS)
DEFAULT_THRESHOLD = 0.5
DEFAULT_COOLDOWN_S = 2.0


def _ensure_models(wake_phrase: str) -> None:
    """
    Download embedding/melspectrogram ONNX assets plus the requested wake model (ONNX + tflite URLs).
    Filenames use underscores (e.g. hey_jarvis_v0.1); download_models matches substrings.
    """
    key = wake_phrase.strip().lower().replace(" ", "_")
    # Map common phrase to release asset basename fragment
    name_map = {
        "hey_jarvis": "hey_jarvis_v0.1",
        "alexa": "alexa_v0.1",
        "hey_mycroft": "hey_mycroft_v0.1",
        "hey_rhasspy": "hey_rhasspy_v0.1",
    }
    fragment = name_map.get(key, f"{key}_v0.1")
    download_models(model_names=[fragment])


def _build_model(wake_phrase: str) -> Model:
    """
    ONNX + explicit CPU providers only (no CUDA) so the NVIDIA GPU stays idle.
    `device='cpu'` is forwarded to AudioFeatures inside openWakeWord.
    """
    return Model(
        wakeword_models=["/home/yactouat/.openclaw/overlay/wake_daemon/yo_chrome.onnx"],
        inference_framework="onnx",
        device="cpu",
        ncpu=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only openWakeWord listener (PyAudio + ONNX).")
    parser.add_argument(
        "--wake",
        default=DEFAULT_WAKE_PHRASE,
        help='Wake phrase key, e.g. "hey jarvis" or "alexa" (passed to openWakeWord pretrained lookup)',
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Detection threshold 0..1")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_S, help="Seconds to ignore after a hit")
    parser.add_argument(
        "--beep",
        action="store_true",
        help="Play a desktop bell via paplay instead of notify-send",
    )
    args = parser.parse_args()

    # Extra guard: hide CUDA devices from any loaded library that might probe for GPUs.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    print("Downloading / verifying openWakeWord models (first run may take a minute)...", flush=True)
    _ensure_models(args.wake)

    print("Loading ONNX models on CPU...", flush=True)
    model = _build_model(args.wake)

    # Ring / transfer buffer: callback thread appends raw PCM bytes; main thread consumes.
    chunk_queue: Deque[bytes] = collections.deque()
    queue_lock = threading.Lock()

    audio = pyaudio.PyAudio()

    def audio_callback(in_data, _frame_count, _time_info, status):
        # PortAudio invokes this from a high-priority thread; keep work minimal (enqueue only).
        if status:
            # Non-zero status (e.g. input overflow): log in a real daemon; here we still keep the stream alive.
            pass
        with queue_lock:
            chunk_queue.append(in_data)
        return (None, pyaudio.paContinue)

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
        stream_callback=audio_callback,
    )

    pcm_accumulator = bytearray()
    running = True
    cooldown_until = 0.0

    def stop_handler(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    stream.start_stream()
    print(
        f"Listening for {args.wake!r} (threshold={args.threshold}, cooldown={args.cooldown}s). Ctrl+C to quit.",
        flush=True,
    )

    try:
        while running and stream.is_active():
            with queue_lock:
                while chunk_queue:
                    pcm_accumulator.extend(chunk_queue.popleft())

            if len(pcm_accumulator) < INFERENCE_BYTES:
                # Avoid busy-waiting while the callback fills the deque.
                time.sleep(0.005)
                continue

            # Take exactly one 80 ms frame from the front of the byte buffer (FIFO over the queued chunks).
            frame_bytes = pcm_accumulator[:INFERENCE_BYTES]
            del pcm_accumulator[:INFERENCE_BYTES]

            # openWakeWord expects int16 mono @ 16 kHz (shape (1280,) for one frame).
            audio_i16 = np.frombuffer(frame_bytes, dtype=np.int16)

            now = time.monotonic()
            if now < cooldown_until:
                # Keep calling predict so mel/embedding preprocessor state stays consistent with real time.
                _ = model.predict(audio_i16)
                continue

            predictions = model.predict(audio_i16)
            # With one wakeword model this dict has a single scalar; multi-class models expose several keys.
            score = max(float(v) for v in predictions.values()) if predictions else 0.0

            if score > args.threshold:
                cooldown_until = now + args.cooldown
                # Highly visible terminal line (ANSI reverse video)
                print(
                    "\n\033[7m[WAKE WORD DETECTED]\033[0m "
                    f"score={score:.3f} phrase={args.wake!r}\n",
                    flush=True,
                )
                if args.beep:
                    subprocess.Popen(
                        [
                            "paplay",
                            "/usr/share/sounds/freedesktop/stereo/bell.oga",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    subprocess.Popen(
                        ["notify-send", "Wake word", f"Detected ({args.wake!r}, score={score:.2f})"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    return 0


if __name__ == "__main__":
    sys.exit(main())
