# Wake word daemon (POC)

Local, always-on **wake-word listener** for Linux desktops using **PyAudio** (works with **PipeWire** via the PulseAudio compatibility layer) and **[openWakeWord](https://github.com/dscripka/openWakeWord)**. Inference uses **ONNX Runtime on the CPU** only so a discrete GPU can stay idle (better battery and thermals on laptops).

This is a minimal Step-1 proof of concept: stable capture loop, ring-style buffering, threshold + cooldown, and a visible trigger (`notify-send` or a system bell).

## System dependencies (Ubuntu 24.04)

| Package | Why |
|--------|-----|
| `portaudio19-dev` | Headers and libs to **build PyAudio** from source |
| `python3-dev` | Python development headers for native extensions |
| `build-essential` | Compiler toolchain (`gcc`, etc.) |
| `libnotify-bin` | **`notify-send`** (default action on detection) |

Optional (only if you use `--beep`):

- `paplay` is usually provided by **PipeWire** or **PulseAudio** (`pipewire-pulse` / `pulseaudio-utils`).

Install:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev build-essential libnotify-bin
```

Ensure a microphone is available and the default input is correct (e.g. **Settings → Sound** or `pavucontrol` under PipeWire).

## Python environment

Use a virtual environment inside this directory.

```bash
cd /path/to/openclaw/overlay/wake_daemon

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "openwakeword>=0.6.0" --no-deps
```

### Why `openwakeword` is installed twice

On **Python 3.12**, the `openwakeword` package still lists **`tflite-runtime`** for Linux, which has **no compatible wheel**, so a plain `pip install openwakeword` can fail. This project uses **ONNX only**, so install dependencies from `requirements.txt`, then install openWakeWord with **`--no-deps`**.

If you use **Python 3.11**, you can try a single command instead:

```bash
pip install -r requirements.txt
pip install "openwakeword>=0.6.0"
```

(If that still conflicts, keep the `--no-deps` approach.)

### Keep inference on CPU

- Install **`onnxruntime`** from PyPI (already in `requirements.txt`).
- Do **not** install **`onnxruntime-gpu`**.

Quick check inside the venv:

```bash
python -c "import onnxruntime as ort; print('device:', ort.get_device())"
```

You should see **`CPU`**.

## Usage

```bash
source venv/bin/activate
python main.py
```

Default wake phrase is **“hey jarvis”**. On first run, models are downloaded into openWakeWord’s package `resources/` (embedding, melspectrogram, and the chosen wake model).

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--wake` | `hey jarvis` | Pretrained phrase, e.g. `alexa`, `hey mycroft`, `hey rhasspy` |
| `--threshold` | `0.5` | Score above this counts as a detection |
| `--cooldown` | `2.0` | Seconds to suppress repeat triggers |
| `--beep` | off | Use `paplay` on a freedesktop bell instead of `notify-send` |

Examples:

```bash
python main.py --wake "alexa" --threshold 0.55 --cooldown 3
python main.py --beep
```

Stop with **Ctrl+C** (SIGINT) or send **SIGTERM**.

## Troubleshooting

- **`portaudio.h: No such file`** when installing PyAudio: install `portaudio19-dev`, then `pip install -r requirements.txt` again.
- **No desktop notification**: install `libnotify-bin`; on SSH or headless sessions, `notify-send` may not show anything (use `--beep` or point the subprocess in `main.py` at your own command).
- **No sound for `--beep`**: install PipeWire/Pulse tools or change the `paplay` path to a `.wav` you have locally.
- **False positives / misses**: adjust `--threshold` and `--cooldown`; reduce background noise; confirm sample rate is effectively 16 kHz mono from the default input.

## Files

| File | Role |
|------|------|
| `main.py` | Capture loop, deque buffer, openWakeWord ONNX inference, trigger + cooldown |
| `requirements.txt` | Python dependencies (plus comment for the `openwakeword --no-deps` step) |
