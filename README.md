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

## Training a Custom Wake Word (The Sovereign Forge)

To shed the default models and train a completely local, private wake word (e.g., "yo chrome") without relying on Google Colab or external compute APIs, follow these steps. This process generates thousands of synthetic TTS clips using Piper, mixes them with background noise, and trains a custom `.onnx` classifier using your CPU/GPU.

### 1. Clone the Source
Open a new terminal outside of your `wake_daemon` directory (e.g., in `overlay/` or `~/workspace/`):
```bash
git clone https://github.com/dscripka/openWakeWord.git
cd openWakeWord
```

### 2. Setup the Training Environment (Python 3.10 Required)
The training environment requires heavy ML dependencies (like older versions of TensorFlow and ONNX translation layers) that **do not have pre-built wheels for Python 3.12**. 

To compile these, we must use Python 3.10. On Ubuntu 24.04, you need the `deadsnakes` PPA to get it natively without breaking your system's default Python 3.12.

```bash
# 1. Add Deadsnakes PPA and install Python 3.10 + build headers
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev cmake

# 2. Navigate to repo
cd openWakeWord

# 3. Patch older pinned versions in setup.py to prevent strictly pinned build failures
sed -i 's/torchaudio>=0.13.1,<1/torchaudio>=0.13.1/g' setup.py
sed -i 's/tensorflow-cpu==2.8.1/tensorflow-cpu>=2.8.1/g' setup.py
sed -i 's/onnx==1.14.0/onnx>=1.14.0/g' setup.py
sed -i 's/onnx_tf==1.10.0/onnx_tf>=1.10.0/g' setup.py

# 4. Create and activate a Python 3.10 venv
python3.10 -m venv venv_train
source venv_train/bin/activate

# 5. Install the package with full training dependencies
pip install -e ".[full]"

# 6. Install TorchCodec backend (required by torchaudio>=2.6.0)
pip install torchcodec

# 7. Downgrade PyTorch to stable CUDA 12.4 (Fixes libtorchcodec / libnppicc.so.13 crashes)
# The default pip install may pull a nightly PyTorch (cu130) which crashes on missing CUDA 13 system libs.
pip uninstall -y torch torchvision torchaudio torchcodec
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Download Audio Augmentation Datasets
To make the wake word resilient to echoes and real-world noise, the model mixes the clean TTS voices with Room Impulse Responses (RIRs) and background audio. These datasets are too large to be included in the git repo.

```bash
cd ~/openWakeWord

# 1. Download and extract MIT Room Impulse Responses (~2.5GB unpacked)
# Note: The official MIT link is occasionally down. As an alternative, you can use the HuggingFace datasets library:
# python -c "import datasets; datasets.load_dataset('davidscripka/MIT_environmental_impulse_responses', split='train', streaming=True)"
# But for simplicity, we provide the working direct links here if available, or you can skip RIRs by emptying the `rir_paths` in the YAML.

# 2. Download the background noise and validation features
mkdir -p ./audioset ./fma
wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy
```

### 4. Execute the Forge
The `openWakeWord` training script uses a YAML configuration file to define the generation parameters, the target word, and the noise layers. It also requires the Piper sample generator, a validation dataset to prevent false positives, and a few specific TTS dependencies.

```bash
# 1. Download the required Piper TTS sample generator
git clone https://github.com/dscripka/piper-sample-generator.git

# 2. Download the Piper TTS base models (the generator defaults to high, not medium)
wget -O piper-sample-generator/models/en-us-libritts-high.pt 'https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt'
wget -O piper-sample-generator/models/en-us-libritts-high.pt.json 'https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt.json'

# 3. Install the Piper TTS generator dependencies into the training venv
pip install piper-phonemize webrtcvad

# 4. Patch PyTorch 2.6 security restriction to allow loading the Piper .pt model
sed -i 's/model = torch.load(model_path)/model = torch.load(model_path, weights_only=False)/g' piper-sample-generator/generate_samples.py

# 5. Modify the file to set your target phrase, model name, and remove redundant RIR paths
sed -i 's/"hey jarvis"/"yo chrome"/g' yo_chrome.yml
sed -i 's/"my_model"/"yo_chrome"/g' yo_chrome.yml
sed -i 's/"\.\/my_custom_model"/"\.\/custom_models"/g' yo_chrome.yml
sed -i 's/ - ".\/mit_rirs"/\[\]/g' yo_chrome.yml
sed -i 's/ - ".\/background_clips"/\[\]/g' yo_chrome.yml

# 5. Download the correct Piper TTS high model (the generator defaults to high, not medium)
wget -O piper-sample-generator/models/en-us-libritts-high.pt 'https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt'
wget -O piper-sample-generator/models/en-us-libritts-high.pt.json 'https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt.json'

# 6. Fetch Foundational Inference Models
# openWakeWord requires core models (melspectrogram.onnx, etc.) for the training script to convert audio to features.
# It doesn't download them automatically when training from source.
python3.10 -c "from openwakeword.utils import download_models; download_models()"

# 7. Run the end-to-end training pipeline
# Note: You MUST pass the flags to generate clips, augment them, and train.
python3.10 openwakeword/train.py --training_config yo_chrome.yml --generate_clips --augment_clips --train_model
```
*(Note: On the first run, it will download several GBs of background noise datasets and Piper TTS models. Depending on your hardware, this can take 20-45 minutes as it aggressively pins your CPU/GPU to mix the audio).*

### 4. Integration
When finished, it will output a `.onnx` file in `./custom_models/yo_chrome/` (or similar). Copy that file into your `wake_daemon` directory.

> **Note on Training Errors:** If the training script crashes at the very end with `ValueError: Arg specs do not match` (in `tensorflow_probability` or TFLite conversion), you can completely ignore it. This crash happens *after* the `.onnx` file is successfully saved. Because our daemon uses ONNX Runtime on the CPU, we do not need the TFLite conversion step anyway.

Update `main.py` to bypass the default downloading logic and load your local model directly, while changing the default phrase mapping so your terminal output echoes correctly:
```python
# 1. Update the Model constructor to load the local ONNX path:
# Change this:
# model = Model(wakeword_models=["hey jarvis"])
# To this:
# model = Model(wakeword_models=["/path/to/overlay/wake_daemon/yo_chrome.onnx"])

# 2. Update the default text display:
# Change this:
# DEFAULT_WAKE_PHRASE = "hey jarvis"
# To this:
# DEFAULT_WAKE_PHRASE = "yo Chrome"
```
