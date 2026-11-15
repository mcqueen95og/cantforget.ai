#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# cantforget.ai — one-shot installer for Linux (Ubuntu/Mint/Debian/others)
#
#   bash scripts/install.sh
#
# Does everything: checks Python, creates a private virtual environment,
# installs every library, and downloads the AI model files. Internet is
# needed once, for the downloads; the app itself then runs fully offline.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

# Always run from the project root even if invoked from elsewhere.
PROJECT_DIR=$(pwd)
MODELS_DIR="$PROJECT_DIR/data/models"
SPEAKER_DIR="$MODELS_DIR/speaker"
VENV_DIR="$PROJECT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
ok()   { printf '  \033[32m✔\033[0m %s\n' "$1"; }
info() { printf '  \033[36m→\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }

echo ""
bold "cantforget.ai — installer (Linux)"
echo ""

# --- 1. Python ------------------------------------------------------------
bold "1/4  Checking Python"
if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 was not found."
    echo "Install it first, e.g.:  sudo apt install python3 python3-venv python3-pip"
    exit 1
fi
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
MIN_MAJOR=${PY_VERSION%%.*}; MIN_MINOR=${PY_VERSION##*.}
if [ "$MIN_MAJOR" -lt 3 ] || { [ "$MIN_MAJOR" -eq 3 ] && [ "$MIN_MINOR" -lt 10 ]; }; then
    warn "Python $PY_VERSION found — 3.10 or newer is required."
    echo "Please upgrade Python and re-run this script."
    exit 1
fi
ok "Python $PY_VERSION found"

# --- 2. Virtual environment -----------------------------------------------
bold "2/4  Creating a private Python environment (.venv)"
if [ ! -x "$VENV_PY" ]; then
    python3 -m venv "$VENV_DIR"
    ok "Environment created at $VENV_DIR"
else
    ok "Environment already exists — reusing it"
fi

# --- 3. Libraries ----------------------------------------------------------
bold "3/4  Installing the Python libraries (this can take a few minutes)"
"$VENV_PY" -m pip install --upgrade pip --quiet
"$VENV_PY" -m pip install -r "$PROJECT_DIR/requirements.txt"
ok "Libraries installed (FastAPI, OpenCV, Whisper, sherpa-onnx, ...)"

# --- 4. AI model files -----------------------------------------------------
bold "4/4  Downloading the AI model files"
mkdir -p "$SPEAKER_DIR"

fetch() { # fetch <url> <destination> <friendly-name>
    local url="$1" dest="$2" name="$3"
    if [ -s "$dest" ]; then
        ok "$name already present"
        return 0
    fi
    info "downloading $name ..."
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "$dest.part" "$url" && mv "$dest.part" "$dest"
    else
        wget -q -O "$dest.part" "$url" && mv "$dest.part" "$dest"
    fi
    ok "$name saved"
}

fetch "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
      "$MODELS_DIR/face_detection_yunet.onnx" \
      "face detector (YuNet)"

fetch "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
      "$MODELS_DIR/face_recognition_sface_2021dec.onnx" \
      "face recogniser (SFace)"

fetch "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" \
      "$MODELS_DIR/silero_vad.onnx" \
      "voice detector (Silero VAD)"

fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" \
      "$SPEAKER_DIR/eres2net.onnx" \
      "voice recogniser (ERes2Net)"

# The speech-to-text model (Whisper "tiny") downloads itself on first use.
echo ""
bold "Install complete ✔"
echo ""
echo "To start the app:"
echo "    cd $PROJECT_DIR"
echo "    python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"
echo ""
echo "Then open  http://127.0.0.1:8000/  in your browser."
echo ""
