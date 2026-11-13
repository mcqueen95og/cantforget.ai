"""Application configuration (environment-variable driven)."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

# --- Face detection (Phase 1) ---
YUNET_MODEL_PATH = os.environ.get(
    "YUNET_MODEL_PATH", str(MODELS_DIR / "face_detection_yunet.onnx")
)
YUNET_INPUT_SIZE = (320, 320)
YUNET_SCORE_THRESHOLD = float(os.environ.get("YUNET_SCORE_THRESHOLD", "0.6"))
YUNET_NMS_THRESHOLD = float(os.environ.get("YUNET_NMS_THRESHOLD", "0.3"))

# --- Face recognition (Phase 2) ---
SFACE_MODEL_PATH = os.environ.get(
    "SFACE_MODEL_PATH", str(MODELS_DIR / "face_recognition_sface_2021dec.onnx")
)
DB_PATH = os.environ.get("DB_PATH", str(DATA_DIR / "forgetmenot.db"))

# Cosine similarity between SFace embeddings (higher = more similar).
# A person is only named when the best score >= MATCH_THRESHOLD AND it beats
# the runner-up person by MATCH_MARGIN. Scores in [UNKNOWN_BELOW, MATCH_THRESHOLD)
# or ambiguous results are "uncertain" (never named, never auto-registered).
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.45"))
MATCH_MARGIN = float(os.environ.get("MATCH_MARGIN", "0.05"))
UNKNOWN_BELOW = float(os.environ.get("UNKNOWN_BELOW", "0.30"))
# A name is shown only after this many consecutive agreeing recognitions.
RECOG_CONFIRM_FRAMES = int(os.environ.get("RECOG_CONFIRM_FRAMES", "2"))
# Re-verify an already-recognised track every N detect calls.
RECOG_REVERIFY_FRAMES = int(os.environ.get("RECOG_REVERIFY_FRAMES", "10"))
# Faces smaller than this (px, shorter side) or less confident are not embedded.
MIN_FACE_PX = int(os.environ.get("MIN_FACE_PX", "70"))
MIN_FACE_CONFIDENCE = float(os.environ.get("MIN_FACE_CONFIDENCE", "0.8"))

# --- New-face candidates (held in memory only until a caregiver registers) ---
PENDING_MIN_SAMPLES = int(os.environ.get("PENDING_MIN_SAMPLES", "4"))
PENDING_MAX_SAMPLES = int(os.environ.get("PENDING_MAX_SAMPLES", "5"))
PENDING_TTL_SECONDS = int(os.environ.get("PENDING_TTL_SECONDS", "900"))
PENDING_MAX_CANDIDATES = int(os.environ.get("PENDING_MAX_CANDIDATES", "8"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))

# --- Tracking ---
TRACKER_MAX_DISAPPEARED = int(os.environ.get("TRACKER_MAX_DISAPPEARED", "15"))
TRACKER_MAX_DISTANCE = float(os.environ.get("TRACKER_MAX_DISTANCE", "80.0"))
TRACKER_IOU_THRESHOLD = float(os.environ.get("TRACKER_IOU_THRESHOLD", "0.25"))

# --- Voice / conversation (Phase 3) ---
SAMPLE_RATE = 16000
VAD_MODEL_PATH = os.environ.get(
    "VAD_MODEL_PATH", str(MODELS_DIR / "silero_vad.onnx")
)
SPEAKER_MODEL_PATH = os.environ.get(
    "SPEAKER_MODEL_PATH", str(MODELS_DIR / "speaker" / "eres2net.onnx")
)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")  # tiny | base | small
# Optional fixed language (e.g. "en"). Empty = auto-detect per segment.
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "").strip() or None
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.35"))
VAD_MIN_SPEECH_MS = int(os.environ.get("VAD_MIN_SPEECH_MS", "200"))
VAD_MIN_SILENCE_MS = int(os.environ.get("VAD_MIN_SILENCE_MS", "400"))
VAD_SPEECH_PAD_MS = int(os.environ.get("VAD_SPEECH_PAD_MS", "150"))
DIARIZATION_MAX_SPEAKERS = int(os.environ.get("DIARIZATION_MAX_SPEAKERS", "4"))
DIARIZATION_MIN_SEGMENT_MS = int(os.environ.get("DIARIZATION_MIN_SEGMENT_MS", "500"))
DIARIZATION_THRESHOLD = float(os.environ.get("DIARIZATION_THRESHOLD", "0.5"))
VOICE_ENROLL_MIN_SAMPLES = int(os.environ.get("VOICE_ENROLL_MIN_SAMPLES", "3"))
VOICE_ENROLL_MAX_SAMPLES = int(os.environ.get("VOICE_ENROLL_MAX_SAMPLES", "10"))
VOICE_SIMILARITY_THRESHOLD = float(os.environ.get("VOICE_SIMILARITY_THRESHOLD", "0.45"))
VOICE_ASSOC_CONFIRM_FRAMES = int(os.environ.get("VOICE_ASSOC_CONFIRM_FRAMES", "3"))
VOICE_ASSOC_MAX_OFFSET_MS = int(os.environ.get("VOICE_ASSOC_MAX_OFFSET_MS", "1000"))

# --- Conversation summaries (local, CPU-only) ---
SUMMARIZER_ENABLED = os.environ.get("SUMMARIZER_ENABLED", "1") == "1"
# Prefer flan-t5-base if present, otherwise flan-t5-small.
SUMMARIZER_MODEL_PATH = os.environ.get(
    "SUMMARIZER_MODEL_PATH", str(MODELS_DIR / "flan-t5-base")
)
SUMMARIZER_FALLBACK_PATH = os.environ.get(
    "SUMMARIZER_FALLBACK_PATH", str(MODELS_DIR / "flan-t5-small")
)
SUMMARIZER_MAX_INPUT_CHARS = int(os.environ.get("SUMMARIZER_MAX_INPUT_CHARS", "1200"))
SUMMARIZER_MAX_NEW_TOKENS = int(os.environ.get("SUMMARIZER_MAX_NEW_TOKENS", "64"))
# The abstractive model is optional: the extractive summarizer is instant and
# never invents content. Enable the local model explicitly when its quality is
# good enough on your hardware.
SUMMARIZER_USE_MODEL = os.environ.get("SUMMARIZER_USE_MODEL", "0") == "1"
SUMMARY_MIN_SEGMENTS = int(os.environ.get("SUMMARY_MIN_SEGMENTS", "4"))
SUMMARY_MAX_SEGMENTS = int(os.environ.get("SUMMARY_MAX_SEGMENTS", "12"))
SUMMARY_MIN_SPAN_MS = int(os.environ.get("SUMMARY_MIN_SPAN_MS", "15000"))

# --- Server ---
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

# --- Frontend static files ---
FRONTEND_DIR = BASE_DIR / "frontend"
