/**
 * Audio module — microphone capture + HTTP POST streaming.
 * Web Audio API (ScriptProcessorNode) for raw PCM capture, no encoding issues.
 * Sends float32 PCM chunks via HTTP POST to /api/audio/stream.
 */

const BACKEND_URL = window.location.origin;
const TARGET_SAMPLE_RATE = 16000;
const CHUNK_MS = 300;
const CHUNK_SAMPLES = Math.floor((TARGET_SAMPLE_RATE * CHUNK_MS) / 1000);
const PROCESSOR_BUFFER_SIZE = 4096;

let mediaStream = null;
let audioCtx = null;
let processor = null;
let sourceNode = null;
let isRecording = false;
let wsSessionId = null;
let chunksSent = 0;
let sampleBuffer = new Float32Array(0);

const statusEl = document.getElementById("micStatus");
const startBtn = document.getElementById("startMicBtn");
const stopBtn = document.getElementById("stopMicBtn");
const vadIndicator = document.getElementById("vadIndicator");

export function showMicError(message) {
  const box = document.getElementById("micErrorBox");
  if (box) { box.textContent = message; box.classList.remove("hidden"); }
}
export function hideMicError() {
  const box = document.getElementById("micErrorBox");
  if (box) box.classList.add("hidden");
}
export function setMicStatus(text, kind = "idle") {
  if (statusEl) { statusEl.textContent = text; statusEl.className = `status status--${kind}`; }
}
export function setVadActive(active) {
  if (vadIndicator) {
    vadIndicator.classList.toggle("active", active);
    vadIndicator.classList.toggle("inactive", !active);
  }
}

function resample(samples, fromRate, toRate) {
  if (fromRate === toRate) return samples;
  const ratio = fromRate / toRate;
  const newLen = Math.floor(samples.length / ratio);
  const out = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const src = i * ratio;
    const i0 = Math.floor(src);
    const i1 = Math.min(i0 + 1, samples.length - 1);
    const frac = src - i0;
    out[i] = samples[i0] * (1 - frac) + samples[i1] * frac;
  }
  return out;
}

async function sendSamples(samples) {
  if (!samples || samples.length === 0) return;
  try {
    const resp = await fetch(`${BACKEND_URL}/api/audio/stream?session_id=${wsSessionId}`, {
      method: "POST",
      body: samples,
      headers: { "Content-Type": "application/octet-stream" },
    });
    if (!resp.ok) {
      console.log("[audio] HTTP error: " + resp.status);
    }
  } catch (err) {
    console.log("[audio] Send error: " + err.message);
  }
}

export async function startMic() {
  hideMicError();
  if (isRecording) return;

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    console.log("[audio] Microphone access granted");
  } catch (err) {
    const msg = err.name === "NotAllowedError"
      ? "Microphone permission denied. Please allow mic access and refresh."
      : err.name === "NotFoundError"
        ? "No microphone found on this device."
        : `Microphone error: ${err.message}`;
    showMicError(msg);
    setMicStatus("Mic error", "error");
    throw err;
  }

  wsSessionId = "sess_" + Math.random().toString(36).slice(2, 10);
  console.log("[audio] Session: " + wsSessionId);

  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") await audioCtx.resume();
    sourceNode = audioCtx.createMediaStreamSource(mediaStream);
    processor = audioCtx.createScriptProcessor(PROCESSOR_BUFFER_SIZE, 1, 1);

    processor.onaudioprocess = (e) => {
      if (!isRecording) return;
      const input = e.inputBuffer.getChannelData(0);
      const resampled = resample(input, audioCtx.sampleRate, TARGET_SAMPLE_RATE);

      const combined = new Float32Array(sampleBuffer.length + resampled.length);
      combined.set(sampleBuffer);
      combined.set(resampled, sampleBuffer.length);
      sampleBuffer = combined;

      const numChunks = Math.floor(sampleBuffer.length / CHUNK_SAMPLES);
      if (numChunks > 0) {
        const toSend = sampleBuffer.slice(0, numChunks * CHUNK_SAMPLES);
        sampleBuffer = sampleBuffer.slice(numChunks * CHUNK_SAMPLES);
        sendSamples(toSend);
        chunksSent += numChunks;
        if (chunksSent <= 1 || chunksSent % 10 === 0) {
          console.log("[audio] Sent chunks: " + chunksSent);
        }
      }

      setVadActive(true);
      setTimeout(() => setVadActive(false), 150);
    };

    sourceNode.connect(processor);
    processor.connect(audioCtx.destination);
    console.log("[audio] Audio graph connected");

    setMicStatus("Microphone live", "live");
    startBtn.disabled = true;
    stopBtn.disabled = false;
    isRecording = true;
  } catch (err) {
    console.error("Audio node setup failed:", err);
    showMicError("Failed to set up audio processing: " + err.message);
    stopMic();
    throw err;
  }
}

export function stopMic() {
  console.log("[audio] stopMic called");
  isRecording = false;
  try { if (processor) { processor.disconnect(); processor.onaudioprocess = null; } } catch (_) {}
  processor = null;
  try { if (sourceNode) sourceNode.disconnect(); } catch (_) {}
  sourceNode = null;
  try { if (audioCtx) audioCtx.close(); } catch (_) {}
  audioCtx = null;
  if (mediaStream) { try { mediaStream.getTracks().forEach((t) => t.stop()); } catch (_) {} mediaStream = null; }
  sampleBuffer = new Float32Array(0);
  wsSessionId = null;
  setMicStatus("Mic stopped", "idle");
  setVadActive(false);
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

export function isMicRecording() { return isRecording; }
export function getMicSessionId() { return wsSessionId; }
