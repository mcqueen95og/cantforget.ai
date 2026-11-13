/**
 * Debug module: wires camera, face detection polling, and detailed
 * pipeline metrics display for debugging the face recognition subsystem.
 */

import {
  startCamera,
  stopCamera,
  isCameraActive,
  showError,
  hideError,
  setStatus,
  getVideoStream,
} from "./camera.js";
import {
  bindVideo,
  updateTracks,
  draw,
  onResize,
  clearTracks,
} from "./face-overlay.js";

const startBtn = document.getElementById("startCameraBtn");
const stopBtn = document.getElementById("stopCameraBtn");
const fpsEl = document.getElementById("dbgFps");
const faceCountEl = document.getElementById("dbgFaceCount");
const detectMsEl = document.getElementById("dbgDetectMs");
const trackMsEl = document.getElementById("dbgTrackMs");
const recogMsEl = document.getElementById("dbgRecogMs");
const totalMsEl = document.getElementById("dbgTotalMs");
const detModelEl = document.getElementById("dbgDetModel");
const recModelEl = document.getElementById("dbgRecModel");
const gallerySizeEl = document.getElementById("dbgGallerySize");
const frameSizeEl = document.getElementById("dbgFrameSize");
const faceListEl = document.getElementById("dbgFaceList");
const videoEl = document.getElementById("localVideo");

const BACKEND_URL = window.location.origin;
const DETECT_INTERVAL_MS = 150;

let animationId = null;
let lastTime = 0;
let frameCount = 0;
let lastFpsTime = 0;
let isDetectionRunning = false;
let lastDetectTime = 0;

function captureFrame() {
  const offscreen = document.createElement("canvas");
  offscreen.width = videoEl.videoWidth || 640;
  offscreen.height = videoEl.videoHeight || 480;
  const octx = offscreen.getContext("2d");
  octx.drawImage(videoEl, 0, 0, offscreen.width, offscreen.height);

  return new Promise((resolve) => {
    offscreen.toBlob((blob) => resolve(blob), "image/jpeg", 0.75);
  });
}

async function detectFrame() {
  if (isDetectionRunning) return;
  isDetectionRunning = true;
  const start = performance.now();

  try {
    const blob = await captureFrame();
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    const resp = await fetch(`${BACKEND_URL}/api/faces/detect`, {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      throw new Error(`Backend responded ${resp.status}`);
    }

    const data = await resp.json();

    if (data.ok) {
      updateTracks(data.faces, data.frame_w || 640, data.frame_h || 480);

      // Update metrics
      faceCountEl.textContent = data.faces.length;
      if (data.timing_ms) {
        detectMsEl.textContent = data.timing_ms.detection;
        trackMsEl.textContent = data.timing_ms.tracking;
        recogMsEl.textContent = data.timing_ms.recognition;
        totalMsEl.textContent = data.timing_ms.total;
      }
      if (data.frame_w && data.frame_h) {
        frameSizeEl.textContent = `${data.frame_w}×${data.frame_h}`;
      }
      if (data.models) {
        detModelEl.innerHTML = `<span class="model-badge">${shortName(data.models.detector)}</span>`;
        recModelEl.innerHTML = `<span class="model-badge">${shortName(data.models.recognizer)}</span>`;
      }
      renderFaceList(data.faces);
    }
  } catch (err) {
    console.error("Debug detection loop error:", err);
  } finally {
    isDetectionRunning = false;
  }
}

function shortName(modelPath) {
  if (!modelPath) return "unknown";
  const parts = modelPath.split("/");
  const file = parts[parts.length - 1];
  if (file.includes("yunet")) return "YuNet";
  if (file.includes("scrfd")) return "SCRFD";
  if (file.includes("sface")) return "SFace 128d";
  if (file.includes("arcface")) return "ArcFace 512d";
  return file;
}

function renderFaceList(faces) {
  faceListEl.innerHTML = "";
  if (!faces || faces.length === 0) {
    faceListEl.innerHTML = `<div style="color: var(--muted); font-style: italic;">No faces detected</div>`;
    return;
  }
  for (const f of faces) {
    const el = document.createElement("div");
    el.className = "face-entry";
    const recognized = f.is_recognized;
    const sim = f.recognition_confidence != null ? (f.recognition_confidence * 100).toFixed(1) : "—";
    const detConf = (f.confidence * 100).toFixed(1);
    const lmCount = f.landmarks ? f.landmarks.length : 0;
    el.innerHTML = `
      <div class="face-name ${recognized ? "recognized" : "unknown"}">
        Face ${f.track_id} &middot; ${recognized ? f.person_name || "Person" : "Unknown"}
      </div>
      <div class="face-detail">
        Box: (${f.x}, ${f.y}, ${f.w}×${f.h}) &middot;
        Det conf: ${detConf}% &middot;
        Sim: ${sim}% &middot;
        Landmarks: ${lmCount} &middot;
        Recog: ${f.recognition_ms != null ? f.recognition_ms + "ms" : "—"}
      </div>
    `;
    faceListEl.appendChild(el);
  }
}

function loop(timestamp) {
  if (!isCameraActive()) {
    cancelAnimationFrame(animationId);
    animationId = null;
    return;
  }

  if (!lastTime) lastTime = timestamp;
  lastTime = timestamp;

  frameCount++;
  if (timestamp - lastFpsTime >= 1000) {
    fpsEl.textContent = frameCount.toFixed(0);
    frameCount = 0;
    lastFpsTime = timestamp;
  }

  if (timestamp - lastDetectTime >= DETECT_INTERVAL_MS) {
    detectFrame();
    lastDetectTime = timestamp;
  }

  draw();
  animationId = requestAnimationFrame(loop);
}

async function start() {
  hideError();
  try {
    await startCamera();
  } catch (err) {
    return;
  }

  bindVideo(videoEl);
  videoEl.addEventListener("loadedmetadata", () => bindVideo(videoEl));
  window.addEventListener("resize", () => onResize(videoEl));

  startBtn.disabled = true;
  stopBtn.disabled = false;

  lastTime = 0;
  frameCount = 0;
  lastFpsTime = performance.now();
  lastDetectTime = performance.now();

  if (animationId) cancelAnimationFrame(animationId);
  animationId = requestAnimationFrame(loop);

  // Load gallery size
  try {
    const resp = await fetch(`${BACKEND_URL}/api/health`);
    if (resp.ok) {
      const data = await resp.json();
      gallerySizeEl.textContent = data.gallery_size;
    }
  } catch (err) {
    console.error(err);
  }
}

function stop() {
  stopCamera();
  clearTracks();
  faceCountEl.textContent = "0";
  fpsEl.textContent = "--";
  detectMsEl.textContent = "--";
  trackMsEl.textContent = "--";
  recogMsEl.textContent = "--";
  totalMsEl.textContent = "--";
  frameSizeEl.textContent = "--";
  faceListEl.innerHTML = `<div style="color: var(--muted); font-style: italic;">No faces detected</div>`;

  startBtn.disabled = false;
  stopBtn.disabled = true;

  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  lastTime = 0;
  lastDetectTime = 0;
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

window.addEventListener("beforeunload", () => {
  stop();
});
