/**
 * cantforget.ai — Face & Voice app
 * Wires camera + mic + conversation display together.
 */

import { startCamera, stopCamera, isCameraActive, showError, hideError, getVideoStream } from "./camera.js";
import { bindVideo, updateTracks, draw, onResize, clearTracks } from "./face-overlay.js?v=2";
import { startMic, stopMic, isMicRecording, hideMicError, getMicSessionId } from "./audio.js";
import { startVoiceSession, stopVoiceSession, isVoiceConnected, summarizeSession } from "./conversation.js";
import { initPeople } from "./people.js";
import { setIntelligenceSession, clearIntelligenceSession } from "./conversation-intelligence.js";

const cameraToggleBtn = document.getElementById("cameraToggleBtn");
const micToggleBtn = document.getElementById("micToggleBtn");
const menuBtn = document.getElementById("menuBtn");
const closeMenuBtn = document.getElementById("closeMenuBtn");
const sidePanel = document.getElementById("sidePanel");
const sideOverlay = document.getElementById("sideOverlay");
const fpsEl = document.getElementById("fpsDisplay");
const faceCountEl = document.getElementById("faceCountDisplay");
const detectMsEl = document.getElementById("detectMsDisplay");
const videoEl = document.getElementById("localVideo");

// --- Side panel drawer (menu button) ---
function openMenu() {
  if (!sidePanel) return;
  sidePanel.classList.add("open");
  if (sideOverlay) sideOverlay.classList.remove("hidden");
}
function closeMenu() {
  if (!sidePanel) return;
  sidePanel.classList.remove("open");
  if (sideOverlay) sideOverlay.classList.add("hidden");
}
if (menuBtn) menuBtn.addEventListener("click", openMenu);
if (closeMenuBtn) closeMenuBtn.addEventListener("click", closeMenu);
if (sideOverlay) sideOverlay.addEventListener("click", closeMenu);

// --- Panel tabs (Conversation / People) ---
const tabBar = document.getElementById("panelTabs");
if (tabBar) {
  const tabs = [...tabBar.querySelectorAll(".panel-tab")];
  tabs.forEach((btn) =>
    btn.addEventListener("click", () => {
      tabs.forEach((b) => b.classList.toggle("active", b === btn));
      for (const page of document.querySelectorAll(".tab-page")) {
        page.classList.toggle("hidden", page.id !== btn.dataset.tab);
      }
    })
  );
}

const BACKEND_URL = window.location.origin;
const DETECT_INTERVAL_MS = 150;

let animationId = null;
let lastTime = 0;
let frameCount = 0;
let lastFpsTime = 0;
let isDetectionRunning = false;
let lastDetectTime = 0;

function endVoiceSession(sid) {
  if (!sid) return;
  try {
    fetch(`${BACKEND_URL}/api/voice/sessions/${encodeURIComponent(sid)}/stop`, {
      method: "POST",
      keepalive: true,
    });
  } catch (_) { /* best effort */ }
}

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
    const resp = await fetch(`${BACKEND_URL}/api/faces/detect`, { method: "POST", body: formData });
    if (!resp.ok) throw new Error(`Backend responded ${resp.status}`);
    const data = await resp.json();
    if (data.ok && Array.isArray(data.faces)) {
      updateTracks(data.faces, data.frame_w || 640, data.frame_h || 480);
      if (faceCountEl) faceCountEl.textContent = data.faces.length;
      if (detectMsEl) detectMsEl.textContent = (performance.now() - start).toFixed(0);
    }
  } catch (err) {
    console.error("Detection error:", err);
  } finally {
    isDetectionRunning = false;
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
    if (fpsEl) fpsEl.textContent = frameCount.toFixed(0);
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

function refreshCameraButton() {
  if (!cameraToggleBtn) return;
  const on = isCameraActive();
  cameraToggleBtn.classList.toggle("on", on);
  cameraToggleBtn.setAttribute("aria-label", on ? "Turn camera off" : "Turn camera on");
}

function refreshMicButton() {
  if (!micToggleBtn) return;
  const on = isMicRecording();
  micToggleBtn.classList.toggle("on", on);
  micToggleBtn.setAttribute("aria-label", on ? "Turn microphone off" : "Turn microphone on");
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

  if (micToggleBtn) micToggleBtn.disabled = false;

  lastTime = 0;
  frameCount = 0;
  lastFpsTime = performance.now();
  lastDetectTime = performance.now();

  if (animationId) cancelAnimationFrame(animationId);
  animationId = requestAnimationFrame(loop);
  refreshCameraButton();
}

function stop() {
  const sid = getMicSessionId();
  stopCamera();
  stopMic();
  stopVoiceSession();
  endVoiceSession(sid);
  clearTracks();
  if (faceCountEl) faceCountEl.textContent = "0";
  if (fpsEl) fpsEl.textContent = "--";
  if (detectMsEl) detectMsEl.textContent = "--";

  if (micToggleBtn) micToggleBtn.disabled = true;

  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  lastTime = 0;
  lastDetectTime = 0;
  refreshCameraButton();
  refreshMicButton();
}

if (cameraToggleBtn) {
  cameraToggleBtn.addEventListener("click", async () => {
    if (isCameraActive()) {
      stop();
    } else {
      await start();
    }
  });
}

if (micToggleBtn) {
  micToggleBtn.addEventListener("click", async () => {
    if (isMicRecording()) {
      const sid = getMicSessionId();
      stopMic();
      stopVoiceSession();
      endVoiceSession(sid);
      clearIntelligenceSession();
      // Auto-summarize the conversation once the mic session ends.
      try { await summarizeSession(); } catch (_) { /* best effort */ }
    } else {
      try {
        await startMic();
        // Subscribe to transcription events for the SAME session the audio is sent to.
        const sid = getMicSessionId();
        if (sid) {
          console.log("[app] Starting voice session:", sid);
          startVoiceSession(sid);
          setIntelligenceSession(sid);
        }
      } catch (err) {
        // startMic already showed an error
      }
    }
    refreshMicButton();
  });
}

window.addEventListener("beforeunload", () => {
  stop();
});

// Global error handler
window.addEventListener("error", (ev) => {
  console.error("[app] Error:", ev.message);
});
window.addEventListener("unhandledrejection", (ev) => {
  console.error("[app] Unhandled rejection:", ev.reason);
});

// Load the registered-people list so stored faces are visible.
initPeople();
