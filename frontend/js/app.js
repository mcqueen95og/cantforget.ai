/**
 * ForgetMeNot — Face & Voice app
 * Wires camera + mic + conversation display together.
 */

import { startCamera, stopCamera, isCameraActive, showError, hideError, setStatus, getVideoStream } from "./camera.js";
import { bindVideo, updateTracks, draw, onResize, clearTracks } from "./face-overlay.js";
import { startMic, stopMic, isMicRecording, setMicStatus, showMicError, hideMicError, setVadActive, getMicSessionId } from "./audio.js";
import { startVoiceSession, stopVoiceSession, isVoiceConnected, summarizeSession } from "./conversation.js";
import { initPeople } from "./people.js";
import { setIntelligenceSession, clearIntelligenceSession } from "./conversation-intelligence.js";

const startBtn = document.getElementById("startCameraBtn");
const stopBtn = document.getElementById("stopCameraBtn");
const startMicBtn = document.getElementById("startMicBtn");
const stopMicBtn = document.getElementById("stopMicBtn");
const fpsEl = document.getElementById("fpsDisplay");
const faceCountEl = document.getElementById("faceCountDisplay");
const detectMsEl = document.getElementById("detectMsDisplay");
const videoEl = document.getElementById("localVideo");

// --- Sidebar tabs (Conversation / People / History) ---
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

// --- Person notification card (AR-demo style) ---
// When a face is recognised with a name, announce it in a large, readable
// card at the bottom-right. Auto-dismisses; same person is not repeated
// within a short cooldown so the card does not flicker.
const personCard = document.getElementById("personNotification");
const personNotifyCooldown = new Map(); // person_uid -> timestamp
let personCardTimer = null;

function showPersonNotification({ name, tag, description }) {
  if (!personCard || !name) return;
  personCard.innerHTML = "";
  const h = document.createElement("h3");
  h.textContent = name;
  personCard.appendChild(h);
  if (tag) {
    const b = document.createElement("span");
    b.className = "pn-tag";
    b.textContent = tag;
    personCard.appendChild(b);
  }
  const p = document.createElement("p");
  p.className = "pn-desc";
  p.textContent = description || "Recognised from stored face data.";
  personCard.appendChild(p);
  personCard.classList.remove("hidden");
  clearTimeout(personCardTimer);
  personCardTimer = setTimeout(() => personCard.classList.add("hidden"), 8000);
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
      faceCountEl.textContent = data.faces.length;
      detectMsEl.textContent = (performance.now() - start).toFixed(0);

      // Announce recognised people (max one card at a time, 30 s cooldown).
      const now = Date.now();
      for (const face of data.faces) {
        const id = face.identity || {};
        if (id.status === "recognized" && id.person_uid) {
          const last = personNotifyCooldown.get(id.person_uid) || 0;
          if (now - last > 30000) {
            personNotifyCooldown.set(id.person_uid, now);
            showPersonNotification(id);
            break;
          }
        }
      }
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
  startMicBtn.disabled = false;

  lastTime = 0;
  frameCount = 0;
  lastFpsTime = performance.now();
  lastDetectTime = performance.now();

  if (animationId) cancelAnimationFrame(animationId);
  animationId = requestAnimationFrame(loop);
}

function stop() {
  const sid = getMicSessionId();
  stopCamera();
  stopMic();
  stopVoiceSession();
  endVoiceSession(sid);
  clearTracks();
  faceCountEl.textContent = "0";
  fpsEl.textContent = "--";
  detectMsEl.textContent = "--";

  startBtn.disabled = false;
  stopBtn.disabled = true;
  startMicBtn.disabled = true;
  stopMicBtn.disabled = true;

  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  lastTime = 0;
  lastDetectTime = 0;
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

startMicBtn.addEventListener("click", async () => {
  try {
    await startMic();
    startMicBtn.disabled = true;
    stopMicBtn.disabled = false;
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
});

stopMicBtn.addEventListener("click", async () => {
  const sid = getMicSessionId();
  stopMic();
  stopVoiceSession();
  endVoiceSession(sid);
  startMicBtn.disabled = false;
  stopMicBtn.disabled = true;
  clearIntelligenceSession();
  // Auto-summarize the conversation once the mic session ends.
  try { await summarizeSession(); } catch (_) { /* best effort */ }
});
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
