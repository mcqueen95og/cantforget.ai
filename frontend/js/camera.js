/**
 * Camera module — getUserMedia + stream lifecycle.
 */

const videoEl = document.getElementById("localVideo");
const statusEl = document.getElementById("cameraStatus");
const errorBox = document.getElementById("errorBox");

let stream = null;

export function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}
export function hideMicError() {
  errorBox.classList.add("hidden");
}
export function hideError() {
  errorBox.classList.add("hidden");
}
export function setStatus(text, kind = "idle") {
  statusEl.textContent = text;
  statusEl.className = `status status--${kind}`;
}

export async function startCamera() {
  errorBox.classList.add("hidden");
  let lastError = null;

  const constraintSets = [
    { video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" } },
    { video: { width: { ideal: 1280 }, height: { ideal: 720 } } },
    { video: { facingMode: "user" } },
    { video: true },
  ];

  for (const constraints of constraintSets) {
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      break;
    } catch (err) {
      lastError = err;
    }
  }

  if (!stream) {
    const msg = lastError.name === "NotAllowedError"
      ? "Camera permission denied. Please allow camera access and refresh."
      : lastError.name === "NotFoundError"
        ? "No camera found on this device."
        : `Camera error: ${lastError.message}`;
    showError(msg);
    setStatus("Camera error", "error");
    throw lastError;
  }

  videoEl.srcObject = stream;
  setStatus("Camera live", "live");
  return stream;
}

export function stopCamera() {
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  if (videoEl.srcObject) videoEl.srcObject = null;
  setStatus("Camera stopped", "idle");
}

export function isCameraActive() {
  return !!stream && stream.getVideoTracks().some((t) => t.readyState === "live");
}
export function getVideoStream() { return stream; }
