/**
 * Face overlay module: draws bounding boxes and track IDs on a canvas
 * positioned over the video element.
 *
 * Boxes are smoothed every animation frame using linear interpolation
 * toward the latest detection targets, so faces move fluidly instead of
 * jumping. Tracks persist for a few frames after a missed detection so
 * boxes don't flicker on/off.
 */

const canvas = document.getElementById("overlayCanvas");
const ctx = canvas.getContext("2d");

let frameW = 640;
let frameH = 480;

// Smoothing factor (0 = instant snap, 1 = never move). 0.2 = smooth.
const SMOOTH = 0.18;

// How many animation frames to keep showing a track after last detection.
const PERSIST_FRAMES = 12;

// Internal track state keyed by track_id.
const tracks = new Map(); // id -> {x,y,w,h, tx,ty,tw,th, persist, hits, age}

function bindVideo(videoEl) {
  if (!videoEl) return;
  const rect = videoEl.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  // Keep a copy of the video's intrinsic resolution for coordinate scaling.
  frameW = videoEl.videoWidth || 640;
  frameH = videoEl.videoHeight || 480;
}

function clearOverlay() {
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * Called by the app when a detection response arrives.
 * Merges new detections into the smoothing state.
 */
function updateTracks(faces, fw, fh) {
  if (fw) frameW = fw;
  if (fh) frameH = fh;

  const seenIds = new Set();

  for (const face of faces) {
    const id = face.track_id;
    seenIds.add(id);
    const sx = face.x / frameW;
    const sy = face.y / frameH;
    const sw = face.w / frameW;
    const sh = face.h / frameH;

    let t = tracks.get(id);
    if (!t) {
      t = {
        x: sx, y: sy, w: sw, h: sh,
        tx: sx, ty: sy, tw: sw, th: sh,
        persist: 0, hits: face.hits, age: face.age,
        identity: face.identity || { status: "checking" },
      };
      tracks.set(id, t);
    } else {
      t.tx = sx; t.ty = sy; t.tw = sw; t.th = sh;
      t.persist = 0;
      t.hits = face.hits;
      t.age = face.age;
      t.identity = face.identity || t.identity;
    }
  }

  // Age out tracks not seen in this detection batch.
  for (const [id, t] of tracks) {
    if (!seenIds.has(id)) {
      t.persist += 1;
    }
  }

  // Remove tracks that have been missing too long.
  for (const [id, t] of tracks) {
    if (t.persist > PERSIST_FRAMES) {
      tracks.delete(id);
    }
  }
}

/**
 * Draw all tracks with interpolated positions. Called every animation frame.
 */
function draw() {
  if (!ctx) return;
  clearOverlay();

  const cw = canvas.width;
  const ch = canvas.height;

  ctx.lineWidth = 4;
  ctx.font = "700 18px system-ui, sans-serif";

  for (const [id, t] of tracks) {
    // Interpolate current position toward target.
    t.x += (t.tx - t.x) * SMOOTH;
    t.y += (t.ty - t.y) * SMOOTH;
    t.w += (t.tw - t.w) * SMOOTH;
    t.h += (t.th - t.h) * SMOOTH;

    const x = t.x * cw;
    const y = t.y * ch;
    const w = t.w * cw;
    const h = t.h * ch;

    const { text, color } = describe(t.identity);
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    // Fade out slightly while persisting without detection.
    const alpha = t.persist === 0 ? 1 : Math.max(0.5, 1 - t.persist * 0.08);
    ctx.globalAlpha = alpha;

    const textWidth = ctx.measureText(text).width;
    const pad = 8;
    const labelH = 30;
    const ly = y - labelH - 2 < 0 ? y + 2 : y - labelH - 2;

    ctx.fillStyle = color;
    ctx.fillRect(x, ly, textWidth + pad * 2, labelH);
    ctx.fillStyle = "#08101e";
    ctx.fillText(text, x + pad, ly + labelH - 9);
    ctx.globalAlpha = 1;
  }
}

// What to show for each recognition status. Only a confident match shows a
// name. Temporary track IDs are never shown: they are not identities.
function describe(identity) {
  switch (identity.status) {
    case "recognized":
      return { text: identity.tag ? `${identity.name} · ${identity.tag}` : identity.name, color: "#4ade80" };
    case "new_face":
      return { text: "New face - register?", color: "#4cc2ff" };
    case "unknown":
      return { text: "Not sure who this is", color: "#e2e8f0" };
    case "uncertain":
      return { text: "Not sure who this is", color: "#fbbf24" };
    case "recognition_unavailable":
      return { text: "Face detected", color: "#e2e8f0" };
    default:
      return { text: "Looking...", color: "#fbbf24" };
  }
}

function onResize(videoEl) {
  bindVideo(videoEl);
}

export { bindVideo, updateTracks, draw, onResize, clearTracks };

function clearTracks() {
  tracks.clear();
  if (ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}
