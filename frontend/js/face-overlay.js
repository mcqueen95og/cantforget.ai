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

    // Fade out slightly while persisting without detection.
    const alpha = t.persist === 0 ? 1 : Math.max(0.5, 1 - t.persist * 0.08);

    drawPersonCard(x, y, w, h, t.identity, alpha);
  }
}

// --------------------------------------------------------------- person card

const CARD_WIDTH = 240;
const CARD_PAD_X = 14;
const CARD_PAD_Y = 12;
const TITLE_FONT = "600 20px Georgia, 'Palatino Linotype', serif";
const ROLE_FONT = "500 12px Georgia, 'Palatino Linotype', serif";
const DESC_FONT = "400 13px Georgia, 'Palatino Linotype', serif";
const DESC_LINE_H = 18;

function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

function wrapText(text, maxW, maxLines) {
  const words = String(text).split(/\s+/).filter(Boolean);
  const lines = [];
  let line = "";
  for (const word of words) {
    const test = line ? line + " " + word : word;
    if (ctx.measureText(test).width > maxW && line) {
      lines.push(line);
      line = word;
    } else {
      line = test;
    }
  }
  if (line) lines.push(line);
  return lines.slice(0, maxLines);
}

// What to show on the card. Only a confident match shows a name and profile;
// temporary track IDs are never shown: they are not identities.
function describe(identity) {
  if (identity && identity.status === "recognized" && identity.name) {
    return {
      title: identity.name,
      role: identity.tag || "",
      description: identity.description || "",
    };
  }
  return { title: "Unknown Person", role: "", description: "" };
}

/**
 * Draw a translucent rounded info card right next to a face:
 *   heading = person name, small rounded tab with their role next to it,
 *   and the description/summary underneath.
 * Unrecognised faces get a card titled "Unknown Person".
 */
function drawPersonCard(fx, fy, fw, fh, identity, alpha) {
  const { title, role, description } = describe(identity);

  // ---- measure ----
  ctx.font = TITLE_FONT;
  const titleW = ctx.measureText(title).width;
  let roleW = 0;
  if (role) {
    ctx.font = ROLE_FONT;
    roleW = ctx.measureText(role).width + 16; // pill padding
  }
  const textW = CARD_WIDTH - CARD_PAD_X * 2;
  let descLines = [];
  if (description) {
    ctx.font = DESC_FONT;
    descLines = wrapText(description, textW, 3);
  }

  const headerH = 26;
  const descH = descLines.length ? descLines.length * DESC_LINE_H + 4 : 0;
  const cardH = CARD_PAD_Y * 2 + headerH + descH;

  // ---- position: to the right of the face, flipped left when overflowing ----
  let cx = fx + fw + 14;
  if (cx + CARD_WIDTH > canvas.width - 10) cx = fx - CARD_WIDTH - 14;
  if (cx < 10) cx = Math.max(10, Math.min(canvas.width - CARD_WIDTH - 10, fx));
  let cy = fy;
  if (cy + cardH > canvas.height - 10) cy = canvas.height - cardH - 10;
  if (cy < 10) cy = 10;

  // ---- card body (semi-transparent) ----
  ctx.globalAlpha = alpha;
  ctx.fillStyle = "rgba(255, 255, 255, 0.82)";
  roundRect(ctx, cx, cy, CARD_WIDTH, cardH, 16);
  ctx.fill();
  ctx.strokeStyle = "rgba(29, 43, 58, 0.18)";
  ctx.lineWidth = 1.2;
  ctx.stroke();

  // ---- heading: name, with the role tab right next to it ----
  ctx.font = TITLE_FONT;
  ctx.fillStyle = "#1d2b3a";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(title, cx + CARD_PAD_X, cy + CARD_PAD_Y + 18);
  const titleEnd = cx + CARD_PAD_X + titleW;

  if (role) {
    ctx.font = ROLE_FONT;
    const pw = roleW;
    const ph = 20;
    let px = titleEnd + 8;
    if (px + pw > cx + CARD_WIDTH - CARD_PAD_X) {
      px = cx + CARD_WIDTH - CARD_PAD_X - pw;
    }
    if (px >= titleEnd + 4) {
      const py = cy + CARD_PAD_Y + 1;
      ctx.fillStyle = "rgba(47, 111, 219, 0.14)";
      roundRect(ctx, px, py, pw, ph, 10);
      ctx.fill();
      ctx.fillStyle = "#2559b5";
      ctx.fillText(role, px + 8, py + 14);
    }
  }

  // ---- description / summary underneath ----
  if (descLines.length) {
    ctx.font = DESC_FONT;
    ctx.fillStyle = "rgba(29, 43, 58, 0.78)";
    let dy = cy + CARD_PAD_Y + headerH + 6;
    for (const line of descLines) {
      ctx.fillText(line, cx + CARD_PAD_X, dy);
      dy += DESC_LINE_H;
    }
  }

  ctx.globalAlpha = 1;
}

// (describe/drawPersonCard live above; the overlay shows only the card.)

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
