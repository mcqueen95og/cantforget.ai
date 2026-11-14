/**
 * Conversation module — displays SSE transcription events in a clean panel.
 */

const BACKEND_URL = window.location.origin;
const SSE_URL = `${BACKEND_URL}/api/voice/events`;

let eventSource = null;
let sessionId = null;
let lastSessionId = null;  // Persisted across mic start/stop so summarize works after stopping
let isConnected = false;

const transcriptEl = document.getElementById("transcriptList");
const speakerStatsEl = document.getElementById("speakerStats");
const keyPointsEl = document.getElementById("keyPointsList");

const speakerStats = new Map();
const liveSummaries = new Map();
// speaker_id -> DOM nodes of transcript entries shown with a temporary
// label, so they can be relabelled once the voice is linked to a person.
const speakerEntries = new Map();

/** Turn backend labels into readable ones ("SPEAKER_01" -> "Speaker 2"). */
function friendlySpeaker(speaker) {
  if (!speaker || speaker === "SPEAKER_UNKNOWN") return "Unknown voice";
  const m = /^SPEAKER_(\d+)$/.exec(speaker);
  if (m) {
    const n = parseInt(m[1], 10) + 1;
    return `Speaker ${isNaN(n) ? m[1] : n}`;
  }
  return speaker;
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

export function setVoiceConnectionStatus(text, kind = "idle") {}

export function startVoiceSession(sid) {
  sessionId = sid;
  lastSessionId = sid;  // Persist so summarize works after mic stops
  if (eventSource) eventSource.close();

  const url = `${SSE_URL}?session_id=${encodeURIComponent(sid)}`;
  console.log("[conversation] SSE:", url);
  eventSource = new EventSource(url);

  eventSource.onopen = () => {
    isConnected = true;
    console.log("[conversation] Connected");
  };

  eventSource.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      console.log("[conversation] SSE event:", data.event_type, data);
      if (data.event_type === "transcription") {
        handleTranscriptionEvent(data);
      } else if (data.event_type === "summary") {
        handleSummaryEvent(data);
      } else if (data.event_type === "status") {
        setVoiceConnectionStatus(data.message || "", data.kind || "idle");
      } else if (data.event_type === "error") {
        console.error("[conversation] Pipeline error:", data.message);
        setVoiceConnectionStatus(data.message || "Pipeline error", "error");
      }
    } catch (err) {
      console.error("[conversation] Parse error:", err);
    }
  };

  eventSource.onerror = () => {
    isConnected = false;
    console.warn("[conversation] SSE connection dropped; browser will retry");
  };
}

export function stopVoiceSession() {
  if (eventSource) { eventSource.close(); eventSource = null; }
  isConnected = false;
  sessionId = null;
}

export function isVoiceConnected() { return isConnected; }

export async function summarizeSession(sid) {
  // Use passed sid, or fall back to last persisted session ID (in case mic
  // was already stopped before clicking summarize).
  const sessionIdToUse = sid || lastSessionId;
  if (!sessionIdToUse) {
    console.error("[conversation] No session ID available for summarization");
    return;
  }
  try {
    // Phase 4: Process conversation through intelligence engine
    // (summarization + extraction + approval workflow)
    const resp = await fetch(`${BACKEND_URL}/api/intelligence/session/${encodeURIComponent(sessionIdToUse)}/process`, {
      method: "POST",
    });
    if (!resp.ok) throw new Error(`Process failed (${resp.status})`);
    const data = await resp.json();
    console.log("[conversation] Process result:", data);
    // Load both summaries and pending extracted info for approval
    await loadSummaries(sessionIdToUse);
    await loadPendingExtracted(sessionIdToUse);
  } catch (err) {
    console.error("[conversation] Process error:", err);
  }
}

export async function loadSummaries(sid) {
  if (!sid || !keyPointsEl) return;
  try {
    const resp = await fetch(`${BACKEND_URL}/api/intelligence/session/${encodeURIComponent(sid)}/summaries`);
    const data = await resp.json();
    renderSummaries(data.summaries || []);
  } catch (err) {
    console.error("[conversation] Load summaries error:", err);
  }
}

export async function loadPendingExtracted(sid) {
  if (!sid) return;
  try {
    const resp = await fetch(`${BACKEND_URL}/api/intelligence/session/${encodeURIComponent(sid)}/extracted?status=pending`);
    const data = await resp.json();
    renderPendingExtracted(data.records || []);
  } catch (err) {
    console.error("[conversation] Load pending extracted error:", err);
  }
}

function renderPendingExtracted(records) {
  const container = document.getElementById("pendingExtractedList");
  if (!container) return;
  container.replaceChildren();
  if (!records.length) {
    container.appendChild(el("div", { class: "pending-extracted-empty" }, "No information awaiting approval."));
    return;
  }
  for (const r of records) {
    const personName = r.person_name || friendlySpeaker(r.speaker_id) || "Unknown speaker";
    const correctionId = `corr-${r.id}`;
    const item = el(
      "div", { class: "pending-record" },
      el("div", { class: "pending-record-statement" }, `"${escapeHtml(r.original_statement)}"`),
      el("div", { class: "pending-record-meta" },
        el("span", { class: "pending-record-category" }, r.category),
        el("span", null, `· ${personName}`),
        el("span", null, `· ${r.model_version || "extractive"}`)
      ),
      el("div", { class: "pending-record-actions" },
        el("input", {
          class: "pending-record-correction",
          type: "text",
          id: correctionId,
          placeholder: "Optional correction",
          value: r.corrected_info || ""
        }),
        el("button", {
          class: "btn btn--small btn--primary",
          onclick: async () => {
            const corrEl = document.getElementById(correctionId);
            try {
              await fetch(`${BACKEND_URL}/api/intelligence/extracted/${encodeURIComponent(r.id)}/approve`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ corrected_info: corrEl.value.trim() || null }),
              });
              await loadPendingExtracted(sid);
              await loadSummaries(sid);
            } catch (err) {
              alert("Failed to approve: " + err.message);
            }
          }
        }, "Approve"),
        el("button", {
          class: "btn btn--small btn--danger",
          onclick: async () => {
            try {
              await fetch(`${BACKEND_URL}/api/intelligence/extracted/${encodeURIComponent(r.id)}/reject`, { method: "POST" });
              await loadPendingExtracted(sid);
            } catch (err) {
              alert("Failed to reject: " + err.message);
            }
          }
        }, "Reject")
      )
    );
    container.appendChild(item);
  }
}

function renderSummaries(summaries) {
  // Load saved summaries into the same map used for live updates, so a later
  // SSE summary merges with them instead of replacing the whole panel.
  liveSummaries.clear();
  for (const s of summaries) {
    const name = s.person_name || friendlySpeaker(s.speaker_id) || "Unknown speaker";
    const key = s.identity_uid || s.speaker_id || name;
    let entry = liveSummaries.get(key);
    if (!entry) {
      entry = { name, isUnknown: !s.identity_uid, points: [] };
      liveSummaries.set(key, entry);
    }
    for (const p of s.key_points || []) {
      if (p && !entry.points.includes(p)) entry.points.push(p);
    }
  }
  renderLiveSummaries();
}

function handleSummaryEvent(data) {
  if (!keyPointsEl) return;
  const key = data.identity_id || data.speaker_id || "unknown";
  const name = data.identity_name || friendlySpeaker(data.speaker_id) || "Unknown speaker";
  const isUnknown = !data.identity_id;
  if (data.identity_id && data.identity_name) {
    window.dispatchEvent(new CustomEvent("cf-person-known", {
      detail: { identityId: data.identity_id, identityName: data.identity_name }
    }));
  }
  let entry = liveSummaries.get(key);
  if (!entry) {
    entry = { name, isUnknown, points: [] };
    liveSummaries.set(key, entry);
  }
  entry.name = name;
  entry.isUnknown = isUnknown;
  for (const p of data.key_points || []) {
    if (p && !entry.points.includes(p)) entry.points.push(p);
  }
  // Keep the panel small and gentle: only the latest few points per person.
  entry.points = entry.points.slice(-4);
  renderLiveSummaries();
}

function renderLiveSummaries() {
  keyPointsEl.replaceChildren();
  if (!liveSummaries.size) {
    keyPointsEl.appendChild(el("div", { class: "key-points-empty" }, "No key points yet."));
    return;
  }
  for (const entry of liveSummaries.values()) {
    keyPointsEl.appendChild(
      el(
        "div",
        { class: "key-point-group" + (entry.isUnknown ? " unknown" : "") },
        el(
          "div",
          { class: "key-point-person" + (entry.isUnknown ? " unknown" : "") },
          entry.name
        ),
        el("ul", { class: "key-points" }, ...entry.points.slice(-4).map((p) => el("li", {}, p)))
      )
    );
  }
}

function handleTranscriptionEvent(data) {
  const speaker = data.speaker_id || "SPEAKER_UNKNOWN";
  const identityName = data.identity_name || null;
  const text = data.text || "";
  const associated = data.associated || false;
  const asrEmpty = data.asr_empty || false;

  // Log but don't display empty ASR segments (inaudible / failed transcription).
  if (asrEmpty && !text) {
    console.log("[conversation] ASR-empty segment:", speaker, data.start_ms, "-", data.end_ms);
    return;
  }

  const wasKnown = associated && !!identityName;
  updateSpeakerStats(speaker, identityName, associated);

  // A person just got registered/recognised from this very sentence:
  // refresh the People tab so the new name + tag appear immediately.
  if (wasKnown) {
    window.dispatchEvent(new CustomEvent("cf-person-known", {
      detail: { identityId: data.identity_id, identityName }
    }));
  }

  const entry = document.createElement("div");
  entry.className = `transcript-entry ${associated ? "associated" : "unassociated"}`;
  const speakerLabel = wasKnown ? identityName : friendlySpeaker(speaker);
  const speakerClass = associated ? "speaker-known" : "speaker-unknown";

  entry.innerHTML = `
    <div class="transcript-meta">
      <span class="speaker ${speakerClass}">${escapeHtml(speakerLabel)}</span>
      ${wasKnown ? '<span class="badge badge--known">Verified</span>' : '<span class="badge badge--unknown">Unknown</span>'}
    </div>
    <div class="transcript-text">${escapeHtml(text)}</div>
  `;

  if (transcriptEl) {
    const empty = transcriptEl.querySelector(".transcript-empty");
    if (empty) empty.remove();
    transcriptEl.prepend(entry);
  }

  // Track this entry so a later "this voice is actually X" event can
  // relabel the earlier lines in place.
  let list = speakerEntries.get(speaker);
  if (!list) { list = []; speakerEntries.set(speaker, list); }
  list.unshift({ entry, labelEl: entry.querySelector(".speaker"), badgeEl: entry.querySelector(".badge") });
  if (list.length > 60) list.length = 60;

  if (wasKnown) relabelSpeakerEntries(speaker, identityName);
}

/** Retro-actively name a speaker's earlier transcript lines. */
function relabelSpeakerEntries(speaker, identityName) {
  const list = speakerEntries.get(speaker);
  if (!list || !identityName) return;
  for (const item of list) {
    if (item.labelEl) item.labelEl.textContent = identityName;
    if (item.labelEl) item.labelEl.classList.remove("speaker-unknown");
    if (item.labelEl) item.labelEl.classList.add("speaker-known");
    if (item.badgeEl) item.badgeEl.outerHTML = '<span class="badge badge--known">Verified</span>';
    item.entry.classList.remove("unassociated");
    item.entry.classList.add("associated");
  }
  // The same voice under a fresh temporary label (e.g. after a session
  // restart) should also adopt the known name on its next line.
  const stats = speakerStats.get(speaker);
  if (stats) { stats.name = identityName; stats.associated = true; }
}

function updateSpeakerStats(speaker, identityName, associated) {
  if (!speakerStatsEl) return;
  let stats = speakerStats.get(speaker);
  if (!stats) {
    stats = { name: identityName || speaker, segments: 0, associated: false };
    speakerStats.set(speaker, stats);
  }
  stats.segments++;
  if (associated && identityName) { stats.name = identityName; stats.associated = true; }

  speakerStatsEl.innerHTML = "";
  for (const [spk, st] of speakerStats) {
    const row = document.createElement("div");
    row.className = `speaker-stat ${st.associated ? "known" : "unknown"}`;
    row.innerHTML = `
      <span class="speaker-name">${escapeHtml(st.name)}</span>
      <span class="segment-count">${st.segments} segment${st.segments !== 1 ? "s" : ""}</span>
    `;
    speakerStatsEl.appendChild(row);
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
