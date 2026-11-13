/**
 * Conversation Intelligence frontend — Phase 4.
 *
 * Displays conversation windows, structured summaries, extracted information
 * awaiting approval, and participant memory records. Provides approve/reject/
 * correct controls for extracted records and edit/delete for participant
 * memories.
 *
 * All user-supplied text is inserted via textContent / DOM APIs — never
 * innerHTML — to prevent markup injection.
 */

const API = window.location.origin;

const processBtn = document.getElementById("processConvBtn");
const convWindowsEl = document.getElementById("convWindowsList");
const convSummariesEl = document.getElementById("convSummariesList");
const pendingExtractedEl = document.getElementById("pendingExtractedList");
const participantMemoriesEl = document.getElementById("participantMemoriesList");

let currentSessionId = null;
let lastSessionId = null;  // Persisted across mic start/stop

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  for (const c of children) if (c !== null && c !== undefined) node.append(c);
  return node;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

async function api(path, options = {}) {
  const resp = await fetch(API + path, options);
  let body = null;
  try { body = await resp.json(); } catch (_) {}
  if (!resp.ok) {
    const err = new Error((body && body.detail) || `Request failed (${resp.status})`);
    err.status = resp.status;
    throw err;
  }
  return body;
}

export function setIntelligenceSession(sid) {
  currentSessionId = sid;
  lastSessionId = sid;  // Persist so processing works after mic stops
  if (processBtn) processBtn.disabled = !sid;
  refreshAll();
}

export function clearIntelligenceSession() {
  currentSessionId = null;
  if (processBtn) processBtn.disabled = true;
  if (convWindowsEl) convWindowsEl.replaceChildren(
    el("div", { class: "conv-windows-empty" }, "No conversation session active.")
  );
  if (convSummariesEl) convSummariesEl.replaceChildren(
    el("div", { class: "conv-summaries-empty" }, "No session active.")
  );
  if (pendingExtractedEl) pendingExtractedEl.replaceChildren(
    el("div", { class: "pending-extracted-empty" }, "No session active.")
  );
  if (participantMemoriesEl) participantMemoriesEl.replaceChildren(
    el("div", { class: "participant-memories-empty" }, "No session active.")
  );
}

function getEffectiveSessionId() {
  return currentSessionId || lastSessionId;
}

async function refreshAll() {
  const sid = getEffectiveSessionId();
  if (!sid) return;
  await Promise.all([
    renderConvWindows(),
    renderConvSummaries(),
    renderPendingExtracted(),
    renderParticipantMemories(),
  ]);
}

async function renderConvWindows() {
  if (!convWindowsEl) return;
  try {
    const data = await api(`/api/intelligence/session/${encodeURIComponent(currentSessionId)}/windows`);
    convWindowsEl.replaceChildren();
    if (!data.windows.length) {
      convWindowsEl.appendChild(el("div", { class: "conv-windows-empty" }, "No conversation windows yet."));
      return;
    }
    for (const w of data.windows) {
      const statusCls = w.status === "completed" ? "completed" : "open";
      const item = el(
        "div", { class: "conv-window" },
        el("div", { class: "conv-window-header" },
          el("span", null, `Window ${w.window_index + 1}`),
          el("span", { class: `conv-window-status ${statusCls}` }, w.status),
          el("span", null, `${w.segment_count || 0} segments`)
        ),
        el("div", null, `Started: ${(w.started_at || "").slice(0, 16)}` + (w.ended_at ? ` · Ended: ${w.ended_at.slice(0, 16)}` : ""))
      );
      convWindowsEl.appendChild(item);
    }
  } catch (err) {
    convWindowsEl.replaceChildren(el("div", { class: "conv-windows-empty" }, "Failed to load windows."));
    console.error("[intelligence] windows error:", err);
  }
}

async function renderConvSummaries() {
  if (!convSummariesEl) return;
  try {
    const data = await api(`/api/intelligence/session/${encodeURIComponent(currentSessionId)}/summaries`);
    convSummariesEl.replaceChildren();
    if (!data.summaries.length) {
      convSummariesEl.appendChild(el("div", { class: "conv-summaries-empty" }, "No summaries yet."));
      return;
    }
    for (const s of data.summaries) {
      const personName = s.person_name || s.speaker_id || "Unknown speaker";
      const item = el(
        "div", { class: "conv-summary" },
        el("div", { class: "conv-summary-meta" },
          el("span", null, personName),
          el("span", null, `· model: ${s.model_version || "extractive"}`)
        ),
        el("div", { class: "conv-summary-text" }, escapeHtml(s.summary_text || "")),
        s.topics && s.topics.length ? el(
          "ul", { class: "conv-summary-topics" },
          ...s.topics.map(t => el("li", null, t))
        ) : null,
        s.decisions && s.decisions.length ? el(
          "ul", { class: "conv-summary-decisions" },
          ...s.decisions.map(d => el("li", null, d))
        ) : null
      );
      convSummariesEl.appendChild(item);
    }
  } catch (err) {
    convSummariesEl.replaceChildren(el("div", { class: "conv-summaries-empty" }, "Failed to load summaries."));
    console.error("[intelligence] summaries error:", err);
  }
}

async function renderPendingExtracted() {
  if (!pendingExtractedEl) return;
  try {
    const data = await api(`/api/intelligence/session/${encodeURIComponent(currentSessionId)}/extracted?status=pending`);
    pendingExtractedEl.replaceChildren();
    if (!data.records.length) {
      pendingExtractedEl.appendChild(el("div", { class: "pending-extracted-empty" }, "No information awaiting approval."));
      return;
    }
    for (const r of data.records) {
      const personName = r.person_name || r.speaker_id || "Unknown speaker";
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
                await api(`/api/intelligence/extracted/${encodeURIComponent(r.id)}/approve`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ corrected_info: corrEl.value.trim() || null }),
                });
                await refreshAll();
              } catch (err) {
                alert("Failed to approve: " + err.message);
              }
            }
          }, "Approve"),
          el("button", {
            class: "btn btn--small btn--danger",
            onclick: async () => {
              try {
                await api(`/api/intelligence/extracted/${encodeURIComponent(r.id)}/reject`, { method: "POST" });
                await refreshAll();
              } catch (err) {
                alert("Failed to reject: " + err.message);
              }
            }
          }, "Reject")
        )
      );
      pendingExtractedEl.appendChild(item);
    }
  } catch (err) {
    pendingExtractedEl.replaceChildren(el("div", { class: "pending-extracted-empty" }, "Failed to load pending records."));
    console.error("[intelligence] pending error:", err);
  }
}

async function renderParticipantMemories() {
  if (!participantMemoriesEl) return;
  try {
    // We don't know all participant UIDs from this endpoint alone, so show
    // a placeholder. In a fuller UI this would list all persons with memories.
    participantMemoriesEl.replaceChildren(
      el("div", { class: "participant-memories-empty" },
        "Participant memories appear here after approval. Use the people list to view a person, then their memories are shown on their profile.")
    );
  } catch (err) {
    console.error("[intelligence] memories error:", err);
  }
}

if (processBtn) {
  processBtn.addEventListener("click", async () => {
    const sid = getEffectiveSessionId();
    if (!sid) return;
    processBtn.disabled = true;
    processBtn.textContent = "Processing…";
    try {
      await api(`/api/intelligence/session/${encodeURIComponent(sid)}/process`, { method: "POST" });
      await refreshAll();
    } catch (err) {
      alert("Failed to process conversation: " + err.message);
    } finally {
      processBtn.disabled = false;
      processBtn.textContent = "Process Conversation";
    }
  });
}
