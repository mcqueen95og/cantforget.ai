/**
 * People module: registered-people list, new-face (pending) registration cards,
 * "who's here" reminder card, and add-from-photos form.
 *
 * All user-supplied text (names, tags, descriptions) is inserted with
 * textContent / DOM APIs -- never innerHTML -- so it cannot inject markup.
 */

const API = window.location.origin;

const peopleListEl = document.getElementById("peopleList");
const pendingListEl = document.getElementById("pendingList");
const pendingEmptyEl = document.getElementById("pendingEmpty");
const whoHereEl = document.getElementById("whoHere");
const addForm = document.getElementById("addPersonForm");
const addMsgEl = document.getElementById("addPersonMsg");
const toastEl = document.getElementById("toast");

const CONSENT_TEXT =
  "I have this person's permission, and the user's, to store their face data on this device for recognition.";
const PRESENCE_HOLD_MS = 5000; // keep a "who's here" entry briefly so it doesn't flicker
const PENDING_HOLD_MS = 30000; // drop an untouched new-face card this long after it was last seen

const pendingCards = new Map(); // pending_id -> {el, lastSeen, dirty}
const present = new Map(); // person_uid -> {name, tag, description, lastSeen}

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

let toastTimer = null;
function toast(message) {
  toastEl.textContent = message;
  toastEl.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.add("hidden"), 4000);
}

async function api(path, options = {}) {
  const resp = await fetch(API + path, options);
  let body = null;
  try {
    body = await resp.json();
  } catch (_) {
    /* non-JSON body */
  }
  if (!resp.ok) {
    const err = new Error((body && body.detail) || `Request failed (${resp.status})`);
    err.status = resp.status;
    throw err;
  }
  return body;
}

function jsonBody(method, obj) {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj) };
}

function labelledInput(labelText, input) {
  return el("label", { class: "field" }, el("span", {}, labelText), input);
}

/* ---------------------------------------------------------------- people */

export async function refreshPeople() {
  let data;
  try {
    data = await api("/api/people");
  } catch (err) {
    peopleListEl.replaceChildren(el("p", { class: "muted" }, err.status === 503 ? "Registration is unavailable." : "Could not load people."));
    return;
  }
  if (!data.people.length) {
    peopleListEl.replaceChildren(el("p", { class: "muted" }, "Nobody is registered yet."));
    return;
  }
  peopleListEl.replaceChildren(...data.people.map(personRow));
}

function personRow(p) {
  const row = el("div", { class: "person" });
  const view = () => {
    const thumb = p.has_thumbnail
      ? el("img", { class: "thumb", alt: "", src: `${API}/api/people/${p.uid}/thumbnail?v=${encodeURIComponent(p.updated_at)}` })
      : el("div", { class: "thumb thumb--empty" });
    const info = el(
      "div",
      { class: "person-info" },
      el("div", { class: "person-name" }, p.name),
      p.tag ? el("div", { class: "person-tag" }, p.tag) : "",
      p.description ? el("div", { class: "person-desc" }, p.description) : "",
      el("div", { class: "person-meta" }, `ID ${p.uid} · ${p.samples} sample${p.samples === 1 ? "" : "s"}`)
    );
    const actions = el(
      "div",
      { class: "person-actions" },
      el("button", { class: "btn btn--small", onclick: () => row.replaceChildren(...edit()) }, "Edit"),
      el("button", { class: "btn btn--small btn--danger", onclick: remove }, "Delete")
    );
    return [thumb, info, actions];
  };
  const edit = () => {
    const name = el("input", { type: "text", maxlength: "80", value: p.name });
    const tag = el("input", { type: "text", maxlength: "40", value: p.tag });
    const desc = el("textarea", { maxlength: "500", rows: "3" });
    desc.value = p.description;
    const msg = el("div", { class: "form-msg" });
    return [
      el(
        "div",
        { class: "person-edit" },
        labelledInput("Name", name),
        labelledInput("Tag (e.g. relationship)", tag),
        labelledInput("Short description", desc),
        msg,
        el(
          "div",
          { class: "person-actions" },
          el(
            "button",
            {
              class: "btn btn--small btn--primary",
              onclick: async () => {
                try {
                  await api(`/api/people/${p.uid}`, jsonBody("PATCH", { name: name.value, tag: tag.value, description: desc.value }));
                  refreshPeople();
                } catch (err) {
                  msg.textContent = err.message;
                }
              },
            },
            "Save"
          ),
          el("button", { class: "btn btn--small", onclick: () => row.replaceChildren(...view()) }, "Cancel")
        )
      ),
    ];
  };
  const remove = async () => {
    if (!confirm(`Delete ${p.name} and their stored face data? This cannot be undone.`)) return;
    try {
      await api(`/api/people/${p.uid}`, { method: "DELETE" });
      toast(`${p.name} was deleted.`);
      refreshPeople();
    } catch (err) {
      toast(err.message);
    }
  };
  row.replaceChildren(...view());
  return row;
}

/* ------------------------------------------------- add person from photos */

if (addForm) {
  addForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    addMsgEl.textContent = "";
    const fd = new FormData(addForm);
    fd.set("consent", addForm.elements.consent.checked ? "true" : "false");
    const button = addForm.querySelector("button[type=submit]");
    button.disabled = true;
    try {
      const data = await api("/api/people", { method: "POST", body: fd });
      toast(`${data.person.name} was registered.`);
      addForm.reset();
      if (data.notes && data.notes.length) addMsgEl.textContent = data.notes.join(" ");
      refreshPeople();
    } catch (err) {
      addMsgEl.textContent = err.message;
    } finally {
      button.disabled = false;
    }
  });
}

/* ------------------------------------------------ new faces (pending cards) */

function updatePendingEmpty() {
  if (!pendingEmptyEl) return;
  pendingEmptyEl.classList.toggle("hidden", pendingCards.size > 0);
}

function removePendingCard(id) {
  const c = pendingCards.get(id);
  if (c) c.el.remove();
  pendingCards.delete(id);
  updatePendingEmpty();
}

function addPendingCard(id) {
  const name = el("input", { type: "text", maxlength: "80", placeholder: "Name" });
  const tag = el("input", { type: "text", maxlength: "40", placeholder: "e.g. Daughter, Neighbour" });
  const desc = el("textarea", { maxlength: "500", rows: "2", placeholder: "A short, accurate description" });
  const consent = el("input", { type: "checkbox" });
  const msg = el("div", { class: "form-msg" });
  const register = el("button", { class: "btn btn--small btn--primary", disabled: "" }, "Register");
  const dismiss = el("button", { class: "btn btn--small" }, "Not now");

  const card = { el: null, lastSeen: Date.now(), dirty: false };
  const sync = () => {
    card.dirty = !!(name.value || tag.value || desc.value);
    register.disabled = !(name.value.trim() && consent.checked);
  };
  [name, tag, desc].forEach((i) => i.addEventListener("input", sync));
  consent.addEventListener("change", sync);

  register.addEventListener("click", async () => {
    register.disabled = true;
    msg.textContent = "";
    try {
      const data = await api(
        `/api/pending/${id}/register`,
        jsonBody("POST", { name: name.value, tag: tag.value, description: desc.value, consent: consent.checked })
      );
      toast(`${data.person.name} was registered.`);
      removePendingCard(id);
      refreshPeople();
    } catch (err) {
      msg.textContent = err.message;
      if (err.status === 404) removePendingCard(id);
      else sync();
    }
  });
  dismiss.addEventListener("click", async () => {
    try {
      await api(`/api/pending/${id}`, { method: "DELETE" });
    } catch (_) {
      /* already gone */
    }
    removePendingCard(id);
  });

  card.el = el(
    "div",
    { class: "pending" },
    el("img", { class: "thumb", alt: "A face the camera has not seen before", src: `${API}/api/pending/${id}/thumbnail?t=${Date.now()}` }),
    el(
      "div",
      { class: "pending-form" },
      el("div", { class: "pending-title" }, "New face seen"),
      el("p", { class: "muted" }, "Nothing is saved unless you register this person."),
      labelledInput("Name", name),
      labelledInput("Tag", tag),
      labelledInput("Short description", desc),
      el("label", { class: "consent" }, consent, el("span", {}, CONSENT_TEXT)),
      msg,
      el("div", { class: "person-actions" }, register, dismiss)
    )
  );
  pendingListEl.prepend(card.el);
  pendingCards.set(id, card);
  updatePendingEmpty();
}

/* ------------------------------------------------------ per-frame handling */

/** Called with the `faces` array from every detect response. */
export function handleFaces(faces) {
  const now = Date.now();

  for (const f of faces) {
    const idn = f.identity || {};
    if (idn.status === "new_face" && idn.pending_id) {
      const c = pendingCards.get(idn.pending_id);
      if (c) c.lastSeen = now;
      else addPendingCard(idn.pending_id);
    } else if (idn.status === "recognized") {
      present.set(idn.person_uid, { name: idn.name, tag: idn.tag, description: idn.description, lastSeen: now });
    }
  }

  for (const [id, c] of pendingCards) {
    if (!c.dirty && now - c.lastSeen > PENDING_HOLD_MS) removePendingCard(id);
  }
  renderPresence(now);
}

function renderPresence(now = Date.now()) {
  if (!whoHereEl) return;
  for (const [uid, p] of present) if (now - p.lastSeen > PRESENCE_HOLD_MS) present.delete(uid);
  if (!present.size) {
    whoHereEl.replaceChildren(el("p", { class: "muted" }, "No familiar person in view."));
    return;
  }
  whoHereEl.replaceChildren(
    ...[...present.values()].map((p) =>
      el(
        "div",
        { class: "reminder" },
        el("div", { class: "reminder-main" }, `This is ${p.name}`),
        p.tag ? el("div", { class: "reminder-tag" }, p.tag) : "",
        p.description ? el("div", { class: "reminder-desc" }, p.description) : ""
      )
    )
  );
}

export function clearPresence() {
  present.clear();
  renderPresence();
}

export function initPeople() {
  renderPresence();
  updatePendingEmpty();
  refreshPeople();
  // When someone is registered or recognised from the conversation
  // (voice self-introduction), refresh so their name + tag show up
  // immediately without a manual reload.
  window.addEventListener("cf-person-known", () => {
    refreshPeople();
  });
}
