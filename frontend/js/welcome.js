/**
 * Welcome tutorial: a multi-page popup shown every time the app opens.
 * Re-openable at any time with the "?" button. All text is inserted with
 * textContent (never innerHTML).
 */

const SCREENS = [
  {
    icon: "🌷",
    title: "Welcome!",
    lead: "You're in good hands. 💙",
    paras: [
      "I remember faces, names and conversations for you.",
    ],
    button: "Let's Begin →",
  },
  {
    title: "What it does",
    items: [
      "👀 Sees who's with you",
      "🎤 Hears your talks",
      "💙 Keeps memories safe",
    ],
    button: "Show me how →",
  },
  {
    title: "Step 1 — Camera 📷",
    lead: "Press the camera button.",
    paras: [
      "Names appear above faces. You're always in control.",
    ],
    button: "Next →",
  },
  {
    title: "Step 2 — Microphone 🎤",
    lead: "Press the microphone button.",
    paras: [
      "Spoken words appear as text, with who said them.",
    ],
    button: "Next →",
  },
  {
    title: "Step 3 — Meeting someone 🤝",
    lead: "Just say:",
    quote: "“Hello, I'm Mary, I'm your daughter.”",
    paras: [
      "The app remembers their name, face and voice by itself.",
    ],
    button: "Next →",
  },
  {
    title: "Step 4 — The menu ☰",
    lead: "Open the menu (☰) to find:",
    items: [
      "👥 People — everyone you've met",
      "💬 Conversation — your talks, written down",
    ],
    button: "Next →",
  },
  {
    icon: "🌻",
    title: "That's everything!",
    paras: [
      "Press ? anytime to see this again.",
      "Enjoy your day 💙",
    ],
    button: "Start Using It →",
    last: true,
  },
];

const root = document.getElementById("welcome");
const body = document.getElementById("welcomeBody");
const dots = document.getElementById("welcomeDots");
const nextBtn = document.getElementById("welcomeNext");
const backBtn = document.getElementById("welcomeBack");
const againBtn = document.getElementById("welcomeAgain");
const closeBtn = document.getElementById("welcomeClose");
const helpBtn = document.getElementById("helpBtn");

let index = 0;
let opener = null;

function node(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function render() {
  const s = SCREENS[index];
  const frag = document.createDocumentFragment();
  if (s.icon) frag.append(node("div", "welcome-icon", s.icon));
  const h = node("h2", "welcome-title", s.title);
  h.id = "welcomeTitle";
  frag.append(h);
  if (s.lead) frag.append(node("p", "welcome-lead", s.lead));
  if (s.quote) frag.append(node("p", "welcome-quote", s.quote));
  if (s.items) {
    const ul = node("ul", "welcome-list");
    for (const it of s.items) ul.append(node("li", null, it));
    frag.append(ul);
  }
  for (const p of s.paras || []) frag.append(node("p", "welcome-para", p));
  body.replaceChildren(frag);
  body.classList.remove("welcome-body--in");
  void body.offsetWidth; // restart the entrance animation
  body.classList.add("welcome-body--in");

  dots.replaceChildren(
    ...SCREENS.map((_, i) => node("span", "welcome-dot" + (i === index ? " welcome-dot--on" : "")))
  );
  nextBtn.textContent = s.button;
  nextBtn.classList.toggle("welcome-next--big", !!s.last);
  backBtn.style.visibility = index === 0 ? "hidden" : "visible";
  againBtn.hidden = !s.last;
  closeBtn.style.visibility = s.last ? "hidden" : "visible";
}

export function openWelcome(from = 0) {
  opener = document.activeElement;
  index = from;
  render();
  root.hidden = false;
  nextBtn.focus();
}

function closeWelcome() {
  root.hidden = true;
  if (opener && typeof opener.focus === "function") opener.focus();
}

function next() {
  if (SCREENS[index].last) closeWelcome();
  else {
    index += 1;
    render();
  }
}

function back() {
  if (index > 0) {
    index -= 1;
    render();
  }
}

nextBtn.addEventListener("click", next);
backBtn.addEventListener("click", back);
closeBtn.addEventListener("click", closeWelcome);
againBtn.addEventListener("click", () => {
  index = 0;
  render();
  nextBtn.focus();
});
if (helpBtn) helpBtn.addEventListener("click", () => openWelcome(0));

document.addEventListener("keydown", (ev) => {
  if (root.hidden) return;
  if (ev.key === "Escape") closeWelcome();
  else if (ev.key === "ArrowRight") next();
  else if (ev.key === "ArrowLeft") back();
  else if (ev.key === "Tab") {
    // keep keyboard focus inside the popup
    const f = [...root.querySelectorAll("button")].filter((b) => !b.hidden && b.style.visibility !== "hidden");
    const first = f[0];
    const last = f[f.length - 1];
    if (ev.shiftKey && document.activeElement === first) { ev.preventDefault(); last.focus(); }
    else if (!ev.shiftKey && document.activeElement === last) { ev.preventDefault(); first.focus(); }
  }
});

// Every time the app opens.
openWelcome(0);
