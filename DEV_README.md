# cantforget.ai — Developer Guide

This is the technical companion to `README.md` (which is written for
non-technical users). It explains how the system actually works: the
architecture, the processing pipelines, the data model, and the operational
details you need when developing or debugging.

---

## 1. What the product does

A local, privacy-first web app for people with memory difficulties:

1. **Face recognition** — the camera feed is processed continuously; known
   people are shown with a name card above their face.
2. **Live speech-to-text** — microphone audio is streamed to the backend,
   transcribed with Whisper, and shown sentence-by-sentence in a panel.
3. **Speaker diarisation** — each sentence gets a speaker label
   (`Speaker 1`, `Speaker 2`, …) from voice embeddings.
4. **Automatic person registration** — when someone says a self-introduction
   ("hey, I'm Mary, I'm your daughter"), the app extracts name + relationship,
   snapshots their face, enrols their voice, and creates a persistent person.
   Afterwards both face and voice are recognised.
5. **Conversation intelligence** — transcript segments are grouped into
   windows, summarised per window and per person; salient facts (plans, phone
   numbers, etc.) are extracted as candidate records that require explicit
   user approval before being stored.
6. **Cumulative person descriptions** — everything a person states about
   themselves ("I'm a mathematician and the son of Watkins") is captured as a
   third-person fact and accumulated in a per-person ledger; the profile
   description shown on the video card and in the People tab is regenerated
   from that ledger as bullet points and grows with every conversation.

Everything runs on `127.0.0.1`; no data leaves the machine.

---

## 2. Architecture at a glance

```
Browser (frontend/)                         Backend (FastAPI, uvicorn)
┌────────────────────────┐                   ┌─────────────────────────────────┐
│ camera.js              │  JPEG frames      │ /api/faces/detect               │
│  → captureFrame()      │ ────────────────► │  detector (YuNet)               │
│ face-overlay.js        │  ~150 ms cadence  │  tracker (IoU/distance)         │
│  (canvas overlay,      │ ◄──────────────── │  identity (SFace + matcher)     │
│   person cards)        │  tracks+identity  │   → pending store for new faces │
├────────────────────────┤                   ├─────────────────────────────────┤
│ audio.js               │  float32 PCM      │ /api/audio/stream               │
│  (ScriptProcessorNode, │ ────────────────► │  AudioPipeline (per session)    │
│   16 kHz, ~300 ms POST │                   │   VAD → ASR → diarize →         │
│   chunks)              │                   │   associate → persist → summary │
│ conversation.js        │  SSE events       │ /api/voice/events (SSE)         │
│  (transcript panel)    │ ◄──────────────── │  (event bus fan-out)            │
├────────────────────────┤                   ├─────────────────────────────────┤
│ conversation-          │  REST             │ /api/intelligence/...           │
│   intelligence.js      │ ────────────────► │  ConversationIntelligenceEngine │
│  (windows, summaries,  │                   │  (windows → summaries →         │
│   approvals)           │                   │   extracted records → approval) │
└────────────────────────┘                   └─────────────────────────────────┘
                                                      │
                                              ┌───────▼────────┐
                                              │ SQLite database │
                                              │ data/cantforget.db
                                              └────────────────┘
```

Two independent, concurrent loops exist on the server:

- the **face path**: `POST /api/faces/detect` per frame (stateless per request,
  stateful tracker/identity inside `svc`), and
- the **audio path**: one `AudioPipeline` task per session, fed by audio POSTs,
  running its own `_process_loop` coroutine.

They meet in two places: voice-to-face **association** (the pipeline reads the
latest face tracks to link a speaker with a face) and the **database**.

---

## 3. Tech stack & models

| Layer | Technology | Model file (data/models/) | Notes |
|---|---|---|---|
| Web framework | FastAPI + Uvicorn | — | async endpoints, SSE via `StreamingResponse` |
| DB | SQLite (stdlib `sqlite3`) | `data/cantforget.db` | single writer, thread-locked (`db.py`) |
| Face detection | OpenCV DNN | `face_detection_yunet.onnx` | YuNet, 320×320 input, score ≥ 0.6 |
| Face recognition | OpenCV DNN | `face_recognition_sface_2021dec.onnx` | SFace, 128-d embeddings, cosine similarity |
| Face tracking | custom `faces/tracker.py` | — | centroid + IoU matching, coasting tracks |
| VAD | Silero (ONNX via `sherpa-onnx`… see `vad.py`) | `silero_vad.onnx` | threshold 0.35, min speech 200 ms |
| Speech-to-text | `faster-whisper` (CTranslate2) | auto-downloaded `tiny` | int8, CPU, beam_size 1, 2 threads |
| Speaker embedding | 3D-Speaker ERes2Net (ONNX) | `speaker/eres2net.onnx` | 512-d embeddings, centroid clustering |
| Summarisation | extractive (built-in) or FLAN-T5 | `flan-t5-small/` (optional) | `SUMMARIZER_USE_MODEL=1` enables the model |
| Conversation intelligence | LLM-with-extractive-fallback (`intelligence.py`) | — | works fully offline without an LLM |
| Frontend | vanilla ES modules, no build step | — | `index.html` + `js/*` + `css/main.css` |

`requirements.txt` covers the Python side; `requirements-dev.txt` adds pytest.

### Library map (what each dependency is for)

| Library | Used for | Where |
|---|---|---|
| `fastapi` | HTTP endpoints, SSE, request/file handling | `main.py` |
| `uvicorn[standard]` | ASGI server that runs the FastAPI app | launch command |
| `opencv-python-headless` | image decode, ONNX DNN face models (YuNet/SFace) | `faces/*`, `main.py` |
| `numpy` | audio buffers, embeddings, all array math | everywhere |
| `python-multipart` | file-upload form parsing (`UploadFile`) | `/api/faces/detect`, `/api/people` |
| `faster-whisper` | CTranslate2 Whisper speech-to-text | `audio/asr.py` |
| `sherpa-onnx` | ONNX speaker-embedding model runtime (ERes2Net) | `audio/diarize.py`, `audio/enrollment.py` |
| `silero-vad` | voice-activity-detection model + runtime | `audio/vad.py` |
| `torch` | runtime for the FLAN-T5 description/summary model | `audio/summarize.py` |
| `transformers` | loads FLAN-T5 tokenizer + model (also pulls `huggingface_hub`, used for model download) | `audio/summarize.py` |
| `sentencepiece` | FLAN-T5 tokenizer backend | loaded automatically |
| `pytest` (dev) | test runner | `tests/` |

Everything runs CPU-only; no CUDA/GPU is required or used.

---

## 4. The audio pipeline in detail (`backend/app/audio/pipeline.py`)

One `AudioPipeline` per session id, created lazily by
`_get_or_create_pipeline()` in `main.py` on the first audio POST.

### 4.1 Ingestion

- The browser captures float32 PCM at 16 kHz mono (Web Audio
  `ScriptProcessorNode`, 4096-sample buffers, downsampled in `audio.js`) and
  POSTs ~300 ms binary chunks to `/api/audio/stream?session_id=sess_xxx`.
- `feed_audio()` appends to `self._buf` (a numpy ring-ish buffer, hard cap
  `MAX_BUFFER_SECONDS = 30`) and timestamps `_last_audio`.

### 4.2 Processing loop

`_process_loop` polls every `POLL_SECONDS = 0.15 s` and calls `_process_once`:

1. **Buffering**: wait until ≥ `MIN_BUFFER_SECONDS (0.3 s)` is buffered.
2. **Sender-pause flush**: if no audio for 0.75 s, flush the tail
   (`_flush_tail`) — this is what emits the final sentence after you stop
   talking.
3. **VAD throttling**: run Silero VAD at most every 0.25 s
   (`VAD_MIN_INTERVAL_SECONDS`), via `asyncio.to_thread`.
4. **Segmentation**: VAD speech regions become `(start, end)` sample ranges;
   very long continuous speech is force-split at `MAX_SEGMENT_SECONDS = 6 s`.
   If VAD finds nothing but RMS is loud (`SILENCE_RMS = 0.006`,
   `FALLBACK_PEAK = 0.05`), an energy-based fallback segmenter is used.
   Pure silence advances the buffer cursor (`SILENCE_ADVANCE_SECONDS = 1.2 s`)
   so memory stays bounded.
5. **Per segment → `_process_segment`**:
   - **ASR**: `await asyncio.to_thread(self._asr.transcribe_segments, ...)`
     — Whisper returns *sentences*, each with timestamps, language and
     confidence. Low-confidence output is dropped (`no_speech_prob > 0.6 &&
     avg_logprob < −0.7`, or `avg_logprob < −1.2`), then `_is_artifact()`
     filters known Whisper hallucinations on silence ("thank you",
     "subscribe", lone filler words, …).
   - **Per-sentence diarisation**: the sentence's audio slice is embedded by
     ERes2Net (`to_thread`) and clustered against previous speakers
     (cosine ≥ `DIARIZATION_THRESHOLD = 0.5`) → `SPEAKER_00`, `SPEAKER_01`, …
     Each sentence gets its own label.
   - **Speaker-change splitting**: if one "sentence" contains a speaker change
     (embedding distance spike), it is split at the change point and each half
     re-processed (depth-guarded recursion).
   - **Voice ↔ face association** (`associate.py`): the pipeline reads the
     latest face tracks (with embeddings + crops) pushed by
     `/api/faces/detect`; a voice within `VOICE_ASSOC_MAX_OFFSET_MS = 1000`
     of a single visible face gets associated; after
     `VOICE_ASSOC_CONFIRM_FRAMES = 3` agreements the speaker is linked to that
     identity persistently (`db.link_speaker_to_identity`, upsert).
   - **Self-intro registration** (`summarize.py: extract_person_intro`):
     case-insensitive regexes over the sentence text detect
     "I'm/I am/my name is \<name\>" plus an optional role
     ("your son/daughter/friend …"). On a match the pipeline enrols the
     speaker's voice (`enrollment.py`, `VOICE_ENROLL_MIN_SAMPLES = 3`) and the
     face crop, creates the person (`db.create_person`, consent implied by the
     introduction flow), attaches the role tag, and calls
     `pipeline.on_person_created` (= `identity.reload`) so the face matcher
     gallery includes the new person immediately.
   - **Persist + publish**: the sentence is stored in
     `transcription_segments` and pushed onto the SSE event bus with
     `{type: "segment", speaker_id, identity_uid, identity_name, text,
     start_ms, end_ms, language, associated, asr_empty, segment_id}`.

### 4.3 Summaries

`_maybe_schedule_summary()` batches finalised sentences; when
≥ `SUMMARY_MIN_SEGMENTS (4)` segments spanning ≥ `SUMMARY_MIN_SPAN_MS (15 s)`
(or ≥ `SUMMARY_MAX_SEGMENTS (12)`) are pending, `_run_summary` runs
`_flush_summary` in the background: utterances are joined per speaker and
summarised by `LocalModelSummarizer` (FLAN-T5 if `SUMMARIZER_USE_MODEL=1` and
the model exists, otherwise instant extractive) plus
`PersonDescriptionWriter`, which keeps a rolling per-person description in the
DB. Everything heavy runs through `asyncio.to_thread`.

### 4.4 Threading / event-loop model (important)

- **All** CPU-heavy steps (VAD, Whisper, ERes2Net, summarisation, and — since
  the responsiveness fix — face detect/track/recognise in `main.py`) run in
  worker threads via `asyncio.to_thread`, guarded where state is shared
  (`_detect_lock` serialises the face path).
- **Never** add a blocking ONNX/torch/whisper call directly inside an `async
  def` handler or the pipeline loop: this machine has 2 cores, and blocking
  the event loop starves every other endpoint (this caused the "face boxes
  freeze while speaking" bug).
- The SQLite connection is guarded by a `threading.Lock` inside `db.py`;
  all writes go through `PeopleDB` methods.

### 4.5 SSE event bus (`events.py`)

`ConversationEventBus` keeps `dict[session_id → set[asyncio.Queue]]`.
`GET /api/voice/events?session_id=...` subscribes and streams JSON lines;
keep-alive comments are sent every 30 s; queue cap is 100 events (oldest
dropped with a warning). The frontend `conversation.js` consumes this and
renders transcript rows + speaker stats; `conversation-intelligence.js`
tracks the session for the intelligence panel.

### 4.6 The person-description engine (fact ledger)

The profile description is **not** a paragraph written once — it is a list of
facts that grows with every conversation. Data flow:

```
finalised sentence ──► extract_self_claims()        deterministic, every sentence
                            │  (regex claim extractor, third-person facts)
summary batch      ──► PersonDescriptionWriter.write_facts()
                            │  model FIRST (FLAN-T5), extractor always as well
                            ▼
                    db.add_person_fact(uid, fact, source)   ← person_facts table
                            │  exact-duplicate check; near-dup check at extraction
                            ▼
                    _refresh_person_description(uid)
                            │  render_description(): "- He is a mathematician"
                            ▼
                    persons.description (bullets, ≤8 facts, ≤460 chars)
                            ▼
                    identity.reload()  → video card + /api/faces/detect
```

- **Claim extractor** (`summarize.py: extract_self_claims`): splits compound
  first-person sentences at clause boundaries ("I'm a mathematician and I
  have dementia" → two facts), classifies each predicate fragment
  (role-of, "your <role>", article-noun, from/place, age, bare
  name/state) and converts dedicated patterns (`I work as`, `I live in`,
  `I like/love/enjoy`, `I have`). Filler/junk/second-person fragments are
  rejected; the pronoun ("He"/"She") is chosen from the person's
  relationship tag.
- **Model path** (`PersonDescriptionWriter.write_facts`): FLAN-T5 is
  prompted for third-person fact lines; output is validated (length,
  no first/second person) and near-duplicate-checked against the ledger
  *and* against already-accepted candidates from the same batch, so model
  variants collapse into the deterministic facts. Known caveat: the small
  model occasionally **invents** details — test with real speech and set
  `DESCRIPTION_USE_MODEL=0` if invented facts outweigh the good ones. The
  deterministic extractor is always applied too, so the description works
  correctly even with the model disabled.
- **Ledger** (`person_facts` table): `person_uid, fact, source
  ('model'|'claim'|'intro'), session_id, created_at`, exact-duplicate-proof,
  cascading delete with the person. `add_person_fact` / `list_person_facts`
  (limit keeps the most recent) / `count_person_facts` in `db.py`.
- **Rendering**: `render_description()` joins bullets oldest-first; the
  card (`face-overlay.js: wrapFactLines`) renders one bullet per line
  (≤6 lines), the People tab uses `white-space: pre-line`.
- **Config**: `DESCRIPTION_USE_MODEL` (default 1), `DESCRIPTION_MAX_FACTS`
  (8), `DESCRIPTION_MAX_CHARS` (460).

### 4.7 Face-path responsiveness (i3 constraint)

`POST /api/faces/detect` runs YuNet detection, tracking and SFace
recognition **inside `asyncio.to_thread` under an `_detect_lock`**, so the
event loop stays responsive for audio POSTs and SSE while the CPU is busy.
Do not regress this to synchronous calls: on the 2-core i3 it starves the
audio pipeline and the face boxes freeze while the mic is on.

---

## 5. The face path (`backend/app/faces/`)

Per `POST /api/faces/detect` (JPEG blob from the browser, ~6–7 fps effective):

1. `detector.detect(frame)` — YuNet finds faces (`YUNET_SCORE_THRESHOLD 0.6`,
   NMS 0.3).
2. `tracker.update(boxes)` — assigns stable `track_id`s; tracks coast for up
   to `TRACKER_MAX_DISAPPEARED = 15` frames.
3. `identity.process(frame, faces)` (`identity.py`) — the state machine:
   - faces ≥ `MIN_FACE_PX (70)` and confidence ≥ 0.8 get a SFace 128-d
     embedding;
   - the `matcher.py` compares against the gallery: best cosine ≥
     `MATCH_THRESHOLD (0.45)` **and** margin ≥ `MATCH_MARGIN (0.05)` over the
     runner-up → `recognized` (name shown only after
     `RECOG_CONFIRM_FRAMES = 2` consecutive agreeing frames; re-verified every
     `RECOG_REVERIFY_FRAMES = 10`);
   - scores in `[UNKNOWN_BELOW (0.30), 0.45)` or ambiguous → `uncertain`
     (never named, never auto-registered);
   - unknown faces feed the `pending.py` store (`PENDING_MIN_SAMPLES 4` …
     `PENDING_MAX_SAMPLES 5`, TTL 900 s) → surfaces in the UI "New face"
     flow for caregiver registration.
4. The response also ships face embeddings/crops to the *pipelines* (kept in
   memory, not returned to the browser) so audio-side association can link
   voices to faces and enrol new people.

---

## 6. Conversation intelligence (`backend/app/audio/intelligence.py`)

`POST /api/intelligence/session/{sid}/process` (called by the frontend when
the mic is turned off, and on demand):

1. Reads persisted transcript segments for the session (independent of the
   live pipeline).
2. Groups them into **windows** by inactivity timeout →
   `conversation_windows` rows.
3. Per window: **summary** + **information extraction** (structured records:
   plans, people mentioned, contact details). Both use an LLM when one is
   configured and fall back to deterministic extractive heuristics otherwise —
   the feature works fully offline.
4. Extracted records are validated, filtered, de-duplicated against previous
   records for the session, and stored with status `pending`.
5. The UI (`conversation-intelligence.js`) shows windows, summaries, pending
   extractions (approve / reject / correct) and per-participant **memories**
   (view / edit / delete). Approved records feed the participant memory
   records shown in the People panel.

All user text is inserted with `textContent`/DOM APIs in the frontend (never
`innerHTML`) to prevent markup injection.

---

## 7. Database schema (SQLite, `data/cantforget.db`)

Managed by `backend/app/db.py` (`PeopleDB`). Key tables:

- `people` — uid, name, tag (relationship), description, consent, timestamps.
- `face_embeddings` — person uid ↔ 128-d SFace vector (blob).
- `voice_embeddings` — person uid ↔ 512-d ERes2Net vector (blob).
- `thumbnails` — consented face JPEG (blob) served via
  `/api/people/{uid}/thumbnail`.
- `sessions` — one row per mic session (`sess_xxxxxxxxxxxx`).
- `transcription_segments` — per sentence: session, speaker_id, identity_uid,
  text, start_ms/end_ms, language.
- `voice_associations` — session ↔ speaker_id ↔ identity_uid (+method);
  the source for `get_speaker_identity` / intelligence attribution.
- `conversation_windows`, `conversation_summaries` — intelligence output.
- `extracted_info` — candidate records with approval status.
- `participant_memories` — per-person long-term memory records.
- `person_facts` — cumulative self-stated facts per person
  (`fact`, `source` = model/claim/intro, `session_id`); the source for the
  bullet descriptions (§4.6).

> Note: `data/forgetmenot.db` is a stale legacy file from before the project
> was renamed. The server only reads/writes `data/cantforget.db`
> (`config.DB_PATH`). Safe to archive/delete.

---

## 8. Frontend modules (`frontend/js/`)

| Module | Responsibility |
|---|---|
| `app.js` | wiring: camera/mic icon toggles, panel tabs, side drawer, FPS/detect-MS HUD, mic-session lifecycle |
| `camera.js` | `getUserMedia` video, error display |
| `face-overlay.js` | canvas overlay: boxes + person name cards (name, role pill, **fact bullets** each on its own line), `updateTracks`, `draw`, `clearTracks` |
| `audio.js` | mic capture graph (osc→downsample→16 kHz float32), chunked POSTs, session id |
| `conversation.js` | SSE subscription, transcript rendering, speaker stats, `summarizeSession` |
| `conversation-intelligence.js` | intelligence panel: windows, summaries, approvals, memories |
| `people.js` | People tab: list, register-from-photos, edit/delete |
| `welcome.js` | first-run tutorial popup, re-openable via the **?** button |

The feed also shows a "Your camera is turned off" overlay whenever the
camera is not running (`#cameraOffOverlay`, toggled in
`app.js: refreshCameraButton`, hidden with the global `.hidden` class).
| `debug.js` | legacy debug panel helpers |

Caching: modules are loaded with `?v=N` query strings (`index.html`); **bump
the version** when you edit any JS/CSS file, otherwise browsers serve the old
copy (this app has no service worker, but the browser cache is aggressive).

Mic-toggle lifecycle (`app.js`) — order matters:

```js
if (isMicRecording()) {
  const sid = getMicSessionId();   // BEFORE stopMic() — stopMic clears the id
  stopMic();
  refreshMicButton();              // synchronous: UI must flip before awaits
  stopVoiceSession();
  endVoiceSession(sid);            // POST /api/voice/sessions/{sid}/stop
  clearIntelligenceSession();
  await summarizeSession();        // may take a while — button already updated
}
```

---

## 9. Configuration reference (`backend/app/config.py`)

All values are environment-overridable. Highlights (see the file for the full
list):

| Group | Variables |
|---|---|
| Face | `YUNET_*`, `MATCH_THRESHOLD`, `MATCH_MARGIN`, `UNKNOWN_BELOW`, `RECOG_*`, `MIN_FACE_PX`, `MIN_FACE_CONFIDENCE`, `TRACKER_*`, `PENDING_*` |
| Whisper | `WHISPER_MODEL` (default `tiny`), `WHISPER_LANGUAGE` (default *auto-detect*, `""`), `WHISPER_CPU_THREADS` (default 2), `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE` |
| VAD / diarisation | `VAD_*`, `DIARIZATION_*`, `VOICE_ENROLL_*`, `VOICE_SIMILARITY_THRESHOLD`, `VOICE_ASSOC_*` |
| Summaries | `SUMMARIZER_*`, `SUMMARY_*` |
| Description engine | `DESCRIPTION_USE_MODEL` (1 = try FLAN-T5 first), `DESCRIPTION_MAX_FACTS` (8), `DESCRIPTION_MAX_CHARS` (460) |
| Storage / server | `DB_PATH` (default `data/cantforget.db`), `HOST`, `PORT` |

**Hardware budget note.** The reference machine is an Intel i3-5005U
(2 cores / 4 threads). Whisper with `base` + `beam_size=5` + 4 threads
saturates all cores and starves the face-detect endpoint (frozen overlay) and
slows transcription to minutes. The defaults (`tiny`, beam 1, 2 threads) are
the verified-working configuration on this hardware. On faster machines you
can raise `WHISPER_MODEL`/threads.

---

## 10. Running & debugging

### One-command install

```bash
# Linux
bash scripts/install.sh

# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

Each script: checks Python ≥ 3.10, creates `.venv`, installs every library
from `requirements.txt`, downloads the four model files (YuNet, SFace,
Silero-VAD, ERes2Net) into `data/models/` and — best effort — the FLAN-T5
description model. They are idempotent: existing files are not re-downloaded.

### Manual run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
# app at http://127.0.0.1:8000/
```

### Logging gotcha

App loggers (`cantforget.*`) emit at INFO/DEBUG but the root logger has no
handler when running under uvicorn's default config — **module log messages
do not appear in the uvicorn output** (only uvicorn's own access/error logs).
This is why "nothing suspicious in the server log" can hide real problems.
To see app logs:

```bash
LOGURU_LEVEL=INFO python -m uvicorn backend.app:app ... \
  2>&1 | grep -a cantforget          # or
python - <<'PY'
import logging; logging.basicConfig(level=logging.INFO)   # in a shell wrapper
PY
```

or temporarily add `logging.basicConfig(level=logging.INFO)` at the top of
`backend/app/main.py` while debugging.

### Health checks

- `GET /api/health` — detector/recognizer/db readiness, people count,
  `voice_pipelines` count.
- `GET /api/intelligence/health` — intelligence engine readiness.

### Manual smoke tests

```bash
# feed 2 s of fake audio (creates a session + loads models):
python3 - <<'PY'
import urllib.request, numpy as np
req = urllib.request.Request(
    'http://127.0.0.1:8000/api/audio/stream?session_id=sess_smoke',
    data=(np.random.randn(16000*2).astype(np.float32)*0.2).tobytes(),
    headers={'Content-Type': 'application/octet-stream'}, method='POST')
print(urllib.request.urlopen(req, timeout=60).read().decode())
PY
# check it was persisted:
sqlite3 data/cantforget.db "select id from sessions order by started_at desc limit 3"
```

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```

Known: some old `backend/tests/test_faces.py` cases target the pre-rename
`database.py` API and fail regardless of recent changes; the live API is
`db.py`.

---

## 11. Project history / conventions

- The project was renamed **ForgetMeNot → cantforget.ai** (commit `f312f57`).
  Renames touched: DB filename, UI theme (blue accent `#2f6fdb` on a B&W
  design), welcome tutorial, and the conversation-intelligence phase.
- Commit `f312f57` introduced three regressions, fixed afterwards:
  Whisper `base` + `beam_size=5` + 4 threads saturated the 2-core i3
  (frozen face boxes, stalled transcription); forced `WHISPER_LANGUAGE=en`
  garbled non-English speech into nothing; the mic button waited for the
  awaited summarize call before un-highlighting. Restored: `tiny`, beam 1,
  2 threads, auto language, face-detect moved into `to_thread`, synchronous
  mic-button refresh.
- The person-description engine was then rebuilt around the cumulative
  **fact ledger** (§4.6): model-first extraction with a deterministic claim
  extractor as fallback, bullets rendered on the card and People tab,
  compound first-person sentences split into one fact per claim.
- Conventions: async endpoints + `asyncio.to_thread` for blocking work;
  user text only via `textContent`; cache-bust versions on frontend assets;
  consent is required before any biometric registration.
