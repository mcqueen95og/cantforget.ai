# cantforget.ai

A friendly web app that runs entirely on your own computer. It watches through the
camera, listens through the microphone, and helps you remember the people in your
life: who they are, what they said, and what matters to them.

It is "CPU-friendly" and "privacy-first": nothing is uploaded to the internet, and
everything is stored in a small file on your machine.

## What it does

- **See who is in the room.** Open the app, click **Start Camera**, and faces are
  outlined on the video with a label. Someone who has been registered before is
  greeted by name; a new face is marked "New face".
- **Hear what people say.** Click **Start Mic** and every sentence that is spoken
  is typed out live in the Conversation panel (speech-to-text).
- **Know who said what.** The app can tell different voices apart and shows a
  speaker label next to each sentence, so the transcript reads like "who said this line".
- **Introduce someone just by talking.** If someone says *"hey, I'm Mary, I'm your
  daughter"*, the app automatically:
  1. hears the name (**Mary**) and the relationship (**daughter**) from the sentence,
  2. takes a snapshot of their face from the camera,
  3. creates a person in its memory with the name, the relationship, and the snapshot,
  4. remembers their voice too.
  From the next conversation onwards, both their face and their voice are recognised.
- **Remember what was talked about.** While you chat, the app quietly writes a few
  short, gentle key points per person (for example: *"Mary is coming for dinner."*).
  These are saved with the person, so you can read them back later.
- **Ask before saving.** Facts like phone numbers or plans that could be worth
  remembering are put on an "Awaiting approval" list — you approve or reject each one.
- **Register people from photos too.** If you prefer, you can add a person by
  uploading one or more clear photos of their face.
- **You stay in control.** People can be edited or deleted at any time. Deleting
  someone removes everything stored about them — face data, voice data, notes.

## Directory layout

```
cantforget.ai/
├── requirements.txt          # the extra parts Python needs
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── config.py         # settings you can tweak
│       ├── db.py             # the memory file (SQLite): people, faces, voices, notes
│       ├── main.py           # the web server + all its routes
│       ├── faces/            # seeing people
│       │   ├── detector.py   # finds faces in a frame (YuNet)
│       │   ├── tracker.py    # follows the same face between frames
│       │   ├── recognizer.py # turns a face into numbers (SFace)
│       │   ├── matcher.py    # decides "is this someone we know?"
│       │   ├── pending.py    # holds "new faces" until you decide
│       │   └── identity.py   # ties all of the above together
│       └── audio/            # hearing people
│           ├── asr.py        # speech-to-text (Whisper)
│           ├── vad.py        # tells speech apart from silence
│           ├── diarize.py    # tells different voices apart
│           ├── enrollment.py # saves a person's voice (with consent)
│           ├── associate.py  # links a voice to a face
│           ├── summarize.py  # writes the gentle key points
│           ├── intelligence.py # longer-term conversation notes
│           ├── pipeline.py   # the audio assembly line
│           └── events.py     # live updates sent to the web page
├── frontend/                 # the web page you interact with
│   ├── index.html
│   ├── css/main.css
│   └── js/                   # camera, mic, transcript, people list...
├── data/
│   ├── forgetmenot.db        # everything the app remembers lives here
│   └── models/               # the "brains": small AI model files (see Quick start)
└── tests/                    # checks that the code still works
```

## Quick start

You need three things, all free:

1. **Python 3.10 or newer** — get it from <https://www.python.org/downloads/>
2. A webcam and a microphone
3. The Chrome, Edge or Firefox browser

### 1. Get the project files

If you have the project folder already, open it. Otherwise download/unzip it and
open a terminal in that folder.

**Linux (Ubuntu/Mint/Debian):** press `Ctrl+Alt+T` to open a terminal.
**Windows:** press the Start button, type `PowerShell`, and open Windows PowerShell.

### 2. Make a private Python folder (a "virtual environment")

This keeps the app's Python parts separate from the rest of your system.

**Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

When it works, you will see `(.venv)` at the start of the line.

### 3. Install the pieces

**Linux and Windows (same command):**
```bash
pip install -r requirements.txt
```

On some Linux systems (for example Arch), you may first need to install `portaudio`
headers or use the venv as shown above — the venv approach is the simplest and is
what we recommend.

### 4. Download the small AI model files

The app needs a few free model files (a few MB each) in the `data/models/` folder.
Download these four links and save them with the shown file names:

```
https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
   -> data/models/face_detection_yunet.onnx

https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
   -> data/models/face_recognition_sface_2021dec.onnx

https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
   -> data/models/silero_vad.onnx

https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
   -> data/models/speaker/eres2net.onnx
```

(The `data/models/speaker/` folder may need to be created first.)

The speech-to-text model (Whisper "tiny") is downloaded automatically the first
time you use the microphone — no action needed.

### 5. Start the app

**Linux and Windows (same command):**
```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Keep this window open — it is the app running.

### 6. Open the app in your browser

Visit <http://127.0.0.1:8000/> and:

1. Click **Start Camera** and allow camera access.
2. Click **Start Mic** and allow microphone access.
3. Talk! Sentences appear in the Conversation panel with a speaker label.
4. Say *"hey, I'm [your name], I'm your [relationship]"* — watch the People tab:
   a new person appears with your name, your relationship, and your snapshot.
5. When you stop the mic, key points are written to the Key points panel.

### 7. Try the built-in checks (optional)

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```

## Endpoints

For the curious: these are the "addresses" the web page uses to talk to the app.
You normally never need to type these yourself.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | The web page |
| GET | `/api/health` | Is everything working? (camera model, database, people count) |
| POST | `/api/faces/detect` | One camera frame -> faces found, with "do we know them?" info |
| GET | `/api/people` | List everyone the app remembers |
| POST | `/api/people` | Register someone from photos |
| GET / PATCH / DELETE | `/api/people/{uid}` | Read / edit / delete one person |
| DELETE | `/api/people` | Remove everyone and all their stored data |
| POST | `/api/people/{uid}/samples` | Add more photos for a person |
| GET | `/api/people/{uid}/thumbnail` | A person's small photo |
| GET | `/api/pending` | New faces waiting for your decision |
| POST | `/api/pending/{id}/register` | Say "yes, remember this face as..." |
| DELETE | `/api/pending/{id}` | Say "not now" |
| POST | `/api/audio/stream` | Send one piece of microphone sound |
| GET | `/api/voice/events` | Live conversation updates (sentences, speakers, key points) |
| GET | `/api/voice/sessions/{id}/transcript` | The full transcript of a session |
| GET | `/api/voice/sessions/{id}/summaries` | The key points of a session |
| POST | `/api/voice/sessions/{id}/summarize` | Ask for the key points now |
| POST | `/api/voice/sessions/{id}/stop` | Finish a conversation session |
| POST/GET | `/api/intelligence/...` | Longer-term notes, approval list, memories |

## Tuning (optional)

If you want to tweak how the app behaves, set these before starting the server
(on Linux: `VAR=value python -m uvicorn ...`; on Windows PowerShell:
`$env:VAR="value"` before the start command).

| Variable | Default | Meaning |
|----------|---------|---------|
| `MATCH_THRESHOLD` | 0.45 | How strict the face match must be before a name is shown |
| `WHISPER_MODEL` | tiny | Speech-to-text quality: `tiny` (fastest) → `base` → `small` (best, needs a better PC) |
| `WHISPER_LANGUAGE` | (auto) | Leave empty to auto-detect the spoken language, or set e.g. `en`, `de` |
| `DIARIZATION_THRESHOLD` | 0.5 | How different two voices must be to count as two people |
| `SUMMARIZER_USE_MODEL` | 0 | `0` = simple instant key points; `1` = use a downloaded AI summariser |
| `DB_PATH` | data/forgetmenot.db | Where the memory file lives |

## Performance notes

Everything is designed to run on an ordinary desktop or laptop (even an older
i3 with 8 GB of RAM). The camera loop is throttled to keep the CPU cool, and the
speech-to-text model is the smallest fast one by default. If speech recognition
feels slow, let it be — it catches up in the background; or upgrade `WHISPER_MODEL`
only if your PC has room to spare.

## Privacy

Everything runs on your machine and nothing is sent to the internet:

- Camera frames are processed in memory and never saved.
- What is saved: a small photo you consented to, a list of numbers describing
  each face and voice (these cannot be turned back into a picture or audio),
  and the conversation notes — all inside `data/forgetmenot.db`.
- Deleting a person removes every piece of their data.
- The app only listens and watches while you press the Start buttons and the
  browser shows the permission prompt.

Note: the server binds to your machine only (`127.0.0.1`). Do not expose it to
a network or the internet without adding a login system first.
