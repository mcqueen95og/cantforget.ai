"""ForgetMeNot backend -- Phase 2: face detection, recognition and registration.

Routes
  GET    /                              frontend
  GET    /api/health                    readiness (detector, recognizer, people count)
  POST   /api/faces/detect              JPEG frame -> tracked faces + identity per face
  GET    /api/people                    list registered people
  POST   /api/people                    register from photos (consent required)
  GET    /api/people/{uid}              one person
  PATCH  /api/people/{uid}              edit name / tag / description
  DELETE /api/people/{uid}              delete person + their face data
  POST   /api/people/{uid}/samples      add more reference photos
  GET    /api/people/{uid}/thumbnail    consented face thumbnail (JPEG)
  GET    /api/pending                   new faces seen by the camera, awaiting a decision
  GET    /api/pending/{id}/thumbnail    thumbnail of a pending face
  POST   /api/pending/{id}/register     register a pending face (consent required)
  DELETE /api/pending/{id}              dismiss a pending face

All handlers are ``async def`` on purpose: they run one at a time on the event
loop, so the shared OpenCV detector/recognizer and in-memory state need no
extra locking. Embeddings are never returned by any endpoint.
"""

from __future__ import annotations

import logging
import time
import asyncio
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import (
    DB_PATH,
    FRONTEND_DIR,
    MATCH_MARGIN,
    MATCH_THRESHOLD,
    MAX_UPLOAD_BYTES,
    MIN_FACE_CONFIDENCE,
    MIN_FACE_PX,
    PENDING_MAX_CANDIDATES,
    PENDING_MAX_SAMPLES,
    PENDING_MIN_SAMPLES,
    PENDING_TTL_SECONDS,
    RECOG_CONFIRM_FRAMES,
    RECOG_REVERIFY_FRAMES,
    SFACE_MODEL_PATH,
    TRACKER_IOU_THRESHOLD,
    TRACKER_MAX_DISAPPEARED,
    TRACKER_MAX_DISTANCE,
    UNKNOWN_BELOW,
    YUNET_INPUT_SIZE,
    YUNET_MODEL_PATH,
    YUNET_NMS_THRESHOLD,
    YUNET_SCORE_THRESHOLD,
)
from .db import PeopleDB, ValidationError
from .faces.detector import FaceDetector
from .faces.identity import DuplicatePersonError, IdentityService, NoUsableFaceError
from .faces.matcher import Matcher
from .faces.pending import PendingStore
from .faces.recognizer import FaceRecognizer
from .faces.tracker import CentroidTracker
from .audio.events import ConversationEventBus
from .audio.pipeline import AudioPipeline
from .audio.summarize import ConversationSummarizer
from .audio.intelligence import (
    ConversationIntelligenceEngine,
    LLMClient,
    ConversationWindow,
    TranscriptSegment,
    CONVERSATION_INACTIVITY_MS,
    CONVERSATION_MAX_WINDOW_SECONDS,
)
from .audio.enrollment import (
    start_voice_enrollment,
    get_voice_enrollment,
    delete_voice_enrollment_session,
)
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

logger = logging.getLogger("forgetmenot.backend")


class Services:
    """Process-wide singletons, created at startup."""

    detector: Optional[FaceDetector] = None
    recognizer: Optional[FaceRecognizer] = None
    tracker: Optional[CentroidTracker] = None
    db: Optional[PeopleDB] = None
    identity: Optional[IdentityService] = None
    # Phase 3: voice / conversation
    event_bus: Optional[ConversationEventBus] = None
    pipelines: dict[str, AudioPipeline] = {}  # session_id -> AudioPipeline
    # Phase 4: conversation intelligence
    intelligence: Optional[ConversationIntelligenceEngine] = None
    llm_client: Optional[LLMClient] = None


svc = Services()


@asynccontextmanager
async def lifespan(_: FastAPI):
    svc.tracker = CentroidTracker(
        max_disappeared=TRACKER_MAX_DISAPPEARED,
        max_distance=TRACKER_MAX_DISTANCE,
        iou_threshold=TRACKER_IOU_THRESHOLD,
    )
    try:
        svc.detector = FaceDetector(
            model_path=YUNET_MODEL_PATH,
            input_size=YUNET_INPUT_SIZE,
            score_threshold=YUNET_SCORE_THRESHOLD,
            nms_threshold=YUNET_NMS_THRESHOLD,
        )
    except Exception:  # noqa: BLE001
        logger.exception("YuNet detector failed to load; detection is unavailable")
    try:
        svc.recognizer = FaceRecognizer(SFACE_MODEL_PATH)
    except Exception:  # noqa: BLE001
        logger.exception("SFace recognizer failed to load; recognition is unavailable")
    try:
        svc.db = PeopleDB(DB_PATH)
    except Exception:  # noqa: BLE001
        logger.exception("Database failed to open; registration is unavailable")
    try:
        svc.event_bus = ConversationEventBus()
    except Exception:  # noqa: BLE001
        logger.exception("Event bus failed to initialize")
    try:
        svc.llm_client = LLMClient()
        svc.intelligence = ConversationIntelligenceEngine(svc.db, llm=svc.llm_client)
    except Exception:  # noqa: BLE001
        logger.exception("Conversation intelligence engine failed to initialize")
    if svc.recognizer and svc.db:
        svc.identity = IdentityService(
            svc.recognizer,
            svc.db,
            Matcher(MATCH_THRESHOLD, MATCH_MARGIN, UNKNOWN_BELOW),
            PendingStore(
                same_person_threshold=MATCH_THRESHOLD,
                ttl_seconds=PENDING_TTL_SECONDS,
                max_candidates=PENDING_MAX_CANDIDATES,
                max_samples=PENDING_MAX_SAMPLES,
            ),
            detector=svc.detector,
            confirm_frames=RECOG_CONFIRM_FRAMES,
            reverify_frames=RECOG_REVERIFY_FRAMES,
            min_face_px=MIN_FACE_PX,
            min_face_confidence=MIN_FACE_CONFIDENCE,
            pending_min_samples=PENDING_MIN_SAMPLES,
            pending_max_samples=PENDING_MAX_SAMPLES,
        )
    logger.info(
        "Startup: detector=%s recognizer=%s db=%s",
        svc.detector is not None,
        svc.recognizer is not None,
        svc.db is not None,
    )
    yield
    if svc.db:
        svc.db.close()


app = FastAPI(title="ForgetMeNot — Phase 2: Recognition & Registration", version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# --- helpers ---------------------------------------------------------------


def _identity() -> IdentityService:
    if svc.identity is None:
        raise HTTPException(
            status_code=503,
            detail="Recognition is unavailable (SFace model or database not loaded).",
        )
    return svc.identity


async def _read_image(upload: UploadFile) -> np.ndarray:
    raw = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large.")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Unable to decode image")
    return img


def _jpeg(data: Optional[bytes]) -> Response:
    if not data:
        raise HTTPException(status_code=404, detail="No thumbnail")
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def _crop_jpeg(frame: np.ndarray, row, max_side: int = 112, quality: int = 80) -> Optional[bytes]:
    """Crop a detected face from the frame and encode a small JPEG.

    ``row`` is the detector's rectangle (x, y, w, h in frame coordinates).
    Used so a person introduced by speech also gets a face thumbnail.
    """
    h, w = frame.shape[:2]
    x, y, fw, fh = [int(v) for v in row[:4]]
    pad = int(0.18 * max(fw, fh))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + fw + pad), min(h, y + fh + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = frame[y0:y1, x0:x1]
    if max(crop.shape[:2]) > max_side:
        scale = max_side / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return bytes(buf) if ok else None


def _handle(exc: Exception) -> HTTPException:
    """Map domain errors to HTTP errors without leaking internals."""
    if isinstance(exc, ValidationError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="Not found")
    if isinstance(exc, DuplicatePersonError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, NoUsableFaceError):
        return HTTPException(status_code=422, detail=str(exc))
    logger.exception("Unexpected error")
    return HTTPException(status_code=500, detail="Something went wrong.")


_DOMAIN_ERRORS = (ValidationError, KeyError, DuplicatePersonError, NoUsableFaceError)


class ProfileBody(BaseModel):
    name: str
    tag: str = ""
    description: str = ""


class RegisterBody(ProfileBody):
    consent: bool = False


class ApproveBody(BaseModel):
    corrected_info: Optional[str] = None


class MemoriesBody(BaseModel):
    description: str = ""
    interests: list[str] = []
    preferences: list[str] = []


# --- basic routes ----------------------------------------------------------


async def _health_payload() -> dict:
    return {
        "status": "ok",
        "phase": 3,
        "detector_ready": svc.detector is not None,
        "recognizer_ready": svc.recognizer is not None,
        "database_ready": svc.db is not None,
        "people_count": len(svc.db.list_people()) if svc.db else 0,
        "event_bus_ready": svc.event_bus is not None,
        "voice_pipelines": len(svc.pipelines),
    }


@app.get("/api/health")
async def health() -> dict:
    return await _health_payload()


@app.get("/health")
async def health_alias() -> dict:
    """Alias for /api/health so bare /health probes also succeed."""
    return await _health_payload()


@app.get("/v1/models")
async def v1_models() -> dict:
    """OpenAI-compatible model list.

    External tools (IDE extensions, local AI dashboards) probe
    ``/v1/models`` on any server that answers ``/health``. Returning a real
    list instead of 404 keeps those probes quiet and documents what this
    app runs locally.
    """
    return {
        "object": "list",
        "data": [
            {"id": "forgetmenot-whisper", "object": "model", "owned_by": "forgetmenot"},
            {"id": "forgetmenot-speaker-eres2net", "object": "model", "owned_by": "forgetmenot"},
            {"id": "forgetmenot-face-yunet-sface", "object": "model", "owned_by": "forgetmenot"},
        ],
    }


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/")
async def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built yet")
    return FileResponse(index_path)


@app.post("/api/faces/detect")
async def detect_faces(file: UploadFile = File(...)) -> JSONResponse:
    """Accept a JPEG frame; detect, track, and (if available) recognise faces.

    Each face carries an ``identity`` object. ``status`` is one of:
    recognized | checking | unknown | uncertain | new_face | low_quality |
    recognition_unavailable. Only ``recognized`` includes a name.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file")
    if svc.detector is None or svc.tracker is None:
        raise HTTPException(status_code=503, detail="Face detector is unavailable")

    start = time.perf_counter()
    frame = await _read_image(file)

    try:
        raw = svc.detector.detect(frame)
    except Exception:  # noqa: BLE001
        logger.exception("Face detection failed")
        raise HTTPException(status_code=500, detail="Detection error")

    dets = [d for d in raw if d["confidence"] >= YUNET_SCORE_THRESHOLD]
    tracks = svc.tracker.update([(d["x"], d["y"], d["w"], d["h"]) for d in dets])

    faces = []
    for t in tracks:
        if t["det_index"] is None:  # coasting track, not seen this frame
            continue
        d = dets[t["det_index"]]
        faces.append(
            {
                "track_id": t["track_id"],
                "x": d["x"],
                "y": d["y"],
                "w": d["w"],
                "h": d["h"],
                "confidence": d["confidence"],
                "hits": t["hits"],
                "age": t["age"],
                "row": d["row"],
            }
        )

    identities: dict[int, dict] = {}
    if svc.identity is not None:
        try:
            identities = svc.identity.process(frame, faces)
        except Exception:  # noqa: BLE001 - recognition trouble must not stop detection
            logger.exception("Recognition step failed")

    results = []
    for f in faces:
        public = {k: v for k, v in f.items() if k != "row"}
        public["identity"] = identities.get(
            f["track_id"],
            {"status": "recognition_unavailable" if svc.identity is None else "checking"},
        )
        results.append(public)

    # Sync face tracks with audio pipeline for voice-to-face association.
    # Include face embeddings (and a small JPEG crop) so self-introduced
    # persons can be enrolled with voice AND face biometrics.
    if svc.pipelines:
        now_ms = time.time() * 1000.0
        pipeline_tracks = []
        for public, f in zip(results, faces):
            ident = public.get("identity", {})
            # Compute the face embedding for this track so the audio pipeline
            # can enrol it when a self-introduction occurs.
            face_emb = None
            try:
                emb = svc.recognizer.embed(frame, f["row"]) if svc.recognizer else None
                if emb is not None:
                    face_emb = emb.astype(np.float32).tobytes()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to compute face embedding for track %s", public["track_id"])
            # Small JPEG crop for the person thumbnail.
            face_crop = None
            try:
                face_crop = _crop_jpeg(frame, f["row"], max_side=112, quality=80)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to crop face for track %s", public["track_id"])
            pipeline_tracks.append(
                {
                    "track_id": public["track_id"],
                    "person_uid": ident.get("person_uid"),
                    "status": ident.get("status", "checking"),
                    "embedding": face_emb,
                    "face_crop": face_crop,
                    "x": public["x"],
                    "y": public["y"],
                    "w": public["w"],
                    "h": public["h"],
                    "start_ms": now_ms,
                    "end_ms": now_ms + 500,
                }
            )
        for pipeline in svc.pipelines.values():
            try:
                asyncio.create_task(pipeline.update_face_tracks(pipeline_tracks))
            except Exception:  # noqa: BLE001
                logger.exception("Failed to sync face tracks to pipeline")

    return JSONResponse(
        {
            "ok": True,
            "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 2),
            "frame_w": frame.shape[1],
            "frame_h": frame.shape[0],
            "faces": results,
        }
    )


# --- people ----------------------------------------------------------------


@app.get("/api/people")
async def list_people() -> dict:
    return {"people": _identity().db.list_people()}


@app.post("/api/people", status_code=201)
async def create_person(
    name: str = Form(...),
    tag: str = Form(""),
    description: str = Form(""),
    consent: bool = Form(False),
    photos: list[UploadFile] = File(...),
) -> dict:
    ident = _identity()
    if len(photos) > 10:
        raise HTTPException(status_code=413, detail="At most 10 photos at a time.")
    images = [await _read_image(p) for p in photos]
    try:
        person, notes = ident.enroll_from_images(name, tag, description, consent, images)
    except _DOMAIN_ERRORS as exc:
        raise _handle(exc)
    return {"person": person, "notes": notes}


@app.get("/api/people/{uid}")
async def get_person(uid: str) -> dict:
    person = _identity().db.get_person(uid)
    if person is None:
        raise HTTPException(status_code=404, detail="Not found")
    return person


@app.patch("/api/people/{uid}")
async def update_person(uid: str, body: ProfileBody) -> dict:
    try:
        return _identity().update_person(uid, body.name, body.tag, body.description)
    except _DOMAIN_ERRORS as exc:
        raise _handle(exc)


@app.delete("/api/people/{uid}")
async def delete_person(uid: str) -> dict:
    if not _identity().delete_person(uid):
        raise HTTPException(status_code=404, detail="Not found")
    return {"deleted": uid}


@app.delete("/api/people")
async def delete_all_people() -> dict:
    """Remove every person and all their stored face/voice data."""
    count = _identity().db.delete_all_people()
    if svc.identity is not None:
        svc.identity.reload()
    for pipeline in svc.pipelines.values():
        if pipeline._associator:
            pipeline._associator._reload_voice_gallery()
    return {"deleted": count}


@app.post("/api/people/{uid}/samples")
async def add_samples(uid: str, photos: list[UploadFile] = File(...)) -> dict:
    ident = _identity()
    if len(photos) > 10:
        raise HTTPException(status_code=413, detail="At most 10 photos at a time.")
    images = [await _read_image(p) for p in photos]
    try:
        person, notes = ident.add_samples(uid, images)
    except _DOMAIN_ERRORS as exc:
        raise _handle(exc)
    return {"person": person, "notes": notes}


@app.get("/api/people/{uid}/thumbnail")
async def person_thumbnail(uid: str) -> Response:
    return _jpeg(_identity().db.get_thumbnail(uid))


# --- pending (new faces seen by the camera) --------------------------------


@app.get("/api/pending")
async def list_pending() -> dict:
    ident = _identity()
    return {
        "pending": [{"pending_id": c.id, "sightings": c.sightings} for c in ident.pending.list_active()]
    }


@app.get("/api/pending/{pending_id}/thumbnail")
async def pending_thumbnail(pending_id: str) -> Response:
    cand = _identity().pending.get(pending_id)
    if cand is None:
        raise HTTPException(status_code=404, detail="Not found")
    return _jpeg(cand.thumbnail)


@app.post("/api/pending/{pending_id}/register", status_code=201)
async def register_pending(pending_id: str, body: RegisterBody) -> dict:
    try:
        person = _identity().register_pending(
            pending_id, body.name, body.tag, body.description, body.consent
        )
    except _DOMAIN_ERRORS as exc:
        raise _handle(exc)
    return {"person": person}


@app.delete("/api/pending/{pending_id}")
async def dismiss_pending(pending_id: str) -> dict:
    if not _identity().dismiss_pending(pending_id):
        raise HTTPException(status_code=404, detail="Not found")
    return {"dismissed": pending_id}


# ------------------------------------------------------------------ voice ---

def _event_bus() -> ConversationEventBus:
    if svc.event_bus is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation event bus is unavailable",
        )
    return svc.event_bus


def _get_or_create_pipeline(session_id: str) -> AudioPipeline:
    if session_id not in svc.pipelines:
        if svc.db is None:
            raise HTTPException(
                status_code=503, detail="Database is unavailable"
            )
        pipeline = AudioPipeline(
            db=svc.db,
            event_bus=_event_bus(),
            session_id=session_id,
        )
        # When a person is auto-created from a self-introduction ("hey im X"),
        # the face matcher must reload its gallery so the new face is
        # recognised on subsequent frames.
        if svc.identity is not None:
            pipeline.on_person_created = svc.identity.reload
        svc.pipelines[session_id] = pipeline
    return svc.pipelines[session_id]


@app.post("/api/audio/stream")
async def audio_stream(request: Request, session_id: str = "default"):
    """HTTP POST endpoint for streaming audio chunks from the frontend.

    Expects one binary chunk per request: float32 PCM, 16 kHz mono.
    """
    data = await request.body()
    if not data:
        return {"ok": True, "samples": 0}
    if len(data) % 4 != 0:
        raise HTTPException(status_code=400, detail="Audio body must be float32-aligned")
    samples = np.frombuffer(data, dtype=np.float32)
    pipeline = _get_or_create_pipeline(session_id)
    await pipeline.start()
    await pipeline.feed_audio(samples)
    logger.debug("HTTP audio: session=%s chunk=%d samples buffered=%.1fs",
                 session_id, samples.size, pipeline.buffered_seconds())
    return {"ok": True, "samples": int(samples.size)}



@app.get("/api/voice/events")
async def voice_events(session_id: str = "default"):
    """Server-Sent Events stream of conversation events for a session."""
    bus = _event_bus()

    async def event_generator():
        queue = await bus.subscribe(session_id)
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            await bus.unsubscribe(session_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/voice/enroll/start")
async def voice_enroll_start(
    person_uid: str,
    name: str,
    min_samples: int = 3,
    max_samples: int = 10,
) -> dict:
    """Start a voice enrollment session for an already-registered person."""
    if person_uid not in (_identity()._people if _identity() else {}):
        raise HTTPException(status_code=404, detail="Person not found")
    eid = start_voice_enrollment(person_uid, name, min_samples, max_samples)
    return {"enrollment_id": eid, "person_uid": person_uid, "name": name}


@app.post("/api/voice/enroll/sample")
async def voice_enroll_sample(
    enrollment_id: str,
    file: UploadFile = File(...),
) -> dict:
    """Submit one voice sample (WAV/Opus/Ogg) for enrollment.

    The backend will run VAD to extract speech, then compute a voice
    fingerprint and add it to the enrollment session.
    """
    session = get_voice_enrollment(enrollment_id)
    if not session:
        raise HTTPException(status_code=404, detail="Enrollment session not found")

    try:
        raw = await file.read()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Unable to read audio: {exc}")

    # Decode audio using OpenCV-free approach: use soundfile if available,
    # otherwise fallback to raw float32 interpretation.
    try:
        import soundfile as sf
        import io
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if sr != SAMPLE_RATE:
            # Simple resampling: repeat or drop samples (crude but OK for enrollment)
            ratio = SAMPLE_RATE / sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except ImportError:
        # soundfile not installed: interpret raw bytes as float32 PCM at 16 kHz
        audio = np.frombuffer(raw, dtype=np.float32)

    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio")

    duration_ms = round(len(audio) / SAMPLE_RATE * 1000.0, 1)
    result = session.add_sample(audio, duration_ms)
    if not result["accepted"]:
        raise HTTPException(status_code=400, detail=result["reason"])

    return {
        "ok": True,
        "enrollment_id": enrollment_id,
        "sample_count": session.sample_count,
        "duration_ms": duration_ms,
    }


@app.post("/api/voice/enroll/commit")
async def voice_enroll_commit(enrollment_id: str) -> dict:
    """Commit a voice enrollment session: store the voice profile in SQLite."""
    session = get_voice_enrollment(enrollment_id)
    if not session:
        raise HTTPException(status_code=404, detail="Enrollment session not found")

    now = datetime.now(timezone.utc).isoformat()
    ok, message = session.commit(svc.db, now)
    if not ok:
        raise HTTPException(status_code=400, detail=message)

    delete_voice_enrollment_session(enrollment_id)
    # Reload voice gallery in the associator if it exists.
    for pipeline in svc.pipelines.values():
        if pipeline._associator:
            pipeline._associator._reload_voice_gallery()

    return {
        "ok": True,
        "person_uid": session.person_uid,
        "person_name": session.name,
        "samples": session.sample_count,
    }


@app.delete("/api/voice/enroll/{enrollment_id}")
async def voice_enroll_cancel(enrollment_id: str) -> dict:
    """Cancel a voice enrollment session without saving."""
    ok = delete_voice_enrollment_session(enrollment_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Enrollment session not found")
    return {"ok": True}


@app.delete("/api/voice/person/{person_uid}")
async def voice_person_delete(person_uid: str) -> dict:
    """Delete all voice enrollments for a person."""
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    count = svc.db.delete_voice_enrollments(person_uid)
    # Reload voice gallery in all pipelines.
    for pipeline in svc.pipelines.values():
        if pipeline._associator:
            pipeline._associator._reload_voice_gallery()
    return {"ok": True, "deleted": count}


@app.post("/api/voice/sessions/{session_id}/stop")
async def voice_session_stop(session_id: str) -> dict:
    """Stop and release the audio pipeline for a finished mic session."""
    pipeline = svc.pipelines.pop(session_id, None)
    if pipeline is None:
        return {"ok": True, "stopped": False}
    await pipeline.stop()
    return {"ok": True, "stopped": True}


@app.get("/api/voice/sessions/{session_id}/status")
async def voice_session_status(session_id: str) -> dict:
    """Return pipeline component readiness for a session."""
    pipeline = svc.pipelines.get(session_id)
    if pipeline is None:
        return {
            "session_id": session_id,
            "running": False,
            "vad_ready": svc.event_bus is not None,
            "asr_ready": False,
            "diarizer_ready": False,
            "associator_ready": False,
            "buffered_seconds": 0.0,
        }
    return {
        "session_id": session_id,
        "running": pipeline._running,
        "vad_ready": pipeline._vad is not None and pipeline._vad.is_loaded(),
        "asr_ready": pipeline._asr is not None and pipeline._asr.is_loaded(),
        "diarizer_ready": pipeline._diarizer is not None,
        "associator_ready": pipeline._associator is not None,
        "buffered_seconds": round(pipeline.buffered_seconds(), 2),
    }


@app.get("/api/voice/sessions/{session_id}/transcript")
async def voice_transcript(session_id: str, limit: int = 100) -> dict:
    """Return recent transcription segments for a session."""
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    segments = svc.db.get_transcript_segments(session_id, limit)
    return {"session_id": session_id, "segments": segments}


@app.get("/api/voice/sessions/{session_id}/associations")
async def voice_associations(session_id: str) -> dict:
    """Return voice-to-face associations for a session."""
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    assocs = svc.db.get_voice_associations(session_id)
    return {"session_id": session_id, "associations": assocs}


@app.post("/api/voice/sessions/{session_id}/summarize")
async def summarize_session(session_id: str) -> dict:
    """Summarize the conversation into gentle key points per person.

    Groups the persisted transcript segments by who said what (the diarized
    speaker label, linked to a recognised face identity where the voice was
    associated with a face), runs a lightweight extractive summarizer over
    each person's words, and saves the key points to the database keyed by
    person.
    """
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    segments = svc.db.get_transcript_segments(session_id)
    if not segments:
        return {"session_id": session_id, "summaries": []}

    # Group by person: recognised identity first, otherwise the diarized
    # speaker label (who said which sentence).
    by_person: dict[str, dict] = {}
    for seg in segments:
        identity_uid = seg.get("identity_uid")
        speaker_id = seg.get("speaker_id") or "SPEAKER_UNKNOWN"
        key = identity_uid or speaker_id
        entry = by_person.setdefault(key, {"speaker_id": speaker_id, "texts": []})
        text = (seg.get("text") or "").strip()
        if text:
            entry["texts"].append(text)

    summarizer = ConversationSummarizer()
    results = summarizer.summarize_by_person(
        {key: entry["texts"] for key, entry in by_person.items()}
    )

    saved: list[dict] = []
    for key, points in results.items():
        if not points:
            continue
        entry = by_person[key]
        identity_uid = key if key.startswith("p_") else None
        speaker_id = entry["speaker_id"]
        summary_id = svc.db.add_conversation_summary(
            session_id=session_id,
            speaker_id=speaker_id,
            key_points=points,
            identity_uid=identity_uid,
        )
        person_name = None
        if identity_uid:
            person = svc.db.get_person(identity_uid)
            if person:
                person_name = person.get("name")
        saved.append(
            {
                "id": summary_id,
                "speaker_id": speaker_id,
                "identity_uid": identity_uid,
                "person_name": person_name,
                "key_points": points,
            }
        )

    return {"session_id": session_id, "summaries": saved}


@app.get("/api/voice/sessions/{session_id}/summaries")
async def voice_summaries(session_id: str, limit: int = 50) -> dict:
    """Return saved key-point summaries for a session."""
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    summaries = svc.db.get_conversation_summaries(session_id, limit)
    return {"session_id": session_id, "summaries": summaries}


# ============================================================================
# Phase 4: Conversation Intelligence & Memory Engine
# ============================================================================

def _intelligence() -> ConversationIntelligenceEngine:
    if svc.intelligence is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation intelligence engine is unavailable",
        )
    return svc.intelligence


@app.post("/api/intelligence/session/{session_id}/process")
async def intelligence_process_session(
    session_id: str,
) -> dict:
    """Process all transcript segments for a session into windows,
    then run structured summarization + information extraction.

    Reads persisted transcript segments from the database (independent of the
    live audio pipeline), groups them into conversation windows, and produces
    summaries and candidate extracted-information records for approval.
    """
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    engine = _intelligence()

    segments = svc.db.get_transcript_segments(session_id, limit=1000)
    if not segments:
        return {
            "session_id": session_id,
            "windows": 0,
            "summaries": 0,
            "extracted": 0,
            "message": "No transcript segments found.",
        }

    # Build attribution map from existing voice associations.
    assocs = svc.db.get_voice_associations(session_id)
    assoc_map: dict[str, str] = {}
    for a in assocs:
        assoc_map[a["speaker_id"]] = a["identity_uid"]

    # Group segments into windows by inactivity timeout.
    seg_objs = [
        {
            "speaker_id": s["speaker_id"],
            "identity_uid": s.get("identity_uid"),
            "text": s["text"],
            "start_ms": s["start_ms"],
            "end_ms": s["end_ms"],
        }
        for s in segments
    ]
    seg_objs.sort(key=lambda x: x["start_ms"])

    windows: list[dict] = []
    current: list[dict] = []
    window_start_ms = seg_objs[0]["start_ms"]
    last_end_ms = seg_objs[0]["end_ms"]
    window_index = 0

    async def flush_window() -> None:
        nonlocal window_index, current, window_start_ms
        if not current:
            return
        window_id = "cw_" + uuid.uuid4().hex[:12]
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        svc.db.create_conversation_window(
            session_id, window_index, started, window_id=window_id
        )
        # Process through the intelligence engine.
        result = await engine.process_window(
            ConversationWindow(
                window_id=window_id,
                session_id=session_id,
                window_index=window_index,
                segments=[
                    TranscriptSegment(
                        speaker_id=s["speaker_id"],
                        identity_uid=s.get("identity_uid"),
                        identity_name=None,
                        text=s["text"],
                        start_ms=s["start_ms"],
                        end_ms=s["end_ms"],
                    )
                    for s in current
                ],
                started_at=started,
                ended_at=None,
            )
        )
        windows.append(result)
        window_index += 1
        current = []
        window_start_ms = 0

    for s in seg_objs:
        if current and (s["start_ms"] - last_end_ms) >= CONVERSATION_INACTIVITY_MS:
            await flush_window()
        current.append(s)
        last_end_ms = s["end_ms"]
        if (s["end_ms"] - window_start_ms) >= CONVERSATION_MAX_WINDOW_SECONDS * 1000:
            await flush_window()
    if current:
        await flush_window()

    return {
        "session_id": session_id,
        "windows": len(windows),
        "summaries": len(windows),
        "extracted": sum(len(w.get("extracted", [])) for w in windows),
        "windows_detail": windows,
    }


@app.get("/api/intelligence/session/{session_id}/windows")
async def intelligence_windows(session_id: str, limit: int = 50) -> dict:
    """Return conversation windows for a session."""
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    windows = svc.db.get_windows(session_id, limit)
    return {"session_id": session_id, "windows": windows}


@app.get("/api/intelligence/session/{session_id}/summaries")
async def intelligence_summaries(session_id: str, limit: int = 50) -> dict:
    """Return structured summaries for a session."""
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    summaries = svc.db.get_conversation_summaries(session_id, limit)
    return {"session_id": session_id, "summaries": summaries}


@app.get("/api/intelligence/session/{session_id}/extracted")
async def intelligence_extracted(
    session_id: str,
    status: Optional[str] = None,
    limit: int = 100,
) -> dict:
    """Return extracted-information records for a session."""
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    records = svc.db.get_extracted_information(session_id, status, limit)
    return {"session_id": session_id, "records": records}


@app.get("/api/intelligence/extracted/pending")
async def intelligence_pending_extracted(limit: int = 100) -> dict:
    """Return all extracted-information records awaiting approval."""
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    records = svc.db.get_pending_extracted_information(limit=limit)
    return {"records": records}


@app.post("/api/intelligence/extracted/{record_id}/approve")
async def intelligence_approve_extracted(
    record_id: str,
    body: ApproveBody = ApproveBody(),
) -> dict:
    """Approve a candidate extracted-information record."""
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    ok = svc.db.approve_extracted_information(record_id, body.corrected_info)
    if not ok:
        raise HTTPException(status_code=404, detail="Record not found or already processed")
    return {"ok": True, "record_id": record_id, "status": "approved"}


@app.post("/api/intelligence/extracted/{record_id}/reject")
async def intelligence_reject_extracted(record_id: str) -> dict:
    """Reject (discard) a candidate extracted-information record."""
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    ok = svc.db.reject_extracted_information(record_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Record not found or already processed")
    return {"ok": True, "record_id": record_id, "status": "rejected"}


@app.get("/api/intelligence/participants/{participant_uid}/memories")
async def intelligence_participant_memories(participant_uid: str) -> dict:
    """Return a participant's approved memory record."""
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    memories = svc.db.get_participant_memories(participant_uid)
    if memories is None:
        return {"participant_uid": participant_uid, "memories": None}
    return {"participant_uid": participant_uid, "memories": memories}


@app.patch("/api/intelligence/participants/{participant_uid}/memories")
async def intelligence_update_memories(
    participant_uid: str,
    body: MemoriesBody,
) -> dict:
    """Create or update a participant's approved memory record."""
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    rid = svc.db.upsert_participant_memories(
        participant_uid,
        description=body.description,
        interests=body.interests,
        preferences=body.preferences,
    )
    return {"ok": True, "record_id": rid}


@app.delete("/api/intelligence/participants/{participant_uid}/memories")
async def intelligence_delete_memories(participant_uid: str) -> dict:
    """Delete a participant's memory record."""
    if svc.db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    ok = svc.db.delete_participant_memories(participant_uid)
    if not ok:
        raise HTTPException(status_code=404, detail="No memory record found")
    return {"ok": True, "participant_uid": participant_uid}


@app.get("/api/intelligence/health")
async def intelligence_health() -> dict:
    """Report intelligence engine readiness."""
    engine = _intelligence()
    return {
        "engine_ready": True,
        "llm_provider": engine.llm.provider,
        "llm_model": engine.llm.model,
        "llm_available": engine.llm.is_available(),
    }
