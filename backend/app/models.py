"""Pydantic request/response schemas for Phase 2."""

from __future__ import annotations

from pydantic import BaseModel
from typing import Optional, List


class FaceDetection(BaseModel):
    track_id: int
    x: int
    y: int
    w: int
    h: int
    confidence: float
    hits: int
    age: int
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    person_relationship: Optional[str] = None
    recognition_confidence: Optional[float] = None


class DetectResponse(BaseModel):
    ok: bool
    elapsed_ms: float
    frame_w: int
    frame_h: int
    faces: List[FaceDetection]


class PersonBase(BaseModel):
    name: str
    relationship: Optional[str] = None


class PersonCreate(PersonBase):
    pass


class PersonOut(PersonBase):
    id: str
    created_at: str
    updated_at: str
    embedding_count: int = 0


class EnrollRequest(BaseModel):
    track_id: int
    name: str
    relationship: Optional[str] = None


class EnrollResponse(BaseModel):
    ok: bool
    person_id: str
    person_name: str
    samples: int
