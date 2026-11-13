"""Identity service tests with a fake recognizer (no SFace model needed)."""

import numpy as np
import pytest

from backend.app.db import PeopleDB, ValidationError
from backend.app.faces.identity import DuplicatePersonError, IdentityService
from backend.app.faces.matcher import Matcher
from backend.app.faces.pending import PendingStore

FRAME = np.full((480, 640, 3), 90, np.uint8)


def vec(seed):
    return np.random.default_rng(seed).normal(size=128).astype(np.float32)


class FakeRecognizer:
    """Embedding is chosen by the 'x' of the detection row: row[0] = person seed."""

    def embed(self, frame, row):
        base = vec(int(row[0]))
        return base + np.random.default_rng(int(row[1])).normal(size=128).astype(np.float32) * 0.05


def face(track_id, seed, size=120, conf=0.95):
    row = np.zeros(15, np.float32)
    row[0], row[1] = seed, track_id  # fake: identity seed + tiny noise seed
    return {"track_id": track_id, "x": 100, "y": 100, "w": size, "h": size, "confidence": conf, "row": row}


def make():
    db = PeopleDB(":memory:")
    matcher = Matcher(0.45, 0.05, 0.30)
    pending = PendingStore(0.45, ttl_seconds=900)
    svc = IdentityService(FakeRecognizer(), db, matcher, pending, detector=None, confirm_frames=2)
    return svc


def run(svc, seed, n, track_id=1, **kw):
    out = None
    for _ in range(n):
        out = svc.process(FRAME, [face(track_id, seed, **kw)])[track_id]
    return out


def test_new_face_is_flagged_not_stored():
    svc = make()
    out = run(svc, 11, 6)
    assert out["status"] == "new_face" and out["pending_id"].startswith("n_")
    assert svc.db.list_people() == [] and svc.db.load_gallery() == {}  # nothing persisted


def test_register_requires_consent_then_recognizes_by_name():
    svc = make()
    pid = run(svc, 11, 6)["pending_id"]
    with pytest.raises(ValidationError):
        svc.register_pending(pid, "Asha", "Daughter", "Visits on Sundays", consent=False)
    assert svc.db.list_people() == []
    person = svc.register_pending(pid, "Asha", "Daughter", "Visits on Sundays", consent=True)
    assert person["uid"].startswith("p_")
    out = run(svc, 11, 4)
    assert out["status"] == "recognized"
    assert (out["name"], out["tag"], out["description"]) == ("Asha", "Daughter", "Visits on Sundays")
    assert out["person_uid"] == person["uid"] and "embedding" not in out


def test_name_needs_consecutive_confirmations():
    svc = make()
    svc.register_pending(run(svc, 11, 6)["pending_id"], "Asha", "", "", True)
    first = svc.process(FRAME, [face(7, 11)])[7]  # brand-new track
    assert first["status"] == "checking" and "name" not in first
    assert svc.process(FRAME, [face(7, 11)])[7]["status"] == "recognized"


def test_second_new_person_gets_own_candidate_and_stranger_not_named():
    svc = make()
    svc.register_pending(run(svc, 11, 6)["pending_id"], "Asha", "", "", True)
    out = run(svc, 22, 6, track_id=2)
    assert out["status"] == "new_face" and "name" not in out


def test_same_face_new_track_reuses_pending_id():
    svc = make()
    a = run(svc, 33, 6, track_id=1)["pending_id"]
    b = run(svc, 33, 6, track_id=5)["pending_id"]
    assert a == b and len(svc.pending.list_active()) == 1


def test_dismissed_face_is_not_offered_again():
    svc = make()
    pid = run(svc, 44, 6)["pending_id"]
    assert svc.dismiss_pending(pid)
    assert run(svc, 44, 8, track_id=9)["status"] == "unknown"


def test_duplicate_registration_is_refused():
    svc = make()
    svc.register_pending(run(svc, 11, 6)["pending_id"], "Asha", "", "", True)
    # Same person seen on a track whose gallery check hasn't confirmed yet:
    cand = svc.pending.upsert([vec(11)] * 4, None, 1.0)
    with pytest.raises(DuplicatePersonError):
        svc.register_pending(cand.id, "Asha again", "", "", True)
    assert len(svc.db.list_people()) == 1


def test_uncertain_never_named_and_never_offered_for_registration():
    svc = make()
    svc.register_pending(run(svc, 11, 6)["pending_id"], "Asha", "", "", True)
    # A face whose embedding sits between Asha and a stranger
    class Mid(FakeRecognizer):
        def embed(self, frame, row):
            return vec(11) + 2.2 * vec(55)

    svc.recognizer = Mid()
    out = run(svc, 0, 8, track_id=3)
    assert out["status"] == "uncertain" and "name" not in out
    assert svc.pending.list_active() == []


def test_small_or_low_confidence_faces_are_skipped():
    svc = make()
    assert run(svc, 11, 6, size=30)["status"] == "low_quality"
    assert run(svc, 11, 6, conf=0.5, track_id=2)["status"] == "low_quality"
    assert svc.pending.list_active() == []


def test_delete_person_removes_name_and_biometrics():
    svc = make()
    person = svc.register_pending(run(svc, 11, 6)["pending_id"], "Asha", "", "", True)
    assert run(svc, 11, 4)["status"] == "recognized"
    assert svc.delete_person(person["uid"]) is True
    assert svc.db.load_gallery() == {}
    out = run(svc, 11, 1)
    assert out["status"] != "recognized" and "name" not in out


def test_edit_updates_displayed_name():
    svc = make()
    person = svc.register_pending(run(svc, 11, 6)["pending_id"], "Asha", "", "", True)
    run(svc, 11, 4)
    svc.update_person(person["uid"], "Asha K", "Daughter", "New description")
    out = svc.process(FRAME, [face(1, 11)])[1]
    assert out["name"] == "Asha K" and out["description"] == "New description"


def test_pending_expires():
    clock = {"t": 0.0}
    store = PendingStore(0.45, ttl_seconds=10, clock=lambda: clock["t"])
    c = store.upsert([vec(1)] * 4, None, 1.0)
    clock["t"] = 11.0
    assert store.get(c.id) is None
