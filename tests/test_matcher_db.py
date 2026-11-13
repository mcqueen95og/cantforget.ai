import numpy as np
import pytest

from backend.app.db import PeopleDB, ValidationError
from backend.app.faces.matcher import MATCH, UNCERTAIN, UNKNOWN, Matcher


def vec(seed):
    return np.random.default_rng(seed).normal(size=128).astype(np.float32)


def noisy(v, amount, seed=0):
    return v + np.random.default_rng(seed).normal(size=128).astype(np.float32) * amount


def gallery():
    return {"a": np.vstack([vec(1), noisy(vec(1), 0.2, 5)]), "b": vec(2).reshape(1, -1)}


def test_empty_gallery_is_unknown():
    assert Matcher().match(vec(1)).status == UNKNOWN


def test_match_uncertain_unknown():
    m = Matcher(0.45, 0.05, 0.30)
    m.set_gallery(gallery())
    assert m.match(noisy(vec(1), 0.3, 9)).person_uid == "a"
    assert m.match(vec(99)).status == UNKNOWN  # random vector: near-zero similarity
    # Halfway between a and a stranger -> weak similarity -> uncertain, never named
    mid = vec(1) + 2.2 * vec(77)
    r = m.match(mid)
    assert r.status == UNCERTAIN and r.person_uid is None


def test_ambiguous_between_two_people_is_not_named():
    m = Matcher(0.45, 0.05, 0.30)
    base = vec(10)
    m.set_gallery({"a": noisy(base, 0.05, 1).reshape(1, -1), "b": noisy(base, 0.05, 2).reshape(1, -1)})
    r = m.match(base)
    assert r.status == UNCERTAIN and r.person_uid is None


def test_db_requires_consent_and_validates():
    db = PeopleDB(":memory:")
    with pytest.raises(ValidationError):
        db.create_person("Ann", "", "", [vec(1)], consent=False)
    with pytest.raises(ValidationError):
        db.create_person("  ", "", "", [vec(1)], consent=True)
    with pytest.raises(ValidationError):
        db.create_person("Ann", "", "x" * 501, [vec(1)], consent=True)
    with pytest.raises(ValidationError):
        db.create_person("Ann", "", "", [], consent=True)
    assert db.list_people() == []


def test_db_create_update_delete_cascades_embeddings():
    db = PeopleDB(":memory:")
    p = db.create_person("Ann", "Daughter", "Visits Sundays", [vec(1), vec(2)], b"jpg", consent=True)
    assert p["uid"].startswith("p_") and p["samples"] == 2 and "embedding" not in str(p)
    db.update_person(p["uid"], "Anne", "Daughter", "Visits Sundays")
    assert db.get_person(p["uid"])["name"] == "Anne"
    assert p["uid"] in db.load_gallery()
    assert db.delete_person(p["uid"]) is True
    assert db.load_gallery() == {}
    left = db._conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
    assert left == 0
    assert db.delete_person(p["uid"]) is False


def test_unique_ids():
    db = PeopleDB(":memory:")
    ids = {db.create_person(f"P{i}", "", "", [vec(i)], consent=True)["uid"] for i in range(20)}
    assert len(ids) == 20
