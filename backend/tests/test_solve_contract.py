"""Contract tests for the /presets and /solve config schema.

Covers the 0.2 breaking change: max_exercises/min_exercises (renamed from
C_max/C_min) with the fixed 150-exercise cap removed in favor of a
work-budget check.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schema import SolveRequest

client = TestClient(app)


def calibrated(**overrides):
    cfg = SolveRequest().model_dump()
    cfg.update(overrides)
    return cfg


def test_full_flexibility_contract_accepted():
    """T=1 with max_exercises=252 (=N) was rejected by the old 150 cap."""
    r = client.post("/solve", json=calibrated(max_exercises=252, min_exercises=0))
    assert r.status_code == 200
    boundary = r.json()["boundary"]
    assert len(boundary) == 252
    assert len(boundary[0]) == 253


def test_max_exercises_above_N_rejected_with_N_message():
    r = client.post("/solve", json=calibrated(max_exercises=253, min_exercises=0))
    assert r.status_code == 422
    detail = r.json()["detail"]
    msg = detail[0]["msg"] if isinstance(detail, list) else detail
    assert "N" in msg or "252" in msg
    assert "budget" not in msg.lower()


def test_over_budget_rejected_with_budget_message_not_N():
    """T=5, max_exercises=1260 (=N exactly, so the N check alone would pass)
    with kappa at its floor should be caught by the work-budget check."""
    r = client.post(
        "/solve",
        json=calibrated(T=5.0, kappa=0.5, max_exercises=1260, min_exercises=0),
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    msg = detail if isinstance(detail, str) else detail[0]["msg"]
    assert "budget" in msg.lower()
    assert "exceeds N" not in msg


def test_q_max_above_cap_rejected():
    r = client.post("/solve", json=calibrated(q_max=1e7))
    assert r.status_code == 422


def test_min_exceeding_max_rejected():
    r = client.post("/solve", json=calibrated(min_exercises=999))
    assert r.status_code == 422
    detail = r.json()["detail"]
    msg = detail if isinstance(detail, str) else detail[0]["msg"]
    assert "min_exercises" in msg


def test_calibrated_config_still_solves_to_expected_shape():
    r = client.post("/solve", json=calibrated())
    assert r.status_code == 200
    boundary = r.json()["boundary"]
    assert len(boundary) == 252
    assert len(boundary[0]) == 127


def test_presets_payload_shape():
    r = client.get("/presets")
    assert r.status_code == 200
    body = r.json()

    assert "C_max_cap" not in body
    assert "q_max" in body["ranges"]
    assert "min" in body["ranges"]["q_max"] and "max" in body["ranges"]["q_max"]
    assert "work_budget" in body
    assert "max_exercises_by_T" in body

    by_T = body["max_exercises_by_T"]
    assert {entry["T"] for entry in by_T} == set(body["T_choices"])
    for entry in by_T:
        assert entry["max_exercises"] >= 1


@pytest.mark.parametrize("bad_preset", ["not_a_real_preset", ""])
def test_unknown_alpha_preset_rejected(bad_preset):
    r = client.post("/solve", json=calibrated(alpha_preset=bad_preset))
    assert r.status_code == 422


@pytest.mark.parametrize("bad_T", [0.75, 3.0, -1.0])
def test_non_enumerated_T_rejected(bad_T):
    r = client.post("/solve", json=calibrated(T=bad_T))
    assert r.status_code == 422


def test_openapi_renders_enums_for_dropdowns():
    schema = client.get("/openapi.json").json()
    props = schema["components"]["schemas"]["SolveRequest"]["properties"]
    assert set(props["alpha_preset"]["enum"]) == {
        "henry_hub", "henry_hub_hi", "henry_hub_lo", "wti", "power",
    }
    assert set(props["T"]["enum"]) == {0.5, 1.0, 2.0, 5.0}
