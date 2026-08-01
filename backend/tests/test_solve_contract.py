"""Contract tests for the /presets and /solve config schema.

Covers the 0.2 breaking change: max_exercises/min_exercises (renamed from
C_max/C_min) with the fixed 150-exercise cap removed in favor of a
work-budget check, and the 1.0 change: max_exercises/min_exercises restricted
to multiples of exercise_ladder_step (21) to bound the /solve cache-key
space, and the stub solver replaced by the real thesis DP.
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
    """T=1 with max_exercises=252 (=N, and a multiple of 21) was rejected by
    the old 150 cap."""
    r = client.post("/solve", json=calibrated(max_exercises=252, min_exercises=0))
    assert r.status_code == 200
    boundary = r.json()["boundary"]
    assert len(boundary) == 253  # N + 1 dates, spanning [0, T] inclusive
    assert len(boundary[0]) == 253  # max_exercises + 1


def test_max_exercises_above_N_rejected_with_N_message():
    """273 (a multiple of 21) exceeds N=252 at T=1, so this must fail on the
    N bound specifically, not on the (also-violated, but unrelated) ladder
    ceiling -- see the ordering comment in SolveRequest._check()."""
    r = client.post("/solve", json=calibrated(max_exercises=273, min_exercises=0))
    assert r.status_code == 422
    detail = r.json()["detail"]
    msg = detail[0]["msg"] if isinstance(detail, list) else detail
    assert "N" in msg or "252" in msg
    assert "budget" not in msg.lower()
    assert "ladder" not in msg.lower()


def test_off_ladder_max_exercises_rejected():
    """130 is not a multiple of 21."""
    r = client.post("/solve", json=calibrated(max_exercises=130, min_exercises=0))
    assert r.status_code == 422
    detail = r.json()["detail"]
    msg = detail[0]["msg"] if isinstance(detail, list) else detail
    assert "multiple" in msg.lower()


def test_over_budget_rejected_with_budget_message_not_N():
    """T=5, max_exercises=252 (the ladder ceiling -- min(N, 252) -- so the N
    check alone would pass) with kappa at its floor should be caught by the
    work-budget check."""
    r = client.post(
        "/solve",
        json=calibrated(T=5.0, kappa=0.5, max_exercises=252, min_exercises=0),
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    msg = detail if isinstance(detail, str) else detail[0]["msg"]
    assert "budget" in msg.lower()
    assert "exceeds N" not in msg


def test_q_max_above_cap_rejected():
    r = client.post("/solve", json=calibrated(q_max=int(1e7)))
    assert r.status_code == 422


def test_min_exceeding_max_rejected():
    """147 (7 x 21) is on-ladder but exceeds the default max_exercises=126,
    so this exercises the min<=max check specifically, not the multiple_of
    field constraint."""
    r = client.post("/solve", json=calibrated(min_exercises=147))
    assert r.status_code == 422
    detail = r.json()["detail"]
    msg = detail if isinstance(detail, str) else detail[0]["msg"]
    assert "min_exercises" in msg


def test_calibrated_config_still_solves_to_expected_shape():
    r = client.post("/solve", json=calibrated())
    assert r.status_code == 200
    boundary = r.json()["boundary"]
    assert len(boundary) == 253  # N + 1
    assert len(boundary[0]) == 127


def test_presets_payload_shape():
    r = client.get("/presets")
    assert r.status_code == 200
    body = r.json()

    assert "C_max_cap" not in body
    assert "q_max" in body["ranges"]
    assert "min" in body["ranges"]["q_max"] and "max" in body["ranges"]["q_max"]
    assert "work_budget" in body
    assert "exercise_ladder_step" in body
    assert "max_exercises_by_T" in body

    ladder_step = body["exercise_ladder_step"]
    by_T = body["max_exercises_by_T"]
    assert {entry["T"] for entry in by_T} == set(body["T_choices"])
    for entry in by_T:
        assert entry["max_exercises"] >= ladder_step
        assert entry["max_exercises"] % ladder_step == 0


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
        "henry_hub", "henry_hub_hi", "henry_hub_lo", "henry_hub_flat",
    }
    assert set(props["T"]["enum"]) == {0.5, 1.0, 2.0, 5.0}
