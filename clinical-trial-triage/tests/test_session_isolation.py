from __future__ import annotations

from fastapi.testclient import TestClient

from server.app import app


def test_http_sessions_are_isolated() -> None:
    client = TestClient(app)

    session_a = {"X-Session-ID": "session-a"}
    session_b = {"X-Session-ID": "session-b"}

    reset_a = client.post("/reset", headers=session_a, json={"task_id": "adverse_event_triage"})
    assert reset_a.status_code == 200

    reset_b = client.post("/reset", headers=session_b, json={"task_id": "protocol_deviation_audit"})
    assert reset_b.status_code == 200

    state_a_before = client.get("/state", headers=session_a)
    state_b_before = client.get("/state", headers=session_b)
    assert state_a_before.status_code == 200
    assert state_b_before.status_code == 200

    assert state_a_before.json()["task_id"] == "adverse_event_triage"
    assert state_b_before.json()["task_id"] == "protocol_deviation_audit"
    assert state_a_before.json()["step_count"] == 0
    assert state_b_before.json()["step_count"] == 0

    step_a = client.post(
        "/step",
        headers=session_a,
        json={
            "task_id": "adverse_event_triage",
            "ae_triage": {
                "severity_classification": "severe",
                "reporting_timeline": "15-day",
                "meddra_soc": "Cardiac disorders",
                "meddra_preferred_term": "Myocardial infarction",
                "is_serious": True,
                "rationale": "session isolation test action",
            },
        },
    )
    assert step_a.status_code == 200

    state_a_after = client.get("/state", headers=session_a)
    state_b_after = client.get("/state", headers=session_b)

    assert state_a_after.status_code == 200
    assert state_b_after.status_code == 200
    assert state_a_after.json()["step_count"] == 1
    assert state_b_after.json()["step_count"] == 0
