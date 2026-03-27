from __future__ import annotations

from fastapi.testclient import TestClient

from models import (
    AESeverity,
    AdverseEventTriageAction,
    ReportingTimeline,
    TaskID,
    TriageAction,
)
from server.app import app
from server.openenv_env import ClinicalTrialOpenEnv


def test_openenv_adapter_reset_step_state() -> None:
    env = ClinicalTrialOpenEnv()
    obs = env.reset(task_id=TaskID.ADVERSE_EVENT_TRIAGE)
    assert obs.task_id == TaskID.ADVERSE_EVENT_TRIAGE
    assert obs.done is False

    action = TriageAction(
        task_id=TaskID.ADVERSE_EVENT_TRIAGE,
        ae_triage=AdverseEventTriageAction(
            severity_classification=AESeverity.SEVERE,
            reporting_timeline=ReportingTimeline.FIFTEEN_DAY,
            meddra_soc="Cardiac disorders",
            meddra_preferred_term="Myocardial infarction",
            is_serious=True,
            rationale="test action",
        ),
    )

    step_obs = env.step(action)
    assert isinstance(step_obs.reward, float)
    assert step_obs.task_id == TaskID.ADVERSE_EVENT_TRIAGE

    state = env.state
    assert state.step_count >= 1
    assert state.task_id == TaskID.ADVERSE_EVENT_TRIAGE


def test_openenv_http_endpoints_available() -> None:
    client = TestClient(app)

    metadata_resp = client.get("/openenv/metadata")
    assert metadata_resp.status_code == 200

    schema_resp = client.get("/openenv/schema")
    assert schema_resp.status_code == 200

    reset_resp = client.post("/openenv/reset", json={"task_id": "adverse_event_triage"})
    assert reset_resp.status_code == 200

    step_resp = client.post(
        "/openenv/step",
        json={
            "action": {
                "task_id": "adverse_event_triage",
                "ae_triage": {
                    "severity_classification": "severe",
                    "reporting_timeline": "15-day",
                    "meddra_soc": "Cardiac disorders",
                    "meddra_preferred_term": "Myocardial infarction",
                    "is_serious": True,
                    "rationale": "http test action",
                },
            }
        },
    )
    assert step_resp.status_code == 200
