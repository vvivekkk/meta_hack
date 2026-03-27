"""
Pre-submission validator for Clinical Trial Triage OpenEnv.

Checks:
1. Core endpoints respond and return expected shapes.
2. /tasks returns >= 3 tasks.
3. Each task can be completed and /grader returns score in [0.0, 1.0].
4. Baseline script runs without errors and produces outputs/baseline_results.json.

Usage:
    python scripts/validate_submission.py

Notes:
    - Requires the API server to be running (default: http://localhost:8000).
    - Uses deterministic heuristic actions for endpoint and grader checks.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import httpx

# Ensure project root import resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import TaskID
from scripts.heuristic_baseline import (
    _heuristic_ae_triage,
    _heuristic_deviation_audit,
    _heuristic_narrative,
)
from tasks.case_bank import AE_CASES, DEVIATION_CASES, NARRATIVE_CASES


BASE_URL = os.environ.get("VALIDATOR_BASE_URL", "http://localhost:8000").rstrip("/")
OUTPUT_FILE = ROOT / "outputs" / "baseline_results.json"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _post_json(client: httpx.Client, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = client.post(path, json=payload)
    _assert(response.status_code == 200, f"{path} returned {response.status_code}: {response.text}")
    return response.json()


def _run_episode(client: httpx.Client, task_id: str) -> float:
    reset_data = _post_json(client, "/reset", {"task_id": task_id})
    _assert("observation" in reset_data, f"/reset missing observation for task {task_id}")

    if task_id == TaskID.ADVERSE_EVENT_TRIAGE:
        for case in AE_CASES:
            step_payload = _heuristic_ae_triage(case).model_dump()
            step_response = _post_json(client, "/step", step_payload)
            if step_response.get("done"):
                break
    elif task_id == TaskID.PROTOCOL_DEVIATION_AUDIT:
        for case in DEVIATION_CASES:
            step_payload = _heuristic_deviation_audit(case).model_dump()
            step_response = _post_json(client, "/step", step_payload)
            if step_response.get("done"):
                break
    elif task_id == TaskID.SAFETY_NARRATIVE_GENERATION:
        for case in NARRATIVE_CASES:
            step_payload = _heuristic_narrative(case).model_dump()
            step_response = _post_json(client, "/step", step_payload)
            if step_response.get("done"):
                break
    else:
        raise AssertionError(f"Unknown task_id: {task_id}")

    grader_response = client.get("/grader")
    _assert(grader_response.status_code == 200, f"/grader failed for task {task_id}: {grader_response.text}")
    grader_data = grader_response.json()
    score = grader_data.get("normalized_score")
    _assert(isinstance(score, (int, float)), f"normalized_score missing for task {task_id}")
    _assert(0.0 <= float(score) <= 1.0, f"normalized_score out of range for task {task_id}: {score}")
    return float(score)


def _check_openenv_endpoints(client: httpx.Client) -> None:
    metadata = client.get("/openenv/metadata")
    _assert(metadata.status_code == 200, f"/openenv/metadata returned {metadata.status_code}")

    schema = client.get("/openenv/schema")
    _assert(schema.status_code == 200, f"/openenv/schema returned {schema.status_code}")

    reset = client.post("/openenv/reset", json={"task_id": TaskID.ADVERSE_EVENT_TRIAGE})
    _assert(reset.status_code == 200, f"/openenv/reset returned {reset.status_code}: {reset.text}")
    reset_payload = reset.json()
    _assert("observation" in reset_payload, "/openenv/reset missing observation")

    step = client.post(
        "/openenv/step",
        json={
            "action": {
                "task_id": TaskID.ADVERSE_EVENT_TRIAGE,
                "ae_triage": {
                    "severity_classification": "severe",
                    "reporting_timeline": "15-day",
                    "meddra_soc": "Cardiac disorders",
                    "meddra_preferred_term": "Myocardial infarction",
                    "is_serious": True,
                    "rationale": "validator openenv smoke action",
                },
            }
        },
    )
    _assert(step.status_code == 200, f"/openenv/step returned {step.status_code}: {step.text}")

    state = client.get("/openenv/state")
    _assert(state.status_code == 200, f"/openenv/state returned {state.status_code}: {state.text}")

    health = client.get("/openenv/health")
    _assert(health.status_code == 200, f"/openenv/health returned {health.status_code}")


def _run_baseline_script() -> Dict[str, Any]:
    cmd = [sys.executable, str(ROOT / "scripts" / "baseline_inference.py")]
    process = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    _assert(process.returncode == 0, f"baseline_inference.py failed:\n{process.stderr}\n{process.stdout}")
    _assert(OUTPUT_FILE.exists(), f"Missing baseline output file: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    tasks = data.get("tasks", {})
    _assert(len(tasks) >= 3, "Baseline output does not contain all 3 tasks")
    _assert("overall_mean_reward" in data, "Baseline output missing overall_mean_reward")
    return data


def main() -> None:
    print("Running pre-submission validator")
    print(f"Base URL: {BASE_URL}")

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        root = client.get("/")
        _assert(root.status_code == 200, f"/ returned {root.status_code}")

        health = client.get("/health")
        _assert(health.status_code == 200, f"/health returned {health.status_code}")

        tasks = client.get("/tasks")
        _assert(tasks.status_code == 200, f"/tasks returned {tasks.status_code}")
        tasks_data = tasks.json()
        task_list = tasks_data.get("tasks", [])
        _assert(len(task_list) >= 3, f"Expected >=3 tasks, found {len(task_list)}")

        _check_openenv_endpoints(client)

        scores: Dict[str, float] = {}
        for task in [
            TaskID.ADVERSE_EVENT_TRIAGE,
            TaskID.PROTOCOL_DEVIATION_AUDIT,
            TaskID.SAFETY_NARRATIVE_GENERATION,
        ]:
            scores[task] = _run_episode(client, task)

    baseline_data = _run_baseline_script()

    print("All checks passed")
    print("Episode grader scores:")
    for task_id, score in scores.items():
        print(f"  - {task_id}: {score:.4f}")
    print(f"Baseline overall mean: {baseline_data.get('overall_mean_reward')}")


if __name__ == "__main__":
    main()