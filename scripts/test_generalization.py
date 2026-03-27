"""
Generalization test runner for unseen (OOD) production cases.

Strategy:
- Use production_cases.py only as unknown test set.
- Reuse inference.py action policy path (single LLM attempt + deterministic fallback).
- Grade with deterministic task graders.

Output:
- outputs/generalization_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import choose_action, heuristic_action
from models import AdverseEventTriageAction, ProtocolDeviationAction, SafetyNarrativeAction
from tasks.graders import grade_ae_triage, grade_protocol_deviation, grade_safety_narrative
from tasks.production_cases import EXTRA_AE_CASES, EXTRA_DEVIATION_CASES, EXTRA_NARRATIVE_CASES

OUTPUT_FILE = ROOT / "outputs" / "generalization_results.json"

TASK_ID_AE = "adverse_event_triage"
TASK_ID_DEV = "protocol_deviation_audit"
TASK_ID_NAR = "safety_narrative_generation"


def _build_observation(task_id: str, case: Dict[str, Any], step_idx: int, total_steps: int) -> Dict[str, Any]:
    if task_id == TASK_ID_AE:
        payload = dict(case)
        payload["step_count"] = step_idx
        payload["max_steps"] = total_steps
        return {
            "task_id": task_id,
            "ae_observation": payload,
            "message": "Unknown production AE case",
        }

    if task_id == TASK_ID_DEV:
        payload = dict(case)
        payload["step_count"] = step_idx
        payload["max_steps"] = total_steps
        return {
            "task_id": task_id,
            "deviation_observation": payload,
            "message": "Unknown production deviation case",
        }

    payload = dict(case)
    payload["step_count"] = step_idx
    payload["max_steps"] = total_steps
    return {
        "task_id": task_id,
        "narrative_observation": payload,
        "message": "Unknown production narrative case",
    }


def _ensure_valid_action(task_id: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    try:
        action = choose_action(task_id, observation)
        if isinstance(action, dict) and action.get("task_id") == task_id:
            return action
    except Exception:  # noqa: BLE001
        pass

    print("LLM failed, using heuristic fallback")
    return heuristic_action(task_id, observation)


def _score_ae_case(case: Dict[str, Any], step_idx: int, total_steps: int) -> float:
    observation = _build_observation(TASK_ID_AE, case, step_idx, total_steps)
    action = _ensure_valid_action(TASK_ID_AE, observation)

    try:
        action_model = AdverseEventTriageAction(**action["ae_triage"])
        return float(grade_ae_triage(action_model, case).total)
    except Exception:  # noqa: BLE001
        fallback = heuristic_action(TASK_ID_AE, observation)
        action_model = AdverseEventTriageAction(**fallback["ae_triage"])
        return float(grade_ae_triage(action_model, case).total)


def _score_deviation_case(case: Dict[str, Any], step_idx: int, total_steps: int) -> float:
    observation = _build_observation(TASK_ID_DEV, case, step_idx, total_steps)
    action = _ensure_valid_action(TASK_ID_DEV, observation)

    try:
        action_model = ProtocolDeviationAction(**action["deviation_audit"])
        return float(grade_protocol_deviation(action_model, case).total)
    except Exception:  # noqa: BLE001
        fallback = heuristic_action(TASK_ID_DEV, observation)
        action_model = ProtocolDeviationAction(**fallback["deviation_audit"])
        return float(grade_protocol_deviation(action_model, case).total)


def _score_narrative_case(case: Dict[str, Any], step_idx: int, total_steps: int) -> float:
    observation = _build_observation(TASK_ID_NAR, case, step_idx, total_steps)
    action = _ensure_valid_action(TASK_ID_NAR, observation)

    try:
        action_model = SafetyNarrativeAction(**action["safety_narrative"])
        return float(grade_safety_narrative(action_model, case).total)
    except Exception:  # noqa: BLE001
        fallback = heuristic_action(TASK_ID_NAR, observation)
        action_model = SafetyNarrativeAction(**fallback["safety_narrative"])
        return float(grade_safety_narrative(action_model, case).total)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def run_generalization() -> Dict[str, Any]:
    print("Running generalization test...")

    ae_scores = [
        _score_ae_case(case, idx, len(EXTRA_AE_CASES))
        for idx, case in enumerate(EXTRA_AE_CASES, start=1)
    ]
    dev_scores = [
        _score_deviation_case(case, idx, len(EXTRA_DEVIATION_CASES))
        for idx, case in enumerate(EXTRA_DEVIATION_CASES, start=1)
    ]
    nar_scores = [
        _score_narrative_case(case, idx, len(EXTRA_NARRATIVE_CASES))
        for idx, case in enumerate(EXTRA_NARRATIVE_CASES, start=1)
    ]

    per_task_scores = {
        TASK_ID_AE: round(_mean(ae_scores), 4),
        TASK_ID_DEV: round(_mean(dev_scores), 4),
        TASK_ID_NAR: round(_mean(nar_scores), 4),
    }

    mean_score = round(_mean(list(per_task_scores.values())), 4)

    print(f"{TASK_ID_AE}: {per_task_scores[TASK_ID_AE]:.4f}")
    print(f"{TASK_ID_DEV}: {per_task_scores[TASK_ID_DEV]:.4f}")
    print(f"{TASK_ID_NAR}: {per_task_scores[TASK_ID_NAR]:.4f}")
    print(f"Final mean score: {mean_score:.4f}")

    result = {
        "per_task_scores": per_task_scores,
        "mean_score": mean_score,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved results to: {OUTPUT_FILE}")

    return result


if __name__ == "__main__":
    run_generalization()
