from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List

import numpy as np


FEATURE_DIM = 128


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _hash_bucket(values: Iterable[str], dim: int = 24) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for value in values:
        text = (value or "").strip().lower()
        if not text:
            continue
        digest = hashlib.md5(text.encode("utf-8")).digest()
        for byte in digest[:4]:
            vec[byte % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else (vec / norm)


def _numeric_summary(values: List[float], width: int = 6) -> np.ndarray:
    if not values:
        return np.zeros(width, dtype=np.float32)
    arr = np.array(values, dtype=np.float32)
    q25 = float(np.quantile(arr, 0.25))
    q75 = float(np.quantile(arr, 0.75))
    return np.array(
        [
            float(arr.mean()),
            float(arr.std()),
            float(arr.min()),
            float(arr.max()),
            q25,
            q75,
        ],
        dtype=np.float32,
    )


def encode_observation(observation: Dict[str, Any]) -> np.ndarray:
    task_id = observation.get("task_id", "")

    head = np.zeros(16, dtype=np.float32)
    head[0] = 1.0 if task_id == "adverse_event_triage" else 0.0
    head[1] = 1.0 if task_id == "protocol_deviation_audit" else 0.0
    head[2] = 1.0 if task_id == "safety_narrative_generation" else 0.0

    step_count = 0.0
    max_steps = 1.0

    branch = np.zeros(72, dtype=np.float32)

    ae_obs = observation.get("ae_observation") or {}
    dev_obs = observation.get("deviation_observation") or {}
    nar_obs = observation.get("narrative_observation") or {}

    if ae_obs:
        step_count = _safe_float(ae_obs.get("step_count", 0.0), 0.0)
        max_steps = _safe_float(ae_obs.get("max_steps", 1.0), 1.0)
        branch[0] = _safe_float(ae_obs.get("patient_age", 0.0)) / 100.0
        branch[1] = 1.0 if str(ae_obs.get("patient_sex", "")).lower().startswith("m") else 0.0
        branch[2] = _safe_float(ae_obs.get("dose_mg", 0.0)) / 1000.0
        branch[3] = _safe_float(ae_obs.get("days_on_drug", 0.0)) / 365.0

        labs = ae_obs.get("lab_values", {})
        lab_vals = [_safe_float(v) for v in labs.values() if isinstance(v, (int, float))]
        branch[4:10] = _numeric_summary(lab_vals)

        hx = [str(x) for x in ae_obs.get("relevant_medical_history", [])]
        meds = [str(x) for x in ae_obs.get("concomitant_medications", [])]
        text = [ae_obs.get("narrative", ""), ae_obs.get("ae_description", ""), *hx, *meds]
        branch[10:34] = _hash_bucket(text, dim=24)

    elif dev_obs:
        step_count = _safe_float(dev_obs.get("step_count", 0.0), 0.0)
        max_steps = _safe_float(dev_obs.get("max_steps", 1.0), 1.0)
        branch[0] = _safe_float(dev_obs.get("prior_deviations", 0.0)) / 20.0
        branch[1] = _safe_float(dev_obs.get("active_subjects", 0.0)) / 200.0

        findings = dev_obs.get("findings", [])
        branch[2] = min(len(findings) / 10.0, 1.0)

        categories = []
        descriptions = []
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            categories.append(str(finding.get("category", "")))
            descriptions.append(str(finding.get("description", "")))
        branch[10:34] = _hash_bucket(categories + descriptions, dim=24)

    elif nar_obs:
        step_count = _safe_float(nar_obs.get("step_count", 0.0), 0.0)
        max_steps = _safe_float(nar_obs.get("max_steps", 1.0), 1.0)

        demographics = nar_obs.get("patient_demographics", {})
        branch[0] = _safe_float(demographics.get("age", 0.0)) / 100.0
        branch[1] = 1.0 if str(demographics.get("sex", "")).lower().startswith("m") else 0.0
        branch[2] = _safe_float(demographics.get("weight_kg", 0.0)) / 150.0

        ae = nar_obs.get("adverse_event", {})
        branch[3] = _safe_float(ae.get("ctcae_grade", 0.0)) / 5.0

        timeline = nar_obs.get("lab_values_timeline", [])
        labs = []
        for item in timeline:
            if not isinstance(item, dict):
                continue
            for value in item.values():
                if isinstance(value, (int, float)):
                    labs.append(float(value))
        branch[4:10] = _numeric_summary(labs)

        texts = [
            str(nar_obs.get("study_drug", "")),
            str(ae.get("term", "")),
            str(ae.get("action_taken", "")),
            str(nar_obs.get("outcome_at_last_followup", "")),
        ]
        texts.extend([str(x) for x in nar_obs.get("medical_history", [])])
        branch[10:34] = _hash_bucket(texts, dim=24)

    head[3] = step_count / max(max_steps, 1.0)
    head[4] = max_steps / 10.0

    msg = str(observation.get("message", ""))
    tail = _hash_bucket([msg], dim=40)

    vec = np.concatenate([head, branch, tail], axis=0)
    if vec.shape[0] < FEATURE_DIM:
        padding = np.zeros(FEATURE_DIM - vec.shape[0], dtype=np.float32)
        vec = np.concatenate([vec, padding], axis=0)

    return vec[:FEATURE_DIM].astype(np.float32)
