"""
inference.py - Clinical Trial Triage OpenEnv Baseline
=====================================================
Reliable, deterministic baseline runner for OpenEnv submission.

Design goals:
- Keep OpenAI SDK compatibility with HF router variables.
- Never crash when LLM/API fails.
- Deterministic fallback for all tasks.
- Always write outputs/baseline_results.json.
"""

from __future__ import annotations

import json
import os
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


# Keep required OpenAI/HF compatibility variables.
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"

SERVER_URL = os.getenv("ENV_SERVER_URL") or "http://localhost:8000"
TEMPERATURE = 0.0
MAX_TOKENS = 1000
OUTPUT_FILE = Path("outputs/baseline_results.json")

TASK_IDS = [
    "adverse_event_triage",
    "protocol_deviation_audit",
    "safety_narrative_generation",
]

VALID_AE_SEVERITY = {"mild", "moderate", "severe", "life_threatening", "fatal"}
VALID_TIMELINE = {"7-day", "15-day", "routine"}
VALID_DEV_TYPE = {"major", "minor", "protocol_amendment"}
VALID_CAUSALITY = {
    "definitely_related",
    "probably_related",
    "possibly_related",
    "unlikely_related",
    "not_related",
    "unassessable",
}


def _make_client() -> Optional[OpenAI]:
    if not API_KEY:
        return None
    try:
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception:  # noqa: BLE001
        return None


CLIENT = _make_client()

SYSTEM_PROMPT = textwrap.dedent(
    """
You are a clinical pharmacovigilance specialist.
Return only a valid JSON action object for the provided task.
No markdown, no prose, no explanations.
"""
).strip()

AE_TASK_PROMPT = """
TASK: Adverse Event Triage
Observation:
{observation}

Return JSON:
{{
  "task_id": "adverse_event_triage",
  "ae_triage": {{
    "severity_classification": "mild|moderate|severe|life_threatening|fatal",
    "reporting_timeline": "7-day|15-day|routine",
    "meddra_soc": "string",
    "meddra_preferred_term": "string",
    "is_serious": true,
    "rationale": "string"
  }}
}}
"""

DEV_TASK_PROMPT = """
TASK: Protocol Deviation Audit
Observation:
{observation}

Return JSON:
{{
  "task_id": "protocol_deviation_audit",
  "deviation_audit": {{
    "deviation_type": "major|minor|protocol_amendment",
    "capa_required": true,
    "site_risk_score": 6.5,
    "flagged_finding_ids": ["F001"],
    "recommended_action": "string"
  }}
}}
"""

NARRATIVE_TASK_PROMPT = """
TASK: Safety Narrative Generation
Observation:
{observation}

Return JSON:
{{
  "task_id": "safety_narrative_generation",
  "safety_narrative": {{
    "narrative_text": "string",
    "causality_assessment": "definitely_related|probably_related|possibly_related|unlikely_related|not_related|unassessable",
    "key_temporal_flags": ["string"],
    "dechallenge_positive": true,
    "rechallenge_positive": null
  }}
}}
"""


def observation_to_text(obs: dict) -> str:
    lines: list[str] = []

    def flatten(item: object, prefix: str = "") -> None:
        if isinstance(item, dict):
            for key, value in item.items():
                child_prefix = f"{prefix}{key}: " if not prefix else f"{prefix}  {key}: "
                flatten(value, child_prefix)
        elif isinstance(item, list):
            for i, value in enumerate(item):
                flatten(value, f"{prefix}[{i}] ")
        else:
            lines.append(f"{prefix}{item}")

    flatten(obs)
    return "\n".join(lines)


def build_prompt(task_id: str, obs: dict) -> str:
    obs_text = observation_to_text(obs)
    if task_id == "adverse_event_triage":
        return AE_TASK_PROMPT.format(observation=obs_text)
    if task_id == "protocol_deviation_audit":
        return DEV_TASK_PROMPT.format(observation=obs_text)
    return NARRATIVE_TASK_PROMPT.format(observation=obs_text)


def parse_json_action(text: str) -> Optional[dict]:
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
    cleaned = cleaned.strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except Exception:  # noqa: BLE001
                return None
    return None


def safe_llm_call(prompt: str) -> Optional[dict]:
    """Single-attempt LLM call that never throws and returns parsed JSON or None."""
    if CLIENT is None:
        return None

    try:
        response = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw_text = response.choices[0].message.content or ""
        return parse_json_action(raw_text)
    except Exception:
        return None


def _to_bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def extract_finding_ids(obs: dict) -> list[str]:
    findings = obs.get("deviation_observation", {}).get("findings", [])
    return [str(item.get("id", "")) for item in findings if isinstance(item, dict) and item.get("id")]


def heuristic_action(task_id: str, obs: dict) -> dict:
    """Deterministic fallback policy that always returns valid action JSON."""
    if task_id == "adverse_event_triage":
        ae = obs.get("ae_observation", {})
        narrative = f"{ae.get('narrative', '')} {ae.get('ae_description', '')}".lower()
        labs = ae.get("lab_values", {}) if isinstance(ae.get("lab_values"), dict) else {}

        def _f(name: str, fallback: float = 0.0) -> float:
            try:
                return float(labs.get(name, fallback) or fallback)
            except Exception:  # noqa: BLE001
                return fallback

        alt = _f("ALT_U_L")
        alt_uln = _f("ALT_ULN")
        bilirubin = _f("Bilirubin_mg_dL")
        severe_liver_signal = (alt_uln > 0 and alt / alt_uln >= 5.0) or bilirubin >= 2.0

        if any(kw in narrative for kw in ["fatal", "death", "died"]):
            severity, timeline, serious = "fatal", "7-day", True
        elif any(kw in narrative for kw in ["stemi", "cardiac arrest", "icu", "life-threatening", "hypotension"]):
            severity, timeline, serious = "life_threatening", "7-day", True
        elif any(kw in narrative for kw in ["hospital", "encephalopathy", "grade 3", "severe", "jaundice"]):
            severity, timeline, serious = "severe", "15-day", True
        elif any(kw in narrative for kw in ["moderate", "grade 2", "nausea", "vomiting"]):
            severity, timeline, serious = "moderate", "routine", False
        else:
            severity, timeline, serious = "mild", "routine", False

        if any(kw in narrative for kw in ["cardiac", "myocardial", "stemi", "heart"]):
            soc, pt = "Cardiac disorders", "Myocardial infarction"
        elif any(kw in narrative for kw in ["encephalopathy", "neurolog", "ataxia", "hallucination"]):
            soc, pt = "Nervous system disorders", "Encephalopathy"
        elif any(kw in narrative for kw in ["anaphyl", "urticaria", "immune"]):
            soc, pt = "Immune system disorders", "Anaphylactic reaction"
        elif any(kw in narrative for kw in ["nausea", "vomiting"]) and not severe_liver_signal:
            soc, pt = "Gastrointestinal disorders", "Nausea"
        elif any(kw in narrative for kw in ["liver", "bilirubin", "alt", "ast", "jaundice"]):
            soc, pt = "Hepatobiliary disorders", "Drug-induced liver injury"
        else:
            soc, pt = "General disorders", "Adverse event"

        return {
            "task_id": "adverse_event_triage",
            "ae_triage": {
                "severity_classification": severity,
                "reporting_timeline": timeline,
                "meddra_soc": soc,
                "meddra_preferred_term": pt,
                "is_serious": serious,
                "rationale": "Deterministic heuristic triage based on narrative and labs.",
            },
        }

    if task_id == "protocol_deviation_audit":
        dev = obs.get("deviation_observation", {})
        findings = dev.get("findings", [])
        risk_keywords = {
            "eligibility",
            "blinding",
            "unblind",
            "sae",
            "integrity",
            "consent",
            "accountability",
            "endpoint",
            "source",
            "edc",
            "temperature",
        }

        flagged: list[str] = []
        risk_hits = 0
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            text = f"{finding.get('category', '')} {finding.get('description', '')}".lower()
            if any(token in text for token in risk_keywords):
                risk_hits += 1
                fid = str(finding.get("id", "")).strip()
                if fid:
                    flagged.append(fid)

        prior = float(dev.get("prior_deviations", 0) or 0)
        score = min(10.0, risk_hits * 1.8 + prior * 0.35)
        dev_type = "major" if risk_hits >= 2 or score >= 6.0 else "minor"
        capa = dev_type == "major"

        if dev_type == "minor":
            flagged = []

        return {
            "task_id": "protocol_deviation_audit",
            "deviation_audit": {
                "deviation_type": dev_type,
                "capa_required": capa,
                "site_risk_score": round(score if dev_type == "major" else min(score, 4.5), 2),
                "flagged_finding_ids": flagged,
                "recommended_action": (
                    "Escalate to sponsor QA and execute CAPA with effectiveness check."
                    if capa
                    else "Document minor findings and trend under routine monitoring."
                ),
            },
        }

    nr = obs.get("narrative_observation", {})
    demographics = nr.get("patient_demographics", {})
    adverse_event = nr.get("adverse_event", {})
    conmeds = nr.get("concomitant_medications", [])
    labs = nr.get("lab_values_timeline", [])

    age = demographics.get("age", "unknown")
    sex = demographics.get("sex", "unknown")
    case_id = nr.get("case_id", "unknown")
    study_drug = str(nr.get("study_drug", "investigational product"))
    event_term = adverse_event.get("term", "adverse event")
    onset = adverse_event.get("onset_date", "unknown")
    report_date = adverse_event.get("report_date", "unknown")
    seriousness = ", ".join(adverse_event.get("seriousness_criteria", [])) or "medically significant"
    action_taken = str(nr.get("action_taken", "managed per protocol"))
    outcome = str(nr.get("outcome_at_last_followup", "outcome pending"))

    meds = []
    for med in conmeds:
        if isinstance(med, dict):
            meds.append(f"{med.get('name', 'Unknown')} {med.get('dose', '')}".strip())
        else:
            meds.append(str(med))

    lab_lines = []
    for row in labs:
        if not isinstance(row, dict):
            continue
        parts = [f"{k}={v}" for k, v in row.items() if k != "date"]
        if parts:
            lab_lines.append(f"{row.get('date', 'unknown')}: " + ", ".join(parts))

    causality = "probably_related" if str(adverse_event.get("dechallenge_positive", "")).lower() == "true" else "possibly_related"

    narrative_text = (
        f"Case {case_id}: a {age}-year-old {sex} participant received {study_drug}. "
        f"Relevant medical history: {'; '.join(str(x) for x in nr.get('medical_history', [])) or 'as recorded in source documents'}. "
        f"Concomitant medications: {', '.join(meds) or 'none reported'}. "
        f"The subject developed {event_term} with onset on {onset} and report date {report_date}. "
        f"Seriousness criteria included {seriousness}. "
        f"Laboratory timeline: {'; '.join(lab_lines) or 'laboratory data reviewed'}. "
        f"Action taken: {action_taken}. "
        f"Outcome at follow-up: {outcome}. "
        f"Causality assessment: {causality.replace('_', ' ')} based on temporal relationship and clinical course."
    )

    return {
        "task_id": "safety_narrative_generation",
        "safety_narrative": {
            "narrative_text": narrative_text,
            "causality_assessment": causality,
            "key_temporal_flags": [
                f"event onset on {onset}",
                f"report date on {report_date}",
                "timeline reviewed from exposure through outcome",
            ],
            "dechallenge_positive": _to_bool_or_none(adverse_event.get("dechallenge_positive")),
            "rechallenge_positive": _to_bool_or_none(adverse_event.get("rechallenge_done")),
        },
    }


def normalize_action(task_id: str, action: dict, obs: dict) -> Optional[dict]:
    if not isinstance(action, dict):
        return None
    if action.get("task_id") != task_id:
        return None

    if task_id == "adverse_event_triage":
        payload = action.get("ae_triage")
        if not isinstance(payload, dict):
            return None
        severity = str(payload.get("severity_classification", "")).strip().lower()
        timeline = str(payload.get("reporting_timeline", "")).strip().lower()
        if severity not in VALID_AE_SEVERITY or timeline not in VALID_TIMELINE:
            return None
        return {
            "task_id": task_id,
            "ae_triage": {
                "severity_classification": severity,
                "reporting_timeline": timeline,
                "meddra_soc": str(payload.get("meddra_soc", "")).strip() or "General disorders",
                "meddra_preferred_term": str(payload.get("meddra_preferred_term", "")).strip() or "Adverse event",
                "is_serious": bool(payload.get("is_serious", False)),
                "rationale": (str(payload.get("rationale", "")).strip() or "LLM-assisted triage")[:500],
            },
        }

    if task_id == "protocol_deviation_audit":
        payload = action.get("deviation_audit")
        if not isinstance(payload, dict):
            return None
        dev_type = str(payload.get("deviation_type", "")).strip().lower()
        if dev_type not in VALID_DEV_TYPE:
            return None
        try:
            risk = float(payload.get("site_risk_score", 0.0))
        except Exception:  # noqa: BLE001
            return None
        allowed_ids = set(extract_finding_ids(obs))
        flagged = payload.get("flagged_finding_ids", [])
        if not isinstance(flagged, list):
            flagged = []
        filtered = [str(x) for x in flagged if str(x) in allowed_ids]
        return {
            "task_id": task_id,
            "deviation_audit": {
                "deviation_type": dev_type,
                "capa_required": bool(payload.get("capa_required", dev_type == "major")),
                "site_risk_score": max(0.0, min(10.0, risk)),
                "flagged_finding_ids": filtered,
                "recommended_action": (str(payload.get("recommended_action", "")).strip() or "Escalate and track CAPA actions.")[:300],
            },
        }

    payload = action.get("safety_narrative")
    if not isinstance(payload, dict):
        return None
    causality = str(payload.get("causality_assessment", "")).strip().lower()
    if causality not in VALID_CAUSALITY:
        return None

    text = str(payload.get("narrative_text", "")).strip()
    if len(text) < 120:
        return None

    flags = payload.get("key_temporal_flags", [])
    if not isinstance(flags, list):
        flags = []

    return {
        "task_id": task_id,
        "safety_narrative": {
            "narrative_text": text[:4000],
            "causality_assessment": causality,
            "key_temporal_flags": [str(x) for x in flags if str(x).strip()][:8],
            "dechallenge_positive": _to_bool_or_none(payload.get("dechallenge_positive")),
            "rechallenge_positive": _to_bool_or_none(payload.get("rechallenge_positive")),
        },
    }


def choose_action(task_id: str, obs: dict) -> dict:
    prompt = build_prompt(task_id, obs)
    print(f"  Trying LLM for {task_id} step...")
    llm_action = safe_llm_call(prompt)
    if llm_action is not None:
        normalized = normalize_action(task_id, llm_action, obs)
        if normalized is not None:
            print("  LLM action accepted")
            return normalized

    print("  LLM failed, using heuristic fallback")
    return heuristic_action(task_id, obs)


def env_reset(task_id: str, session_id: str) -> dict:
    response = requests.post(
        f"{SERVER_URL}/reset",
        json={"task_id": task_id},
        headers={"X-Session-ID": session_id},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def env_step(action: dict, session_id: str) -> dict:
    response = requests.post(
        f"{SERVER_URL}/step",
        json=action,
        headers={"X-Session-ID": session_id},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def env_grader(session_id: str) -> dict:
    response = requests.get(
        f"{SERVER_URL}/grader",
        headers={"X-Session-ID": session_id},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def run_task(task_id: str) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Task: {task_id}")
    print(f"{'=' * 60}")

    session_id = f"infer-{task_id}-{uuid.uuid4().hex[:8]}"
    rewards: list[float] = []
    error: Optional[str] = None

    try:
        payload = env_reset(task_id, session_id)
    except Exception as exc:  # noqa: BLE001
        error = f"reset_failed: {exc}"
        print(f"  {error}")
        return {
            "mean_reward": 0.0,
            "n_steps": 0,
            "per_step_rewards": [],
            "error": error,
        }

    max_steps = 6
    for _ in range(max_steps):
        done = bool(payload.get("done", False))
        obs = payload.get("observation", payload)
        if done:
            break

        action = choose_action(task_id, obs)
        try:
            step_result = env_step(action, session_id)
        except Exception as exc:  # noqa: BLE001
            error = f"step_failed: {exc}"
            print(f"  {error}")
            break

        reward = float(step_result.get("reward", 0.0))
        rewards.append(reward)
        payload = step_result
        print(f"  reward={reward:.4f} done={bool(step_result.get('done', False))}")

        if bool(step_result.get("done", False)):
            break

    score = 0.0
    try:
        grader = env_grader(session_id)
        score = float(
            grader.get(
                "normalized_score",
                sum(rewards) / max(len(rewards), 1),
            )
        )
    except Exception:  # noqa: BLE001
        score = sum(rewards) / max(len(rewards), 1)

    print(f"  final_score={score:.4f}")
    return {
        "mean_reward": round(score, 6),
        "n_steps": len(rewards),
        "per_step_rewards": rewards,
        "error": error,
    }


def run_all() -> Dict[str, Any]:
    task_results: Dict[str, dict] = {}
    for task_id in TASK_IDS:
        try:
            task_results[task_id] = run_task(task_id)
        except Exception as exc:  # noqa: BLE001
            # Hard fail-safe: one task failure should never crash whole script.
            task_results[task_id] = {
                "mean_reward": 0.0,
                "n_steps": 0,
                "per_step_rewards": [],
                "error": f"task_runner_exception: {exc}",
            }

    means = [float(item.get("mean_reward", 0.0)) for item in task_results.values()]
    mean_score = round(sum(means) / max(len(means), 1), 4)

    return {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "llm_enabled": CLIENT is not None,
        "mean_score": mean_score,
        "overall_mean_reward": mean_score,
        "tasks": task_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_results(summary: Dict[str, Any]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {OUTPUT_FILE}")


def main() -> None:
    print(f"Model : {MODEL_NAME}")
    print(f"Server: {SERVER_URL}")
    print(f"API   : {API_BASE_URL}")
    if CLIENT is None:
        print("LLM disabled (missing/invalid HF_TOKEN or client init failure). Fallback-only mode.")

    summary: Dict[str, Any]
    try:
        summary = run_all()
    except Exception as exc:  # noqa: BLE001
        # Absolute fail-safe: still emit valid output shape.
        summary = {
            "model": MODEL_NAME,
            "api_base_url": API_BASE_URL,
            "llm_enabled": False,
            "mean_score": 0.0,
            "overall_mean_reward": 0.0,
            "tasks": {task_id: {"mean_reward": 0.0, "n_steps": 0, "per_step_rewards": [], "error": str(exc)} for task_id in TASK_IDS},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    write_results(summary)

    print("\nSummary")
    print(f"  mean_score={summary['mean_score']:.4f}")
    print(f"  overall_mean_reward={summary['overall_mean_reward']:.4f}")
    for task_id, task_result in summary["tasks"].items():
        print(f"  {task_id}: {float(task_result.get('mean_reward', 0.0)):.4f}")


if __name__ == "__main__":
    main()
