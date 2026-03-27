"""
LLM Baseline Inference Script (GroqCloud)
=========================================
Runs a GroqCloud-backed LLM baseline against all 3 tasks and produces
reproducible scores. If GROQ_API_KEY is not set, it falls back to the
deterministic heuristic baseline so the script always completes.

Usage:
    GROQ_API_KEY=... python scripts/baseline_inference.py

Optional environment variables:
    GROQ_BASE_URL=https://api.groq.com
    GROQ_API_KEYS=gsk_key_1,gsk_key_2,gsk_key_3
    BASELINE_MODEL=llama-3.3-70b-versatile
    GROQ_KEY_STATE_FILE=outputs/groq_key_usage.json

Output:
    outputs/baseline_results.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    AdverseEventTriageAction,
    ProtocolDeviationAction,
    SafetyNarrativeAction,
    TaskID,
    TriageAction,
)
from scripts.groq_key_pool import GroqKeyPool, parse_groq_keys
from server.environment import ClinicalTrialEnvironment

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

API_KEY = os.environ.get("GROQ_API_KEY", "")
API_KEYS_CSV = os.environ.get("GROQ_API_KEYS", "")
BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com")
MODEL = os.environ.get("BASELINE_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE = 0.0
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
KEY_STATE_FILE = Path(
    os.environ.get(
        "GROQ_KEY_STATE_FILE",
        str(OUTPUT_DIR / "groq_key_usage.json"),
    )
)


# -----------------------------------------------------------------------------
# SYSTEM PROMPTS
# -----------------------------------------------------------------------------

AE_SYSTEM_PROMPT = """You are an expert clinical research pharmacovigilance specialist.

Return only a valid JSON object with these exact fields:
{
  "severity_classification": "mild|moderate|severe|life_threatening|fatal",
  "reporting_timeline": "7-day|15-day|routine",
  "meddra_soc": "string",
  "meddra_preferred_term": "string",
  "is_serious": true,
  "rationale": "string up to 500 chars"
}
"""

DEVIATION_SYSTEM_PROMPT = """You are a senior GCP auditor.

Return only a valid JSON object with these exact fields:
{
  "deviation_type": "major|minor|protocol_amendment",
  "capa_required": true,
  "site_risk_score": 0.0,
  "flagged_finding_ids": ["F001"],
  "recommended_action": "string up to 300 chars"
}
"""

NARRATIVE_SYSTEM_PROMPT = """You are a regulatory medical writer.

Return only a valid JSON object with these exact fields:
{
  "narrative_text": "100-4000 chars",
  "causality_assessment": "definitely_related|probably_related|possibly_related|unlikely_related|not_related|unassessable",
  "key_temporal_flags": ["string"],
  "dechallenge_positive": true,
  "rechallenge_positive": null
}
"""


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction when model returns extra text around JSON."""
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class LLMAgent:
    """GroqCloud-backed agent with strict JSON parsing and retries."""

    def __init__(self, key_pool: GroqKeyPool, model: str = MODEL):
        self.key_pool = key_pool
        self.model = model

    def _call(self, system_prompt: str, user_content: str, retries: int = 3) -> Optional[Dict[str, Any]]:
        for attempt in range(retries):
            key_id = self.key_pool.acquire_key()
            if key_id is None:
                print(f"  [Attempt {attempt + 1}] No Groq API key available")
                time.sleep(2**attempt)
                continue

            client = self.key_pool.get_client(key_id)
            self.key_pool.mark_request(key_id)
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    temperature=TEMPERATURE,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                raw = response.choices[0].message.content or ""
                parsed = _extract_json_object(raw)
                if parsed is not None:
                    self.key_pool.mark_success(key_id)
                    return parsed
                self.key_pool.mark_failure(key_id, "invalid json response")
                print(f"  [Attempt {attempt + 1}] Could not parse JSON response")
            except Exception as exc:
                self.key_pool.mark_failure(key_id, str(exc))
                print(f"  [Attempt {attempt + 1}] API error: {exc}")

            time.sleep(2**attempt)

        return None

    def act_ae_triage(self, obs_dict: Dict[str, Any]) -> Optional[TriageAction]:
        ae_obs = obs_dict.get("ae_observation", {})
        user_content = f"""
ADVERSE EVENT CASE
==================
Case ID: {ae_obs.get('case_id')}
Patient: {ae_obs.get('patient_age')}y {ae_obs.get('patient_sex')}
Study Drug: {ae_obs.get('study_drug')} {ae_obs.get('dose_mg')}mg
Days on Drug: {ae_obs.get('days_on_drug')}

Narrative:
{ae_obs.get('narrative')}

AE Description: {ae_obs.get('ae_description')}
Outcome: {ae_obs.get('outcome')}
Medical History: {', '.join(ae_obs.get('relevant_medical_history', []))}
Concomitant Medications: {', '.join(ae_obs.get('concomitant_medications', []))}
Lab Values: {json.dumps(ae_obs.get('lab_values', {}), indent=2)}
"""

        result = self._call(AE_SYSTEM_PROMPT, user_content)
        if not result:
            return None

        try:
            return TriageAction(
                task_id=TaskID.ADVERSE_EVENT_TRIAGE,
                ae_triage=AdverseEventTriageAction(**result),
            )
        except Exception as exc:
            print(f"  Action parse error: {exc}")
            return None

    def act_deviation_audit(self, obs_dict: Dict[str, Any]) -> Optional[TriageAction]:
        dev_obs = obs_dict.get("deviation_observation", {})
        findings_str = "\n".join(
            f"  [{f['id']}] {f['category']}: {f['description']}"
            for f in dev_obs.get("findings", [])
        )

        user_content = f"""
SITE AUDIT FINDINGS
===================
Site: {dev_obs.get('site_id')} - {dev_obs.get('site_name')}
Visit Type: {dev_obs.get('visit_type')}
Study Phase: {dev_obs.get('study_phase')}
Active Subjects: {dev_obs.get('active_subjects')}
Prior Deviations: {dev_obs.get('prior_deviations')}
Last Monitoring: {dev_obs.get('last_monitoring_visit')}

Findings:
{findings_str}
"""

        result = self._call(DEVIATION_SYSTEM_PROMPT, user_content)
        if not result:
            return None

        try:
            return TriageAction(
                task_id=TaskID.PROTOCOL_DEVIATION_AUDIT,
                deviation_audit=ProtocolDeviationAction(**result),
            )
        except Exception as exc:
            print(f"  Action parse error: {exc}")
            return None

    def act_safety_narrative(self, obs_dict: Dict[str, Any]) -> Optional[TriageAction]:
        nr_obs = obs_dict.get("narrative_observation", {})
        user_content = f"""
CASE FOR ICSR NARRATIVE
=======================
Case ID: {nr_obs.get('case_id')}
Patient: {json.dumps(nr_obs.get('patient_demographics', {}), indent=2)}
Study Drug: {nr_obs.get('study_drug')}
Suspect Drugs: {nr_obs.get('suspect_drugs')}
Concomitant Medications: {json.dumps(nr_obs.get('concomitant_medications', []), indent=2)}
Adverse Event: {json.dumps(nr_obs.get('adverse_event', {}), indent=2)}
Lab Values Timeline: {json.dumps(nr_obs.get('lab_values_timeline', []), indent=2)}
Medical History: {nr_obs.get('medical_history')}
Action Taken: {nr_obs.get('action_taken')}
Outcome: {nr_obs.get('outcome_at_last_followup')}
Reference Documents: {nr_obs.get('reference_documents')}
"""

        result = self._call(NARRATIVE_SYSTEM_PROMPT, user_content)
        if not result:
            return None

        try:
            return TriageAction(
                task_id=TaskID.SAFETY_NARRATIVE_GENERATION,
                safety_narrative=SafetyNarrativeAction(**result),
            )
        except Exception as exc:
            print(f"  Action parse error: {exc}")
            return None


def _run_task(env: ClinicalTrialEnvironment, task_id: TaskID, agent: LLMAgent, max_steps: int) -> Dict[str, Any]:
    """Run one task and return per-step rewards and details."""
    obs_dict = env.reset(task_id=task_id).model_dump()
    rewards: List[float] = []
    details: List[Dict[str, Any]] = []

    for _ in range(max_steps):
        if task_id == TaskID.ADVERSE_EVENT_TRIAGE:
            action = agent.act_ae_triage(obs_dict)
        elif task_id == TaskID.PROTOCOL_DEVIATION_AUDIT:
            action = agent.act_deviation_audit(obs_dict)
        else:
            action = agent.act_safety_narrative(obs_dict)

        if action is None:
            rewards.append(0.0)
            details.append({"error": "agent_failed_to_produce_valid_action"})
            continue

        step_result = env.step(action)
        rewards.append(step_result.reward)
        details.append(step_result.reward_detail.model_dump())
        obs_dict = step_result.observation.model_dump()
        if step_result.done:
            break

    return {
        "per_step_rewards": rewards,
        "mean_reward": round(sum(rewards) / max(len(rewards), 1), 4),
        "n_steps": len(rewards),
        "details": details,
    }


def run_llm_baseline() -> Dict[str, Any]:
    """Run GroqCloud LLM baseline or deterministic fallback."""
    api_keys = parse_groq_keys(api_key=API_KEY, api_keys_csv=API_KEYS_CSV)
    if not api_keys:
        from scripts.heuristic_baseline import run_heuristic_baseline

        fallback = run_heuristic_baseline()
        fallback["baseline_type"] = "heuristic_fallback"
        fallback["reason"] = "No Groq key found (GROQ_API_KEY or GROQ_API_KEYS); used deterministic heuristic baseline."

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "baseline_results.json"
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(fallback, file, indent=2)

        print(json.dumps(fallback, indent=2))
        print(f"\nResults saved to: {output_path}")
        return fallback

    key_pool = GroqKeyPool(
        api_keys=api_keys,
        base_url=BASE_URL,
        state_file=KEY_STATE_FILE,
    )
    agent = LLMAgent(key_pool=key_pool, model=MODEL)
    env = ClinicalTrialEnvironment()

    results: Dict[str, Any] = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "baseline_type": "groq_llm",
        "provider": "groqcloud",
        "key_pool": key_pool.snapshot(),
        "tasks": {},
    }

    results["tasks"][TaskID.ADVERSE_EVENT_TRIAGE] = _run_task(
        env=env,
        task_id=TaskID.ADVERSE_EVENT_TRIAGE,
        agent=agent,
        max_steps=3,
    )
    results["tasks"][TaskID.PROTOCOL_DEVIATION_AUDIT] = _run_task(
        env=env,
        task_id=TaskID.PROTOCOL_DEVIATION_AUDIT,
        agent=agent,
        max_steps=3,
    )
    results["tasks"][TaskID.SAFETY_NARRATIVE_GENERATION] = _run_task(
        env=env,
        task_id=TaskID.SAFETY_NARRATIVE_GENERATION,
        agent=agent,
        max_steps=1,
    )

    all_means = [task_result["mean_reward"] for task_result in results["tasks"].values()]
    results["overall_mean_reward"] = round(sum(all_means) / len(all_means), 4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "baseline_results.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    run_llm_baseline()