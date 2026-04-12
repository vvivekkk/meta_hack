"""
Clinical Trial Triage — Environment (Server-Side)
===================================================
Implements the OpenEnv Environment base with reset(), step(), state().
Full episode management, reward shaping, and multi-task support.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from models import (
    AdverseEventObservation,
    AdverseEventTriageAction,
    ProtocolDeviationAction,
    ProtocolDeviationObservation,
    SafetyNarrativeAction,
    SafetyNarrativeObservation,
    StepResult,
    TaskID,
    TriageAction,
    TriageObservation,
    TriageReward,
    TriageState,
)
from tasks.case_bank import AE_CASES, DEVIATION_CASES, NARRATIVE_CASES  # noqa: E402
from tasks.graders import (  # noqa: E402
    grade_ae_triage,
    grade_protocol_deviation,
    grade_safety_narrative,
)

# Task → max steps configuration
TASK_MAX_STEPS: Dict[str, int] = {
    TaskID.ADVERSE_EVENT_TRIAGE: 3,         # 3 AE cases per episode
    TaskID.PROTOCOL_DEVIATION_AUDIT: 3,     # 3 site audits per episode
    TaskID.SAFETY_NARRATIVE_GENERATION: 1,  # 1 complex narrative per episode
}

TASK_CASES: Dict[str, list] = {
    TaskID.ADVERSE_EVENT_TRIAGE: AE_CASES,
    TaskID.PROTOCOL_DEVIATION_AUDIT: DEVIATION_CASES,
    TaskID.SAFETY_NARRATIVE_GENERATION: NARRATIVE_CASES,
}

_SCORE_EPS = 1e-6


_sessions: Dict[str, "ClinicalTrialEnvironment"] = {}
_sessions_lock = threading.Lock()


def get_or_create_session(session_id: str = "default") -> "ClinicalTrialEnvironment":
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = ClinicalTrialEnvironment()
        return _sessions[session_id]


def clear_session(session_id: str = "default") -> None:
    with _sessions_lock:
        _sessions.pop(session_id, None)


class ClinicalTrialEnvironment:
    """
    Main environment class implementing OpenEnv-compatible APIs.

    Episode lifecycle:
        reset(task_id) → initial observation
        step(action)   → observation, reward, done, info
        state()        → episode metadata
    """

    def __init__(self) -> None:
        self._state: Optional[TriageState] = None
        self._case_index: int = 0
        self._current_task: Optional[TaskID] = None
        self._cumulative_reward: float = 0.0
        self._actions_log: list = []
        self._last_action_signature: Optional[str] = None

    # ─────────────────────────────────────
    # PUBLIC OPENENV API
    # ─────────────────────────────────────

    def reset(self, task_id: str = TaskID.ADVERSE_EVENT_TRIAGE) -> TriageObservation:
        """Initialize a new episode for the given task."""
        task = TaskID(task_id)
        self._current_task = task
        self._case_index = 0
        self._cumulative_reward = 0.0
        self._actions_log = []
        self._last_action_signature = None

        self._state = TriageState(
            episode_id=str(uuid.uuid4()),
            task_id=task,
            step_count=0,
            max_steps=TASK_MAX_STEPS[task],
            done=False,
            cumulative_reward=0.0,
            actions_taken=[],
            current_case_id=self._get_current_case_id(),
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        return self._build_observation()

    def step(self, action: TriageAction) -> StepResult:
        """Execute one action and return (observation, reward, done, info)."""
        if self._state is None or self._state.done:
            raise RuntimeError("Call reset() before step(), or episode is already done.")

        if TaskID(action.task_id) != self._current_task:
            raise ValueError(
                f"Action task_id '{action.task_id}' does not match "
                f"current episode task '{self._current_task}'."
            )

        current_observation = self._build_observation()

        # Grade this step
        reward_detail = self._grade(action)
        step_reward = reward_detail.total

        # Reward shaping: small partial reward for being in the right direction
        # This gives signal across the trajectory, not just at episode end
        shaped_reward = self._shape_reward(step_reward, reward_detail, action, current_observation)

        # Update state
        self._cumulative_reward += shaped_reward
        self._case_index += 1
        self._state.step_count += 1
        self._state.cumulative_reward = self._cumulative_reward
        self._state.actions_taken.append(
            {"step": self._state.step_count, "action": action.model_dump(), "reward": shaped_reward}
        )

        done = self._state.step_count >= self._state.max_steps
        self._state.done = done
        if done:
            self._state.completed_at = datetime.now(timezone.utc).isoformat()
            self._state.current_case_id = None
        else:
            self._state.current_case_id = self._get_current_case_id()

        obs = self._build_observation()

        return StepResult(
            observation=obs,
            reward=shaped_reward,
            reward_detail=reward_detail,
            done=done,
            info={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
                "cumulative_reward": self._cumulative_reward,
                "done": done,
            },
        )

    def state(self) -> TriageState:
        """Return current episode state metadata."""
        if self._state is None:
            raise RuntimeError("No active episode. Call reset() first.")
        return self._state

    # ─────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────

    def _get_current_case_id(self) -> Optional[str]:
        cases = TASK_CASES.get(self._current_task, [])
        if self._case_index < len(cases):
            case = cases[self._case_index]
            return case.get("case_id") or case.get("site_id")
        return None

    def _build_observation(self) -> TriageObservation:
        """Build the typed observation for the current state."""
        if self._state is None:
            raise RuntimeError("No active state.")

        cases = TASK_CASES[self._current_task]
        step = self._state.step_count
        max_steps = self._state.max_steps

        if self._state.done or self._case_index >= len(cases):
            return TriageObservation(
                task_id=self._current_task,
                message=f"Episode complete. Cumulative reward: {self._cumulative_reward:.4f}",
            )

        case = cases[self._case_index]

        if self._current_task == TaskID.ADVERSE_EVENT_TRIAGE:
            obs = AdverseEventObservation(
                case_id=case["case_id"],
                narrative=case["narrative"],
                patient_age=case["patient_age"],
                patient_sex=case["patient_sex"],
                study_drug=case["study_drug"],
                dose_mg=case["dose_mg"],
                days_on_drug=case["days_on_drug"],
                relevant_medical_history=case["relevant_medical_history"],
                concomitant_medications=case["concomitant_medications"],
                lab_values=case["lab_values"],
                ae_onset_date=case["ae_onset_date"],
                ae_description=case["ae_description"],
                outcome=case["outcome"],
                step_count=step,
                max_steps=max_steps,
            )
            return TriageObservation(
                task_id=self._current_task,
                ae_observation=obs,
                message=f"Step {step + 1}/{max_steps}: Classify the adverse event.",
            )

        elif self._current_task == TaskID.PROTOCOL_DEVIATION_AUDIT:
            obs = ProtocolDeviationObservation(
                site_id=case["site_id"],
                site_name=case["site_name"],
                visit_type=case["visit_type"],
                findings=case["findings"],
                prior_deviations=case["prior_deviations"],
                active_subjects=case["active_subjects"],
                study_phase=case["study_phase"],
                last_monitoring_visit=case["last_monitoring_visit"],
                step_count=step,
                max_steps=max_steps,
            )
            return TriageObservation(
                task_id=self._current_task,
                deviation_observation=obs,
                message=f"Step {step + 1}/{max_steps}: Audit the site findings.",
            )

        elif self._current_task == TaskID.SAFETY_NARRATIVE_GENERATION:
            obs = SafetyNarrativeObservation(
                case_id=case["case_id"],
                patient_demographics=case["patient_demographics"],
                study_drug=case["study_drug"],
                suspect_drugs=case["suspect_drugs"],
                concomitant_medications=case["concomitant_medications"],
                adverse_event=case["adverse_event"],
                lab_values_timeline=case["lab_values_timeline"],
                medical_history=case["medical_history"],
                action_taken=case["action_taken"],
                outcome_at_last_followup=case["outcome_at_last_followup"],
                reference_documents=case["reference_documents"],
                step_count=step,
                max_steps=max_steps,
            )
            return TriageObservation(
                task_id=self._current_task,
                narrative_observation=obs,
                message=f"Step {step + 1}/{max_steps}: Write the ICSR safety narrative.",
            )

        return TriageObservation(task_id=self._current_task, message="Unknown task state.")

    def _grade(self, action: TriageAction) -> TriageReward:
        """Route grading to the correct task grader."""
        cases = TASK_CASES[self._current_task]
        if self._case_index >= len(cases):
            return TriageReward(total=0.0)

        case = cases[self._case_index]

        if self._current_task == TaskID.ADVERSE_EVENT_TRIAGE:
            if action.ae_triage is None:
                return TriageReward(
                    total=0.0, penalty_applied=True,
                    penalty_reason="No ae_triage action provided for adverse_event_triage task."
                )
            return grade_ae_triage(action.ae_triage, case)

        elif self._current_task == TaskID.PROTOCOL_DEVIATION_AUDIT:
            if action.deviation_audit is None:
                return TriageReward(
                    total=0.0, penalty_applied=True,
                    penalty_reason="No deviation_audit action provided for protocol_deviation_audit task."
                )
            return grade_protocol_deviation(action.deviation_audit, case)

        elif self._current_task == TaskID.SAFETY_NARRATIVE_GENERATION:
            if action.safety_narrative is None:
                return TriageReward(
                    total=0.0, penalty_applied=True,
                    penalty_reason="No safety_narrative action provided for safety_narrative_generation task."
                )
            return grade_safety_narrative(action.safety_narrative, case)

        return TriageReward(total=0.0)

    def _shape_reward(
        self,
        raw_reward: float,
        detail: TriageReward,
        action: TriageAction,
        current_observation: TriageObservation,
    ) -> float:
        """
        Apply reward shaping to ensure dense signals across the trajectory.
        - Small bonus for partial progress (>0.3 total)
        - Deduction for penalty-flagged actions
        - No bonus for trivially wrong answers
        """
        shaped = raw_reward

        # Partial progress signal
        if 0.3 <= raw_reward < 0.6:
            shaped += 0.02  # small encouragement signal

        # Penalty deduction
        if detail.penalty_applied:
            shaped = max(0.0, shaped - 0.05)

        # Anti-gaming penalty for obviously inflated severity when narrative implies mild signal.
        if (
            action.task_id == TaskID.ADVERSE_EVENT_TRIAGE
            and action.ae_triage is not None
            and current_observation.ae_observation is not None
        ):
            ae_action = action.ae_triage
            narrative = current_observation.ae_observation.narrative.lower()
            narrative_implies_mild = any(word in narrative for word in ["mild", "minor", "slight", "minimal"])
            narrative_has_critical = any(
                word in narrative for word in ["life-threatening", "critical", "icu", "intubat", "cardiac arrest"]
            )
            severity_value = getattr(ae_action.severity_classification, "value", ae_action.severity_classification)
            agent_says_life_threatening = str(severity_value) == "life_threatening"
            if narrative_implies_mild and not narrative_has_critical and agent_says_life_threatening:
                shaped -= 0.15

        # Anti-loop penalty for repeating identical consecutive actions.
        action_signature = json.dumps(action.model_dump(mode="json"), sort_keys=True)
        if self._last_action_signature == action_signature:
            shaped -= 0.05
        self._last_action_signature = action_signature

        return round(min(1.0 - _SCORE_EPS, max(_SCORE_EPS, shaped)), 6)