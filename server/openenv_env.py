"""
OpenEnv-native environment adapter for Clinical Trial Triage.

This module exposes a Meta OpenEnv Environment implementation while reusing
existing domain logic from server.environment. It enables native OpenEnv
HTTP/WebSocket operation via openenv.core.env_server.create_fastapi_app.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Optional

from pydantic import Field, model_validator

from openenv.core.env_server import (
    Action,
    Environment,
    Observation,
    State,
)
from openenv.core.env_server.types import EnvironmentMetadata

from models import (
    AdverseEventTriageAction,
    ProtocolDeviationAction,
    SafetyNarrativeAction,
    TaskID,
    TriageAction,
)
from server.environment import ClinicalTrialEnvironment


_SCORE_EPS = 1e-3


def _clamp_open_score(value: float) -> float:
    return max(_SCORE_EPS, min(1.0 - _SCORE_EPS, float(value)))


class OpenEnvTriageAction(Action):
    """OpenEnv action wrapper for the clinical triage tasks."""

    task_id: TaskID = Field(..., description="Task to execute action against")
    ae_triage: Optional[AdverseEventTriageAction] = None
    deviation_audit: Optional[ProtocolDeviationAction] = None
    safety_narrative: Optional[SafetyNarrativeAction] = None

    @model_validator(mode="after")
    def validate_task_payload(self) -> "OpenEnvTriageAction":
        has_ae = self.ae_triage is not None
        has_dev = self.deviation_audit is not None
        has_nr = self.safety_narrative is not None
        if sum([has_ae, has_dev, has_nr]) != 1:
            raise ValueError(
                "Exactly one payload must be provided: ae_triage, deviation_audit, or safety_narrative"
            )

        if self.task_id == TaskID.ADVERSE_EVENT_TRIAGE and not has_ae:
            raise ValueError("task_id=adverse_event_triage requires ae_triage payload")
        if self.task_id == TaskID.PROTOCOL_DEVIATION_AUDIT and not has_dev:
            raise ValueError("task_id=protocol_deviation_audit requires deviation_audit payload")
        if self.task_id == TaskID.SAFETY_NARRATIVE_GENERATION and not has_nr:
            raise ValueError("task_id=safety_narrative_generation requires safety_narrative payload")
        return self


class OpenEnvTriageObservation(Observation):
    """OpenEnv observation wrapper with full domain payload."""

    task_id: TaskID
    payload: Dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class OpenEnvTriageState(State):
    """OpenEnv state object for environment introspection."""

    task_id: Optional[TaskID] = None
    max_steps: int = 0
    done: bool = False
    cumulative_reward: float = 0.0
    current_case_id: Optional[str] = None


class ClinicalTrialOpenEnv(
    Environment[OpenEnvTriageAction, OpenEnvTriageObservation, OpenEnvTriageState]
):
    """
    Native OpenEnv environment implementation for clinical trial triage.

    Supports:
    - task-specific episodes via reset(task_id=...)
    - mixed-task curriculum via reset(task_id="mixed")
    - complete reward + done semantics in OpenEnv Observation
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        super().__init__()
        self._core = ClinicalTrialEnvironment()
        self._task_rng = random.Random(42)
        self._last_task_id: Optional[TaskID] = None

    def _to_openenv_observation(
        self,
        payload: Dict[str, Any],
        task_id: TaskID,
        reward: Optional[float],
        done: bool,
        message: str,
        reward_detail: Optional[Dict[str, Any]] = None,
    ) -> OpenEnvTriageObservation:
        metadata = {}
        if reward_detail is not None:
            metadata["reward_detail"] = reward_detail

        return OpenEnvTriageObservation(
            task_id=task_id,
            payload=payload,
            message=message,
            reward=reward,
            done=done,
            metadata=metadata,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OpenEnvTriageObservation:
        if seed is not None:
            self._task_rng.seed(seed)

        requested_task = kwargs.get("task_id", TaskID.ADVERSE_EVENT_TRIAGE)
        if requested_task == "mixed":
            chosen_task = self._task_rng.choice(
                [
                    TaskID.ADVERSE_EVENT_TRIAGE,
                    TaskID.PROTOCOL_DEVIATION_AUDIT,
                    TaskID.SAFETY_NARRATIVE_GENERATION,
                ]
            )
        else:
            chosen_task = TaskID(requested_task)

        self._last_task_id = chosen_task
        obs = self._core.reset(task_id=chosen_task)
        payload = obs.model_dump()
        return self._to_openenv_observation(
            payload=payload,
            task_id=chosen_task,
            reward=None,
            done=False,
            message=obs.message,
        )

    def step(
        self,
        action: OpenEnvTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> OpenEnvTriageObservation:
        triage_action = TriageAction(
            task_id=action.task_id,
            ae_triage=action.ae_triage,
            deviation_audit=action.deviation_audit,
            safety_narrative=action.safety_narrative,
        )

        step_result = self._core.step(triage_action)
        obs = step_result.observation
        payload = obs.model_dump()
        return self._to_openenv_observation(
            payload=payload,
            task_id=TaskID(obs.task_id),
            reward=step_result.reward,
            done=step_result.done,
            message=obs.message,
            reward_detail=step_result.reward_detail.model_dump(),
        )

    @property
    def state(self) -> OpenEnvTriageState:
        state = self._core.state()
        normalized_cumulative = _clamp_open_score(
            state.cumulative_reward / state.step_count if state.step_count > 0 else _SCORE_EPS
        )
        return OpenEnvTriageState(
            episode_id=state.episode_id,
            step_count=state.step_count,
            task_id=TaskID(state.task_id),
            max_steps=state.max_steps,
            done=state.done,
            cumulative_reward=normalized_cumulative,
            current_case_id=state.current_case_id,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Clinical Trial Triage OpenEnv",
            description=(
                "Production-grade multi-task environment for adverse event triage, "
                "protocol deviation auditing, and safety narrative generation."
            ),
            version="2.0.0",
            author="OpenEnv Hackathon Submission",
            documentation_url="/docs",
        )

    def close(self) -> None:
        # Core env has no external handles to close today.
        return
