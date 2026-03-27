from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models import TaskID, TriageObservation
from rl.action_templates import action_from_vector, action_space_nvec
from rl.featurizer import FEATURE_DIM, encode_observation
from server.environment import ClinicalTrialEnvironment


_TASKS = [
    TaskID.ADVERSE_EVENT_TRIAGE.value,
    TaskID.PROTOCOL_DEVIATION_AUDIT.value,
    TaskID.SAFETY_NARRATIVE_GENERATION.value,
]


class ClinicalTrialGymEnv(gym.Env):
    """
    Gymnasium wrapper around the typed clinical-trial environment.

    Features:
    - mixed-task curriculum mode
    - deterministic seeding
    - standardized continuous observation vector
    - shared parameterized MultiDiscrete policy head across all tasks
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        task_mode: str = "mixed",
        reward_scale: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.task_mode = task_mode
        self.reward_scale = reward_scale
        self._rng = random.Random(seed)
        self._episode_idx = 0

        self._env = ClinicalTrialEnvironment()
        self._last_observation: Optional[TriageObservation] = None

        self.action_space = spaces.MultiDiscrete(action_space_nvec())
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(FEATURE_DIM,),
            dtype=np.float32,
        )

    def _sample_task(self) -> str:
        if self.task_mode in _TASKS:
            return self.task_mode
        # Curriculum: AE-heavy early, then balanced.
        if self._episode_idx < 200:
            return self._rng.choices(
                population=_TASKS,
                weights=[0.65, 0.25, 0.10],
                k=1,
            )[0]
        return self._rng.choice(_TASKS)

    def _observation_payload(self, obs: TriageObservation) -> Dict[str, Any]:
        if obs.ae_observation is not None:
            return obs.ae_observation.model_dump()
        if obs.deviation_observation is not None:
            return obs.deviation_observation.model_dump()
        if obs.narrative_observation is not None:
            return obs.narrative_observation.model_dump()
        return {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        requested_task = None
        if options:
            requested_task = options.get("task_id")
        task_id = requested_task or self._sample_task()

        self._episode_idx += 1
        obs = self._env.reset(task_id=task_id)
        self._last_observation = obs
        vec = encode_observation(obs.model_dump())

        return vec, {"task_id": task_id, "episode_index": self._episode_idx}

    def step(self, action: Any):
        if self._last_observation is None:
            raise RuntimeError("reset() must be called before step().")

        task_value = self._last_observation.task_id
        task_id = task_value.value if hasattr(task_value, "value") else str(task_value)
        payload = self._observation_payload(self._last_observation)
        triage_action = action_from_vector(task_id, action, payload)

        result = self._env.step(triage_action)
        self._last_observation = result.observation

        vec = encode_observation(result.observation.model_dump())
        reward = float(result.reward) * self.reward_scale
        terminated = bool(result.done)
        truncated = False

        info = {
            "task_id": task_id,
            "raw_action": np.asarray(action).tolist() if not np.isscalar(action) else int(action),
            "raw_reward": float(result.reward),
            "reward_detail": result.reward_detail.model_dump(),
            "done": bool(result.done),
        }
        return vec, reward, terminated, truncated, info

    def render(self):
        if self._last_observation is None:
            print("No active episode")
            return
        print(self._last_observation.message)

    def close(self):
        return
