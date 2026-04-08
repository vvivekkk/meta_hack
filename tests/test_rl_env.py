from __future__ import annotations

import numpy as np

from rl.featurizer import FEATURE_DIM
from rl.gym_env import ClinicalTrialGymEnv


def test_rl_env_reset_and_step() -> None:
    env = ClinicalTrialGymEnv(task_mode="mixed", seed=11)
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (FEATURE_DIM,)
    assert "task_id" in info

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (FEATURE_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "task_id" in step_info

    env.close()
