from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    total_timesteps: int = 120_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 256
    batch_size: int = 128
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    clip_range: float = 0.2
    num_envs: int = 4
    seed: int = 42
    task_mode: str = "mixed"
    reward_scale: float = 1.0


@dataclass
class EvalConfig:
    episodes_per_task: int = 20
    deterministic: bool = True
    seed: int = 123
