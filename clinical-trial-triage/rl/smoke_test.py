from __future__ import annotations

import json
from pathlib import Path

from rl.config import TrainingConfig
from rl.train import train_policy


def main() -> None:
    output_dir = Path("outputs/rl/smoke")
    config = TrainingConfig(
        total_timesteps=2048,
        num_envs=2,
        n_steps=128,
        batch_size=64,
        task_mode="mixed",
        seed=7,
    )
    summary = train_policy(config=config, output_dir=str(output_dir))
    print(json.dumps(summary["evaluation"], indent=2))


if __name__ == "__main__":
    main()
