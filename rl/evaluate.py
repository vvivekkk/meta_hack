from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from stable_baselines3 import PPO

from models import TaskID
from rl.config import EvalConfig
from rl.gym_env import ClinicalTrialGymEnv


def evaluate_model(
    model_path: str,
    episodes_per_task: int = 20,
    deterministic: bool = True,
    seed: int = 123,
) -> Dict[str, Any]:
    model = PPO.load(model_path)

    per_task: Dict[str, Dict[str, Any]] = {}
    task_rewards: List[float] = []

    for task_id in [
        TaskID.ADVERSE_EVENT_TRIAGE,
        TaskID.PROTOCOL_DEVIATION_AUDIT,
        TaskID.SAFETY_NARRATIVE_GENERATION,
    ]:
        task_value = task_id.value
        env = ClinicalTrialGymEnv(task_mode=task_value, seed=seed)
        episode_rewards: List[float] = []
        penalty_events = 0

        for _ in range(episodes_per_task):
            obs, _ = env.reset(options={"task_id": task_value})
            done = False
            reward_sum = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                reward_sum += float(reward)
                if info.get("reward_detail", {}).get("penalty_applied"):
                    penalty_events += 1
            episode_rewards.append(reward_sum)

        env.close()
        per_task[task_value] = {
            "mean_episode_reward": round(mean(episode_rewards), 6),
            "min_episode_reward": round(min(episode_rewards), 6),
            "max_episode_reward": round(max(episode_rewards), 6),
            "penalty_event_count": penalty_events,
            "episodes": episodes_per_task,
        }
        task_rewards.append(mean(episode_rewards))

    report = {
        "model_path": model_path,
        "episodes_per_task": episodes_per_task,
        "deterministic": deterministic,
        "overall_mean_reward": round(mean(task_rewards), 6),
        "per_task": per_task,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy for clinical triage.")
    parser.add_argument("--model-path", required=True, help="Path to PPO model zip file")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy sampling")
    parser.add_argument("--output", default="outputs/rl/eval_report.json")
    args = parser.parse_args()

    config = EvalConfig(
        episodes_per_task=args.episodes_per_task,
        deterministic=not args.stochastic,
        seed=args.seed,
    )
    report = evaluate_model(
        model_path=args.model_path,
        episodes_per_task=config.episodes_per_task,
        deterministic=config.deterministic,
        seed=config.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
