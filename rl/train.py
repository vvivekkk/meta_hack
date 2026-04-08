from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.config import TrainingConfig
from rl.evaluate import evaluate_model
from rl.gym_env import ClinicalTrialGymEnv


def _make_env(task_mode: str, reward_scale: float, seed: int):
    def _factory():
        env = ClinicalTrialGymEnv(task_mode=task_mode, reward_scale=reward_scale, seed=seed)
        return Monitor(env)

    return _factory


def train_policy(config: TrainingConfig, output_dir: str) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv(
        [_make_env(config.task_mode, config.reward_scale, config.seed + i) for i in range(config.num_envs)]
    )

    eval_env = DummyVecEnv([_make_env(config.task_mode, config.reward_scale, config.seed + 999)])

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        clip_range=config.clip_range,
        seed=config.seed,
        verbose=1,
    )

    best_model_dir = out / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(out / "logs"),
        eval_freq=max(1000, config.n_steps),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=config.total_timesteps, callback=eval_callback)

    final_model_path = out / "ppo_clinical_triage"
    model.save(str(final_model_path))

    final_eval = evaluate_model(
        model_path=str(final_model_path) + ".zip",
        episodes_per_task=10,
        deterministic=True,
        seed=config.seed,
    )

    summary = {
        "config": {
            "total_timesteps": config.total_timesteps,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "ent_coef": config.ent_coef,
            "vf_coef": config.vf_coef,
            "clip_range": config.clip_range,
            "num_envs": config.num_envs,
            "seed": config.seed,
            "task_mode": config.task_mode,
            "reward_scale": config.reward_scale,
        },
        "artifacts": {
            "final_model": str(final_model_path) + ".zip",
            "best_model_dir": str(best_model_dir),
        },
        "evaluation": final_eval,
    }

    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    train_env.close()
    eval_env.close()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO policy for clinical trial triage environment")
    parser.add_argument("--total-timesteps", type=int, default=120000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task-mode",
        choices=[
            "mixed",
            "adverse_event_triage",
            "protocol_deviation_audit",
            "safety_narrative_generation",
        ],
        default="mixed",
    )
    parser.add_argument("--output-dir", default="outputs/rl/train")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    args = parser.parse_args()

    config = TrainingConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        clip_range=args.clip_range,
        num_envs=args.num_envs,
        seed=args.seed,
        task_mode=args.task_mode,
        reward_scale=args.reward_scale,
    )

    summary = train_policy(config=config, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
