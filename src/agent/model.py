from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True


def create_model(
    policy: str,
    env,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
    net_arch: list[int] | None = None,
    tensorboard_log: str | None = None,
) -> PPO:
    net_arch = net_arch or [128, 128]
    policy_kwargs = {"net_arch": dict(pi=net_arch, vf=net_arch)}
    return PPO(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1,
    )


def save_model(model: PPO, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_model(path: str, env=None) -> PPO:
    return PPO.load(path, env=env)
