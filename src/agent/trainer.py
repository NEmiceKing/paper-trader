from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.agent.env import TradingEnv
from src.agent.model import create_model, save_model


def split_timeseries(data: np.ndarray, prices: np.ndarray, train_pct=0.7, val_pct=0.15) -> tuple:
    n = len(data)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    return (
        (data[:train_end], prices[:train_end]),
        (data[train_end:val_end], prices[train_end:val_end]),
        (data[val_end:], prices[val_end:]),
    )


def train_agent(
    features: np.ndarray,
    prices: np.ndarray,
    config,
    model_dir: str = "models",
) -> tuple[PPO, dict]:
    (train_feat, train_prices), (val_feat, val_prices), (test_feat, test_prices) = split_timeseries(
        features, prices
    )

    train_env = Monitor(TradingEnv(
        train_feat, train_prices,
        window=config.agent.observation_bars,
        reward_fn=config.agent.reward_fn,
    ))

    val_env = Monitor(TradingEnv(
        val_feat, val_prices,
        window=config.agent.observation_bars,
        reward_fn=config.agent.reward_fn,
    ))

    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(Path(model_dir) / "best"),
        log_path=str(Path(model_dir) / "logs"),
        eval_freq=max(5000 // train_env.env.window, 1000),
        n_eval_episodes=1,
        deterministic=True,
    )

    t_cfg = config.agent.training
    model = create_model(
        policy=config.agent.policy,
        env=train_env,
        learning_rate=t_cfg.learning_rate,
        n_steps=t_cfg.n_steps,
        batch_size=t_cfg.batch_size,
        n_epochs=t_cfg.n_epochs,
        gamma=t_cfg.gamma,
        ent_coef=t_cfg.ent_coef,
        net_arch=config.agent.net_arch,
        tensorboard_log=str(Path(model_dir) / "tensorboard"),
    )

    model.learn(total_timesteps=t_cfg.total_timesteps, callback=eval_callback, progress_bar=True)
    model_path = str(Path(model_dir) / "ppo_trader.zip")
    save_model(model, model_path)

    test_env = TradingEnv(
        test_feat, test_prices,
        window=config.agent.observation_bars,
        reward_fn=config.agent.reward_fn,
    )
    obs, _ = test_env.reset()
    test_equity = [test_env.initial_capital]
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_equity.append(info["equity"])
        if terminated or truncated:
            break

    test_returns = np.diff(test_equity) / (np.array(test_equity[:-1]) + 1e-12)
    sharpe = (np.mean(test_returns) * 252 - 0.05) / (np.std(test_returns, ddof=1) * np.sqrt(252) + 1e-12)
    peak = np.maximum.accumulate(test_equity)
    drawdown_pct = abs((np.array(test_equity) - peak) / (peak + 1e-12))
    max_dd = float(np.max(drawdown_pct))

    metrics = {"test_sharpe": float(sharpe), "test_max_drawdown": float(max_dd), "test_equity": test_equity}
    return model, metrics


def incremental_train(
    features: np.ndarray,
    prices: np.ndarray,
    config,
    model_path: str = "models/ppo_trader.zip",
    timesteps: int = 10000,
) -> tuple:
    """Fine-tune an existing PPO model on new data (incremental/online learning).

    Loads the current model, trains on the most recent data for a small number
    of timesteps, and saves the updated model. Designed to be run daily after
    market close to adapt the model to latest market conditions.

    Args:
        features: Feature array (use only recent data, e.g., last 60 days)
        prices: Price array
        config: AppConfig
        model_path: Path to existing model to fine-tune
        timesteps: Number of timesteps for fine-tuning (default 10k)

    Returns:
        (updated_model, metrics_dict)
    """
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    if not Path(model_path).exists():
        logger.warning(f"No existing model at {model_path}, training from scratch")
        return train_agent(features, prices, config)

    # Load existing model
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    logger.info(f"Loaded existing model from {model_path} for incremental training")

    # Use only recent data for fine-tuning
    if len(features) > 126:  # ~6 months of daily data
        features = features[-126:]
        prices = prices[-126:]
        logger.info(f"Using last {len(features)} bars for incremental training")

    # Create env with the loaded model's environment
    train_env = Monitor(TradingEnv(
        features, prices,
        window=config.agent.observation_bars,
        reward_fn=config.agent.reward_fn,
    ))

    # Update the model's environment
    model.set_env(train_env)

    # Fine-tune with reduced learning rate
    t_cfg = config.agent.training
    model.learning_rate = t_cfg.learning_rate * 0.5  # lower LR for fine-tuning
    model.ent_coef = t_cfg.ent_coef * 0.5  # less exploration during fine-tuning

    logger.info(f"Incremental training: {timesteps} timesteps, lr={model.learning_rate:.6f}")
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # Back up old model
    backup_path = model_path.replace(".zip", f"_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip")
    import shutil
    shutil.copy(model_path, backup_path)

    # Save fine-tuned model
    save_model(model, model_path)
    logger.info(f"Saved fine-tuned model to {model_path} (backup: {backup_path})")

    # Quick evaluation
    test_start = max(0, len(features) - 252)
    test_feat = features[test_start:]
    test_prices = prices[test_start:]
    test_env = TradingEnv(
        test_feat, test_prices,
        window=config.agent.observation_bars,
        reward_fn=config.agent.reward_fn,
    )
    obs, _ = test_env.reset()
    test_equity = [test_env.initial_capital]
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_equity.append(info["equity"])
        if terminated or truncated:
            break

    test_returns = np.diff(test_equity) / (np.array(test_equity[:-1]) + 1e-12)
    sharpe = (np.mean(test_returns) * 252 - 0.05) / (np.std(test_returns, ddof=1) * np.sqrt(252) + 1e-12)
    peak = np.maximum.accumulate(test_equity)
    max_dd = float(np.max(abs((np.array(test_equity) - peak) / (peak + 1e-12))))

    metrics = {
        "test_sharpe": float(sharpe),
        "test_max_drawdown": max_dd,
        "test_equity": test_equity,
        "incremental_timesteps": timesteps,
    }

    logger.info(f"Incremental training complete: Sharpe={sharpe:.4f}, MaxDD={max_dd:.4f}")
    return model, metrics
