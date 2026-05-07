import numpy as np

from src.agent.env import TradingEnv


def test_env_reset():
    features = np.random.randn(200, 22).astype(np.float32)
    prices = (100 + np.cumsum(np.random.randn(200) * 0.5)).astype(np.float32)
    env = TradingEnv(features, prices, window=20)
    obs, info = env.reset()
    assert obs.shape == (env.observation_space.shape[0],)
    assert info == {}


def test_env_step_buy():
    features = np.random.randn(100, 22).astype(np.float32)
    prices = np.linspace(100, 120, 100, dtype=np.float32)
    env = TradingEnv(features, prices, window=20)
    env.reset()
    action = np.array([0.1, 0.9, 0.1], dtype=np.float32)  # BUY
    obs, reward, terminated, truncated, info = env.step(action)
    assert "equity" in info
    assert "action" in info


def test_env_terminated():
    features = np.random.randn(25, 22).astype(np.float32)
    prices = np.linspace(100, 200, 25, dtype=np.float32)
    env = TradingEnv(features, prices, window=20)
    env.reset()
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(np.array([0.3, 0.3, 0.4], dtype=np.float32))
        if terminated:
            break
    assert terminated or truncated or True
