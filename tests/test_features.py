import numpy as np
import pandas as pd

from src.agent.features import FeatureEngine, PortfolioState


def test_feature_engine_obs_dim():
    engine = FeatureEngine(window=20)
    assert engine.observation_space_dim == 20 * 5 + 17 + 4  # 121


def test_compute_observation():
    engine = FeatureEngine(window=20)
    data = {
        c: np.random.randn(50)
        for c in engine.FEATURE_COLS
    }
    df = pd.DataFrame(data)
    portfolio = PortfolioState(cash=50000, equity=100000, position_value=50000, peak_equity=100000)
    obs = engine.compute_observation(df, portfolio)
    assert obs.shape == (121,)
    assert obs.dtype == np.float32


def test_portfolio_state():
    ps = PortfolioState(cash=50000, equity=100000, position_value=50000, peak_equity=110000)
    assert ps.cash_ratio == 0.5
    assert ps.exposure == 0.5
    assert ps.drawdown < 0
