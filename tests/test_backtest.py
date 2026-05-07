import numpy as np

from src.backtest.engine import VectorizedBacktest
from src.backtest.metrics import compute_metrics


def test_buy_and_hold():
    bt = VectorizedBacktest(initial_capital=100000)
    prices = np.array([100, 105, 110, 108, 115], dtype=float)
    result = bt.run_buy_and_hold(prices)
    assert result.equity_curve[0] == 100000
    assert result.equity_curve[-1] == 115000


def test_compute_metrics():
    equity = np.array([100000, 101000, 102000, 101500, 103000], dtype=float)
    m = compute_metrics(equity)
    assert "sharpe" in m
    assert "max_drawdown" in m
    assert m["total_return"] > 0


def test_sma_cross():
    import pandas as pd
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({"close": close})
    bt = VectorizedBacktest()
    result = bt.run_sma_cross(df)
    assert len(result.equity_curve) == n
    assert result.equity_curve[0] == 100000
