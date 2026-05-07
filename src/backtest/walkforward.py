from typing import Callable

import numpy as np


def walkforward_validate(
    features: np.ndarray,
    prices: np.ndarray,
    train_fn: Callable[[np.ndarray, np.ndarray], object],
    test_fn: Callable[[object, np.ndarray, np.ndarray], np.ndarray],
    n_splits: int = 5,
    train_window: int = 756,
    test_window: int = 126,
) -> list[dict]:
    metrics = []
    for split in range(n_splits):
        train_end = train_window + split * test_window
        test_end = min(train_end + test_window, len(prices))

        train_features = features[:train_end]
        train_prices = prices[:train_end]
        test_features = features[train_end:test_end]
        test_prices = prices[train_end:test_end]

        model = train_fn(train_features, train_prices)
        equity = test_fn(model, test_features, test_prices)

        returns = np.diff(equity) / (equity[:-1] + 1e-12)
        sharpe = (np.mean(returns) * 252 - 0.05) / (np.std(returns, ddof=1) * np.sqrt(252) + 1e-12)
        peak = np.maximum.accumulate(equity)
        max_dd = abs((equity - peak) / (peak + 1e-12)).min()

        metrics.append({"split": split, "sharpe": sharpe, "max_drawdown": max_dd, "equity": equity})

    return metrics
