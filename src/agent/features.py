from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioState:
    cash: float = 1.0
    equity: float = 1.0
    position_value: float = 0.0
    peak_equity: float = 1.0
    entry_price: float = 0.0

    @property
    def cash_ratio(self) -> float:
        return self.cash / (self.equity + 1e-12)

    @property
    def exposure(self) -> float:
        return self.position_value / (self.equity + 1e-12)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.position_value <= 0 or self.entry_price <= 0:
            return 0.0
        return (self.equity - self.cash - self.position_value) / (self.position_value + 1e-12)

    @property
    def drawdown(self) -> float:
        return (self.equity - self.peak_equity) / (self.peak_equity + 1e-12)


class FeatureEngine:
    BASE_FEATURE_COLS = [
        "close", "open", "high", "low", "volume",
        "return_1d", "return_5d", "return_20d",
        "rsi_14", "macd_line", "macd_signal", "macd_hist",
        "sma_20", "sma_50", "sma_ratio",
        "atr_14", "atr_ratio",
        "volume_sma_20", "volume_ratio",
        "bb_upper", "bb_lower", "bb_width",
    ]

    def __init__(self, window: int = 20, extra_feature_cols: list[str] | None = None):
        self.window = window
        self.extra_feature_cols = extra_feature_cols or []
        self.FEATURE_COLS = self.BASE_FEATURE_COLS + self.extra_feature_cols
        self.price_cols = 5
        self.indicator_cols = len(self.FEATURE_COLS) - self.price_cols
        self.portfolio_cols = 4
        self.obs_dim = self.window * self.price_cols + self.indicator_cols + self.portfolio_cols
        self._mean = None
        self._std = None
        self._fitted = False

    def fit(self, features_df: pd.DataFrame):
        available = [c for c in self.FEATURE_COLS if c in features_df.columns]
        data = features_df[available].values.astype(np.float32)
        self._mean = np.nanmean(data, axis=0)
        self._std = np.nanstd(data, axis=0)
        self._std[self._std < 1e-8] = 1.0
        self._mean[np.isnan(self._mean)] = 0.0
        self._fitted = True

    def compute_observation(self, features_df: pd.DataFrame, portfolio: PortfolioState) -> np.ndarray:
        available = [c for c in self.FEATURE_COLS if c in features_df.columns]
        window_df = features_df[available].iloc[-self.window:]

        if len(window_df) < self.window:
            pad = np.zeros((self.window - len(window_df), len(available)))
            data = np.vstack([pad, window_df.values.astype(np.float32)])
        else:
            data = window_df.values.astype(np.float32)

        if self._fitted:
            idx = [self.FEATURE_COLS.index(c) for c in available]
            m = self._mean[idx]
            s = self._std[idx]
            data = (data - m) / s
            data = np.clip(data, -3.0, 3.0)

        price_part = data[:, :self.price_cols].flatten() if data.shape[1] >= self.price_cols else np.zeros(self.window * self.price_cols)
        indicator_part = data[-1, self.price_cols:] if data.shape[1] > self.price_cols else np.zeros(self.indicator_cols)

        if len(indicator_part) < self.indicator_cols:
            pad = np.zeros(self.indicator_cols - len(indicator_part))
            indicator_part = np.concatenate([indicator_part, pad])

        port = np.array([
            portfolio.cash_ratio,
            portfolio.exposure,
            portfolio.unrealized_pnl_pct,
            portfolio.drawdown,
        ], dtype=np.float32)

        obs = np.concatenate([price_part, indicator_part, port])
        return np.nan_to_num(obs, nan=0.0).astype(np.float32)

    @property
    def observation_space_dim(self) -> int:
        return self.obs_dim
