from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd
# Pure numpy implementation — no scipy dependency


@dataclass
class FactorTemplate:
    name: str
    expression: str
    category: str
    description: str


@dataclass
class AlphaFactor:
    name: str
    values: np.ndarray
    ic: float
    rank_ic: float
    sharpe: float
    turnover: float = 0.0


@dataclass
class FactorMiningConfig:
    max_factors: int = 50
    top_k: int = 10
    ic_threshold: float = 0.02
    population_size: int = 100
    generations: int = 20


class FactorRegistry:
    """Registry of factor expression templates (inspired by Qlib's alpha factor DSL)."""

    @staticmethod
    def compute(df: pd.DataFrame, expression: str, period: int) -> np.ndarray:
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)
        open_ = df["open"].values.astype(float) if "open" in df.columns else close

        n = len(close)
        result = np.full(n, np.nan)

        if expression == "roc":
            # Rate of change
            result[period:] = close[period:] / close[:-period] - 1
        elif expression == "mom":
            # Momentum (difference)
            result[period:] = close[period:] - close[:-period]
        elif expression == "wma_sma_ratio":
            # WMA / SMA ratio (trend strength)
            wma = _weighted_ma(close, period)
            sma = _sma(close, period)
            mask = sma > 0
            result[mask] = wma[mask] / sma[mask]
        elif expression == "sma_dev":
            # Deviation from SMA
            sma = _sma(close, period)
            mask = (sma > 0) & (np.arange(n) >= period)
            result[mask] = (close[mask] - sma[mask]) / sma[mask]
        elif expression == "bb_position":
            # Position within Bollinger Bands
            sma = _sma(close, period)
            std = _rolling_std(close, period)
            mask = std > 0
            result[mask] = (close[mask] - sma[mask]) / (2 * std[mask])
        elif expression == "atr_ratio":
            tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)),
                                                    abs(low - np.roll(close, 1))))
            atr = _sma(tr, period)
            mask = (atr > 0) & (np.arange(n) >= period)
            result[mask] = atr[mask] / close[mask]
        elif expression == "high_low_ratio":
            result[period:] = high[period:] / (low[period:] + 1e-12) - 1
        elif expression == "return_std":
            ret = np.full(n, np.nan)
            ret[1:] = close[1:] / close[:-1] - 1
            result[period:] = _rolling_std(ret, period)[period:]
        elif expression == "vol_ratio":
            sma_vol = _sma(volume, period)
            mask = (sma_vol > 0) & (np.arange(n) >= period)
            result[mask] = volume[mask] / sma_vol[mask]
        elif expression == "dollar_vol":
            dv = close * volume
            sma_dv = _sma(dv, period)
            mask = (sma_dv > 0) & (np.arange(n) >= period)
            result[mask] = dv[mask] / sma_dv[mask]
        elif expression == "corr_price_vol":
            ret = _returns(close)
            vol_chg = _returns(volume)
            result[period:] = _rolling_corr(ret, vol_chg, period)[period:]
        elif expression == "corr_price_sma":
            sma = _sma(close, period)
            result[period:] = _rolling_corr(close, sma, period)[period:]
        elif expression == "rsi_reversal":
            # RSI-based mean reversion signal
            rsi = _rsi(close, period)
            rsi = np.nan_to_num(rsi, nan=50)
            result[:] = -(rsi - 50) / 50  # -1 to +1, negative at overbought
        elif expression == "volume_price_trend":
            ret = _returns(close)
            result[period:] = _sma(ret * volume, period)[period:]
        elif expression == "intraday_range":
            result[:] = (high - low) / (open_ + 1e-12)
        elif expression == "open_close_gap":
            result[1:] = (open_[1:] - close[:-1]) / (close[:-1] + 1e-12)
        elif expression == "efficiency_ratio":
            direction = np.abs(close[period:] - close[:-period])
            volatility = _rolling_sum(np.abs(_returns(close)), period)[period:]
            mask = volatility > 0
            result_slice = result[period:]
            result_slice[mask] = direction[mask] / volatility[mask]
            result[period:] = result_slice

        return result

    @staticmethod
    def templates() -> list[FactorTemplate]:
        periods = [5, 10, 20, 30, 60]
        categories = {
            "momentum": ["roc", "mom", "wma_sma_ratio", "sma_dev"],
            "mean_reversion": ["bb_position", "rsi_reversal", "efficiency_ratio"],
            "volatility": ["atr_ratio", "return_std", "high_low_ratio", "intraday_range"],
            "volume": ["vol_ratio", "dollar_vol", "volume_price_trend"],
            "correlation": ["corr_price_vol", "corr_price_sma"],
            "price_pattern": ["open_close_gap"],
        }
        templates = []
        for cat, expressions in categories.items():
            for expr in expressions:
                for p in periods:
                    templates.append(FactorTemplate(
                        name=f"{expr}_{p}",
                        expression=expr,
                        category=cat,
                        description=f"{expr}(close/high/low/vol, period={p})",
                    ))
        return templates


class FactorMiner:
    """Alpha factor discovery engine using brute-force IC evaluation."""

    def __init__(self, config: FactorMiningConfig):
        self.config = config
        self.registry = FactorRegistry()

    def brute_force_search(self, features: pd.DataFrame, prices: np.ndarray) -> list[AlphaFactor]:
        """Evaluate all factor templates and rank by Information Coefficient."""
        templates = self.registry.templates()
        forward_returns = _forward_returns(prices, horizon=5)

        factors: list[AlphaFactor] = []
        for tpl in templates[:self.config.max_factors]:
            values = self.registry.compute(features, tpl.expression, self._parse_period(tpl.name))
            if values is None or np.all(np.isnan(values)):
                continue

            ic, rank_ic = evaluate_factor(values, forward_returns)
            if abs(ic) < self.config.ic_threshold:
                continue

            sharpe = _factor_sharpe(values, forward_returns)
            turnover = _factor_turnover(values)

            factors.append(AlphaFactor(
                name=f"{tpl.category}/{tpl.name}",
                values=values,
                ic=float(ic),
                rank_ic=float(rank_ic),
                sharpe=float(sharpe),
                turnover=float(turnover),
            ))

        factors.sort(key=lambda f: abs(f.rank_ic), reverse=True)
        return factors

    def _parse_period(self, name: str) -> int:
        parts = name.split("_")
        try:
            return int(parts[-1])
        except (ValueError, IndexError):
            return 20


def evaluate_factor(factor_values: np.ndarray, forward_returns: np.ndarray) -> tuple[float, float]:
    """Compute IC (Pearson) and Rank IC (Spearman) for a factor."""
    valid = ~np.isnan(factor_values) & ~np.isnan(forward_returns)
    if valid.sum() < 30:
        return 0.0, 0.0

    fv = factor_values[valid]
    fr = forward_returns[valid]

    ic = np.corrcoef(fv, fr)[0, 1]
    rank_ic = _spearmanr(fv, fr)

    return float(ic if not np.isnan(ic) else 0.0), float(rank_ic if not np.isnan(rank_ic) else 0.0)


def select_top_k(factors: list[AlphaFactor], k: int = 10) -> list[AlphaFactor]:
    """Select top-K factors considering both IC and turnover."""
    if len(factors) <= k:
        return factors

    # Penalize high-turnover factors
    scored = []
    for f in factors:
        score = abs(f.rank_ic) - 0.1 * f.turnover
        scored.append((score, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in scored[:k]]


def add_factors_to_features(df: pd.DataFrame, factors: list[AlphaFactor]) -> pd.DataFrame:
    """Merge selected factor values into the feature DataFrame."""
    result = df.copy()
    for f in factors:
        col_name = f.name.replace("/", "_")
        result[col_name] = f.values
    return result


# ── Numerical helpers ──────────────────────────────────────────────


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """Pure-numpy Spearman rank correlation (avoids scipy dependency)."""
    from numpy import argsort
    x_rank = argsort(argsort(x)).astype(float)
    y_rank = argsort(argsort(y)).astype(float)
    n = len(x_rank)
    numerator = n * np.sum(x_rank * y_rank) - np.sum(x_rank) * np.sum(y_rank)
    denominator = np.sqrt(
        (n * np.sum(x_rank ** 2) - np.sum(x_rank) ** 2) *
        (n * np.sum(y_rank ** 2) - np.sum(y_rank) ** 2)
    )
    if denominator < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _sma(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= window:
        kernel = np.ones(window) / window
        out[window - 1:] = np.convolve(x, kernel, mode="valid")
    return out


def _weighted_ma(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= window:
        weights = np.arange(1, window + 1)
        kernel = weights / weights.sum()
        out[window - 1:] = np.convolve(x, kernel, mode="valid")
    return out


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= window:
        ret = np.lib.stride_tricks.sliding_window_view(x, window)
        out[window - 1:] = np.std(ret, axis=1, ddof=1)
    return out


def _rolling_corr(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= window:
        a_win = np.lib.stride_tricks.sliding_window_view(a, window)
        b_win = np.lib.stride_tricks.sliding_window_view(b, window)
        for i in range(len(a_win)):
            idx = i + window - 1
            mask = ~np.isnan(a_win[i]) & ~np.isnan(b_win[i])
            if mask.sum() >= max(window // 2, 5):
                c = np.corrcoef(a_win[i][mask], b_win[i][mask])[0, 1]
                out[idx] = c if not np.isnan(c) else 0.0
    return out


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= window:
        kernel = np.ones(window)
        out[window - 1:] = np.convolve(x, kernel, mode="valid")
    return out


def _returns(x: np.ndarray) -> np.ndarray:
    ret = np.full(len(x), np.nan)
    ret[1:] = x[1:] / x[:-1] - 1
    return ret


def _rsi(close: np.ndarray, window: int) -> np.ndarray:
    ret = _returns(close)
    gain = np.where(ret > 0, ret, 0)
    loss = np.where(ret < 0, -ret, 0)
    avg_gain = _sma(gain, window)
    avg_loss = _sma(loss, window)
    mask = avg_loss > 0
    rsi = np.full(len(close), 50.0)
    rsi[mask] = 100 - 100 / (1 + avg_gain[mask] / avg_loss[mask])
    return rsi


def _forward_returns(prices: np.ndarray, horizon: int = 5) -> np.ndarray:
    ret = np.full(len(prices), np.nan)
    ret[:-horizon] = (prices[horizon:] - prices[:-horizon]) / (prices[:-horizon] + 1e-12)
    return ret


def _factor_sharpe(values: np.ndarray, forward_returns: np.ndarray) -> float:
    """Estimate a factor's Sharpe by long-short decile spread."""
    valid = ~np.isnan(values) & ~np.isnan(forward_returns)
    if valid.sum() < 50:
        return 0.0
    fv = values[valid]
    fr = forward_returns[valid]
    top_cut = np.percentile(fv, 80)
    bot_cut = np.percentile(fv, 20)
    top_rets = fr[fv >= top_cut]
    bot_rets = fr[fv <= bot_cut]
    if len(top_rets) < 5 or len(bot_rets) < 5:
        return 0.0
    spread = np.mean(top_rets) - np.mean(bot_rets)
    vol = np.std(fr, ddof=1)
    return float(spread / (vol + 1e-12))


def _factor_turnover(values: np.ndarray) -> float:
    """Estimate factor turnover (correlation with lagged values)."""
    if len(values) < 10:
        return 0.5
    valid = ~np.isnan(values)
    v = values[valid]
    if len(v) < 10:
        return 0.5
    corr = np.corrcoef(v[1:], v[:-1])[0, 1]
    return float(1 - abs(corr)) if not np.isnan(corr) else 0.5
