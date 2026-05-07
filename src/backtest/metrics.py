import numpy as np
import pandas as pd


def compute_metrics(
    equity_curve: np.ndarray, benchmark_returns: np.ndarray | None = None, periods_per_year: int = 252
) -> dict[str, float]:
    if len(equity_curve) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "calmar": 0.0}

    returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-12)

    ann_return = np.mean(returns) * periods_per_year
    ann_vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    sharpe = (ann_return - 0.05) / (ann_vol + 1e-12)

    downside = returns[returns < 0]
    sortino = (ann_return - 0.05) / (np.std(downside, ddof=1) * np.sqrt(periods_per_year) + 1e-12) if len(downside) > 1 else 0.0

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (peak + 1e-12)
    max_dd = abs(drawdown.min())

    calmar = ann_return / (max_dd + 1e-12)

    n_years = len(equity_curve) / periods_per_year
    total_return = equity_curve[-1] / equity_curve[0] - 1
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    result = {
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "ann_return": round(ann_return, 4),
        "ann_vol": round(ann_vol, 4),
        "total_return": round(total_return, 4),
    }

    if benchmark_returns is not None and len(benchmark_returns) > 1:
        min_len = min(len(returns), len(benchmark_returns))
        strat = returns[-min_len:]
        bench = benchmark_returns[-min_len:]
        alpha_beta = compute_alpha_beta(strat, bench)
        result.update(alpha_beta)

    return result


def compute_alpha_beta(strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> dict[str, float]:
    """Compute alpha, beta, tracking error, and information ratio."""
    min_len = min(len(strategy_returns), len(benchmark_returns))
    strat = strategy_returns[-min_len:]
    bench = benchmark_returns[-min_len:]

    if min_len < 2:
        return {"alpha": 0.0, "beta": 0.0, "tracking_error": 0.0, "information_ratio": 0.0}

    cov = np.cov(strat, bench, ddof=1)
    bench_var = np.var(bench, ddof=1)
    beta = cov[0, 1] / (bench_var + 1e-12)

    alpha = (np.mean(strat) - beta * np.mean(bench)) * 252

    excess = strat - bench
    tracking_error = float(np.std(excess, ddof=1) * np.sqrt(252))
    info_ratio = (np.mean(excess) * 252) / (tracking_error + 1e-12)

    return {
        "alpha": round(float(alpha), 4),
        "beta": round(float(beta), 4),
        "tracking_error": round(tracking_error, 4),
        "information_ratio": round(float(info_ratio), 4),
    }
