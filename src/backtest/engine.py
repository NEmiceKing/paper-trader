from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: np.ndarray
    trades: list[dict] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class SlippageModel:
    fixed_cost: float = 0.005
    pct_cost: float = 0.0005
    price_impact_pct: float = 0.001


@dataclass
class TradeStatistics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_holding_period: float = 0.0
    max_consecutive_losses: int = 0
    expectancy: float = 0.0


class VectorizedBacktest:
    def __init__(self, initial_capital: float = 100_000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission

    def apply_slippage(self, price: float, is_buy: bool, volume: float = 0,
                       model: SlippageModel | None = None) -> float:
        """Apply slippage and market impact to execution price."""
        if model is None:
            return price
        sign = 1 if is_buy else -1
        impact = model.price_impact_pct * volume if volume > 0 else 0
        return price * (1 + sign * (model.pct_cost + impact)) + sign * model.fixed_cost

    def run_buy_and_hold(self, prices: np.ndarray) -> BacktestResult:
        equity = self.initial_capital * prices / prices[0]
        return BacktestResult(equity_curve=equity)

    def run_sma_cross(
        self, df: pd.DataFrame, fast: int = 20, slow: int = 50,
        slippage: SlippageModel | None = None,
    ) -> BacktestResult:
        close = df["close"].values.astype(float)
        sma_fast = df["close"].rolling(fast).mean().values
        sma_slow = df["close"].rolling(slow).mean().values

        position = 0.0
        cash = self.initial_capital
        equity = np.zeros(len(close))
        trades = []

        for i in range(len(close)):
            if i < slow:
                equity[i] = cash
                continue

            prev_fast = sma_fast[i - 1]
            prev_slow = sma_slow[i - 1]
            curr_fast = sma_fast[i]
            curr_slow = sma_slow[i]

            price = close[i]

            if prev_fast <= prev_slow and curr_fast > curr_slow and position == 0:
                exec_price = self.apply_slippage(price, True, volume=0, model=slippage)
                position = cash * 0.95 / exec_price
                cash -= position * exec_price * (1 + self.commission)
                trades.append({"date": i, "action": "BUY", "price": exec_price,
                               "raw_price": price, "shares": position})
            elif prev_fast >= prev_slow and curr_fast < curr_slow and position > 0:
                exec_price = self.apply_slippage(price, False, volume=0, model=slippage)
                cash += position * exec_price * (1 - self.commission)
                trades.append({"date": i, "action": "SELL", "price": exec_price,
                               "raw_price": price, "shares": position})
                position = 0.0

            equity[i] = cash + position * close[i]

        return BacktestResult(equity_curve=equity, trades=trades)

    def compute_trade_statistics(self, trades: list[dict]) -> TradeStatistics:
        """Compute detailed trade-level statistics from a list of paired trades."""
        if len(trades) < 2:
            return TradeStatistics()

        stats = TradeStatistics()
        wins = []
        losses = []
        consecutive_losses = 0
        max_consec = 0
        last_action = None

        # Pair BUY->SELL to compute PnL per round-trip
        i = 0
        while i < len(trades) - 1:
            buy = trades[i]
            sell = None
            for j in range(i + 1, len(trades)):
                if trades[j]["action"] == "SELL":
                    sell = trades[j]
                    break
            if sell is None:
                break

            buy_value = buy["shares"] * buy["price"]
            sell_value = sell["shares"] * sell["price"]
            pnl = sell_value - buy_value
            pnl_pct = pnl / (buy_value + 1e-12)

            if pnl > 0:
                wins.append(pnl_pct)
                consecutive_losses = 0
            else:
                losses.append(abs(pnl_pct))
                consecutive_losses += 1
                max_consec = max(max_consec, consecutive_losses)

            i = j + 1

        stats.total_trades = len(wins) + len(losses)
        stats.winning_trades = len(wins)
        stats.losing_trades = len(losses)
        stats.win_rate = stats.winning_trades / max(stats.total_trades, 1)
        stats.avg_win = float(np.mean(wins)) if wins else 0.0
        stats.avg_loss = float(np.mean(losses)) if losses else 0.0
        gross_profit = float(np.sum(wins)) if wins else 0.0
        gross_loss = float(np.sum(losses)) if losses else 1e-12
        stats.profit_factor = gross_profit / gross_loss
        stats.max_consecutive_losses = max_consec
        stats.expectancy = stats.win_rate * stats.avg_win - (1 - stats.win_rate) * stats.avg_loss

        return stats

    def compare_to_benchmark(self, equity_curve: np.ndarray,
                             benchmark_prices: np.ndarray) -> dict[str, float]:
        """Compute alpha, beta, tracking error, and information ratio vs benchmark."""
        if len(equity_curve) < 2 or len(benchmark_prices) < 2:
            return {"alpha": 0.0, "beta": 0.0, "tracking_error": 0.0, "information_ratio": 0.0}

        strat_rets = np.diff(equity_curve) / (equity_curve[:-1] + 1e-12)
        bench_rets = np.diff(benchmark_prices) / (benchmark_prices[:-1] + 1e-12)

        min_len = min(len(strat_rets), len(bench_rets))
        strat_rets = strat_rets[-min_len:]
        bench_rets = bench_rets[-min_len:]

        cov = np.cov(strat_rets, bench_rets, ddof=1)
        bench_var = np.var(bench_rets, ddof=1)
        beta = cov[0, 1] / (bench_var + 1e-12)

        alpha = (np.mean(strat_rets) - beta * np.mean(bench_rets)) * 252

        excess = strat_rets - bench_rets
        tracking_error = float(np.std(excess, ddof=1) * np.sqrt(252))
        info_ratio = (np.mean(excess) * 252) / (tracking_error + 1e-12)

        return {
            "alpha": round(float(alpha), 4),
            "beta": round(float(beta), 4),
            "tracking_error": round(tracking_error, 4),
            "information_ratio": round(float(info_ratio), 4),
        }

    def run_multi_timeframe(
        self,
        daily_df: pd.DataFrame,
        intraday_df: pd.DataFrame | None = None,
        slippage: SlippageModel | None = None,
    ) -> BacktestResult:
        """Multi-timeframe backtest using daily signals with optional intraday execution."""
        close = daily_df["close"].values.astype(float)
        equity = np.full(len(close), self.initial_capital)
        cash = self.initial_capital
        position = 0.0
        trades = []

        for i in range(1, len(close)):
            price = close[i]
            prev_price = close[i - 1]
            ret = price / prev_price - 1

            if position > 0:
                cash_prev = cash
                cash = 0
                position_value = position * price
            else:
                position_value = 0

            equity[i] = cash + position_value

        # Simple momentum strategy on daily data
        sma_20 = daily_df["close"].rolling(20).mean().values
        sma_50 = daily_df["close"].rolling(50).mean().values

        cash = self.initial_capital
        position = 0.0
        equity = np.zeros(len(close))

        for i in range(len(close)):
            if i < 50:
                equity[i] = cash
                continue

            price = close[i]
            exec_price = price

            if sma_20[i] > sma_50[i] and sma_20[i - 1] <= sma_50[i - 1] and position == 0:
                if slippage:
                    exec_price = self.apply_slippage(price, True, volume=0, model=slippage)
                position = cash * 0.95 / exec_price
                cash -= position * exec_price * (1 + self.commission)
                trades.append({"date": i, "action": "BUY", "price": exec_price,
                               "raw_price": price, "shares": position})
            elif sma_20[i] < sma_50[i] and sma_20[i - 1] >= sma_50[i - 1] and position > 0:
                if slippage:
                    exec_price = self.apply_slippage(price, False, volume=0, model=slippage)
                cash += position * exec_price * (1 - self.commission)
                trades.append({"date": i, "action": "SELL", "price": exec_price,
                               "raw_price": price, "shares": position})
                position = 0.0

            equity[i] = cash + position * price

        return BacktestResult(equity_curve=equity, trades=trades)
