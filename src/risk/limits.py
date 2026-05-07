import logging
from dataclasses import dataclass

from src.ibkr.order_manager import AccountSummary, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    modified: bool = False
    reason: str = ""
    modified_size: float | None = None


@dataclass
class RiskLimits:
    max_position_pct: float = 0.10
    max_daily_loss_pct: float = 0.05
    max_total_drawdown_pct: float = 0.15
    min_hold_bars: int = 1
    max_trades_per_day: int = 10
    min_trade_value: float = 500.0


class RiskEngine:
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.daily_trade_count = 0
        self.starting_day_equity = 0.0
        self.peak_equity = 0.0

    def reset_daily(self, current_equity: float):
        self.daily_trade_count = 0
        self.starting_day_equity = current_equity
        self.peak_equity = max(self.peak_equity, current_equity)

    def check_signal(self, signal: TradeSignal, account: AccountSummary) -> RiskDecision:
        if signal.direction == 0:
            return RiskDecision(approved=True, reason="HOLD")

        if self.daily_trade_count >= self.limits.max_trades_per_day:
            return RiskDecision(approved=False, reason="Max daily trades reached")

        if signal.direction == 1:
            trade_value = account.cash * signal.size
            if trade_value < self.limits.min_trade_value:
                return RiskDecision(approved=False, reason=f"Trade value ${trade_value:.0f} below minimum")

            position_pct = trade_value / (account.equity + 1e-12)
            if position_pct > self.limits.max_position_pct:
                capped_size = self.limits.max_position_pct * account.equity / account.cash
                return RiskDecision(approved=True, modified=True, reason="Position size capped", modified_size=capped_size)

        if self.starting_day_equity > 0:
            daily_return = (account.equity - self.starting_day_equity) / self.starting_day_equity
            if daily_return < -self.limits.max_daily_loss_pct:
                return RiskDecision(approved=False, reason=f"Daily loss limit hit: {daily_return:.3f}")

        drawdown = (account.equity - self.peak_equity) / (self.peak_equity + 1e-12)
        if drawdown < -self.limits.max_total_drawdown_pct:
            return RiskDecision(approved=False, reason=f"Max drawdown hit: {drawdown:.3f}")

        return RiskDecision(approved=True)
