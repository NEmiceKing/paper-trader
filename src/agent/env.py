import gymnasium
import numpy as np
from gymnasium import spaces

from src.agent.features import FeatureEngine, PortfolioState
from src.agent.reward import conservative_reward


class TradingEnv(gymnasium.Env):
    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        initial_capital: float = 100_000.0,
        window: int = 20,
        reward_fn: str = "conservative",
        commission: float = 0.001,
        max_position_pct: float = 0.95,
        llm_features: np.ndarray | None = None,
    ):
        super().__init__()
        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.initial_capital = initial_capital
        self.window = window
        self.reward_fn = reward_fn
        self.commission = commission
        self.max_position_pct = max_position_pct
        self.llm_features = llm_features  # shape (T, n_llm_feats) or None

        self.n_features = features.shape[1]
        feature_extra_cols = ["llm_0", "llm_1", "llm_2", "llm_3"] if llm_features is not None else None
        self.feature_engine = FeatureEngine(window=window, extra_feature_cols=feature_extra_cols)
        obs_dim = self.feature_engine.observation_space_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.cash = initial_capital
        self.shares = 0.0
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.current_step = 0
        self.entry_price = 0.0
        self.prev_exposure = 0.0
        self.episode_returns = []

    def _get_portfolio_state(self) -> PortfolioState:
        position_value = self.shares * self.prices[self.current_step]
        return PortfolioState(
            cash=self.cash,
            equity=self.equity,
            position_value=position_value,
            peak_equity=self.peak_equity,
            entry_price=self.entry_price,
        )

    def _get_obs(self) -> np.ndarray:
        end = self.current_step + 1
        start = max(0, end - self.window * 2)
        feat_slice = self.features[start:end]

        import pandas as pd

        if self.llm_features is not None and self.current_step < len(self.llm_features):
            llm_slice = self.llm_features[self.current_step]
            sfx = pd.DataFrame([llm_slice], columns=["llm_0", "llm_1", "llm_2", "llm_3"])
            sfx_repeated = pd.DataFrame(
                np.tile(llm_slice, (feat_slice.shape[0], 1)),
                columns=["llm_0", "llm_1", "llm_2", "llm_3"],
            )
            feat_slice = np.hstack([feat_slice, sfx_repeated.values.astype(np.float32)])

        col_names = self.feature_engine.FEATURE_COLS[: feat_slice.shape[1]]
        df = pd.DataFrame(feat_slice, columns=col_names)
        return self.feature_engine.compute_observation(df, self._get_portfolio_state())

    def _decode_action(self, action: np.ndarray) -> tuple[int, float]:
        direction = int(np.argmax(action))
        size = np.clip(abs(action[direction]), 0.0, 1.0)
        if size < 0.02:
            direction = 0
        return direction, float(size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = self.initial_capital
        self.shares = 0.0
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.current_step = self.window
        self.entry_price = 0.0
        self.prev_exposure = 0.0
        self.episode_returns = []
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        direction, size = self._decode_action(action)
        price = self.prices[self.current_step]
        old_equity = self.equity

        position_value = self.shares * price

        if direction == 1:
            buy_value = self.cash * size * self.max_position_pct
            if buy_value > 0:
                qty = buy_value / (price * (1 + self.commission))
                cost = qty * price * (1 + self.commission)
                if cost <= self.cash:
                    self.shares += qty
                    self.cash -= cost
                    if self.shares > 0:
                        self.entry_price = (self.entry_price * (self.shares - qty) + cost) / self.shares
        elif direction == 2:
            if self.shares > 0:
                sell_qty = self.shares * size
                proceeds = sell_qty * price * (1 - self.commission)
                self.shares -= sell_qty
                self.cash += proceeds
                if self.shares <= 1e-8:
                    self.shares = 0.0
                    self.entry_price = 0.0

        self.current_step += 1
        new_price = self.prices[self.current_step] if self.current_step < len(self.prices) else price
        self.equity = self.cash + self.shares * new_price
        self.peak_equity = max(self.peak_equity, self.equity)

        step_return = (self.equity - old_equity) / (old_equity + 1e-12)
        curr_exposure = (self.shares * new_price) / (self.equity + 1e-12)
        drawdown = (self.equity - self.peak_equity) / (self.peak_equity + 1e-12)

        if self.reward_fn == "conservative":
            reward = conservative_reward(
                step_return=step_return,
                prev_exposure=self.prev_exposure,
                curr_exposure=curr_exposure,
                drawdown=drawdown,
            )
        else:
            reward = step_return

        self.prev_exposure = curr_exposure
        self.episode_returns.append(step_return)

        terminated = self.current_step >= len(self.prices) - 1
        truncated = self.equity <= self.initial_capital * 0.5

        info = {
            "equity": self.equity,
            "cash": self.cash,
            "shares": self.shares,
            "action": int(direction),
            "size": size,
            "step_return": step_return,
        }
        return self._get_obs(), reward, terminated, truncated, info
