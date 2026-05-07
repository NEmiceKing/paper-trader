from src.agent.reward import conservative_reward


def test_reward_positive_return():
    r = conservative_reward(0.01, 0.0, 0.5, drawdown=0.0)
    assert r > 0


def test_reward_negative_return():
    r = conservative_reward(-0.01, 0.0, 0.0, drawdown=0.0)
    assert r < 0


def test_reward_drawdown_penalty():
    r_no_dd = conservative_reward(0.01, 0.0, 0.5, drawdown=0.0)
    r_dd = conservative_reward(0.01, 0.0, 0.5, drawdown=-0.10)
    assert r_dd < r_no_dd


def test_reward_trade_penalty():
    r_no_trade = conservative_reward(0.0, 0.5, 0.5, drawdown=0.0)
    r_trade = conservative_reward(0.0, 0.0, 0.5, drawdown=0.0)
    assert r_trade < r_no_trade
