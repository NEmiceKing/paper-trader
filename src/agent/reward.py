import numpy as np


def conservative_reward(
    step_return: float,
    prev_exposure: float,
    curr_exposure: float,
    drawdown: float,
    lambda_risk: float = 0.5,
    trade_penalty: float = 0.001,
    drawdown_threshold: float = 0.05,
    drawdown_penalty: float = 2.0,
) -> float:
    reward = step_return

    if step_return != 0:
        reward -= lambda_risk * abs(step_return)

    exposure_change = abs(curr_exposure - prev_exposure)
    reward -= trade_penalty * exposure_change

    if drawdown < -drawdown_threshold:
        reward -= drawdown_penalty * abs(drawdown + drawdown_threshold)

    return float(reward)
