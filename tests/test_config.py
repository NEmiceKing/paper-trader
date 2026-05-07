from src.config.loader import load_config


def test_load_config():
    config = load_config()
    assert len(config.symbols) > 0
    assert config.agent.training.learning_rate == 0.0003
    assert config.risk.max_position_pct == 0.40  # small account
    assert config.ibkr.port == 4001  # IB Gateway paper
