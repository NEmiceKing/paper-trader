from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, field_validator


class IBKRConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1


class DataConfig(BaseModel):
    default_start: str = "2020-01-01"
    historical_source: str = "yfinance"
    training_bar_size: str = "1d"
    live_bar_size: str = "30 mins"
    provider_priority: list[str] = field(default_factory=lambda: ["yfinance"])
    openbb_api_key: str = ""


class AgentTrainingConfig(BaseModel):
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    ent_coef: float = 0.01


class AgentInferenceConfig(BaseModel):
    deterministic: bool = True


class AgentConfig(BaseModel):
    model_name: str = "PPO"
    policy: str = "MlpPolicy"
    observation_bars: int = 20
    reward_fn: str = "conservative"
    net_arch: list[int] = [128, 128]
    training: AgentTrainingConfig = field(default_factory=AgentTrainingConfig)
    inference: AgentInferenceConfig = field(default_factory=AgentInferenceConfig)


class RiskConfig(BaseModel):
    max_position_pct: float = 0.10
    max_sector_pct: float = 0.30
    max_daily_loss_pct: float = 0.05
    max_total_drawdown_pct: float = 0.15
    min_hold_bars: int = 1
    max_trades_per_day: int = 10
    min_trade_value: float = 500.0


class LoggingConfig(BaseModel):
    level: str = "INFO"
    trades_db: str = "logs/trades.db"
    dashboard_port: int = 8501
    trading_interval_seconds: int = 300


class LLMConfig(BaseModel):
    enabled: bool = False
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key_env_var: str = "ANTHROPIC_API_KEY"
    fundamentals_enabled: bool = True
    news_enabled: bool = True
    sentiment_enabled: bool = True
    technical_llm_enabled: bool = True
    analysis_interval_minutes: int = 60
    cache_ttl_minutes: int = 60


class DebateConfig(BaseModel):
    enabled: bool = False
    max_rounds: int = 3
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    api_key_env_var: str = "DEEPSEEK_API_KEY"
    min_confidence_to_trade: float = 0.6


class FactorMiningConfig(BaseModel):
    enabled: bool = False
    max_factors: int = 50
    top_k: int = 10
    ic_threshold: float = 0.02
    population_size: int = 100
    generations: int = 20


class SignalFusionConfig(BaseModel):
    rl_weight: float = 0.7
    llm_override_threshold: float = 0.8
    agreement_threshold: float = 0.6
    debate_boost: float = 0.15
    min_override_confidence: float = 0.6


class NLConfig(BaseModel):
    enabled: bool = False
    parser_mode: str = "pattern"


class AppConfig(BaseModel):
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    data: DataConfig = field(default_factory=DataConfig)
    symbols: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL"])
    agent: AgentConfig = field(default_factory=AgentConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    factor_mining: FactorMiningConfig = field(default_factory=FactorMiningConfig)
    signal_fusion: SignalFusionConfig = field(default_factory=SignalFusionConfig)
    nlp: NLConfig = field(default_factory=NLConfig)
    paper_trading_capital: float = 100000.0


def load_config(path: str = "config/settings.yaml") -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    ibkr = IBKRConfig(**raw.get("ibkr", {}))
    data_raw = raw.get("data", {})
    ds = data_raw.get("sources", {})
    bs = data_raw.get("bar_sizes", {})
    data = DataConfig(
        default_start=data_raw.get("default_start", "2020-01-01"),
        historical_source=ds.get("historical", "yfinance"),
        training_bar_size=bs.get("training", "1d"),
        live_bar_size=bs.get("live", "30 mins"),
        provider_priority=data_raw.get("provider_priority", ["yfinance"]),
        openbb_api_key=data_raw.get("openbb_api_key", ""),
    )

    symbols = raw.get("symbols", {}).get("watchlist", ["SPY", "QQQ", "AAPL"])

    agent_raw = raw.get("agent", {})
    agent_train_raw = agent_raw.get("training", {})
    agent_inf_raw = agent_raw.get("inference", {})
    agent = AgentConfig(
        model_name=agent_raw.get("model_name", "PPO"),
        policy=agent_raw.get("policy", "MlpPolicy"),
        observation_bars=agent_raw.get("observation_bars", 20),
        reward_fn=agent_raw.get("reward_fn", "conservative"),
        net_arch=agent_raw.get("net_arch", [128, 128]),
        training=AgentTrainingConfig(**agent_train_raw),
        inference=AgentInferenceConfig(**agent_inf_raw),
    )

    risk = RiskConfig(**raw.get("risk", {}))

    log_raw = raw.get("logging", {})
    log = LoggingConfig(
        level=log_raw.get("level", "INFO"),
        trades_db=log_raw.get("trades_db", "logs/trades.db"),
        dashboard_port=log_raw.get("dashboard_port", 8501),
        trading_interval_seconds=log_raw.get("trading_interval_seconds", 300),
    )

    llm = LLMConfig(**raw.get("llm", {}))
    debate_raw = raw.get("debate", {})
    debate = DebateConfig(
        enabled=debate_raw.get("enabled", False),
        max_rounds=debate_raw.get("max_rounds", 3),
        llm_provider=debate_raw.get("llm_provider", raw.get("llm", {}).get("provider", "deepseek")),
        llm_model=debate_raw.get("llm_model", raw.get("llm", {}).get("model", "deepseek-chat")),
        api_key_env_var=debate_raw.get("api_key_env_var", raw.get("llm", {}).get("api_key_env_var", "DEEPSEEK_API_KEY")),
        min_confidence_to_trade=debate_raw.get("min_confidence_to_trade", 0.6),
    )
    factor_mining = FactorMiningConfig(**raw.get("factor_mining", {}))
    signal_fusion = SignalFusionConfig(**raw.get("signal_fusion", {}))
    nlp = NLConfig(**raw.get("nlp", {}))

    paper_capital = raw.get("paper_trading_capital", 100000.0)

    return AppConfig(
        ibkr=ibkr, data=data, symbols=symbols, agent=agent, risk=risk,
        logging=log, llm=llm, debate=debate, factor_mining=factor_mining,
        signal_fusion=signal_fusion, nlp=nlp,
        paper_trading_capital=paper_capital,
    )
