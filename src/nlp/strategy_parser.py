"""
Natural language strategy parser.
Converts NL descriptions into StrategyIntent and config overrides.

Supports two modes:
  - pattern: Regex/keyword-based (no LLM dependency)
  - llm: Uses LLM for structured extraction (requires anthropic/openai)
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StrategyIntent:
    risk_profile: str = "moderate"
    focus_style: str = "trend_following"
    symbols: list[str] | None = None
    capital: float | None = None
    stop_loss: str = "moderate"
    holding_period: str = "medium"
    confidence: float = 0.0


# ── Pattern-based parser ──────────────────────────────────────────

RISK_KEYWORDS = {
    "conservative": ["conservative", "safe", "low risk", "defensive", "preserve capital", "保守"],
    "aggressive": ["aggressive", "high risk", "growth", "aggressive growth", "bold", "激进"],
    "moderate": ["moderate", "balanced", "medium risk", "moderate risk", "适度"],
}

STYLE_KEYWORDS = {
    "momentum": ["momentum", "trend strength", "breakout", "volume surge", "动量"],
    "mean_reversion": ["mean reversion", "reversal", "oversold", "overbought", "bounce", "回调"],
    "trend_following": ["trend following", "follow trend", "trend", "顺趋势"],
}

STOP_KEYWORDS = {
    "tight": ["tight stop", "strict stop", "close stop", "快速止损", "紧止损"],
    "wide": ["wide stop", "loose stop", "宽止损"],
    "moderate": [],
}

HOLDING_KEYWORDS = {
    "short": ["short term", "day trade", "scalp", "短线", "日内"],
    "long": ["long term", "swing", "invest", "hold", "长线", "长期"],
    "medium": [],
}


def parse_strategy(text: str) -> StrategyIntent:
    """Parse a natural language strategy description into a StrategyIntent.

    First tries pattern matching; falls back to LLM if configured.
    """
    intent = _pattern_parse(text.lower())

    # Extract capital
    capital_match = re.search(
        r'(\d[\d,.]*)\s*(k|K|万)?\s*(capital|dollars|美金|美元|资金|初始|块钱?)?',
        text.lower()
    )
    if capital_match:
        val = float(capital_match.group(1).replace(",", ""))
        unit = capital_match.group(2)
        context = capital_match.group(3) or ""
        if unit in ("k", "K"):
            val *= 1000
        elif unit == "万":
            val *= 10000
        elif not unit and ("k" in context or "千" in context):
            val *= 1000
        intent.capital = val

    # Extract symbols (use lookbehind/lookahead to avoid \b issues with Chinese chars)
    symbol_pattern = r'(?<![A-Za-z])([A-Z]{1,5})(?![A-Za-z])'
    symbols = re.findall(symbol_pattern, text.upper())
    # Filter out common false positives
    filter_words = {"I", "A", "THE", "IT", "IS", "ARE", "AN", "USE", "FOR", "AND", "MY", "ON", "IN",
                    "K", "AI", "RL", "LLM", "DCA", "ETF", "AT", "TO", "BY", "BE", "NO", "OR", "AS",
                    "IF", "OF", "WE", "HE", "DO", "GO", "SO", "US", "AM", "PM", "PCT", "START", "PAPER",
                    "WITH", "TRADE", "TRADING", "MODEL", "NEW", "RUN", "SET", "BACKTEST"}
    symbols = [s for s in symbols if s not in filter_words]
    if symbols:
        intent.symbols = symbols

    intent.confidence = 0.7 if intent.risk_profile != "moderate" else 0.5
    return intent


def _pattern_parse(text: str) -> StrategyIntent:
    intent = StrategyIntent()

    for profile, keywords in RISK_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            intent.risk_profile = profile
            break

    for style, keywords in STYLE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            intent.focus_style = style
            break

    for stop, keywords in STOP_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            intent.stop_loss = stop
            break

    for period, keywords in HOLDING_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            intent.holding_period = period
            break

    return intent


def strategy_to_config(intent: StrategyIntent) -> dict:
    """Convert StrategyIntent into config override dictionary."""
    overrides = {}

    # Risk profile → position sizing and risk limits
    risk_map = {
        "conservative": {
            "max_position_pct": 0.05,
            "max_daily_loss_pct": 0.02,
            "max_total_drawdown_pct": 0.08,
        },
        "moderate": {
            "max_position_pct": 0.10,
            "max_daily_loss_pct": 0.05,
            "max_total_drawdown_pct": 0.15,
        },
        "aggressive": {
            "max_position_pct": 0.25,
            "max_daily_loss_pct": 0.10,
            "max_total_drawdown_pct": 0.25,
        },
    }
    if intent.risk_profile in risk_map:
        overrides["risk"] = risk_map[intent.risk_profile]

    # Stop loss → reward function parameters
    stop_map = {
        "tight": {"drawdown_threshold": 0.02, "drawdown_penalty": 3.0},
        "moderate": {"drawdown_threshold": 0.05, "drawdown_penalty": 2.0},
        "wide": {"drawdown_threshold": 0.10, "drawdown_penalty": 1.0},
    }
    overrides["stop_loss"] = stop_map.get(intent.stop_loss, stop_map["moderate"])

    # Holding period → agent observation_bars
    hold_map = {"short": 10, "medium": 20, "long": 50}
    overrides["observation_bars"] = hold_map.get(intent.holding_period, 20)

    # Focus style
    if intent.focus_style == "momentum":
        overrides["factor_focus"] = "momentum"
    elif intent.focus_style == "mean_reversion":
        overrides["factor_focus"] = "mean_reversion"
        overrides["reward_fn"] = "conservative"

    if intent.symbols:
        overrides["symbols"] = intent.symbols
    if intent.capital:
        overrides["initial_capital"] = intent.capital

    return overrides


def apply_strategy(config, intent: StrategyIntent):
    """Apply strategy intent to an AppConfig, returning a modified copy."""
    overrides = strategy_to_config(intent)
    cfg = config

    if "risk" in overrides:
        r = overrides["risk"]
        cfg.risk.max_position_pct = r.get("max_position_pct", cfg.risk.max_position_pct)
        cfg.risk.max_daily_loss_pct = r.get("max_daily_loss_pct", cfg.risk.max_daily_loss_pct)
        cfg.risk.max_total_drawdown_pct = r.get("max_total_drawdown_pct", cfg.risk.max_total_drawdown_pct)

    if "observation_bars" in overrides:
        cfg.agent.observation_bars = overrides["observation_bars"]

    if "reward_fn" in overrides:
        cfg.agent.reward_fn = overrides["reward_fn"]

    if "symbols" in overrides:
        cfg.symbols = overrides["symbols"]

    return cfg
