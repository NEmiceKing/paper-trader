"""
Natural language command handler for paper-trader CLI and GUI.

Parses NL commands and dispatches them to the appropriate action.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParsedCommand:
    command_type: str
    parameters: dict = field(default_factory=dict)
    raw_text: str = ""


def parse_command(nl_text: str) -> ParsedCommand:
    """Classify a natural language command and extract parameters.

    Examples:
        "start paper trading with 50k on SPY and QQQ"
        "train a new model with conservative settings"
        "run backtest on NVDA"
        "show me the portfolio"
        "set aggressive risk mode"
    """
    text = nl_text.lower().strip()

    if any(kw in text for kw in ["start", "paper trade", "live trade", "开始交易"]):
        cmd = "start_trading"
    elif any(kw in text for kw in ["stop trading", "停止交易", "halt"]):
        cmd = "stop_trading"
    elif any(kw in text for kw in ["train", "训练"]):
        cmd = "train"
    elif any(kw in text for kw in ["backtest", "回测"]):
        cmd = "backtest"
    elif any(kw in text for kw in ["portfolio", "持仓", "show me", "status", "状态"]):
        cmd = "show_status"
    elif any(kw in text for kw in ["set conservative", "set aggressive", "risk mode", "风控"]):
        cmd = "change_risk"
    elif any(kw in text for kw in ["analyze", "分析"]):
        cmd = "analyze"
    else:
        cmd = "unknown"

    # Extract symbols (use lookbehind/lookahead to avoid \b issues with Chinese chars)
    import re
    symbols = re.findall(r'(?<![A-Za-z])([A-Z]{1,5})(?![A-Za-z])', nl_text.upper())
    filter_words = {"I", "A", "THE", "IT", "IS", "ARE", "AN", "USE", "FOR", "AND", "MY", "ON", "IN",
                    "K", "AI", "RL", "LLM", "DCA", "ETF", "AT", "TO", "BY", "BE", "NO", "OR", "AS",
                    "IF", "OF", "WE", "HE", "DO", "GO", "SO", "US", "AM", "PM", "PCT", "NEW",
                    "START", "PAPER", "WITH", "TRADE", "TRADING", "MODEL", "RUN", "SET", "BACKTEST"}
    symbols = [s for s in symbols if s not in filter_words]

    # Extract capital
    capital = None
    cap_match = re.search(r'(\d[\d,.]*)\s*(k|K|万)?\s*(capital|dollars|美金|美元|资金)?', text)
    if cap_match:
        val = float(cap_match.group(1).replace(",", ""))
        unit = cap_match.group(2)
        if unit in ("k", "K"):
            val *= 1000
        elif unit == "万":
            val *= 10000
        capital = val

    params = {}
    if symbols:
        params["symbols"] = symbols
    if capital:
        params["capital"] = capital

    if "conservative" in text or "保守" in text:
        params["risk_profile"] = "conservative"
    elif "aggressive" in text or "激进" in text:
        params["risk_profile"] = "aggressive"

    return ParsedCommand(command_type=cmd, parameters=params, raw_text=nl_text)


def execute_command(command: ParsedCommand, config=None, state: dict | None = None) -> str:
    """Execute a parsed command and return a response message.

    Args:
        command: Parsed NL command
        config: Optional AppConfig
        state: Optional state dict (for GUI integration)

    Returns:
        Response message string
    """
    if state is None:
        state = {}

    if command.command_type == "start_trading":
        syms = command.parameters.get("symbols", config.symbols if config else ["SPY"])
        cap = command.parameters.get("capital", 100000)
        return f"Paper trading started for {', '.join(syms)} with ${cap:,.0f} capital. Run `make paper` in terminal."

    elif command.command_type == "stop_trading":
        state["trading_active"] = False
        return "Paper trading stopped."

    elif command.command_type == "train":
        risk = command.parameters.get("risk_profile", "moderate")
        return f"Training requested with {risk} profile. Run `make train` in terminal."

    elif command.command_type == "backtest":
        syms = command.parameters.get("symbols", config.symbols if config else ["SPY"])
        return f"Backtest queued for {', '.join(syms)}. Run `make backtest` in terminal."

    elif command.command_type == "change_risk":
        risk = command.parameters.get("risk_profile", "moderate")
        return f"Risk mode changed to {risk}. Settings will apply on next trading session."

    elif command.command_type == "analyze":
        syms = command.parameters.get("symbols", config.symbols if config else ["SPY"])
        return f"Analysis requested for {', '.join(syms)}. Run `make analyze` in terminal."

    elif command.command_type == "show_status":
        return "Portfolio status view requested. Open dashboard to view."

    else:
        return f"Unknown command: '{command.raw_text}'. Try: start trading, stop trading, train, backtest, analyze, set risk mode."
