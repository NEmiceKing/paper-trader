import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DebateRound:
    round_number: int
    bull_argument: str
    bear_argument: str
    bull_score: float
    bear_score: float


@dataclass
class DebateResult:
    symbol: str
    timestamp: datetime
    rounds: list[DebateRound] = field(default_factory=list)
    composite_signal: int = 0
    confidence: float = 0.5
    bull_summary: str = ""
    bear_summary: str = ""


@dataclass
class DebateConfig:
    enabled: bool = False
    max_rounds: int = 3
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    api_key_env_var: str = "ANTHROPIC_API_KEY"
    min_confidence_to_trade: float = 0.6


def _get_llm_client(config: DebateConfig):
    provider = config.llm_provider.lower()
    api_key = os.environ.get(config.api_key_env_var) or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        def _call(system: str, prompt: str) -> str:
            resp = client.messages.create(
                model=config.llm_model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        return _call
    elif provider in ("openai", "deepseek"):
        import openai
        if provider == "deepseek":
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
            )
        else:
            client = openai.OpenAI(api_key=api_key)
        def _call(system: str, prompt: str) -> str:
            resp = client.chat.completions.create(
                model=config.llm_model,
                max_tokens=1024,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        return _call
    else:
        raise ImportError(f"Unsupported LLM provider: {provider}")


BULL_SYSTEM = """You are a bullish equity researcher. Your job is to find reasons to BUY the stock.
Argue convincingly using data provided. Consider technical trends, growth potential, and positive catalysts.
Focus on upside potential and why bears are wrong.

Respond ONLY in JSON:
{"argument": "your bullish case (2-3 sentences)", "score": float 0.0-1.0}
Score represents your conviction in the bullish case."""

BEAR_SYSTEM = """You are a bearish equity researcher. Your job is to find reasons to SELL or AVOID the stock.
Argue convincingly using data provided. Consider risks, overvaluation, negative trends, and downside catalysts.
Focus on what could go wrong and why bulls are too optimistic.

Respond ONLY in JSON:
{"argument": "your bearish case (2-3 sentences)", "score": float 0.0-1.0}
Score represents your conviction in the bearish case."""

JUDGE_SYSTEM = """You are an impartial market judge. After reviewing bull and bear arguments, synthesize a final composite view.
Determine the overall signal direction (-1=BEARISH, 0=NEUTRAL, +1=BULLISH) and your confidence (0.0-1.0).

Respond ONLY in JSON:
{"signal": int, "confidence": float, "bull_summary": "one-sentence bull position", "bear_summary": "one-sentence bear position"}"""


class DebateManager:
    """Orchestrates multi-round Bull vs Bear debate for trading decisions."""

    def __init__(self, config):
        self.config = config
        self.min_confidence = config.min_confidence_to_trade
        try:
            self.llm = _get_llm_client(config)
            self._ready = True
        except (ImportError, Exception) as e:
            logger.warning(f"LLM client not available for debate: {e}")
            self.llm = None
            self._ready = False

    def debate(self, symbol: str, df: pd.DataFrame,
               analyst_reports: list | None = None) -> DebateResult:
        """Run multi-round bull vs bear debate for a symbol.

        Args:
            symbol: Stock symbol
            df: Market data DataFrame with OHLCV and indicators
            analyst_reports: Optional list of AnalystReport from prior analysis phase

        Returns:
            DebateResult with composite signal and confidence
        """
        if not self._ready or self.llm is None:
            return DebateResult(
                symbol=symbol, timestamp=datetime.now(timezone.utc),
                composite_signal=0, confidence=0.5,
                bull_summary="LLM not available", bear_summary="LLM not available",
            )

        # Build context from data
        close = df["close"].values[-1] if len(df) > 0 else 0
        ret_5d = (df["close"].iloc[-1] / df["close"].iloc[-min(6, len(df))] - 1) if len(df) > 5 else 0
        ret_20d = (df["close"].iloc[-1] / df["close"].iloc[-min(21, len(df))] - 1) if len(df) > 20 else 0

        indicators_str = ""
        for col in ["rsi_14", "macd_hist", "sma_ratio", "atr_ratio", "bb_width"]:
            if col in df.columns and not pd.isna(df[col].iloc[-1]):
                indicators_str += f"  {col}: {df[col].iloc[-1]:.4f}\n"

        # Incorporate analyst reports if available
        reports_str = ""
        if analyst_reports:
            for r in analyst_reports:
                sign = "BULLISH" if r.signal > 0 else ("BEARISH" if r.signal < 0 else "NEUTRAL")
                reports_str += f"  {r.analyst_name}: {sign} (conf={r.confidence:.2f}) — {r.summary[:150]}\n"

        base_context = (
            f"Symbol: {symbol}\n"
            f"Latest Close: ${close:.2f}\n"
            f"5-day Return: {ret_5d:.2%}\n"
            f"20-day Return: {ret_20d:.2%}\n"
            f"Technical Indicators:\n{indicators_str}"
        )
        if reports_str:
            base_context += f"\nAnalyst Reports:\n{reports_str}"

        rounds = []
        bull_history = ""
        bear_history = ""

        for i in range(self.config.max_rounds):
            # Bull argues
            bull_prompt = base_context
            if i > 0:
                bull_prompt += f"\nPrevious round:\nBull: {bull_history}\nBear: {bear_history}\nRebutt the bear's arguments."
            else:
                bull_prompt += "\nMake your opening bullish case."

            try:
                bull_resp = json.loads(self._extract_json(self.llm(BULL_SYSTEM, bull_prompt)))
                bull_arg = bull_resp.get("argument", "")
                bull_score = float(bull_resp.get("score", 0.5))
            except Exception as e:
                logger.warning(f"Bull round {i+1} failed: {e}")
                bull_arg = f"Bullish case for {symbol} at ${close:.2f}."
                bull_score = 0.5

            # Bear argues
            bear_prompt = base_context
            if i > 0:
                bear_prompt += f"\nPrevious round:\nBull: {bull_arg}\nBear: {bear_history}\nRebutt the bull's arguments."
            else:
                bear_prompt += f"\nThe bull just argued: \"{bull_arg}\"\nMake your opening bearish case and rebuttal."

            try:
                bear_resp = json.loads(self._extract_json(self.llm(BEAR_SYSTEM, bear_prompt)))
                bear_arg = bear_resp.get("argument", "")
                bear_score = float(bear_resp.get("score", 0.5))
            except Exception as e:
                logger.warning(f"Bear round {i+1} failed: {e}")
                bear_arg = f"Bearish case for {symbol} at ${close:.2f}."
                bear_score = 0.5

            rounds.append(DebateRound(
                round_number=i + 1,
                bull_argument=bull_arg,
                bear_argument=bear_arg,
                bull_score=bull_score,
                bear_score=bear_score,
            ))
            bull_history = bull_arg
            bear_history = bear_arg

        # Judge synthesizes final view
        final_bull = rounds[-1].bull_argument
        final_bear = rounds[-1].bear_argument
        judge_prompt = (
            f"After {len(rounds)} rounds of debate for {symbol}:\n\n"
            f"Final Bull Argument: {final_bull}\n\n"
            f"Final Bear Argument: {final_bear}\n\n"
            f"Synthesize the final composite view. Consider the conviction scores and argument quality."
        )

        try:
            judge_resp = json.loads(self._extract_json(self.llm(JUDGE_SYSTEM, judge_prompt)))
            signal = int(judge_resp.get("signal", 0))
            confidence = float(judge_resp.get("confidence", 0.5))
            bull_summary = judge_resp.get("bull_summary", final_bull[:100])
            bear_summary = judge_resp.get("bear_summary", final_bear[:100])
        except Exception as e:
            logger.warning(f"Judge synthesis failed: {e}")
            # Fallback: compute from debate scores
            avg_bull = sum(r.bull_score for r in rounds) / len(rounds)
            avg_bear = sum(r.bear_score for r in rounds) / len(rounds)
            if avg_bull > avg_bear + 0.15:
                signal, confidence = 1, avg_bull
            elif avg_bear > avg_bull + 0.15:
                signal, confidence = -1, avg_bear
            else:
                signal, confidence = 0, max(avg_bull, avg_bear)
            bull_summary = final_bull[:100]
            bear_summary = final_bear[:100]

        result = DebateResult(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            rounds=rounds,
            composite_signal=signal,
            confidence=confidence,
            bull_summary=bull_summary,
            bear_summary=bear_summary,
        )

        # Persist result
        try:
            from src.monitor.llm_store import LLMAnalysisStore
            LLMAnalysisStore().save_debate_result(symbol, result)
        except Exception:
            pass

        return result

    def _extract_json(self, text: str) -> str:
        text = text.strip()
        if "```json" in text:
            return text.split("```json")[1].split("```")[0]
        elif "```" in text:
            return text.split("```")[1].split("```")[0]
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]
        return text
