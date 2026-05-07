import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnalystReport:
    timestamp: datetime
    symbol: str
    analyst_name: str
    signal: int
    confidence: float
    summary: str
    details: dict | None = None


@dataclass
class AnalystConfig:
    fundamentals_enabled: bool = True
    news_enabled: bool = True
    sentiment_enabled: bool = True
    technical_llm_enabled: bool = True
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    api_key_env_var: str = "ANTHROPIC_API_KEY"
    cache_ttl_minutes: int = 60


# ── LLM Client Abstraction ────────────────────────────────────────


def _get_llm_client(config: AnalystConfig):
    """Return a callable `llm(messages) -> str` for the configured provider."""
    provider = config.llm_provider.lower()
    api_key = os.environ.get(config.api_key_env_var) or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")

    if provider == "anthropic":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            def _call(messages: list[dict], system: str = "") -> str:
                resp = client.messages.create(
                    model=config.llm_model,
                    max_tokens=1024,
                    system=system,
                    messages=messages,
                )
                return resp.content[0].text
            return _call
        except ImportError:
            pass
    elif provider in ("openai", "deepseek"):
        try:
            import openai
            if provider == "deepseek":
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com",
                )
            else:
                client = openai.OpenAI(api_key=api_key)
            def _call(messages: list[dict], system: str = "") -> str:
                full = [{"role": "system", "content": system}] + messages if system else messages
                resp = client.chat.completions.create(
                    model=config.llm_model,
                    max_tokens=1024,
                    messages=full,
                )
                return resp.choices[0].message.content
            return _call
        except ImportError:
            pass

    raise ImportError(f"No LLM provider available. Install anthropic or openai package.")


# ── Base Analyst ──────────────────────────────────────────────────


class BaseAnalyst(ABC):
    """Abstract analyst that produces an AnalystReport for a symbol."""

    def __init__(self, llm, config: AnalystConfig):
        self.llm = llm
        self.config = config

    @abstractmethod
    def analyze(self, symbol: str, df: pd.DataFrame) -> AnalystReport:
        ...

    def _make_llm_call(self, system_prompt: str, user_prompt: str) -> dict:
        """Make a structured LLM call, expecting JSON response with signal, confidence, summary."""
        messages = [{"role": "user", "content": user_prompt}]
        try:
            text = self.llm(messages, system=system_prompt)
            # Try to extract JSON from response
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return {"signal": 0, "confidence": 0.3, "summary": text[:200]}


# ── Analyst Implementations ───────────────────────────────────────


class FundamentalsAnalyst(BaseAnalyst):
    """Analyzes company fundamentals: valuation, growth, financial health."""

    SYSTEM = """You are a fundamental equity analyst. Analyze the given financial metrics and provide:
1. A trading signal: -1 (bearish), 0 (neutral), or +1 (bullish)
2. Confidence: 0.0 to 1.0
3. A one-sentence summary of your analysis.

Respond in JSON: {"signal": int, "confidence": float, "summary": "string"}
Do NOT include any other text. Only valid JSON."""

    def analyze(self, symbol: str, df: pd.DataFrame) -> AnalystReport:
        close = df["close"].values[-1] if len(df) > 0 else 0
        ret_20d = (df["close"].iloc[-1] / df["close"].iloc[-min(21, len(df))] - 1) if len(df) > 20 else 0
        ret_60d = (df["close"].iloc[-1] / df["close"].iloc[-min(61, len(df))] - 1) if len(df) > 60 else 0
        vol = float(df["volume"].mean()) if "volume" in df.columns else 0

        # Try to get fundamental data via yfinance
        fundamental_text = ""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            pe = info.get("trailingPE", "N/A")
            pb = info.get("priceToBook", "N/A")
            eps_growth = info.get("earningsGrowth", "N/A")
            de = info.get("debtToEquity", "N/A")
            roe = info.get("returnOnEquity", "N/A")
            fundamental_text = (
                f"PE Ratio: {pe}, PB Ratio: {pb}, EPS Growth: {eps_growth}, "
                f"Debt/Equity: {de}, ROE: {roe}"
            )
        except Exception:
            pass

        user = (
            f"Symbol: {symbol}\n"
            f"Latest Close: ${close:.2f}\n"
            f"20-day Return: {ret_20d:.2%}\n"
            f"60-day Return: {ret_60d:.2%}\n"
            f"Avg Volume: {vol:,.0f}\n"
        )
        if fundamental_text:
            user += f"Fundamentals: {fundamental_text}\n"
        else:
            user += "Fundamentals: Not available. Use price action only.\n"
        user += "Evaluate whether this stock is undervalued, fairly valued, or overvalued."

        result = self._make_llm_call(self.SYSTEM, user)
        return AnalystReport(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            analyst_name="fundamentals",
            signal=result.get("signal", 0),
            confidence=result.get("confidence", 0.5),
            summary=result.get("summary", ""),
            details={"fundamental_text": fundamental_text},
        )


class NewsAnalyst(BaseAnalyst):
    """Analyzes real news headlines and summaries for a symbol.

    Uses yfinance's built-in news feed (free, no API key required).
    """

    SYSTEM = """You are a financial news analyst. Based on the provided REAL news headlines and summaries, analyze:
1. The overall sentiment: bullish (-1 to +1, where +1 is very bullish)
2. Whether there are specific catalysts mentioned (earnings, product launches, regulatory changes, etc.)
3. The confidence of your assessment (0.0 to 1.0)

Respond in JSON: {"signal": int, "confidence": float, "summary": "one-sentence key takeaway from the news"}
Do NOT include any other text. Only valid JSON."""

    def analyze(self, symbol: str, df: pd.DataFrame) -> AnalystReport:
        from src.data.news_fetcher import NewsFetcher

        # Fetch real news
        fetcher = NewsFetcher()
        articles = fetcher.fetch(symbol, max_articles=8)
        news_text = fetcher.format_for_llm(articles)

        close = df["close"].values[-1] if len(df) > 0 else 0
        ret_5d = (df["close"].iloc[-1] / df["close"].iloc[-min(6, len(df))] - 1) if len(df) > 5 else 0

        user = (
            f"Symbol: {symbol}\n"
            f"Latest Close: ${close:.2f}\n"
            f"5-day Return: {ret_5d:.2%}\n\n"
            f"RECENT NEWS HEADLINES:\n{news_text}\n\n"
            f"Analyze the news sentiment. Are the headlines predominantly positive, negative, or mixed?\n"
            f"Is there any specific catalyst (earnings beat, product launch, lawsuit, regulatory issue, etc.)?\n"
            f"Weigh the credibility of sources and the potential market impact."
        )

        result = self._make_llm_call(self.SYSTEM, user)

        # Count articles for additional context
        article_count = len(articles)

        return AnalystReport(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            analyst_name="news",
            signal=result.get("signal", 0),
            confidence=result.get("confidence", 0.5),
            summary=result.get("summary", ""),
            details={"article_count": article_count},
        )


class SentimentAnalyst(BaseAnalyst):
    """Analyzes market sentiment indicators."""

    SYSTEM = """You are a market sentiment analyst. Based on technical and volume indicators, provide:
1. A trading signal: -1 (bearish), 0 (neutral), or +1 (bullish)
2. Confidence: 0.0 to 1.0
3. A one-sentence summary of market sentiment.

Respond in JSON: {"signal": int, "confidence": float, "summary": "string"}
Do NOT include any other text. Only valid JSON."""

    def analyze(self, symbol: str, df: pd.DataFrame) -> AnalystReport:
        close = df["close"].values[-1] if len(df) > 0 else 0
        bb_width = 0
        rsi = 50
        atr_ratio = 0

        if "bb_width" in df.columns:
            bb_width = float(df["bb_width"].iloc[-1]) if not pd.isna(df["bb_width"].iloc[-1]) else 0
        if "rsi_14" in df.columns:
            rsi = float(df["rsi_14"].iloc[-1]) if not pd.isna(df["rsi_14"].iloc[-1]) else 50
        if "atr_ratio" in df.columns:
            atr_ratio = float(df["atr_ratio"].iloc[-1]) if not pd.isna(df["atr_ratio"].iloc[-1]) else 0

        user = (
            f"Symbol: {symbol}\n"
            f"Latest Close: ${close:.2f}\n"
            f"RSI(14): {rsi:.1f}\n"
            f"Bollinger Band Width: {bb_width:.4f}\n"
            f"ATR Ratio: {atr_ratio:.4f}\n"
            f"RSI > 70 is overbought, RSI < 30 is oversold. High BB width means high volatility.\n"
            f"Assess overall market sentiment for this stock."
        )

        result = self._make_llm_call(self.SYSTEM, user)
        return AnalystReport(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            analyst_name="sentiment",
            signal=result.get("signal", 0),
            confidence=result.get("confidence", 0.5),
            summary=result.get("summary", ""),
        )


class TechnicalLLMAnalyst(BaseAnalyst):
    """LLM-powered technical analysis that interprets chart patterns."""

    SYSTEM = """You are a technical chart analyst. Based on the provided technical indicators, provide:
1. A trading signal: -1 (bearish), 0 (neutral), or +1 (bullish)
2. Confidence: 0.0 to 1.0
3. A one-sentence summary of your technical analysis.

Respond in JSON: {"signal": int, "confidence": float, "summary": "string"}
Do NOT include any other text. Only valid JSON."""

    def analyze(self, symbol: str, df: pd.DataFrame) -> AnalystReport:
        close = df["close"].values[-1] if len(df) > 0 else 0

        indicators = {}
        indicator_cols = ["rsi_14", "macd_line", "macd_signal", "macd_hist", "sma_20", "sma_50",
                          "sma_ratio", "atr_14", "atr_ratio", "bb_upper", "bb_lower", "bb_width",
                          "volume_ratio"]
        for col in indicator_cols:
            if col in df.columns:
                val = df[col].iloc[-1]
                indicators[col] = round(float(val), 4) if not pd.isna(val) else "N/A"
            else:
                indicators[col] = "N/A"

        ret_1d = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) if len(df) >= 2 else 0
        ret_5d = (df["close"].iloc[-1] / df["close"].iloc[-min(6, len(df))] - 1) if len(df) > 5 else 0
        ret_20d = (df["close"].iloc[-1] / df["close"].iloc[-min(21, len(df))] - 1) if len(df) > 20 else 0

        user = (
            f"Symbol: {symbol}\n"
            f"Latest Close: ${close:.2f}\n"
            f"Returns: 1d={ret_1d:.2%}, 5d={ret_5d:.2%}, 20d={ret_20d:.2%}\n"
            f"Indicators: {json.dumps(indicators)}\n"
            f"Look for MACD crossovers, RSI divergences, trend confirmations, and chart patterns."
        )

        result = self._make_llm_call(self.SYSTEM, user)
        return AnalystReport(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            analyst_name="technical_llm",
            signal=result.get("signal", 0),
            confidence=result.get("confidence", 0.5),
            summary=result.get("summary", ""),
        )


# ── Analyst Manager ───────────────────────────────────────────────


class LLMAnalystManager:
    """Orchestrates all analysts with caching, rate limiting, and batch processing."""

    def __init__(self, config):
        """
        Args:
            config: LLMConfig from config/loader.py
        """
        self.analyst_config = AnalystConfig(
            fundamentals_enabled=config.fundamentals_enabled,
            news_enabled=config.news_enabled,
            sentiment_enabled=config.sentiment_enabled,
            technical_llm_enabled=config.technical_llm_enabled,
            llm_provider=config.provider,
            llm_model=config.model,
            api_key_env_var=config.api_key_env_var,
            cache_ttl_minutes=config.cache_ttl_minutes,
        )
        self.analysis_interval = config.analysis_interval_minutes

        # Initialize LLM client
        try:
            self.llm = _get_llm_client(self.analyst_config)
            self._ready = True
        except ImportError as e:
            logger.warning(f"LLM client not available: {e}. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
            self.llm = None
            self._ready = False

        # Initialize analysts
        self._analysts: dict[str, BaseAnalyst] = {}
        if self._ready:
            if config.fundamentals_enabled:
                self._analysts["fundamentals"] = FundamentalsAnalyst(self.llm, self.analyst_config)
            if config.news_enabled:
                self._analysts["news"] = NewsAnalyst(self.llm, self.analyst_config)
            if config.sentiment_enabled:
                self._analysts["sentiment"] = SentimentAnalyst(self.llm, self.analyst_config)
            if config.technical_llm_enabled:
                self._analysts["technical_llm"] = TechnicalLLMAnalyst(self.llm, self.analyst_config)

        # Store for persistence
        from src.monitor.llm_store import LLMAnalysisStore
        self.store = LLMAnalysisStore()

    @property
    def ready(self) -> bool:
        return self._ready and len(self._analysts) > 0

    def analyze_symbol(self, symbol: str, df: pd.DataFrame,
                       force_refresh: bool = False) -> list[AnalystReport]:
        """Run all enabled analysts for a symbol. Uses cache unless force_refresh=True."""
        if not self.ready:
            return []

        # Check cache
        if not force_refresh:
            try:
                cached = self.store.load_reports(symbol)
                if not cached.empty:
                    latest = cached["timestamp"].max()
                    age_minutes = (datetime.now(timezone.utc) - latest).total_seconds() / 60
                    if age_minutes < self.analyst_config.cache_ttl_minutes:
                        reports = []
                        for _, row in cached[cached["timestamp"] == latest].iterrows():
                            reports.append(AnalystReport(
                                timestamp=row["timestamp"],
                                symbol=symbol,
                                analyst_name=row["analyst_name"],
                                signal=int(row["signal"]),
                                confidence=float(row["confidence"]),
                                summary=str(row.get("summary", "")),
                            ))
                        return reports
            except Exception:
                pass

        reports: list[AnalystReport] = []
        for name, analyst in self._analysts.items():
            try:
                report = analyst.analyze(symbol, df)
                reports.append(report)
                logger.info(f"[{symbol}] {name}: signal={report.signal}, conf={report.confidence:.2f}")
            except Exception as e:
                logger.warning(f"[{symbol}] {name} analysis failed: {e}")

        if reports:
            try:
                self.store.save_reports(symbol, reports)
            except Exception as e:
                logger.warning(f"Failed to save reports for {symbol}: {e}")

        return reports

    def get_llm_features(self, symbol: str) -> dict[str, float]:
        """Get LLM feature vector for injection into RL observation space.

        Returns dict with keys: llm_sentiment_avg, llm_agreement, llm_fundamentals_score, llm_technical_score
        """
        reports = self.analyze_symbol(symbol, pd.DataFrame())  # reads cached
        if not reports:
            return {
                "llm_sentiment_avg": 0.0,
                "llm_agreement": 0.0,
                "llm_fundamentals_score": 0.0,
                "llm_technical_score": 0.0,
            }

        scores = [r.signal * r.confidence for r in reports]
        fundamentals = next((r for r in reports if r.analyst_name == "fundamentals"), None)
        technical = next((r for r in reports if r.analyst_name == "technical_llm"), None)

        return {
            "llm_sentiment_avg": float(np.mean(scores)) if scores else 0.0,
            "llm_agreement": float(1.0 - np.std(scores)) if len(scores) > 1 else 0.0,
            "llm_fundamentals_score": float(fundamentals.signal * fundamentals.confidence) if fundamentals else 0.0,
            "llm_technical_score": float(technical.signal * technical.confidence) if technical else 0.0,
        }
