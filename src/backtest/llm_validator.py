"""
Validate LLM analyst signals against historical returns.

Tests:
  1. Cached LLM signals vs next-period returns (hit rate, IC)
  2. Simple news headline sentiment vs next-day returns
  3. Technical rule-based signals as baseline comparison
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    analyst: str
    hit_rate: float  # % of times signal direction matched return direction
    mean_return_when_bullish: float
    mean_return_when_bearish: float
    ic: float  # Information Coefficient (correlation signal ~ future return)
    n_samples: int
    signal_distribution: dict  # {BULLISH: N, NEUTRAL: N, BEARISH: N}


def validate_llm_signals(symbol: str = "SPY", horizon_days: int = 1) -> list[ValidationResult]:
    """Compare cached LLM analysis signals to actual subsequent returns."""
    from src.monitor.llm_store import LLMAnalysisStore
    from src.data.store import MarketDataStore

    store = LLMAnalysisStore()
    data_store = MarketDataStore()

    reports_df = store.load_reports(symbol)
    if reports_df.empty:
        logger.warning(f"No cached LLM reports for {symbol}")
        return []

    try:
        prices = data_store.read_bars(symbol)
    except FileNotFoundError:
        logger.warning(f"No price data for {symbol}")
        return []

    # For each analyst, compute signal vs future return
    results = []
    for analyst_name in reports_df["analyst_name"].unique():
        analyst_reports = reports_df[reports_df["analyst_name"] == analyst_name].copy()
        analyst_reports = analyst_reports.sort_values("timestamp")

        hits = 0
        total = 0
        bullish_returns = []
        bearish_returns = []
        signals = []
        future_returns = []

        for _, row in analyst_reports.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            signal = int(row["signal"])
            confidence = float(row["confidence"])

            # Find the price at analysis time and horizon_days later
            try:
                price_idx = prices.index.get_indexer([ts], method="ffill")[0]
            except (IndexError, KeyError):
                continue

            if price_idx < 0 or price_idx + horizon_days >= len(prices):
                continue

            current_price = float(prices.iloc[price_idx]["close"])
            future_price = float(prices.iloc[price_idx + horizon_days]["close"])
            future_ret = (future_price - current_price) / current_price

            signals.append(signal * confidence)
            future_returns.append(future_ret)
            total += 1

            if signal > 0:
                bullish_returns.append(future_ret)
            elif signal < 0:
                bearish_returns.append(future_ret)

            if signal * future_ret > 0:
                hits += 1
            elif signal == 0:
                total -= 1  # neutral signals don't count for hit rate

        if total == 0:
            continue

        # Compute metrics
        hit_rate = hits / max(total, 1)
        mean_bull = float(np.mean(bullish_returns)) if bullish_returns else 0.0
        mean_bear = float(np.mean(bearish_returns)) if bearish_returns else 0.0

        # IC: correlation between signal*confidence and future return
        if len(signals) >= 3:
            ic = float(np.corrcoef(signals, future_returns)[0, 1])
            if np.isnan(ic):
                ic = 0.0
        else:
            ic = 0.0

        # Signal distribution
        dist = {
            "BULLISH": int((analyst_reports["signal"] > 0).sum()),
            "NEUTRAL": int((analyst_reports["signal"] == 0).sum()),
            "BEARISH": int((analyst_reports["signal"] < 0).sum()),
        }

        results.append(ValidationResult(
            analyst=analyst_name,
            hit_rate=round(hit_rate, 4),
            mean_return_when_bullish=round(mean_bull, 4),
            mean_return_when_bearish=round(mean_bear, 4),
            ic=round(ic, 4),
            n_samples=total,
            signal_distribution=dist,
        ))

    return results


def validate_news_sentiment(symbols: list[str] | None = None) -> dict:
    """Test if simple keyword sentiment from yfinance news predicts next-day returns."""
    from src.data.news_fetcher import NewsFetcher, NewsArticle
    from src.data.store import MarketDataStore

    if symbols is None:
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    fetcher = NewsFetcher(cache_ttl_minutes=0)  # bypass cache
    data_store = MarketDataStore()

    # Simple keyword-based sentiment
    positive_words = {
        "beat", "surge", "soar", "jump", "rally", "record high", "upgrade",
        "bullish", "strong", "profit", "growth", "beat estimates", "buyback",
        "dividend", "launch", "breakthrough", "partnership", "approval",
        "best-selling", "outperform", "positive", "momentum",
    }
    negative_words = {
        "plunge", "crash", "downgrade", "lawsuit", "investigation", "fine",
        "layoff", "loss", "decline", "drop", "weak", "bearish", "sell-off",
        "risk", "warning", "miss", "cut", "concern", "overvalued", "bubble",
        "slowdown", "tariff", "sanction", "probe", "recall", "delay",
    }

    results = {}
    for sym in symbols:
        articles = fetcher.fetch(sym, max_articles=20)
        if not articles:
            results[sym] = {"n_articles": 0, "sentiment": 0, "error": "no news"}
            continue

        pos_count = 0
        neg_count = 0
        for a in articles:
            text = (a.title + " " + a.summary).lower()
            p = sum(1 for w in positive_words if w in text)
            n = sum(1 for w in negative_words if w in text)
            pos_count += p
            neg_count += n

        total = pos_count + neg_count
        sentiment = (pos_count - neg_count) / max(total, 1)

        # Get next-day return
        try:
            prices = data_store.read_bars(sym)
            latest_close = float(prices["close"].iloc[-1])
            prev_close = float(prices["close"].iloc[-2]) if len(prices) >= 2 else latest_close
            next_day_ret = (latest_close - prev_close) / prev_close
        except FileNotFoundError:
            next_day_ret = None

        results[sym] = {
            "n_articles": len(articles),
            "positive_keywords": pos_count,
            "negative_keywords": neg_count,
            "keyword_sentiment": round(sentiment, 4),
            "next_day_return": round(next_day_ret, 4) if next_day_ret else None,
            "sentiment_correct": (
                (sentiment > 0 and next_day_ret and next_day_ret > 0) or
                (sentiment < 0 and next_day_ret and next_day_ret < 0)
            ) if next_day_ret else None,
        }

    return results


def validate_technical_baseline(symbol: str = "SPY") -> dict:
    """Benchmark: how well do simple technical rules predict next-day returns?"""
    from src.data.store import MarketDataStore
    from src.data.indicators import compute_features

    data_store = MarketDataStore()
    try:
        df = data_store.read_bars(symbol)
    except FileNotFoundError:
        return {"error": f"No data for {symbol}"}

    feat = compute_features(df)
    ret = df["close"].pct_change().shift(-1)  # next-day returns

    results = {}
    # Align indices
    common_idx = feat.index.intersection(ret.index)
    feat_aligned = feat.loc[common_idx]
    ret_aligned = ret.loc[common_idx]

    # RSI rule: RSI<30 = bullish, RSI>70 = bearish
    if "rsi_14" in feat_aligned.columns:
        rsi = feat_aligned["rsi_14"]
        rsi_signal = pd.Series(0, index=rsi.index)
        rsi_signal[rsi < 30] = 1
        rsi_signal[rsi > 70] = -1
        rsi_valid = rsi_signal != 0
        if rsi_valid.sum() > 0:
            rsi_hits = int(((rsi_signal[rsi_valid] * ret_aligned[rsi_valid]) > 0).sum())
            rsi_total = int(rsi_valid.sum())
            results["rsi"] = {
                "hit_rate": round(rsi_hits / max(rsi_total, 1), 4),
                "n_signals": rsi_total,
                "description": "RSI<30=BUY, RSI>70=SELL",
            }

    # MACD crossover rule
    if "macd_line" in feat_aligned.columns and "macd_signal" in feat_aligned.columns:
        macd_signal = pd.Series(0, index=feat_aligned.index)
        crossover_up = (feat_aligned["macd_line"] > feat_aligned["macd_signal"]) & (feat_aligned["macd_line"].shift(1) <= feat_aligned["macd_signal"].shift(1))
        crossover_down = (feat_aligned["macd_line"] < feat_aligned["macd_signal"]) & (feat_aligned["macd_line"].shift(1) >= feat_aligned["macd_signal"].shift(1))
        macd_signal[crossover_up] = 1
        macd_signal[crossover_down] = -1
        macd_valid = macd_signal != 0
        if macd_valid.sum() > 0:
            macd_hits = int(((macd_signal[macd_valid] * ret_aligned[macd_valid]) > 0).sum())
            macd_total = int(macd_valid.sum())
            results["macd_cross"] = {
                "hit_rate": round(macd_hits / max(macd_total, 1), 4),
                "n_signals": macd_total,
                "description": "MACD crossover",
            }

    # SMA crossover
    if "sma_20" in feat_aligned.columns and "sma_50" in feat_aligned.columns:
        sma_signal = pd.Series(0, index=feat_aligned.index)
        sma_signal[feat_aligned["sma_20"] > feat_aligned["sma_50"]] = 1
        sma_signal[feat_aligned["sma_20"] < feat_aligned["sma_50"]] = -1
        sma_changed = sma_signal.diff() != 0
        sma_valid = sma_changed & (sma_signal != 0)
        if sma_valid.sum() > 0:
            sma_hits = int(((sma_signal[sma_valid] * ret_aligned[sma_valid]) > 0).sum())
            sma_total = int(sma_valid.sum())
            results["sma_cross"] = {
                "hit_rate": round(sma_hits / max(sma_total, 1), 4),
                "n_signals": sma_total,
                "description": "SMA20/50 crossover",
            }

    # Buy & hold baseline
    buy_hold_hits = int((ret_aligned > 0).sum())
    buy_hold_total = int((~ret_aligned.isna()).sum())
    results["buy_and_hold"] = {
        "hit_rate": round(buy_hold_hits / max(buy_hold_total, 1), 4),
        "n_signals": buy_hold_total,
        "description": f"Up days / total days ({ret_aligned.index[0].date()} → {ret_aligned.index[-1].date()})",
    }

    return results


def run_full_validation(symbols: list[str] | None = None) -> dict:
    """Run all validation tests and return comprehensive results."""
    if symbols is None:
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    output = {
        "llm_signal_validation": {},
        "news_sentiment": {},
        "technical_baseline": {},
        "summary": {},
    }

    # Test 1: LLM cached signals
    logger.info("=" * 60)
    logger.info("TEST 1: LLM Cached Signal Validation")
    logger.info("=" * 60)
    for sym in symbols:
        results = validate_llm_signals(sym)
        if results:
            output["llm_signal_validation"][sym] = [r.__dict__ for r in results]
            for r in results:
                logger.info(
                    f"  [{sym}] {r.analyst:18s} | HitRate={r.hit_rate:.2%} | "
                    f"IC={r.ic:+.4f} | BullRet={r.mean_return_when_bullish:+.4f} | "
                    f"BearRet={r.mean_return_when_bearish:+.4f} | N={r.n_samples}"
                )

    # Test 2: News keyword sentiment
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: News Keyword Sentiment")
    logger.info("=" * 60)
    news_results = validate_news_sentiment(symbols)
    output["news_sentiment"] = news_results
    for sym, r in news_results.items():
        if r.get("next_day_return") is not None:
            logger.info(
                f"  {sym:5s} | Articles={r['n_articles']} | "
                f"Sent={r['keyword_sentiment']:+.2f} | "
                f"NextRet={r['next_day_return']:+.4f} | "
                f"Correct={r['sentiment_correct']}"
            )

    # Test 3: Technical baseline
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Technical Rule Baseline (SPY)")
    logger.info("=" * 60)
    tech_results = validate_technical_baseline("SPY")
    output["technical_baseline"]["SPY"] = tech_results
    for rule, r in tech_results.items():
        logger.info(
            f"  {rule:20s} | HitRate={r['hit_rate']:.2%} | "
            f"Signals={r['n_signals']} | {r.get('description', '')}"
        )

    # Summary
    llm_hit_rates = []
    llm_ics = []
    for sym_results in output["llm_signal_validation"].values():
        for r in sym_results:
            llm_hit_rates.append(r["hit_rate"])
            llm_ics.append(r["ic"])

    tech_hit_rates = []
    for rule, r in output["technical_baseline"].get("SPY", {}).items():
        if rule != "buy_and_hold":
            tech_hit_rates.append(r["hit_rate"])

    output["summary"] = {
        "llm_avg_hit_rate": round(float(np.mean(llm_hit_rates)), 4) if llm_hit_rates else None,
        "llm_avg_ic": round(float(np.mean(llm_ics)), 4) if llm_ics else None,
        "llm_max_ic": round(float(np.max(llm_ics)), 4) if llm_ics else None,
        "tech_avg_hit_rate": round(float(np.mean(tech_hit_rates)), 4) if tech_hit_rates else None,
        "buy_hold_hit_rate": output["technical_baseline"].get("SPY", {}).get("buy_and_hold", {}).get("hit_rate"),
        "n_symbols": len(symbols),
        "n_llm_samples": int(sum(r["n_samples"] for sym_results in output["llm_signal_validation"].values() for r in sym_results)),
    }

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    s = output["summary"]
    logger.info(f"  LLM Avg Hit Rate: {s['llm_avg_hit_rate']:.2%}" if s['llm_avg_hit_rate'] is not None else "  LLM: no data")
    logger.info(f"  LLM Avg IC:       {s['llm_avg_ic']:+.4f}" if s['llm_avg_ic'] is not None else "")
    logger.info(f"  Tech Avg Hit Rate:{s['tech_avg_hit_rate']:.2%}" if s['tech_avg_hit_rate'] is not None else "")
    logger.info(f"  Buy&Hold Up Days: {s['buy_hold_hit_rate']:.2%}" if s['buy_hold_hit_rate'] is not None else "")

    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_full_validation()
