"""Multi-source news aggregator — yfinance + Google News RSS + Finnhub (all free)."""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    title: str
    summary: str
    source: str
    published: str
    url: str = ""
    related_symbols: list[str] = field(default_factory=list)


class NewsFetcher:
    """Aggregate news from multiple free sources with caching and dedup."""

    def __init__(self, cache_ttl_minutes: int = 15):
        self.cache_ttl = cache_ttl_minutes
        self._cache: dict[str, tuple[datetime, list[NewsArticle]]] = {}

    # ── Public API ─────────────────────────────────────────────────

    def fetch(self, symbol: str, max_articles: int = 15) -> list[NewsArticle]:
        """Fetch news from all sources, merge, dedup, and cache."""
        now = datetime.now(timezone.utc)

        if symbol in self._cache:
            cached_time, articles = self._cache[symbol]
            if (now - cached_time).total_seconds() < self.cache_ttl * 60:
                return articles

        articles: list[NewsArticle] = []

        # Source 1: yfinance (structured, symbol-specific) — always works
        articles.extend(self._fetch_yfinance(symbol, max_articles))

        # Source 2: Google News RSS (free, broad coverage) — no API key
        articles.extend(self._fetch_google_news(symbol, max_articles // 2))

        # Source 3: Finnhub (free tier, company news) — optional API key
        articles.extend(self._fetch_finnhub(symbol, max_articles // 2))

        # Deduplicate by title similarity
        articles = self._dedup(articles)

        self._cache[symbol] = (now, articles)
        return articles

    def fetch_all(self, symbols: list[str], max_per_symbol: int = 5) -> dict[str, list[NewsArticle]]:
        result = {}
        for sym in symbols:
            articles = self.fetch(sym, max_articles=max_per_symbol)
            if articles:
                result[sym] = articles
        return result

    def format_for_llm(self, articles: list[NewsArticle], max_chars: int = 2000) -> str:
        if not articles:
            return "No recent news available."

        lines = []
        total = 0
        for a in articles:
            line = f"- [{a.published[:19]}] [{a.source}] {a.title}"
            if a.summary:
                summary_short = re.sub(r"<[^>]+>", "", a.summary)[:200]
                line += f"\n  {summary_short}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)

    # ── Sources ────────────────────────────────────────────────────

    def _fetch_yfinance(self, symbol: str, max_articles: int) -> list[NewsArticle]:
        articles = []
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            raw = ticker.news
            for item in raw[:max_articles]:
                content = item.get("content", item)
                title = content.get("title", "") or ""
                summary = content.get("summary", "") or content.get("description", "") or ""
                pub_date = content.get("pubDate", "") or content.get("providerPublishTime", "")
                source = "Yahoo Finance"
                if isinstance(content.get("provider"), dict):
                    source = content["provider"].get("displayName", "Yahoo Finance")
                url = ""
                if isinstance(content.get("canonicalUrl"), dict):
                    url = content["canonicalUrl"].get("url", "")

                if title:
                    articles.append(NewsArticle(
                        title=str(title), summary=str(summary)[:500],
                        source=str(source), published=str(pub_date), url=str(url),
                    ))
        except Exception as e:
            logger.debug(f"yfinance news failed for {symbol}: {e}")
        return articles

    def _fetch_google_news(self, symbol: str, max_articles: int) -> list[NewsArticle]:
        """Scrape Google News RSS for stock-related headlines (free, no API key)."""
        articles = []
        try:
            import xml.etree.ElementTree as ET
            import requests

            # Search for symbol + "stock" to get relevant news
            query = quote(f"{symbol} stock")
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&ceid=US:en"

            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                return articles

            root = ET.fromstring(resp.text)
            count = 0
            for item in root.iter("item"):
                if count >= max_articles:
                    break
                title = ""
                pub_date = ""
                link = ""
                source = "Google News"
                for child in item:
                    if child.tag == "title":
                        title = child.text or ""
                        # Remove " - SourceName" suffix
                        if " - " in title:
                            parts = title.rsplit(" - ", 1)
                            title = parts[0]
                            source = parts[1]
                    elif child.tag == "pubDate":
                        pub_date = child.text or ""
                    elif child.tag == "link":
                        link = child.text or ""
                    elif child.tag == "source":
                        source = child.text or source

                if title:
                    articles.append(NewsArticle(
                        title=title, summary="", source=source,
                        published=pub_date, url=link,
                    ))
                    count += 1
        except Exception as e:
            logger.debug(f"Google News failed for {symbol}: {e}")
        return articles

    def _fetch_finnhub(self, symbol: str, max_articles: int) -> list[NewsArticle]:
        """Fetch from Finnhub free tier (needs FINNHUB_API_KEY env var)."""
        import os
        api_key = os.environ.get("FINNHUB_API_KEY", "")
        if not api_key:
            return []

        articles = []
        try:
            import requests
            from datetime import datetime as dt, timedelta

            to_date = dt.now().strftime("%Y-%m-%d")
            from_date = (dt.now() - timedelta(days=2)).strftime("%Y-%m-%d")

            resp = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": symbol, "from": from_date, "to": to_date, "token": api_key},
                timeout=10,
            )
            if resp.status_code != 200:
                return articles

            for item in resp.json()[:max_articles]:
                title = item.get("headline", "")
                summary = item.get("summary", "")
                source = item.get("source", "Finnhub")
                published = dt.fromtimestamp(item.get("datetime", 0)).isoformat()
                url = item.get("url", "")

                if title:
                    articles.append(NewsArticle(
                        title=str(title), summary=str(summary)[:500],
                        source=str(source), published=published, url=str(url),
                        related_symbols=item.get("related", "").split(",") if item.get("related") else [],
                    ))
        except Exception as e:
            logger.debug(f"Finnhub news failed for {symbol}: {e}")
        return articles

    # ── Dedup ──────────────────────────────────────────────────────

    def _dedup(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """Remove duplicate articles by title similarity."""
        seen = set()
        unique = []
        for a in articles:
            # Normalize title for comparison
            key = re.sub(r"[^a-z0-9]", "", a.title.lower())[:80]
            if key and key not in seen:
                seen.add(key)
                unique.append(a)
        # Sort by most recent first
        unique.sort(key=lambda a: a.published, reverse=True)
        return unique
