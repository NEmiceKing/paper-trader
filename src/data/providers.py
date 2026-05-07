from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    @abstractmethod
    def download(self, symbols: list[str], start: str, end: str | None) -> dict[str, pd.DataFrame]:
        ...


class YFinanceProvider(DataProvider):
    def download(self, symbols: list[str], start: str, end: str | None = None) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            ticker = yf.Ticker(sym)
            df = ticker.history(start=start, end=end, auto_adjust=True)
            if df.empty:
                continue
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            result[sym] = df
        return result


class IBKRProvider(DataProvider):
    def __init__(self, ib_client):
        self._ib = ib_client

    def download(self, symbols: list[str], start: str, end: str | None = None) -> dict[str, pd.DataFrame]:
        from ib_insync import Stock

        result: dict[str, pd.DataFrame] = {}
        duration = "5 Y"
        bar_size = "1 day"

        for sym in symbols:
            contract = Stock(sym, "SMART", "USD")
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime=end or "",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="ADJUSTED_LAST",
                useRTH=True,
                formatDate=1,
            )
            if not bars:
                continue
            records = [
                {"date": b.date, "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}
                for b in bars
            ]
            df = pd.DataFrame(records).set_index("date")
            df.index = pd.to_datetime(df.index)
            result[sym] = df
        return result


class OpenBBProvider(DataProvider):
    """Data provider using OpenBB's unified financial data API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self._obb = None

    def _ensure_obb(self):
        if self._obb is None:
            try:
                from openbb import obb
                self._obb = obb
            except ImportError:
                raise ImportError(
                    "OpenBB is not installed. Run: pip install openbb"
                )

    def download(self, symbols: list[str], start: str, end: str | None = None) -> dict[str, pd.DataFrame]:
        self._ensure_obb()

        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                kwargs = {"symbol": sym, "provider": "yfinance"}
                if start:
                    kwargs["start_date"] = start
                if end:
                    kwargs["end_date"] = end
                data = self._obb.equity.price.historical(**kwargs)
                df = data.to_dataframe()
                if df is not None and not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    if "date" not in [c.lower() for c in df.columns] and df.index.name != "date":
                        df.index.name = "date"
                    result[sym] = df
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"OpenBB download failed for {sym}: {e}")
        return result

    def get_fundamentals(self, symbol: str) -> dict:
        """Fetch fundamental data for a symbol."""
        self._ensure_obb()
        try:
            income = self._obb.equity.fundamental.income(symbol, provider="yfinance")
            balance = self._obb.equity.fundamental.balance(symbol, provider="yfinance")
            return {
                "income": income.to_dataframe() if income else None,
                "balance": balance.to_dataframe() if balance else None,
            }
        except Exception:
            return {}

    def get_options_data(self, symbol: str) -> pd.DataFrame | None:
        """Fetch options chain data."""
        self._ensure_obb()
        try:
            chains = self._obb.derivatives.options.chains(symbol, provider="yfinance")
            return chains.to_dataframe() if chains else None
        except Exception:
            return None


class MultiSourceProvider(DataProvider):
    """Wrapper that tries providers in priority order, falling back on failure."""

    def __init__(self, providers: list[DataProvider]):
        self.providers = providers

    def download(self, symbols: list[str], start: str, end: str | None = None) -> dict[str, pd.DataFrame]:
        for provider in self.providers:
            try:
                result = provider.download(symbols, start, end)
                if result:
                    return result
            except Exception:
                continue
        raise RuntimeError(f"All providers ({len(self.providers)}) failed to download data")
