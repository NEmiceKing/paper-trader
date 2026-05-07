import numpy as np
import pandas as pd

from src.data.indicators import compute_features
from src.data.providers import DataProvider, MultiSourceProvider, YFinanceProvider
from src.data.store import MarketDataStore


def _build_provider(config=None) -> DataProvider:
    """Build a provider chain from config priority list."""
    if config is None or not hasattr(config, "data"):
        return YFinanceProvider()

    priority = getattr(config.data, "provider_priority", ["yfinance"])
    providers: list[DataProvider] = []

    for name in priority:
        name = name.strip().lower()
        if name == "yfinance":
            providers.append(YFinanceProvider())
        elif name == "openbb":
            try:
                from src.data.providers import OpenBBProvider
                providers.append(OpenBBProvider(api_key=getattr(config.data, "openbb_api_key", None)))
            except ImportError:
                pass
        elif name == "ibkr":
            try:
                from src.ibkr.client import IBKRClient
                from src.data.providers import IBKRProvider
                client = IBKRClient(
                    host=config.ibkr.host, port=config.ibkr.port, client_id=config.ibkr.client_id
                )
                client.connect()
                providers.append(IBKRProvider(client.ib))
            except (ImportError, Exception):
                pass

    if not providers:
        providers.append(YFinanceProvider())

    return MultiSourceProvider(providers) if len(providers) > 1 else providers[0]


class DataPipeline:
    def __init__(self, store: MarketDataStore | None = None, provider: DataProvider | None = None, config=None):
        self.store = store or MarketDataStore()
        self.provider = provider or _build_provider(config)

    def download_and_store(self, symbols: list[str], start: str = "2020-01-01", end: str | None = None):
        data = self.provider.download(symbols, start, end)
        for sym, df in data.items():
            self.store.write_bars(sym, df)

    def refresh_data(self, symbols: list[str]) -> dict[str, dict]:
        """Incrementally refresh data — only fetches new bars since last stored date.

        Returns a dict of {symbol: {"before": last_date, "after": new_last_date, "new_bars": N}}
        """
        results = {}
        for sym in symbols:
            last_date = self.store.last_date(sym)
            if last_date is None:
                # No existing data — full download
                self.download_and_store([sym])
                new_date = self.store.last_date(sym)
                results[sym] = {"before": None, "after": new_date, "new_bars": "full"}
            else:
                # Incremental: download only from day after last stored date
                next_day_ts = pd.Timestamp(last_date) + pd.Timedelta(days=1)
                today = pd.Timestamp.now().normalize()
                if next_day_ts > today:
                    # Last data is already up to date (includes today or future)
                    results[sym] = {"before": last_date, "after": last_date, "new_bars": 0}
                    continue
                next_day = next_day_ts.strftime("%Y-%m-%d")
                data = self.provider.download([sym], start=next_day)
                if sym in data and not data[sym].empty:
                    df = data[sym]
                    if len(df) > 0:
                        self.store.write_bars(sym, df)
                        new_bars = len(df)
                    else:
                        new_bars = 0
                else:
                    new_bars = 0
                new_date = self.store.last_date(sym)
                results[sym] = {"before": last_date, "after": new_date, "new_bars": new_bars}
        return results

    def get_freshness(self, symbols: list[str]) -> pd.DataFrame:
        """Return a DataFrame showing data freshness for all symbols."""
        rows = []
        for sym in symbols:
            info = self.store.data_freshness(sym)
            info["symbol"] = sym
            rows.append(info)
        return pd.DataFrame(rows)

    def prepare_training_data(
        self, symbols: list[str], start: str = "2020-01-01", end: str | None = None
    ) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for sym in symbols:
            df = self.store.read_bars(sym, start, end)
            features = compute_features(df)
            result[sym] = features.values.astype(np.float32)
        return result

    def get_latest_observation(self, symbol: str, window: int = 20) -> np.ndarray:
        df = self.store.read_bars(symbol)
        features = compute_features(df)
        return features.values[-window:].astype(np.float32)
