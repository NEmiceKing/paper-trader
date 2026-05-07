from pathlib import Path

import pandas as pd


class MarketDataStore:
    def __init__(self, base_dir: str = "data"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def _sym_dir(self, symbol: str) -> Path:
        p = self.base / symbol
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write_bars(self, symbol: str, df: pd.DataFrame):
        if df.empty:
            return
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("date")
        path = self._sym_dir(symbol) / "daily.parquet"
        existing = pd.read_parquet(path) if path.exists() else None
        if existing is not None:
            combined = pd.concat([existing, df[~df.index.isin(existing.index)]])
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = df
        combined.sort_index(inplace=True)
        combined.to_parquet(path)

    def read_bars(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        path = self._sym_dir(symbol) / "daily.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No data for {symbol}")
        df = pd.read_parquet(path)
        if start:
            ts = pd.Timestamp(start)
            ts = ts.tz_localize(df.index.tz) if df.index.tz else ts
            df = df[df.index >= ts]
        if end:
            ts = pd.Timestamp(end)
            ts = ts.tz_localize(df.index.tz) if df.index.tz else ts
            df = df[df.index <= ts]
        return df

    def list_symbols(self) -> list[str]:
        return [d.name for d in self.base.iterdir() if d.is_dir() and (d / "daily.parquet").exists()]

    def last_date(self, symbol: str) -> str | None:
        """Return the most recent date (YYYY-MM-DD format) for a symbol's stored data."""
        path = self._sym_dir(symbol) / "daily.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if df.empty:
            return None
        latest = pd.Timestamp(df.index.max())
        return latest.strftime("%Y-%m-%d")

    def data_freshness(self, symbol: str) -> dict:
        """Return freshness info: last date, bar count, age in days."""
        path = self._sym_dir(symbol) / "daily.parquet"
        if not path.exists():
            return {"exists": False, "last_date": None, "bars": 0, "age_days": None}
        df = pd.read_parquet(path)
        if df.empty:
            return {"exists": True, "last_date": None, "bars": 0, "age_days": None}
        latest = df.index.max()
        latest_ts = pd.Timestamp(latest)
        age = (pd.Timestamp.now(tz=latest_ts.tz) - latest_ts).days if latest_ts.tz else (pd.Timestamp.now() - latest_ts).days
        return {
            "exists": True,
            "last_date": str(latest_ts.date()),
            "bars": len(df),
            "age_days": age,
        }
