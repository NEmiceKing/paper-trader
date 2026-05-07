from datetime import datetime
from pathlib import Path

import pandas as pd


class LLMAnalysisStore:
    """Persist and retrieve LLM analysis results using Parquet."""

    def __init__(self, base_dir: str = "data/llm_analysis"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def _sym_dir(self, symbol: str) -> Path:
        p = self.base / symbol
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_reports(self, symbol: str, reports: list) -> None:
        """Save analyst reports to Parquet. Reports must be a list of AnalystReport dataclasses."""
        if not reports:
            return
        records = []
        for r in reports:
            records.append({
                "timestamp": r.timestamp.isoformat() if isinstance(r.timestamp, datetime) else str(r.timestamp),
                "symbol": symbol,
                "analyst_name": r.analyst_name,
                "signal": r.signal,
                "confidence": r.confidence,
                "summary": r.summary,
            })
        df = pd.DataFrame(records)
        path = self._sym_dir(symbol) / "analyst_reports.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp", "analyst_name"], keep="last")
        else:
            combined = df
        combined.to_parquet(path)

    def load_reports(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        path = self._sym_dir(symbol) / "analyst_reports.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if start:
                df = df[df["timestamp"] >= pd.Timestamp(start)]
            if end:
                df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df

    def load_latest_report(self, symbol: str, analyst_name: str | None = None) -> dict | None:
        df = self.load_reports(symbol)
        if df.empty:
            return None
        if analyst_name:
            df = df[df["analyst_name"] == analyst_name]
        if df.empty:
            return None
        return df.sort_values("timestamp", ascending=False).iloc[0].to_dict()

    def save_debate_result(self, symbol: str, result) -> None:
        """Save debate result to Parquet."""
        records = [{
            "timestamp": result.timestamp.isoformat() if isinstance(result.timestamp, datetime) else str(result.timestamp),
            "symbol": symbol,
            "composite_signal": result.composite_signal,
            "confidence": result.confidence,
            "total_rounds": len(result.rounds),
            "bull_summary": result.bull_summary,
            "bear_summary": result.bear_summary,
        }]
        df = pd.DataFrame(records)
        path = self._sym_dir(symbol) / "debate_results.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df
        combined.to_parquet(path)

    def load_latest_debate(self, symbol: str) -> dict | None:
        path = self._sym_dir(symbol) / "debate_results.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return df.sort_values("timestamp", ascending=False).iloc[0].to_dict()
