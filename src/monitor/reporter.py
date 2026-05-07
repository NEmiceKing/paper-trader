import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def generate_report(db_path: str = "logs/trades.db") -> dict:
    if not Path(db_path).exists():
        return {"error": "No trade database found"}

    conn = sqlite3.connect(db_path)

    trades = pd.read_sql("SELECT * FROM trades ORDER BY timestamp", conn)
    snapshots = pd.read_sql("SELECT * FROM portfolio_snapshots ORDER BY timestamp", conn)
    conn.close()

    if snapshots.empty:
        return {"error": "No portfolio data"}

    snapshots["timestamp"] = pd.to_datetime(snapshots["timestamp"])
    snapshots = snapshots.set_index("timestamp")

    equity = snapshots["total_equity"]
    returns = equity.pct_change().dropna()

    sharpe = (returns.mean() * 252 - 0.05) / (returns.std() * 252**0.5 + 1e-12)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 1 else 0
    peak = equity.cummax()
    max_dd = (equity - peak) / peak

    filled_trades = trades[trades["status"] == "FILLED"] if not trades.empty else trades
    win_rate = (filled_trades["pnl"] > 0).mean() if not filled_trades.empty else 0

    return {
        "total_return": round(float(total_return), 4),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd.min()), 4),
        "total_trades": len(filled_trades),
        "win_rate": round(float(win_rate), 4),
        "current_equity": round(float(equity.iloc[-1]), 2) if len(equity) > 0 else 0,
    }
