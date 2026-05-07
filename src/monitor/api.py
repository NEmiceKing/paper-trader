"""
FastAPI server for paper-trader v2 — production-ready REST API.

Provides:
  GET  /api/status         — system status, IBKR connection, data freshness
  GET  /api/portfolio      — current simulated portfolio
  GET  /api/signals        — latest LLM + RL signals
  GET  /api/reflection     — analyst performance report
  GET  /api/trades         — recent trades
  POST /api/refresh        — trigger data refresh
  POST /api/strategy       — parse natural language strategy
"""

from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Paper Trader v2 API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _get_config():
    from src.config.loader import load_config
    return load_config()


def _get_db(db_path: str = "logs/trades.db"):
    import sqlite3
    if not Path(db_path).exists():
        return None
    return sqlite3.connect(db_path)


# ── Status ────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    config = _get_config()
    from src.data.pipeline import DataPipeline
    pipeline = DataPipeline(config=config)
    freshness = pipeline.get_freshness(config.symbols)

    ibkr_connected = False
    try:
        from src.ibkr.client import IBKRClient
        c = IBKRClient(config.ibkr.host, config.ibkr.port, 99)
        c.connect()
        ibkr_connected = c.is_connected
        c.disconnect()
    except Exception:
        pass

    return {
        "status": "running",
        "ibkr_connected": ibkr_connected,
        "symbols": config.symbols,
        "data_freshness": freshness.to_dict(orient="records") if not freshness.empty else [],
        "llm_enabled": config.llm.enabled,
        "debate_enabled": config.debate.enabled,
        "paper_capital": config.paper_trading_capital,
    }


# ── Portfolio ─────────────────────────────────────────────────────

@app.get("/api/portfolio")
def api_portfolio():
    db = _get_db()
    if db is None:
        return {"equity": 0, "cash": 0, "exposure": 0, "snapshots": []}

    import pandas as pd
    snaps = pd.read_sql("SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 50", db)
    db.close()

    if snaps.empty:
        return {"equity": 0, "cash": 0, "exposure": 0, "snapshots": []}

    latest = snaps.iloc[0]
    return {
        "equity": float(latest["total_equity"]),
        "cash": float(latest["cash"]),
        "exposure": float(latest["exposure"]),
        "snapshots": snaps[["timestamp", "total_equity", "cash", "exposure"]].to_dict(orient="records"),
    }


# ── Signals ───────────────────────────────────────────────────────

@app.get("/api/signals/{symbol}")
def api_signals(symbol: str):
    db = _get_db()
    if db is None:
        return {"symbol": symbol, "llm_signals": [], "fused_signals": []}

    import pandas as pd
    llm = pd.read_sql(
        f"SELECT * FROM llm_signals WHERE symbol='{symbol}' ORDER BY timestamp DESC LIMIT 10", db
    )
    fused = pd.read_sql(
        f"SELECT * FROM fused_signals WHERE symbol='{symbol}' ORDER BY timestamp DESC LIMIT 5", db
    )
    db.close()

    return {
        "symbol": symbol,
        "llm_signals": llm.to_dict(orient="records") if not llm.empty else [],
        "fused_signals": fused.to_dict(orient="records") if not fused.empty else [],
    }


# ── Reflection ────────────────────────────────────────────────────

@app.get("/api/reflection")
def api_reflection():
    try:
        from src.agent.reflection import ReflectionTracker
        tracker = ReflectionTracker()
        return {
            "report": tracker.get_report(),
            "weights": tracker.get_all_weights(),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Trades ────────────────────────────────────────────────────────

@app.get("/api/trades")
def api_trades(limit: int = 20):
    db = _get_db()
    if db is None:
        return {"trades": []}

    import pandas as pd
    trades = pd.read_sql(f"SELECT * FROM trades ORDER BY timestamp DESC LIMIT {limit}", db)
    db.close()
    return {"trades": trades.to_dict(orient="records") if not trades.empty else []}


# ── Actions ───────────────────────────────────────────────────────

@app.post("/api/refresh")
def api_refresh():
    config = _get_config()
    from src.data.pipeline import DataPipeline
    pipeline = DataPipeline(config=config)
    results = pipeline.refresh_data(config.symbols)
    total = sum(r.get("new_bars", 0) if isinstance(r.get("new_bars"), int) else 0 for r in results.values())
    return {"status": "ok", "new_bars": total, "details": {k: v for k, v in results.items()}}


@app.post("/api/strategy")
def api_parse_strategy(text: str):
    from src.nlp.strategy_parser import parse_strategy, strategy_to_config
    intent = parse_strategy(text)
    overrides = strategy_to_config(intent)
    return {
        "intent": {
            "risk_profile": intent.risk_profile,
            "focus_style": intent.focus_style,
            "symbols": intent.symbols,
            "capital": intent.capital,
        },
        "config_overrides": overrides,
    }


# ── Run ───────────────────────────────────────────────────────────

def run_api(host: str = "0.0.0.0", port: int = 8090):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_api()
