import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class TradeLogger:
    def __init__(self, db_path: str = "logs/trades.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    order_id INTEGER,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    pnl REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    risk_decision TEXT DEFAULT 'APPROVED'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TEXT PRIMARY KEY,
                    total_equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    exposure REAL NOT NULL,
                    daily_pnl REAL DEFAULT 0.0,
                    drawdown REAL DEFAULT 0.0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    analyst TEXT NOT NULL,
                    signal INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    summary TEXT,
                    debate_round INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fused_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    rl_direction INTEGER,
                    rl_size REAL,
                    llm_composite REAL,
                    debate_composite REAL,
                    final_direction INTEGER,
                    final_size REAL,
                    final_confidence REAL
                )
            """)

    def log_trade(self, symbol: str, action: str, quantity: float, price: float,
                  order_id: int | None = None, status: str = "PENDING",
                  pnl: float = 0.0, confidence: float = 0.0, risk_decision: str = "APPROVED"):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO trades (timestamp, symbol, action, quantity, price, order_id, status, pnl, confidence, risk_decision) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), symbol, action, quantity, price, order_id, status, pnl, confidence, risk_decision),
            )

    def log_snapshot(self, equity: float, cash: float, exposure: float, daily_pnl: float = 0.0, drawdown: float = 0.0):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio_snapshots (timestamp, total_equity, cash, exposure, daily_pnl, drawdown) VALUES (?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), equity, cash, exposure, daily_pnl, drawdown),
            )

    def log_llm_signal(self, symbol: str, analyst: str, signal: int, confidence: float,
                       summary: str = "", debate_round: int = 0):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO llm_signals (timestamp, symbol, analyst, signal, confidence, summary, debate_round) VALUES (?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), symbol, analyst, signal, confidence, summary, debate_round),
            )

    def log_fused_signal(self, symbol: str, rl_direction: int, rl_size: float,
                         llm_composite: float, debate_composite: float,
                         final_direction: int, final_size: float, final_confidence: float):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO fused_signals (timestamp, symbol, rl_direction, rl_size, llm_composite, debate_composite, final_direction, final_size, final_confidence) VALUES (?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), symbol, rl_direction, rl_size, llm_composite, debate_composite, final_direction, final_size, final_confidence),
            )
