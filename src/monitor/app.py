import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on the Python path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def run_dashboard(config=None):
    st.set_page_config(page_title="Paper Trader Control Panel", layout="wide")
    st.title("Paper Trader Control Panel")

    # ═══ Auto-refresh control ═══
    if "auto_refresh_interval" not in st.session_state:
        st.session_state.auto_refresh_interval = 0
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()

    with st.sidebar:
        st.caption("Language / 语言")
        lang = st.selectbox(
            "Language",
            ["English", "中文"],
            index=0,
            key="lang_select",
        )
        is_cn = (lang == "中文")
        st.session_state.is_cn = is_cn

        st.divider()

        st.caption("Auto-Refresh / 自动刷新")
        interval = st.selectbox(
            "Interval / 间隔",
            ["Off", "1s", "5s", "10s", "30s", "1min", "5min"],
            index=0,
            key="refresh_interval_select",
        )
        interval_map = {"Off": 0, "1s": 1, "5s": 5, "10s": 10, "30s": 30, "1min": 60, "5min": 300}
        st.session_state.auto_refresh_interval = interval_map[interval]
        if st.session_state.auto_refresh_interval > 0:
            remaining = st.session_state.auto_refresh_interval - (time.time() - st.session_state.last_refresh)
            label = f"Next in {max(0, int(remaining))}s" if not st.session_state.get("is_cn", False) else f"{max(0, int(remaining))}秒后刷新"
            st.caption(label)

    # Resolve config
    if config is None:
        try:
            from src.config.loader import load_config
            # Resolve config path relative to this file's location
            config_path = str(_project_root / "config" / "settings.yaml")
            config = load_config(config_path)
        except Exception as e:
            st.warning(f"Could not load config: {e}")
            config = None

    # Initialize session state
    defaults = {
        "trading_active": False,
        "training_active": False,
        "backtest_cache": {},
        "config_overrides": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    db_path = config.logging.trades_db if config else "logs/trades.db"

    # ═══ Data Freshness Bar ═══
    if config is not None:
        try:
            from src.data.pipeline import DataPipeline
            pipeline = DataPipeline(config=config)
            freshness = pipeline.get_freshness(config.symbols)
            if not freshness.empty:
                oldest = freshness["age_days"].max()
                freshest = freshness["age_days"].min()
                stale_count = int((freshness["age_days"] > 1).sum())

                col_fresh, col_btn = st.columns([4, 1])
                with col_fresh:
                    last_date = freshness["last_date"].dropna().iloc[0] if not freshness["last_date"].dropna().empty else "N/A"
                    if oldest <= 1:
                        st.success(f"EOD data through {last_date} | {len(config.symbols)} symbols ready | IBKR 1-min poll (free)")
                    elif stale_count > 0:
                        st.warning(
                            f"EOD data: {stale_count}/{len(config.symbols)} symbols need refresh "
                            f"(last: {last_date}). Click Refresh."
                        )
                    else:
                        st.info(f"EOD data through {last_date}")
                with col_btn:
                    if st.button("Refresh Data", type="primary", use_container_width=True):
                        with st.spinner("Downloading latest data..."):
                            results = pipeline.refresh_data(config.symbols)
                            total_new = sum(
                                r["new_bars"] if isinstance(r["new_bars"], int) else 0 for r in results.values()
                            )
                            st.success(f"Added {total_new} new bars. Refreshing page...")
                            time.sleep(1)
                            st.rerun()
        except Exception as e:
            st.caption(f"Data freshness check unavailable: {e}")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Training", "Paper Trading", "Backtest", "Signals", "Risk", "Logs", "Manual / 手册"
    ])

    with tab1:
        _render_training_tab(config)
    with tab2:
        _render_paper_trading_tab(config, db_path)
    with tab3:
        _render_backtest_tab(config)
    with tab4:
        _render_signals_tab(config, db_path)
    with tab5:
        _render_risk_tab(config)
    with tab6:
        _render_logs_tab()
    with tab7:
        _render_manual_tab()


def _render_training_tab(config):
    st.subheader("Training Configuration")

    if config is None:
        st.info("Load config to enable training controls.")
        return

    agent_cfg = config.agent
    train_cfg = agent_cfg.training

    with st.form("training_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            total_timesteps = st.number_input("Total Timesteps", 10000, 2000000, train_cfg.total_timesteps, 50000)
            learning_rate = st.number_input("Learning Rate", 1e-5, 1e-2, train_cfg.learning_rate, format="%.5f")
            n_steps = st.number_input("N Steps", 128, 8192, train_cfg.n_steps, 256)
        with col2:
            batch_size = st.number_input("Batch Size", 16, 512, train_cfg.batch_size, 16)
            n_epochs = st.number_input("N Epochs", 1, 50, train_cfg.n_epochs)
            gamma = st.slider("Gamma", 0.8, 1.0, train_cfg.gamma, 0.01)
        with col3:
            ent_coef = st.slider("Entropy Coef", 0.0, 0.1, train_cfg.ent_coef, 0.005)
            reward_fn = st.selectbox("Reward Function", ["conservative", "simple"], index=0)
            net_arch_str = st.text_input("Net Arch (comma-sep)", ",".join(str(x) for x in agent_cfg.net_arch))

        submitted = st.form_submit_button("Start Training")
        if submitted:
            st.session_state.training_active = True
            st.session_state.training_config = {
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "gamma": gamma,
                "ent_coef": ent_coef,
            }
            st.success("Training started. Check TensorBoard for progress.")
            st.info("Run `make train` in the terminal to execute training with these parameters.")

    # TensorBoard file display
    tb_dir = Path("models/tensorboard")
    if tb_dir.exists():
        st.subheader("Training Runs")
        runs = list(tb_dir.glob("*"))
        if runs:
            run_info = []
            for r in runs:
                events = list(r.glob("events.out.*"))
                if events:
                    mtime = datetime.fromtimestamp(events[0].stat().st_mtime)
                    run_info.append({"run": r.name, "last_modified": mtime})
            if run_info:
                st.dataframe(pd.DataFrame(run_info), use_container_width=True, height=150)
        else:
            st.info("No training runs found. Run `make train` to start.")


def _render_paper_trading_tab(config, db_path):
    _cn = st.session_state.get("is_cn", False)
    T = lambda en, cn: cn if _cn else en

    st.subheader(T("Live Paper Trading", "模拟交易"))

    if config is None:
        st.info(T("Load config to enable trading controls.", "加载配置以启用交易控制。"))
        return

    # ── Market hours auto-start check ──
    from datetime import timezone as _tz
    now_et = datetime.now(_tz.utc).astimezone().strftime("%H:%M")
    market_msg = ""
    try:
        # US market: 9:30 AM - 4:00 PM ET (13:30 - 20:00 UTC)
        import subprocess
        paper_proc = subprocess.run(["pgrep", "-f", "src.main paper"], capture_output=True, text=True)
        paper_is_running = bool(paper_proc.stdout.strip())

        now_utc = datetime.now(_tz.utc)
        hour_et = (now_utc.hour - 4) % 24  # rough ET conversion
        market_open = 9 <= hour_et < 16 and now_utc.weekday() < 5
        if market_open and not paper_is_running:
            market_msg = T("Market OPEN — auto-start available", "美股开盘中 — 可自动启动")
        elif not market_open:
            market_msg = T("Market CLOSED — prices static until next open", "美股休市中 — 开盘前价格不变")
    except Exception:
        paper_is_running = False

    # ── Controls ──
    col1, col2, col3 = st.columns(3)
    with col1:
        symbols = st.multiselect(T("Symbols", "交易标的"), config.symbols, default=config.symbols[:3])
    with col2:
        capital = st.number_input(T("Capital ($)", "本金 ($)"), 100, 1000000, int(config.paper_trading_capital), 100)
    with col3:
        interval = st.selectbox(T("Poll interval", "轮询间隔"), ["30s", "60s", "5min", "15min"], index=1,
                               help=T("How often to check for new IBKR bars. IBKR 1-min bars are always fetched on each poll.",
                                      "检查IBKR新bar的频率。每次轮询都获取最新1分钟bar。"))

    if market_msg:
        st.caption(market_msg)

    btn_start = st.button(T("Start Paper Trading", "启动模拟交易"), type="primary", disabled=paper_is_running,
                          use_container_width=True)
    btn_stop = st.button(T("Stop Paper Trading", "停止模拟交易"), disabled=not paper_is_running,
                         use_container_width=False)

    if btn_start:
        import os, subprocess
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        env_cmd = f"DEEPSEEK_API_KEY={api_key}" if api_key else ""
        interval_sec = {"30s": 30, "60s": 60, "5min": 300, "15min": 900}[interval]
        # Write interval override
        cmd = (f"cd {_project_root} && source .venv/bin/activate && "
               f"{env_cmd} TRADING_INTERVAL={interval_sec} python -m src.main paper")
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        st.session_state.trading_active = True
        st.success(T("Paper trading started!", "模拟交易已启动！"))
        st.rerun()

    if btn_stop:
        import subprocess
        subprocess.run(["pkill", "-f", "src.main paper"], capture_output=True)
        st.session_state.trading_active = False
        st.warning(T("Paper trading stopped.", "模拟交易已停止。"))
        st.rerun()

    # Status
    if paper_is_running:
        st.success(T("Status: RUNNING — IBKR 1-min bars polled every cycle", "状态：运行中 — 每轮获取IBKR最新1分钟bar"))
    else:
        st.info(T("Status: IDLE — click button above to start", "状态：空闲 — 点击上方按钮启动"))

    # ── Live Trading Entry (password-protected) ──
    with st.expander(T("Go LIVE / 实盘交易入口", "实盘交易入口"), expanded=False):
        st.warning(T(
            "WARNING: This will use REAL money from your IBKR account. Only enable after model validation.",
            "警告：将使用IBKR账户真实资金。仅在模型验证通过后启用。"
        ))
        live_pw = st.text_input(T("Enter password to enable live trading", "输入密码启用实盘交易"),
                                type="password", key="live_pw")
        if live_pw == "live2026":
            st.error(T(
                "LIVE TRADING UNLOCKED — Not yet implemented. Model must prove profitable first.",
                "实盘交易已解锁 — 尚未实现。模型必须先证明盈利能力。"
            ))
        elif live_pw and live_pw != "live2026":
            st.caption(T("Incorrect password", "密码错误"))

    # ── Live price ticker ──
    import json as _json
    live_prices = {}
    price_source = T("Parquet EOD", "Parquet 收盘数据")
    try:
        lp_file = Path("logs/live_prices.json")
        if lp_file.exists():
            with open(lp_file) as f:
                live_prices = _json.load(f)
            ts = live_prices.pop("_timestamp", "")
            if ts:
                age_sec = (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
                if age_sec < 600:
                    price_source = T(f"IBKR 1-min bar ({int(age_sec)}s ago)", f"IBKR 1分钟bar ({int(age_sec)}秒前)")
    except Exception:
        pass

    st.markdown("---")
    label = T("Prices", "实时价格")
    if not paper_is_running:
        st.caption(T(f"{label} — {price_source} | Click Start to get IBKR live prices",
                     f"{label} — {price_source} | 点击「启动模拟交易」获取IBKR实时价格"))
    else:
        st.caption(f"{label} — {price_source}")

    # Compact price grid — 10 per row, all symbols
    all_symbols = config.symbols if config else []
    per_row = 10
    for row_start in range(0, len(all_symbols), per_row):
        row_symbols = all_symbols[row_start:row_start + per_row]
        price_cols = st.columns(per_row)
        for i, sym in enumerate(row_symbols):
            try:
                if sym in live_prices and live_prices[sym] > 0:
                    close = live_prices[sym]
                else:
                    from src.data.store import MarketDataStore
                    store = MarketDataStore()
                    df = store.read_bars(sym)
                    close = float(df["close"].iloc[-1])

                try:
                    from src.data.store import MarketDataStore
                    store2 = MarketDataStore()
                    df2 = store2.read_bars(sym)
                    prev_close = float(df2["close"].iloc[-2]) if len(df2) >= 2 else close
                except Exception:
                    prev_close = close

                change_pct = (close - prev_close) / prev_close * 100 if prev_close and prev_close > 0 else 0
                with price_cols[i]:
                    color = "#4CAF50" if change_pct >= 0 else "#F44336"
                    st.markdown(
                        f"**{sym}**<br><span style='color:{color};font-size:0.95em'>${close:.2f}</span>"
                        f"<br><small style='color:{color}'>{change_pct:+.2f}%</small>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                with price_cols[i]:
                    st.caption(f"{sym}: N/A")

    # Portfolio data from SQLite
    db_exists = Path(db_path).exists()
    if db_exists:
        try:
            conn = sqlite3.connect(db_path)
            snapshots = pd.read_sql("SELECT * FROM portfolio_snapshots ORDER BY timestamp", conn)
            trades = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20", conn)
            conn.close()

            if not snapshots.empty:
                snapshots["timestamp"] = pd.to_datetime(snapshots["timestamp"])

                col1, col2, col3, col4 = st.columns(4)
                _cn2 = st.session_state.get("is_cn", False)
                with col1:
                    st.metric("净值" if _cn2 else "Equity", f"${snapshots['total_equity'].iloc[-1]:,.2f}")
                with col2:
                    st.metric("现金" if _cn2 else "Cash", f"${snapshots['cash'].iloc[-1]:,.2f}")
                with col3:
                    st.metric("仓位" if _cn2 else "Exposure", f"{snapshots['exposure'].iloc[-1]:.1%}")
                with col4:
                    dd = snapshots["drawdown"].iloc[-1] if "drawdown" in snapshots.columns else 0
                    st.metric("回撤" if _cn2 else "Drawdown", f"{dd:.2%}")

                st.subheader("Equity Curve")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=snapshots["timestamp"], y=snapshots["total_equity"],
                                         name="Equity", line=dict(color="#2196F3")))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            if not trades.empty:
                st.subheader("Recent Trades")
                st.dataframe(trades, use_container_width=True, height=250)
            else:
                st.info("No trades recorded yet.")
        except Exception as e:
            st.warning(f"Could not read trade database: {e}")
    else:
        st.info("No trade database found. Run paper trading to generate data.")


def _render_backtest_tab(config):
    st.subheader("Backtest Runner")

    if config is None:
        st.info("Load config to enable backtesting.")
        return

    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Symbol", config.symbols)
    with col2:
        strategy = st.selectbox("Strategy", ["SMA Crossover", "Buy & Hold", "RL Model"])

    with st.expander("Backtest Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            capital = st.number_input("Initial Capital", 10000, 1000000, 100000, 10000, key="bt_capital")
            commission = st.number_input("Commission %", 0.0, 1.0, 0.1, 0.01) / 100
        with col2:
            fixed_slippage = st.number_input("Fixed Slippage ($/share)", 0.0, 0.1, 0.005, 0.001)
            pct_slippage = st.number_input("Pct Slippage %", 0.0, 1.0, 0.05, 0.01) / 100
        with col3:
            impact_pct = st.number_input("Price Impact %", 0.0, 1.0, 0.1, 0.01) / 100
            fast_period = st.number_input("Fast SMA", 5, 50, 20)
            slow_period = st.number_input("Slow SMA", 20, 200, 50)

    if st.button("Run Backtest", type="primary"):
        try:
            from src.backtest.engine import VectorizedBacktest, SlippageModel
            from src.backtest.metrics import compute_metrics
            from src.data.store import MarketDataStore

            store = MarketDataStore()
            df = store.read_bars(symbol, start=config.data.default_start)

            slippage = SlippageModel(
                fixed_cost=fixed_slippage,
                pct_cost=pct_slippage,
                price_impact_pct=impact_pct,
            )
            bt = VectorizedBacktest(initial_capital=capital, commission=commission)

            if strategy == "SMA Crossover":
                result = bt.run_sma_cross(df, fast=fast_period, slow=slow_period, slippage=slippage)
            elif strategy == "Buy & Hold":
                result = bt.run_buy_and_hold(df["close"].values)
                # Add no-slippage benchmark for comparison
            else:
                st.info("RL Model backtesting not yet available in GUI. Run `make backtest` in terminal.")
                return

            metrics = compute_metrics(result.equity_curve)
            if result.trades:
                trade_stats = bt.compute_trade_statistics(result.trades)
                metrics.update({
                    "win_rate": round(trade_stats.win_rate, 4),
                    "profit_factor": round(trade_stats.profit_factor, 4),
                    "total_trades": trade_stats.total_trades,
                    "avg_win": round(trade_stats.avg_win, 4),
                    "avg_loss": round(trade_stats.avg_loss, 4),
                })

            alpha_beta = bt.compare_to_benchmark(result.equity_curve, df["close"].values)
            metrics.update(alpha_beta)

            st.session_state.backtest_cache = {
                "result": result,
                "metrics": metrics,
                "symbol": symbol,
                "strategy": strategy,
                "df": df,
            }
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            return

    cache = st.session_state.backtest_cache
    if cache and cache.get("symbol") == symbol and cache.get("strategy") == strategy:
        metrics = cache["metrics"]
        result = cache["result"]

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Sharpe", f"{metrics.get('sharpe', 0):.2f}")
        with col2:
            st.metric("Max DD", f"{metrics.get('max_drawdown', 0):.2%}")
        with col3:
            st.metric("CAGR", f"{metrics.get('cagr', 0):.2%}")
        with col4:
            st.metric("Alpha", f"{metrics.get('alpha', 0):.4f}")
        with col5:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")

        col6, col7, col8, col9, col10 = st.columns(5)
        with col6:
            st.metric("Beta", f"{metrics.get('beta', 0):.2f}")
        with col7:
            st.metric("Info Ratio", f"{metrics.get('information_ratio', 0):.2f}")
        with col8:
            st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        with col9:
            st.metric("Trades", f"{metrics.get('total_trades', 0)}")
        with col10:
            st.metric("Sortino", f"{metrics.get('sortino', 0):.2f}")

        st.subheader("Equity Curve")
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(y=result.equity_curve, name="Strategy", line=dict(color="#2196F3")))
        prices = cache["df"]["close"].values
        bench_equity = capital * prices / prices[0]
        fig_equity.add_trace(go.Scatter(y=bench_equity, name="Buy & Hold", line=dict(color="#9E9E9E", dash="dash")))
        fig_equity.update_layout(height=350, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_equity, use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Underwater Drawdown")
            peak = np.maximum.accumulate(result.equity_curve)
            dd = (np.array(result.equity_curve) - peak) / (peak + 1e-12)
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(y=dd, fill="tozeroy", name="Drawdown", line=dict(color="#F44336")))
            fig_dd.update_layout(height=250, margin=dict(l=20, r=20, t=10, b=10),
                                 yaxis_tickformat=".0%")
            st.plotly_chart(fig_dd, use_container_width=True)
        with col_right:
            if result.trades:
                st.subheader("Trade Log")
                trades_df = pd.DataFrame(result.trades)
                st.dataframe(trades_df, use_container_width=True, height=250)


def _render_signals_tab(config, db_path):
    st.subheader("Signal Analysis")

    if config is None:
        st.info("Load config to view signals.")
        return

    symbol = st.selectbox("Symbol", config.symbols, key="sig_symbol")

    # Load market data for this symbol
    from src.data.store import MarketDataStore
    from src.data.indicators import compute_features

    store = MarketDataStore()
    try:
        df = store.read_bars(symbol, start=config.data.default_start)
    except FileNotFoundError:
        st.warning(f"No market data for {symbol}. Run `make download` first.")
        return

    feat = compute_features(df)
    if feat.empty:
        st.warning("Could not compute features.")
        return

    latest_price = float(df["close"].iloc[-1])
    ret_1d = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) if len(df) >= 2 else 0
    ret_5d = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) if len(df) >= 6 else 0

    st.markdown(f"**{symbol}** — Latest Close: ${latest_price:.2f} | 1d: {ret_1d:.2%} | 5d: {ret_5d:.2%}")

    # ═══ RL Model Signal ═══
    st.markdown("### RL Model Signal")
    try:
        from stable_baselines3 import PPO
        from src.agent.env import TradingEnv

        model_path = Path("models/ppo_trader.zip")
        if model_path.exists():
            model = PPO.load(str(model_path))
            env = TradingEnv(feat.values.astype(np.float32), feat["close"].values.astype(np.float32),
                             window=config.agent.observation_bars)
            obs, _ = env.reset()
            action_arr, _ = model.predict(obs, deterministic=True)
            direction = int(np.argmax(action_arr))
            size = float(np.clip(abs(action_arr[direction]), 0.0, 1.0))
            confidence = float(max(action_arr))
            if size < 0.02:
                direction = 0
            dir_label = {0: "HOLD", 1: "BUY", 2: "SELL"}
            dir_color = {0: "#9E9E9E", 1: "#4CAF50", 2: "#F44336"}
        else:
            direction, size, confidence = 0, 0.0, 0.0
            dir_label = {0: "HOLD"}
            dir_color = {0: "#9E9E9E"}
            st.info("No trained model found. Run `make train` first.")
    except Exception as e:
        direction, size, confidence = 0, 0.0, 0.0
        dir_label = {0: "HOLD"}
        dir_color = {0: "#9E9E9E"}
        st.info(f"RL model not available: {e}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<h2 style='color:{dir_color.get(direction, '#9E9E9E')};margin:0'>{dir_label.get(direction, 'HOLD')}</h2>",
            unsafe_allow_html=True,
        )
        st.caption("Direction")
    with col2:
        st.metric("Size", f"{size:.2f}")
    with col3:
        st.metric("Confidence", f"{confidence:.2f}")

    # ═══ Technical Indicators ═══
    st.markdown("### Technical Indicators")
    indicators_row = feat.iloc[-1]
    cols = st.columns(5)
    indicators_display = [
        ("RSI(14)", "rsi_14", lambda v: f"{v:.1f}"),
        ("MACD Hist", "macd_hist", lambda v: f"{v:.4f}"),
        ("SMA20/SMA50", "sma_ratio", lambda v: f"{v:.4f}"),
        ("ATR Ratio", "atr_ratio", lambda v: f"{v:.4f}"),
        ("BB Width", "bb_width", lambda v: f"{v:.4f}"),
    ]
    for i, (label, col_name, fmt) in enumerate(indicators_display):
        with cols[i]:
            if col_name in indicators_row:
                val = indicators_row[col_name]
                if not pd.isna(val):
                    st.metric(label, fmt(val))
                else:
                    st.metric(label, "N/A")
            else:
                st.metric(label, "N/A")

    # ═══ LLM Analyst Signals ═══
    st.markdown("### LLM Analyst Signals")
    db_exists = Path(db_path).exists()
    has_llm = False
    has_fused = False
    llm_signals = pd.DataFrame()
    fused_signals = pd.DataFrame()

    if db_exists:
        try:
            conn = sqlite3.connect(db_path)
            llm_signals = pd.read_sql(
                "SELECT * FROM llm_signals WHERE symbol=? ORDER BY timestamp DESC LIMIT 10",
                conn, params=(symbol,),
            )
            fused_signals = pd.read_sql(
                "SELECT * FROM fused_signals WHERE symbol=? ORDER BY timestamp DESC LIMIT 5",
                conn, params=(symbol,),
            )
            conn.close()
            has_llm = not llm_signals.empty
            has_fused = not fused_signals.empty
        except Exception:
            pass

    if has_llm:
        for _, row in llm_signals.iterrows():
            sign = "BULLISH" if row["signal"] > 0 else ("BEARISH" if row["signal"] < 0 else "NEUTRAL")
            color = "#4CAF50" if row["signal"] > 0 else ("#F44336" if row["signal"] < 0 else "#9E9E9E")
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**{row['analyst']}** — {row['summary'][:120] if pd.notna(row['summary']) else ''}")
                with col2:
                    st.markdown(f"<span style='color:{color};font-weight:bold'>{sign}</span>", unsafe_allow_html=True)
                with col3:
                    st.text(f"Conf: {row['confidence']:.2f}")
                st.divider()
    else:
        st.info("No LLM signals yet. Enable LLM in config and run `python -m src.main analyze`.")

    # ═══ Fused Signals ═══
    st.markdown("### Fused Signal Output")
    if has_fused:
        last = fused_signals.iloc[0]
        dir_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RL", f"{dir_map.get(int(last['rl_direction']), '?')} ({last['rl_size']:.2f})")
        with col2:
            st.metric("LLM Composite", f"{last['llm_composite']:.3f}")
        with col3:
            final_dir = dir_map.get(int(last['final_direction']), '?')
            st.metric("Final", f"{final_dir} ({last['final_size']:.2f}, conf={last['final_confidence']:.2f})")
    else:
        st.info("No fused signals yet. Signal fusion activates when LLM analysis is running.")


def _render_risk_tab(config):
    st.subheader("Risk Management")

    if config is None:
        st.info("Load config to configure risk limits.")
        return

    risk = config.risk

    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_pos = st.slider("Max Position %", 1, 50, int(risk.max_position_pct * 100)) / 100
            max_daily = st.slider("Max Daily Loss %", 1, 20, int(risk.max_daily_loss_pct * 100)) / 100
        with col2:
            max_dd = st.slider("Max Total Drawdown %", 5, 50, int(risk.max_total_drawdown_pct * 100)) / 100
            min_hold = st.number_input("Min Hold Bars", 1, 20, risk.min_hold_bars)
        with col3:
            max_trades = st.number_input("Max Trades/Day", 1, 100, risk.max_trades_per_day)
            min_trade = st.number_input("Min Trade Value ($)", 10, 10000, max(10, int(risk.min_trade_value)), 10)

        if st.form_submit_button("Apply Risk Settings"):
            st.session_state.config_overrides["risk"] = {
                "max_position_pct": max_pos,
                "max_daily_loss_pct": max_daily,
                "max_total_drawdown_pct": max_dd,
                "min_hold_bars": min_hold,
                "max_trades_per_day": max_trades,
                "min_trade_value": min_trade,
            }
            st.success("Risk settings applied for next session.")

    st.markdown("### Current Risk State")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Daily Trades", "0")
    with col2:
        st.metric("Daily P&L", "$0.00")
    with col3:
        st.metric("Drawdown from Peak", "0.00%")
    with col4:
        st.metric("Max Drawdown Limit", f"{risk.max_total_drawdown_pct:.1%}")

    # Risk limit gauges
    st.markdown("---")
    limits_data = {
        "Position Limit": risk.max_position_pct,
        "Daily Loss Limit": risk.max_daily_loss_pct,
        "Total Drawdown Limit": risk.max_total_drawdown_pct,
    }
    for name, val in limits_data.items():
        st.text(f"{name}: {val:.1%}")
        st.progress(min(val * 5, 1.0))


def _render_logs_tab():
    st.subheader("Application Logs")
    st.caption("Auto-refresh controlled by sidebar — select an interval to keep logs live.")

    log_file = Path("logs/trader.log")
    if log_file.exists():
        try:
            with open(log_file) as f:
                lines = f.readlines()[-100:]
            log_text = "".join(lines)
        except Exception:
            log_text = "Could not read log file."
    else:
        log_text = "No log file found. Logs will appear here when the trader is running."

    log_level = st.selectbox("Filter", ["ALL", "INFO", "WARNING", "ERROR", "DEBUG"], index=0)

    filtered = log_text
    if log_level != "ALL":
        filtered = "\n".join(
            line for line in log_text.split("\n") if f"[{log_level}]" in line
        ) or f"No {log_level} messages."

    st.code(filtered, language="log", line_numbers=False)


    # ═══ Auto-refresh trigger ═══
    if st.session_state.auto_refresh_interval > 0:
        elapsed = time.time() - st.session_state.last_refresh
        wait = max(0.1, st.session_state.auto_refresh_interval - elapsed)
        time.sleep(wait)
        st.session_state.last_refresh = time.time()
        st.rerun()


@st.cache_data
def _get_manual_content():
    return """
## Paper Trader v2.0 — 系统概览 / System Overview

Paper Trader v2 是一个 **RL + LLM 混合自动交易系统**，使用 PPO 强化学习模型和 DeepSeek 大语言模型
进行多源分析，通过 Interactive Brokers (IBKR) 执行模拟交易。

Paper Trader v2 is an **RL + LLM hybrid automated trading system** that uses a PPO
reinforcement learning model and DeepSeek LLM for multi-source analysis, executing
paper trades through Interactive Brokers (IBKR).

---

### 当前架构 / Current Architecture

| 组件 Component | 技术 Technology | 说明 Description |
|---|---|---|
| 数据源 Data | yfinance + IBKR | 日线数据 + 1分钟历史 bar（免费/无需订阅） |
| 新闻 News | 29 sources | Yahoo Finance 聚合 + Google News RSS / Multi-source aggregation |
| RL 模型 | PPO (stable-baselines3) | 200k timesteps 训练 / 200k timestep training |
| LLM 分析 | DeepSeek V4 Pro | 4 个分析师并行 / 4 parallel analysts |
| 辩论 Debate | Bull vs Bear | 2 轮结构化辩论 / 2-round structured debate |
| 信号融合 Fusion | 3-Tier | RL注入 → LLM覆盖 → 辩论加成 / Injection → Override → Boost |
| 风控 Risk | 5 limits | 仓位/日内止损/最大回撤/最小交易额/日交易次数 |
| 反思学习 Reflection | Weight tracking | 追踪分析师准确率，动态调权 / Track accuracy, adjust weights |
| 执行 Execution | IBKR Paper Trading | 限价单 / Limit orders |
| 监控 Dashboard | Streamlit | 每秒刷新 / 1s refresh |

---

### 启动指南 / Startup Guide

**步骤 1 / Step 1: 启动 IB Gateway → 登录 Paper Trading 账户**
```bash
# IB Gateway 端口 / Port: 4001
# 确保 API 设置中取消勾选 "Read-Only API"
```

**步骤 2 / Step 2: 设置 DeepSeek API Key**
```bash
export DEEPSEEK_API_KEY="sk-your-key"
```

**步骤 3 / Step 3: 启动系统**
```bash
cd ~/paper-trader
source .venv/bin/activate

# 数据刷新（确保数据最新）
make refresh

# 启动 LLM 分析（首次初始化）
make analyze

# 启动模拟交易（后台持续运行）
make paper

# 启动 Dashboard
make dashboard
```

**步骤 4 / Step 4: 实时监控**
```
Dashboard: http://localhost:8501  →  侧边栏选 "1s" 自动刷新
API Server: http://localhost:8090/docs  →  FastAPI 自动文档
```

---

### 常用命令 / Common Commands

| 命令 / Command | 功能 / Function |
|---|---|
| `make refresh` | 增量更新数据 / Incremental data refresh |
| `make analyze` | 运行 LLM 分析师 / Run LLM analysts |
| `make debate` | 运行多空辩论 / Run bull vs bear debate |
| `make train` | 全量训练 PPO 模型 / Full PPO training |
| `make paper` | 启动模拟交易 / Start paper trading |
| `make dashboard` | 启动控制面板 / Start dashboard |
| `make api` | 启动 REST API / Start API server |
| `make reflect` | 查看分析师表现 / View analyst performance |
| `python -m src.main incremental-train` | 增量训练模型 / Incremental model training |

---

### 交易节奏 / Trading Cadence

| 操作 | 频率 | 说明 |
|---|---|---|
| 价格检查 | 每 5 分钟 | 检测 IBKR 新 bar |
| 新闻刷新 | 每 15 分钟 | 29 个来源聚合 |
| LLM 分析 | 每 60 分钟 | 4 个分析师 + 辩论 |
| 下单执行 | 有新 bar 且距上次交易 >15 分钟 | 信号强度足够时 |
| 反思学习 | 每轮自动 | 动态调整分析师权重 |
| 自动调权 | 每 10 轮 | 根据准确率更新融合权重 |

| Operation | Frequency | Note |
|---|---|---|
| Price check | Every 5 min | Detect new IBKR bar |
| News refresh | Every 15 min | 29 source aggregation |
| LLM analysis | Every 60 min | 4 analysts + debate |
| Order execution | On new bar + >15min since last trade | Sufficient signal strength |
| Reflection | Every iteration | Auto-adjust analyst weights |
| Auto-weight | Every 10 iterations | Update fusion weights by accuracy |

---

### 风控参数 / Risk Parameters

| 参数 Parameter | 当前值 Current | 说明 Description |
|---|---|---|
| max_position_pct | 40% | 单笔最大仓位 / Max single position |
| max_daily_loss_pct | 5% | 日内止损线 / Daily loss limit |
| max_total_drawdown_pct | 15% | 最大回撤线 / Max drawdown limit |
| min_trade_value | $50 | 最小交易金额 / Min trade value |
| max_trades_per_day | 20 | 每日最大交易次数 / Max trades per day |
| paper_trading_capital | $1,276 | 模拟本金 / Simulated capital |

---

### 信号融合权重 / Signal Fusion Weights

| 权重 Weight | 当前值 Current | 说明 Description |
|---|---|---|
| rl_weight | 0.2 | RL 模型权重 (0-1, 越低 LLM 越主导) |
| llm_override_threshold | 0.3 | LLM 覆盖 RL 的共识阈值 |
| agreement_threshold | 0.3 | 分析师一致度阈值 |
| debate_boost | 0.25 | 辩论对最终信号的加成 |
| min_override_confidence | 0.3 | 最小覆盖置信度 |

---

### Dashboard 各标签页说明 / Tab Guide

| 标签 Tab | 内容 Content |
|---|---|
| **Training** | 配置并启动 PPO 模型训练 / Configure and start PPO training |
| **Paper Trading** | 实时净值曲线、持仓、价格条、交易记录 / Live equity curve, positions, prices, trades |
| **Backtest** | 回测策略（SMA/买入持有）含滑点和交易统计 / Backtest with slippage and trade stats |
| **Signals** | 每个标的的 RL 信号 + LLM 分析师信号 + 辩论结果 / Per-symbol RL+LLM+Debate signals |
| **Risk** | 配置风控参数 / Configure risk limits |
| **Logs** | 实时日志查看 / Live log viewer |
| **Manual** | 本手册 / This manual |

---

### 故障排查 / Troubleshooting

| 问题 Problem | 解决方案 Solution |
|---|---|
| IBKR 连接失败 | 检查 IB Gateway 是否运行，端口是否为 4001，Read-Only API 是否取消勾选 |
| LLM 分析未运行 | 确认 `DEEPSEEK_API_KEY` 环境变量已设置，`pip install openai` 已安装 |
| Equity Curve 为 0 或异常 | 可能混入了旧数据，运行 `python -c "import sqlite3; db=sqlite3.connect('logs/trades.db'); db.execute('DELETE FROM portfolio_snapshots WHERE total_equity>10000'); db.commit()"` |
| 数据过期 | 运行 `make refresh` 增量更新 |
| Dashboard 无数据 | 确认 `make paper` 正在后台运行 |
| SELL 订单未执行 | 系统使用模拟持仓进行卖出（不依赖 IBKR 实际持仓） |
| API 调用成本高 | DeepSeek 每 60 分钟一次分析，约 28 次 API 调用/小时，成本极低（~$0.01/小时） |
"""


def _render_manual_tab():
    st.subheader("Operation Manual / 操作手册")
    content = _get_manual_content()
    st.markdown(content, unsafe_allow_html=True)


if __name__ == "__main__":
    run_dashboard()
