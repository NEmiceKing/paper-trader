import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.config.loader import load_config
from src.data.indicators import compute_features
from src.data.pipeline import DataPipeline
from src.data.store import MarketDataStore
from src.agent.env import TradingEnv
from src.agent.trainer import train_agent
from src.backtest.engine import VectorizedBacktest, SlippageModel
from src.backtest.metrics import compute_metrics
from src.monitor.logger import TradeLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("paper-trader")


def cmd_download(config):
    pipeline = DataPipeline()
    pipeline.download_and_store(config.symbols, config.data.default_start)
    logger.info(f"Downloaded data for {config.symbols}")


def cmd_backtest(config):
    store = MarketDataStore()
    all_metrics = {}
    slippage = SlippageModel()
    for sym in config.symbols:
        try:
            df = store.read_bars(sym, start=config.data.default_start)
        except FileNotFoundError:
            logger.warning(f"No data for {sym}, skipping")
            continue
        bt = VectorizedBacktest()

        # SMA crossover with slippage
        result = bt.run_sma_cross(df, slippage=slippage)
        m = compute_metrics(result.equity_curve)
        m["symbol"] = sym

        # Trade statistics
        if result.trades:
            trade_stats = bt.compute_trade_statistics(result.trades)
            m["win_rate"] = round(trade_stats.win_rate, 4)
            m["profit_factor"] = round(trade_stats.profit_factor, 4)
            m["total_trades"] = trade_stats.total_trades

        # Benchmark comparison vs buy-and-hold
        bench = bt.run_buy_and_hold(df["close"].values)
        bench_rets = np.diff(bench.equity_curve) / (bench.equity_curve[:-1] + 1e-12)
        alpha_beta = bt.compare_to_benchmark(result.equity_curve, df["close"].values)
        m.update(alpha_beta)

        all_metrics[sym] = m
        logger.info(
            f"{sym}: Sharpe={m.get('sharpe', 0):.4f}, MaxDD={m.get('max_drawdown', 0):.4f}, "
            f"Alpha={m.get('alpha', 0):.4f}, Beta={m.get('beta', 0):.4f}, "
            f"WinRate={m.get('win_rate', 0):.2%}, PF={m.get('profit_factor', 0):.2f}"
        )
    return all_metrics


def cmd_train(config):
    store = MarketDataStore()
    pipeline = DataPipeline(store)

    all_features = []
    all_prices = []
    for sym in config.symbols:
        try:
            df = store.read_bars(sym, start=config.data.default_start)
        except FileNotFoundError:
            logger.warning(f"No data for {sym}, skipping")
            continue
        feat = compute_features(df)
        all_features.append(feat.values.astype(np.float32))
        all_prices.append(df["close"].values.astype(np.float32))

    if not all_features:
        logger.error("No data available for training")
        return

    combined_features = np.concatenate(all_features, axis=0)
    combined_prices = np.concatenate(all_prices, axis=0)

    logger.info(f"Training on {len(combined_features)} total bars across {len(all_features)} symbols")
    model, metrics = train_agent(combined_features, combined_prices, config)
    logger.info(f"Test Sharpe: {metrics['test_sharpe']:.4f}, Test MaxDD: {metrics['test_max_drawdown']:.4f}")
    return model, metrics


def cmd_paper_trade(config):
    try:
        from src.ibkr.client import IBKRClient
        from src.ibkr.market_data import MarketDataFeed
        from src.ibkr.order_manager import AccountSummary, OrderManager, TradeSignal
        from src.risk.limits import RiskEngine, RiskLimits
    except ImportError:
        logger.error("IBKR integration not available. Install ib_insync.")
        return

    project_root = Path(__file__).resolve().parent.parent
    model_path = str(project_root / "models" / "ppo_trader.zip")
    if not Path(model_path).exists():
        logger.error("No trained model found. Run 'train' first.")
        return

    model = PPO.load(model_path)

    client = IBKRClient(config.ibkr.host, config.ibkr.port, config.ibkr.client_id)
    client.connect()

    # Initialize optional signal fusion components
    fusion_config = config.signal_fusion
    llm_mgr = None
    debate_mgr = None

    if config.llm.enabled:
        try:
            from src.agent.llm_analyst import LLMAnalystManager
            llm_mgr = LLMAnalystManager(config.llm)
            logger.info("LLM Analyst Manager initialized")
        except ImportError:
            logger.warning("LLM features enabled but openai package not installed. Run: pip install openai")

    if config.debate.enabled:
        try:
            from src.agent.debate import DebateManager
            debate_mgr = DebateManager(config.debate)
            logger.info("Debate Manager initialized")
        except ImportError:
            logger.warning("Debate features enabled but dependencies missing")

    # Initialize reflection tracker
    reflection_tracker = None
    try:
        from src.agent.reflection import ReflectionTracker
        reflection_tracker = ReflectionTracker()
        logger.info("Reflection Tracker initialized")
    except ImportError:
        pass

    feed = None
    try:
        feed = MarketDataFeed(client.ib)
        order_mgr = OrderManager(client.ib)
        order_mgr._min_trade_value = config.risk.min_trade_value  # from config

        risk_limits = RiskLimits(
            max_position_pct=config.risk.max_position_pct,
            max_daily_loss_pct=config.risk.max_daily_loss_pct,
            max_total_drawdown_pct=config.risk.max_total_drawdown_pct,
            min_hold_bars=config.risk.min_hold_bars,
            max_trades_per_day=config.risk.max_trades_per_day,
            min_trade_value=config.risk.min_trade_value,
        )
        risk_engine = RiskEngine(risk_limits)
        trade_logger = TradeLogger(config.logging.trades_db)

        store = MarketDataStore()
        pipeline = DataPipeline(store)

        # Use historical 1-min bars for pricing (free, no subscription needed)
        # Real-time streaming requires $4.50/month market data subscription
        use_realtime = False  # Will be set to True if any symbol gets live bars
        for sym in config.symbols:
            try:
                # Try real-time first; fallback to historical polling
                feed.subscribe_realtime_bars(
                    sym, bar_size=5, target_resample="1 min"
                )
                # Wait briefly to see if real-time bars arrive
                ib_sleep_seconds = 1
                import time as _time
                _time.sleep(ib_sleep_seconds)
                buf = feed.get_buffer(sym)
                if buf and buf.latest_price > 0:
                    use_realtime = True
            except Exception as e:
                logger.warning(f"Could not subscribe to bars for {sym}: {e}")

        if use_realtime:
            logger.info(f"Live pricing active for {len(config.symbols)} symbols")
        else:
            logger.info("Using historical 1-min bar polling for pricing (free)")

        # Use simulated capital for position sizing (paper trading)
        simulated_capital = getattr(config, "paper_trading_capital", 100000.0)
        simulated_equity = simulated_capital
        simulated_cash = simulated_capital
        simulated_positions: dict[str, dict] = {}  # sym -> {shares, avg_cost}
        logger.info(f"Paper trading with simulated capital: ${simulated_capital:,.2f} USD")
        logger.info("Paper trading started. Press Ctrl+C to stop.")
        first_run = True
        iteration = 0
        import os as _os
        interval = int(_os.environ.get("TRADING_INTERVAL", config.logging.trading_interval_seconds))
        last_llm_run: dict[str, datetime] = {}  # per-symbol LLM analysis timestamps
        last_trade: dict[str, datetime] = {}     # per-symbol last trade timestamps
        last_new_bar: dict[str, str] = {}         # per-symbol last bar timestamp for change detection

        while True:
            if first_run:
                pipeline.refresh_data(config.symbols)
                order_mgr.sync_positions()
                risk_engine.reset_daily(simulated_capital)
                first_run = False

            # ── Decision gate: should we run full analysis this cycle? ──
            now = datetime.now(timezone.utc)
            market_open = now.hour >= 13 or now.hour < 20  # ~9:30 AM - 4:00 PM ET
            is_llm_cycle = (iteration % 12 == 0)  # every 12 cycles (~60 min) run LLM
            is_news_cycle = (iteration % 3 == 0)   # every 3 cycles (~15 min) refresh news

            for sym in config.symbols:
                # ── Safe data load ──
                try:
                    historical = store.read_bars(sym)
                except FileNotFoundError:
                    continue  # skip symbols with no data

                # Detect new bar: only run PPO if latest bar changed
                latest_bar_id = f"{historical.index[-1]}" if len(historical) > 0 else ""
                new_bar_detected = (sym not in last_new_bar or last_new_bar[sym] != latest_bar_id)
                last_new_bar[sym] = latest_bar_id

                # Enough time since last LLM analysis? (per analyst cache TTL)
                min_since_llm = 999
                if sym in last_llm_run:
                    min_since_llm = (now - last_llm_run[sym]).total_seconds() / 60
                should_run_llm = is_llm_cycle or min_since_llm > 60

                # Enough time/change since last trade?
                min_since_trade = 999
                if sym in last_trade:
                    min_since_trade = (now - last_trade[sym]).total_seconds() / 60
                should_execute = new_bar_detected and (first_run or min_since_trade > 15)

                if not (new_bar_detected or should_run_llm or should_execute):
                    continue  # Nothing changed, skip this symbol

                if use_realtime:
                    combined = feed.get_current_bar_series(sym, historical, lookback=config.agent.observation_bars)
                    if combined is not None and not combined.empty:
                        df = combined
                    else:
                        df = historical  # fallback if intraday data not yet available
                else:
                    df = historical

                feat = compute_features(df)
                if feat.empty:
                    continue

                # Use latest price from real-time buffer if available
                rt_price = feed.get_latest_price(sym)
                if rt_price > 0:
                        # Replace last close with real-time price
                    feat_data = feat.values.astype(np.float32)
                    if "close" in feat.columns:
                        feat_data[-1, list(feat.columns).index("close")] = rt_price
                else:
                    feat_data = feat.values.astype(np.float32)

                env = TradingEnv(feat_data, feat["close"].values.astype(np.float32),
                                 window=config.agent.observation_bars)
                obs, _ = env.reset()

                action_arr, _ = model.predict(obs, deterministic=config.agent.inference.deterministic)

                direction = int(np.argmax(action_arr))
                size = float(np.clip(abs(action_arr[direction]), 0.0, 1.0))
                rl_confidence = float(max(action_arr))
                if size < 0.02:
                    direction = 0

                rl_signal = TradeSignal(symbol=sym, direction=direction, size=size, confidence=rl_confidence)

                # ── Reflection: evaluate previous predictions ──
                current_price = float(historical["close"].iloc[-1])
                if reflection_tracker is not None:
                    reflection_tracker.evaluate_predictions(sym, current_price)

                # ── LLM analysis: only run on schedule or when needed ──
                llm_reports = []
                debate_result = None
                if llm_mgr is not None and (should_run_llm or is_news_cycle):
                    try:
                        llm_reports = llm_mgr.analyze_symbol(sym, df)
                        # Apply reflection weights to adjust confidence
                        if reflection_tracker is not None:
                            from src.agent.reflection import apply_reflection_weights
                            llm_reports = apply_reflection_weights(llm_reports, reflection_tracker)
                        for report in llm_reports:
                            trade_logger.log_llm_signal(
                                sym, report.analyst_name, report.signal,
                                report.confidence, report.summary[:200]
                            )
                    except Exception as e:
                        logger.debug(f"LLM analysis failed for {sym}: {e}")

                    if debate_mgr is not None and llm_reports:
                        try:
                            debate_result = debate_mgr.debate(sym, df, llm_reports)
                            trade_logger.log_llm_signal(
                                sym, "debate", debate_result.composite_signal,
                                debate_result.confidence, debate_result.bull_summary[:200]
                            )
                        except Exception as e:
                            logger.debug(f"Debate failed for {sym}: {e}")

                # ── Signal fusion ──
                if fusion_config.rl_weight < 1.0 and (llm_reports or debate_result):
                    try:
                        from src.agent.signal_fusion import fuse_signals
                        final_signal = fuse_signals(rl_signal, llm_reports, debate_result, fusion_config)
                        trade_logger.log_fused_signal(
                            sym, rl_signal.direction, rl_signal.size,
                            float(np.mean([r.signal * r.confidence for r in llm_reports])) if llm_reports else 0.0,
                            debate_result.confidence if debate_result else 0.0,
                            final_signal.direction, final_signal.size, final_signal.confidence,
                        )
                    except ImportError:
                        final_signal = rl_signal
                else:
                    final_signal = rl_signal

                # Use simulated portfolio for position sizing
                # Inject simulated positions so SELL orders can size correctly
                if sym in simulated_positions:
                    from src.ibkr.order_manager import Position
                    sp = simulated_positions[sym]
                    order_mgr.positions[sym] = Position(
                        symbol=sym, quantity=sp["shares"],
                        avg_cost=sp["avg_cost"], market_price=current_price,
                    )
                else:
                    order_mgr.positions.pop(sym, None)

                account = AccountSummary(
                    cash=simulated_cash,
                    equity=simulated_equity,
                    buying_power=simulated_equity * 2,
                )
                decision = risk_engine.check_signal(final_signal, account)
                if not decision.approved:
                    logger.info(f"Risk rejected {sym}: {decision.reason}")
                    continue

                final_size = decision.modified_size if decision.modified else final_signal.size
                exec_signal = TradeSignal(symbol=sym, direction=final_signal.direction, size=final_size)

                # ── Decision gate: only execute if signal is strong enough ──
                if not should_execute:
                    continue

                try:
                    order_id = order_mgr.place_trade(exec_signal, account)
                except Exception as e:
                    logger.warning(f"Order placement failed for {sym}: {e}")
                    order_id = None
                    continue
                if order_id:
                    last_trade[sym] = now
                if should_run_llm:
                    last_llm_run[sym] = now

                # ── Record predictions for reflection learning ──
                if reflection_tracker is not None and llm_reports:
                    for report in llm_reports:
                        reflection_tracker.record_prediction(
                            sym, report.analyst_name, report.signal,
                            report.confidence, current_price,
                        )

                action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}

                # Update simulated portfolio
                if order_id and final_signal.direction == 1:  # BUY
                    price = feed.get_latest_price(sym)
                    if price <= 0:
                        price = float(historical["close"].iloc[-1])
                    qty = int(simulated_cash * final_signal.size / price) if price > 0 else 0
                    if qty > 0:
                        cost = qty * price
                        simulated_cash -= cost
                        if sym in simulated_positions:
                            old_qty = simulated_positions[sym]["shares"]
                            old_cost = simulated_positions[sym]["avg_cost"] * old_qty
                            new_qty = old_qty + qty
                            simulated_positions[sym] = {"shares": new_qty, "avg_cost": (old_cost + cost) / new_qty}
                        else:
                            simulated_positions[sym] = {"shares": qty, "avg_cost": price}
                elif order_id and final_signal.direction == 2:  # SELL
                    if sym in simulated_positions:
                        price = feed.get_latest_price(sym)
                        if price <= 0:
                            price = float(historical["close"].iloc[-1])
                        pos = simulated_positions[sym]
                        sell_qty = int(pos["shares"] * final_signal.size)
                        if sell_qty > 0:
                            proceeds = sell_qty * price
                            simulated_cash += proceeds
                            remaining = pos["shares"] - sell_qty
                            if remaining <= 0:
                                del simulated_positions[sym]
                            else:
                                simulated_positions[sym] = {"shares": remaining, "avg_cost": pos["avg_cost"]}

                # Calculate simulated equity
                simulated_equity = simulated_cash
                position_value = 0.0
                for s, p in simulated_positions.items():
                    price = feed.get_latest_price(s)
                    if price <= 0:
                        try:
                            price = float(store.read_bars(s)["close"].iloc[-1])
                        except Exception:
                            price = p["avg_cost"]
                    position_value += p["shares"] * price
                simulated_equity += position_value

                trade_logger.log_trade(
                    symbol=sym,
                    action=action_map[final_signal.direction],
                    quantity=0.0,
                    price=rt_price if rt_price > 0 else 0.0,
                    order_id=order_id,
                    status="PENDING" if order_id else "SKIPPED",
                    confidence=final_signal.confidence,
                    risk_decision=decision.reason or "APPROVED",
                )

            order_mgr.sync_positions()
            # Log SIMULATED portfolio values
            sim_exposure = (simulated_equity - simulated_cash) / (simulated_equity + 1e-12) if simulated_equity > 0 else 0.0
            # ── Save live prices for dashboard ──
            try:
                import json as _json
                live_prices = {}
                for sym in config.symbols:
                    lp = feed.get_latest_price(sym)
                    if lp <= 0:
                        try:
                            lp = float(store.read_bars(sym)["close"].iloc[-1])
                        except Exception:
                            lp = 0.0
                    live_prices[sym] = round(lp, 2)
                live_prices["_timestamp"] = datetime.now(timezone.utc).isoformat()
                with open("logs/live_prices.json", "w") as f:
                    _json.dump(live_prices, f)
            except Exception:
                pass

            trade_logger.log_snapshot(
                equity=simulated_equity,
                cash=simulated_cash,
                exposure=sim_exposure,
            )

            # ── Auto-adjust weights from reflection data ──
            iteration += 1
            if reflection_tracker is not None and iteration % 10 == 0:
                try:
                    from src.agent.signal_fusion import auto_adjust_weights
                    updates = auto_adjust_weights(fusion_config, reflection_tracker, iteration)
                    if updates:
                        logger.info(f"Weight updates: {updates}")
                except Exception as e:
                    logger.debug(f"Auto-weight adjustment failed: {e}")

            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Paper trading stopped")
    finally:
        if feed is not None:
            feed.unsubscribe_all()
        client.disconnect()


def cmd_analyze(config):
    """Run LLM analysts for all configured symbols."""
    if not config.llm.enabled:
        logger.error("LLM features not enabled in config. Set llm.enabled=true")
        return

    try:
        from src.agent.llm_analyst import LLMAnalystManager
    except ImportError:
        logger.error("LLM dependencies not installed. Run: pip install anthropic")
        return

    store = MarketDataStore()
    pipeline = DataPipeline(store)
    pipeline.download_and_store(config.symbols, config.data.default_start)

    mgr = LLMAnalystManager(config.llm)
    for sym in config.symbols:
        try:
            df = store.read_bars(sym, start=config.data.default_start)
        except FileNotFoundError:
            logger.warning(f"No data for {sym}, skipping")
            continue
        reports = mgr.analyze_symbol(sym, df)
        for r in reports:
            sign = "BULLISH" if r.signal > 0 else ("BEARISH" if r.signal < 0 else "NEUTRAL")
            logger.info(f"[{sym}] {r.analyst_name}: {sign} (conf={r.confidence:.2f}) — {r.summary[:100]}")


def cmd_debate(config):
    """Run bull vs bear debate for a symbol."""
    if not config.debate.enabled:
        logger.error("Debate not enabled in config. Set debate.enabled=true")
        return

    try:
        from src.agent.debate import DebateManager
        from src.agent.llm_analyst import LLMAnalystManager
    except ImportError:
        logger.error("LLM dependencies not installed")
        return

    store = MarketDataStore()
    pipeline = DataPipeline(store)
    pipeline.download_and_store(config.symbols, config.data.default_start)

    llm_mgr = LLMAnalystManager(config.llm) if config.llm.enabled else None
    debate_mgr = DebateManager(config.debate)

    for sym in config.symbols:
        try:
            df = store.read_bars(sym, start=config.data.default_start)
        except FileNotFoundError:
            continue
        reports = llm_mgr.analyze_symbol(sym, df) if llm_mgr else []
        result = debate_mgr.debate(sym, df, reports)
        sign = "BULLISH" if result.composite_signal > 0 else ("BEARISH" if result.composite_signal < 0 else "NEUTRAL")
        logger.info(f"[{sym}] DEBATE RESULT: {sign} (conf={result.confidence:.2f})")
        for r in result.rounds:
            logger.info(f"  Round {r.round_number}: Bull={r.bull_score:.2f} Bear={r.bear_score:.2f}")
            logger.info(f"    Bull: {r.bull_argument[:120]}...")
            logger.info(f"    Bear: {r.bear_argument[:120]}...")


def cmd_mine_factors(config):
    """Run alpha factor mining."""
    if not config.factor_mining.enabled:
        logger.error("Factor mining not enabled. Set factor_mining.enabled=true")
        return

    from src.data.factor_miner import FactorMiner, evaluate_factors, select_top_k
    from src.data.factor_miner import FactorMiningConfig

    store = MarketDataStore()
    pipeline = DataPipeline(store)
    pipeline.download_and_store(config.symbols, config.data.default_start)

    fc = FactorMiningConfig(
        max_factors=config.factor_mining.max_factors,
        top_k=config.factor_mining.top_k,
        ic_threshold=config.factor_mining.ic_threshold,
        population_size=config.factor_mining.population_size,
        generations=config.factor_mining.generations,
    )

    for sym in config.symbols:
        try:
            df = store.read_bars(sym, start=config.data.default_start)
        except FileNotFoundError:
            continue
        feat = compute_features(df)
        miner = FactorMiner(fc)
        factors = miner.brute_force_search(feat, df["close"].values.astype(float))
        top_factors = select_top_k(factors, fc.top_k)
        logger.info(f"[{sym}] Top {len(top_factors)} factors:")
        for f in top_factors:
            logger.info(f"  {f.name}: IC={f.ic:.4f}, RankIC={f.rank_ic:.4f}, Sharpe={f.sharpe:.4f}")


def cmd_parse_strategy(config):
    """Parse a natural language strategy description."""
    if not config.nlp.enabled:
        logger.error("NLP not enabled. Set nlp.enabled=true")
        return

    from src.nlp.strategy_parser import parse_strategy, strategy_to_config

    text = input("Describe your trading strategy: ").strip()
    if not text:
        logger.error("No input provided")
        return

    intent = parse_strategy(text)
    overrides = strategy_to_config(intent)
    logger.info(f"Parsed strategy intent: {intent}")
    logger.info(f"Config overrides: {overrides}")
    return overrides


def cmd_refresh(config):
    """Incrementally refresh market data for all symbols."""
    pipeline = DataPipeline(config=config)
    results = pipeline.refresh_data(config.symbols)
    total_new = 0
    for sym, info in results.items():
        if info["new_bars"] == "full":
            logger.info(f"  {sym}: full download → {info['after']}")
        else:
            logger.info(f"  {sym}: {info['before']} → {info['after']} (+{info['new_bars']} bars)")
            total_new += info["new_bars"]
    logger.info(f"Refresh complete. {total_new} new bars total across {len(config.symbols)} symbols.")


def cmd_api(config):
    """Start FastAPI REST server."""
    from src.monitor.api import run_api
    logger.info(f"Starting API server on port 8090")
    run_api(port=8090)


def cmd_incremental_train(config):
    """Fine-tune existing PPO model on latest data."""
    from src.data.store import MarketDataStore
    from src.data.indicators import compute_features
    from src.agent.trainer import incremental_train

    store = MarketDataStore()
    all_feat, all_prices = [], []
    for sym in config.symbols:
        try:
            df = store.read_bars(sym, start=config.data.default_start)
        except FileNotFoundError:
            continue
        feat = compute_features(df)
        all_feat.append(feat.values.astype(np.float32))
        all_prices.append(df["close"].values.astype(np.float32))

    if not all_feat:
        logger.error("No data available")
        return

    features = np.concatenate(all_feat, axis=0)
    prices = np.concatenate(all_prices, axis=0)
    logger.info(f"Incremental training on {len(features)} bars")
    model, metrics = incremental_train(features, prices, config, timesteps=8000)
    logger.info(f"Done: Sharpe={metrics['test_sharpe']:.4f}, MaxDD={metrics['test_max_drawdown']:.4f}")


def cmd_reflect(config):
    """Show reflection report (analyst performance tracking)."""
    try:
        from src.agent.reflection import ReflectionTracker
        tracker = ReflectionTracker()
        print(tracker.get_report())
    except ImportError:
        logger.error("Reflection module not available.")


def main():
    parser = argparse.ArgumentParser(description="Paper Trader - RL stock trading agent")
    parser.add_argument("command", choices=[
        "download", "backtest", "train", "paper", "dashboard",
        "analyze", "debate", "mine-factors", "parse-strategy", "refresh",
        "api", "reflect", "incremental-train",
    ], help="Action to run")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.command == "download":
        cmd_download(config)
    elif args.command == "backtest":
        cmd_backtest(config)
    elif args.command == "train":
        cmd_train(config)
    elif args.command == "paper":
        cmd_paper_trade(config)
    elif args.command == "dashboard":
        from src.monitor.app import run_dashboard
        run_dashboard(config)
    elif args.command == "analyze":
        cmd_analyze(config)
    elif args.command == "debate":
        cmd_debate(config)
    elif args.command == "mine-factors":
        cmd_mine_factors(config)
    elif args.command == "parse-strategy":
        cmd_parse_strategy(config)
    elif args.command == "refresh":
        cmd_refresh(config)
    elif args.command == "api":
        cmd_api(config)
    elif args.command == "reflect":
        cmd_reflect(config)
    elif args.command == "incremental-train":
        cmd_incremental_train(config)


if __name__ == "__main__":
    main()
