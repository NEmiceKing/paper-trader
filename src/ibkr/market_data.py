import logging
import threading
from datetime import datetime, timezone
from typing import Callable

import pandas as pd
from ib_insync import IB, Stock, util

logger = logging.getLogger(__name__)


class RealtimeBar:
    """A single real-time bar from IBKR."""
    __slots__ = ("time", "open", "high", "low", "close", "volume", "wap", "count")

    def __init__(self, time_val, open_, high, low, close, volume, wap, count):
        self.time = time_val
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.wap = wap
        self.count = count


class RealtimeBuffer:
    """Thread-safe buffer that collects IBKR 5-second bars and resamples to target timeframe."""

    def __init__(self, symbol: str, target_bar_size: str = "5 mins"):
        self.symbol = symbol
        self.target_bar_size = target_bar_size
        self._bars: list[RealtimeBar] = []
        self._lock = threading.Lock()
        self._latest_price: float = 0.0
        self._last_update: datetime | None = None

    def add_bar(self, bar: RealtimeBar):
        with self._lock:
            self._bars.append(bar)
            self._latest_price = bar.close
            self._last_update = datetime.now(timezone.utc)

    @property
    def latest_price(self) -> float:
        return self._latest_price

    @property
    def last_update(self) -> datetime | None:
        return self._last_update

    def to_dataframe(self) -> pd.DataFrame:
        """Convert accumulated raw 5s bars to a DataFrame."""
        with self._lock:
            if not self._bars:
                return pd.DataFrame()
            records = [
                {"date": b.time, "open": b.open, "high": b.high, "low": b.low,
                 "close": b.close, "volume": b.volume}
                for b in self._bars
            ]
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df

    def to_ohlcv(self, bar_size: str | None = None) -> pd.DataFrame:
        """Resample raw bars to target bar size (e.g. '5min', '30min', '1h')."""
        df = self.to_dataframe()
        if df.empty:
            return df

        size = bar_size or self.target_bar_size
        resampled = df.resample(size).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        resampled.columns = [c.lower() for c in resampled.columns]
        return resampled

    def bars_since(self, timestamp: datetime) -> int:
        """Count raw bars received since a given timestamp."""
        with self._lock:
            return sum(1 for b in self._bars if b.time > timestamp)


class MarketDataFeed:
    """IBKR market data feed with historical and real-time bar support."""

    def __init__(self, ib: IB):
        self.ib = ib
        self._subscriptions: dict[str, object] = {}
        self._buffers: dict[str, RealtimeBuffer] = {}
        self._bar_size: str = "5 mins"

    # ── Historical ────────────────────────────────────────────────

    def request_historical_bars(
        self, symbol: str, duration: str = "2 Y", bar_size: str = "1 day", end: str = ""
    ) -> pd.DataFrame:
        contract = Stock(symbol, "SMART", "USD")
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="ADJUSTED_LAST",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            logger.warning(f"No historical data for {symbol}")
            return pd.DataFrame()

        df = util.df(bars)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        df.columns = [c.lower() for c in df.columns]
        return df

    # ── Real-time ─────────────────────────────────────────────────

    def subscribe_realtime_bars(
        self,
        symbol: str,
        bar_size: int = 5,
        target_resample: str = "5 mins",
        callback: Callable | None = None,
    ):
        """Subscribe to IBKR 5-second real-time bars and buffer them.

        Falls back to periodic snapshot polling if real-time subscription is unavailable.

        Args:
            symbol: Stock symbol
            bar_size: IBKR bar size in seconds (5 is the only valid value for stocks)
            target_resample: Resample target (e.g. '5min', '30min', '1H')
            callback: Optional callback receiving (symbol, bar) on each update
        """
        self._bar_size = target_resample
        buffer = RealtimeBuffer(symbol, target_bar_size=target_resample)
        self._buffers[symbol] = buffer

        contract = Stock(symbol, "SMART", "USD")

        def _on_bar(bars, has_new_bar):
            if not has_new_bar:
                return
            try:
                latest = bars[-1]
                rt_bar = RealtimeBar(
                    time_val=latest.time,
                    open_=latest.open_,
                    high=latest.high,
                    low=latest.low,
                    close=latest.close,
                    volume=latest.volume,
                    wap=latest.wap,
                    count=latest.count,
                )
                buffer.add_bar(rt_bar)
                if callback:
                    callback(symbol, rt_bar)
            except Exception as e:
                logger.debug(f"Error processing real-time bar for {symbol}: {e}")

        # Try streaming subscription; fall back to historical polling
        try:
            bars = self.ib.reqRealTimeBars(
                contract, barSize=bar_size, whatToShow="TRADES", useRTH=False
            )
            bars.updateEvent += _on_bar
            self._subscriptions[symbol] = bars
            logger.info(f"Subscribed to real-time {bar_size}s bars for {symbol} (resample={target_resample})")
        except Exception:
            # Real-time requires market data subscription — use historical polling instead
            logger.info(f"Real-time bars need subscription for {symbol}, using historical polling (free)")
            self._start_snapshot_polling(symbol, contract, buffer, interval=bar_size)

    def _start_snapshot_polling(self, symbol: str, contract, buffer: RealtimeBuffer, interval: int = 5):
        """Fallback: use historical 1-min bars to get latest prices (free, no subscription needed)."""
        import threading

        def poll():
            while symbol in self._buffers:
                try:
                    bars = self.ib.reqHistoricalData(
                        contract, endDateTime="", durationStr="1 D",
                        barSizeSetting="1 min", whatToShow="TRADES",
                        useRTH=False, formatDate=1,
                    )
                    if bars and len(bars) > 0:
                        latest = bars[-1]
                        bar = RealtimeBar(
                            time_val=latest.date,
                            open_=latest.open,
                            high=latest.high,
                            low=latest.low,
                            close=latest.close,
                            volume=latest.volume,
                            wap=latest.close,
                            count=1,
                        )
                        buffer.add_bar(bar)
                except Exception as e:
                    logger.debug(f"Historical poll error for {symbol}: {e}")
                import time as _time
                _time.sleep(interval)

        t = threading.Thread(target=poll, daemon=True)
        t.start()
        logger.info(f"Started historical polling for {symbol} (interval={interval}s)")

    def unsubscribe_all(self):
        for symbol, bars in self._subscriptions.items():
            self.ib.cancelRealTimeBars(bars)
        self._subscriptions.clear()
        self._buffers.clear()
        logger.info("Unsubscribed all real-time bars")

    # ── Buffer access ─────────────────────────────────────────────

    def get_buffer(self, symbol: str) -> RealtimeBuffer | None:
        return self._buffers.get(symbol)

    def get_latest_price(self, symbol: str) -> float:
        buf = self._buffers.get(symbol)
        return buf.latest_price if buf else 0.0

    def get_all_prices(self) -> dict[str, float]:
        return {sym: buf.latest_price for sym, buf in self._buffers.items()}

    def get_intraday_ohlcv(self, symbol: str, bar_size: str | None = None) -> pd.DataFrame:
        """Get resampled intraday OHLCV DataFrame for a symbol."""
        buf = self._buffers.get(symbol)
        if buf is None:
            return pd.DataFrame()
        return buf.to_ohlcv(bar_size or self._bar_size)

    def build_combined_features(
        self,
        symbol: str,
        historical_daily: pd.DataFrame,
        lookback_days: int = 20,
    ) -> pd.DataFrame | None:
        """Merge historical daily data with today's intraday bars for feature computation.

        Takes the last N days from historical + today's intraday resampled bars.
        """
        intraday = self.get_intraday_ohlcv(symbol)
        if intraday.empty or historical_daily.empty:
            return None

        # Ensure daily data index is tz-naive for merging
        daily = historical_daily.copy()
        if daily.index.tz is not None:
            daily.index = daily.index.tz_localize(None)

        # Get last N-1 days (today will come from intraday)
        last_daily = daily.iloc[-lookback_days + 1:] if len(daily) >= lookback_days else daily

        # Combine: yesterday's last bar → today's intraday bars
        combined = pd.concat([last_daily, intraday])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        return combined

    def get_current_bar_series(self, symbol: str, historical_daily: pd.DataFrame,
                               lookback: int = 20) -> pd.DataFrame | None:
        """Build a complete feature-ready DataFrame: historical daily + latest intraday.

        This is the primary method used by the paper trading loop.
        """
        buf = self._buffers.get(symbol)
        if buf is None:
            return None

        intraday = buf.to_ohlcv()
        if intraday.empty:
            # No intraday data yet — use historical only
            return historical_daily.copy()

        daily = historical_daily.copy()
        if daily.index.tz is not None:
            daily.index = daily.index.tz_localize(None)

        # Use last N-1 daily bars + all of today's intraday bars
        last_daily = daily.iloc[-lookback + 1:] if len(daily) > lookback else daily

        # Combine
        combined = pd.concat([last_daily, intraday])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        # Ensure we have all required columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in combined.columns:
                return None

        return combined
