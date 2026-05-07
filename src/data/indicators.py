import numpy as np
import pandas as pd
import ta


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    close = df["close"].astype(float)
    df_close_nona = close.dropna()
    if df_close_nona.empty:
        return pd.DataFrame()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    result = pd.DataFrame(index=df.index)
    result["close"] = close
    result["open"] = high * 0  # placeholder, filled below
    result["high"] = high
    result["low"] = low
    result["volume"] = volume

    if "open" in df.columns:
        result["open"] = df["open"].astype(float)
    else:
        result["open"] = close.shift(1).fillna(close)

    result["return_1d"] = close.pct_change()
    result["return_5d"] = close.pct_change(5)
    result["return_20d"] = close.pct_change(20)

    result["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close)
    result["macd_line"] = macd.macd()
    result["macd_signal"] = macd.macd_signal()
    result["macd_hist"] = macd.macd_diff()

    result["sma_20"] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    result["sma_50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    result["sma_ratio"] = result["sma_20"] / result["sma_50"].replace(0, np.nan)

    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    result["atr_14"] = atr
    result["atr_ratio"] = atr / close.replace(0, np.nan)

    result["volume_sma_20"] = ta.trend.SMAIndicator(volume, window=20).sma_indicator()
    result["volume_ratio"] = volume / result["volume_sma_20"].replace(0, np.nan)

    result["bb_upper"] = ta.volatility.BollingerBands(close, window=20, window_dev=2).bollinger_hband()
    result["bb_lower"] = ta.volatility.BollingerBands(close, window=20, window_dev=2).bollinger_lband()
    result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / close

    return result.dropna()
