import pandas as pd
import numpy as np


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def compute_all_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"]

    macd_line, signal_line, hist = macd(close)
    upper, mid, lower = bollinger_bands(close)

    result = {
        "sma_20": sma(close, 20),
        "sma_50": sma(close, 50),
        "ema_12": ema(close, 12),
        "ema_26": ema(close, 26),
        "rsi_14": rsi(close, 14),
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": hist,
        "bb_upper": upper,
        "bb_mid": mid,
        "bb_lower": lower,
    }

    out = {}
    for key, series in result.items():
        out[key] = [None if pd.isna(v) else round(float(v), 4) for v in series]
    return out
