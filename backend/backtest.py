import pandas as pd
import numpy as np
from indicators import sma, rsi


def backtest_sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                            initial_capital: float = 100000.0, cost_pct: float = 0.001):
    close = df["Close"]
    fast_ma = sma(close, fast)
    slow_ma = sma(close, slow)

    signal = (fast_ma > slow_ma).astype(int)
    signal = signal.shift(1).fillna(0)

    returns = close.pct_change().fillna(0)
    strategy_returns = signal * returns

    position_change = signal.diff().abs().fillna(0)
    strategy_returns = strategy_returns - position_change * cost_pct

    equity_curve = (1 + strategy_returns).cumprod() * initial_capital
    buy_hold_curve = (1 + returns).cumprod() * initial_capital

    total_return = (equity_curve.iloc[-1] / initial_capital - 1) * 100
    buy_hold_return = (buy_hold_curve.iloc[-1] / initial_capital - 1) * 100

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    if strategy_returns.std() > 0:
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    num_trades = int(position_change.sum() / 2)

    return {
        "strategy": f"SMA Crossover ({fast}/{slow})",
        "total_return_pct": round(float(total_return), 2),
        "buy_hold_return_pct": round(float(buy_hold_return), 2),
        "max_drawdown_pct": round(float(max_drawdown), 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "num_trades": num_trades,
        "equity_curve": [round(float(v), 2) for v in equity_curve.fillna(initial_capital)],
        "buy_hold_curve": [round(float(v), 2) for v in buy_hold_curve.fillna(initial_capital)],
        "dates": [str(d.date()) for d in df.index],
    }


def backtest_rsi_reversion(df: pd.DataFrame, oversold: int = 30, overbought: int = 70,
                            initial_capital: float = 100000.0, cost_pct: float = 0.001):
    close = df["Close"]
    rsi_val = rsi(close, 14)

    signal = pd.Series(0, index=df.index)
    position = 0
    for i in range(len(df)):
        if rsi_val.iloc[i] < oversold:
            position = 1
        elif rsi_val.iloc[i] > overbought:
            position = 0
        signal.iloc[i] = position

    signal = signal.shift(1).fillna(0)
    returns = close.pct_change().fillna(0)
    strategy_returns = signal * returns
    position_change = signal.diff().abs().fillna(0)
    strategy_returns = strategy_returns - position_change * cost_pct

    equity_curve = (1 + strategy_returns).cumprod() * initial_capital
    buy_hold_curve = (1 + returns).cumprod() * initial_capital

    total_return = (equity_curve.iloc[-1] / initial_capital - 1) * 100
    buy_hold_return = (buy_hold_curve.iloc[-1] / initial_capital - 1) * 100
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() > 0 else 0.0
    num_trades = int(position_change.sum() / 2)

    return {
        "strategy": f"RSI Mean Reversion ({oversold}/{overbought})",
        "total_return_pct": round(float(total_return), 2),
        "buy_hold_return_pct": round(float(buy_hold_return), 2),
        "max_drawdown_pct": round(float(max_drawdown), 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "num_trades": num_trades,
        "equity_curve": [round(float(v), 2) for v in equity_curve.fillna(initial_capital)],
        "buy_hold_curve": [round(float(v), 2) for v in buy_hold_curve.fillna(initial_capital)],
        "dates": [str(d.date()) for d in df.index],
    }
