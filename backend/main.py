from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd

from indicators import compute_all_indicators
from backtest import backtest_sma_crossover, backtest_rsi_reversion
from predict import predict_next_direction

app = FastAPI(title="Indian Stock Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not (symbol.endswith(".NS") or symbol.endswith(".BO")):
        symbol += ".NS"
    return symbol


def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    ticker = normalize_symbol(symbol)
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for symbol '{symbol}'. "
                             f"Try the NSE code, e.g. RELIANCE, TCS, INFY.")
    df = df.dropna()
    return df


@app.get("/")
def root():
    return {"status": "ok", "message": "Indian Stock Analyzer API is running"}


@app.get("/api/stock/{symbol}")
def get_stock_data(symbol: str, period: str = Query("1y"), interval: str = Query("1d")):
    df = fetch_data(symbol, period, interval)
    return {
        "symbol": normalize_symbol(symbol),
        "dates": [str(d.date()) for d in df.index],
        "open": df["Open"].round(2).tolist(),
        "high": df["High"].round(2).tolist(),
        "low": df["Low"].round(2).tolist(),
        "close": df["Close"].round(2).tolist(),
        "volume": df["Volume"].tolist(),
    }


@app.get("/api/indicators/{symbol}")
def get_indicators(symbol: str, period: str = Query("1y"), interval: str = Query("1d")):
    df = fetch_data(symbol, period, interval)
    indicators = compute_all_indicators(df)
    return {
        "symbol": normalize_symbol(symbol),
        "dates": [str(d.date()) for d in df.index],
        "close": df["Close"].round(2).tolist(),
        **indicators,
    }


@app.get("/api/backtest/{symbol}")
def get_backtest(symbol: str, strategy: str = Query("sma"), period: str = Query("2y"),
                  fast: int = Query(20), slow: int = Query(50),
                  oversold: int = Query(30), overbought: int = Query(70)):
    df = fetch_data(symbol, period, "1d")
    if strategy == "sma":
        result = backtest_sma_crossover(df, fast=fast, slow=slow)
    elif strategy == "rsi":
        result = backtest_rsi_reversion(df, oversold=oversold, overbought=overbought)
    else:
        raise HTTPException(status_code=400, detail="strategy must be 'sma' or 'rsi'")
    result["symbol"] = normalize_symbol(symbol)
    return result


@app.get("/api/predict/{symbol}")
def get_prediction(symbol: str, period: str = Query("2y")):
    df = fetch_data(symbol, period, "1d")
    result = predict_next_direction(df)
    result["symbol"] = normalize_symbol(symbol)
    return result
