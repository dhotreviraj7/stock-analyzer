import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from indicators import sma, ema, rsi, macd, bollinger_bands


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    feat = pd.DataFrame(index=df.index)
    feat["return_1d"] = close.pct_change()
    feat["sma_20_ratio"] = close / sma(close, 20)
    feat["sma_50_ratio"] = close / sma(close, 50)
    feat["rsi_14"] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    feat["macd_hist"] = hist
    upper, mid, lower = bollinger_bands(close)
    feat["bb_position"] = (close - lower) / (upper - lower)
    feat["volatility_10d"] = close.pct_change().rolling(10).std()
    return feat


def predict_next_direction(df: pd.DataFrame) -> dict:
    feat = build_features(df)
    target = (df["Close"].shift(-1) > df["Close"]).astype(int)

    data = feat.copy()
    data["target"] = target
    data = data.dropna()

    if len(data) < 60:
        return {
            "error": "Not enough historical data to train a reliable model (need 60+ clean rows)."
        }

    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    latest_features = feat.iloc[[-1]].fillna(method="ffill")
    proba = model.predict_proba(latest_features)[0]
    prediction = "UP" if proba[1] > proba[0] else "DOWN"
    confidence = round(float(max(proba)) * 100, 1)

    return {
        "prediction": prediction,
        "confidence_pct": confidence,
        "backtested_accuracy_pct": round(float(test_accuracy) * 100, 1),
        "disclaimer": (
            "This is a simple statistical model, not a reliable trading signal. "
            "Daily direction prediction is inherently close to a coin flip. "
            "Backtested accuracy shown reflects out-of-sample test performance only."
        ),
    }