import numpy as np
import pandas as pd

# All feature columns produced by compute_features — referenced by config.FEATURE_COLS
ALL_FEATURE_COLS = [
    "return",
    "hl_range",
    "rel_volume",
    "momentum_z",
    "bb_pct_b",
    "atr_norm",
    "clv",
    "vol_regime",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Base features (existing) ---
    df["return"]     = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["hl_range"]   = (df["High"] - df["Low"]) / df["Close"]
    df["rel_volume"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # --- Phase 1: Alpha Factors ---

    # Momentum Z-score: 20-day rolling mean return normalised by 60-day rolling std
    # Captures trend persistence relative to its own volatility
    df["momentum_z"] = (
        df["return"].rolling(20).mean() / df["return"].rolling(60).std()
    )

    # Bollinger Band %B: where price sits within its 2-std envelope
    # 0 = at lower band, 1 = at upper band, >1 or <0 = outside bands
    sma20      = df["Close"].rolling(20).mean()
    std20      = df["Close"].rolling(20).std()
    band_width = 4 * std20  # upper - lower = 4 * std
    df["bb_pct_b"] = np.where(
        band_width != 0,
        (df["Close"] - (sma20 - 2 * std20)) / band_width,
        0.5,
    )

    # ATR(14) normalised by close — volatility regime, scale-free
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift(1)).abs()
    lpc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["atr_norm"] = tr.rolling(14).mean() / df["Close"]

    # Close Location Value: buy vs. sell pressure within each bar
    # +1 = closed at high (pure buying), -1 = closed at low (pure selling)
    denom = df["High"] - df["Low"]
    df["clv"] = np.where(
        denom != 0,
        (df["Close"] - df["Low"] - (df["High"] - df["Close"])) / denom,
        0.0,
    )

    # Volatility regime: short-term vol relative to medium-term vol
    # >1 = vol expanding (regime shift risk), <1 = vol contracting (trending)
    df["vol_regime"] = (
        df["return"].rolling(5).std() / df["return"].rolling(20).std()
    )

    # Drop rows where any feature is NaN (first ~60 bars due to rolling windows)
    return df.dropna(subset=ALL_FEATURE_COLS)
