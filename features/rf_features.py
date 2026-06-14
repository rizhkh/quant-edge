"""
Jansen-style feature engineering for Random Forest.

Includes all [BOTH] features shared with XGBoost, plus [RF]-specific features:
ATR ratio, dollar volume, price-to-SMA ratios, Bollinger %B, Stochastic %K/%D.

Random Forest favors bounded/normalized features (BB position, stochastic %K,
52wk high ratio) and handles these cleanly without needing log transforms.

Cross-sectional normalization approximated via 252-bar rolling z-score (single-stock).
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(s: pd.Series, window: int = 252) -> pd.Series:
    mean = s.rolling(window).mean()
    std  = s.rolling(window).std()
    return (s - mean) / (std + 1e-10)


def _rsi_raw(close: pd.Series, period: int) -> pd.Series:
    """RSI in raw avg_gain/avg_loss ratio form — not 0-100 scaled."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return avg_gain / (avg_loss + 1e-10)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    """Stochastic %K and %D — price position within recent range, bounded 0-1."""
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    k  = (close - ll) / (hh - ll + 1e-10)
    k_smooth = k.rolling(smooth_k).mean()
    d_smooth = k_smooth.rolling(smooth_d).mean()
    return k_smooth, d_smooth


def _rolling_beta(asset_ret: pd.Series, spy_ret: pd.Series, window: int = 60) -> pd.Series:
    cov = asset_ret.rolling(window).cov(spy_ret)
    var = spy_ret.rolling(window).var()
    return cov / (var + 1e-10)


def _fetch_spy(start, end) -> pd.Series | None:
    try:
        import yfinance as yf
        raw = yf.download("SPY", start=start, end=end,
                          interval="1d", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw["Close"].dropna() if "Close" in raw.columns else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_rf_features(df: pd.DataFrame, use_spy: bool = True) -> pd.DataFrame:
    """
    Build Jansen-style feature matrix for Random Forest.

    [BOTH] shared with XGBoost:
      Normalized returns 1/5/21/63d, momentum lags t-1/5/21,
      realized vol 5/21/63d, beta vs SPY, volume trend,
      52-week high ratio, RSI(14)/RSI(21), RS vs SPY

    [RF] specific:
      ATR ratio, log dollar volume, price/SMA(50), price/SMA(200),
      Bollinger Band %B, Stochastic %K and %D

    Skipped (no data source): log market cap, sector dummies.
    """
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    ret    = close.pct_change()

    feat = pd.DataFrame(index=df.index)

    # ------------------------------------------------------------------
    # Return-based [BOTH]
    # ------------------------------------------------------------------
    for n in [1, 5, 21, 63]:
        feat[f"ret_{n}d_z"] = _rolling_zscore(close.pct_change(n))

    for lag in [1, 5, 21]:
        feat[f"ret_lag_{lag}"] = ret.shift(lag)

    # ------------------------------------------------------------------
    # Volatility [BOTH] + [RF]
    # ------------------------------------------------------------------
    for n in [5, 21, 63]:
        rvol = ret.rolling(n).std() * np.sqrt(252)
        feat[f"rvol_{n}d_z"] = _rolling_zscore(rvol)

    # [RF] ATR ratio — normalized intraday range proxy
    feat["atr_ratio"] = _atr(high, low, close, 14) / (close + 1e-10)

    # ------------------------------------------------------------------
    # Volume [BOTH] + [RF]
    # ------------------------------------------------------------------
    feat["vol_trend"] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-10)

    # [RF] Dollar volume — Jansen's preferred liquidity proxy
    dollar_vol = close * volume
    feat["log_dollar_vol"] = np.log(dollar_vol.rolling(21).mean() + 1)

    # ------------------------------------------------------------------
    # Price-level & trend [BOTH] + [RF]
    # ------------------------------------------------------------------
    feat["high_52w_ratio"] = close / (close.rolling(252).max() + 1e-10)

    # [RF] Price-to-SMA ratios — normalized distance from trend
    feat["price_sma50_ratio"]  = close / (close.rolling(50).mean()  + 1e-10)
    feat["price_sma200_ratio"] = close / (close.rolling(200).mean() + 1e-10)

    # ------------------------------------------------------------------
    # Technical [BOTH] + [RF]
    # ------------------------------------------------------------------
    feat["rsi_14"] = _rsi_raw(close, 14)
    feat["rsi_21"] = _rsi_raw(close, 21)

    # [RF] Bollinger Band %B — bounded 0-1, mean reversion signal
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # [RF] Stochastic %K and %D — bounded 0-1
    stoch_k, stoch_d = _stochastic(high, low, close)
    feat["stoch_k"] = stoch_k
    feat["stoch_d"] = stoch_d

    # ------------------------------------------------------------------
    # SPY-based [BOTH]
    # ------------------------------------------------------------------
    if use_spy:
        spy_s = _fetch_spy(df.index[0], df.index[-1] + pd.Timedelta(days=5))
        if spy_s is not None:
            spy_aligned = spy_s.reindex(df.index, method="ffill")
            spy_ret     = spy_aligned.pct_change()

            feat["beta_60d"] = _rolling_beta(ret, spy_ret, window=60)
            feat["rs_21d"]   = close.pct_change(21) - spy_aligned.pct_change(21)
            feat["rs_63d"]   = close.pct_change(63) - spy_aligned.pct_change(63)

    return feat
