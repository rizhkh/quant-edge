import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def _stoch_rsi(close: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    rsi     = _rsi(close, period)
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    stoch   = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
    k       = stoch.rolling(smooth_k).mean()
    d       = k.rolling(smooth_d).mean()
    return k, d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll + 1e-10)


def _ttm_squeeze_momentum(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    bb_mid = close.rolling(period).mean()
    hh     = high.rolling(period).max()
    ll     = low.rolling(period).min()
    delta  = close - ((hh + ll) / 2 + bb_mid) / 2

    def _linreg_last(x):
        if np.isnan(x).any():
            return np.nan
        t = np.arange(len(x))
        slope, intercept = np.polyfit(t, x, 1)
        return slope * (len(x) - 1) + intercept

    return delta.rolling(period).apply(_linreg_last, raw=True)


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 10, multiplier: float = 3.0) -> pd.Series:
    atr_val = _atr(high, low, close, period).values
    hl2     = ((high + low) / 2).values
    c       = close.values
    n       = len(c)

    basic_upper = hl2 + multiplier * atr_val
    basic_lower = hl2 - multiplier * atr_val
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    direction   = np.ones(n)

    for i in range(1, n):
        final_upper[i] = (basic_upper[i]
                          if basic_upper[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]
                          else final_upper[i - 1])
        final_lower[i] = (basic_lower[i]
                          if basic_lower[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]
                          else final_lower[i - 1])
        if direction[i - 1] == -1:
            direction[i] = 1 if c[i] > final_upper[i] else -1
        else:
            direction[i] = -1 if c[i] < final_lower[i] else 1

    return pd.Series(direction, index=close.index)


def _ichimoku(high: pd.Series, low: pd.Series, close: pd.Series):
    tenkan   = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun    = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    above = (close > senkou_a) & (close > senkou_b)
    below = (close < senkou_a) & (close < senkou_b)
    vs_cloud = pd.Series(np.where(above, 1, np.where(below, -1, 0)), index=close.index)
    tk_cross = pd.Series(np.sign((tenkan - kijun).values), index=close.index)
    return vs_cloud, tk_cross


def _vwap_dev(high: pd.Series, low: pd.Series, close: pd.Series,
              volume: pd.Series, period: int = 20) -> pd.Series:
    tp    = (high + low + close) / 3
    vwap  = (tp * volume).rolling(period).sum() / volume.rolling(period).sum()
    return (close - vwap) / (vwap + 1e-10) * 100


def _keltner_width(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 20, multiplier: float = 1.5) -> pd.Series:
    mid   = close.ewm(span=period, adjust=False).mean()
    atr   = _atr(high, low, close, period)
    upper = mid + multiplier * atr
    lower = mid - multiplier * atr
    return (upper - lower) / (mid + 1e-10)


def _obv_slope(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()

    def _slope(x):
        if np.isnan(x).any() or len(x) < 2:
            return np.nan
        return np.polyfit(range(len(x)), x, 1)[0]

    return obv.rolling(period).apply(_slope, raw=True)


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, period: int = 20) -> pd.Series:
    clv = ((close - low) - (high - close)) / (high - low + 1e-10)
    return (clv * volume).rolling(period).sum() / (volume.rolling(period).sum() + 1e-10)


def _fetch_vix_spy(start, end) -> tuple:
    """Fetch VIX and SPY daily closes. Returns (vix_series, spy_series), either may be None."""
    import yfinance as yf
    vix, spy = None, None
    try:
        raw = yf.download(["^VIX", "SPY"], start=start, end=end,
                          interval="1d", progress=False, auto_adjust=True)
        if "Close" in raw.columns:
            closes = raw["Close"]
            if "^VIX" in closes.columns:
                vix = closes["^VIX"].dropna()
            if "SPY" in closes.columns:
                spy = closes["SPY"].dropna()
    except Exception:
        pass
    return vix, spy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_lgbm_features(df: pd.DataFrame, use_vix: bool = True, use_spy: bool = True) -> pd.DataFrame:
    """
    Build feature matrix for LightGBM from a daily OHLCV DataFrame.
    Covers momentum, trend, volatility, volume, lagged returns, and
    optional VIX / SPY regime features.
    All features use only past data — no lookahead.
    """
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    feat = pd.DataFrame(index=df.index)

    # ------------------------------------------------------------------
    # Category 1 — Momentum
    # ------------------------------------------------------------------
    for w in [5, 10, 21, 63]:
        feat[f"ret_{w}d"] = close.pct_change(w)

    rsi = _rsi(close, 14)
    feat["rsi_14"]        = rsi
    feat["rsi_dev_50"]    = rsi - 50

    hist = _macd_histogram(close)
    feat["macd_hist"]        = hist
    feat["macd_sign_change"] = (np.sign(hist) != np.sign(hist.shift(1))).astype(int)

    for w in [5, 10, 21]:
        feat[f"roc_{w}d"] = (close / close.shift(w) - 1) * 100

    feat["williams_r"] = _williams_r(high, low, close, 14)

    stoch_k, stoch_d = _stoch_rsi(close)
    feat["stoch_rsi_k"] = stoch_k
    feat["stoch_rsi_d"] = stoch_d

    feat["ttm_squeeze_mom"] = _ttm_squeeze_momentum(high, low, close, 20)

    # ------------------------------------------------------------------
    # Category 2 — Trend
    # ------------------------------------------------------------------
    for span in [21, 50, 200]:
        ema = close.ewm(span=span, adjust=False).mean()
        feat[f"ema{span}_dev"] = (close - ema) / (ema + 1e-10) * 100

    ema21  = close.ewm(span=21,  adjust=False).mean()
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    feat["ema_stack"] = np.where(
        (ema21 > ema50) & (ema50 > ema200),  1,
        np.where((ema21 < ema50) & (ema50 < ema200), -1, 0)
    )

    feat["vwap_dev"]    = _vwap_dev(high, low, close, volume, 20)
    feat["supertrend"]  = _supertrend(high, low, close, 10, 3.0)

    vs_cloud, tk_cross = _ichimoku(high, low, close)
    feat["ichimoku_vs_cloud"] = vs_cloud
    feat["ichimoku_tk_cross"] = tk_cross

    # ------------------------------------------------------------------
    # Category 3 — Volatility
    # ------------------------------------------------------------------
    atr = _atr(high, low, close, 14)
    feat["atr_14"]  = atr
    feat["atr_pct"] = atr / (close + 1e-10) * 100

    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    feat["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    feat["kc_width"] = _keltner_width(high, low, close, 20, 1.5)
    feat["squeeze"]  = feat["bb_width"] / (feat["kc_width"] + 1e-10)  # <1 = squeeze active

    for w in [10, 21]:
        feat[f"realized_vol_{w}d"] = close.pct_change().rolling(w).std() * np.sqrt(252)

    # ------------------------------------------------------------------
    # Category 4 — Volume
    # ------------------------------------------------------------------
    vol_avg20 = volume.rolling(20).mean()
    feat["rvol"] = volume / (vol_avg20 + 1e-10)

    feat["obv_slope_5d"]  = _obv_slope(close, volume, 5)
    feat["obv_slope_10d"] = _obv_slope(close, volume, 10)

    feat["cmf_20"]    = _cmf(high, low, close, volume, 20)
    feat["vol_force"] = volume * np.sign(close.pct_change())

    rvol_5d  = feat["rvol"].rolling(5).mean()
    rvol_10d = feat["rvol"].rolling(10).mean()
    feat["rvol_trend"] = np.where(rvol_5d > rvol_10d, 1, -1).astype(float)

    # ------------------------------------------------------------------
    # Category 5 — Lagged returns
    # ------------------------------------------------------------------
    daily_ret = close.pct_change()
    for lag in [1, 2, 3, 5]:
        feat[f"ret_lag_{lag}"] = daily_ret.shift(lag)

    # ------------------------------------------------------------------
    # Category 7 — Regime (VIX + SPY)
    # ------------------------------------------------------------------
    if use_vix or use_spy:
        import datetime
        end_date   = df.index[-1] + pd.Timedelta(days=5)
        start_date = df.index[0]
        vix_s, spy_s = _fetch_vix_spy(start_date, end_date)

        if use_vix and vix_s is not None:
            vix_aligned = vix_s.reindex(df.index, method="ffill")
            feat["vix_level"] = vix_aligned.values
            feat["vix_pct"]   = (vix_aligned.rank(pct=True) * 100).values

        if use_spy and spy_s is not None:
            spy_aligned  = spy_s.reindex(df.index, method="ffill")
            spy_ret_20   = spy_aligned.pct_change(20)
            feat["spy_trend"] = np.where(spy_ret_20 > 0.02, 1,
                                np.where(spy_ret_20 < -0.02, -1, 0)).astype(float)

    return feat
