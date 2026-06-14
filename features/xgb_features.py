"""
XGBoost feature engineering — v2 (13 features).

Designed via feature audit (see ML_FEATURE_AUDIT_PLAN.txt). Drops the
generic Jansen tech-indicator pile and instead emphasises:
  - Compact momentum (2 z-scored returns)
  - Compact volatility (1 rvol z-score + regime ratio)
  - Compact trend (RSI(14), MACD norm, 52w ratio)
  - Mean-reversion (Bollinger %B) — interacts with RSI
  - Sector COHORT (per-ticker sector ETF return + relative strength)
  - Macro regime (VIX bucket — categorical)
  - Catalyst proximity (days_to_earnings)

13 features. With ~1500 training bars this leaves ~115 bars per feature.
"""
import numpy as np
import pandas as pd


def _rolling_zscore(s: pd.Series, window: int = 252) -> pd.Series:
    mean = s.rolling(window).mean()
    std  = s.rolling(window).std()
    return (s - mean) / (std + 1e-10)


def _rsi_raw(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI returned as 0–100 scaled (standard form)."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd_norm(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram normalized by price."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal, adjust=False).mean()
    return (macd - sig) / (close + 1e-10)


def _bb_pct_b(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.Series:
    """Bollinger Band %B — 0 at lower band, 1 at upper, can extend beyond."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + k * std
    lower = mid - k * std
    return (close - lower) / (upper - lower + 1e-10)


# Module-level caches — populated once per process. Critical for backtest /
# param_sweep performance: without these, yfinance is hit on every feature
# build (~1600 calls per ticker per sweep). With caches: 1 call per symbol.
_CLOSE_CACHE: dict = {}
_EARNINGS_CACHE: dict = {}


def _fetch_yf_close(symbol: str, start=None, end=None) -> pd.Series | None:
    """
    Fetch yfinance closes for `symbol`. Cached per symbol at module level.
    `start`/`end` are accepted for API back-compat but ignored — we always
    fetch the maximum-available history once, then callers slice locally.
    """
    if symbol in _CLOSE_CACHE:
        return _CLOSE_CACHE[symbol]
    try:
        import yfinance as yf
        raw = yf.download(symbol, period="max",
                          interval="1d", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        s = raw["Close"].dropna() if "Close" in raw.columns else None
        if s is not None and hasattr(s.index, "tz") and s.index.tz is not None:
            s = s.tz_localize(None)
        _CLOSE_CACHE[symbol] = s
        return s
    except Exception:
        _CLOSE_CACHE[symbol] = None
        return None


def _vix_bucket(vix: pd.Series) -> pd.Series:
    """Categorical VIX regime: 0 = LOW (<15), 1 = MID (15-25), 2 = HIGH (>25)."""
    return pd.Series(
        np.where(vix < 15, 0, np.where(vix < 25, 1, 2)).astype(float),
        index=vix.index,
    )


def _days_to_earnings(ticker: str, index: pd.DatetimeIndex) -> pd.Series:
    """
    For each date in `index`, days until the next earnings report.
    Falls back to a 99-day constant if yfinance has no data — model treats
    that as "no catalyst in sight." Earnings date is cached per ticker.
    """
    fallback = pd.Series(99.0, index=index)
    if ticker in _EARNINGS_CACHE:
        next_date = _EARNINGS_CACHE[ticker]
        if next_date is None:
            return fallback
        days = (next_date - index).days.astype(float)
        days = np.where(days < 0, 99.0, days)
        return pd.Series(np.clip(days, 0, 99), index=index)
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        cal = getattr(t, "calendar", None)
        if cal is None:
            _EARNINGS_CACHE[ticker] = None
            return fallback
        # yfinance .calendar can be DataFrame (legacy) or dict (newer versions)
        next_date = None
        if isinstance(cal, dict):
            v = cal.get("Earnings Date")
            if isinstance(v, list) and v:
                next_date = pd.to_datetime(v[0]).tz_localize(None)
            elif v is not None:
                next_date = pd.to_datetime(v).tz_localize(None)
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                next_date = pd.to_datetime(cal.loc["Earnings Date"].iloc[0]).tz_localize(None)
        if next_date is None:
            _EARNINGS_CACHE[ticker] = None
            return fallback
        _EARNINGS_CACHE[ticker] = next_date
        days = (next_date - index).days.astype(float)
        days = np.where(days < 0, 99.0, days)
        return pd.Series(np.clip(days, 0, 99), index=index)
    except Exception:
        _EARNINGS_CACHE[ticker] = None
        return fallback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_xgb_features(df: pd.DataFrame,
                       ticker: str | None = None,
                       use_sector: bool = True,
                       use_vix: bool = True,
                       use_earnings: bool = True,
                       sector_map: dict | None = None) -> pd.DataFrame:
    """
    Build the 13-feature matrix for XGBoost.

    Features:
      Returns       : ret_5d_z, ret_21d_z
      Volatility    : rvol_21d_z, vol_regime
      Trend         : rsi_14, macd_norm, high_52w_ratio
      Mean-reversion: bb_pct_b
      Cohort        : sector_ret_5d, sector_ret_21d, rel_strength_vs_sector
      Regime        : vix_bucket
      Catalyst      : days_to_earnings
    """
    close  = df["Close"]
    ret    = close.pct_change()

    feat = pd.DataFrame(index=df.index)

    # Returns (z-scored over 252d)
    feat["ret_5d_z"]  = _rolling_zscore(close.pct_change(5))
    feat["ret_21d_z"] = _rolling_zscore(close.pct_change(21))

    # Volatility
    rvol_21 = ret.rolling(21).std() * np.sqrt(252)
    feat["rvol_21d_z"] = _rolling_zscore(rvol_21)
    feat["vol_regime"] = ret.rolling(5).std() / (ret.rolling(63).std() + 1e-10)

    # Trend
    feat["rsi_14"]         = _rsi_raw(close, 14)
    feat["macd_norm"]      = _macd_norm(close)
    feat["high_52w_ratio"] = close / (close.rolling(252).max() + 1e-10)

    # Mean-reversion
    feat["bb_pct_b"] = _bb_pct_b(close, 20, 2.0)

    # Cohort — sector ETF
    sector_etf = None
    if use_sector and ticker:
        sm = sector_map or {}
        sector_etf = sm.get(ticker, "SPY")
        sector_s = _fetch_yf_close(sector_etf, df.index[0], df.index[-1] + pd.Timedelta(days=5))
        if sector_s is not None and len(sector_s) > 30:
            sector_aligned = sector_s.reindex(df.index, method="ffill")
            sector_ret_5d  = sector_aligned.pct_change(5)
            sector_ret_21d = sector_aligned.pct_change(21)
            feat["sector_ret_5d"]          = sector_ret_5d
            feat["sector_ret_21d"]         = sector_ret_21d
            feat["rel_strength_vs_sector"] = close.pct_change(5) - sector_ret_5d
        else:
            # Cohort fetch failed — fill with zeros (neutral signal)
            feat["sector_ret_5d"]          = 0.0
            feat["sector_ret_21d"]         = 0.0
            feat["rel_strength_vs_sector"] = 0.0

    # Regime — VIX bucket
    if use_vix:
        vix_s = _fetch_yf_close("^VIX", df.index[0], df.index[-1] + pd.Timedelta(days=5))
        if vix_s is not None and len(vix_s) > 30:
            vix_aligned = vix_s.reindex(df.index, method="ffill")
            feat["vix_bucket"] = _vix_bucket(vix_aligned)
        else:
            feat["vix_bucket"] = 1.0  # default MID if fetch fails

    # Catalyst — days to earnings
    if use_earnings and ticker:
        feat["days_to_earnings"] = _days_to_earnings(ticker, df.index)

    return feat
