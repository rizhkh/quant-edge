import os
import pandas as pd
import yfinance as yf
from features.engineer import compute_features

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _download(ticker: str, interval: str, period: str = None,
              start: pd.Timestamp = None, target_bars: int = None) -> pd.DataFrame:
    """Download OHLCV from yfinance, with fallback periods when target_bars is set."""
    if start is not None:
        # Incremental fetch from a known start date
        raw = yf.download(ticker, start=start, interval=interval,
                          auto_adjust=True, progress=False)
    elif target_bars:
        raw = None
        for p in ["10y", "3y", "max"]:
            raw = yf.download(ticker, period=p, interval=interval,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw[raw["Close"].notna()]
            if len(raw) >= target_bars or p == "max":
                break
    else:
        raw = yf.download(ticker, period=period, interval=interval,
                          auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    return raw[raw["Close"].notna()]


def fetch_data(ticker: str, period: str, interval: str, cache_path: str,
               target_bars: int = None) -> pd.DataFrame:
    """Fetch OHLCV data, appending only new rows to the local CSV cache.

    - If the cache exists, fetch only dates after the last cached row and append.
    - If the cache does not exist, do a full download and create it.
    """

    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)

    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path, index_col=0, parse_dates=True)

        # Keep only raw OHLCV so derived columns are always recomputed cleanly
        existing_raw = existing[[c for c in _OHLCV if c in existing.columns]].copy()
        last_date = existing_raw.index[-1]

        # Fetch rows newer than the last cached date
        new_raw = _download(ticker, interval, start=last_date)
        new_rows = new_raw[[c for c in _OHLCV if c in new_raw.columns]].copy()
        new_rows.index = new_rows.index.tz_localize(None) if new_rows.index.tz is not None else new_rows.index
        new_rows = new_rows[new_rows.index > last_date]

        if len(new_rows) > 0:
            combined = pd.concat([existing_raw, new_rows])
            combined.to_csv(cache_path)
            print(f"> Appended {len(new_rows)} new bar(s) to {cache_path}  "
                  f"(total {len(combined)} bars)")
            df = combined
        else:
            print(f"> Cache up to date — loaded {len(existing_raw)} bars of {ticker} "
                  f"from cache ({cache_path})")
            df = existing_raw
    else:
        raw = _download(ticker, interval, period=period, target_bars=target_bars)
        df = raw[[c for c in _OHLCV if c in raw.columns]].copy()
        df.to_csv(cache_path)
        print(f"> Cached {len(df)} bars to {cache_path}")

    df = compute_features(df)

    # Flag extreme single-day moves (potential data errors / splits)
    extreme = df["return"].abs() > 0.50
    if extreme.any():
        print(f"  WARNING: {extreme.sum()} bar(s) with >50% single-day move detected "
              f"— check for splits/errors:")
        print(df.loc[extreme, ["Close", "return"]].to_string())

    start_date = df.index[0].date()
    end_date   = df.index[-1].date()
    print(f"> {len(df)} bars of {ticker} ({start_date} to {end_date})")
    print(f"> Current close: ${df['Close'].iloc[-1]:.2f}")

    return df
