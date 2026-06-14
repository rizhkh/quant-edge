"""
Run the k-NN weekly forecast for every ticker in TICKER_ALL and print
those with >= 60% probability of price being above current close on each
trading day of the upcoming week.
"""

import io
import sys
import warnings
import pandas as pd
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

import config as cfg
from data.fetcher   import fetch_data
from forecaster.knn import run_knn_forecast

TICKER_ALL = ["HUT", "KKRNF", "MRAM", "PATH", "IREN", "AAOI", "MU", "AMD", "QBTS", "ARM"]
BULLISH_THRESHOLD = 0.60


def upcoming_week_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    today = pd.Timestamp.now().normalize()
    # If weekend, next Monday; else next Monday (start of upcoming week)
    days_to_monday = (7 - today.weekday()) % 7
    if days_to_monday == 0:
        days_to_monday = 7  # already Monday → go to next Monday
    monday = today + pd.Timedelta(days=days_to_monday)
    friday = monday + pd.Timedelta(days=4)
    return monday, friday


def build_config(ticker: str) -> dict:
    return {
        "TICKER":            ticker,
        "PERIOD":            cfg.PERIOD,
        "INTERVAL":          cfg.INTERVAL,
        "CACHE_PATH":        f"data/{ticker}_max.csv",
        "WINDOW_LEN":        cfg.WINDOW_LEN,
        "FORECAST_LEN":      cfg.FORECAST_LEN,
        "BARS_BACK":         cfg.BARS_BACK,
        "SIMILARITY_METHOD": cfg.SIMILARITY_METHOD,
        "INPUT_TYPE":        cfg.INPUT_TYPE,
        "K":                 cfg.K,
        "MIN_GAP":           cfg.MIN_GAP,
        "CONFIDENCE_BANDS":  cfg.CONFIDENCE_BANDS,
        "SHOW_ALL_PATHS":    cfg.SHOW_ALL_PATHS,
        "SHOW_ZIGZAG":       cfg.SHOW_ZIGZAG,
        "ZIGZAG_LEGS":       cfg.ZIGZAG_LEGS,
        "FEATURE_COLS":      cfg.FEATURE_COLS,
    }


def prob_up_per_day(all_paths: np.ndarray, current_close: float,
                    forecast_dates: pd.DatetimeIndex,
                    week_start: pd.Timestamp, week_end: pd.Timestamp) -> dict:
    """Return {date: p_up} for trading days in [week_start, week_end]."""
    result = {}
    for i, date in enumerate(forecast_dates):
        d = date.normalize()
        if week_start <= d <= week_end:
            prices_on_day = all_paths[:, i]
            p_up = float(np.mean(prices_on_day > current_close))
            result[d] = p_up
    return result


def main() -> None:
    monday, friday = upcoming_week_dates()
    print(f"\nUpcoming week: {monday.date()} (Mon) → {friday.date()} (Fri)")
    print(f"Bullish threshold: {BULLISH_THRESHOLD*100:.0f}%  (price > current close)\n")
    print(f"Tickers: {', '.join(TICKER_ALL)}\n")
    print("=" * 70)

    bullish_tickers: list[dict] = []

    for ticker in TICKER_ALL:
        sys.stdout.write(f"  {ticker:<8} fetching... ")
        sys.stdout.flush()
        config = build_config(ticker)

        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = fetch_data(
                        ticker,
                        config["PERIOD"],
                        config["INTERVAL"],
                        config["CACHE_PATH"],
                        target_bars=5000,
                    )
        except Exception as e:
            print(f"SKIP (fetch error: {str(e)[:50]})")
            continue

        current_close = float(df["Close"].iloc[-1])

        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    knn = run_knn_forecast(df, config)
        except Exception as e:
            print(f"SKIP (forecast error: {str(e)[:50]})")
            continue

        all_paths      = knn["all_paths"]          # (K, FORECAST_LEN)
        forecast_dates = knn["forecast_cone"]["date"]

        day_probs = prob_up_per_day(all_paths, current_close, forecast_dates, monday, friday)

        if not day_probs:
            print("SKIP (no forecast dates in target week)")
            continue

        avg_p_up = float(np.mean(list(day_probs.values())))
        days_above = sum(1 for p in day_probs.values() if p >= BULLISH_THRESHOLD)
        total_days = len(day_probs)

        print(f"avg P(up)={avg_p_up*100:.1f}%  days≥60%: {days_above}/{total_days}  close=${current_close:.2f}")

        if avg_p_up >= BULLISH_THRESHOLD:
            bullish_tickers.append({
                "ticker":        ticker,
                "current_close": current_close,
                "avg_p_up":      avg_p_up,
                "days_above":    days_above,
                "total_days":    total_days,
                "day_probs":     day_probs,
            })

    print()
    print("=" * 70)
    if not bullish_tickers:
        print("No tickers met the ≥60% bullish threshold for the upcoming week.")
        return

    bullish_tickers.sort(key=lambda x: x["avg_p_up"], reverse=True)

    print(f"BULLISH TICKERS (avg P(up) >= 60%) — week of {monday.date()}")
    print("=" * 70)
    print(f"{'Ticker':<8}  {'Close':>8}  {'Avg P(up)':>10}  {'Days ≥60%':<12}  Per-Day Breakdown")
    print("-" * 70)

    for t in bullish_tickers:
        day_str = "  ".join(
            f"{d.strftime('%a')} {p*100:.0f}%"
            for d, p in sorted(t["day_probs"].items())
        )
        print(
            f"{t['ticker']:<8}  ${t['current_close']:>7.2f}  "
            f"{t['avg_p_up']*100:>9.1f}%  "
            f"{t['days_above']}/{t['total_days']} days     "
            f"{day_str}"
        )

    print()


if __name__ == "__main__":
    main()
