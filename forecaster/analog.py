import pandas as pd
import numpy as np
from similarity.search import find_best_match


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "HIGH"
    if score >= 0.70:
        return "MEDIUM"
    return "LOW — treat with caution"


def run_analog_forecast(df: pd.DataFrame, config: dict) -> dict:
    method       = config["SIMILARITY_METHOD"]
    window_len   = config["WINDOW_LEN"]
    forecast_len = config["FORECAST_LEN"]
    input_type   = config["INPUT_TYPE"]

    feature_cols = config.get("FEATURE_COLS")
    if feature_cols:
        series = df[feature_cols].values          # 2D: (N, F)
    elif input_type == "pct_change":
        series = df["return"].values
    else:
        series = df["Close"].values

    best_idx, best_score = find_best_match(series, window_len, forecast_len, method)

    match_start    = df.index[best_idx]
    match_end      = df.index[best_idx + window_len - 1]
    forecast_start = best_idx + window_len
    forecast_end   = forecast_start + forecast_len

    analog_returns  = df["return"].values[forecast_start:forecast_end]
    analog_segment  = df["Close"].iloc[best_idx : best_idx + window_len]

    current_close   = float(df["Close"].iloc[-1])
    forecast_prices = [current_close]
    for r in analog_returns:
        forecast_prices.append(forecast_prices[-1] * (1 + r))
    forecast_prices = forecast_prices[1:]

    forecast_dates = pd.bdate_range(
        start   = df.index[-1] + pd.Timedelta(days=1),
        periods = forecast_len,
    )

    forecast_df = pd.DataFrame({
        "day":   range(1, forecast_len + 1),
        "date":  forecast_dates,
        "price": forecast_prices,
    })

    # --- Console output ---
    conf  = _confidence_label(best_score)
    bias  = "BULLISH" if forecast_prices[-1] > current_close * 1.02 else (
            "BEARISH" if forecast_prices[-1] < current_close * 0.98 else "CHOPPY")

    print()
    print("=" * 44)
    print(f"[ANALOG FORECAST] Method: {method}")
    print("=" * 44)
    print(f"Best Match: {match_start.date()} to {match_end.date()}")
    print(f"Similarity Score: {best_score:.3f}")
    print(f"Confidence: {conf}")
    print()
    print(f"{'Day':<5}  {'Date':<12}  {'Forecast Price':>14}")
    print(f"{'-'*5}  {'-'*12}  {'-'*14}")
    milestones = {1, 5, 10, 20, 30}
    for _, row in forecast_df.iterrows():
        if int(row["day"]) in milestones:
            print(f"{int(row['day']):<5}  {str(row['date'].date()):<12}  ${row['price']:>13.2f}")
    print("=" * 44)
    print(f"Directional Bias: {bias}")

    return {
        "method":         method,
        "best_score":     best_score,
        "match_start":    match_start,
        "match_end":      match_end,
        "analog_segment": analog_segment,
        "forecast_df":    forecast_df,
    }
