import pandas as pd
import numpy as np
from similarity.search import find_top_k_matches


def run_knn_forecast(df: pd.DataFrame, config: dict) -> dict:
    method       = config["SIMILARITY_METHOD"]
    window_len   = config["WINDOW_LEN"]
    forecast_len = config["FORECAST_LEN"]
    input_type   = config["INPUT_TYPE"]
    k            = config["K"]
    min_gap      = config["MIN_GAP"]
    bands        = config["CONFIDENCE_BANDS"]

    bars_back = config.get("BARS_BACK", 750)
    if method == "kendall" and bars_back > 500:
        bars_back = 500
        print(f"  WARNING: kendall is O(n²) per window — capping search to 500 bars")

    feature_cols = config.get("FEATURE_COLS")
    if feature_cols:
        series = df[feature_cols].values          # 2D: (N, F)
    elif input_type == "pct_change":
        series = df["return"].values
    else:
        series = df["Close"].values

    # Limit search space to bars_back (keeps the query window + runway intact)
    max_len = bars_back + window_len + forecast_len
    if len(series) > max_len:
        series = series[-max_len:]

    matches = find_top_k_matches(series, window_len, forecast_len, method, k, min_gap)

    current_close = float(df["Close"].iloc[-1])
    paths         = []
    analog_info   = []

    for rank, (idx, score) in enumerate(matches, start=1):
        start_ts = df.index[idx]
        end_ts   = df.index[idx + window_len - 1]
        analog_returns = df["return"].values[idx + window_len : idx + window_len + forecast_len]

        path = [current_close]
        for r in analog_returns:
            path.append(path[-1] * (1 + r))
        path = path[1:]

        paths.append(path)
        analog_info.append({
            "rank":        rank,
            "idx":         idx,
            "score":       score,
            "match_start": start_ts,
            "match_end":   end_ts,
        })

    all_paths = np.array(paths)  # shape (K, FORECAST_LEN)

    forecast_dates = pd.bdate_range(
        start   = df.index[-1] + pd.Timedelta(days=1),
        periods = forecast_len,
    )

    # bands = probability values (e.g. 90 means 90% chance price closes above this level)
    # numpy percentile = 100 - probability
    cone_data = {"day": range(1, forecast_len + 1), "date": forecast_dates}
    for prob in bands:
        cone_data[f"p{prob}"] = np.percentile(all_paths, 100 - prob, axis=0)
    if 50 not in bands:
        cone_data["p50"] = np.percentile(all_paths, 50, axis=0)

    forecast_cone = pd.DataFrame(cone_data)

    # PDCP: p-band prices filtered to only those above current_close (prev day close)
    # null means that confidence level has no upside target above yesterday's close
    for prob in bands:
        col = f"p{prob}"
        forecast_cone[f"pdcp{prob}"] = forecast_cone[col].where(forecast_cone[col] > current_close, other=np.nan)

    # low/median/high aliases for backward compatibility
    # p90 = lowest price (90% likely), p50 = median, p10 = highest price (10% likely)
    forecast_cone["low"]    = forecast_cone["p90"] if 90 in bands else np.percentile(all_paths, 10, axis=0)
    forecast_cone["median"] = forecast_cone["p50"]
    forecast_cone["high"]   = forecast_cone["p10"] if 10 in bands else np.percentile(all_paths, 90, axis=0)

    median_band = forecast_cone["median"].values
    slope = (median_band[-1] - median_band[0]) / median_band[0]
    if slope > 0.02:
        bias = "BULLISH"
    elif slope < -0.02:
        bias = "BEARISH"
    else:
        bias = "CHOPPY"

    # --- Console output ---
    print()
    print("=" * 44)
    print(f"[k-NN FORECAST] Method: {method}  |  k={k}")
    print("=" * 44)
    print(f"Top {k} Analog Matches:")
    print(f"{'Rank':<6}  {'Score':<7}  {'Match Period'}")
    print(f"{'-'*6}  {'-'*7}  {'-'*27}")
    for a in analog_info:
        print(f"{a['rank']:<6}  {a['score']:<7.3f}  {str(a['match_start'].date())} to {str(a['match_end'].date())}")

    sorted_bands = sorted(bands, reverse=True)  # p90 first (highest prob = lowest price)
    print()
    print("Forecast Cone (% = chance price closes above this level):")
    header = "  ".join(f"p{p}({p}%)".rjust(12) for p in sorted_bands)
    print(f"{'Day':<5}  {'Date':<12}  {header}")
    print(f"{'-'*5}  {'-'*12}  " + "  ".join(["-"*12] * len(sorted_bands)))
    milestones = {1, 5, 10, 20, 30}
    for _, row in forecast_cone.iterrows():
        if int(row["day"]) in milestones:
            vals = "  ".join(f"${row[f'p{p}']:>10.2f}" for p in sorted_bands)
            print(f"{int(row['day']):<5}  {str(row['date'].date()):<12}  {vals}")

    print()
    print(f"Directional Bias: {bias}")
    print("=" * 44)

    return {
        "k":             k,
        "method":        method,
        "analogs":       analog_info,
        "all_paths":     all_paths,
        "forecast_cone": forecast_cone,
        "bias":          bias,
    }
