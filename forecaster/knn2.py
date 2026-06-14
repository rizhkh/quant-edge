"""
Enhanced kNN forecaster (knn2).

Implements Steps 2-7 from the KNN2_PLAN:
  Step 2 — Feature-weighted distance
  Step 3 — Direction/magnitude separation + conviction score
  Step 4 — Regime-conditional matching (4 buckets)
  Step 5 — Recency-weighted aggregation
  Step 6 — Adaptive k via distance threshold
  Step 7 — Metric ensemble (Manhattan 40% + Spearman 35% + Euclidean 25%)

Output format is identical to existing kNN:
  same p-band / pdcp columns, same DataFrame structure.
  Additional columns: direction, regime, conviction, n_neighbors.
"""

import numpy as np
import pandas as pd

from features.engineer       import compute_features
from features.feature_weights import compute_feature_weights, load_knn2_params
from similarity.search_enhanced import find_top_k_enhanced


# ---------------------------------------------------------------------------
# Ensemble configuration (Step 7)
# ---------------------------------------------------------------------------
_ENSEMBLE_METHODS  = ["manhattan", "spearman", "euclidean"]
_ENSEMBLE_WEIGHTS  = [0.40,        0.35,        0.25]


# ---------------------------------------------------------------------------
# Weighted percentile
# ---------------------------------------------------------------------------

def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute weighted q-th percentile (q in 0–100)."""
    if len(values) == 0:
        return float("nan")
    sorted_idx   = np.argsort(values)
    sorted_vals  = values[sorted_idx]
    sorted_w     = weights[sorted_idx]
    cumsum        = np.cumsum(sorted_w)
    cumsum       /= cumsum[-1]
    return float(np.interp(q / 100.0, cumsum, sorted_vals))


# ---------------------------------------------------------------------------
# Main forecaster
# ---------------------------------------------------------------------------

def run_knn2_forecast(df: pd.DataFrame, config: dict,
                      params: dict | None = None) -> dict:
    """
    Run the full knn2 enhanced forecast pipeline.

    Parameters
    ----------
    df     : OHLCV DataFrame with feature columns already computed
             (output of compute_features). Must include 'return' column.
    config : same config dict as run_knn_forecast
    params : optional override for half_life_days / distance_threshold_pct.
             If None, loaded from feature_weights_knn2.yaml.

    Returns dict with:
      forecast_cone  : pd.DataFrame — same columns as existing kNN cone
                       + regime, conviction, n_neighbors per row
      bias           : str
      regime         : str
      conviction     : float  0–1
      n_neighbors    : int
      feature_weights: np.ndarray
      direction_vote : float  0–1  (>0.5 = UP)
    """
    ticker       = config.get("TICKER", "")
    feature_cols = config.get("FEATURE_COLS")
    window_len   = config["WINDOW_LEN"]
    forecast_len = config["FORECAST_LEN"]
    k_max        = config.get("K", 30)
    min_gap      = config.get("MIN_GAP", 20)
    bands        = config.get("CONFIDENCE_BANDS", [20, 60, 90])
    lookback     = config.get("ML_TRAINING_LOOKBACK_BARS", 1500)

    if not feature_cols:
        raise ValueError("knn2 requires FEATURE_COLS to be set in config.")

    # --- Load runtime params (half-life, distance threshold) ---
    saved = params or load_knn2_params(ticker)
    half_life_days  = int(saved.get("half_life_days", 250))
    distance_pct    = float(saved.get("distance_threshold_pct", 20.0))
    k_min           = max(5, k_max // 4)
    lam             = np.log(2) / half_life_days

    current_close = float(df["Close"].iloc[-1])

    # --- Step 2: Feature weights ---
    print(f"  [knn2] Computing feature weights... ", end="", flush=True)
    feat_weights = compute_feature_weights(df, feature_cols, ticker, lookback)
    print(", ".join(f"{c}={w:.3f}" for c, w in zip(feature_cols, feat_weights)))

    # --- Build feature series ---
    series = df[feature_cols].values   # (N, 8)

    # Forecast date range
    forecast_dates = pd.bdate_range(
        start   = df.index[-1] + pd.Timedelta(days=1),
        periods = forecast_len,
    )

    # -------------------------------------------------------------------
    # Step 7: Ensemble — run enhanced search with 3 methods independently
    # -------------------------------------------------------------------
    method_votes    = []   # direction vote per method
    primary_paths   = None # paths from 1st method for the cone
    primary_weights = None

    for i, method in enumerate(_ENSEMBLE_METHODS):
        matches, meta = find_top_k_enhanced(
            series,
            window_len   = window_len,
            forecast_len = forecast_len,
            method       = method,
            weights      = feat_weights,
            feature_cols = feature_cols,
            distance_pct = distance_pct,
            k_min        = k_min,
            k_max        = k_max,
            min_gap      = min_gap,
        )

        if not matches:
            method_votes.append(0.5)
            continue

        idxs     = [m[0] for m in matches]
        scores   = np.array([m[1] for m in matches], dtype=float)
        age_bars = np.array([m[2] for m in matches], dtype=float)

        # Step 5: Recency weights — older matches discounted
        recency_w = np.exp(-lam * age_bars)

        # Combined weight = score × recency
        dist_w    = np.maximum(scores, 0.0)
        combined  = dist_w * recency_w
        w_sum     = combined.sum()
        combined  = combined / w_sum if w_sum > 0 else np.ones(len(matches)) / len(matches)

        # Build forward price paths for each neighbor
        return_series = df["return"].values
        paths, valid_w = [], []
        for j, idx in enumerate(idxs):
            start = idx + window_len
            end   = start + forecast_len
            if end > len(df):
                continue
            analog_returns = return_series[start:end]
            path = [current_close]
            for r in analog_returns:
                path.append(path[-1] * (1.0 + r))
            paths.append(path[1:])
            valid_w.append(combined[j])

        if not paths:
            method_votes.append(0.5)
            continue

        paths_arr = np.array(paths)
        w_arr     = np.array(valid_w)
        w_arr     = w_arr / w_arr.sum()

        # Step 3: Direction vote at Day 5
        day5_col   = min(4, paths_arr.shape[1] - 1)
        day5_prices = paths_arr[:, day5_col]
        bullish     = (day5_prices > current_close).astype(float)
        vote        = float(np.dot(w_arr, bullish))
        method_votes.append(vote)

        # Keep primary method's data for the price cone
        if primary_paths is None:
            primary_paths   = paths_arr
            primary_weights = w_arr
            primary_meta    = meta

    # -------------------------------------------------------------------
    # Ensemble direction vote (Step 7)
    # -------------------------------------------------------------------
    n_methods = len(_ENSEMBLE_METHODS)
    ew = _ENSEMBLE_WEIGHTS[:n_methods]
    mv = method_votes[:n_methods]
    ensemble_vote = sum(w * v for w, v in zip(ew, mv)) / sum(ew[:len(mv)])

    final_direction = "UP" if ensemble_vote > 0.5 else "DOWN"
    # Vote % for the predicted direction (e.g. 73.5 means 73.5% of neighbors agreed)
    conviction      = round((ensemble_vote if ensemble_vote >= 0.5 else 1 - ensemble_vote) * 100, 1)

    regime      = primary_meta["regime"]      if primary_paths is not None else "UNKNOWN"
    n_neighbors = primary_meta["n_neighbors"] if primary_paths is not None else 0

    # -------------------------------------------------------------------
    # Build forecast cone from primary method paths (Step 3: magnitude)
    # -------------------------------------------------------------------
    if primary_paths is None or len(primary_paths) == 0:
        raise RuntimeError("knn2: no neighbors found — cannot build forecast cone.")

    cone_data = {"day": list(range(1, forecast_len + 1)), "date": forecast_dates}

    for prob in bands:
        q = 100 - prob   # p90 → 10th percentile
        cone_data[f"p{prob}"] = [
            _weighted_percentile(primary_paths[:, d], primary_weights, q)
            for d in range(forecast_len)
        ]

    if 50 not in bands:
        cone_data["p50"] = [
            _weighted_percentile(primary_paths[:, d], primary_weights, 50)
            for d in range(forecast_len)
        ]

    forecast_cone = pd.DataFrame(cone_data)

    # Aliases
    highest_prob = max(bands)
    lowest_prob  = min(bands)
    forecast_cone["low"]    = forecast_cone[f"p{highest_prob}"]   # p90 = floor
    forecast_cone["median"] = forecast_cone["p50"]
    forecast_cone["high"]   = forecast_cone[f"p{lowest_prob}"]    # p20 = ceiling

    # Monotonicity guard (higher quantile → higher price)
    quant_cols = sorted(
        [c for c in forecast_cone.columns if c.startswith("p") and c[1:].isdigit()],
        key=lambda c: int(c[1:]),
    )
    for d_idx in range(len(forecast_cone)):
        prices = [forecast_cone.at[d_idx, c] for c in quant_cols]
        sorted_prices = sorted(prices)
        for ci, col in enumerate(quant_cols):
            forecast_cone.at[d_idx, col] = sorted_prices[ci]

    # PDCP — upside-only targets
    for prob in bands:
        col = f"p{prob}"
        if col in forecast_cone.columns:
            forecast_cone[f"pdcp{prob}"] = forecast_cone[col].where(
                forecast_cone[col] > current_close, other=np.nan
            )

    # Metadata columns
    forecast_cone["direction"]   = final_direction
    forecast_cone["conviction"]  = conviction
    forecast_cone["regime"]      = regime
    forecast_cone["n_neighbors"] = n_neighbors

    # Directional bias from median slope
    median_vals = forecast_cone["median"].values
    slope = (median_vals[-1] - median_vals[0]) / (abs(median_vals[0]) + 1e-10)
    bias  = "BULLISH" if slope > 0.02 else ("BEARISH" if slope < -0.02 else "CHOPPY")

    # Console output
    print()
    print("=" * 50)
    print("[knn2 FORECAST]")
    print("=" * 50)
    print(f"Regime:      {regime}")
    print(f"Neighbors:   {n_neighbors}  (k_max={k_max}, adaptive threshold {distance_pct}th pct)")
    print(f"Vote:        {conviction:.1f}%  ({'STRONG' if conviction >= 70 else 'MODERATE' if conviction >= 60 else 'WEAK'})")
    print(f"Direction:   {final_direction}  (vote={ensemble_vote:.2f})")
    print()

    sorted_probs = sorted(bands, reverse=True)
    header = "  ".join(f"p{p}({p}%)".rjust(12) for p in sorted_probs)
    print(f"{'Day':<5}  {'Date':<12}  {header}")
    print(f"{'-'*5}  {'-'*12}  " + "  ".join(["-" * 12] * len(sorted_probs)))
    for _, row in forecast_cone.iterrows():
        if int(row["day"]) in {1, 5, 10, 20, 30}:
            vals = "  ".join(f"${row[f'p{p}']:>10.2f}" for p in sorted_probs)
            print(f"{int(row['day']):<5}  {str(row['date'].date()):<12}  {vals}")

    print()
    print(f"Directional Bias: {bias}")
    print("=" * 50)

    return {
        "model":           "knn2",
        "forecast_cone":   forecast_cone,
        "bias":            bias,
        "regime":          regime,
        "conviction":      conviction,
        "n_neighbors":     n_neighbors,
        "feature_weights": feat_weights,
        "direction_vote":  ensemble_vote,
        "method_votes":    dict(zip(_ENSEMBLE_METHODS, method_votes)),
    }
