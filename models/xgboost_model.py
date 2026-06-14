import os
import numpy as np
import pandas as pd
import xgboost as xgb
from features.xgb_features import build_xgb_features

# Days at which we train individual models; remaining days are interpolated
_MILESTONE_DAYS = [1, 5, 10, 20, 30, 40]
_QUANTILES      = [0.10, 0.50, 0.90]  # fallback; overridden by CONFIDENCE_BANDS in config

_DEFAULT_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
}


def load_best_xgb_params(ticker: str) -> dict:
    """Read XGBoost hyperparams from params/best_params_xgb.yaml."""
    from utils.params_io import load_xgb_params
    return load_xgb_params(ticker)


def _build_feature_matrix(df: pd.DataFrame, feature_cols: list, window_len: int) -> pd.DataFrame:
    """
    Build XGBoost input features for each bar:
      - Current bar values for each feature (spot signal)
      - Rolling mean over window_len bars (regime context)
      - Rolling std over window_len bars  (regime volatility)

    This gives XGBoost the same window-level context that k-NN sees
    when it compares 20-bar patterns.
    """
    parts = [df[feature_cols].copy()]
    for col in feature_cols:
        parts.append(df[col].rolling(window_len).mean().rename(f"{col}_rmean"))
        parts.append(df[col].rolling(window_len).std().rename(f"{col}_rstd"))
    return pd.concat(parts, axis=1)


def _train_quantile_models(X: np.ndarray, y: np.ndarray, quantiles: list, params: dict = None) -> dict:
    """Train one XGBoost quantile regressor per quantile level."""
    p = {**_DEFAULT_PARAMS, **(params or {})}
    models = {}
    for q in quantiles:
        m = xgb.XGBRegressor(
            objective        = "reg:quantileerror",
            quantile_alpha   = q,
            n_estimators     = p["n_estimators"],
            max_depth        = p["max_depth"],
            learning_rate    = p["learning_rate"],
            subsample        = p["subsample"],
            colsample_bytree = p["colsample_bytree"],
            min_child_weight = p.get("min_child_weight", 5),
            random_state     = 42,
            verbosity        = 0,
        )
        m.fit(X, y)
        models[q] = m
    return models


def run_xgboost_forecast(df: pd.DataFrame, config: dict, params: dict = None) -> dict:
    forecast_len   = config["FORECAST_LEN"]
    ticker         = config.get("TICKER", "")
    conf_bands     = config.get("CONFIDENCE_BANDS", [10, 50, 90])
    use_sector     = config.get("XGB_USE_SECTOR", True)
    use_vix        = config.get("XGB_USE_VIX", True)
    use_earnings   = config.get("XGB_USE_EARNINGS", True)
    sector_map     = config.get("SECTOR_MAP", {})
    lookback       = config.get("ML_TRAINING_LOOKBACK_BARS", 1500)
    quantiles      = sorted(set([(100 - prob) / 100 for prob in conf_bands] + [0.50]))

    # Load saved best params unless caller already passed explicit params
    if params is None:
        saved = load_best_xgb_params(ticker)
        if saved:
            print(f"> Loaded XGBoost best params from output/{ticker}/best_params_xgb.txt  "
                  f"(n_estimators={saved.get('n_estimators')}, "
                  f"max_depth={saved.get('max_depth')}, "
                  f"lr={saved.get('learning_rate')})")
        params = saved or {}

    X_all     = build_xgb_features(df, ticker=ticker,
                                   use_sector=use_sector, use_vix=use_vix,
                                   use_earnings=use_earnings, sector_map=sector_map)
    X_current = X_all.iloc[[-1]].values  # shape (1, n_features)

    current_close = float(df["Close"].iloc[-1])
    forecast_dates = pd.bdate_range(
        start   = df.index[-1] + pd.Timedelta(days=1),
        periods = forecast_len,
    )

    milestone_days = [d for d in _MILESTONE_DAYS if d <= forecast_len]
    milestone_preds: dict[int, dict] = {}
    n_train_samples = 0

    for day in milestone_days:
        # Target: cumulative return from bar t to bar t+day
        fwd_return = df["Close"].pct_change(day).shift(-day)

        combined = X_all.copy()
        combined["_y"] = fwd_return
        combined = combined.dropna()
        if lookback and len(combined) > lookback:
            combined = combined.iloc[-lookback:]

        if len(combined) < 60:
            continue

        X_train = combined.drop(columns=["_y"]).values
        y_train = combined["_y"].values
        n_train_samples = len(X_train)

        models = _train_quantile_models(X_train, y_train, quantiles, params)
        raw = {q: float(current_close * (1 + models[q].predict(X_current)[0]))
               for q in quantiles}
        # Enforce monotonicity: higher quantile level → higher price (prevents quantile crossing)
        sorted_qs     = sorted(raw.keys())
        sorted_prices = sorted(raw[q] for q in sorted_qs)
        milestone_preds[day] = dict(zip(sorted_qs, sorted_prices))

    if not milestone_preds:
        raise RuntimeError("XGBoost: insufficient training data to produce any forecast.")

    # Interpolate across all forecast days from milestone predictions
    days_avail = sorted(milestone_preds.keys())
    all_days   = np.arange(1, forecast_len + 1)

    cone_data = {"day": all_days, "date": forecast_dates}
    for q in quantiles:
        prob = int(round(100 - q * 100))
        cone_data[f"p{prob}"] = np.interp(all_days, days_avail, [milestone_preds[d][q] for d in days_avail])

    forecast_cone = pd.DataFrame(cone_data)

    # low/median/high aliases for backward compatibility
    outermost_low  = min(quantiles)
    outermost_high = max(quantiles)
    forecast_cone["low"]    = forecast_cone[f"p{int(round(100 - outermost_low  * 100))}"]
    forecast_cone["median"] = forecast_cone["p50"]
    forecast_cone["high"]   = forecast_cone[f"p{int(round(100 - outermost_high * 100))}"]

    # PDCP: p-band prices filtered to only those above current_close (prev day close)
    for prob in conf_bands:
        col = f"p{prob}"
        if col in forecast_cone.columns:
            forecast_cone[f"pdcp{prob}"] = forecast_cone[col].where(forecast_cone[col] > current_close, other=np.nan)

    median_vals = forecast_cone["median"].values
    slope = (median_vals[-1] - median_vals[0]) / median_vals[0]
    bias  = "BULLISH" if slope > 0.02 else ("BEARISH" if slope < -0.02 else "CHOPPY")

    n_features = X_all.shape[1]

    # --- Console output ---
    print()
    print("=" * 44)
    print("[XGBoost FORECAST]")
    print("=" * 44)
    print(f"Training samples : {n_train_samples}")
    print(f"Input features   : {n_features}  (Jansen-style: returns, vol, volume, trend, technical, SPY)")
    prob_cols    = sorted([c for c in forecast_cone.columns if c.startswith("p") and c[1:].isdigit()],
                          key=lambda c: -int(c[1:]))
    print(f"Quantiles        : {[f'p{c[1:]}' for c in prob_cols]} (% chance price closes above)")
    print()
    print("Forecast Cone (% = chance price closes above this level):")
    header = "  ".join(f"{c}({c[1:]}%)".rjust(12) for c in prob_cols)
    print(f"{'Day':<5}  {'Date':<12}  {header}")
    print(f"{'-'*5}  {'-'*12}  " + "  ".join(["-"*12] * len(prob_cols)))
    milestones = {1, 5, 10, 20, 30}
    for _, row in forecast_cone.iterrows():
        if int(row["day"]) in milestones:
            vals = "  ".join(f"${row[c]:>10.2f}" for c in prob_cols)
            print(f"{int(row['day']):<5}  {str(row['date'].date()):<12}  {vals}")
    print()
    print(f"Directional Bias : {bias}")
    print("=" * 44)

    return {
        "model":           "xgboost",
        "forecast_cone":   forecast_cone,
        "bias":            bias,
        "milestone_preds": milestone_preds,
        "n_train_samples": n_train_samples,
        "n_features":      n_features,
    }
