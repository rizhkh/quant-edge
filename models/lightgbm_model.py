import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from features.lgbm_features import build_lgbm_features

_MILESTONE_DAYS = [1, 5, 10, 20, 30, 40]

_DEFAULT_PARAMS = {
    "n_estimators":      500,
    "max_depth":         6,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_samples": 10,
}


def load_best_lgbm_params(ticker: str) -> dict:
    """Read LightGBM hyperparams from params/best_params_lgbm.yaml."""
    from utils.params_io import load_lgbm_params
    return load_lgbm_params(ticker)


def _train_quantile_models(X: np.ndarray, y: np.ndarray,
                           quantiles: list, params: dict = None) -> dict:
    p = {**_DEFAULT_PARAMS, **(params or {})}
    models = {}
    for q in quantiles:
        m = lgb.LGBMRegressor(
            objective        = "quantile",
            alpha            = q,
            n_estimators     = p["n_estimators"],
            max_depth        = p["max_depth"],
            learning_rate    = p["learning_rate"],
            num_leaves       = p["num_leaves"],
            subsample        = p["subsample"],
            colsample_bytree = p["colsample_bytree"],
            min_child_samples= p["min_child_samples"],
            random_state     = 42,
            verbose          = -1,
        )
        m.fit(X, y)
        models[q] = m
    return models


def run_lgbm_forecast(df: pd.DataFrame, config: dict, params: dict = None) -> dict:
    forecast_len = config["FORECAST_LEN"]
    ticker       = config.get("TICKER", "")
    conf_bands   = config.get("CONFIDENCE_BANDS", [20, 60, 90])
    use_vix      = config.get("LGBM_USE_VIX", True)
    use_spy      = config.get("LGBM_USE_SPY", True)
    lookback     = config.get("ML_TRAINING_LOOKBACK_BARS", 1500)

    quantiles = sorted(set([(100 - prob) / 100 for prob in conf_bands] + [0.50]))

    if params is None:
        saved = load_best_lgbm_params(ticker)
        if saved:
            print(f"> Loaded LightGBM best params from output/{ticker}/best_params_lgbm.txt")
        params = saved or {}

    X_all = build_lgbm_features(df, use_vix=use_vix, use_spy=use_spy)
    X_current = X_all.iloc[[-1]]  # keep as DataFrame so feature names match fit

    current_close = float(df["Close"].iloc[-1])
    forecast_dates = pd.bdate_range(
        start   = df.index[-1] + pd.Timedelta(days=1),
        periods = forecast_len,
    )

    milestone_days  = [d for d in _MILESTONE_DAYS if d <= forecast_len]
    milestone_preds: dict[int, dict] = {}
    n_train = 0

    for day in milestone_days:
        fwd_return = df["Close"].pct_change(day).shift(-day)
        combined   = X_all.copy()
        combined["_y"] = fwd_return
        combined   = combined.dropna()
        if lookback and len(combined) > lookback:
            combined = combined.iloc[-lookback:]

        if len(combined) < 60:
            continue

        X_train = combined.drop(columns=["_y"])  # keep DataFrame — consistent with X_current
        y_train = combined["_y"].values
        n_train = len(X_train)

        models = _train_quantile_models(X_train, y_train, quantiles, params)
        raw = {q: float(current_close * (1 + models[q].predict(X_current)[0]))
               for q in quantiles}
        # Enforce monotonicity: higher quantile level → higher price (prevents quantile crossing)
        sorted_qs     = sorted(raw.keys())
        sorted_prices = sorted(raw[q] for q in sorted_qs)
        milestone_preds[day] = dict(zip(sorted_qs, sorted_prices))

    if not milestone_preds:
        raise RuntimeError("LightGBM: insufficient training data to produce any forecast.")

    days_avail = sorted(milestone_preds.keys())
    all_days   = np.arange(1, forecast_len + 1)

    cone_data = {"day": all_days, "date": forecast_dates}
    for q in quantiles:
        prob = int(round(100 - q * 100))
        cone_data[f"p{prob}"] = np.interp(
            all_days, days_avail, [milestone_preds[d][q] for d in days_avail]
        )

    forecast_cone = pd.DataFrame(cone_data)

    outermost_low  = min(quantiles)
    outermost_high = max(quantiles)
    forecast_cone["low"]    = forecast_cone[f"p{int(round(100 - outermost_low  * 100))}"]
    forecast_cone["median"] = forecast_cone["p50"]
    forecast_cone["high"]   = forecast_cone[f"p{int(round(100 - outermost_high * 100))}"]

    # PDCP: p-band prices only if above current_close
    for prob in conf_bands:
        col = f"p{prob}"
        if col in forecast_cone.columns:
            forecast_cone[f"pdcp{prob}"] = forecast_cone[col].where(
                forecast_cone[col] > current_close, other=np.nan
            )

    median_vals = forecast_cone["median"].values
    slope = (median_vals[-1] - median_vals[0]) / (median_vals[0] + 1e-10)
    bias  = "BULLISH" if slope > 0.02 else ("BEARISH" if slope < -0.02 else "CHOPPY")

    n_features = X_all.shape[1]

    # Console output
    print()
    print("=" * 44)
    print("[LightGBM FORECAST]")
    print("=" * 44)
    print(f"Training samples : {n_train}")
    print(f"Input features   : {n_features}")
    prob_cols = sorted(
        [c for c in forecast_cone.columns if c.startswith("p") and c[1:].isdigit()],
        key=lambda c: -int(c[1:]),
    )
    print(f"Quantiles        : {[f'p{c[1:]}' for c in prob_cols]} (% chance price closes above)")
    print()
    print("Forecast Cone (% = chance price closes above this level):")
    header = "  ".join(f"{c}({c[1:]}%)".rjust(12) for c in prob_cols)
    print(f"{'Day':<5}  {'Date':<12}  {header}")
    print(f"{'-'*5}  {'-'*12}  " + "  ".join(["-"*12] * len(prob_cols)))
    for _, row in forecast_cone.iterrows():
        if int(row["day"]) in {1, 5, 10, 20, 30}:
            vals = "  ".join(f"${row[c]:>10.2f}" for c in prob_cols)
            print(f"{int(row['day']):<5}  {str(row['date'].date()):<12}  {vals}")
    print()
    print(f"Directional Bias : {bias}")
    print("=" * 44)

    return {
        "model":           "lightgbm",
        "forecast_cone":   forecast_cone,
        "bias":            bias,
        "milestone_preds": milestone_preds,
        "n_train_samples": n_train,
        "n_features":      n_features,
    }
