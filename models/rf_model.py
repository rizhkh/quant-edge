import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from features.rf_features import build_rf_features

_MILESTONE_DAYS = [1, 5, 10, 20, 30, 40]

_DEFAULT_PARAMS = {
    "n_estimators":     200,
    "max_depth":        None,
    "min_samples_leaf": 5,
    "max_features":     "sqrt",
    "random_state":     42,
    "n_jobs":           -1,
}


def load_best_rf_params(ticker: str) -> dict:
    """Read RF hyperparams from params/best_params_rf.yaml."""
    from utils.params_io import load_rf_params
    return load_rf_params(ticker)


def _train_rf(X: np.ndarray, y: np.ndarray, params: dict = None) -> RandomForestRegressor:
    p = {**_DEFAULT_PARAMS, **(params or {})}
    rf = RandomForestRegressor(
        n_estimators     = p["n_estimators"],
        max_depth        = p["max_depth"],
        min_samples_leaf = p["min_samples_leaf"],
        max_features     = p["max_features"],
        random_state     = p.get("random_state", 42),
        n_jobs           = p.get("n_jobs", -1),
    )
    rf.fit(X, y)
    return rf


def _rf_quantile_predict(rf: RandomForestRegressor, X_current: np.ndarray,
                         quantiles: list, current_close: float) -> dict:
    """
    Derive quantile predictions from the distribution of individual tree outputs.
    Each tree independently predicts a return; the distribution across all trees
    approximates the predictive uncertainty — the proper RF uncertainty method.
    """
    tree_preds  = np.array([tree.predict(X_current)[0] for tree in rf.estimators_])
    tree_prices = current_close * (1 + tree_preds)
    return {q: float(np.percentile(tree_prices, q * 100)) for q in quantiles}


def run_rf_forecast(df: pd.DataFrame, config: dict, params: dict = None) -> dict:
    forecast_len = config["FORECAST_LEN"]
    ticker       = config.get("TICKER", "")
    conf_bands   = config.get("CONFIDENCE_BANDS", [20, 60, 90])
    use_spy      = config.get("RF_USE_SPY", True)
    lookback     = config.get("ML_TRAINING_LOOKBACK_BARS", 1500)
    quantiles    = sorted(set([(100 - prob) / 100 for prob in conf_bands] + [0.50]))

    if params is None:
        saved = load_best_rf_params(ticker)
        if saved:
            print(f"> Loaded RF best params from output/{ticker}/best_params_rf.txt")
        params = saved or {}

    X_all     = build_rf_features(df, use_spy=use_spy)
    X_current = X_all.iloc[[-1]].values

    current_close  = float(df["Close"].iloc[-1])
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

        X_train = combined.drop(columns=["_y"]).values
        y_train = combined["_y"].values
        n_train = len(X_train)

        rf  = _train_rf(X_train, y_train, params)
        raw = _rf_quantile_predict(rf, X_current, quantiles, current_close)

        # Enforce monotonicity: higher quantile → higher price
        sorted_qs     = sorted(raw.keys())
        sorted_prices = sorted(raw[q] for q in sorted_qs)
        milestone_preds[day] = dict(zip(sorted_qs, sorted_prices))

    if not milestone_preds:
        raise RuntimeError("RF: insufficient training data to produce any forecast.")

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

    print()
    print("=" * 44)
    print("[Random Forest FORECAST]")
    print("=" * 44)
    print(f"Training samples : {n_train}")
    print(f"Input features   : {n_features}  (Jansen-style RF features + SPY)")
    prob_cols = sorted(
        [c for c in forecast_cone.columns if c.startswith("p") and c[1:].isdigit()],
        key=lambda c: -int(c[1:]),
    )
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
        "model":           "rf",
        "forecast_cone":   forecast_cone,
        "bias":            bias,
        "milestone_preds": milestone_preds,
        "n_train_samples": n_train,
        "n_features":      n_features,
    }
