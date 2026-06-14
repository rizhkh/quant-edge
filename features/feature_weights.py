"""
XGBoost-based feature importance weights for knn2.
Trains a quick regressor on 5-day forward returns and extracts
feature_importances_ as the weight vector for similarity search.

Cached per ticker in output/{TICKER}/params/feature_weights_knn2.yaml.
Recomputes only when new bars have arrived since the last computation.
"""

import os
import numpy as np
import pandas as pd
import yaml
from datetime import date as _date


def compute_feature_weights(
    df: pd.DataFrame,
    feature_cols: list,
    ticker: str,
    lookback: int = 1500,
    target_days: int = 5,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Return a normalized weight vector (sums to 1) aligned to feature_cols order.
    Trains XGBoost on 5-day forward returns and uses feature_importances_.
    Falls back to uniform weights if XGBoost unavailable or data is too short.
    """
    cache_path = f"output/{ticker}/params/feature_weights_knn2.yaml"

    if not force_recompute and os.path.exists(cache_path):
        cached = _load_yaml(cache_path)
        if cached.get("computed_on") == str(_date.today()):
            try:
                weights = np.array([cached["feature_weights"][c] for c in feature_cols])
                if len(weights) == len(feature_cols) and not np.any(np.isnan(weights)):
                    return weights
            except (KeyError, TypeError):
                pass

    weights = _fit_weights(df, feature_cols, lookback, target_days)
    _save_weights(cache_path, weights, feature_cols, lookback, target_days)
    return weights


def _fit_weights(df, feature_cols, lookback, target_days):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return _uniform(feature_cols)

    X = df[feature_cols].copy()
    y = df["Close"].pct_change(target_days).shift(-target_days)

    combined = X.copy()
    combined["_y"] = y
    combined = combined.dropna()
    if lookback and len(combined) > lookback:
        combined = combined.iloc[-lookback:]
    if len(combined) < 60:
        return _uniform(feature_cols)

    try:
        model = XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, verbosity=0, n_jobs=-1,
        )
        model.fit(combined[feature_cols].values, combined["_y"].values)
        imp = model.feature_importances_
        total = imp.sum()
        return imp / total if total > 0 else _uniform(feature_cols)
    except Exception:
        return _uniform(feature_cols)


def _uniform(feature_cols):
    n = len(feature_cols)
    return np.ones(n) / n


def _save_weights(cache_path, weights, feature_cols, lookback, target_days):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    existing = _load_yaml(cache_path)
    existing.update({
        "feature_weights": {c: float(w) for c, w in zip(feature_cols, weights)},
        "computed_on":      str(_date.today()),
        "lookback_bars":    lookback,
        "target_days":      target_days,
    })
    with open(cache_path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)


def _load_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_knn2_params(ticker: str) -> dict:
    """Load all knn2 runtime params (half_life, distance_pct, weights, etc.)."""
    return _load_yaml(f"output/{ticker}/params/feature_weights_knn2.yaml")


def save_knn2_params(ticker: str, updates: dict) -> None:
    """Merge updates into the knn2 params yaml."""
    path = f"output/{ticker}/params/feature_weights_knn2.yaml"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = _load_yaml(path)
    existing.update(updates)
    with open(path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)
