"""
Centralised read/write for best_params YAML files stored in output/{TICKER}/params/.

File naming convention:
  params/best_params_knn.yaml          — global kNN fallback (K, MIN_GAP)
  params/best_params_{method}.yaml     — per-method kNN (spearman, pearson, …)
  params/best_params_xgb.yaml          — XGBoost
  params/best_params_lgbm.yaml         — LightGBM
  params/best_params_rf.yaml           — Random Forest
  params/param_sweep_*.csv             — sweep result tables (CSV, reference only)
"""

import os
from datetime import date
from typing import Optional
import yaml


def _params_dir(ticker: str) -> str:
    return f"output/{ticker}/params"


def _yaml_path(ticker: str, filename: str) -> str:
    return f"{_params_dir(ticker)}/{filename}"


# ---------------------------------------------------------------------------
# Generic read / write
# ---------------------------------------------------------------------------

def read_params_yaml(ticker: str, filename: str) -> dict:
    """Return the 'params' dict from a YAML file, or {} if not found."""
    path = _yaml_path(ticker, filename)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("params", {})


def write_params_yaml(ticker: str, filename: str, model: str,
                      params: dict, metrics: dict = None,
                      method: str = None) -> str:
    """Write a best_params YAML file. Returns the path written."""
    os.makedirs(_params_dir(ticker), exist_ok=True)
    path = _yaml_path(ticker, filename)

    data = {"model": model, "generated": str(date.today())}
    if method:
        data["method"] = method
    if metrics:
        data["metrics"] = metrics
    data["params"] = params

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# kNN (global + per-method)
# ---------------------------------------------------------------------------

def load_knn_params(ticker: str, method: str = None) -> dict:
    """
    Load K and MIN_GAP. Priority:
      1. params/best_params_{method}.yaml  (per-method)
      2. params/best_params_knn.yaml       (global fallback)
    Returns {} if neither exists.
    """
    if method:
        p = read_params_yaml(ticker, f"best_params_{method}.yaml")
        if p:
            return p
    return read_params_yaml(ticker, "best_params_knn.yaml")


def save_knn_params(ticker: str, params: dict, metrics: dict,
                    method: str = None) -> str:
    filename = f"best_params_{method}.yaml" if method else "best_params_knn.yaml"
    return write_params_yaml(ticker, filename, model="knn",
                             params=params, metrics=metrics, method=method)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def load_xgb_params(ticker: str) -> dict:
    return read_params_yaml(ticker, "best_params_xgb.yaml")


def save_xgb_params(ticker: str, params: dict, metrics: dict) -> str:
    return write_params_yaml(ticker, "best_params_xgb.yaml",
                             model="xgboost", params=params, metrics=metrics)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def load_lgbm_params(ticker: str) -> dict:
    return read_params_yaml(ticker, "best_params_lgbm.yaml")


def save_lgbm_params(ticker: str, params: dict, metrics: dict) -> str:
    return write_params_yaml(ticker, "best_params_lgbm.yaml",
                             model="lightgbm", params=params, metrics=metrics)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def load_rf_params(ticker: str) -> dict:
    return read_params_yaml(ticker, "best_params_rf.yaml")


def save_rf_params(ticker: str, params: dict, metrics: dict) -> str:
    return write_params_yaml(ticker, "best_params_rf.yaml",
                             model="randomforest", params=params, metrics=metrics)
