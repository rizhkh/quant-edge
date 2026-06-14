"""
One-time migration: convert existing best_params*.txt files to YAML
and move param_sweep_*.csv files into output/{TICKER}/params/.

Run once:
    venv/bin/python migrate_params.py
"""

import os
import re
import shutil
import glob
import yaml
from datetime import date

SIMILARITY_METHODS = ["spearman", "pearson", "cosine", "euclidean", "kendall"]


def _parse_txt(path: str) -> tuple[dict, dict]:
    """Parse a best_params*.txt file → (params dict, metrics dict)."""
    params  = {}
    metrics = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#   MAE"):
                m = re.search(r"\$?([\d.]+)", line.split("=", 1)[-1])
                if m:
                    metrics["mae"] = float(m.group(1))
            elif line.startswith("#   MAPE"):
                m = re.search(r"([\d.]+)", line.split("=", 1)[-1])
                if m:
                    metrics["mape_pct"] = float(m.group(1))
            elif line.startswith("#   Direction"):
                m = re.search(r"([\d.]+)", line.split("=", 1)[-1])
                if m:
                    metrics["direction_pct"] = float(m.group(1))
            elif line.startswith("#   Cone"):
                m = re.search(r"([\d.]+)", line.split("=", 1)[-1])
                if m:
                    metrics["cone_hit_pct"] = float(m.group(1))
            elif not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                k = key.strip().lower()
                v = val.strip()
                if v.lower() in ("none", ""):
                    params[k] = None
                else:
                    try:
                        params[k] = int(v)
                    except ValueError:
                        try:
                            params[k] = float(v)
                        except ValueError:
                            params[k] = v
    return params, metrics


def _normalise_params(raw: dict, model: str) -> dict:
    """Map lowercase txt keys to the canonical param names used by each model."""
    if model == "knn":
        return {
            "K":       raw.get("k"),
            "MIN_GAP": raw.get("min_gap"),
        }
    if model == "xgboost":
        return {k: raw.get(k) for k in
                ["n_estimators", "max_depth", "learning_rate",
                 "subsample", "colsample_bytree"] if k in raw}
    if model == "lightgbm":
        return {k: raw.get(k) for k in
                ["n_estimators", "max_depth", "learning_rate",
                 "num_leaves", "subsample", "colsample_bytree",
                 "min_child_samples"] if k in raw}
    if model == "randomforest":
        return {k: raw.get(k) for k in
                ["n_estimators", "max_depth", "min_samples_leaf",
                 "max_features"] if k in raw}
    return raw


def migrate_ticker(ticker_dir: str) -> None:
    ticker     = os.path.basename(ticker_dir)
    params_dir = os.path.join(ticker_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    # --- Map txt files to (yaml_filename, model, method) ---
    migrations = [
        ("best_params.txt",      "best_params_knn.yaml",   "knn",          None),
        ("best_params_xgb.txt",  "best_params_xgb.yaml",   "xgboost",      None),
        ("best_params_lgbm.txt", "best_params_lgbm.yaml",  "lightgbm",     None),
        ("best_params_rf.txt",   "best_params_rf.yaml",    "randomforest", None),
    ]
    for method in SIMILARITY_METHODS:
        migrations.append(
            (f"best_params_{method}.txt",
             f"best_params_{method}.yaml",
             "knn", method)
        )

    for txt_name, yaml_name, model, method in migrations:
        txt_path  = os.path.join(ticker_dir, txt_name)
        yaml_path = os.path.join(params_dir, yaml_name)

        if not os.path.exists(txt_path):
            continue
        if os.path.exists(yaml_path):
            print(f"  [{ticker}] {yaml_name} already exists — skipping")
            continue

        raw_params, metrics = _parse_txt(txt_path)
        params = _normalise_params(raw_params, model)
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        data = {"model": model, "generated": str(date.today())}
        if method:
            data["method"] = method
        if metrics:
            data["metrics"] = metrics
        data["params"] = params

        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"  [{ticker}] {txt_name} → params/{yaml_name}")

    # --- Move param_sweep_*.csv files to params/ ---
    for csv_path in glob.glob(os.path.join(ticker_dir, "param_sweep_*.csv")):
        filename = os.path.basename(csv_path)
        dest     = os.path.join(params_dir, filename)
        if os.path.exists(dest):
            print(f"  [{ticker}] {filename} already in params/ — skipping")
            continue
        shutil.move(csv_path, dest)
        print(f"  [{ticker}] {filename} → params/{filename}")


def main() -> None:
    if not os.path.isdir("output"):
        print("No output/ directory found. Nothing to migrate.")
        return

    tickers = [
        d for d in sorted(os.listdir("output"))
        if os.path.isdir(f"output/{d}")
        and "copy" not in d.lower()
        and "test" not in d.lower()
    ]

    if not tickers:
        print("No ticker directories found in output/.")
        return

    print(f"Migrating {len(tickers)} tickers: {', '.join(tickers)}")
    print()
    for ticker in tickers:
        migrate_ticker(f"output/{ticker}")
    print()
    print("Migration complete.")
    print("Run your forecasts/backtests as normal — code now reads from params/*.yaml")


if __name__ == "__main__":
    main()
