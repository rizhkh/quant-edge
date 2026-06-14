"""
Parameter sweep for k-NN (K × MIN_GAP) and XGBoost (n_estimators × max_depth × learning_rate).

Usage:
    venv/bin/python param_sweep.py            ← k-NN sweep
    venv/bin/python param_sweep.py xgboost   ← XGBoost sweep
"""

import io
import os
import sys
import pandas as pd
import config as cfg
from contextlib import redirect_stdout, redirect_stderr
from data.fetcher              import fetch_data
from backtester.backtest       import run_backtest
from backtester.backtest_xgb   import run_xgb_backtest
from backtester.backtest_lgbm  import run_lgbm_backtest
from backtester.backtest_rf    import run_rf_backtest


K_VALUES          = [5, 8, 10, 12, 15, 20, 25, 30]
MIN_GAP_VALUES    = [10, 20, 30, 40]
SIMILARITY_METHODS = ["spearman", "pearson", "cosine", "euclidean", "kendall", "manhattan"]


def build_config(k: int, min_gap: int) -> dict:
    return {
        "TICKER":            cfg.TICKER,
        "PERIOD":            cfg.PERIOD,
        "INTERVAL":          cfg.INTERVAL,
        "CACHE_PATH":        cfg.CACHE_PATH,
        "WINDOW_LEN":        cfg.WINDOW_LEN,
        "FORECAST_LEN":      cfg.FORECAST_LEN,
        "BARS_BACK":         cfg.BARS_BACK,
        "SIMILARITY_METHOD": cfg.SIMILARITY_METHOD,
        "INPUT_TYPE":        cfg.INPUT_TYPE,
        "K":                 k,
        "MIN_GAP":           min_gap,
        "CONFIDENCE_BANDS":  cfg.CONFIDENCE_BANDS,
        "BACKTEST_MONTHS":   cfg.BACKTEST_MONTHS,
        "BACKTEST_STEP":     cfg.BACKTEST_STEP,
        "BACKTEST_FORECAST_DAYS": cfg.BACKTEST_FORECAST_DAYS,
        "FEATURE_COLS":      getattr(cfg, "FEATURE_COLS", None),
    }


def main() -> None:
    print()
    print("=" * 70)
    print("PARAMETER SWEEP: K  x  MIN_GAP")
    print(f"K values:       {K_VALUES}")
    print(f"MIN_GAP values: {MIN_GAP_VALUES}")
    print(f"Total combos:   {len(K_VALUES) * len(MIN_GAP_VALUES)}")
    print("=" * 70)
    print()

    # Fetch data once — reused for every combo
    base_config = build_config(cfg.K, cfg.MIN_GAP)
    df = fetch_data(
        base_config["TICKER"], base_config["PERIOD"],
        base_config["INTERVAL"], base_config["CACHE_PATH"],
    )
    print()

    total = len(K_VALUES) * len(MIN_GAP_VALUES)
    done  = 0
    rows  = []

    for k in K_VALUES:
        for gap in MIN_GAP_VALUES:
            done += 1
            print(f"[{done:>2}/{total}]  K={k:<3}  MIN_GAP={gap:<3} ...", end="", flush=True)

            config = build_config(k, gap)
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    result = run_backtest(df, config)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            if not result:
                print("  no results")
                continue

            rows.append({
                "K":           k,
                "MIN_GAP":     gap,
                "MAE":         round(result["mae"],   2),
                "RMSE":        round(result["rmse"],  2),
                "MAPE_%":      round(result["mape"],  2),
                "Dir_Acc_%":   round(result["directional_accuracy"], 1),
                "Cone_Hit_%":  round(result["cone_hit_rate"], 1),
                "N_Forecasts": result["total_forecasts"],
            })
            print(f"  MAE=${result['mae']:.2f}  MAPE={result['mape']:.1f}%  Dir={result['directional_accuracy']:.1f}%  Cone={result['cone_hit_rate']:.1f}%")

    if not rows:
        print("No results — something went wrong.")
        return

    results_df = pd.DataFrame(rows)

    # Composite score: lower MAPE is better, higher direction accuracy is better
    # Normalise each metric to 0-1 then combine
    mape_norm = (results_df["MAPE_%"] - results_df["MAPE_%"].min()) / (results_df["MAPE_%"].max() - results_df["MAPE_%"].min() + 1e-9)
    dir_norm  = (results_df["Dir_Acc_%"].max() - results_df["Dir_Acc_%"]) / (results_df["Dir_Acc_%"].max() - results_df["Dir_Acc_%"].min() + 1e-9)
    cone_norm = (results_df["Cone_Hit_%"].max() - results_df["Cone_Hit_%"]) / (results_df["Cone_Hit_%"].max() - results_df["Cone_Hit_%"].min() + 1e-9)
    # Weights: MAPE 40%, direction 40%, cone 20%
    results_df["Score"] = (0.4 * mape_norm + 0.4 * dir_norm + 0.2 * cone_norm).round(4)
    results_df = results_df.sort_values("Score").reset_index(drop=True)
    results_df.insert(0, "Rank", results_df.index + 1)

    W = 90
    print()
    print("=" * W)
    print("RESULTS  (sorted best → worst  |  Score: lower = better)")
    print("=" * W)
    print(f"{'Rank':<5} {'K':<5} {'MIN_GAP':<9} {'MAE':>7} {'RMSE':>7} {'MAPE%':>7} {'Dir%':>6} {'Cone%':>7} {'Score':>7}")
    print("-" * W)
    for _, row in results_df.iterrows():
        marker = "  <-- BEST" if row["Rank"] == 1 else ""
        print(
            f"{int(row['Rank']):<5} {int(row['K']):<5} {int(row['MIN_GAP']):<9} "
            f"${row['MAE']:>6.2f} ${row['RMSE']:>6.2f} {row['MAPE_%']:>6.1f}% "
            f"{row['Dir_Acc_%']:>5.1f}% {row['Cone_Hit_%']:>6.1f}% {row['Score']:>7.4f}"
            + marker
        )

    best = results_df.iloc[0]
    print()
    print("=" * W)
    print("WINNER")
    print("=" * W)
    print(f"  K = {int(best['K'])}   MIN_GAP = {int(best['MIN_GAP'])}")
    print(f"  MAE=${best['MAE']:.2f}  MAPE={best['MAPE_%']:.1f}%  "
          f"Direction accuracy={best['Dir_Acc_%']:.1f}%  Cone hit={best['Cone_Hit_%']:.1f}%")
    print()
    print(f"  Current config:  K={cfg.K}  MIN_GAP={cfg.MIN_GAP}")

    current_row = results_df[(results_df["K"] == cfg.K) & (results_df["MIN_GAP"] == cfg.MIN_GAP)]
    if not current_row.empty:
        cur = current_row.iloc[0]
        print(f"  Current rank:    #{int(cur['Rank'])} of {len(results_df)}")
    print()

    from utils.params_io import save_knn_params
    os.makedirs(f"output/{cfg.TICKER}/params", exist_ok=True)
    csv_path = f"output/{cfg.TICKER}/params/param_sweep_knn_{cfg.TICKER}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"> Full results saved to {csv_path}")

    metrics = {"mae": float(best["MAE"]), "mape_pct": float(best["MAPE_%"]),
               "direction_pct": float(best["Dir_Acc_%"]), "cone_hit_pct": float(best["Cone_Hit_%"])}
    params  = {"K": int(best["K"]), "MIN_GAP": int(best["MIN_GAP"])}
    yaml_path = save_knn_params(cfg.TICKER, params, metrics)
    print(f"> Best params written to {yaml_path}")
    print()


def _run_knn_sweep(method: str, df) -> None:
    """
    Run K × MIN_GAP grid search for one similarity method.
    Saves best_params_{method}.txt and param_sweep_{method}_{TICKER}.csv.
    """
    total = len(K_VALUES) * len(MIN_GAP_VALUES)
    done  = 0
    rows  = []

    ticker = cfg.TICKER
    print(f"\n  [{ticker}] {method.upper():<12}  ({total} combos)")
    for k in K_VALUES:
        for gap in MIN_GAP_VALUES:
            done += 1
            print(f"    [{ticker}] [{done:>2}/{total}]  K={k:<3}  MIN_GAP={gap:<3} ...", end="", flush=True)
            config = build_config(k, gap)
            config["SIMILARITY_METHOD"] = method
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    result = run_backtest(df, config)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            if not result:
                print("  no results")
                continue
            rows.append({
                "K":          k,
                "MIN_GAP":    gap,
                "MAE":        round(result["mae"],  2),
                "RMSE":       round(result["rmse"], 2),
                "MAPE_%":     round(result["mape"], 2),
                "Dir_Acc_%":  round(result["directional_accuracy"], 1),
                "Cone_Hit_%": round(result["cone_hit_rate"], 1),
                "N_Forecasts": result["total_forecasts"],
            })
            print(f"  MAE=${result['mae']:.2f}  Dir={result['directional_accuracy']:.1f}%")

    if not rows:
        print(f"    [{ticker}] No results for {method}.")
        return

    from datetime import date
    results_df = pd.DataFrame(rows)
    mape_n = (results_df["MAPE_%"] - results_df["MAPE_%"].min()) / (results_df["MAPE_%"].max() - results_df["MAPE_%"].min() + 1e-9)
    dir_n  = (results_df["Dir_Acc_%"].max() - results_df["Dir_Acc_%"]) / (results_df["Dir_Acc_%"].max() - results_df["Dir_Acc_%"].min() + 1e-9)
    cone_n = (results_df["Cone_Hit_%"].max() - results_df["Cone_Hit_%"]) / (results_df["Cone_Hit_%"].max() - results_df["Cone_Hit_%"].min() + 1e-9)
    results_df["Score"] = (0.4 * mape_n + 0.4 * dir_n + 0.2 * cone_n).round(4)
    results_df = results_df.sort_values("Score").reset_index(drop=True)
    results_df.insert(0, "Rank", results_df.index + 1)
    best = results_df.iloc[0]

    from utils.params_io import save_knn_params
    os.makedirs(f"output/{cfg.TICKER}/params", exist_ok=True)
    csv_path = f"output/{cfg.TICKER}/params/param_sweep_{method}_{cfg.TICKER}.csv"
    results_df.to_csv(csv_path, index=False)

    metrics   = {"mae": float(best["MAE"]), "mape_pct": float(best["MAPE_%"]),
                 "direction_pct": float(best["Dir_Acc_%"]), "cone_hit_pct": float(best["Cone_Hit_%"])}
    params    = {"K": int(best["K"]), "MIN_GAP": int(best["MIN_GAP"])}
    yaml_path = save_knn_params(cfg.TICKER, params, metrics, method=method)
    print(f"    [{ticker}] ✓ Best: K={int(best['K'])}  MIN_GAP={int(best['MIN_GAP'])}  "
          f"Dir={best['Dir_Acc_%']:.1f}%  MAE=${best['MAE']:.2f}  → {yaml_path}")


def _knn_sweep_worker(args: tuple) -> tuple[str, str]:
    """Multiprocessing worker: run one method's sweep, return (method, captured_output)."""
    method, df = args
    buf = io.StringIO()
    with redirect_stdout(buf):
        _run_knn_sweep(method, df)
    return method, buf.getvalue()


def knn_all_sweep(n_workers: int = 6) -> None:
    """Run K × MIN_GAP sweep for all 6 similarity methods in parallel."""
    import multiprocessing
    n_workers = min(n_workers, len(SIMILARITY_METHODS))
    ticker = cfg.TICKER

    print()
    print("=" * 70)
    print(f"k-NN PARAMETER SWEEP — ALL SIMILARITY METHODS — {ticker}")
    print(f"K values:       {K_VALUES}")
    print(f"MIN_GAP values: {MIN_GAP_VALUES}")
    print(f"Total combos per method: {len(K_VALUES) * len(MIN_GAP_VALUES)}")
    print(f"Methods: {SIMILARITY_METHODS}")
    print(f"Workers: {n_workers} parallel")
    print("=" * 70)

    import time
    base_config = build_config(cfg.K, cfg.MIN_GAP)
    df = fetch_data(base_config["TICKER"], base_config["PERIOD"],
                    base_config["INTERVAL"], base_config["CACHE_PATH"])

    total   = len(SIMILARITY_METHODS)
    tasks   = [(m, df) for m in SIMILARITY_METHODS]
    t_start = time.time()

    print()
    print(f"[{ticker}] Dispatching {total} methods across {n_workers} workers — "
          f"progress shown as each completes:")
    print(f"  [{ticker}] Methods: {', '.join(SIMILARITY_METHODS)}")
    print()

    completed_outputs = []
    with multiprocessing.Pool(n_workers) as pool:
        for done, (method, output) in enumerate(
                pool.imap_unordered(_knn_sweep_worker, tasks), 1):
            elapsed = time.time() - t_start
            remaining = total - done
            avg = elapsed / done
            eta = avg * remaining
            print(f"  [{ticker}] [{done}/{total}] {method.upper():<12} done  "
                  f"(elapsed {elapsed:.0f}s"
                  + (f"  ETA ~{eta:.0f}s" if remaining else "") + ")",
                  flush=True)
            completed_outputs.append((method, output))

    # Print full per-method output in original method order
    print()
    completed_outputs.sort(key=lambda x: SIMILARITY_METHODS.index(x[0]))
    for method, output in completed_outputs:
        print(output, end="")

    total_time = time.time() - t_start
    print()
    print("=" * 70)
    print(f"[{ticker}] ALL METHODS COMPLETE  ({total_time:.0f}s total)")
    print(f"[{ticker}] params/best_params_{{method}}.yaml saved in output/{ticker}/params/")
    print("  These are auto-loaded by main.py and backtest.py for each method.")
    print("=" * 70)
    print()


def xgb_sweep() -> None:
    """Grid search over XGBoost hyperparameters using walk-forward backtest."""

    N_ESTIMATORS_VALUES  = [100, 200, 400]
    MAX_DEPTH_VALUES     = [3, 4, 5]
    LEARNING_RATE_VALUES = [0.03, 0.07, 0.15]

    total = len(N_ESTIMATORS_VALUES) * len(MAX_DEPTH_VALUES) * len(LEARNING_RATE_VALUES)

    print()
    print("=" * 70)
    print("PARAMETER SWEEP: XGBoost  (n_estimators × max_depth × learning_rate)")
    print(f"n_estimators:  {N_ESTIMATORS_VALUES}")
    print(f"max_depth:     {MAX_DEPTH_VALUES}")
    print(f"learning_rate: {LEARNING_RATE_VALUES}")
    print(f"Total combos:  {total}")
    print("=" * 70)
    print()

    base_config = {
        "TICKER":               cfg.TICKER,
        "PERIOD":               cfg.PERIOD,
        "INTERVAL":             cfg.INTERVAL,
        "CACHE_PATH":           cfg.CACHE_PATH,
        "WINDOW_LEN":           cfg.WINDOW_LEN,
        "FORECAST_LEN":         cfg.FORECAST_LEN,
        "FEATURE_COLS":         getattr(cfg, "FEATURE_COLS", None),
        "BACKTEST_MONTHS":      cfg.BACKTEST_MONTHS,
        "BACKTEST_STEP":        cfg.BACKTEST_STEP,
        "BACKTEST_FORECAST_DAYS": cfg.BACKTEST_FORECAST_DAYS,
        "ML_TRAINING_LOOKBACK_BARS": getattr(cfg, "ML_TRAINING_LOOKBACK_BARS", 1500),
        "XGB_USE_SPY":          getattr(cfg, "XGB_USE_SPY", False),
        "XGB_USE_SECTOR":       getattr(cfg, "XGB_USE_SECTOR", True),
        "XGB_USE_VIX":          getattr(cfg, "XGB_USE_VIX", True),
        "XGB_USE_EARNINGS":     getattr(cfg, "XGB_USE_EARNINGS", True),
        "SECTOR_MAP":           getattr(cfg, "SECTOR_MAP", {}),
        "TICKER":               cfg.TICKER,
    }

    df = fetch_data(cfg.TICKER, cfg.PERIOD, cfg.INTERVAL, cfg.CACHE_PATH)
    print()

    done = 0
    rows = []

    for n_est in N_ESTIMATORS_VALUES:
        for depth in MAX_DEPTH_VALUES:
            for lr in LEARNING_RATE_VALUES:
                done += 1
                print(f"[{done:>2}/{total}]  n_est={n_est:<4}  depth={depth}  lr={lr:<5} ...",
                      end="", flush=True)

                params = {
                    "n_estimators":     n_est,
                    "max_depth":        depth,
                    "learning_rate":    lr,
                    "subsample":        0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 5,
                }

                try:
                    result = run_xgb_backtest(df, base_config, params=params)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

                if not result:
                    print("  no results")
                    continue

                rows.append({
                    "n_estimators":      n_est,
                    "max_depth":         depth,
                    "learning_rate":     lr,
                    "MAE":               round(result["mae"],   2),
                    "MAPE_%":            round(result["mape"],  2),
                    "Dir_Acc_%":         round(result["directional_accuracy"], 1),
                    "Cone_Hit_%":        round(result["cone_hit_rate"], 1),
                    "N_Forecasts":       result["total_forecasts"],
                })
                print(f"  MAE=${result['mae']:.2f}  MAPE={result['mape']:.1f}%"
                      f"  Dir={result['directional_accuracy']:.1f}%"
                      f"  Cone={result['cone_hit_rate']:.1f}%")

    if not rows:
        print("No results — something went wrong.")
        return

    results_df = pd.DataFrame(rows)

    # Composite score: same weighting as kNN sweep
    mape_norm = (results_df["MAPE_%"] - results_df["MAPE_%"].min()) / (results_df["MAPE_%"].max() - results_df["MAPE_%"].min() + 1e-9)
    dir_norm  = (results_df["Dir_Acc_%"].max() - results_df["Dir_Acc_%"]) / (results_df["Dir_Acc_%"].max() - results_df["Dir_Acc_%"].min() + 1e-9)
    cone_norm = (results_df["Cone_Hit_%"].max() - results_df["Cone_Hit_%"]) / (results_df["Cone_Hit_%"].max() - results_df["Cone_Hit_%"].min() + 1e-9)
    results_df["Score"] = (0.4 * mape_norm + 0.4 * dir_norm + 0.2 * cone_norm).round(4)
    results_df = results_df.sort_values("Score").reset_index(drop=True)
    results_df.insert(0, "Rank", results_df.index + 1)

    W = 90
    print()
    print("=" * W)
    print("RESULTS  (sorted best → worst  |  Score: lower = better)")
    print("=" * W)
    print(f"{'Rank':<5} {'n_est':<7} {'depth':<7} {'lr':<7} {'MAE':>7} {'MAPE%':>7} {'Dir%':>6} {'Cone%':>7} {'Score':>7}")
    print("-" * W)
    for _, row in results_df.iterrows():
        marker = "  <-- BEST" if row["Rank"] == 1 else ""
        print(
            f"{int(row['Rank']):<5} {int(row['n_estimators']):<7} {int(row['max_depth']):<7} "
            f"{row['learning_rate']:<7} ${row['MAE']:>6.2f} {row['MAPE_%']:>6.1f}% "
            f"{row['Dir_Acc_%']:>5.1f}% {row['Cone_Hit_%']:>6.1f}% {row['Score']:>7.4f}"
            + marker
        )

    best = results_df.iloc[0]
    print()
    print("=" * W)
    print("WINNER")
    print("=" * W)
    print(f"  n_estimators={int(best['n_estimators'])}  max_depth={int(best['max_depth'])}  learning_rate={best['learning_rate']}")
    print(f"  MAE=${best['MAE']:.2f}  MAPE={best['MAPE_%']:.1f}%  "
          f"Direction={best['Dir_Acc_%']:.1f}%  Cone={best['Cone_Hit_%']:.1f}%")

    from utils.params_io import save_xgb_params
    os.makedirs(f"output/{cfg.TICKER}/params", exist_ok=True)
    csv_path = f"output/{cfg.TICKER}/params/param_sweep_xgb_{cfg.TICKER}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n> Full results saved to {csv_path}")

    metrics   = {"mae": float(best["MAE"]), "mape_pct": float(best["MAPE_%"]),
                 "direction_pct": float(best["Dir_Acc_%"]), "cone_hit_pct": float(best["Cone_Hit_%"])}
    params    = {"n_estimators": int(best["n_estimators"]), "max_depth": int(best["max_depth"]),
                 "learning_rate": float(best["learning_rate"]),
                 "subsample": 0.8, "colsample_bytree": 0.8}
    yaml_path = save_xgb_params(cfg.TICKER, params, metrics)
    print(f"> Best params written to {yaml_path}\n")


def lgbm_sweep() -> None:
    """Grid search over LightGBM hyperparameters using walk-forward backtest."""

    N_ESTIMATORS_VALUES  = [200, 400, 600]
    MAX_DEPTH_VALUES     = [4, 6, 8]
    LEARNING_RATE_VALUES = [0.03, 0.07, 0.15]
    NUM_LEAVES           = 31   # fixed — most impactful on speed; tune manually if needed

    total = len(N_ESTIMATORS_VALUES) * len(MAX_DEPTH_VALUES) * len(LEARNING_RATE_VALUES)

    print()
    print("=" * 70)
    print("PARAMETER SWEEP: LightGBM  (n_estimators × max_depth × learning_rate)")
    print(f"n_estimators:  {N_ESTIMATORS_VALUES}")
    print(f"max_depth:     {MAX_DEPTH_VALUES}")
    print(f"learning_rate: {LEARNING_RATE_VALUES}")
    print(f"num_leaves:    {NUM_LEAVES}  (fixed)")
    print(f"Total combos:  {total}")
    print("=" * 70)
    print()

    base_config = {
        "TICKER":               cfg.TICKER,
        "PERIOD":               cfg.PERIOD,
        "INTERVAL":             cfg.INTERVAL,
        "CACHE_PATH":           cfg.CACHE_PATH,
        "WINDOW_LEN":           cfg.WINDOW_LEN,
        "FORECAST_LEN":         cfg.FORECAST_LEN,
        "CONFIDENCE_BANDS":     cfg.CONFIDENCE_BANDS,
        "BACKTEST_MONTHS":      cfg.BACKTEST_MONTHS,
        "BACKTEST_STEP":        cfg.BACKTEST_STEP,
        "BACKTEST_FORECAST_DAYS": cfg.BACKTEST_FORECAST_DAYS,
        "ML_TRAINING_LOOKBACK_BARS": getattr(cfg, "ML_TRAINING_LOOKBACK_BARS", 1500),
        "LGBM_USE_VIX":         getattr(cfg, "LGBM_USE_VIX", True),
        "LGBM_USE_SPY":         getattr(cfg, "LGBM_USE_SPY", True),
    }

    df = fetch_data(cfg.TICKER, cfg.PERIOD, cfg.INTERVAL, cfg.CACHE_PATH)
    print()

    done = 0
    rows = []

    for n_est in N_ESTIMATORS_VALUES:
        for depth in MAX_DEPTH_VALUES:
            for lr in LEARNING_RATE_VALUES:
                done += 1
                print(f"[{done:>2}/{total}]  n_est={n_est:<4}  depth={depth}  lr={lr:<5} ...",
                      end="", flush=True)

                params = {
                    "n_estimators":      n_est,
                    "max_depth":         depth,
                    "learning_rate":     lr,
                    "num_leaves":        NUM_LEAVES,
                    "subsample":         0.8,
                    "colsample_bytree":  0.8,
                    "min_child_samples": 10,
                }

                try:
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        result = run_lgbm_backtest(df, base_config, params=params)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

                if not result:
                    print("  no results")
                    continue

                rows.append({
                    "n_estimators":  n_est,
                    "max_depth":     depth,
                    "learning_rate": lr,
                    "num_leaves":    NUM_LEAVES,
                    "MAE":           round(result["mae"],  2),
                    "MAPE_%":        round(result["mape"], 2),
                    "Dir_Acc_%":     round(result["directional_accuracy"], 1),
                    "Cone_Hit_%":    round(result["cone_hit_rate"], 1),
                    "N_Forecasts":   result["total_forecasts"],
                })
                print(f"  MAE=${result['mae']:.2f}  MAPE={result['mape']:.1f}%"
                      f"  Dir={result['directional_accuracy']:.1f}%"
                      f"  Cone={result['cone_hit_rate']:.1f}%")

    if not rows:
        print("No results — something went wrong.")
        return

    results_df = pd.DataFrame(rows)

    mape_norm = (results_df["MAPE_%"] - results_df["MAPE_%"].min()) / (results_df["MAPE_%"].max() - results_df["MAPE_%"].min() + 1e-9)
    dir_norm  = (results_df["Dir_Acc_%"].max() - results_df["Dir_Acc_%"]) / (results_df["Dir_Acc_%"].max() - results_df["Dir_Acc_%"].min() + 1e-9)
    cone_norm = (results_df["Cone_Hit_%"].max() - results_df["Cone_Hit_%"]) / (results_df["Cone_Hit_%"].max() - results_df["Cone_Hit_%"].min() + 1e-9)
    results_df["Score"] = (0.4 * mape_norm + 0.4 * dir_norm + 0.2 * cone_norm).round(4)
    results_df = results_df.sort_values("Score").reset_index(drop=True)
    results_df.insert(0, "Rank", results_df.index + 1)

    W = 95
    print()
    print("=" * W)
    print("RESULTS  (sorted best → worst  |  Score: lower = better)")
    print("=" * W)
    print(f"{'Rank':<5} {'n_est':<7} {'depth':<7} {'lr':<7} {'leaves':<8} {'MAE':>7} {'MAPE%':>7} {'Dir%':>6} {'Cone%':>7} {'Score':>7}")
    print("-" * W)
    for _, row in results_df.iterrows():
        marker = "  <-- BEST" if row["Rank"] == 1 else ""
        print(
            f"{int(row['Rank']):<5} {int(row['n_estimators']):<7} {int(row['max_depth']):<7} "
            f"{row['learning_rate']:<7} {int(row['num_leaves']):<8} ${row['MAE']:>6.2f} "
            f"{row['MAPE_%']:>6.1f}% {row['Dir_Acc_%']:>5.1f}% {row['Cone_Hit_%']:>6.1f}% "
            f"{row['Score']:>7.4f}" + marker
        )

    best = results_df.iloc[0]
    print()
    print("=" * W)
    print("WINNER")
    print("=" * W)
    print(f"  n_estimators={int(best['n_estimators'])}  max_depth={int(best['max_depth'])}  "
          f"learning_rate={best['learning_rate']}  num_leaves={int(best['num_leaves'])}")
    print(f"  MAE=${best['MAE']:.2f}  MAPE={best['MAPE_%']:.1f}%  "
          f"Direction={best['Dir_Acc_%']:.1f}%  Cone={best['Cone_Hit_%']:.1f}%")

    from utils.params_io import save_lgbm_params
    os.makedirs(f"output/{cfg.TICKER}/params", exist_ok=True)
    csv_path = f"output/{cfg.TICKER}/params/param_sweep_lgbm_{cfg.TICKER}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n> Full results saved to {csv_path}")

    metrics   = {"mae": float(best["MAE"]), "mape_pct": float(best["MAPE_%"]),
                 "direction_pct": float(best["Dir_Acc_%"]), "cone_hit_pct": float(best["Cone_Hit_%"])}
    params    = {"n_estimators": int(best["n_estimators"]), "max_depth": int(best["max_depth"]),
                 "learning_rate": float(best["learning_rate"]),
                 "num_leaves": int(best["num_leaves"]),
                 "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 10}
    yaml_path = save_lgbm_params(cfg.TICKER, params, metrics)
    print(f"> Best params written to {yaml_path}\n")


def rf_sweep() -> None:
    """Grid search over Random Forest hyperparameters using walk-forward backtest."""

    N_ESTIMATORS_VALUES  = [100, 300]
    MAX_DEPTH_VALUES     = [None, 15]
    MIN_SAMPLES_VALUES   = [3, 5, 10]

    total = len(N_ESTIMATORS_VALUES) * len(MAX_DEPTH_VALUES) * len(MIN_SAMPLES_VALUES)

    print()
    print("=" * 70)
    print("PARAMETER SWEEP: Random Forest  (n_estimators × max_depth × min_samples_leaf)")
    print(f"n_estimators:     {N_ESTIMATORS_VALUES}")
    print(f"max_depth:        {MAX_DEPTH_VALUES}  (None = unlimited)")
    print(f"min_samples_leaf: {MIN_SAMPLES_VALUES}")
    print(f"max_features:     sqrt  (fixed)")
    print(f"Total combos:     {total}")
    print("=" * 70)
    print()

    base_config = {
        "TICKER":               cfg.TICKER,
        "PERIOD":               cfg.PERIOD,
        "INTERVAL":             cfg.INTERVAL,
        "CACHE_PATH":           cfg.CACHE_PATH,
        "WINDOW_LEN":           cfg.WINDOW_LEN,
        "FORECAST_LEN":         cfg.FORECAST_LEN,
        "CONFIDENCE_BANDS":     cfg.CONFIDENCE_BANDS,
        "BACKTEST_MONTHS":      cfg.BACKTEST_MONTHS,
        "BACKTEST_STEP":        cfg.BACKTEST_STEP,
        "BACKTEST_FORECAST_DAYS": cfg.BACKTEST_FORECAST_DAYS,
        "ML_TRAINING_LOOKBACK_BARS": getattr(cfg, "ML_TRAINING_LOOKBACK_BARS", 1500),
        "RF_USE_SPY":           getattr(cfg, "RF_USE_SPY", True),
    }

    df = fetch_data(cfg.TICKER, cfg.PERIOD, cfg.INTERVAL, cfg.CACHE_PATH)
    print()

    done = 0
    rows = []

    for n_est in N_ESTIMATORS_VALUES:
        for depth in MAX_DEPTH_VALUES:
            for min_samp in MIN_SAMPLES_VALUES:
                done += 1
                depth_s = str(depth) if depth is not None else "None"
                print(f"[{done:>2}/{total}]  n_est={n_est:<4}  depth={depth_s:<5}  min_leaf={min_samp} ...",
                      end="", flush=True)

                params = {
                    "n_estimators":     n_est,
                    "max_depth":        depth,
                    "min_samples_leaf": min_samp,
                    "max_features":     "sqrt",
                }

                try:
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        result = run_rf_backtest(df, base_config, params=params)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

                if not result:
                    print("  no results")
                    continue

                rows.append({
                    "n_estimators":     n_est,
                    "max_depth":        depth_s,
                    "min_samples_leaf": min_samp,
                    "MAE":              round(result["mae"],  2),
                    "MAPE_%":           round(result["mape"], 2),
                    "Dir_Acc_%":        round(result["directional_accuracy"], 1),
                    "Cone_Hit_%":       round(result["cone_hit_rate"], 1),
                    "N_Forecasts":      result["total_forecasts"],
                })
                print(f"  MAE=${result['mae']:.2f}  MAPE={result['mape']:.1f}%"
                      f"  Dir={result['directional_accuracy']:.1f}%"
                      f"  Cone={result['cone_hit_rate']:.1f}%")

    if not rows:
        print("No results — something went wrong.")
        return

    results_df = pd.DataFrame(rows)
    mape_norm = (results_df["MAPE_%"] - results_df["MAPE_%"].min()) / (results_df["MAPE_%"].max() - results_df["MAPE_%"].min() + 1e-9)
    dir_norm  = (results_df["Dir_Acc_%"].max() - results_df["Dir_Acc_%"]) / (results_df["Dir_Acc_%"].max() - results_df["Dir_Acc_%"].min() + 1e-9)
    cone_norm = (results_df["Cone_Hit_%"].max() - results_df["Cone_Hit_%"]) / (results_df["Cone_Hit_%"].max() - results_df["Cone_Hit_%"].min() + 1e-9)
    results_df["Score"] = (0.4 * mape_norm + 0.4 * dir_norm + 0.2 * cone_norm).round(4)
    results_df = results_df.sort_values("Score").reset_index(drop=True)
    results_df.insert(0, "Rank", results_df.index + 1)

    W = 95
    print()
    print("=" * W)
    print("RESULTS  (sorted best → worst  |  Score: lower = better)")
    print("=" * W)
    print(f"{'Rank':<5} {'n_est':<7} {'depth':<7} {'min_leaf':<10} {'MAE':>7} {'MAPE%':>7} {'Dir%':>6} {'Cone%':>7} {'Score':>7}")
    print("-" * W)
    for _, row in results_df.iterrows():
        marker = "  <-- BEST" if row["Rank"] == 1 else ""
        print(
            f"{int(row['Rank']):<5} {int(row['n_estimators']):<7} {str(row['max_depth']):<7} "
            f"{int(row['min_samples_leaf']):<10} ${row['MAE']:>6.2f} "
            f"{row['MAPE_%']:>6.1f}% {row['Dir_Acc_%']:>5.1f}% {row['Cone_Hit_%']:>6.1f}% "
            f"{row['Score']:>7.4f}" + marker
        )

    best = results_df.iloc[0]
    print()
    print("=" * W)
    print("WINNER")
    print("=" * W)
    print(f"  n_estimators={int(best['n_estimators'])}  max_depth={best['max_depth']}  min_samples_leaf={int(best['min_samples_leaf'])}")
    print(f"  MAE=${best['MAE']:.2f}  MAPE={best['MAPE_%']:.1f}%  "
          f"Direction={best['Dir_Acc_%']:.1f}%  Cone={best['Cone_Hit_%']:.1f}%")

    from utils.params_io import save_rf_params
    os.makedirs(f"output/{cfg.TICKER}/params", exist_ok=True)
    csv_path = f"output/{cfg.TICKER}/params/param_sweep_rf_{cfg.TICKER}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n> Full results saved to {csv_path}")

    depth_val = best["max_depth"]
    metrics   = {"mae": float(best["MAE"]), "mape_pct": float(best["MAPE_%"]),
                 "direction_pct": float(best["Dir_Acc_%"]), "cone_hit_pct": float(best["Cone_Hit_%"])}
    params    = {"n_estimators": int(best["n_estimators"]),
                 "max_depth": None if str(depth_val).lower() == "none" else int(depth_val),
                 "min_samples_leaf": int(best["min_samples_leaf"]),
                 "max_features": "sqrt"}
    yaml_path = save_rf_params(cfg.TICKER, params, metrics)
    print(f"> Best params written to {yaml_path}\n")


def knn2_sweep() -> None:
    """
    Grid search over knn2 parameters:
      half_life_days        — recency decay (Step 5)
      distance_threshold_pct — adaptive k cutoff (Step 6)

    Feature weights are computed once (cached) and reused across all combos.
    Scoring: Direction 60% (primary target) + MAPE 25% + Cone 15%.
    Saves best params to output/{TICKER}/params/feature_weights_knn2.yaml.
    """
    from features.engineer        import compute_features
    from backtester.backtest_knn2 import run_knn2_backtest
    from features.feature_weights import save_knn2_params

    HALF_LIFE_VALUES   = [60, 120, 250, 500, 750]
    DISTANCE_PCT_VALUES = [10, 15, 20, 30, 40]
    total = len(HALF_LIFE_VALUES) * len(DISTANCE_PCT_VALUES)

    print()
    print("=" * 70)
    print(f"PARAMETER SWEEP: knn2  (half_life_days × distance_threshold_pct)")
    print(f"half_life_days:         {HALF_LIFE_VALUES}")
    print(f"distance_threshold_pct: {DISTANCE_PCT_VALUES}")
    print(f"Total combos:           {total}")
    print(f"Scoring:                Direction 60% + MAPE 25% + Cone 15%  (Day 5 dir is primary)")
    print("=" * 70)
    print()

    base_config = {
        "TICKER":                 cfg.TICKER,
        "PERIOD":                 cfg.PERIOD,
        "INTERVAL":               cfg.INTERVAL,
        "CACHE_PATH":             cfg.CACHE_PATH,
        "WINDOW_LEN":             cfg.WINDOW_LEN,
        "FORECAST_LEN":           cfg.FORECAST_LEN,
        "BARS_BACK":              cfg.BARS_BACK,
        "K":                      cfg.K,
        "MIN_GAP":                cfg.MIN_GAP,
        "CONFIDENCE_BANDS":       cfg.CONFIDENCE_BANDS,
        "BACKTEST_MONTHS":        cfg.BACKTEST_MONTHS,
        "BACKTEST_STEP":          cfg.BACKTEST_STEP,
        "BACKTEST_FORECAST_DAYS": cfg.BACKTEST_FORECAST_DAYS,
        "FEATURE_COLS":           getattr(cfg, "FEATURE_COLS", None),
        "ML_TRAINING_LOOKBACK_BARS": getattr(cfg, "ML_TRAINING_LOOKBACK_BARS", 1500),
    }

    df_raw = fetch_data(cfg.TICKER, cfg.PERIOD, cfg.INTERVAL, cfg.CACHE_PATH)
    df     = compute_features(df_raw)
    print()
    print(f"Feature weights will be computed on first combo then cached.")
    print()

    done = 0
    rows = []

    for hl in HALF_LIFE_VALUES:
        for dpct in DISTANCE_PCT_VALUES:
            done += 1
            print(f"[{done:>2}/{total}]  half_life={hl:<4}  dist_pct={dpct:<4} ...",
                  end="", flush=True)

            params = {"half_life_days": hl, "distance_threshold_pct": dpct}

            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    result = run_knn2_backtest(df, base_config, params=params)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            if not result:
                print("  no results")
                continue

            rows.append({
                "half_life_days":         hl,
                "distance_threshold_pct": dpct,
                "MAE":         round(result["mae"],  2),
                "MAPE_%":      round(result["mape"], 2),
                "Dir_Acc_%":   round(result["directional_accuracy"], 1),
                "Cone_Hit_%":  round(result["cone_hit_rate"], 1),
                "N_Forecasts": result["total_forecasts"],
            })
            print(f"  Dir: {result['directional_accuracy']:.1f}%  "
                  f"MAE: ${result['mae']:.2f}  "
                  f"MAPE: {result['mape']:.1f}%  "
                  f"Cone: {result['cone_hit_rate']:.1f}%")

    if not rows:
        print("No results — something went wrong.")
        return

    results_df = pd.DataFrame(rows)

    # Scoring: Direction 60%, MAPE 25%, Cone 15% (direction is primary target)
    dir_n  = (results_df["Dir_Acc_%"].max() - results_df["Dir_Acc_%"]) / (results_df["Dir_Acc_%"].max() - results_df["Dir_Acc_%"].min() + 1e-9)
    mape_n = (results_df["MAPE_%"] - results_df["MAPE_%"].min()) / (results_df["MAPE_%"].max() - results_df["MAPE_%"].min() + 1e-9)
    cone_n = (results_df["Cone_Hit_%"].max() - results_df["Cone_Hit_%"]) / (results_df["Cone_Hit_%"].max() - results_df["Cone_Hit_%"].min() + 1e-9)
    results_df["Score"] = (0.60 * dir_n + 0.25 * mape_n + 0.15 * cone_n).round(4)
    results_df = results_df.sort_values("Score").reset_index(drop=True)
    results_df.insert(0, "Rank", results_df.index + 1)

    W = 85
    print()
    print("=" * W)
    print("RESULTS  (sorted best → worst  |  Score: lower = better  |  Dir weighted 60%)")
    print("=" * W)
    print(f"  {'Rank':<5} {'HalfLife':<10} {'DistPct':<9} {'Dir%':>7}  {'MAE':>8}  {'MAPE%':>7}  {'Cone%':>7}  {'Score':>7}")
    print(f"  {'-'*5} {'-'*10} {'-'*9} {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
    for _, row in results_df.iterrows():
        marker = "  ← BEST" if row["Rank"] == 1 else ""
        print(f"  {int(row['Rank']):<5} {int(row['half_life_days']):<10} "
              f"{row['distance_threshold_pct']:<9.0f} "
              f"{row['Dir_Acc_%']:>6.1f}%  "
              f"${row['MAE']:>7.2f}  {row['MAPE_%']:>6.1f}%  "
              f"{row['Cone_Hit_%']:>6.1f}%  {row['Score']:>7.4f}{marker}")

    best = results_df.iloc[0]
    print()
    print("=" * W)
    print("WINNER")
    print("=" * W)
    print(f"  half_life_days={int(best['half_life_days'])}  "
          f"distance_threshold_pct={best['distance_threshold_pct']:.0f}")
    print(f"  Dir={best['Dir_Acc_%']:.1f}%  MAE=${best['MAE']:.2f}  "
          f"MAPE={best['MAPE_%']:.1f}%  Cone={best['Cone_Hit_%']:.1f}%")

    os.makedirs(f"output/{cfg.TICKER}/params", exist_ok=True)
    csv_path = f"output/{cfg.TICKER}/params/param_sweep_knn2_{cfg.TICKER}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n> Full results saved to {csv_path}")

    save_knn2_params(cfg.TICKER, {
        "half_life_days":         int(best["half_life_days"]),
        "distance_threshold_pct": float(best["distance_threshold_pct"]),
        "sweep_dir_acc":          float(best["Dir_Acc_%"]),
        "sweep_mape":             float(best["MAPE_%"]),
        "sweep_n_forecasts":      int(best["N_Forecasts"]),
    })
    print(f"> Best params saved to output/{cfg.TICKER}/params/feature_weights_knn2.yaml")
    print()


if __name__ == "__main__":
    # Strip --ticker and --knn-workers flags before mode detection
    _argv       = sys.argv[1:]
    _ticker     = None
    _knn_workers = 4
    _step        = None
    _mode_args  = []

    i = 0
    while i < len(_argv):
        if _argv[i] == "--ticker" and i + 1 < len(_argv):
            _ticker = _argv[i + 1]; i += 2
        elif _argv[i] == "--knn-workers" and i + 1 < len(_argv):
            _knn_workers = int(_argv[i + 1]); i += 2
        elif _argv[i] == "--step" and i + 1 < len(_argv):
            _step = int(_argv[i + 1]); i += 2
        else:
            _mode_args.append(_argv[i]); i += 1

    # Override cfg.TICKER if provided (avoids touching config.py)
    if _ticker:
        cfg.TICKER      = _ticker
        cfg.CACHE_PATH  = f"data/{_ticker}_max.csv"

    # Override BACKTEST_STEP if provided (used to trade accuracy for speed)
    if _step:
        cfg.BACKTEST_STEP = _step

    _mode = _mode_args[0] if _mode_args else None

    if _mode == "xgboost":
        xgb_sweep()
    elif _mode == "lgbm":
        lgbm_sweep()
    elif _mode == "rf":
        rf_sweep()
    elif _mode == "knn":
        knn_all_sweep(n_workers=_knn_workers)
    elif _mode == "knn2":
        knn2_sweep()
    else:
        main()
