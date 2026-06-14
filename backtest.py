import os
import sys
import argparse
import pandas as pd
import config as cfg

from data.fetcher              import fetch_data
from features.engineer         import compute_features
from backtester.backtest       import run_backtest
from backtester.backtest_xgb   import run_xgb_backtest
from backtester.backtest_lgbm  import run_lgbm_backtest
from backtester.backtest_rf    import run_rf_backtest
from backtester.backtest_knn2  import run_knn2_backtest
from models.xgboost_model      import load_best_xgb_params
from models.lightgbm_model     import load_best_lgbm_params
from models.rf_model           import load_best_rf_params
from features.feature_weights  import load_knn2_params


def _load_best_knn_params(ticker: str, method: str = None) -> dict:
    """Load K and MIN_GAP from params/best_params_{method}.yaml or params/best_params_knn.yaml."""
    from utils.params_io import load_knn_params
    return load_knn_params(ticker, method=method)


def _build_config(months: int, step: int, forecast_days: int,
                  start_date: str = None, end_date: str = None) -> dict:  # noqa: E501
    best_knn = _load_best_knn_params(cfg.TICKER, method=cfg.SIMILARITY_METHOD)
    k        = best_knn.get("K",       cfg.K)
    min_gap  = best_knn.get("MIN_GAP", cfg.MIN_GAP)
    if best_knn:
        print(f"> Loaded kNN best params for {cfg.SIMILARITY_METHOD}  "
              f"(K={k}, MIN_GAP={min_gap})")
    config = {
        "TICKER":               cfg.TICKER,
        "PERIOD":               cfg.PERIOD,
        "INTERVAL":             cfg.INTERVAL,
        "CACHE_PATH":           cfg.CACHE_PATH,
        "WINDOW_LEN":           cfg.WINDOW_LEN,
        "FORECAST_LEN":         cfg.FORECAST_LEN,
        "BARS_BACK":            cfg.BARS_BACK,
        "SIMILARITY_METHOD":    cfg.SIMILARITY_METHOD,
        "INPUT_TYPE":           cfg.INPUT_TYPE,
        "K":                    k,
        "MIN_GAP":              min_gap,
        "CONFIDENCE_BANDS":     cfg.CONFIDENCE_BANDS,
        "FEATURE_COLS":         getattr(cfg, "FEATURE_COLS", None),
        "BACKTEST_MONTHS":      months,
        "BACKTEST_STEP":        step,
        "BACKTEST_FORECAST_DAYS": forecast_days,
    }
    if start_date:
        config["BACKTEST_START"] = start_date
    if end_date:
        config["BACKTEST_END"] = end_date
    config["ML_TRAINING_LOOKBACK_BARS"] = getattr(cfg, "ML_TRAINING_LOOKBACK_BARS", 1500)
    config["XGB_USE_SPY"]  = getattr(cfg, "XGB_USE_SPY",  True)
    config["LGBM_USE_VIX"] = getattr(cfg, "LGBM_USE_VIX", True)
    config["LGBM_USE_SPY"] = getattr(cfg, "LGBM_USE_SPY", True)
    config["RF_USE_SPY"]   = getattr(cfg, "RF_USE_SPY",   True)
    return config


def run_knn_backtest_cli(args) -> None:
    config = _build_config(args.months, args.step, args.forecast_days,
                           start_date=args.start, end_date=args.end)

    print()
    df = fetch_data(config["TICKER"], config["PERIOD"], config["INTERVAL"], config["CACHE_PATH"])
    print()

    print("=" * 50)
    print("Running Walk-Forward Backtest (k-NN)...")
    print("=" * 50)
    print(f"Date Range:  {df.index[0].date()} to {df.index[-1].date()}")
    if args.start and args.end:
        print(f"Test Period: {args.start} → {args.end} | Step: {args.step} days")
    else:
        print(f"Test Period: Last {args.months} months | Step: {args.step} days")
    print()

    result = run_backtest(df, config)
    if not result:
        print("Backtest failed — no results generated.")
        return

    print("=" * 50)
    print("BACKTEST RESULTS — k-NN (Walk-Forward)")
    print("=" * 50)
    print(f"Date Range:           {result['date_range'][0].date()} to {result['date_range'][1].date()}")
    print(f"Total Forecasts:      {result['total_forecasts']}")
    print(f"Method:               {config['SIMILARITY_METHOD']}  |  k={config['K']}")
    print()
    print(f"MAE  (Day {args.forecast_days}):         ${result['mae']:.2f}")
    print(f"RMSE (Day {args.forecast_days}):         ${result['rmse']:.2f}")
    print(f"MAPE (Day {args.forecast_days}):         {result['mape']:.1f}%")
    print()
    print(f"Directional Accuracy: {result['directional_accuracy']:.1f}%"
          f"  ({int(result['directional_accuracy'] * result['total_forecasts'] / 100)}"
          f" / {result['total_forecasts']})")
    print(f"Cone Hit Rate:        {result['cone_hit_rate']:.1f}%"
          f"  ({int(result['cone_hit_rate'] * result['total_forecasts'] / 100)}"
          f" / {result['total_forecasts']})")
    print()
    if result["daily_accuracy"]:
        print("=== DAILY BREAKDOWN (MAE) ===")
        for day_key, mae_val in sorted(result["daily_accuracy"].items(),
                                       key=lambda x: int(x[0].split("_")[1])):
            print(f"  Day {day_key.replace('day_', ''):<2}:  ${mae_val:.2f}")
    print("=" * 50)

    ticker_dir = f"output/{config['TICKER']}"
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = f"{ticker_dir}/backtest_results_knn_{config['TICKER']}.csv"
    result["summary_metrics"].to_csv(csv_path, index=False)
    print(f"\n> Results saved to {csv_path}")


def run_xgboost_backtest_cli(args) -> None:
    config = _build_config(args.months, args.step, args.forecast_days,
                           start_date=args.start, end_date=args.end)

    print()
    df = fetch_data(config["TICKER"], config["PERIOD"], config["INTERVAL"], config["CACHE_PATH"])
    print()

    # Load saved best XGBoost params if available
    saved_params = load_best_xgb_params(config["TICKER"])
    if saved_params:
        print(f"> Loaded XGBoost best params from output/{config['TICKER']}/best_params_xgb.txt  "
              f"(n_estimators={saved_params.get('n_estimators')}, "
              f"max_depth={saved_params.get('max_depth')}, "
              f"lr={saved_params.get('learning_rate')})")
    else:
        print("> No best_params_xgb.txt found — using default XGBoost params. "
              "Run: venv/bin/python param_sweep.py xgboost")
    print()

    print("=" * 50)
    print("Running Walk-Forward Backtest (XGBoost)...")
    print("=" * 50)
    print(f"Date Range:  {df.index[0].date()} to {df.index[-1].date()}")
    if args.start and args.end:
        print(f"Test Period: {args.start} → {args.end} | Step: {args.step} days")
    else:
        print(f"Test Period: Last {args.months} months | Step: {args.step} days")
    print()

    result = run_xgb_backtest(df, config, params=saved_params or None)
    if not result:
        print("Backtest failed — no results generated.")
        return

    print("=" * 50)
    print("BACKTEST RESULTS — XGBoost (Walk-Forward)")
    print("=" * 50)
    print(f"Date Range:           {result['date_range'][0].date()} to {result['date_range'][1].date()}")
    print(f"Total Forecasts:      {result['total_forecasts']}")
    print()
    print(f"MAE  (Day {args.forecast_days}):         ${result['mae']:.2f}")
    print(f"MAPE (Day {args.forecast_days}):         {result['mape']:.1f}%")
    print()
    print(f"Directional Accuracy: {result['directional_accuracy']:.1f}%"
          f"  ({int(result['directional_accuracy'] * result['total_forecasts'] / 100)}"
          f" / {result['total_forecasts']})")
    print(f"Cone Hit Rate:        {result['cone_hit_rate']:.1f}%"
          f"  ({int(result['cone_hit_rate'] * result['total_forecasts'] / 100)}"
          f" / {result['total_forecasts']})")
    print()
    if result["daily_accuracy"]:
        print("=== DAILY BREAKDOWN (MAE) ===")
        for day_key, mae_val in sorted(result["daily_accuracy"].items(),
                                       key=lambda x: int(x[0].split("_")[1])):
            print(f"  Day {day_key.replace('day_', ''):<2}:  ${mae_val:.2f}")
    print("=" * 50)

    ticker_dir = f"output/{config['TICKER']}"
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = f"{ticker_dir}/backtest_results_xgboost_{config['TICKER']}.csv"
    result["summary_metrics"].to_csv(csv_path, index=False)
    print(f"\n> Results saved to {csv_path}")


def run_lgbm_backtest_cli(args) -> None:
    config = _build_config(args.months, args.step, args.forecast_days,
                           start_date=args.start, end_date=args.end)

    print()
    df = fetch_data(config["TICKER"], config["PERIOD"], config["INTERVAL"], config["CACHE_PATH"])
    print()

    saved_params = load_best_lgbm_params(config["TICKER"])
    if saved_params:
        print(f"> Loaded LightGBM best params from output/{config['TICKER']}/best_params_lgbm.txt")
    else:
        print("> No best_params_lgbm.txt found — using default LightGBM params.")
    print()

    print("=" * 50)
    print("Running Walk-Forward Backtest (LightGBM)...")
    print("=" * 50)
    print(f"Date Range:  {df.index[0].date()} to {df.index[-1].date()}")
    if args.start and args.end:
        print(f"Test Period: {args.start} → {args.end} | Step: {args.step} days")
    else:
        print(f"Test Period: Last {args.months} months | Step: {args.step} days")
    print()

    result = run_lgbm_backtest(df, config, params=saved_params or None)
    if not result:
        print("Backtest failed — no results generated.")
        return

    print("=" * 50)
    print("BACKTEST RESULTS — LightGBM (Walk-Forward)")
    print("=" * 50)
    print(f"Date Range:           {result['date_range'][0].date()} to {result['date_range'][1].date()}")
    print(f"Total Forecasts:      {result['total_forecasts']}")
    print()
    print(f"MAE  (Day {args.forecast_days}):         ${result['mae']:.2f}")
    print(f"MAPE (Day {args.forecast_days}):         {result['mape']:.1f}%")
    print()
    print(f"Directional Accuracy: {result['directional_accuracy']:.1f}%"
          f"  ({int(result['directional_accuracy'] * result['total_forecasts'] / 100)}"
          f" / {result['total_forecasts']})")
    print(f"Cone Hit Rate:        {result['cone_hit_rate']:.1f}%"
          f"  ({int(result['cone_hit_rate'] * result['total_forecasts'] / 100)}"
          f" / {result['total_forecasts']})")
    print()
    if result["daily_accuracy"]:
        print("=== DAILY BREAKDOWN (MAE) ===")
        for day_key, mae_val in sorted(result["daily_accuracy"].items(),
                                       key=lambda x: int(x[0].split("_")[1])):
            print(f"  Day {day_key.replace('day_', ''):<2}:  ${mae_val:.2f}")
    print("=" * 50)

    ticker_dir = f"output/{config['TICKER']}"
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = f"{ticker_dir}/backtest_results_lgbm_{config['TICKER']}.csv"
    result["summary_metrics"].to_csv(csv_path, index=False)
    print(f"\n> Results saved to {csv_path}")


def run_rf_backtest_cli(args) -> None:
    config = _build_config(args.months, args.step, args.forecast_days,
                           start_date=args.start, end_date=args.end)
    print()
    df = fetch_data(config["TICKER"], config["PERIOD"], config["INTERVAL"], config["CACHE_PATH"])
    print()

    saved_params = load_best_rf_params(config["TICKER"])
    if saved_params:
        print(f"> Loaded RF best params from output/{config['TICKER']}/best_params_rf.txt")
    else:
        print("> No best_params_rf.txt found — using default RF params.")
    print()

    print("=" * 50)
    print("Running Walk-Forward Backtest (Random Forest)...")
    print("=" * 50)
    print(f"Date Range:  {df.index[0].date()} to {df.index[-1].date()}")
    if args.start and args.end:
        print(f"Test Period: {args.start} → {args.end} | Step: {args.step} days")
    else:
        print(f"Test Period: Last {args.months} months | Step: {args.step} days")
    print()

    result = run_rf_backtest(df, config, params=saved_params or None)
    if not result:
        print("Backtest failed — no results generated.")
        return

    print("=" * 50)
    print("BACKTEST RESULTS — Random Forest (Walk-Forward)")
    print("=" * 50)
    print(f"Date Range:           {result['date_range'][0].date()} to {result['date_range'][1].date()}")
    print(f"Total Forecasts:      {result['total_forecasts']}")
    print()
    print(f"MAE  (Day {args.forecast_days}):         ${result['mae']:.2f}")
    print(f"MAPE (Day {args.forecast_days}):         {result['mape']:.1f}%")
    print()
    print(f"Directional Accuracy: {result['directional_accuracy']:.1f}%"
          f"  ({int(result['directional_accuracy'] * result['total_forecasts'] / 100)}"
          f" / {result['total_forecasts']})")
    print(f"Cone Hit Rate:        {result['cone_hit_rate']:.1f}%"
          f"  ({int(result['cone_hit_rate'] * result['total_forecasts'] / 100)}"
          f" / {result['total_forecasts']})")
    print()
    if result["daily_accuracy"]:
        print("=== DAILY BREAKDOWN (MAE) ===")
        for day_key, mae_val in sorted(result["daily_accuracy"].items(),
                                       key=lambda x: int(x[0].split("_")[1])):
            print(f"  Day {day_key.replace('day_', ''):<2}:  ${mae_val:.2f}")
    print("=" * 50)

    ticker_dir = f"output/{config['TICKER']}"
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = f"{ticker_dir}/backtest_results_rf_{config['TICKER']}.csv"
    result["summary_metrics"].to_csv(csv_path, index=False)
    print(f"\n> Results saved to {csv_path}")


SIMILARITY_METHODS = ["spearman", "pearson", "cosine", "euclidean", "kendall", "manhattan"]


def run_methods_backtest_cli(args) -> None:
    """Backtest all 6 similarity methods and print a leaderboard."""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    print()
    base_config = _build_config(args.months, args.step, args.forecast_days,
                                start_date=args.start, end_date=args.end)
    df = fetch_data(base_config["TICKER"], base_config["PERIOD"],
                    base_config["INTERVAL"], base_config["CACHE_PATH"])
    print()

    period_str = (f"{args.start} → {args.end}" if args.start and args.end
                  else f"Last {args.months} months")
    print("=" * 65)
    print(f"SIMILARITY METHOD BACKTEST — {base_config['TICKER']}")
    print(f"Period: {period_str}  |  Step: {args.step} days  |  "
          f"Forecast: {args.forecast_days} days")
    print("=" * 65)
    print()

    rows = []
    for method in SIMILARITY_METHODS:
        print(f"  {method.upper():<12} ", end="", flush=True)
        method_params = _load_best_knn_params(base_config["TICKER"], method=method)
        config = {**base_config,
                  "SIMILARITY_METHOD": method,
                  "K":       method_params.get("K",       base_config["K"]),
                  "MIN_GAP": method_params.get("MIN_GAP", base_config["MIN_GAP"])}
        try:
            buf = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(buf):
                result = run_backtest(df, config)
            if not result:
                print("no results")
                continue
            print(f"Dir: {result['directional_accuracy']:.1f}%  "
                  f"MAE: ${result['mae']:.2f}  "
                  f"MAPE: {result['mape']:.1f}%  "
                  f"Cone: {result['cone_hit_rate']:.1f}%  "
                  f"N: {result['total_forecasts']}")
            rows.append({
                "method":           method,
                "dir_accuracy_%":   round(result["directional_accuracy"], 1),
                "mae":              round(result["mae"],  2),
                "mape_%":           round(result["mape"], 2),
                "cone_hit_%":       round(result["cone_hit_rate"], 1),
                "total_forecasts":  result["total_forecasts"],
            })
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")

    if not rows:
        print("No results generated.")
        return

    # Composite score: MAPE 40% + direction 40% + cone 20% (lower = better)
    import pandas as pd, numpy as np
    df_res = pd.DataFrame(rows)
    mape_n = (df_res["mape_%"] - df_res["mape_%"].min()) / (df_res["mape_%"].max() - df_res["mape_%"].min() + 1e-9)
    dir_n  = (df_res["dir_accuracy_%"].max() - df_res["dir_accuracy_%"]) / (df_res["dir_accuracy_%"].max() - df_res["dir_accuracy_%"].min() + 1e-9)
    cone_n = (df_res["cone_hit_%"].max() - df_res["cone_hit_%"]) / (df_res["cone_hit_%"].max() - df_res["cone_hit_%"].min() + 1e-9)
    df_res["score"] = (0.4 * mape_n + 0.4 * dir_n + 0.2 * cone_n).round(4)
    df_res = df_res.sort_values("score").reset_index(drop=True)
    df_res.insert(0, "rank", df_res.index + 1)

    W = 75
    print()
    print("=" * W)
    print("LEADERBOARD  (score: MAPE 40% + Direction 40% + Cone 20% | lower = better)")
    print("=" * W)
    print(f"  {'Rank':<5} {'Method':<12} {'Dir%':>7}  {'MAE':>8}  {'MAPE%':>7}  {'Cone%':>7}  {'Score':>7}")
    print(f"  {'-'*5} {'-'*12} {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
    for _, r in df_res.iterrows():
        marker = "  ← BEST" if r["rank"] == 1 else ""
        print(f"  {int(r['rank']):<5} {r['method']:<12} {r['dir_accuracy_%']:>6.1f}%  "
              f"${r['mae']:>7.2f}  {r['mape_%']:>6.1f}%  {r['cone_hit_%']:>6.1f}%  "
              f"{r['score']:>7.4f}{marker}")

    best = df_res.iloc[0]
    print()
    print(f"  Best method: {best['method'].upper()}  "
          f"(Dir: {best['dir_accuracy_%']:.1f}%  MAE: ${best['mae']:.2f}  "
          f"Cone: {best['cone_hit_%']:.1f}%)")
    print("=" * W)

    # Save
    ticker_dir = f"output/{base_config['TICKER']}"
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = f"{ticker_dir}/backtest_methods_{base_config['TICKER']}.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n> Results saved to {csv_path}")


def run_knn2_backtest_cli(args) -> None:
    config = _build_config(args.months, args.step, args.forecast_days,
                           start_date=args.start, end_date=args.end)
    print()
    df_raw = fetch_data(config["TICKER"], config["PERIOD"], config["INTERVAL"],
                        config["CACHE_PATH"])
    df = compute_features(df_raw)
    print()

    saved_params = load_knn2_params(config["TICKER"])

    period_str = (f"{args.start} → {args.end}" if args.start and args.end
                  else f"Last {args.months} months")
    print("=" * 55)
    print("Running Walk-Forward Backtest (knn2 enhanced)...")
    print("=" * 55)
    print(f"Period: {period_str}  |  Step: {args.step} days  |  "
          f"Forecast: {args.forecast_days} days")
    print(f"Half-life: {saved_params.get('half_life_days', 250)} days  |  "
          f"Distance pct: {saved_params.get('distance_threshold_pct', 20)}")
    print()

    result = run_knn2_backtest(df, config, params=saved_params or None)
    if not result:
        print("Backtest failed — no results generated.")
        return

    W = 55
    print("=" * W)
    print("BACKTEST RESULTS — knn2 (Enhanced Walk-Forward)")
    print("=" * W)
    print(f"Date Range:           {result['date_range'][0].date()} to {result['date_range'][1].date()}")
    print(f"Total Forecasts:      {result['total_forecasts']}")
    print()
    print(f"MAE  (Day {args.forecast_days}):         ${result['mae']:.2f}")
    print(f"MAPE (Day {args.forecast_days}):         {result['mape']:.1f}%")
    print()
    print(f"Directional Accuracy (Day 5): {result['directional_accuracy']:.1f}%"
          f"  ({int(result['directional_accuracy'] * result['total_forecasts'] / 100)}"
          f" / {result['total_forecasts']})")
    print(f"Cone Hit Rate:        {result['cone_hit_rate']:.1f}%")
    print()
    if result["daily_accuracy"]:
        print("=== DAILY BREAKDOWN (MAE) ===")
        for day_key, mae_val in sorted(result["daily_accuracy"].items(),
                                       key=lambda x: int(x[0].split("_")[1])):
            print(f"  Day {day_key.replace('day_', ''):<2}:  ${mae_val:.2f}")
    print("=" * W)

    ticker_dir = f"output/{config['TICKER']}"
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = f"{ticker_dir}/backtest_results_knn2_{config['TICKER']}.csv"
    result["summary_metrics"].to_csv(csv_path, index=False)
    print(f"\n> Results saved to {csv_path}")


def run_leaderboard_cli(args) -> None:
    """Run all models on the same date range and print a unified leaderboard."""
    import io
    import numpy as np
    from contextlib import redirect_stdout, redirect_stderr

    ticker = cfg.TICKER
    print()
    base_config = _build_config(args.months, args.step, args.forecast_days,
                                start_date=args.start, end_date=args.end)
    df = fetch_data(base_config["TICKER"], base_config["PERIOD"],
                    base_config["INTERVAL"], base_config["CACHE_PATH"])
    print()

    period_str = (f"{args.start} → {args.end}" if args.start and args.end
                  else f"Last {args.months} months")
    W = 78
    print("=" * W)
    print(f"FULL MODEL LEADERBOARD — {ticker}")
    print(f"Period: {period_str}  |  Step: {args.step} days  |  Forecast: {args.forecast_days} days")
    print(f"Models: 8 kNN methods + XGBoost + LightGBM + Random Forest")
    print("=" * W)
    print()

    rows = []

    # --- 8 kNN similarity methods ---
    for method in SIMILARITY_METHODS:
        print(f"  {method.upper():<14} ", end="", flush=True)
        method_params = _load_best_knn_params(ticker, method=method)
        config = {**base_config,
                  "SIMILARITY_METHOD": method,
                  "K":       method_params.get("K",       base_config["K"]),
                  "MIN_GAP": method_params.get("MIN_GAP", base_config["MIN_GAP"])}
        try:
            buf = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(buf):
                result = run_backtest(df, config)
            if not result:
                print("no results")
                continue
            print(f"Dir: {result['directional_accuracy']:.1f}%  "
                  f"MAE: ${result['mae']:.2f}  "
                  f"MAPE: {result['mape']:.1f}%  "
                  f"Cone: {result['cone_hit_rate']:.1f}%")
            rows.append({
                "model":          method,
                "dir_accuracy_%": round(result["directional_accuracy"], 1),
                "mae":            round(result["mae"], 2),
                "mape_%":         round(result["mape"], 2),
                "cone_hit_%":     round(result["cone_hit_rate"], 1),
                "n_forecasts":    result["total_forecasts"],
            })
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")

    # --- XGBoost ---
    print(f"  {'XGBOOST':<14} ", end="", flush=True)
    try:
        saved_xgb = load_best_xgb_params(ticker)
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            result = run_xgb_backtest(df, base_config, params=saved_xgb or None)
        if result:
            print(f"Dir: {result['directional_accuracy']:.1f}%  "
                  f"MAE: ${result['mae']:.2f}  "
                  f"MAPE: {result['mape']:.1f}%  "
                  f"Cone: {result['cone_hit_rate']:.1f}%")
            rows.append({
                "model":          "xgboost",
                "dir_accuracy_%": round(result["directional_accuracy"], 1),
                "mae":            round(result["mae"], 2),
                "mape_%":         round(result["mape"], 2),
                "cone_hit_%":     round(result["cone_hit_rate"], 1),
                "n_forecasts":    result["total_forecasts"],
            })
        else:
            print("no results")
    except Exception as e:
        print(f"ERROR: {str(e)[:50]}")

    # --- LightGBM ---
    print(f"  {'LIGHTGBM':<14} ", end="", flush=True)
    try:
        saved_lgbm = load_best_lgbm_params(ticker)
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            result = run_lgbm_backtest(df, base_config, params=saved_lgbm or None)
        if result:
            print(f"Dir: {result['directional_accuracy']:.1f}%  "
                  f"MAE: ${result['mae']:.2f}  "
                  f"MAPE: {result['mape']:.1f}%  "
                  f"Cone: {result['cone_hit_rate']:.1f}%")
            rows.append({
                "model":          "lightgbm",
                "dir_accuracy_%": round(result["directional_accuracy"], 1),
                "mae":            round(result["mae"], 2),
                "mape_%":         round(result["mape"], 2),
                "cone_hit_%":     round(result["cone_hit_rate"], 1),
                "n_forecasts":    result["total_forecasts"],
            })
        else:
            print("no results")
    except Exception as e:
        print(f"ERROR: {str(e)[:50]}")

    # --- Random Forest ---
    print(f"  {'RANDOMFOREST':<14} ", end="", flush=True)
    try:
        saved_rf = load_best_rf_params(ticker)
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            result = run_rf_backtest(df, base_config, params=saved_rf or None)
        if result:
            print(f"Dir: {result['directional_accuracy']:.1f}%  "
                  f"MAE: ${result['mae']:.2f}  "
                  f"MAPE: {result['mape']:.1f}%  "
                  f"Cone: {result['cone_hit_rate']:.1f}%")
            rows.append({
                "model":          "randomforest",
                "dir_accuracy_%": round(result["directional_accuracy"], 1),
                "mae":            round(result["mae"], 2),
                "mape_%":         round(result["mape"], 2),
                "cone_hit_%":     round(result["cone_hit_rate"], 1),
                "n_forecasts":    result["total_forecasts"],
            })
        else:
            print("no results")
    except Exception as e:
        print(f"ERROR: {str(e)[:50]}")

    # --- knn2 (enhanced) ---
    print(f"  {'KNN2':<14} ", end="", flush=True)
    try:
        df_feat    = compute_features(df)
        saved_knn2 = load_knn2_params(ticker)
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            result = run_knn2_backtest(df_feat, base_config, params=saved_knn2 or None)
        if result:
            print(f"Dir: {result['directional_accuracy']:.1f}%  "
                  f"MAE: ${result['mae']:.2f}  "
                  f"MAPE: {result['mape']:.1f}%  "
                  f"Cone: {result['cone_hit_rate']:.1f}%")
            rows.append({
                "model":          "knn2",
                "dir_accuracy_%": round(result["directional_accuracy"], 1),
                "mae":            round(result["mae"], 2),
                "mape_%":         round(result["mape"], 2),
                "cone_hit_%":     round(result["cone_hit_rate"], 1),
                "n_forecasts":    result["total_forecasts"],
            })
        else:
            print("no results")
    except Exception as e:
        print(f"ERROR: {str(e)[:50]}")

    if not rows:
        print("No results generated.")
        return

    # Composite score: MAPE 40% + Direction 40% + Cone 20% (lower = better)
    df_res = pd.DataFrame(rows)
    mape_n = (df_res["mape_%"] - df_res["mape_%"].min()) / (df_res["mape_%"].max() - df_res["mape_%"].min() + 1e-9)
    dir_n  = (df_res["dir_accuracy_%"].max() - df_res["dir_accuracy_%"]) / (df_res["dir_accuracy_%"].max() - df_res["dir_accuracy_%"].min() + 1e-9)
    cone_n = (df_res["cone_hit_%"].max() - df_res["cone_hit_%"]) / (df_res["cone_hit_%"].max() - df_res["cone_hit_%"].min() + 1e-9)
    df_res["score"] = (0.4 * mape_n + 0.4 * dir_n + 0.2 * cone_n).round(4)
    df_res = df_res.sort_values("score").reset_index(drop=True)
    df_res.insert(0, "rank", df_res.index + 1)

    print()
    print("=" * W)
    print("LEADERBOARD — All Models  (MAPE 40% + Direction 40% + Cone 20%  |  lower score = better)")
    print("=" * W)
    print(f"  {'Rank':<5} {'Model':<14} {'Dir%':>7}  {'MAE':>8}  {'MAPE%':>7}  {'Cone%':>7}  {'N':>5}  {'Score':>7}")
    print(f"  {'-'*5} {'-'*14} {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*7}")
    for _, r in df_res.iterrows():
        marker = "  ← BEST" if r["rank"] == 1 else ""
        print(f"  {int(r['rank']):<5} {r['model']:<14} {r['dir_accuracy_%']:>6.1f}%  "
              f"${r['mae']:>7.2f}  {r['mape_%']:>6.1f}%  {r['cone_hit_%']:>6.1f}%  "
              f"{int(r['n_forecasts']):>5}  {r['score']:>7.4f}{marker}")

    best = df_res.iloc[0]
    print()
    print(f"  Best model: {best['model'].upper()}  "
          f"(Dir: {best['dir_accuracy_%']:.1f}%  "
          f"MAE: ${best['mae']:.2f}  "
          f"MAPE: {best['mape_%']:.1f}%  "
          f"Cone: {best['cone_hit_%']:.1f}%)")
    print("=" * W)

    ticker_dir = f"output/{ticker}"
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = f"{ticker_dir}/backtest_leaderboard_{ticker}.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n> Results saved to {csv_path}")


def main() -> None:
    _MODES = ("knn", "knn2", "xgboost", "lgbm", "rf", "methods", "leaderboard")
    mode = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in _MODES else None

    # Strip mode argument before argparse sees it
    argv = [a for a in sys.argv[1:] if a not in _MODES]

    parser = argparse.ArgumentParser(
        description="Walk-forward backtest — kNN, XGBoost, LightGBM, RF, all methods, or unified leaderboard"
    )
    parser.add_argument("--months",        type=int, default=cfg.BACKTEST_MONTHS)
    parser.add_argument("--step",          type=int, default=cfg.BACKTEST_STEP)
    parser.add_argument("--forecast-days", type=int, default=cfg.BACKTEST_FORECAST_DAYS,
                        dest="forecast_days")
    parser.add_argument("--start", type=str, default=getattr(cfg, "BACKTEST_START", None),
                        help="Backtest start date YYYY-MM-DD (overrides --months)")
    parser.add_argument("--end",   type=str, default=getattr(cfg, "BACKTEST_END", None),
                        help="Backtest end date YYYY-MM-DD (overrides --months)")
    args = parser.parse_args(argv)

    if mode == "knn2":
        run_knn2_backtest_cli(args)
    elif mode == "xgboost":
        run_xgboost_backtest_cli(args)
    elif mode == "lgbm":
        run_lgbm_backtest_cli(args)
    elif mode == "rf":
        run_rf_backtest_cli(args)
    elif mode == "methods":
        run_methods_backtest_cli(args)
    elif mode == "leaderboard":
        run_leaderboard_cli(args)
    else:
        run_knn_backtest_cli(args)


if __name__ == "__main__":
    main()
