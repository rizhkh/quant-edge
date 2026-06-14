import os
import io
import sys
import pandas as pd
import config as cfg

from data.fetcher         import fetch_data
from forecaster.analog    import run_analog_forecast
from forecaster.knn       import run_knn_forecast
from forecaster.knn2      import run_knn2_forecast
from features.engineer    import compute_features
from models.xgboost_model  import run_xgboost_forecast
from models.lightgbm_model import run_lgbm_forecast, load_best_lgbm_params
from models.rf_model       import run_rf_forecast,   load_best_rf_params
from contextlib import redirect_stdout, redirect_stderr, nullcontext


SIMILARITY_METHODS = ["spearman", "pearson", "cosine", "euclidean", "kendall", "manhattan"]


def discover_tickers() -> list:
    """Return ticker names from output/ subdirectories, skipping 'copy'/'test' folders."""
    if not os.path.isdir("output"):
        return []
    tickers = []
    for name in sorted(os.listdir("output")):
        low = name.lower()
        if "copy" in low or "test" in low:
            continue
        if os.path.isdir(f"output/{name}"):
            tickers.append(name)
    return tickers


def load_best_params(ticker: str, method: str = None) -> dict:
    """Load K and MIN_GAP for kNN from params/best_params_{method}.yaml or params/best_params_knn.yaml."""
    from utils.params_io import load_knn_params
    return load_knn_params(ticker, method=method)


def build_config(method: str, ticker: str = None) -> dict:
    t    = ticker or cfg.TICKER
    best = load_best_params(t, method=method)
    k       = best.get("K",       cfg.K)
    min_gap = best.get("MIN_GAP", cfg.MIN_GAP)
    method_file = f"best_params_{method}.txt" if os.path.exists(f"output/{t}/best_params_{method}.txt") else "best_params.txt"
    if best:
        print(f"> Loaded best params from output/{t}/{method_file}  "
              f"(K={k}, MIN_GAP={min_gap})")
    else:
        print(f"> No best_params found — using config.py defaults  "
              f"(K={k}, MIN_GAP={min_gap})")
    return {
        "TICKER":            t,
        "PERIOD":            cfg.PERIOD,
        "INTERVAL":          cfg.INTERVAL,
        "CACHE_PATH":        f"data/{t}_max.csv",
        "WINDOW_LEN":        cfg.WINDOW_LEN,
        "FORECAST_LEN":      cfg.FORECAST_LEN,
        "BARS_BACK":         cfg.BARS_BACK,
        "SIMILARITY_METHOD": method,
        "INPUT_TYPE":        cfg.INPUT_TYPE,
        "K":                 k,
        "MIN_GAP":           min_gap,
        "CONFIDENCE_BANDS":  cfg.CONFIDENCE_BANDS,
        "SHOW_ALL_PATHS":    cfg.SHOW_ALL_PATHS,
        "SHOW_ZIGZAG":       cfg.SHOW_ZIGZAG,
        "ZIGZAG_LEGS":       cfg.ZIGZAG_LEGS,
        "FEATURE_COLS":      getattr(cfg, "FEATURE_COLS", None),
        "ML_TRAINING_LOOKBACK_BARS": getattr(cfg, "ML_TRAINING_LOOKBACK_BARS", 1500),
        "XGB_USE_SPY":       getattr(cfg, "XGB_USE_SPY",  False),
        "XGB_USE_SECTOR":    getattr(cfg, "XGB_USE_SECTOR",   True),
        "XGB_USE_VIX":       getattr(cfg, "XGB_USE_VIX",      True),
        "XGB_USE_EARNINGS":  getattr(cfg, "XGB_USE_EARNINGS", True),
        "SECTOR_MAP":        getattr(cfg, "SECTOR_MAP", {}),
        "LGBM_USE_VIX":      getattr(cfg, "LGBM_USE_VIX", True),
        "LGBM_USE_SPY":      getattr(cfg, "LGBM_USE_SPY", True),
        "RF_USE_SPY":        getattr(cfg, "RF_USE_SPY",   True),
    }


# Column-prefix groups for selective merges. Each group's columns can be
# preserved from the existing CSV when only a subset of models is being refreshed.
MODEL_GROUP_PREFIXES = {
    "knn":  ["analog_", "knn_", "spearman_", "pearson_", "cosine_", "euclidean_",
             "mse_", "kendall_", "manhattan_", "mahalanobis_"],
    "xgb":  ["xgb_"],
    "lgb":  ["lgb_"],
    "rf":   ["rf_"],
    "knn2": ["knn2_"],
}
MODEL_GROUP_EXACT = {
    "knn":  ["knn_method"],
}


def _cols_in_group(df_columns, group: str) -> list:
    prefixes = MODEL_GROUP_PREFIXES.get(group, [])
    exact    = MODEL_GROUP_EXACT.get(group, [])
    out = [c for c in df_columns if any(c.startswith(p) for p in prefixes) or c in exact]
    return out


def save_forecast_results(new_forecast_df: pd.DataFrame, ticker: str,
                          preserve_groups: list = None) -> str:
    """
    Append/merge new forecast into forecast_results_{T}.csv.

    Default behaviour (preserve_groups=None): full-row merge, only keeping
    knn2_* columns from history (existing pattern).

    When preserve_groups is set (e.g. ["xgb", "lgb", "rf", "knn2"]), columns
    matching those groups are NOT overwritten — they are read from the existing
    CSV and merged into the result. Use this when only a subset of models is
    being refreshed (e.g. `main.py run_all knn`).
    """
    import shutil
    ticker_dir = f"output/{ticker}"
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = f"{ticker_dir}/forecast_results_{ticker}.csv"

    # Archive existing file before overwriting
    if os.path.exists(csv_path):
        archive_dir = f"{ticker_dir}/archive"
        os.makedirs(archive_dir, exist_ok=True)
        today_str   = pd.Timestamp.now().strftime("%Y-%m-%d")
        archive_path = f"{archive_dir}/{today_str}_forecast_results_{ticker}.csv"
        shutil.copy2(csv_path, archive_path)

    new_forecast_df = new_forecast_df.copy()
    new_forecast_df['date'] = pd.to_datetime(new_forecast_df['date'])

    # Round all price columns to 2 decimal places (cents)
    num_cols = new_forecast_df.select_dtypes(include='number').columns.difference(['day'])
    new_forecast_df[num_cols] = new_forecast_df[num_cols].round(2)

    today = pd.Timestamp.now().normalize()
    today_str = today.strftime("%Y-%m-%d")

    if 'last_updated' not in new_forecast_df.columns:
        date_col_pos = new_forecast_df.columns.get_loc('date')
        new_forecast_df.insert(date_col_pos, 'last_updated', today_str)

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        existing_df['date'] = pd.to_datetime(existing_df['date'])

        if 'last_updated' not in existing_df.columns:
            date_col_pos = existing_df.columns.get_loc('date')
            existing_df.insert(date_col_pos, 'last_updated', '')

        # === Column-merge mode: only some models were refreshed ===
        # Semantic: historical rows (dates not in new_forecast_df) are preserved
        # exactly as-is. For dates IN new_forecast_df, the refreshed model columns
        # come from new_forecast_df, and the preserve_groups columns come from
        # existing_df (so other models' data isn't wiped).
        if preserve_groups:
            preserve_cols = []
            for g in preserve_groups:
                preserve_cols.extend(_cols_in_group(existing_df.columns, g))
            preserve_cols = list(dict.fromkeys(preserve_cols))   # dedupe, preserve order

            new_dates = set(new_forecast_df['date'])

            # Drop columns from new_forecast_df that overlap with preserved (defensive)
            new_cols_to_keep = [c for c in new_forecast_df.columns if c not in preserve_cols]
            new_forecast_df = new_forecast_df[new_cols_to_keep]

            # Untouched rows: existing rows whose date is NOT in the new forecast.
            # Keep them entirely intact — every column preserved.
            untouched = existing_df[~existing_df['date'].isin(new_dates)].copy()

            # Refreshed rows: for dates in new_forecast_df, combine the new
            # model columns with the preserved-group columns pulled from existing_df.
            if preserve_cols:
                preserved_for_new = (
                    existing_df[existing_df['date'].isin(new_dates)][['date'] + preserve_cols]
                    .copy()
                )
                refreshed = new_forecast_df.merge(preserved_for_new, on='date', how='left')
            else:
                refreshed = new_forecast_df.copy()

            # Align columns: union of both, in a stable order (existing first, new appended)
            all_cols = list(existing_df.columns)
            for c in refreshed.columns:
                if c not in all_cols:
                    all_cols.append(c)
            for c in all_cols:
                if c not in untouched.columns:  untouched[c]  = pd.NA
                if c not in refreshed.columns:  refreshed[c]  = pd.NA
            untouched = untouched[all_cols]
            refreshed = refreshed[all_cols]

            combined_df = pd.concat([untouched, refreshed], ignore_index=True)
            combined_df = combined_df.sort_values('date').reset_index(drop=True)

            combined_df.to_csv(csv_path, index=False)
            return csv_path

        # === Full-pipeline mode (default): existing logic, knn2 backup only ===
        knn2_cols  = [c for c in existing_df.columns if c.startswith('knn2_')]
        knn2_backup = None
        if knn2_cols:
            knn2_backup = (existing_df[['date'] + knn2_cols]
                           .dropna(subset=knn2_cols, how='all')
                           .copy())

        historical = existing_df[existing_df['date'] <= today].copy()
        combined_df = pd.concat([historical, new_forecast_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
        combined_df = combined_df.sort_values('date').reset_index(drop=True)

        if knn2_backup is not None and len(knn2_backup) > 0:
            combined_df = combined_df.drop(
                columns=[c for c in combined_df.columns if c.startswith('knn2_')],
                errors='ignore'
            )
            combined_df = combined_df.merge(knn2_backup, on='date', how='left')
    else:
        combined_df = new_forecast_df.copy()

    combined_df.to_csv(csv_path, index=False)
    return csv_path


def run_forecast(ticker: str = None, quiet: bool = False) -> dict:
    """
    Run full forecast pipeline for one ticker.
    Returns a summary dict with success, bias, score, day30_price, etc.
    """
    t = ticker or cfg.TICKER
    summary = {"ticker": t, "success": False, "error": None}

    _buf = io.StringIO()
    _out = redirect_stdout(_buf) if quiet else nullcontext()
    _err = redirect_stderr(_buf) if quiet else nullcontext()

    try:
        with _out, _err:
            # 1. Fetch data once
            base_config = build_config(cfg.SIMILARITY_METHOD, t)
            df = fetch_data(
                base_config["TICKER"],
                base_config["PERIOD"],
                base_config["INTERVAL"],
                base_config["CACHE_PATH"],
                target_bars=5000,
            )
            print()

            # 2. Run forecasts for all similarity methods
            results_by_method = {}
            current_close = df['Close'].iloc[-1]

            print("=" * 70)
            print("RUNNING FORECASTS FOR ALL SIMILARITY METHODS")
            print("=" * 70)
            print()

            for method in SIMILARITY_METHODS:
                print(f"Testing: {method.upper():<10} ", end="", flush=True)

                config = build_config(method, t)

                try:
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        analog_result = run_analog_forecast(df, config)
                        knn_result = run_knn_forecast(df, config)

                    results_by_method[method] = {
                        "analog": analog_result,
                        "knn": knn_result,
                        "best_score": analog_result["best_score"]
                    }
                    print(f"✓ Score: {analog_result['best_score']:.3f}")
                except Exception as e:
                    print(f"✗ Error: {str(e)[:40]}")
                    continue

            # 2b. Run XGBoost forecast
            xgb_result = None
            print(f"Testing: {'XGBOOST':<10} ", end="", flush=True)
            try:
                xgb_config = build_config(cfg.SIMILARITY_METHOD, t)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    xgb_result = run_xgboost_forecast(df, xgb_config)
                print(f"✓ Samples: {xgb_result['n_train_samples']}  Features: {xgb_result['n_features']}")
            except Exception as e:
                print(f"✗ Error: {str(e)[:60]}")

            # 2c. Run LightGBM forecast
            lgb_result = None
            print(f"Testing: {'LIGHTGBM':<10} ", end="", flush=True)
            try:
                lgb_config = build_config(cfg.SIMILARITY_METHOD, t)
                saved_lgbm = load_best_lgbm_params(t)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    lgb_result = run_lgbm_forecast(df, lgb_config, params=saved_lgbm or None)
                print(f"✓ Samples: {lgb_result['n_train_samples']}  Features: {lgb_result['n_features']}")
            except Exception as e:
                print(f"✗ Error: {str(e)[:60]}")

            # 2d. Run Random Forest forecast
            rf_result = None
            print(f"Testing: {'RANDOMFOREST':<10} ", end="", flush=True)
            try:
                rf_config  = build_config(cfg.SIMILARITY_METHOD, t)
                saved_rf   = load_best_rf_params(t)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    rf_result = run_rf_forecast(df, rf_config, params=saved_rf or None)
                print(f"✓ Samples: {rf_result['n_train_samples']}  Features: {rf_result['n_features']}")
            except Exception as e:
                print(f"✗ Error: {str(e)[:60]}")

            print()
            print("=" * 70)
            print("COMPARISON: Day 5 & Day 30 Forecasts by Method")
            print("=" * 70)
            print()
            print(f"{'Model':<12} {'Score':<8} {'Day5 Low':<10} {'Day5 Med':<10} {'Day5 High':<10} {'Day30 Low':<11} {'Day30 Med':<11} {'Day30 High':<11} {'Bias':<10}")
            print("-" * 95)

            method_rows = []
            for method in SIMILARITY_METHODS:
                if method not in results_by_method:
                    continue

                result = results_by_method[method]
                score = result["best_score"]
                knn_result = result["knn"]

                day_prices = {}
                for day in [1, 5, 10, 20, 30]:
                    day_row = knn_result["forecast_cone"][knn_result["forecast_cone"]["day"] == day]
                    if len(day_row) > 0:
                        day_prices[f"day{day}_low"]    = day_row.iloc[0]["low"]
                        day_prices[f"day{day}_price"]  = day_row.iloc[0]["median"]
                        day_prices[f"day{day}_high"]   = day_row.iloc[0]["high"]
                    else:
                        day_prices[f"day{day}_low"]   = None
                        day_prices[f"day{day}_price"] = None
                        day_prices[f"day{day}_high"]  = None

                bias = knn_result["bias"]

                print(
                    f"{method:<12} {score:<8.3f} "
                    f"${day_prices.get('day5_low', 0):<9.2f} "
                    f"${day_prices.get('day5_price', 0):<9.2f} "
                    f"${day_prices.get('day5_high', 0):<9.2f} "
                    f"${day_prices.get('day30_low', 0):<10.2f} "
                    f"${day_prices.get('day30_price', 0):<10.2f} "
                    f"${day_prices.get('day30_high', 0):<10.2f} "
                    f"{bias:<10}"
                )

                method_rows.append({
                    "method":        method,
                    "score":         score,
                    "day1_low":      day_prices.get('day1_low'),
                    "day1_price":    day_prices.get('day1_price'),
                    "day1_high":     day_prices.get('day1_high'),
                    "day5_low":      day_prices.get('day5_low'),
                    "day5_price":    day_prices.get('day5_price'),
                    "day5_high":     day_prices.get('day5_high'),
                    "day10_low":     day_prices.get('day10_low'),
                    "day10_price":   day_prices.get('day10_price'),
                    "day10_high":    day_prices.get('day10_high'),
                    "day20_low":     day_prices.get('day20_low'),
                    "day20_price":   day_prices.get('day20_price'),
                    "day20_high":    day_prices.get('day20_high'),
                    "day30_low":     day_prices.get('day30_low'),
                    "day30_price":   day_prices.get('day30_price'),
                    "day30_high":    day_prices.get('day30_high'),
                    "bias":          bias,
                    "current_close": current_close,
                })

            # Add XGBoost row to comparison table
            if xgb_result is not None:
                xgb_cone = xgb_result["forecast_cone"]
                xgb_day_prices = {}
                for day in [1, 5, 10, 20, 30]:
                    row = xgb_cone[xgb_cone["day"] == day]
                    xgb_day_prices[f"day{day}_low"]   = row.iloc[0]["low"]    if not row.empty else None
                    xgb_day_prices[f"day{day}_price"]  = row.iloc[0]["median"] if not row.empty else None
                    xgb_day_prices[f"day{day}_high"]  = row.iloc[0]["high"]   if not row.empty else None
                print(
                    f"{'xgboost':<12} {'n/a':<8} "
                    f"${xgb_day_prices.get('day5_low', 0):<9.2f} "
                    f"${xgb_day_prices.get('day5_price', 0):<9.2f} "
                    f"${xgb_day_prices.get('day5_high', 0):<9.2f} "
                    f"${xgb_day_prices.get('day30_low', 0):<10.2f} "
                    f"${xgb_day_prices.get('day30_price', 0):<10.2f} "
                    f"${xgb_day_prices.get('day30_high', 0):<10.2f} "
                    f"{xgb_result['bias']:<10}"
                )
                xgb_cone_full = xgb_result["forecast_cone"]
                def _xgb(day, col):
                    r = xgb_cone_full[xgb_cone_full["day"] == day]
                    return r.iloc[0][col] if not r.empty else None
                method_rows.append({
                    "method":         "xgboost",
                    "score":          None,
                    "day1_low":       _xgb(1, "low"),
                    "day1_price":     xgb_day_prices.get('day1_price'),
                    "day1_high":      _xgb(1, "high"),
                    "day5_low":       _xgb(5, "low"),
                    "day5_price":     xgb_day_prices.get('day5_price'),
                    "day5_high":      _xgb(5, "high"),
                    "day10_low":      _xgb(10, "low"),
                    "day10_price":    xgb_day_prices.get('day10_price'),
                    "day10_high":     _xgb(10, "high"),
                    "day20_low":      _xgb(20, "low"),
                    "day20_price":    xgb_day_prices.get('day20_price'),
                    "day20_high":     _xgb(20, "high"),
                    "day30_low":      _xgb(30, "low"),
                    "day30_price":    xgb_day_prices.get('day30_price'),
                    "day30_high":     _xgb(30, "high"),
                    "bias":           xgb_result["bias"],
                    "current_close":  current_close,
                })

            # Add LightGBM row to comparison table
            if lgb_result is not None:
                lgb_cone_full = lgb_result["forecast_cone"]
                def _lgb(day, col):
                    r = lgb_cone_full[lgb_cone_full["day"] == day]
                    return r.iloc[0][col] if not r.empty else None
                lgb_day_prices = {}
                for day in [1, 5, 10, 20, 30]:
                    row = lgb_cone_full[lgb_cone_full["day"] == day]
                    lgb_day_prices[f"day{day}_low"]   = row.iloc[0]["low"]    if not row.empty else None
                    lgb_day_prices[f"day{day}_price"]  = row.iloc[0]["median"] if not row.empty else None
                    lgb_day_prices[f"day{day}_high"]  = row.iloc[0]["high"]   if not row.empty else None
                print(
                    f"{'lightgbm':<12} {'n/a':<8} "
                    f"${lgb_day_prices.get('day5_low', 0):<9.2f} "
                    f"${lgb_day_prices.get('day5_price', 0):<9.2f} "
                    f"${lgb_day_prices.get('day5_high', 0):<9.2f} "
                    f"${lgb_day_prices.get('day30_low', 0):<10.2f} "
                    f"${lgb_day_prices.get('day30_price', 0):<10.2f} "
                    f"${lgb_day_prices.get('day30_high', 0):<10.2f} "
                    f"{lgb_result['bias']:<10}"
                )
                method_rows.append({
                    "method":        "lightgbm",
                    "score":         None,
                    "day1_low":      _lgb(1, "low"),   "day1_price":   lgb_day_prices.get("day1_price"),  "day1_high":  _lgb(1, "high"),
                    "day5_low":      _lgb(5, "low"),   "day5_price":   lgb_day_prices.get("day5_price"),  "day5_high":  _lgb(5, "high"),
                    "day10_low":     _lgb(10, "low"),  "day10_price":  lgb_day_prices.get("day10_price"), "day10_high": _lgb(10, "high"),
                    "day20_low":     _lgb(20, "low"),  "day20_price":  lgb_day_prices.get("day20_price"), "day20_high": _lgb(20, "high"),
                    "day30_low":     _lgb(30, "low"),  "day30_price":  lgb_day_prices.get("day30_price"), "day30_high": _lgb(30, "high"),
                    "bias":          lgb_result["bias"],
                    "current_close": current_close,
                })

            # Add Random Forest row to comparison table
            if rf_result is not None:
                rf_cone_full = rf_result["forecast_cone"]
                def _rf(day, col):
                    r = rf_cone_full[rf_cone_full["day"] == day]
                    return r.iloc[0][col] if not r.empty else None
                rf_day_prices = {}
                for day in [1, 5, 10, 20, 30]:
                    row = rf_cone_full[rf_cone_full["day"] == day]
                    rf_day_prices[f"day{day}_low"]   = row.iloc[0]["low"]    if not row.empty else None
                    rf_day_prices[f"day{day}_price"]  = row.iloc[0]["median"] if not row.empty else None
                    rf_day_prices[f"day{day}_high"]  = row.iloc[0]["high"]   if not row.empty else None
                print(
                    f"{'randomforest':<12} {'n/a':<8} "
                    f"${rf_day_prices.get('day5_low', 0):<9.2f} "
                    f"${rf_day_prices.get('day5_price', 0):<9.2f} "
                    f"${rf_day_prices.get('day5_high', 0):<9.2f} "
                    f"${rf_day_prices.get('day30_low', 0):<10.2f} "
                    f"${rf_day_prices.get('day30_price', 0):<10.2f} "
                    f"${rf_day_prices.get('day30_high', 0):<10.2f} "
                    f"{rf_result['bias']:<10}"
                )
                method_rows.append({
                    "method":        "randomforest",
                    "score":         None,
                    "day1_low":      _rf(1, "low"),   "day1_price":   rf_day_prices.get("day1_price"),  "day1_high":  _rf(1, "high"),
                    "day5_low":      _rf(5, "low"),   "day5_price":   rf_day_prices.get("day5_price"),  "day5_high":  _rf(5, "high"),
                    "day10_low":     _rf(10, "low"),  "day10_price":  rf_day_prices.get("day10_price"), "day10_high": _rf(10, "high"),
                    "day20_low":     _rf(20, "low"),  "day20_price":  rf_day_prices.get("day20_price"), "day20_high": _rf(20, "high"),
                    "day30_low":     _rf(30, "low"),  "day30_price":  rf_day_prices.get("day30_price"), "day30_high": _rf(30, "high"),
                    "bias":          rf_result["bias"],
                    "current_close": current_close,
                })

            print()

            # 3. Save comparison to CSV
            ticker_dir = f"output/{t}"
            os.makedirs(ticker_dir, exist_ok=True)
            comparison_df = pd.DataFrame(method_rows)
            comparison_path = f"{ticker_dir}/method_comparison_{t}.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"> Comparison saved to {comparison_path}")

            # 4. Use the best scoring method for main forecast and visualization
            best_method = max(results_by_method.items(), key=lambda x: x[1]["best_score"])
            method_name = best_method[0]
            best_result = best_method[1]

            print()
            print("=" * 70)
            print(f"BEST METHOD: {method_name.upper()} (Score: {best_result['best_score']:.3f})")
            print("=" * 70)
            print()

            config = build_config(method_name, t)

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                analog_result = run_analog_forecast(df, config)
                knn_result = run_knn_forecast(df, config)

            # 5. Save forecast results
            analog_df = analog_result["forecast_df"].rename(columns={"price": "analog_price"})
            def _prob_cols(fc):
                return [c for c in fc.columns if c.startswith("p") and c[1:].isdigit()]
            def _pdcp_cols(fc):
                return [c for c in fc.columns if c.startswith("pdcp") and c[4:].isdigit()]

            combined_df = analog_df.copy()

            for method in SIMILARITY_METHODS:
                if method not in results_by_method:
                    continue
                m_fc      = results_by_method[method]["knn"]["forecast_cone"]
                m_p       = _prob_cols(m_fc)
                m_pdcp    = _pdcp_cols(m_fc)
                cone      = m_fc[["day", "low", "median", "high"] + m_p + m_pdcp].copy()
                cone      = cone.rename(columns={
                    "low":    f"{method}_low",
                    "median": f"{method}_median",
                    "high":   f"{method}_high",
                    **{c: f"{method}_{c}" for c in m_p},
                    **{c: f"{method}_{c}" for c in m_pdcp},
                })
                combined_df = pd.merge(combined_df, cone, on="day", how="left")

            # Add XGBoost columns to saved forecast
            if xgb_result is not None:
                xgb_fc    = xgb_result["forecast_cone"]
                xgb_p     = _prob_cols(xgb_fc)
                xgb_pdcp  = _pdcp_cols(xgb_fc)
                xgb_cone  = xgb_fc[["day", "low", "median", "high"] + xgb_p + xgb_pdcp].rename(
                    columns={"low": "xgb_low", "median": "xgb_median", "high": "xgb_high",
                             **{c: f"xgb_{c}" for c in xgb_p},
                             **{c: f"xgb_{c}" for c in xgb_pdcp}})
                combined_df = pd.merge(combined_df, xgb_cone, on="day", how="left")

            # Add current_close, knn_method (winning similarity method), and direction columns
            cc = float(current_close)
            combined_df["current_close"] = round(cc, 2)
            combined_df["knn_method"]    = method_name  # which similarity method won this run

            direction_pairs = [("analog_price", "analog_direction")]
            for med_col, dir_col in direction_pairs:
                if med_col in combined_df.columns:
                    combined_df[dir_col] = combined_df[med_col].apply(
                        lambda x: "UP" if pd.notna(x) and float(x) > cc else "DOWN"
                    )
            for method in SIMILARITY_METHODS:
                med_col = f"{method}_median"
                if med_col in combined_df.columns:
                    combined_df[f"{method}_direction"] = combined_df[med_col].apply(
                        lambda x: "UP" if pd.notna(x) and float(x) > cc else "DOWN"
                    )
            if xgb_result is not None and "xgb_median" in combined_df.columns:
                combined_df["xgb_direction"] = combined_df["xgb_median"].apply(
                    lambda x: "UP" if pd.notna(x) and float(x) > cc else "DOWN"
                )
            if lgb_result is not None:
                lgb_fc    = lgb_result["forecast_cone"]
                lgb_p     = _prob_cols(lgb_fc)
                lgb_pdcp  = _pdcp_cols(lgb_fc)
                lgb_cone  = lgb_fc[["day", "low", "median", "high"] + lgb_p + lgb_pdcp].rename(
                    columns={"low": "lgb_low", "median": "lgb_median", "high": "lgb_high",
                             **{c: f"lgb_{c}" for c in lgb_p},
                             **{c: f"lgb_{c}" for c in lgb_pdcp}})
                combined_df = pd.merge(combined_df, lgb_cone, on="day", how="left")
                combined_df["lgb_direction"] = combined_df["lgb_median"].apply(
                    lambda x: "UP" if pd.notna(x) and float(x) > cc else "DOWN"
                )
            if rf_result is not None:
                rf_fc    = rf_result["forecast_cone"]
                rf_p     = _prob_cols(rf_fc)
                rf_pdcp  = _pdcp_cols(rf_fc)
                rf_cone  = rf_fc[["day", "low", "median", "high"] + rf_p + rf_pdcp].rename(
                    columns={"low": "rf_low", "median": "rf_median", "high": "rf_high",
                             **{c: f"rf_{c}" for c in rf_p},
                             **{c: f"rf_{c}" for c in rf_pdcp}})
                combined_df = pd.merge(combined_df, rf_cone, on="day", how="left")
                combined_df["rf_direction"] = combined_df["rf_median"].apply(
                    lambda x: "UP" if pd.notna(x) and float(x) > cc else "DOWN"
                )

            # Reorder: place each _direction column right after its model's _high or _price column
            cols = list(combined_df.columns)
            dir_cols = [c for c in cols if c.endswith("_direction")]
            for dc in dir_cols:
                cols.remove(dc)
            for dc in dir_cols:
                prefix = dc[:-len("_direction")]
                for anchor_suffix in ("_high", "_price", "_median"):
                    anchor = f"{prefix}{anchor_suffix}"
                    if anchor in cols:
                        cols.insert(cols.index(anchor) + 1, dc)
                        break
                else:
                    cols.append(dc)
            combined_df = combined_df[cols]

            csv_path = save_forecast_results(combined_df, t)
            print(f"Results saved to {csv_path}")

            # 6. Summary block
            cone = knn_result["forecast_cone"]
            score_label = (
                "HIGH"   if analog_result["best_score"] >= 0.85 else
                "MEDIUM" if analog_result["best_score"] >= 0.70 else
                "LOW"
            )

            def _row(day: int):
                rows = cone[cone["day"] == day]
                return rows.iloc[0] if not rows.empty else None

            print()
            print("=" * 44)
            print(f"FORECAST SUMMARY — {t}")
            print("=" * 44)
            print(f"Method:            {method_name.upper()}")
            print(f"Current Close:     ${df['Close'].iloc[-1]:.2f}")
            print(f"Best Match Score:  {analog_result['best_score']:.3f}  ({score_label})")
            print()
            print(f"{'Day':<6}  {'Date':<12}  {'kNN Median':>11}  {'XGB Median':>11}  {'LGB Median':>11}  {'RF Median':>11}")
            print(f"{'-'*6}  {'-'*12}  {'-'*11}  {'-'*11}  {'-'*11}  {'-'*11}")
            for day in [5, 10, 20, 30]:
                knn_row = _row(day)
                if knn_row is None:
                    continue
                xgb_med = "n/a"
                if xgb_result is not None:
                    xr = xgb_result["forecast_cone"]
                    xr = xr[xr["day"] == day]
                    xgb_med = f"${xr.iloc[0]['median']:.2f}" if not xr.empty else "n/a"
                lgb_med = "n/a"
                if lgb_result is not None:
                    lr = lgb_result["forecast_cone"]
                    lr = lr[lr["day"] == day]
                    lgb_med = f"${lr.iloc[0]['median']:.2f}" if not lr.empty else "n/a"
                rf_med = "n/a"
                if rf_result is not None:
                    rr = rf_result["forecast_cone"]
                    rr = rr[rr["day"] == day]
                    rf_med = f"${rr.iloc[0]['median']:.2f}" if not rr.empty else "n/a"
                print(f"  Day {day:<2} ({str(knn_row['date'].date())}):  ${knn_row['median']:.2f}  {xgb_med:>11}  {lgb_med:>11}  {rf_med:>11}")
            print()
            xgb_bias = xgb_result["bias"] if xgb_result else "n/a"
            lgb_bias = lgb_result["bias"] if lgb_result else "n/a"
            rf_bias  = rf_result["bias"]  if rf_result  else "n/a"
            print(f"Directional Bias:  kNN={knn_result['bias']}  XGB={xgb_bias}  LGB={lgb_bias}  RF={rf_bias}")
            print("=" * 44)

            # Populate summary for run_all
            day30_row = cone[cone["day"] == 30]
            summary.update({
                "success":       True,
                "method":        method_name,
                "score":         analog_result["best_score"],
                "bias":          knn_result["bias"],
                "current_close": current_close,
                "day30_price":   day30_row.iloc[0]["median"] if not day30_row.empty else None,
            })

    except Exception as e:
        summary["error"] = str(e)

    return summary


def run_all() -> None:
    """Run forecasts for every ticker found in output/ subdirectories."""
    tickers = discover_tickers()
    if not tickers:
        print("No ticker directories found in output/")
        return

    print()
    print("=" * 70)
    print(f"RUN ALL — {len(tickers)} tickers: {', '.join(tickers)}")
    print("=" * 70)
    print()

    results = []
    for ticker in tickers:
        print(f"  {ticker:<8}", end="", flush=True)
        s = run_forecast(ticker, quiet=True)
        results.append(s)
        if s["success"]:
            day30 = f"${s['day30_price']:.2f}" if s["day30_price"] is not None else "N/A"
            # Also run knn2 for this ticker
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    knn2_result = run_knn2_only(ticker)
                knn2_str = (f"  knn2:{knn2_result['direction_vote']*100:.0f}%"
                            if knn2_result else "  knn2:err")
            except Exception:
                knn2_str = "  knn2:err"
            print(
                f"✓  {s['bias']:<8}  Score: {s['score']:.3f}"
                f"  Close: ${s['current_close']:.2f}  Day30: {day30}{knn2_str}"
            )
        else:
            err = (s["error"] or "unknown error")[:55]
            print(f"✗  Error: {err}")

    succeeded = sum(1 for r in results if r["success"])
    print()
    print("=" * 70)
    print(f"RUN ALL COMPLETE  ({succeeded}/{len(tickers)} succeeded)")
    print("=" * 70)
    print()


def _prob_cols(fc):
    return [c for c in fc.columns if c.startswith("p") and c[1:].isdigit()]


def _pdcp_cols(fc):
    return [c for c in fc.columns if c.startswith("pdcp") and c[4:].isdigit()]


def _ml_cone_to_unified_cols(result, prefix: str, current_close: float) -> pd.DataFrame:
    """Convert an XGB/LGB/RF result's forecast cone into unified-CSV column structure."""
    fc      = result["forecast_cone"]
    p_cols  = _prob_cols(fc)
    pdcp    = _pdcp_cols(fc)
    keep    = ["day", "date", "low", "median", "high"] + p_cols + pdcp
    cone    = fc[[c for c in keep if c in fc.columns]].copy()
    rename  = {"low": f"{prefix}_low", "median": f"{prefix}_median", "high": f"{prefix}_high"}
    rename.update({c: f"{prefix}_{c}" for c in p_cols + pdcp})
    cone    = cone.rename(columns=rename)
    cone[f"{prefix}_direction"] = cone[f"{prefix}_median"].apply(
        lambda x: "UP" if pd.notna(x) and float(x) > current_close else "DOWN"
    )
    cone["current_close"] = round(float(current_close), 2)
    return cone


def _knn_results_to_unified_cols(results_by_method: dict, analog_result: dict,
                                  best_method: str, current_close: float) -> pd.DataFrame:
    """Convert kNN results across all methods + analog into unified-CSV column structure."""
    analog_df = analog_result["forecast_df"].rename(columns={"price": "analog_price"}).copy()
    combined  = analog_df.copy()

    for method, res in results_by_method.items():
        m_fc   = res["knn"]["forecast_cone"]
        p_cols = _prob_cols(m_fc)
        pdcp   = _pdcp_cols(m_fc)
        keep   = ["day", "low", "median", "high"] + p_cols + pdcp
        cone   = m_fc[[c for c in keep if c in m_fc.columns]].copy()
        rename = {"low": f"{method}_low", "median": f"{method}_median", "high": f"{method}_high"}
        rename.update({c: f"{method}_{c}" for c in p_cols + pdcp})
        cone   = cone.rename(columns=rename)
        combined = combined.merge(cone, on="day", how="left")

    cc = float(current_close)
    combined["current_close"] = round(cc, 2)
    combined["knn_method"]    = best_method

    # Direction columns
    if "analog_price" in combined.columns:
        combined["analog_direction"] = combined["analog_price"].apply(
            lambda x: "UP" if pd.notna(x) and float(x) > cc else "DOWN"
        )
    for method in results_by_method:
        med_col = f"{method}_median"
        if med_col in combined.columns:
            combined[f"{method}_direction"] = combined[med_col].apply(
                lambda x: "UP" if pd.notna(x) and float(x) > cc else "DOWN"
            )

    return combined


def run_knn_only(ticker: str = None) -> None:
    t = ticker or cfg.TICKER
    config = build_config(cfg.SIMILARITY_METHOD, t)
    df = fetch_data(config["TICKER"], config["PERIOD"], config["INTERVAL"],
                    config["CACHE_PATH"], target_bars=5000)
    print()

    SIMILARITY_METHODS_LOCAL = ["spearman", "pearson", "cosine", "euclidean", "kendall", "manhattan"]
    rows = []
    best_score = -1
    best_knn_result = None
    best_method_name = cfg.SIMILARITY_METHOD
    best_analog_result = None
    results_by_method = {}

    for method in SIMILARITY_METHODS_LOCAL:
        print(f"Testing kNN: {method.upper():<10} ", end="", flush=True)
        cfg_m = build_config(method, t)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                analog_result = run_analog_forecast(df, cfg_m)
                knn_result    = run_knn_forecast(df, cfg_m)
            score = analog_result["best_score"]
            print(f"✓ Score: {score:.3f}")
            results_by_method[method] = {"analog": analog_result, "knn": knn_result, "best_score": score}
            if score > best_score:
                best_score         = score
                best_knn_result    = knn_result
                best_method_name   = method
                best_analog_result = analog_result
            cone = knn_result["forecast_cone"]
            for day in [1, 5, 10, 20, 30]:
                row = cone[cone["day"] == day]
                rows.append({
                    "method": method,
                    "score":  score,
                    "day":    day,
                    "date":   row.iloc[0]["date"] if not row.empty else None,
                    "low":    row.iloc[0]["low"]    if not row.empty else None,
                    "median": row.iloc[0]["median"] if not row.empty else None,
                    "high":   row.iloc[0]["high"]   if not row.empty else None,
                    "bias":   knn_result["bias"],
                })
        except Exception as e:
            print(f"✗ {str(e)[:50]}")

    ticker_dir = f"output/{t}"
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = f"{ticker_dir}/forecast_knn_{t}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print()
    print(f"> kNN standalone results saved to {csv_path}")

    # Merge into unified forecast_results CSV (preserves ML + knn2 columns from history)
    if results_by_method and best_analog_result is not None:
        try:
            current_close = float(df["Close"].iloc[-1])
            unified = _knn_results_to_unified_cols(
                results_by_method, best_analog_result, best_method_name, current_close,
            )
            unified_path = save_forecast_results(
                unified, t, preserve_groups=["xgb", "lgb", "rf", "knn2"],
            )
            print(f"> Merged kNN columns into {unified_path}")
        except Exception as e:
            print(f"⚠ Failed to merge into forecast_results: {e}")

    if best_knn_result is not None:
        cone = best_knn_result["forecast_cone"]
        print(f"> Best method: {best_method_name.upper()}  (score {best_score:.3f})")
        print(f"  Day 5:  ${cone[cone['day']==5].iloc[0]['median']:.2f}")
        print(f"  Day 30: ${cone[cone['day']==30].iloc[0]['median']:.2f}")
        print(f"  Bias:   {best_knn_result['bias']}")


def run_xgboost_only(ticker: str = None) -> None:
    t = ticker or cfg.TICKER
    config = build_config(cfg.SIMILARITY_METHOD, t)
    df = fetch_data(config["TICKER"], config["PERIOD"], config["INTERVAL"],
                    config["CACHE_PATH"], target_bars=5000)
    print()

    result = run_xgboost_forecast(df, config)

    cone = result["forecast_cone"]
    rows = []
    for _, row in cone.iterrows():
        rows.append({
            "model":  "xgboost",
            "day":    int(row["day"]),
            "date":   row["date"],
            "low":    row["low"],
            "median": row["median"],
            "high":   row["high"],
            "bias":   result["bias"],
        })

    ticker_dir = f"output/{t}"
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = f"{ticker_dir}/forecast_xgboost_{t}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print()
    print(f"> XGBoost standalone results saved to {csv_path}")

    # Merge into unified forecast_results CSV (preserves kNN + LGB + RF + knn2 columns)
    try:
        current_close = float(df["Close"].iloc[-1])
        unified = _ml_cone_to_unified_cols(result, "xgb", current_close)
        unified_path = save_forecast_results(
            unified, t, preserve_groups=["knn", "lgb", "rf", "knn2"],
        )
        print(f"> Merged xgb_* columns into {unified_path}")
    except Exception as e:
        print(f"⚠ Failed to merge into forecast_results: {e}")


def run_lgbm_only(ticker: str = None) -> None:
    t = ticker or cfg.TICKER
    config = build_config(cfg.SIMILARITY_METHOD, t)
    config["LGBM_USE_VIX"] = getattr(cfg, "LGBM_USE_VIX", True)
    config["LGBM_USE_SPY"] = getattr(cfg, "LGBM_USE_SPY", True)

    df = fetch_data(config["TICKER"], config["PERIOD"], config["INTERVAL"],
                    config["CACHE_PATH"], target_bars=5000)
    print()

    saved_params = load_best_lgbm_params(t)
    result = run_lgbm_forecast(df, config, params=saved_params or None)

    cone = result["forecast_cone"]
    rows = []
    for _, row in cone.iterrows():
        rows.append({
            "model":  "lightgbm",
            "day":    int(row["day"]),
            "date":   row["date"],
            "low":    row["low"],
            "median": row["median"],
            "high":   row["high"],
            "bias":   result["bias"],
        })

    ticker_dir = f"output/{t}"
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = f"{ticker_dir}/forecast_lgbm_{t}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print()
    print(f"> LightGBM standalone results saved to {csv_path}")

    # Merge into unified forecast_results CSV (preserves kNN + XGB + RF + knn2 columns)
    try:
        current_close = float(df["Close"].iloc[-1])
        unified = _ml_cone_to_unified_cols(result, "lgb", current_close)
        unified_path = save_forecast_results(
            unified, t, preserve_groups=["knn", "xgb", "rf", "knn2"],
        )
        print(f"> Merged lgb_* columns into {unified_path}")
    except Exception as e:
        print(f"⚠ Failed to merge into forecast_results: {e}")


def run_rf_only(ticker: str = None) -> None:
    t = ticker or cfg.TICKER
    config = build_config(cfg.SIMILARITY_METHOD, t)

    df = fetch_data(config["TICKER"], config["PERIOD"], config["INTERVAL"],
                    config["CACHE_PATH"], target_bars=5000)
    print()

    saved_params = load_best_rf_params(t)
    result = run_rf_forecast(df, config, params=saved_params or None)

    cone = result["forecast_cone"]
    rows = []
    for _, row in cone.iterrows():
        rows.append({
            "model":  "randomforest",
            "day":    int(row["day"]),
            "date":   row["date"],
            "low":    row["low"],
            "median": row["median"],
            "high":   row["high"],
            "bias":   result["bias"],
        })

    ticker_dir = f"output/{t}"
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = f"{ticker_dir}/forecast_rf_{t}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print()
    print(f"> Random Forest standalone results saved to {csv_path}")

    # Merge into unified forecast_results CSV (preserves kNN + XGB + LGB + knn2 columns)
    try:
        current_close = float(df["Close"].iloc[-1])
        unified = _ml_cone_to_unified_cols(result, "rf", current_close)
        unified_path = save_forecast_results(
            unified, t, preserve_groups=["knn", "xgb", "lgb", "knn2"],
        )
        print(f"> Merged rf_* columns into {unified_path}")
    except Exception as e:
        print(f"⚠ Failed to merge into forecast_results: {e}")


def run_summary() -> None:
    """
    Read existing forecast CSVs and print a one-line bullish/bearish summary
    per ticker — no re-running, no data fetch.
    """
    tickers = discover_tickers()
    if not tickers:
        print("No ticker directories found in output/")
        return

    today = pd.Timestamp.now().normalize()

    print()
    print("=" * 85)
    print(f"FORECAST SUMMARY  —  {len(tickers)} tickers  |  reading saved CSVs  |  no re-run")
    print("=" * 85)
    print(f"{'Ticker':<10} {'Close':>8}  {'Updated':<12}  {'D5 kNN':<8} {'D5 XGB':<8} {'D5 LGB':<8} {'D5 RF':<8}  {'D30 Bias':<10}  {'Notes'}")
    print(f"{'-'*10} {'-'*8}  {'-'*12}  {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*10}  {'-'*20}")

    for ticker in tickers:
        csv_path = f"output/{ticker}/forecast_results_{ticker}.csv"

        if not os.path.exists(csv_path):
            print(f"{ticker:<10}  {'—':>8}  {'—':<12}  {'n/a':<8} {'n/a':<8} {'n/a':<8}  {'NO FORECAST':<10}")
            continue

        try:
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])

            future = df[df["date"] > today].copy()
            if future.empty:
                print(f"{ticker:<10}  {'—':>8}  {'—':<12}  {'n/a':<8} {'n/a':<8} {'n/a':<8}  {'EXPIRED':<10}  re-run main.py")
                continue

            close_val   = df["current_close"].iloc[-1]  if "current_close"  in df.columns else None
            last_update = str(df["last_updated"].iloc[-1])[:10] if "last_updated" in df.columns else "unknown"

            def _dir(day: int, col: str) -> str:
                row = future[future["day"] == day]
                if row.empty or col not in row.columns:
                    return "n/a"
                val = row.iloc[0][col]
                return str(val) if pd.notna(val) else "n/a"

            knn_method_col = str(df["knn_method"].iloc[-1]) if "knn_method" in df.columns and pd.notna(df["knn_method"].iloc[-1]) else None
            d5_knn = _dir(5, f"{knn_method_col}_direction") if knn_method_col else "n/a"
            d5_xgb = _dir(5,  "xgb_direction")
            d5_lgb = _dir(5,  "lgb_direction")
            d5_rf  = _dir(5,  "rf_direction")

            # Overall Day-30 bias: majority vote across all _direction columns
            dir_cols   = [c for c in future.columns if c.endswith("_direction")]
            day30_row  = future[future["day"] == 30]
            if day30_row.empty:
                day30_row = future.iloc[[-1]]

            if dir_cols and not day30_row.empty:
                r   = day30_row.iloc[0]
                ups = sum(1 for c in dir_cols if r.get(c) == "UP")
                dns = sum(1 for c in dir_cols if r.get(c) == "DOWN")
                if ups > dns:
                    bias = "BULLISH"
                elif dns > ups:
                    bias = "BEARISH"
                else:
                    bias = "MIXED"
                vote_note = f"{ups}↑ {dns}↓ / {len(dir_cols)}"
            else:
                bias      = "n/a"
                vote_note = ""

            close_str = f"${close_val:.2f}" if close_val is not None else "n/a"
            print(
                f"{ticker:<10} {close_str:>8}  {last_update:<12}  "
                f"{d5_knn:<8} {d5_xgb:<8} {d5_lgb:<8} {d5_rf:<8}  "
                f"{bias:<10}  {vote_note}"
            )

        except Exception as e:
            print(f"{ticker:<10}  error: {str(e)[:50]}")

    print()
    print("D5 = Day 5 direction vs close at forecast time.  D30 Bias = majority vote across all models at Day 30.")
    print("To refresh forecasts: venv/bin/python main.py run_all")
    print()


def run_knn2_only(ticker: str = None) -> dict:
    """
    Run the enhanced knn2 forecast for one ticker and save knn2_* columns
    into forecast_results_{TICKER}.csv alongside existing columns.
    """
    t = ticker or cfg.TICKER
    config = build_config(cfg.SIMILARITY_METHOD, t)

    print()
    df_raw = fetch_data(config["TICKER"], config["PERIOD"], config["INTERVAL"],
                        config["CACHE_PATH"], target_bars=5000)
    df = compute_features(df_raw)
    print()

    result = run_knn2_forecast(df, config)

    # --- Merge knn2 columns into forecast_results CSV ---
    fc = result["forecast_cone"].copy()

    def _prob_cols(fc_):
        return [c for c in fc_.columns if c.startswith("p") and c[1:].isdigit()]
    def _pdcp_cols(fc_):
        return [c for c in fc_.columns if c.startswith("pdcp") and c[4:].isdigit()]

    p_cols   = _prob_cols(fc)
    pdcp_cols = _pdcp_cols(fc)
    meta_cols = ["direction", "conviction", "regime", "n_neighbors"]

    col_map = {"low": "knn2_low", "median": "knn2_median", "high": "knn2_high"}
    for c in p_cols:
        col_map[c] = f"knn2_{c}"
    for c in pdcp_cols:
        col_map[c] = f"knn2_{c}"
    col_map["direction"]   = "knn2_direction"
    col_map["conviction"]  = "knn2_vote_pct"
    col_map["regime"]      = "knn2_regime"
    col_map["n_neighbors"] = "knn2_n_neighbors"

    keep_cols = ["day", "date"] + list(col_map.keys())
    knn2_df = fc[[c for c in keep_cols if c in fc.columns]].rename(columns=col_map)
    knn2_df["date"] = pd.to_datetime(knn2_df["date"])
    num_cols = knn2_df.select_dtypes(include="number").columns.difference(["day"])
    knn2_df[num_cols] = knn2_df[num_cols].round(2)

    cc = float(df["Close"].iloc[-1])
    knn2_df["current_close"] = round(cc, 2)

    # Column-merge: add/update knn2_* columns without touching any other columns
    ticker_dir = f"output/{t}"
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = f"{ticker_dir}/forecast_results_{t}.csv"

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing["date"] = pd.to_datetime(existing["date"])

        new_knn2 = knn2_df.drop(columns=["current_close", "day"], errors="ignore")
        knn2_cols = [c for c in new_knn2.columns if c != "date"]

        # Merge new knn2 values into existing — preserve old knn2 values for
        # dates not covered by the new forecast (never wipe historical knn2 data)
        merged = existing.merge(new_knn2, on="date", how="left", suffixes=("", "_new"))
        for col in knn2_cols:
            new_col = f"{col}_new"
            if new_col in merged.columns:
                # Use new value where available, keep existing where not
                merged[col] = merged[new_col].combine_first(merged.get(col, pd.Series(dtype=float)))
                merged.drop(columns=[new_col], inplace=True)
            elif col not in merged.columns:
                merged[col] = pd.NA

        merged.to_csv(csv_path, index=False)
        print(f"> knn2 columns merged into existing {csv_path}")
    else:
        knn2_df.to_csv(csv_path, index=False)
        print(f"> knn2 results saved to new {csv_path}")
    print(f"> Regime: {result['regime']}  Vote: {result['conviction']:.1f}%  "
          f"Direction: {'UP' if result['direction_vote'] > 0.5 else 'DOWN'}  "
          f"Bias: {result['bias']}")
    return result


def run_knn2_all() -> None:
    """Run knn2 for every ticker in output/."""
    tickers = discover_tickers()
    if not tickers:
        print("No ticker directories found in output/")
        return

    print()
    print("=" * 70)
    print(f"KNN2 RUN ALL — {len(tickers)} tickers")
    print("=" * 70)
    print()

    succeeded = 0
    for ticker in tickers:
        print(f"  {ticker:<8}", end="", flush=True)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = run_knn2_only(ticker)
            vote = result["direction_vote"]
            direction = "UP" if vote > 0.5 else "DOWN"
            conviction = result["conviction"]
            print(f"✓  {result['bias']:<8}  Dir: {direction}  "
                  f"Conviction: {conviction:.2f}  Regime: {result['regime']}")
            succeeded += 1
        except Exception as e:
            print(f"✗  Error: {str(e)[:55]}")

    print()
    print("=" * 70)
    print(f"KNN2 ALL COMPLETE  ({succeeded}/{len(tickers)} succeeded)")
    print("=" * 70)
    print()


def _run_for_all(label: str, runner) -> None:
    """Run a single-ticker function across every discovered ticker, with a heading + summary."""
    tickers = discover_tickers()
    if not tickers:
        print("No ticker directories found in output/")
        return

    print()
    print("=" * 70)
    print(f"{label} RUN ALL — {len(tickers)} tickers")
    print("=" * 70)
    print()

    succeeded = 0
    for ticker in tickers:
        print(f"  {ticker:<8}", end="", flush=True)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                runner(ticker)
            print(f"✓  done")
            succeeded += 1
        except Exception as e:
            print(f"✗  Error: {str(e)[:55]}")

    print()
    print("=" * 70)
    print(f"{label} ALL COMPLETE  ({succeeded}/{len(tickers)} succeeded)")
    print("=" * 70)
    print()


def run_knn_all() -> None:
    _run_for_all("KNN", run_knn_only)


def run_xgboost_all() -> None:
    _run_for_all("XGBOOST", run_xgboost_only)


def run_lgbm_all() -> None:
    _run_for_all("LIGHTGBM", run_lgbm_only)


def run_rf_all() -> None:
    _run_for_all("RANDOMFOREST", run_rf_only)


def main() -> None:
    run_forecast(cfg.TICKER)


if __name__ == "__main__":
    arg  = sys.argv[1] if len(sys.argv) > 1 else ""
    arg2 = sys.argv[2] if len(sys.argv) > 2 else ""

    if arg == "run_all":
        # `run_all` (no subcmd) → full pipeline. `run_all <model>` → that model only, all tickers.
        sub = arg2.lower()
        if not sub:
            run_all()
        elif sub == "knn":
            run_knn_all()
        elif sub == "knn2":
            run_knn2_all()
        elif sub in ("xgb", "xgboost"):
            run_xgboost_all()
        elif sub in ("lgb", "lgbm", "lightgbm"):
            run_lgbm_all()
        elif sub == "rf":
            run_rf_all()
        else:
            print(f"Unknown run_all subcommand: '{arg2}'")
            print("Valid: run_all | run_all knn | run_all knn2 | run_all xgb | run_all lgbm | run_all rf")
            sys.exit(1)
    elif arg == "knn":
        run_knn_only()
    elif arg == "knn2":
        run_knn2_only()
    elif arg == "knn2_all":
        run_knn2_all()
    elif arg == "xgboost":
        run_xgboost_only()
    elif arg == "lgbm":
        run_lgbm_only()
    elif arg == "rf":
        run_rf_only()
    elif arg == "summary":
        run_summary()
    else:
        main()
