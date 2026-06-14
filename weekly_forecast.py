"""
Generate a forecast for the remaining trading days of the current week
(today through Friday inclusive) and save to a dated CSV.
"""

import io
import os
import sys
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr, nullcontext

import config as cfg
from data.fetcher         import fetch_data
from forecaster.analog    import run_analog_forecast
from forecaster.knn       import run_knn_forecast
from models.xgboost_model  import run_xgboost_forecast, load_best_xgb_params
from models.lightgbm_model import run_lgbm_forecast, load_best_lgbm_params
from models.rf_model       import run_rf_forecast,   load_best_rf_params

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


def build_config(method: str, ticker: str = None) -> dict:
    t = ticker or cfg.TICKER
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
        "K":                 cfg.K,
        "MIN_GAP":           cfg.MIN_GAP,
        "CONFIDENCE_BANDS":  cfg.CONFIDENCE_BANDS,
        "SHOW_ALL_PATHS":    cfg.SHOW_ALL_PATHS,
        "SHOW_ZIGZAG":       cfg.SHOW_ZIGZAG,
        "ZIGZAG_LEGS":       cfg.ZIGZAG_LEGS,
        "FEATURE_COLS":      getattr(cfg, "FEATURE_COLS", None),
    }


def week_bounds(next_week: bool = False) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, friday) for the current or next trading week."""
    today = pd.Timestamp.now().normalize()
    if next_week:
        days_to_monday = (7 - today.weekday()) % 7 or 7
        start  = today + pd.Timedelta(days=days_to_monday)
        friday = start + pd.Timedelta(days=4)
    else:
        days_until_friday = 4 - today.weekday()
        if days_until_friday < 0:
            days_until_friday = 0
        start  = today
        friday = today + pd.Timedelta(days=days_until_friday)
    return start, friday


def run_weekly_forecast(ticker: str, next_week: bool = False, quiet: bool = False) -> dict:
    """
    Run weekly forecast for one ticker.
    Returns a summary dict with success, best_method, score, current_close, csv_path.
    """
    summary = {"ticker": ticker, "success": False, "error": None}

    _buf = io.StringIO()
    _out = redirect_stdout(_buf) if quiet else nullcontext()
    _err = redirect_stderr(_buf) if quiet else nullcontext()

    try:
        with _out, _err:
            today, friday = week_bounds(next_week)
            monday     = today - pd.Timedelta(days=today.weekday())
            week_label = monday.strftime("%Y-%m-%d")

            print(f"> Weekly forecast [{ticker}]: {today.date()} → {friday.date()}  (week of {week_label})")
            print()

            base_config = build_config(cfg.SIMILARITY_METHOD, ticker)
            df = fetch_data(
                base_config["TICKER"],
                base_config["PERIOD"],
                base_config["INTERVAL"],
                base_config["CACHE_PATH"],
                target_bars=5000,
            )
            print()

            current_close = df["Close"].iloc[-1]
            results_by_method = {}

            print("=" * 70)
            print(f"RUNNING FORECASTS FOR ALL SIMILARITY METHODS — {ticker}")
            print("=" * 70)

            for method in SIMILARITY_METHODS:
                print(f"  {method.upper():<12} ", end="", flush=True)
                config = build_config(method, ticker)
                try:
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        analog_result = run_analog_forecast(df, config)
                        knn_result    = run_knn_forecast(df, config)
                    results_by_method[method] = {
                        "analog": analog_result,
                        "knn":    knn_result,
                        "score":  analog_result["best_score"],
                    }
                    print(f"✓  score={analog_result['best_score']:.3f}")
                except Exception as e:
                    print(f"✗  {str(e)[:50]}")

            if not results_by_method:
                raise RuntimeError("All similarity methods failed — cannot generate forecast.")

            best_method = max(results_by_method, key=lambda m: results_by_method[m]["score"])
            best        = results_by_method[best_method]
            print()
            print(f"> Best method: {best_method.upper()}  (score={best['score']:.3f})")

            cone = best["knn"]["forecast_cone"].copy()
            cone["date"] = pd.to_datetime(cone["date"]).dt.normalize()

            weekly = cone[(cone["date"] >= today) & (cone["date"] <= friday)].copy()

            df_dates = df.index.normalize()
            if today not in df_dates:
                pass
            elif today not in cone["date"].values:
                today_close = float(df["Close"].iloc[-1])
                today_row = pd.DataFrame([{
                    "day":    0,
                    "date":   today,
                    "low":    today_close,
                    "median": today_close,
                    "high":   today_close,
                }])
                weekly = pd.concat([today_row, weekly], ignore_index=True)

            if weekly.empty:
                print(f"WARNING: no forecast dates fall within {today.date()} – {friday.date()}.")

            def _prob_cols(fc):
                return [c for c in fc.columns if c.startswith("p") and c[1:].isdigit()]
            def _pdcp_cols(fc):
                return [c for c in fc.columns if c.startswith("pdcp") and c[4:].isdigit()]

            # All 6 methods — full bands (low/median/high + p-bands + pdcp)
            for method in SIMILARITY_METHODS:
                if method not in results_by_method:
                    continue
                m_fc   = results_by_method[method]["knn"]["forecast_cone"]
                m_p    = _prob_cols(m_fc)
                m_pdcp = _pdcp_cols(m_fc)
                m_cone = m_fc[["day", "low", "median", "high"] + m_p + m_pdcp].copy()
                m_cone = m_cone.rename(columns={
                    "low":    f"{method}_low",
                    "median": f"{method}_median",
                    "high":   f"{method}_high",
                    **{c: f"{method}_{c}" for c in m_p},
                    **{c: f"{method}_{c}" for c in m_pdcp},
                })
                weekly = pd.merge(weekly, m_cone, on="day", how="left")

            # XGBoost — full cone
            print(f"  {'XGBOOST':<12} ", end="", flush=True)
            try:
                xgb_config = build_config(cfg.SIMILARITY_METHOD, ticker)
                saved_params = load_best_xgb_params(ticker)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    xgb_result = run_xgboost_forecast(df, xgb_config, params=saved_params or None)
                xgb_fc   = xgb_result["forecast_cone"]
                xgb_p    = _prob_cols(xgb_fc)
                xgb_pdcp = _pdcp_cols(xgb_fc)
                xgb_cone = xgb_fc[["day", "low", "median", "high"] + xgb_p + xgb_pdcp].rename(columns={
                    "low":    "xgb_low",
                    "median": "xgb_median",
                    "high":   "xgb_high",
                    **{c: f"xgb_{c}" for c in xgb_p},
                    **{c: f"xgb_{c}" for c in xgb_pdcp},
                })
                weekly = pd.merge(weekly, xgb_cone, on="day", how="left")
                print(f"✓  samples={xgb_result['n_train_samples']}")
            except Exception as e:
                print(f"✗  {str(e)[:50]}")

            # LightGBM — full cone
            print(f"  {'LIGHTGBM':<12} ", end="", flush=True)
            try:
                lgb_config   = build_config(cfg.SIMILARITY_METHOD, ticker)
                lgb_config["LGBM_USE_VIX"] = getattr(cfg, "LGBM_USE_VIX", True)
                lgb_config["LGBM_USE_SPY"] = getattr(cfg, "LGBM_USE_SPY", True)
                lgb_saved    = load_best_lgbm_params(ticker)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    lgb_result = run_lgbm_forecast(df, lgb_config, params=lgb_saved or None)
                lgb_fc   = lgb_result["forecast_cone"]
                lgb_p    = _prob_cols(lgb_fc)
                lgb_pdcp = _pdcp_cols(lgb_fc)
                lgb_cone = lgb_fc[["day", "low", "median", "high"] + lgb_p + lgb_pdcp].rename(columns={
                    "low":    "lgb_low",
                    "median": "lgb_median",
                    "high":   "lgb_high",
                    **{c: f"lgb_{c}" for c in lgb_p},
                    **{c: f"lgb_{c}" for c in lgb_pdcp},
                })
                weekly = pd.merge(weekly, lgb_cone, on="day", how="left")
                print(f"✓  samples={lgb_result['n_train_samples']}")
            except Exception as e:
                print(f"✗  {str(e)[:50]}")

            # Random Forest — full cone
            print(f"  {'RANDOMFOREST':<12} ", end="", flush=True)
            try:
                rf_config  = build_config(cfg.SIMILARITY_METHOD, ticker)
                rf_config["RF_USE_SPY"] = getattr(cfg, "RF_USE_SPY", True)
                rf_saved   = load_best_rf_params(ticker)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    rf_result = run_rf_forecast(df, rf_config, params=rf_saved or None)
                rf_fc   = rf_result["forecast_cone"]
                rf_p    = _prob_cols(rf_fc)
                rf_pdcp = _pdcp_cols(rf_fc)
                rf_cone = rf_fc[["day", "low", "median", "high"] + rf_p + rf_pdcp].rename(columns={
                    "low":    "rf_low",
                    "median": "rf_median",
                    "high":   "rf_high",
                    **{c: f"rf_{c}" for c in rf_p},
                    **{c: f"rf_{c}" for c in rf_pdcp},
                })
                weekly = pd.merge(weekly, rf_cone, on="day", how="left")
                print(f"✓  samples={rf_result['n_train_samples']}")
            except Exception as e:
                print(f"✗  {str(e)[:50]}")

            # Direction columns: UP if median > current_close, else DOWN
            cc = float(current_close)
            for med_col, dir_col in [("median", "direction")] + \
                    [(f"{m}_median", f"{m}_direction") for m in SIMILARITY_METHODS] + \
                    [("xgb_median", "xgb_direction"), ("lgb_median", "lgb_direction"),
                     ("rf_median",  "rf_direction")]:
                if med_col in weekly.columns:
                    weekly[dir_col] = weekly[med_col].apply(
                        lambda x: "UP" if pd.notna(x) and float(x) > cc else "DOWN"
                    )

            # Place each direction column right after its model's high column
            cols = list(weekly.columns)
            dir_cols = [c for c in cols if c == "direction" or c.endswith("_direction")]
            for dc in dir_cols:
                cols.remove(dc)
            for dc in dir_cols:
                pfx    = dc[:-len("_direction")] if dc != "direction" else ""
                anchor = f"{pfx}_high" if pfx else "high"
                if anchor in cols:
                    cols.insert(cols.index(anchor) + 1, dc)
                else:
                    cols.append(dc)
            weekly = weekly[cols]

            weekly.insert(0, "ticker",        ticker)
            weekly.insert(1, "current_close", round(cc, 2))
            weekly.insert(2, "best_method",   best_method)

            ticker_dir = f"output/{ticker}"
            os.makedirs(ticker_dir, exist_ok=True)
            csv_path = f"{ticker_dir}/weekof_{week_label}_{ticker}_forecast.csv"
            weekly.to_csv(csv_path, index=False)

            print()
            print("=" * 70)
            print(f"WEEKLY FORECAST — {ticker}  (week of {week_label})")
            print("=" * 70)
            if not weekly.empty:
                print(f"{'Date':<14} {'Day':>4}  {'Low':>8}  {'Median':>8}  {'High':>8}")
                print("-" * 50)
                for _, row in weekly.iterrows():
                    print(
                        f"{str(row['date'].date()):<14} {int(row['day']):>4}  "
                        f"${row['low']:>7.2f}  ${row['median']:>7.2f}  ${row['high']:>7.2f}"
                    )
            print()
            print(f"> Saved to {csv_path}")

            summary.update({
                "success":       True,
                "best_method":   best_method,
                "score":         best["score"],
                "current_close": float(current_close),
                "csv_path":      csv_path,
                "week_label":    week_label,
            })

    except Exception as e:
        summary["error"] = str(e)

    return summary


def run_all(next_week: bool = False) -> None:
    """Run weekly forecast for every ticker found in output/ subdirectories."""
    tickers = discover_tickers()
    if not tickers:
        print("No ticker directories found in output/")
        return

    today, friday = week_bounds(next_week)
    monday     = today - pd.Timedelta(days=today.weekday())
    week_label = monday.strftime("%Y-%m-%d")

    print()
    print("=" * 70)
    print(f"WEEKLY FORECAST — RUN ALL  ({len(tickers)} tickers, week of {week_label})")
    print("=" * 70)
    print()

    results = []
    for ticker in tickers:
        print(f"  {ticker:<8}", end="", flush=True)
        s = run_weekly_forecast(ticker, next_week=next_week, quiet=True)
        results.append(s)
        if s["success"]:
            print(
                f"✓  {s['best_method']:<12}  Score: {s['score']:.3f}"
                f"  Close: ${s['current_close']:.2f}"
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


def main() -> None:
    next_week = "--next" in sys.argv
    run_weekly_forecast(cfg.TICKER, next_week=next_week)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "run_all":
        next_week = "--next" in sys.argv
        run_all(next_week=next_week)
    else:
        main()
