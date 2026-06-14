"""
Experiment: which BACKTEST_MONTHS tuning-window length produces knn2 params
that generalize best to a held-out window?

For each candidate window length N in --windows:
  1. Tune knn2 params on the N months ending right before the holdout
     (param_sweep.py knn2 logic, run in-process with cfg patched)
  2. Save the resulting feature_weights_knn2.yaml under output/{T}/params/_tune_runs/
  3. Backtest knn2 on the holdout with those params
  4. Record holdout metrics

Then print a leaderboard sorted by holdout Day 5 dir accuracy.
The original feature_weights_knn2.yaml is restored at the end.

Usage:
  venv/bin/python tune_window_experiment.py --ticker HUT \
      --windows 3,6,9,12 --holdout-months 2 --step 5 --forecast-days 30
"""
import os
import sys
import shutil
import argparse
import io
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta

import pandas as pd

import config as cfg
from data.fetcher import fetch_data
from features.engineer import compute_features
from backtester.backtest_knn2 import run_knn2_backtest
from features.feature_weights import load_knn2_params


def _params_path(ticker: str) -> str:
    return f"output/{ticker}/params/feature_weights_knn2.yaml"


def _save_dir(ticker: str) -> str:
    p = f"output/{ticker}/params/_tune_runs"
    os.makedirs(p, exist_ok=True)
    return p


def _patch_cfg(months: int, start: str | None, end: str | None,
               step: int, forecast_days: int, ticker: str):
    """Patch cfg attrs in place; return a snapshot to restore later."""
    snap = {
        "TICKER":                 cfg.TICKER,
        "CACHE_PATH":             cfg.CACHE_PATH,
        "BACKTEST_MONTHS":        cfg.BACKTEST_MONTHS,
        "BACKTEST_STEP":          cfg.BACKTEST_STEP,
        "BACKTEST_FORECAST_DAYS": cfg.BACKTEST_FORECAST_DAYS,
        "BACKTEST_START":         getattr(cfg, "BACKTEST_START", None),
        "BACKTEST_END":           getattr(cfg, "BACKTEST_END", None),
    }
    cfg.TICKER                 = ticker
    cfg.CACHE_PATH             = f"data/{ticker}_max.csv"
    cfg.BACKTEST_MONTHS        = months
    cfg.BACKTEST_STEP          = step
    cfg.BACKTEST_FORECAST_DAYS = forecast_days
    cfg.BACKTEST_START         = start
    cfg.BACKTEST_END           = end
    return snap


def _restore_cfg(snap: dict):
    for k, v in snap.items():
        setattr(cfg, k, v)


def run_sweep_quiet():
    """Import + invoke knn2_sweep with stdout suppressed."""
    from param_sweep import knn2_sweep
    buf_o, buf_e = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_o), redirect_stderr(buf_e):
        knn2_sweep()
    return buf_o.getvalue()


def backtest_holdout(ticker: str, holdout_start: str, holdout_end: str,
                     step: int, forecast_days: int) -> dict:
    """Run knn2 backtest on holdout dates with whatever params are saved."""
    df_raw = fetch_data(ticker, cfg.PERIOD, cfg.INTERVAL, f"data/{ticker}_max.csv")
    df     = compute_features(df_raw)

    config = {
        "TICKER":                 ticker,
        "WINDOW_LEN":             cfg.WINDOW_LEN,
        "FORECAST_LEN":           cfg.FORECAST_LEN,
        "BARS_BACK":              cfg.BARS_BACK,
        "K":                      cfg.K,
        "MIN_GAP":                cfg.MIN_GAP,
        "CONFIDENCE_BANDS":       cfg.CONFIDENCE_BANDS,
        "BACKTEST_START":         holdout_start,
        "BACKTEST_END":           holdout_end,
        "BACKTEST_STEP":          step,
        "BACKTEST_FORECAST_DAYS": forecast_days,
        "FEATURE_COLS":           getattr(cfg, "FEATURE_COLS", None),
        "ML_TRAINING_LOOKBACK_BARS": getattr(cfg, "ML_TRAINING_LOOKBACK_BARS", 1500),
    }

    params = load_knn2_params(ticker) or {}
    buf_o, buf_e = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_o), redirect_stderr(buf_e):
        result = run_knn2_backtest(df, config, params=params)
    return result or {}


def date_str(d: pd.Timestamp) -> str:
    return d.strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--windows", default="3,6,9,12",
                    help="comma-separated tuning window lengths in months")
    ap.add_argument("--holdout-months", type=int, default=2,
                    help="how many months at the end to reserve as holdout")
    ap.add_argument("--step", type=int, default=5,
                    help="backtest step (trading days). Smaller = more samples.")
    ap.add_argument("--forecast-days", type=int, default=30,
                    help="forecast horizon during tuning + holdout")
    args = ap.parse_args()

    ticker  = args.ticker.upper()
    windows = [int(x) for x in args.windows.split(",") if x.strip()]

    # Establish data range so we can compute fixed holdout dates.
    # Fetch once to find the latest date in cache.
    df = fetch_data(ticker, cfg.PERIOD, cfg.INTERVAL, f"data/{ticker}_max.csv")
    last_date  = df.index.max().normalize()
    holdout_end   = last_date
    holdout_start = (holdout_end - pd.DateOffset(months=args.holdout_months)).normalize()
    tune_end      = holdout_start - pd.Timedelta(days=1)

    print()
    print("=" * 80)
    print(f"TUNE-WINDOW EXPERIMENT — {ticker}")
    print("=" * 80)
    print(f"  Data range:    {date_str(df.index.min().normalize())} → {date_str(last_date)}")
    print(f"  Holdout:       {date_str(holdout_start)} → {date_str(holdout_end)}  (locked, never tuned on)")
    print(f"  Tuning ends:   {date_str(tune_end)}")
    print(f"  Windows:       {windows} months")
    print(f"  Step:          {args.step} trading days")
    print(f"  Forecast days: {args.forecast_days}")
    print("=" * 80)

    params_path = _params_path(ticker)
    backup_path = params_path + ".pre_experiment.bak"
    if os.path.exists(params_path):
        shutil.copy(params_path, backup_path)
        print(f"  Backed up existing params → {backup_path}")
    print()

    save_dir = _save_dir(ticker)
    rows = []

    for months in windows:
        tune_start = (tune_end - pd.DateOffset(months=months) + pd.Timedelta(days=1)).normalize()
        print(f"--- Window: {months} months  ({date_str(tune_start)} → {date_str(tune_end)}) ---")

        snap = _patch_cfg(
            months=months,
            start=date_str(tune_start),
            end=date_str(tune_end),
            step=args.step,
            forecast_days=args.forecast_days,
            ticker=ticker,
        )
        try:
            print(f"  [1/2] Tuning knn2 ...", flush=True)
            run_sweep_quiet()
            tuned_params = load_knn2_params(ticker) or {}
            saved = os.path.join(save_dir, f"feature_weights_knn2_{months}mo.yaml")
            shutil.copy(params_path, saved)
            print(f"        params: half_life={tuned_params.get('half_life_days')}, "
                  f"dist_pct={tuned_params.get('distance_threshold_pct')}  "
                  f"(saved → {saved})")

            print(f"  [2/2] Backtesting on holdout ...", flush=True)
            holdout_result = backtest_holdout(
                ticker, date_str(holdout_start), date_str(holdout_end),
                args.step, args.forecast_days,
            )
        finally:
            _restore_cfg(snap)

        if not holdout_result:
            print("        no holdout result\n")
            continue

        row = {
            "tune_months":   months,
            "tune_start":    date_str(tune_start),
            "tune_end":      date_str(tune_end),
            "half_life":     tuned_params.get("half_life_days"),
            "dist_pct":      tuned_params.get("distance_threshold_pct"),
            "sweep_dir":     tuned_params.get("sweep_dir_acc"),
            "holdout_dir":   round(holdout_result.get("directional_accuracy", float("nan")), 1),
            "holdout_mape":  round(holdout_result.get("mape", float("nan")), 2),
            "holdout_mae":   round(holdout_result.get("mae", float("nan")), 2),
            "holdout_cone":  round(holdout_result.get("cone_hit_rate", float("nan")), 1),
            "holdout_n":     holdout_result.get("total_forecasts", 0),
        }
        # Generalization gap: how much did dir accuracy drop tune → holdout?
        if row["sweep_dir"] is not None and not pd.isna(row["holdout_dir"]):
            row["gen_gap"] = round(float(row["sweep_dir"]) - float(row["holdout_dir"]), 1)
        else:
            row["gen_gap"] = None
        rows.append(row)
        print(f"        Holdout: Dir={row['holdout_dir']}%  MAPE={row['holdout_mape']}%  "
              f"Cone={row['holdout_cone']}%  N={row['holdout_n']}  "
              f"GenGap={row['gen_gap']}\n")

    # Restore original params
    if os.path.exists(backup_path):
        shutil.copy(backup_path, params_path)
        print(f"Restored original params from {backup_path}")
    print()

    if not rows:
        print("No results.")
        return

    df_out = pd.DataFrame(rows).sort_values("holdout_dir", ascending=False).reset_index(drop=True)
    df_out.insert(0, "rank", df_out.index + 1)

    print("=" * 95)
    print("LEADERBOARD  (ranked by holdout Day-5 directional accuracy, higher = better)")
    print("=" * 95)
    print(f"  {'rank':<5} {'months':<7} {'half_life':<10} {'dist_pct':<9} "
          f"{'sweep_dir':>10} {'holdout_dir':>12} {'gen_gap':>9} "
          f"{'holdout_mape':>13} {'holdout_cone':>13} {'N':>4}")
    for _, r in df_out.iterrows():
        gap = "n/a" if r["gen_gap"] is None else f"{r['gen_gap']:+.1f}%"
        sd  = "n/a" if r["sweep_dir"] is None else f"{r['sweep_dir']:.1f}%"
        print(f"  {int(r['rank']):<5} {int(r['tune_months']):<7} "
              f"{str(r['half_life']):<10} {str(r['dist_pct']):<9} "
              f"{sd:>10} {r['holdout_dir']:>11.1f}% "
              f"{gap:>9} {r['holdout_mape']:>12.1f}% "
              f"{r['holdout_cone']:>12.1f}% {int(r['holdout_n']):>4}")

    out_csv = f"output/{ticker}/params/_tune_runs/window_experiment_{ticker}.csv"
    df_out.to_csv(out_csv, index=False)
    print()
    print(f"> Saved → {out_csv}")
    print()
    print("Interpretation:")
    print("  - Higher holdout_dir = generalizes better.")
    print("  - Small gen_gap (close to 0) = robust. Big positive gap = overfit to tuning window.")
    print("  - Pick the window with the best balance of high holdout_dir AND small gen_gap.")
    print()


if __name__ == "__main__":
    main()
