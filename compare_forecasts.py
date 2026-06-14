import os
import io
import sys
import numpy as np
import pandas as pd
import config as cfg
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from data.fetcher import fetch_data
from validation.compare import compare_forecasts as perform_comparison
from validation.html_report import generate_validation_report


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


_SIMILARITY_METHODS = ["spearman", "pearson", "cosine", "euclidean", "kendall", "manhattan"]

# (display_name, csv_prefix, has_bands)
_ALGO_DEFS = [
    ("analog",      "analog",      False),
    ("spearman",    "spearman",    True),
    ("pearson",     "pearson",     True),
    ("cosine",      "cosine",      True),
    ("euclidean",   "euclidean",   True),
    ("kendall",     "kendall",     True),
    ("manhattan",   "manhattan",   True),
    ("xgboost",     "xgb",         True),
    ("lightgbm",    "lgb",         True),
    ("randomforest","rf",          True),
    ("knn2",        "knn2",        True),
]


def _model_cell(prefix: str, row: pd.Series, actual: float, all_columns: list) -> str:
    """Build a multi-line cell for one model (knn, spearman, xgb, etc.)."""
    med_col  = f"{prefix}_median"
    low_col  = f"{prefix}_low"
    high_col = f"{prefix}_high"

    if med_col not in all_columns or pd.isna(row.get(med_col)):
        return "n/a"

    med  = float(row[med_col])
    low  = float(row[low_col])  if low_col  in all_columns and not pd.isna(row.get(low_col))  else None
    high = float(row[high_col]) if high_col in all_columns and not pd.isna(row.get(high_col)) else None
    err  = float(row[f"{prefix}_error"])     if f"{prefix}_error"     in all_columns else abs(med - actual)
    pct  = float(row[f"{prefix}_pct_error"]) if f"{prefix}_pct_error" in all_columns else err / actual * 100

    lines = [
        f"low: ${low:.4f} - {'HIT' if actual >= low else 'MISS'}" if low  is not None else "low: N/A",
        f"median: ${med:.4f} - {'HIT' if actual >= med else 'MISS'}",
        f"high: ${high:.4f} - {'HIT' if actual >= high else 'MISS'}" if high is not None else "high: N/A",
    ]

    p_cols = sorted(
        [c for c in all_columns if c.startswith(f"{prefix}_p") and c[len(prefix)+2:].isdigit()],
        key=lambda c: -int(c[len(prefix)+2:]),
    )
    for col in p_cols:
        val = row.get(col)
        if val is not None and not pd.isna(val):
            prob = col[len(prefix)+2:]
            lines.append(f"p{prob}: ${float(val):.4f} - {'HIT' if actual >= float(val) else 'MISS'}")

    pdcp_cols = sorted(
        [c for c in all_columns if c.startswith(f"{prefix}_pdcp") and c[len(prefix)+5:].isdigit()],
        key=lambda c: -int(c[len(prefix)+5:]),
    )
    for col in pdcp_cols:
        val = row.get(col)
        prob = col[len(prefix)+5:]
        if val is not None and not pd.isna(val):
            lines.append(f"pdcp{prob}: ${float(val):.4f} - {'HIT' if actual >= float(val) else 'MISS'}")

    # Option A: ABOVE BAND / IN BAND / BELOW BAND
    if low is not None and high is not None:
        if actual > high:
            range_label = "ABOVE BAND"
        elif actual < low:
            range_label = "BELOW BAND"
        else:
            range_label = "IN BAND"
    else:
        range_label = "N/A"
    lines.append(f"range_result: {range_label}")

    lines.append(f"error: ${err:.2f} ({pct:.1f}%)")

    # Direction: one line per model at the bottom
    dir_col     = f"{prefix}_direction"
    dir_ok_col  = f"{prefix}_direction_correct"
    direction   = row.get(dir_col)
    dir_correct = row.get(dir_ok_col)
    if direction and not pd.isna(direction):
        if dir_correct is not None and not pd.isna(dir_correct):
            lines.append(f"direction: {direction} - {'CORRECT' if bool(dir_correct) else 'MISS'}")
        else:
            lines.append(f"direction: {direction}")

    return "\n".join(lines)


def _actual_cell(row: pd.Series) -> str:
    move_dir = "up" if float(row["actual_move_pct"]) >= 0 else "down"
    dir_label = "CORRECT" if bool(row["direction_correct"]) else "WRONG"
    return "\n".join([
        f"C: ${float(row['actual_price']):.2f}",
        f"O: ${float(row['actual_open']):.2f}",
        f"H: ${float(row['actual_high']):.2f}",
        f"L: ${float(row['actual_low']):.2f}",
        f"moved {move_dir} {abs(float(row['actual_move_pct'])):.2f}%",
        f"Direction: {dir_label}",
    ])


def _analog_cell(row: pd.Series, actual: float) -> str:
    val   = float(row["analog_price"])
    err   = float(row["analog_error"])
    pct   = float(row["analog_pct_error"])

    lines = [f"price: ${val:.4f} - {'HIT' if actual >= val else 'MISS'}"]

    # PDCP: analog price vs current_close at forecast time
    current_close = row.get("current_close")
    if current_close is not None and not pd.isna(current_close):
        if val > float(current_close):
            lines.append(f"pdcp: ${val:.4f} - {'HIT' if actual >= val else 'MISS'}")
        else:
            lines.append("pdcp: null")

    lines.append(f"error: ${err:.2f} ({pct:.1f}%)")

    # Direction at the bottom — consistent with other model cells
    direction   = row.get("analog_direction")
    dir_correct = row.get("analog_direction_correct")
    if direction and not pd.isna(direction):
        if dir_correct is not None and not pd.isna(dir_correct):
            lines.append(f"direction: {direction} - {'CORRECT' if bool(dir_correct) else 'MISS'}")
        else:
            lines.append(f"direction: {direction}")

    return "\n".join(lines)


def _leaderboard_cell(row: pd.Series, actual: float, all_columns: list) -> str:
    """Rank all models by absolute error for this single date (lowest error = rank 1)."""
    candidates = [
        ("analog",      "analog_price"),
        ("spearman",    "spearman_median"),
        ("pearson",     "pearson_median"),
        ("cosine",      "cosine_median"),
        ("euclidean",   "euclidean_median"),
        ("kendall",     "kendall_median"),
        ("manhattan",   "manhattan_median"),
        ("xgboost",     "xgb_median"),
        ("lightgbm",    "lgb_median"),
        ("randomforest","rf_median"),
        ("knn2",        "knn2_median"),
    ]
    ranked = []
    for name, col in candidates:
        if col in all_columns and not pd.isna(row.get(col)):
            ranked.append((name, abs(float(row[col]) - actual)))
    ranked.sort(key=lambda x: x[1])
    return "\n".join(f"{i+1}. {name} (${err:.2f})" for i, (name, err) in enumerate(ranked))


def build_output_rows(comparison_df: pd.DataFrame) -> pd.DataFrame:
    all_columns = list(comparison_df.columns)
    total_days  = len(comparison_df)
    rows = []

    for _, row in comparison_df.iterrows():
        actual = float(row["actual_price"])
        out = {
            "date":        pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
            "day":         f"Day {int(row['day'])} of {total_days}",
            "leaderboard": _leaderboard_cell(row, actual, all_columns),
            "prior_close": f"${float(row['prior_close']):.2f}",
            "actual":      _actual_cell(row),
            "analog":      _analog_cell(row, actual),
        }
        for method in _SIMILARITY_METHODS:
            out[method] = _model_cell(method, row, actual, all_columns)
        out["xgboost"]      = _model_cell("xgb",  row, actual, all_columns) if "xgb_median"  in all_columns else "n/a"
        out["lightgbm"]     = _model_cell("lgb",  row, actual, all_columns) if "lgb_median"  in all_columns else "n/a"
        out["randomforest"] = _model_cell("rf",   row, actual, all_columns) if "rf_median"   in all_columns else "n/a"
        out["knn2"]         = _model_cell("knn2", row, actual, all_columns) if "knn2_median" in all_columns else "n/a"
        rows.append(out)

    cols = ["date", "day", "leaderboard", "prior_close", "actual", "analog"] + _SIMILARITY_METHODS + ["xgboost", "lightgbm", "randomforest", "knn2"]
    return pd.DataFrame(rows, columns=cols)


def save_with_upsert(new_rows: pd.DataFrame, csv_path: str) -> None:
    new_dates = new_rows["date"].astype(str).str[:10].tolist()

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path, dtype=str)
        existing = existing[~existing["date"].astype(str).str[:10].isin(new_dates)]
        combined = pd.concat([existing, new_rows.astype(str)], ignore_index=True)
        print(f"> Updating existing file: {csv_path}")
    else:
        combined = new_rows.astype(str)
        print(f"> Creating new file: {csv_path}")

    combined.to_csv(csv_path, index=False)


def run_comparison(ticker: str = None, quiet: bool = False) -> dict:
    """
    Run forecast validation for one ticker.
    Returns a summary dict with success, n_days, dir_accuracy, in_band_rate, avg_error.
    """
    t = ticker or cfg.TICKER
    summary = {"ticker": t, "success": False, "error": None, "n_days": 0}

    _buf = io.StringIO()
    _out = redirect_stdout(_buf) if quiet else nullcontext()
    _err = redirect_stderr(_buf) if quiet else nullcontext()

    try:
        with _out, _err:
            config = {
                "TICKER":     t,
                "PERIOD":     cfg.PERIOD,
                "INTERVAL":   cfg.INTERVAL,
                "CACHE_PATH": f"data/{t}_max.csv",
            }

            print()
            print("=" * 70)
            print("FORECAST VALIDATION: Comparing Predictions to Actual Prices")
            print("=" * 70)
            print()

            df = fetch_data(config["TICKER"], config["PERIOD"], config["INTERVAL"], config["CACHE_PATH"])
            print()

            ticker_dir    = f"output/{t}"
            forecast_file = f"{ticker_dir}/forecast_results_{t}.csv"
            comparison_df = perform_comparison(df, forecast_file)

            if len(comparison_df) == 0:
                print("No forecast dates have passed yet. Check back later!")
                summary["success"] = True
                summary["n_days"] = 0
                return summary

            W = 160
            print("=" * W)
            print("VALIDATION RESULTS: Forecasts vs Actual Prices")
            print("=" * W)
            print()

            header = (
                f"{'Day':<5} {'Date':<12} {'Prior $':<10} {'Forecast':<11} "
                f"{'Band Low':<11} {'Band High':<11} "
                f"{'Open':<9} {'High':<9} {'Low':<9} {'Close':<9} "
                f"{'Err $':<9} {'Err %':<8} {'Dir':<8} {'Band@Close':<11} {'Touched':<8}"
            )
            print(header)
            print("-" * W)

            has_xgb = "xgb_median" in comparison_df.columns

            for _, row in comparison_df.iterrows():
                day       = int(row["day"])
                date      = str(row["date"])[:10]
                prior     = f"${row['prior_close']:.2f}"
                forecast  = f"${row['analog_price']:.2f}"
                knn_low   = f"${row['knn_low']:.2f}"
                knn_high  = f"${row['knn_high']:.2f}"
                act_open  = f"${row['actual_open']:.2f}"
                act_high  = f"${row['actual_high']:.2f}"
                act_low   = f"${row['actual_low']:.2f}"
                actual    = f"${row['actual_price']:.2f}"
                err_d     = f"${row['knn_error']:.2f}"
                err_pct   = f"{row['knn_pct_error']:.1f}%"

                in_band   = row["knn_low"] <= row["actual_price"] <= row["knn_high"]
                touched   = bool(row["band_touched"])
                dir_ok    = bool(row["direction_correct"])

                dir_label     = "UP  " if row["actual_move_pct"] >= 0 else "DOWN"
                dir_tick      = "OK" if dir_ok else "X"
                in_band_label = "YES" if in_band else "NO"
                touched_label = "YES" if touched else "NO"

                print(
                    f"{day:<5} {date:<12} {prior:<10} {forecast:<11} "
                    f"{knn_low:<11} {knn_high:<11} "
                    f"{act_open:<9} {act_high:<9} {act_low:<9} {actual:<9} "
                    f"{err_d:<9} {err_pct:<8} {dir_label+' '+dir_tick:<8} {in_band_label:<11} {touched_label:<8}"
                )

                move_desc  = f"moved {'up' if row['actual_move_pct'] >= 0 else 'down'} {abs(row['actual_move_pct']):.2f}%"
                pred_desc  = f"predicted {'up' if row['predicted_move_pct'] >= 0 else 'down'} {abs(row['predicted_move_pct']):.2f}%"
                band_note  = "closed inside the band" if in_band else "closed outside the band"
                touch_note = "but price did touch the band intraday" if (not in_band and touched) else ""
                xgb_note   = ""
                if has_xgb and not pd.isna(row.get("xgb_median")):
                    xgb_note = f"  XGBoost off by ${row['xgb_error']:.2f} ({row['xgb_pct_error']:.1f}%)."
                print(
                    f"      -> Stock {move_desc} from prior close ${row['prior_close']:.2f} "
                    f"(forecast {pred_desc}). k-NN off by ${row['knn_error']:.2f} "
                    f"({row['knn_pct_error']:.1f}%). {band_note.capitalize()}."
                    + (f" {touch_note.capitalize()}." if touch_note else "")
                    + xgb_note
                )
                print()

            # Compute summary stats from the DataFrame directly
            n               = len(comparison_df)
            knn_in_band     = (comparison_df["knn_low"] <= comparison_df["actual_price"]) & \
                              (comparison_df["actual_price"] <= comparison_df["knn_high"])
            in_band_count          = int(knn_in_band.sum())
            touched_count          = int(comparison_df["band_touched"].sum())
            direction_correct_count = int(comparison_df["direction_correct"].sum())

            if has_xgb:
                xgb_valid             = comparison_df.dropna(subset=["xgb_median"])
                xgb_n                 = len(xgb_valid)
                xgb_dir_correct_count = int(xgb_valid["xgb_direction_correct"].sum())
                xgb_in_band_count     = int(xgb_valid["xgb_in_band"].sum())
                xgb_touched_count     = int(xgb_valid["xgb_band_touched"].sum())
            else:
                xgb_n = 0

            print("=" * W)
            print("SUMMARY STATISTICS")
            print("=" * W)
            print()
            print(f"Total Days Validated:          {n}")
            print()
            print(f"{'Metric':<30} {'k-NN':>10}  {'XGBoost':>10}")
            print(f"{'-'*30} {'-'*10}  {'-'*10}")
            print(f"{'Direction Correct':<30} {direction_correct_count}/{n} ({direction_correct_count/n*100:.1f}%)  "
                  + (f"{xgb_dir_correct_count}/{xgb_n} ({xgb_dir_correct_count/xgb_n*100:.1f}%)" if xgb_n else "n/a"))
            print(f"{'Closed Inside Band':<30} {in_band_count}/{n} ({in_band_count/n*100:.1f}%)  "
                  + (f"{xgb_in_band_count}/{xgb_n} ({xgb_in_band_count/xgb_n*100:.1f}%)" if xgb_n else "n/a"))
            print(f"{'Band Touched Intraday':<30} {touched_count}/{n} ({touched_count/n*100:.1f}%)  "
                  + (f"{xgb_touched_count}/{xgb_n} ({xgb_touched_count/xgb_n*100:.1f}%)" if xgb_n else "n/a"))
            print(f"{'Average Error ($)':<30} ${comparison_df['knn_error'].mean():.2f}        "
                  + (f"${xgb_valid['xgb_error'].mean():.2f}" if xgb_n else "n/a"))
            print(f"{'Average Error (%)':<30} {comparison_df['knn_pct_error'].mean():.1f}%          "
                  + (f"{xgb_valid['xgb_pct_error'].mean():.1f}%" if xgb_n else "n/a"))
            print()
            print(f"Average Error (Analog):        ${comparison_df['analog_error'].mean():.2f}")
            print()
            best_day  = int(comparison_df.loc[comparison_df["knn_error"].idxmin(), "day"])
            worst_day = int(comparison_df.loc[comparison_df["knn_error"].idxmax(), "day"])
            print(f"Best Day  (k-NN, lowest error):  Day {best_day} (${comparison_df['knn_error'].min():.2f})")
            print(f"Worst Day (k-NN, highest error): Day {worst_day} (${comparison_df['knn_error'].max():.2f})")
            print()

            # PDCP accuracy — upside targets only (non-null = price was above prev close at forecast time)
            pdcp_probs = sorted(
                [int(col[8:]) for col in comparison_df.columns if col.startswith("knn_pdcp") and col[8:].isdigit()],
                reverse=True,
            )
            if pdcp_probs:
                print("PDCP — Upside Target Accuracy (actual close >= forecasted target):")
                print(f"  {'Band':<10} {'k-NN':>18}  {'Spearman':>18}  {'XGBoost':>18}  {'LightGBM':>18}  {'RF':>18}")
                print(f"  {'-'*10} {'-'*18}  {'-'*18}  {'-'*18}  {'-'*18}  {'-'*18}")
                for prob in pdcp_probs:
                    row_stats = []
                    for prefix in ("knn", "spearman", "xgb", "lgb", "rf"):
                        col = f"{prefix}_pdcp{prob}"
                        if col not in comparison_df.columns:
                            row_stats.append("n/a")
                            continue
                        valid = comparison_df.dropna(subset=[col])
                        if len(valid) == 0:
                            row_stats.append("no upside targets")
                            continue
                        hits = int((valid["actual_price"] >= valid[col]).sum())
                        row_stats.append(f"{hits}/{len(valid)} ({hits/len(valid)*100:.1f}%) hit")
                    print(f"  pdcp{prob:<6} {row_stats[0]:>18}  {row_stats[1]:>18}  {row_stats[2]:>18}  {row_stats[3]:>18}  {row_stats[4]:>18}")
                print()

            # Leaderboard — rank all models by avg error %
            leaderboard_models = [
                ("analog",      "analog_price",       None,               None),
                ("spearman",    "spearman_median",    "spearman_low",     "spearman_high"),
                ("pearson",     "pearson_median",     "pearson_low",      "pearson_high"),
                ("cosine",      "cosine_median",      "cosine_low",       "cosine_high"),
                ("euclidean",   "euclidean_median",   "euclidean_low",    "euclidean_high"),
                ("kendall",     "kendall_median",     "kendall_low",      "kendall_high"),
                ("manhattan",   "manhattan_median",   "manhattan_low",    "manhattan_high"),
                ("xgboost",     "xgb_median",         "xgb_low",          "xgb_high"),
                ("lightgbm",    "lgb_median",         "lgb_low",          "lgb_high"),
                ("randomforest","rf_median",           "rf_low",           "rf_high"),
                ("knn2",        "knn2_median",         "knn2_low",         "knn2_high"),
            ]
            lb_rows = []
            for name, med_col, low_col, high_col in leaderboard_models:
                if med_col not in comparison_df.columns:
                    continue
                valid = comparison_df.dropna(subset=[med_col])
                if len(valid) == 0:
                    continue
                errors    = (valid[med_col] - valid["actual_price"]).abs()
                avg_err   = errors.mean()
                avg_pct   = (errors / valid["actual_price"] * 100).mean()
                pred_dir  = np.sign(valid[med_col] - valid["prior_close"])
                act_dir   = np.sign(valid["actual_price"] - valid["prior_close"])
                dir_acc   = (pred_dir == act_dir).mean() * 100
                if low_col and high_col and low_col in comparison_df.columns and high_col in comparison_df.columns:
                    in_band_pct = (
                        (valid["actual_price"] >= valid[low_col]) &
                        (valid["actual_price"] <= valid[high_col])
                    ).mean() * 100
                    in_band_str = f"{in_band_pct:.1f}%"
                else:
                    in_band_str = "n/a"
                lb_rows.append((name, dir_acc, avg_err, avg_pct, in_band_str))

            lb_rows.sort(key=lambda x: x[3])  # sort by avg error % ascending

            print("MODEL LEADERBOARD (ranked by avg error %, lower is better):")
            print(f"  {'Rank':<5} {'Model':<12} {'Dir Acc':>9}  {'Avg Err $':>10}  {'Avg Err %':>10}  {'In-Band':>8}")
            print(f"  {'-'*5} {'-'*12} {'-'*9}  {'-'*10}  {'-'*10}  {'-'*8}")
            for rank, (name, dir_acc, avg_err, avg_pct, in_band_str) in enumerate(lb_rows, 1):
                print(f"  {rank:<5} {name:<12} {dir_acc:>8.1f}%  ${avg_err:>9.2f}  {avg_pct:>9.1f}%  {in_band_str:>8}")
            print()

            os.makedirs(ticker_dir, exist_ok=True)
            csv_path    = f"{ticker_dir}/forecast_validation_{t}.csv"
            output_rows = build_output_rows(comparison_df)
            save_with_upsert(output_rows, csv_path)
            print(f"> Results saved to {csv_path}")
            print()

            # Find the BEST model (by conservative score) — used for compare_all summary
            # so users see real model performance, not the per-row knn_method's accuracy.
            best_model = None
            best_score = -1.0
            for name, dir_acc, avg_err, avg_pct, in_band_str in lb_rows:
                # Skip models that don't have direction data or have no in-band info
                in_band_pct = float(in_band_str.replace("%", "")) if in_band_str != "n/a" else 0.0
                # tight = % of forecasts with avg_pct error <= 3%
                med_col = f"{name}_median" if name != "analog" else "analog_price"
                if med_col not in comparison_df.columns:
                    continue
                valid = comparison_df.dropna(subset=[med_col])
                if len(valid) == 0:
                    continue
                errs = (valid[med_col] - valid["actual_price"]).abs() / valid["actual_price"] * 100
                tight_pct = float((errs <= 3).mean() * 100)
                score = (
                    0.40 * dir_acc
                  + 0.30 * tight_pct
                  + 0.20 * in_band_pct
                  + 0.10 * max(0, 100 - avg_pct)
                )
                if score > best_score:
                    best_score = score
                    best_model = {
                        "name":        name,
                        "score":       round(score, 1),
                        "dir":         round(dir_acc, 1),
                        "in_band":     round(in_band_pct, 1),
                        "avg_pct_err": round(avg_pct, 2),
                        "n":           int(len(valid)),
                    }

            summary.update({
                "success":      True,
                "n_days":       n,
                "dir_accuracy": direction_correct_count / n * 100,
                "in_band_rate": in_band_count / n * 100,
                "avg_err":      comparison_df["knn_error"].mean(),
                "avg_pct_err":  comparison_df["knn_pct_error"].mean(),
                # NEW: best-model summary fields used by compare_all
                "best_model":       best_model["name"] if best_model else None,
                "best_score":       best_model["score"] if best_model else None,
                "best_dir":         best_model["dir"] if best_model else None,
                "best_in_band":     best_model["in_band"] if best_model else None,
                "best_avg_pct_err": best_model["avg_pct_err"] if best_model else None,
                "best_n":           best_model["n"] if best_model else None,
            })

    except Exception as e:
        summary["error"] = str(e)

    return summary


def compare_all(verdict_view: bool = False) -> None:
    """Run forecast validation for every ticker found in output/ subdirectories."""
    tickers = discover_tickers()
    if not tickers:
        print("No ticker directories found in output/")
        return

    print()
    print("=" * 70)
    print(f"COMPARE ALL — {len(tickers)} tickers: {', '.join(tickers)}")
    print("=" * 70)
    print()

    results = []
    for ticker in tickers:
        print(f"  Validating {ticker}...", end="\r", flush=True)
        s = run_comparison(ticker, quiet=True)
        results.append(s)

    # Sort: successful with data first (by N desc, then best-model score desc),
    # then no-data tickers, then failed tickers last.
    def _sort_key(s):
        if not s["success"]:
            return (2, 0, 0)
        if s["n_days"] == 0:
            return (1, 0, 0)
        return (0, -s["n_days"], -(s.get("best_score") or 0))

    results.sort(key=_sort_key)

    print(" " * 40, end="\r")  # clear the "Validating..." line

    if verdict_view:
        # Enrich each successful summary with analyzer data (verdict, recent verdict, next-best)
        from validation.analyzer import analyze
        for s in results:
            if not (s["success"] and s["n_days"] > 0):
                continue
            try:
                a = analyze(s["ticker"])
                if "error" in a: continue
                s["verdict"]      = a.get("verdict")
                s["recommended"]  = a.get("recommended")
                s["recent_v"]     = a.get("recent_verdict")
                # next-best model = 2nd in by_score (top3 list)
                top3 = a.get("top3") or []
                s["next_best"]    = top3[1] if len(top3) > 1 else None
            except Exception:
                pass

        print(f"  {'Ticker':<8} {'N':<3}  {'Verdict':<14}  {'Best':<14}  {'Next':<12}  {'Recent':<8}")
        print(f"  {'-'*8} {'-'*3}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*8}")
        for s in results:
            ticker = s["ticker"]
            if not s["success"]:
                err = (s["error"] or "unknown error")[:55]
                print(f"  {ticker:<8} ✗   Error: {err}")
                continue
            if s["n_days"] == 0:
                print(f"  {ticker:<8} -   No elapsed forecast dates yet")
                continue
            v       = s.get("verdict") or "?"
            badge   = {"TRUST": "✅", "WEAK SIGNAL": "⚠️", "NONE": "🛑"}.get(v, "  ")
            v_label = f"{badge} {v}"
            best    = s.get("recommended") or s.get("best_model") or "—"
            nxt     = s.get("next_best") or "—"
            rv      = s.get("recent_v")
            if rv:
                # DECAY when recent verdict is worse than overall, or all-below-50
                decayed = (rv.get("verdict") not in (None, v)) or rv.get("all_below_50")
                recent_label = "⚠ DECAY" if decayed else "STABLE"
            else:
                recent_label = "—"
            print(f"  {ticker:<8} {s['n_days']:<3}  {v_label:<14}  {best:<14}  {nxt:<12}  {recent_label:<8}")
    else:
        print(f"  {'Ticker':<8} {'N':<3}  {'Best model':<20}  {'Dir':<6}  {'In-Band':<8}  {'Err%':<6}  {'Score':<6}")
        print(f"  {'-'*8} {'-'*3}  {'-'*20}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}")
        for s in results:
            ticker = s["ticker"]
            if not s["success"]:
                err = (s["error"] or "unknown error")[:55]
                print(f"  {ticker:<8} ✗   Error: {err}")
            elif s["n_days"] == 0:
                print(f"  {ticker:<8} -   No elapsed forecast dates yet")
            else:
                best   = s.get("best_model") or "—"
                bn     = s.get("best_n") or 0
                best_label = f"{best} (n={bn})"
                bdir   = s.get("best_dir")
                bband  = s.get("best_in_band")
                berr   = s.get("best_avg_pct_err")
                bscore = s.get("best_score")
                dir_s  = f"{bdir:.0f}%"  if bdir  is not None else "—"
                band_s = f"{bband:.0f}%" if bband is not None else "—"
                err_s  = f"{berr:.1f}%"  if berr  is not None else "—"
                score_s= f"{bscore:.1f}" if bscore is not None else "—"
                print(f"  {ticker:<8} {s['n_days']:<3}  {best_label:<20}  {dir_s:<6}  {band_s:<8}  {err_s:<6}  {score_s:<6}")

    # Generate HTML reports for tickers with validation data
    print()
    print("Generating HTML reports...")
    n_reports = 0
    for s in results:
        if s["success"] and s["n_days"] > 0:
            try:
                path = generate_validation_report(s["ticker"])
                if path:
                    n_reports += 1
            except Exception as e:
                print(f"  {s['ticker']:<8}✗  HTML error: {str(e)[:60]}")
    print(f"  → {n_reports} reports written to output/<TICKER>/reports/validate_<TICKER>.html")

    succeeded = sum(1 for r in results if r["success"])
    print()
    print("=" * 70)
    print(f"COMPARE ALL COMPLETE  ({succeeded}/{len(tickers)} succeeded)")
    print("=" * 70)
    print()


def _parse_cell(cell_text: str) -> dict:
    """
    Extract all fields from a model cell string. Skips missing/null values.
    Returns a list of (label, value_string) tuples in original order.
    """
    out = []
    if not cell_text or str(cell_text).strip() in ("n/a", "nan", ""):
        return out

    price_fields = {"low", "median", "high", "price",
                    "p90", "p60", "p50", "p20",
                    "pdcp90", "pdcp60", "pdcp50", "pdcp20", "pdcp"}

    for line in str(cell_text).split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key  = key.strip().lower()
        rest = rest.strip()

        if key in price_fields:
            if rest.lower() in ("null", "n/a", "none", "nan", ""):
                continue  # skip missing pdcp / N/A values
            # rest looks like "$41.1466 - HIT" or "$41.1466 - MISS"
            parts = rest.split(" - ", 1)
            price_s = parts[0].replace("$", "").strip()
            hit_s   = parts[1].strip() if len(parts) > 1 else ""
            label   = "median" if key == "price" else key  # normalise analog "price:" → "median"
            display = f"${float(price_s):.4f}" if price_s else rest
            if hit_s:
                display = f"{display} — {hit_s}"
            try:
                float(price_s)  # only add if parseable
                out.append((label, display))
            except ValueError:
                pass

        elif key == "range_result":
            if rest and rest.lower() not in ("n/a", "nan"):
                out.append(("range_result", rest))

        elif key == "error":
            if rest and rest.lower() not in ("n/a", "nan"):
                out.append(("error", rest))

        elif key == "direction":
            if rest and rest.lower() not in ("n/a", "nan"):
                out.append(("direction", rest))

    return out


def _date_lookup_from_validation(ticker: str, target_dt: pd.Timestamp,
                                  target_date: str, val_path: str) -> None:
    """Show date info from forecast_validation CSV when forecast_results has no entry."""
    try:
        val_df = pd.read_csv(val_path, dtype=str)
        val_df["date"] = pd.to_datetime(val_df["date"])
        row_match = val_df[val_df["date"].dt.normalize() == target_dt]
    except Exception as e:
        print(f"Could not read validation file: {e}")
        return

    if row_match.empty:
        print(f"No data found for {target_date} in forecast_results or forecast_validation.")
        return

    row = row_match.iloc[0]

    actual_str    = str(row.get("actual",      "")).strip()
    prior_str     = str(row.get("prior_close", "")).strip()
    leaderboard   = str(row.get("leaderboard", "")).strip()
    day_str       = str(row.get("day",         "")).strip()

    W = 82
    print()
    print("=" * W)
    print(f"FORECAST LOOKUP — {ticker}  |  Date: {target_date}  |  {day_str}  (from validation history)")
    print("=" * W)
    print(f"  NOTE: Date not in active forecast. Showing from past validation records.")
    print()

    # Actual + prior close (extract price from multi-line actual cell if needed)
    actual_price = None
    prior_price  = None
    if prior_str.startswith("$"):
        try:
            prior_price = float(prior_str.replace("$", ""))
        except ValueError:
            pass
    for line in actual_str.split("\n"):
        if line.strip().startswith("C:"):
            try:
                actual_price = float(line.strip().split("$")[1].split()[0])
            except Exception:
                pass

    if prior_price:
        print(f"  Prior close : ${prior_price:.2f}")
    if actual_price:
        move = ((actual_price - prior_price) / prior_price * 100) if prior_price else 0
        arrow = "▲" if move >= 0 else "▼"
        print(f"  Actual close: ${actual_price:.2f}  {arrow} {abs(move):.1f}%")
    print()

    if leaderboard and leaderboard != "nan":
        print("  LEADERBOARD (closest to actual on this date):")
        for line in leaderboard.split("\n"):
            print(f"    {line}")
        print()

    # Parse each model column
    algo_cols = [
        ("analog",      "analog"),
        ("spearman",    "spearman"),
        ("pearson",     "pearson"),
        ("cosine",      "cosine"),
        ("euclidean",   "euclidean"),
        ("kendall",     "kendall"),
        ("manhattan",   "manhattan"),
        ("xgboost",     "xgboost"),
        ("lightgbm",    "lightgbm"),
        ("knn2",        "knn2"),
    ]

    any_printed = False
    for display, col in algo_cols:
        if col not in val_df.columns:
            continue
        cell   = str(row.get(col, "")).strip()
        fields = _parse_cell(cell)   # list of (label, value_string)
        if not fields:
            continue
        if not any_printed:
            print("  ALGORITHM PREDICTIONS:")
            print()
            any_printed = True
        print(f"  {display.upper()}:")
        for label, value in fields:
            print(f"    {label:<14} {value}")
        print()

    print("=" * W)
    print()


def _match_hit_probability(price: float | None, p_bands: dict[int, float], tol: float = 0.011) -> int | None:
    """
    Match a displayed price level back to its stored percentile band.
    Returns the probability that price closes above that level.
    """
    if price is None:
        return None
    closest_prob = None
    closest_diff = None
    for prob, band_price in p_bands.items():
        diff = abs(float(band_price) - float(price))
        if closest_diff is None or diff < closest_diff:
            closest_prob = prob
            closest_diff = diff
    if closest_diff is not None and closest_diff <= tol:
        return closest_prob
    return None


def run_date_lookup(ticker: str, target_date: str) -> None:
    """
    For a specific date, show all algorithm predictions ranked by historical
    accuracy, plus direction consensus and prediction cluster.
    """
    t = ticker.upper()
    try:
        target_dt = pd.Timestamp(target_date).normalize()
    except Exception:
        print(f"Invalid date: {target_date}. Use YYYY-MM-DD.")
        return

    forecast_path = f"output/{t}/forecast_results_{t}.csv"
    if not os.path.exists(forecast_path):
        print(f"No forecast file found: {forecast_path}")
        print(f"Run: venv/bin/python main.py first.")
        return

    forecast_df = pd.read_csv(forecast_path)
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    target_row = forecast_df[forecast_df["date"].dt.normalize() == target_dt]

    if target_row.empty:
        # Fall back to forecast_validation CSV
        val_path = f"output/{t}/forecast_validation_{t}.csv"
        if os.path.exists(val_path):
            _date_lookup_from_validation(t, target_dt, target_date, val_path)
        else:
            avail = forecast_df["date"].dt.date
            print(f"No forecast for {target_date} in {forecast_path}")
            print(f"Available range: {avail.min()} → {avail.max()}")
        return

    row           = target_row.iloc[0]
    day_num       = int(row["day"]) if "day" in row.index else "?"
    current_close = float(row["current_close"]) if "current_close" in row.index and pd.notna(row.get("current_close")) else None
    last_updated  = str(row.get("last_updated", "unknown"))[:10]

    # --- Historical accuracy (best-effort) ---
    hist_acc  = {}
    n_valid   = 0
    cmp_df    = pd.DataFrame()
    today     = pd.Timestamp.now().normalize()

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            df_hist = fetch_data(t, cfg.PERIOD, cfg.INTERVAL, f"data/{t}_max.csv")
            cmp_df  = perform_comparison(df_hist, forecast_path)

        if len(cmp_df) > 0:
            n_valid = len(cmp_df)
            for name, prefix, has_bands in _ALGO_DEFS:
                med_col = f"{prefix}_median" if has_bands else f"{prefix}_price"
                if med_col not in cmp_df.columns:
                    continue
                valid = cmp_df.dropna(subset=[med_col])
                if len(valid) == 0:
                    continue
                errs    = (valid[med_col] - valid["actual_price"]).abs()
                err_pct = (errs / valid["actual_price"] * 100).mean()
                pred_d  = np.sign(valid[med_col] - valid["prior_close"])
                act_d   = np.sign(valid["actual_price"] - valid["prior_close"])
                dir_acc = (pred_d == act_d).mean() * 100
                hist_acc[name] = {"dir_acc": dir_acc, "err_pct": err_pct, "n": len(valid)}
    except Exception:
        pass

    # --- Extract predictions for target date ---
    preds = []
    for name, prefix, has_bands in _ALGO_DEFS:
        med_col  = f"{prefix}_median" if has_bands else f"{prefix}_price"
        low_col  = f"{prefix}_low"       if has_bands else None
        high_col = f"{prefix}_high"      if has_bands else None
        dir_col  = f"{prefix}_direction" if has_bands else None

        if med_col not in row.index or pd.isna(row.get(med_col)):
            continue

        median_v = float(row[med_col])
        low_v    = float(row[low_col])  if low_col  and low_col  in row.index and pd.notna(row.get(low_col))  else None
        high_v   = float(row[high_col]) if high_col and high_col in row.index and pd.notna(row.get(high_col)) else None
        dir_v    = str(row[dir_col])    if dir_col  and dir_col  in row.index and pd.notna(row.get(dir_col))  else None

        # Infer direction for analog from median vs current_close
        if dir_v is None and current_close is not None:
            dir_v = "UP" if median_v > current_close else "DOWN"

        p_bands = {}
        for prob in [90, 60, 50, 20]:
            pc = f"{prefix}_p{prob}"
            if pc in row.index and pd.notna(row.get(pc)):
                p_bands[prob] = float(row[pc])

        acc = hist_acc.get(name, {})
        preds.append({
            "name":      name,
            "median":    median_v,
            "low":       low_v,
            "high":      high_v,
            "direction": dir_v,
            "p_bands":   p_bands,
            "dir_acc":   acc.get("dir_acc"),
            "err_pct":   acc.get("err_pct"),
            "n_hist":    acc.get("n", 0),
        })

    if not preds:
        print(f"No predictions found for {target_date}.")
        return

    # Sort: ranked by dir_acc desc (algorithms with no history go last)
    preds.sort(key=lambda x: (0 if x["dir_acc"] is not None else 1,
                               -(x["dir_acc"] or 0), x["err_pct"] or 999))

    # Direction consensus
    dirs       = [p["direction"] for p in preds if p["direction"] in ("UP", "DOWN")]
    up_count   = dirs.count("UP")
    down_count = dirs.count("DOWN")
    up_algos   = [p["name"] for p in preds if p["direction"] == "UP"]
    down_algos = [p["name"] for p in preds if p["direction"] == "DOWN"]
    n_dirs     = len(dirs)

    if n_dirs == 0:
        consensus = "UNKNOWN"
    elif up_count > down_count * 1.5:
        consensus = f"BULLISH  ({up_count}/{n_dirs} say UP)"
    elif down_count > up_count * 1.5:
        consensus = f"BEARISH  ({down_count}/{n_dirs} say DOWN)"
    else:
        consensus = f"MIXED  ({up_count} UP vs {down_count} DOWN — no strong consensus)"

    # Cluster: group medians within 1 std of the median-of-medians
    med_vals    = [p["median"] for p in preds]
    med_centre  = float(np.median(med_vals))
    med_std     = float(np.std(med_vals)) or (med_centre * 0.01)
    cluster     = [p for p in preds if abs(p["median"] - med_centre) <= med_std]
    outliers    = [p for p in preds if abs(p["median"] - med_centre) >  med_std]

    # Actual price if date has passed
    actual_price = None
    if target_dt <= today and len(cmp_df) > 0:
        try:
            actual_row = cmp_df[cmp_df["date"].dt.normalize() == target_dt]
            if not actual_row.empty:
                actual_price = float(actual_row.iloc[0]["actual_price"])
        except Exception:
            pass

    # --- Print ---
    W = 82
    print()
    print("=" * W)
    print(f"FORECAST LOOKUP — {t}  |  Date: {target_date}  |  Day {day_num} of forecast")
    print("=" * W)
    if current_close:
        print(f"  Close at forecast time : ${current_close:.2f}    Last updated: {last_updated}")
    if n_valid > 0:
        print(f"  Historical validation  : {n_valid} past dates used for accuracy ranking")
    else:
        print(f"  Historical validation  : none yet — accuracy ranking unavailable")
    if actual_price is not None:
        move = (actual_price - current_close) / current_close * 100 if current_close else 0
        arrow = "▲" if actual_price >= (current_close or actual_price) else "▼"
        print(f"  Actual close (passed)  : ${actual_price:.2f}  {arrow} {abs(move):.1f}% from forecast close")
    print()

    has_acc = any(p["dir_acc"] is not None for p in preds)
    if has_acc:
        print(f"  {'#':<3} {'Algorithm':<12} {'Dir Acc':>8}  {'Err %':>7}  {'Median':>8}  {'Low':>8}  {'High':>8}  Dir")
        print(f"  {'-'*3} {'-'*12} {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  ---")
    else:
        print(f"  {'#':<3} {'Algorithm':<12} {'Median':>8}  {'Low':>8}  {'High':>8}  Dir")
        print(f"  {'-'*3} {'-'*12} {'-'*8}  {'-'*8}  {'-'*8}  ---")

    for i, p in enumerate(preds, 1):
        med_s  = f"${p['median']:.2f}"
        low_s  = f"${p['low']:.2f}"  if p["low"]  is not None else "—"
        high_s = f"${p['high']:.2f}" if p["high"] is not None else "—"
        dir_s  = p["direction"] or "—"
        marker = " ◄ best" if i == 1 and has_acc else ""
        if has_acc and p["dir_acc"] is not None:
            print(f"  {i:<3} {p['name']:<12} {p['dir_acc']:>7.1f}%  {p['err_pct']:>6.1f}%  {med_s:>8}  {low_s:>8}  {high_s:>8}  {dir_s}{marker}")
        elif has_acc:
            print(f"  {i:<3} {p['name']:<12} {'n/a':>8}  {'n/a':>7}  {med_s:>8}  {low_s:>8}  {high_s:>8}  {dir_s}")
        else:
            print(f"  {i:<3} {p['name']:<12} {med_s:>8}  {low_s:>8}  {high_s:>8}  {dir_s}")

    print()
    print(f"DIRECTION CONSENSUS:  {consensus}")
    if up_algos:
        print(f"  UP  : {', '.join(up_algos)}")
    if down_algos:
        print(f"  DOWN: {', '.join(down_algos)}")

    print()
    best = preds[0]
    trust_note = f"  ({best['dir_acc']:.1f}% dir accuracy, {best['err_pct']:.1f}% avg error on {best['n_hist']} past dates)" if best["dir_acc"] is not None else ""
    low_prob = _match_hit_probability(best["low"], best["p_bands"])
    med_prob = _match_hit_probability(best["median"], best["p_bands"])
    high_prob = _match_hit_probability(best["high"], best["p_bands"])
    print(f"MOST TRUSTED: {best['name'].upper()}{trust_note}")
    median_line = f"  Median: ${best['median']:.2f}"
    if med_prob is not None:
        median_line += f"  ({med_prob}% chance price closes above this)"
    median_line += f"   Direction: {best['direction'] or '—'}"
    print(median_line)
    if best["low"] is not None:
        range_line = f"  Range:  ${best['low']:.2f}"
        if low_prob is not None:
            range_line += f" ({low_prob}%)"
        range_line += " – "
        range_line += f"${best['high']:.2f}"
        if high_prob is not None:
            range_line += f" ({high_prob}%)"
        print(range_line)
    for prob in [90, 60, 50, 20]:
        if prob in best["p_bands"]:
            print(f"  p{prob}:   ${best['p_bands'][prob]:.2f}  ({prob}% chance price closes above this)")

    print()
    if cluster:
        cmin = min(p["median"] for p in cluster)
        cmax = max(p["median"] for p in cluster)
        print(f"CLUSTER:  ${cmin:.2f} – ${cmax:.2f}  ({len(cluster)} algorithms: {', '.join(p['name'] for p in cluster)})")
    if outliers:
        for o in outliers:
            print(f"  Outlier: {o['name']} at ${o['median']:.2f}")
    print("=" * W)
    print()


_LEADERBOARD_MODELS = [
    ("analog",       "analog_price",       None,                None),
    ("spearman",     "spearman_median",    "spearman_low",      "spearman_high"),
    ("pearson",      "pearson_median",     "pearson_low",       "pearson_high"),
    ("cosine",       "cosine_median",      "cosine_low",        "cosine_high"),
    ("euclidean",    "euclidean_median",   "euclidean_low",     "euclidean_high"),
    ("kendall",      "kendall_median",     "kendall_low",       "kendall_high"),
    ("manhattan",    "manhattan_median",   "manhattan_low",     "manhattan_high"),
    ("xgboost",      "xgb_median",         "xgb_low",           "xgb_high"),
    ("lightgbm",     "lgb_median",         "lgb_low",           "lgb_high"),
    ("randomforest", "rf_median",          "rf_low",            "rf_high"),
    ("knn2",         "knn2_median",        "knn2_low",          "knn2_high"),
]


def _week_range(anchor: pd.Timestamp) -> tuple:
    """Return (monday, friday) of the ISO week containing anchor."""
    mon = anchor - pd.Timedelta(days=anchor.weekday())
    fri = mon + pd.Timedelta(days=4)
    return mon.normalize(), fri.normalize()


def _leaderboard_from_df(df: pd.DataFrame) -> list:
    """Compute leaderboard rows from a (possibly filtered) comparison DataFrame."""
    rows = []
    for name, med_col, low_col, high_col in _LEADERBOARD_MODELS:
        if med_col not in df.columns:
            continue
        valid = df.dropna(subset=[med_col])
        if len(valid) == 0:
            continue
        errors   = (valid[med_col] - valid["actual_price"]).abs()
        avg_err  = errors.mean()
        avg_pct  = (errors / valid["actual_price"] * 100).mean()
        pred_dir = np.sign(valid[med_col] - valid["prior_close"])
        act_dir  = np.sign(valid["actual_price"] - valid["prior_close"])
        dir_acc  = (pred_dir == act_dir).mean() * 100
        if low_col and high_col and low_col in df.columns and high_col in df.columns:
            in_band = (
                (valid["actual_price"] >= valid[low_col]) &
                (valid["actual_price"] <= valid[high_col])
            ).mean() * 100
            in_band_str = f"{in_band:.1f}%"
        else:
            in_band_str = "n/a"
        rows.append((name, dir_acc, avg_err, avg_pct, in_band_str))
    rows.sort(key=lambda x: x[3])
    return rows


def _print_leaderboard(rows: list) -> None:
    print(f"  {'Rank':<5} {'Model':<12} {'Dir Acc':>9}  {'Avg Err $':>10}  {'Avg Err %':>10}  {'In-Band':>8}")
    print(f"  {'-'*5} {'-'*12} {'-'*9}  {'-'*10}  {'-'*10}  {'-'*8}")
    for rank, (name, dir_acc, avg_err, avg_pct, in_band_str) in enumerate(rows, 1):
        print(f"  {rank:<5} {name:<12} {dir_acc:>8.1f}%  ${avg_err:>9.2f}  {avg_pct:>9.1f}%  {in_band_str:>8}")


def run_validate_week(ticker: str, anchor_date: str, go_back: int = 0) -> None:
    """
    Validate forecasts for the week(s) containing anchor_date.
    go_back=0 → just that week. go_back=N → that week + N prior weeks.
    """
    t = ticker.upper()
    try:
        anchor = pd.Timestamp(anchor_date).normalize()
    except Exception:
        print(f"Invalid date: {anchor_date}. Use YYYY-MM-DD.")
        return

    forecast_path = f"output/{t}/forecast_results_{t}.csv"
    if not os.path.exists(forecast_path):
        print(f"No forecast file: {forecast_path}")
        return

    config = {
        "TICKER":     t,
        "PERIOD":     cfg.PERIOD,
        "INTERVAL":   cfg.INTERVAL,
        "CACHE_PATH": f"data/{t}_max.csv",
    }
    df = fetch_data(config["TICKER"], config["PERIOD"], config["INTERVAL"], config["CACHE_PATH"])

    full_cmp = perform_comparison(df, forecast_path)
    if full_cmp.empty:
        print("No elapsed forecast dates to validate.")
        return

    full_cmp["date"] = pd.to_datetime(full_cmp["date"]).dt.normalize()

    # Build week ranges: anchor week + go_back prior weeks
    weeks = []
    for i in range(go_back + 1):
        target = anchor - pd.Timedelta(weeks=i)
        mon, fri = _week_range(target)
        weeks.append((mon, fri))
    weeks.reverse()  # chronological order

    W = 70
    print()
    print("=" * W)
    n_weeks = go_back + 1
    print(f"WEEK VALIDATION — {t}  |  anchor: {anchor_date}  |  {n_weeks} week(s)")
    print("=" * W)

    combined_frames = []

    for mon, fri in weeks:
        week_df = full_cmp[(full_cmp["date"] >= mon) & (full_cmp["date"] <= fri)].copy()
        label   = f"Week of {mon.strftime('%Y-%m-%d')} → {fri.strftime('%Y-%m-%d')}"
        print()
        print(f"  {label}  ({len(week_df)} trading day(s) with data)")
        print(f"  {'-' * (len(label) + 30)}")

        if week_df.empty:
            print("  No forecast data for this week.")
            continue

        combined_frames.append(week_df)

        # Per-day summary
        for _, row in week_df.sort_values("date").iterrows():
            date_str = str(row["date"])[:10]
            actual   = row.get("actual_price", float("nan"))
            day_num  = int(row["day"]) if "day" in row.index else "?"
            knn_med  = row.get("knn_median", float("nan"))
            knn_err  = abs(knn_med - actual) if pd.notna(knn_med) and pd.notna(actual) else float("nan")
            knn_pct  = knn_err / actual * 100 if pd.notna(knn_err) and actual else float("nan")
            knn_dir  = row.get("knn_direction", "")
            act_dir  = "UP" if pd.notna(actual) and pd.notna(row.get("prior_close")) and actual > row["prior_close"] else "DOWN"
            dir_ok   = "✓" if knn_dir == act_dir else "✗"
            knn_low  = row.get("knn_low", float("nan"))
            knn_high = row.get("knn_high", float("nan"))
            in_band  = (
                pd.notna(knn_low) and pd.notna(knn_high) and pd.notna(actual) and
                knn_low <= actual <= knn_high
            )
            band_str = "IN" if in_band else "OUT"
            err_str  = f"${knn_err:.2f} ({knn_pct:.1f}%)" if pd.notna(knn_err) else "n/a"
            actual_str = f"${actual:.2f}" if pd.notna(actual) else "n/a"
            print(f"    Day {day_num:<3} {date_str}  actual={actual_str:<10}  kNN err={err_str:<18}  dir={dir_ok}  band={band_str}")

        # Per-week leaderboard
        lb = _leaderboard_from_df(week_df)
        if lb:
            print()
            print(f"  Leaderboard — {label}:")
            _print_leaderboard(lb)

    # Combined leaderboard across all weeks
    if len(combined_frames) > 1:
        combined = pd.concat(combined_frames, ignore_index=True)
        lb_all = _leaderboard_from_df(combined)
        print()
        print("=" * W)
        print(f"COMBINED LEADERBOARD — all {n_weeks} weeks ({len(combined)} days):")
        _print_leaderboard(lb_all)

    print()
    print("=" * W)
    print()


def main() -> None:
    run_comparison(cfg.TICKER)


if __name__ == "__main__":
    _args      = sys.argv[1:]
    _ticker    = None
    _date      = None
    _cmd       = None
    _go_back   = 0
    _verdict_view = False

    for _a in _args:
        if _a.lower() == "compare_all":
            _cmd = "compare_all"
        elif _a.lower() == "validate_week":
            _cmd = "validate_week"
        elif _a.lower() in ("--verdict", "verdict"):
            _verdict_view = True
        elif _a.lower().startswith("date="):
            _date = _a[5:]
        elif _a.lower().startswith("go_back="):
            try:
                _go_back = int(_a[8:])
            except ValueError:
                pass
        elif not _a.startswith("-"):
            if _ticker is None:
                _ticker = _a.upper()

    if _cmd == "compare_all":
        compare_all(verdict_view=_verdict_view)
    elif _cmd == "validate_week":
        if not _date:
            print("Usage: compare_forecasts.py TICKER validate_week date=YYYY-MM-DD [go_back=N]")
        else:
            run_validate_week(_ticker or cfg.TICKER, _date, _go_back)
    elif _date:
        run_date_lookup(_ticker or cfg.TICKER, _date)
    elif _ticker:
        run_comparison(_ticker)
    else:
        main()
