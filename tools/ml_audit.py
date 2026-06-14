"""
ML Feature Audit — Step 1 of ML_FEATURE_AUDIT_PLAN.txt

Scans every ticker's forecast_validation_{T}.csv and ranks XGBoost +
LightGBM relative to the kNN stack. Prints two ranked tables + aggregate
summary.

Usage:
  venv/bin/python tools/ml_audit.py
"""
import os
import sys
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.analyzer import _model_stats


def discover_tickers():
    out = []
    for d in sorted(os.listdir("output")):
        low = d.lower()
        if "copy" in low or "test" in low:
            continue
        if os.path.isdir(f"output/{d}"):
            out.append(d)
    return out


def ml_audit_one(ticker: str) -> dict:
    path = f"output/{ticker}/forecast_validation_{ticker}.csv"
    if not os.path.exists(path):
        return {"ticker": ticker, "error": "no validation file"}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"ticker": ticker, "error": f"read error: {e}"}
    if len(df) == 0:
        return {"ticker": ticker, "error": "empty"}

    stats = _model_stats(df)
    if not stats:
        return {"ticker": ticker, "error": "no usable model data"}

    # Sort by conservative score desc, build rank map
    stats_sorted = sorted(stats, key=lambda r: -r["score"])
    rank_map = {r["model"]: i + 1 for i, r in enumerate(stats_sorted)}
    n_models = len(stats_sorted)

    by_model = {r["model"]: r for r in stats}

    def grab(name):
        r = by_model.get(name)
        if not r:
            return None
        return {
            "n":       r["n"],
            "dir":     r["dir"],
            "err":     r["avg_err"],
            "in_band": r["in_band"],
            "score":   r["score"],
            "rank":    rank_map[name],
            "of":      n_models,
        }

    # Best kNN by score (for delta calculation)
    knn_methods = {"spearman", "pearson", "cosine", "euclidean", "kendall", "manhattan", "knn2"}
    knn_rows = [r for r in stats_sorted if r["model"] in knn_methods]
    best_knn = knn_rows[0] if knn_rows else None

    return {
        "ticker":   ticker,
        "xgboost":  grab("xgboost"),
        "lightgbm": grab("lightgbm"),
        "randomforest": grab("randomforest"),
        "best_knn": (best_knn["model"], best_knn["dir"], best_knn["score"]) if best_knn else None,
    }


def _fmt(v, suffix=""):
    if v is None: return "—"
    return f"{v}{suffix}"


def _rank_color(rank, total):
    """Return a label based on rank position."""
    if rank is None: return "—"
    pct = rank / total
    if pct <= 0.25: return f"{rank}/{total} top"
    if pct <= 0.50: return f"{rank}/{total} upper"
    if pct <= 0.75: return f"{rank}/{total} lower"
    return f"{rank}/{total} BOT"


def print_model_table(results, model_key, model_label):
    print()
    print("=" * 100)
    print(f"  {model_label} — per-ticker performance (sorted by dir acc)")
    print("=" * 100)
    rows = []
    for r in results:
        if r.get("error"):
            continue
        m = r.get(model_key)
        if not m:
            continue
        # delta vs best knn
        delta_dir = None
        if r.get("best_knn") and m["dir"] is not None and r["best_knn"][1] is not None:
            delta_dir = round(m["dir"] - r["best_knn"][1], 1)
        rows.append({
            "ticker": r["ticker"],
            "n":      m["n"],
            "dir":    m["dir"],
            "err":    m["err"],
            "ib":     m["in_band"],
            "score":  m["score"],
            "rank":   _rank_color(m["rank"], m["of"]),
            "best_knn": r["best_knn"][0] if r["best_knn"] else "—",
            "delta_dir": delta_dir,
        })
    rows.sort(key=lambda r: (r["dir"] is None, -(r["dir"] or 0)))

    print(f"  {'Ticker':<7} {'N':>3}  {'Dir%':>5}  {'Err%':>5}  {'InBand%':>7}  {'Score':>5}  {'Rank':<12} {'BestKNN':<12} {'ΔDir':>5}")
    print(f"  {'-'*7} {'-'*3}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*12} {'-'*12} {'-'*5}")
    for r in rows:
        ddir = f"{r['delta_dir']:+.1f}" if r['delta_dir'] is not None else "  —"
        print(f"  {r['ticker']:<7} {r['n']:>3}  {_fmt(r['dir']):>5}  {_fmt(r['err']):>5}  {_fmt(r['ib']):>7}  {_fmt(r['score']):>5}  {r['rank']:<12} {r['best_knn']:<12} {ddir:>5}")
    return rows


def print_aggregate(xgb_rows, lgb_rows, rf_rows):
    print()
    print("=" * 100)
    print("  AGGREGATE SUMMARY")
    print("=" * 100)

    def bucket(rows):
        out = {"top": 0, "upper": 0, "lower": 0, "bot": 0, "total": 0,
               "underperform": 0, "outperform": 0,
               "mean_delta": 0.0, "deltas": []}
        for r in rows:
            label = r["rank"].split(" ")[1] if " " in r["rank"] else ""
            if   label == "top":   out["top"]   += 1
            elif label == "upper": out["upper"] += 1
            elif label == "lower": out["lower"] += 1
            elif label == "BOT":   out["bot"]   += 1
            out["total"] += 1
            if r.get("delta_dir") is not None:
                out["deltas"].append(r["delta_dir"])
                if r["delta_dir"] < 0: out["underperform"] += 1
                else:                  out["outperform"]   += 1
        if out["deltas"]:
            out["mean_delta"] = round(sum(out["deltas"]) / len(out["deltas"]), 1)
        return out

    for label, rows in [("XGBoost", xgb_rows), ("LightGBM", lgb_rows), ("RandomForest", rf_rows)]:
        b = bucket(rows)
        if b["total"] == 0:
            print(f"  {label:<14} no data")
            continue
        print(f"\n  {label}")
        print(f"    Total tickers with data:  {b['total']}")
        print(f"    Top quartile rank:        {b['top']}/{b['total']}")
        print(f"    Upper-mid rank:           {b['upper']}/{b['total']}")
        print(f"    Lower-mid rank:           {b['lower']}/{b['total']}")
        print(f"    Bottom quartile rank:     {b['bot']}/{b['total']}")
        print(f"    Beats best-kNN dir:       {b['outperform']}/{b['total']}")
        print(f"    Loses to best-kNN dir:    {b['underperform']}/{b['total']}")
        print(f"    Mean dir gap vs best-kNN: {b['mean_delta']:+.1f}pts")


def print_verdict(xgb_rows, lgb_rows, rf_rows):
    print()
    print("=" * 100)
    print("  HYPOTHESIS VERDICT")
    print("=" * 100)

    def hyp(label, rows):
        if not rows:
            print(f"  {label:<14} NO DATA")
            return
        total = len(rows)
        bot = sum(1 for r in rows if r["rank"].endswith("BOT"))
        underperform = sum(1 for r in rows if (r.get("delta_dir") or 0) < 0)
        top6 = sum(1 for r in rows
                   if r["rank"].endswith("top") or r["rank"].endswith("upper"))
        # Decision rules from plan:
        if top6 >= 10:
            v = "FINE — model ranks top-6 on 10+ tickers. Don't replace features."
        elif bot + underperform >= total * 0.6 and underperform >= 10:
            v = "REPLACE — model in bottom rank or below best-kNN on most tickers. Proceed to Step 2."
        else:
            v = "MIXED — some tickers benefit, some don't. Investigate per-ticker."
        print(f"  {label:<14} {v}")
        print(f"  {'':<14}   top-6 ranks: {top6}/{total} · bottom-quartile: {bot}/{total} · loses to best-kNN: {underperform}/{total}")

    hyp("XGBoost",      xgb_rows)
    hyp("LightGBM",     lgb_rows)
    hyp("RandomForest", rf_rows)


def main():
    tickers = discover_tickers()
    if not tickers:
        print("No tickers in output/")
        return
    print(f"Scanning {len(tickers)} tickers: {', '.join(tickers)}")
    results = [ml_audit_one(t) for t in tickers]

    errors = [r for r in results if r.get("error")]
    if errors:
        print()
        print("Skipped:")
        for r in errors:
            print(f"  {r['ticker']:<8} — {r['error']}")

    xgb_rows = print_model_table(results, "xgboost",  "XGBoost")
    lgb_rows = print_model_table(results, "lightgbm", "LightGBM")
    rf_rows  = print_model_table(results, "randomforest", "RandomForest")
    print_aggregate(xgb_rows, lgb_rows, rf_rows)
    print_verdict(xgb_rows, lgb_rows, rf_rows)
    print()


if __name__ == "__main__":
    main()
