"""
Audit: which tickers have tuned params, which use defaults, which are missing.

For each (ticker, model) cell, returns one of:
  ✓ TUNED     — params file exists AND has sweep/metrics evidence
  ⚠ FILE      — file exists but no sweep metrics (manually set or stale)
  ✗ MISSING   — no params file (model would fall back to config.py default)

Usage:
  venv/bin/python params_audit.py            # full audit
  venv/bin/python params_audit.py compact    # one-line-per-ticker summary
"""
import os
import sys
import yaml


# (display_name, params_filename, "tuned" detection key)
MODEL_FILES = [
    ("knn-global",   "best_params_knn.yaml",         "metrics"),
    ("spearman",     "best_params_spearman.yaml",    "metrics"),
    ("pearson",      "best_params_pearson.yaml",     "metrics"),
    ("cosine",       "best_params_cosine.yaml",      "metrics"),
    ("euclidean",    "best_params_euclidean.yaml",   "metrics"),
    ("kendall",      "best_params_kendall.yaml",     "metrics"),
    ("manhattan",    "best_params_manhattan.yaml",   "metrics"),
    ("xgboost",      "best_params_xgb.yaml",         "metrics"),
    ("lightgbm",     "best_params_lgbm.yaml",        "metrics"),
    ("randomforest", "best_params_rf.yaml",          "metrics"),
    ("knn2",         "feature_weights_knn2.yaml",    "sweep_dir_acc"),
]


def _has_tuned_evidence(yaml_path: str, key: str) -> bool:
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
        if key in data:
            v = data[key]
            return v is not None and v != "" and v != {} and v != []
        return False
    except Exception:
        return False


def discover_tickers() -> list:
    if not os.path.isdir("output"):
        return []
    out = []
    for d in sorted(os.listdir("output")):
        low = d.lower()
        if "copy" in low or "test" in low:
            continue
        if os.path.isdir(f"output/{d}"):
            out.append(d)
    return out


def audit_ticker(ticker: str) -> dict:
    pdir = f"output/{ticker}/params"
    result = {}
    for label, fname, key in MODEL_FILES:
        path = f"{pdir}/{fname}"
        if not os.path.exists(path):
            result[label] = "MISSING"
        elif _has_tuned_evidence(path, key):
            result[label] = "TUNED"
        else:
            result[label] = "FILE"
    return result


def _icon(state: str) -> str:
    return {"TUNED": "✓", "FILE": "⚠", "MISSING": "✗"}.get(state, "?")


def _color_code(state: str) -> str:
    return {"TUNED": "\033[32m", "FILE": "\033[33m", "MISSING": "\033[31m"}.get(state, "")


RESET = "\033[0m"


def print_full(tickers: list) -> None:
    rows = []
    for t in tickers:
        rows.append((t, audit_ticker(t)))

    # Column header — model labels abbreviated to fit
    abbrev = {
        "knn-global": "knn", "spearman": "spr", "pearson": "pea", "cosine": "cos",
        "euclidean": "euc", "kendall": "ken", "manhattan": "man",
        "xgboost": "xgb", "lightgbm": "lgb",
        "randomforest": "rf", "knn2": "knn2",
    }
    header_cols = " ".join(f"{abbrev[m[0]]:>4}" for m in MODEL_FILES)
    print()
    print(f"  {'Ticker':<8} {header_cols}   counts (✓ tuned / ⚠ file / ✗ missing)")
    print(f"  {'-'*8} {'-'*len(header_cols)}   {'-'*42}")

    summary = {"TUNED": 0, "FILE": 0, "MISSING": 0}
    for t, audit in rows:
        cells = []
        c_tuned = c_file = c_miss = 0
        for label, _, _ in MODEL_FILES:
            state = audit[label]
            cells.append(f"{_color_code(state)}{_icon(state):>4}{RESET}")
            summary[state] += 1
            if state == "TUNED":   c_tuned += 1
            elif state == "FILE":  c_file += 1
            else:                  c_miss += 1
        counts = f"{c_tuned:>2}✓  {c_file:>2}⚠  {c_miss:>2}✗"
        print(f"  {t:<8} {' '.join(cells)}   {counts}")

    total = sum(summary.values())
    print()
    print(f"  Total cells: {total}")
    print(f"  ✓ TUNED   (file exists + sweep metrics): {summary['TUNED']:>4}  ({summary['TUNED']/total*100:.1f}%)")
    print(f"  ⚠ FILE    (file exists, no sweep evidence): {summary['FILE']:>4}  ({summary['FILE']/total*100:.1f}%)")
    print(f"  ✗ MISSING (no file → uses config.py default): {summary['MISSING']:>4}  ({summary['MISSING']/total*100:.1f}%)")
    print()
    print("  Legend: ✓ = tuned via param_sweep   ⚠ = yaml present but stale/manual   ✗ = falling back to config.py")


def print_compact(tickers: list) -> None:
    print()
    print(f"  {'Ticker':<8} {'Status':<14} Tuned / Total")
    print(f"  {'-'*8} {'-'*14} {'-'*15}")
    for t in tickers:
        a = audit_ticker(t)
        n_tuned = sum(1 for v in a.values() if v == "TUNED")
        total   = len(a)
        if n_tuned == total:
            status = "FULLY TUNED"
        elif n_tuned == 0:
            status = "ALL DEFAULTS"
        elif n_tuned >= total * 0.8:
            status = "MOSTLY TUNED"
        elif n_tuned >= total * 0.5:
            status = "PARTIAL"
        else:
            status = "MOSTLY DEFAULTS"
        print(f"  {t:<8} {status:<14} {n_tuned}/{total}")
    print()


def main():
    args = sys.argv[1:]
    tickers = discover_tickers()
    if not tickers:
        print("No tickers in output/")
        return
    if args and args[0].lower() == "compact":
        print_compact(tickers)
    else:
        print_full(tickers)


if __name__ == "__main__":
    main()
