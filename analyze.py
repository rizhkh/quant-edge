"""
Detailed validation analysis. Reads forecast_validation_{T}.csv,
applies trust/warning heuristics, prints a verdict + writes markdown report.

Usage:
  venv/bin/python analyze.py HUT          # one ticker
  venv/bin/python analyze.py all          # every ticker in output/
"""
import os
import sys
from validation.analyzer import write_report


def _print_console(result: dict, path: str = None) -> None:
    if "error" in result:
        print(f"  {result['ticker']:<8}  ERROR: {result['error']}")
        return

    badge = {"TRUST": "✅", "WEAK SIGNAL": "⚠️", "NONE": "🛑"}.get(result["verdict"], "")
    rec   = result["recommended"] or "—"
    n     = result["n_dates"]
    print()
    print("=" * 78)
    print(f"  {result['ticker']}  |  {n} days  ({result['date_min']} → {result['date_max']})")
    print("=" * 78)
    print(f"  {badge} OVERALL: {result['verdict']}    Recommended: {rec}")
    if result["reasons"]:
        for r in result["reasons"]:
            print(f"      • {r}")
    if result.get("action"):
        print(f"  ➤ Action: {result['action']}")
    if result["caveats"]:
        print(f"  Caveats:")
        for c in result["caveats"]:
            print(f"      • {c}")

    # Recent verdict
    rv = result.get("recent_verdict")
    if rv:
        rbadge = {"TRUST": "✅", "WEAK SIGNAL": "⚠️", "NONE": "🛑"}.get(rv["verdict"], "")
        print(f"  {rbadge} RECENT (last {rv['n_recent']} days): {rv['verdict']}    "
              f"Best: {rv['best_model']} ({rv['best_dir']:.0f}% dir)")
        if rv["all_below_50"]:
            print(f"      ⚠ All models below 50% direction recently — active regime decay")
        if rv["verdict"] != result["verdict"]:
            print(f"      ⚠ Recent differs from overall — lean on recent for short-horizon trades")

    if result["right"]:
        print(f"\n  ✓ Going right:")
        for r in result["right"]:
            print(f"      • {r}")
    if result["wrong"]:
        print(f"\n  ✗ Going wrong:")
        for w in result["wrong"]:
            print(f"      • {w}")
    if result["observations"]:
        print(f"\n  • Observations:")
        for o in result["observations"]:
            print(f"      • {o}")
    if result["usage"]:
        print(f"\n  Recommended usage:")
        for k, v in result["usage"].items():
            print(f"      {k}: {v}")
    if path:
        print(f"\n  → Saved {path}")


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


def run_one(ticker: str) -> None:
    path, result = write_report(ticker.upper())
    _print_console(result, path)


def run_all() -> None:
    tickers = discover_tickers()
    if not tickers:
        print("No tickers found in output/")
        return
    print(f"Analyzing {len(tickers)} tickers: {', '.join(tickers)}")
    summary_rows = []
    for t in tickers:
        path, result = write_report(t)
        _print_console(result, path)
        summary_rows.append({
            "ticker":   t,
            "verdict":  result.get("verdict", "ERROR"),
            "rec":      result.get("recommended") or "—",
            "n":        result.get("n_dates", 0),
        })

    # Cross-ticker summary
    print()
    print("=" * 78)
    print("CROSS-TICKER SUMMARY")
    print("=" * 78)
    rank = {"TRUST": 0, "WEAK SIGNAL": 1, "NONE": 2, "ERROR": 3}
    badges = {"TRUST": "✅", "WEAK SIGNAL": "⚠️ ", "NONE": "🛑", "ERROR": "❌"}
    summary_rows.sort(key=lambda r: (rank.get(r["verdict"], 9), -r["n"]))
    print(f"  {'Ticker':<8} {'':<3}{'Verdict':<14} {'Recommended':<15} {'N':>4}")
    print(f"  {'-'*8} {'-'*3}{'-'*14} {'-'*15} {'-'*4}")
    for r in summary_rows:
        badge = badges.get(r["verdict"], "  ")
        print(f"  {r['ticker']:<8} {badge:<3}{r['verdict']:<14} {r['rec']:<15} {r['n']:>4}")
    print()
    n_trust = sum(1 for r in summary_rows if r["verdict"] == "TRUST")
    n_weak  = sum(1 for r in summary_rows if r["verdict"] == "WEAK SIGNAL")
    n_none  = sum(1 for r in summary_rows if r["verdict"] == "NONE")
    n_err   = sum(1 for r in summary_rows if r["verdict"] == "ERROR")
    print(f"  ✅ TRUST: {n_trust}   ⚠️  WEAK: {n_weak}   🛑 NONE: {n_none}   ❌ ERROR: {n_err}")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: venv/bin/python analyze.py {TICKER | all}")
        sys.exit(1)
    arg = args[0].lower()
    if arg == "all":
        run_all()
    else:
        run_one(args[0])
