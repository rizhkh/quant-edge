"""
Generate the HTML forecast page from forecast_results_{T}.csv.

Usage:
  venv/bin/python forecast_html.py MU      # one ticker
  venv/bin/python forecast_html.py all     # every ticker in output/
"""
import os
import sys
from validation.forecast_html import generate_forecast_report


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


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: venv/bin/python forecast_html.py {TICKER | all}")
        sys.exit(1)
    if args[0].lower() == "all":
        tickers = discover_tickers()
        n = 0
        for t in tickers:
            path = generate_forecast_report(t)
            if path:
                n += 1
                print(f"  {t:<8}  → {path}")
            else:
                print(f"  {t:<8}  — no forecast CSV, skipped")
        print(f"\n{n}/{len(tickers)} reports generated.")
    else:
        t = args[0].upper()
        path = generate_forecast_report(t)
        if path:
            print(f"Wrote {path}")
        else:
            print(f"No forecast CSV for {t}")


if __name__ == "__main__":
    main()
