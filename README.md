# quant-edge

A personal stock-insight tool I built to generate short-horizon price forecasts
by applying a mix of algorithms to historical market data. It combines a
**historical-analog / k-NN similarity search** with gradient-boosted and
tree-based **machine-learning models** (XGBoost, LightGBM, Random Forest) to
produce a probabilistic view of where a stock might go over the next N trading
days.

Given a ticker, it pulls historical OHLCV data, engineers a set of scale-free
technical features, finds the most similar past market windows, and projects a
probabilistic forecast cone forward. A **walk-forward backtester** is included
to measure how valid each algorithm actually is — running it over historical
data with no lookahead so I can see which methods are worth trusting.

> ⚠️ **Disclaimer** — This is a personal research / educational project. It is
> **not financial advice**, and nothing here is a recommendation to buy or sell
> any security. Markets are not predictable; use at your own risk.

---

## Features

- **Analog forecasting** — finds the `K` most similar historical windows using
  Spearman, Pearson, cosine, Euclidean, Manhattan, or Kendall similarity, then
  projects forward paths into a probabilistic cone.
- **k-NN forecaster** — distance-weighted nearest-neighbour projection with
  configurable confidence bands (e.g. P20 / P60 / P90).
- **ML models** — XGBoost, LightGBM, and Random Forest forecasters, each with
  their own engineered feature sets (sector-cohort, VIX regime, earnings
  proximity, momentum, volatility regime, etc.).
- **Walk-forward backtester** — validates the algorithms by re-running them at
  many historical points with no lookahead, reporting MAE / MAPE, directional
  accuracy, and cone hit-rate so I can tell which methods are actually reliable.
- **Parameter sweeps** — grid search over model/k-NN hyper-parameters per
  ticker, with best params cached to YAML.
- **Validation & reporting** — trust/warning heuristics, live forecast
  validation against realised prices, and HTML / Markdown reports.
- **Batch runners** — sweep or forecast every tracked ticker in parallel.

## Requirements

- Python 3.10+
- An internet connection (data is fetched live from Yahoo Finance via `yfinance`)

Python dependencies are listed in [requirements.txt](requirements.txt):
`yfinance`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`,
`seaborn`, `xgboost`, `lightgbm`, `pyyaml`.

## Installation

```bash
# clone, then from the project root:
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The commands below assume the virtual environment's Python. Either activate the
venv (and use `python ...`) or call it directly with `venv/bin/python ...`
(`venv\Scripts\python.exe` on Windows).

## Usage

Set your ticker in [config.py](config.py), then run:

```bash
venv/bin/python main.py
```

That's it — this generates the forecast for the configured ticker. The repo
also includes scripts for backtesting, parameter tuning, and reporting; explore
them if you want to go deeper.

## How it works

1. **Fetch** — OHLCV history is downloaded from Yahoo Finance and cached to
   `data/<TICKER>_max.csv`.
2. **Engineer features** — returns, intraday range, relative volume, momentum,
   Bollinger %B, ATR-normalised volatility, close-location value, vol regime,
   plus per-model extras (sector cohort, VIX regime, earnings proximity).
3. **Search** — the most recent window is compared against all historical
   windows; the top `K` analogs (with `MIN_GAP` spacing) are selected.
4. **Project** — forward paths from those analogs (or ML model predictions) are
   aggregated into a probabilistic cone with confidence bands.
5. **Score** — walk-forward backtests and validation heuristics measure
   directional accuracy and calibration before any forecast is trusted.

