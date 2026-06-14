# --- Feature Engineering ---
# Columns from features/engineer.py used as the similarity search vector.
# All features are z-score normalised per-window before comparison, so scale
# differences between columns don't bias the similarity metric.
# Set to None to fall back to the legacy single-series INPUT_TYPE mode.
FEATURE_COLS = [
    "return",       # daily pct change — core price signal
    "hl_range",     # (High-Low)/Close — intraday volatility shape
    "rel_volume",   # volume vs 20d avg — participation/conviction
    "momentum_z",   # 20d mean return / 60d std — trend persistence signal
    "bb_pct_b",     # Bollinger %B — mean-reversion / breakout positioning
    "atr_norm",     # ATR(14)/Close — volatility regime (scale-free)
    "clv",          # Close Location Value — intrabar buy/sell pressure
    "vol_regime",   # 5d std / 20d std — vol expansion vs contraction
]

# --- Data ---
TICKER = "AMD"
# TICKER_ALL       = ["HUT", "KRKNF", "MRAM", "PATH", "IREN", "AAOI", "MU", "AMD", "QBTS", "ARM"]
PERIOD            = "max"
INTERVAL          = "1d"
CACHE_PATH = f"data/{TICKER}_max.csv"

# --- Analog Search Parameters ---
WINDOW_LEN        = 20
FORECAST_LEN      = 40
BARS_BACK         = 1000 #750
SIMILARITY_METHOD = "spearman"   # spearman | pearson | cosine | euclidean | mse | kendall
INPUT_TYPE        = "pct_change" # pct_change | price

# --- k-NN Parameters ---
K                 = 20 # 5
MIN_GAP           = 20
CONFIDENCE_BANDS  = [20, 60, 90]  # probability % that price closes above each level

# --- Visualization ---
SHOW_ALL_PATHS    = True
SHOW_ZIGZAG       = True
ZIGZAG_LEGS       = 3

# --- Backtest Parameters ---
BACKTEST_MONTHS         = 3         # How many months of recent history to backtest on
BACKTEST_STEP           = 1        # How often to run a forecast (trading days)
BACKTEST_FORECAST_DAYS  = 30        # Validate this many days forward of each test point
BACKTEST_START          = None      # e.g. "2025-01-01" — if set with BACKTEST_END, overrides BACKTEST_MONTHS
BACKTEST_END            = None      # e.g. "2025-05-01"

# --- ML Training Window (applies to XGBoost, LightGBM, Random Forest) ---
ML_TRAINING_LOOKBACK_BARS = 1000  # rolling window: model trains on last N bars only
                                   # None = use all history (expanding window)
                                   # 1500 ≈ 6 years of daily bars

# --- XGBoost Features (v2 — sector cohort + regime, 13 features) ---
XGB_USE_SPY      = False  # legacy SPY beta/RS features — removed in favor of sector cohort
XGB_USE_SECTOR   = True   # fetch per-ticker sector ETF for cohort features
XGB_USE_VIX      = True   # fetch ^VIX for regime bucket
XGB_USE_EARNINGS = True   # fetch days_to_earnings via yfinance .calendar

# Per-ticker sector ETF mapping for cohort features.
# Used by xgb_features.py and (eventually) rf/lgbm. Falls back to "SPY" for unknown tickers.
SECTOR_MAP = {
    # Semiconductors → SOXX
    "AAOI": "SOXX", "AEHR": "SOXX", "AMD": "SOXX", "AMKR": "SOXX",
    "ARM":  "SOXX", "AVGO": "SOXX", "LSCC": "SOXX", "MRVL": "SOXX",
    "MU":   "SOXX", "QCOM": "SOXX",
    # Crypto miners → BITQ
    "BTDR": "BITQ", "HUT": "BITQ", "IREN": "BITQ", "KRKNF": "BITQ",
    # AI/software / quantum / small-cap
    "PATH": "IGV",
    "QBTS": "IWM",
    # Materials / industrials
    "GLW":  "XLB",
    "UAMY": "XLB",
}

# --- Random Forest Parameters ---
RF_USE_SPY    = True   # fetch SPY for beta and relative strength features (requires internet)

# --- LightGBM Parameters ---
LGBM_USE_VIX  = True   # fetch ^VIX as regime feature (requires internet)
LGBM_USE_SPY  = True   # fetch SPY trend as macro context feature (requires internet)
