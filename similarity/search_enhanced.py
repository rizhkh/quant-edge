"""
Enhanced kNN search with:
  - Feature-weighted distance (Step 2)
  - Regime-conditional matching (Step 4)
  - Adaptive k via distance threshold (Step 6)
  - Returns age_bars per match so caller can apply recency weighting (Step 5)

Only the distance-based methods benefit from feature weighting.
Rank-based methods (spearman, kendall) are unchanged — scaling a feature
before ranking does not change its rank.
"""

import numpy as np
from .search import _build_cand_matrix, _normalize_batch

_RANK_METHODS = {"spearman", "kendall"}


# ---------------------------------------------------------------------------
# Feature weighting helpers
# ---------------------------------------------------------------------------

def _expand_weights(weights: np.ndarray, window_len: int, n_features: int) -> np.ndarray:
    """
    Tile feature weights across the time dimension so the weight for feature f
    applies to every time step in the flattened window.
    Returns array of shape (window_len * n_features,).
    """
    return np.tile(weights, window_len)   # (w₀,w₁,...,w₇, w₀,w₁,...,w₇, ...)


def _weighted_batch_scores(
    method: str,
    q_norm: np.ndarray,
    c_all_norm: np.ndarray,
    weights: np.ndarray | None,
    window_len: int,
    n_features: int,
) -> np.ndarray:
    """
    Vectorized similarity scores with optional feature weighting.
    q_norm   : (dim,)   — normalized, flattened query
    c_all_norm: (N, dim) — normalized, flattened candidates
    weights  : (n_features,) or None
    """
    # Apply sqrt(w) scaling for non-rank methods so weighted distances work
    if weights is not None and method not in _RANK_METHODS:
        w_exp = _expand_weights(np.sqrt(weights), window_len, n_features)
        q = q_norm * w_exp
        c = c_all_norm * w_exp
    else:
        q = q_norm
        c = c_all_norm

    if method == "euclidean":
        return 1.0 / (1.0 + np.sqrt(np.sum((c - q) ** 2, axis=1)))

    if method == "manhattan":
        # True weighted Manhattan: Σ wᵢ|qᵢ-cᵢ| — recompute without sqrt
        if weights is not None:
            w_exp_raw = _expand_weights(weights, window_len, n_features)
            diffs = np.abs(c_all_norm - q_norm)
            return 1.0 / (1.0 + np.sum(w_exp_raw * diffs, axis=1))
        return 1.0 / (1.0 + np.sum(np.abs(c - q), axis=1))

    if method == "mse":
        return 1.0 / (1.0 + np.mean((c - q) ** 2, axis=1))

    if method == "cosine":
        dots  = c @ q
        norms = np.linalg.norm(c, axis=1) * np.linalg.norm(q)
        return np.where(norms == 0, 0.0, dots / norms)

    if method == "pearson":
        q_c   = q - q.mean()
        c_c   = c - c.mean(axis=1, keepdims=True)
        num   = c_c @ q_c
        denom = np.linalg.norm(c_c, axis=1) * np.linalg.norm(q_c)
        return np.where(denom == 0, 0.0, num / denom)

    if method == "spearman":
        from scipy.stats import rankdata
        q_r   = rankdata(q_norm)          # unweighted — ranks unaffected by scaling
        c_r   = np.apply_along_axis(rankdata, 1, c_all_norm)
        n     = len(q_r)
        denom = n * (n ** 2 - 1)
        if denom == 0:
            return np.zeros(len(c_all_norm))
        return 1.0 - 6.0 * ((q_r - c_r) ** 2).sum(axis=1) / denom

    if method == "mahalanobis":
        # Fallback to Euclidean (inv_cov not recomputed here for speed)
        return 1.0 / (1.0 + np.sqrt(np.sum((c - q) ** 2, axis=1)))

    if method == "kendall":
        from scipy.stats import kendalltau
        scores = np.array([
            float(kendalltau(q_norm, c_all_norm[i])[0] or 0.0)
            for i in range(len(c_all_norm))
        ])
        return scores

    raise ValueError(f"Unknown method '{method}'")


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def _detect_regime(
    all_cands: np.ndarray,
    query: np.ndarray,
    feature_cols: list,
    k_min: int,
) -> tuple[np.ndarray, str]:
    """
    Classify all candidate windows and the query into regime buckets.

    Regime = (vol_bucket, trend_bucket):
      vol_bucket   — HIGH if atr_norm > training-set median, else LOW
      trend_bucket — UP   if momentum_z > 0, else DOWN

    Falls back gracefully:
      primary (same vol + same trend) → k_min candidates
      fallback (same vol only)        → k_min candidates
      no filter                       → all candidates

    Returns (boolean mask, regime_label_string).
    """
    n = len(all_cands)

    if feature_cols is None or query.ndim == 1:
        return np.ones(n, dtype=bool), "UNKNOWN"

    try:
        atr_idx = feature_cols.index("atr_norm")
        mom_idx = feature_cols.index("momentum_z")
    except ValueError:
        return np.ones(n, dtype=bool), "UNKNOWN"

    # Per-window mean across time steps (axis=1 of shape N × window_len × n_feat)
    cand_atr = all_cands[:, :, atr_idx].mean(axis=1)
    cand_mom = all_cands[:, :, mom_idx].mean(axis=1)
    atr_threshold = np.median(cand_atr)

    q_atr = query[:, atr_idx].mean()
    q_mom = query[:, mom_idx].mean()
    q_vol_high  = bool(q_atr > atr_threshold)
    q_trend_up  = bool(q_mom > 0)

    regime_str = (
        f"{'HIGH_VOL' if q_vol_high else 'LOW_VOL'}/"
        f"{'UPTREND' if q_trend_up else 'DOWNTREND'}"
    )

    c_vol_high = cand_atr > atr_threshold
    c_trend_up = cand_mom > 0

    # Primary: exact 4-bucket match
    primary = (c_vol_high == q_vol_high) & (c_trend_up == q_trend_up)
    if primary.sum() >= k_min:
        return primary, regime_str

    # Fallback 1: same vol only (relax trend)
    vol_only = c_vol_high == q_vol_high
    if vol_only.sum() >= k_min:
        return vol_only, regime_str

    # Fallback 2: no filter
    return np.ones(n, dtype=bool), regime_str


# ---------------------------------------------------------------------------
# Main enhanced search
# ---------------------------------------------------------------------------

def find_top_k_enhanced(
    series: np.ndarray,
    window_len: int,
    forecast_len: int,
    method: str,
    weights: np.ndarray | None = None,
    feature_cols: list | None = None,
    distance_pct: float = 20.0,
    k_min: int = 5,
    k_max: int = 50,
    min_gap: int = 20,
) -> tuple[list, dict]:
    """
    Enhanced kNN search combining Steps 2, 4, 6.

    Parameters
    ----------
    series        : 2D (N, n_features) or 1D (N,)
    weights       : feature importance weights from feature_weights.py
    feature_cols  : list of feature names (for regime detection)
    distance_pct  : top X% of candidates are considered 'similar enough' (Step 6)
    k_min         : minimum neighbors before falling back / flagging low conviction
    k_max         : hard cap on returned neighbors

    Returns
    -------
    matches : list of (idx, score, age_bars)
              age_bars = 0 for the most recent candidate window
    meta    : {"regime": str, "n_neighbors": int, "regime_filtered": bool}
    """
    query      = series[-window_len:]
    search_end = len(series) - window_len * 2 - forecast_len

    if search_end <= 0:
        return [], {"regime": "UNKNOWN", "n_neighbors": 0, "regime_filtered": False}

    all_cands = _build_cand_matrix(series, window_len, search_end)  # (N, w) or (N, w, f)

    # Determine dimensions for weight expansion
    n_features = series.shape[1] if series.ndim == 2 else 1

    # Normalize using query statistics (same as existing system)
    q_norm, c_all_norm = _normalize_batch(query, all_cands, flatten=True)

    # Step 4: Regime mask
    if series.ndim == 2:
        mask, regime_str = _detect_regime(all_cands, query, feature_cols, k_min)
    else:
        mask, regime_str = np.ones(search_end, dtype=bool), "UNKNOWN"

    regime_filtered = mask.sum() < search_end

    # Step 2: Feature-weighted scores
    scores_arr = _weighted_batch_scores(method, q_norm, c_all_norm, weights, window_len, n_features)

    # Mask out non-regime candidates
    scores_arr = scores_arr.copy()
    scores_arr[~mask] = -np.inf

    # Step 6: Adaptive k — keep only top distance_pct% of regime-masked candidates
    valid_scores = scores_arr[mask]
    if len(valid_scores) > 0:
        threshold = np.percentile(valid_scores, 100.0 - distance_pct)
        too_far   = (scores_arr < threshold) & mask
        scores_arr[too_far] = -np.inf

        # If adaptive k leaves too few, relax threshold
        remaining = (scores_arr > -np.inf).sum()
        if remaining < k_min:
            scores_arr[mask] = scores_arr[mask]  # restore masked-region scores
            # Just use all regime-masked candidates
            scores_arr[~mask] = -np.inf

    # Sort descending and select with min_gap deduplication
    sorted_idx = np.argsort(scores_arr)[::-1]

    selected: list[tuple[int, float, float]] = []
    for idx in sorted_idx:
        score = float(scores_arr[idx])
        if score == -np.inf:
            break
        if not any(abs(int(idx) - s[0]) < min_gap for s in selected):
            age_bars = float(search_end - 1 - int(idx))  # 0 = most recent
            selected.append((int(idx), score, age_bars))
        if len(selected) >= k_max:
            break

    # Emergency fallback: if nothing selected, use unrestricted top-k
    if len(selected) == 0:
        q_n, c_n = _normalize_batch(query, all_cands, flatten=True)
        fb_scores = _weighted_batch_scores(method, q_n, c_n, None, window_len, n_features)
        for idx in np.argsort(fb_scores)[::-1]:
            if not any(abs(int(idx) - s[0]) < min_gap for s in selected):
                age_bars = float(search_end - 1 - int(idx))
                selected.append((int(idx), float(fb_scores[idx]), age_bars))
            if len(selected) >= k_min:
                break

    meta = {
        "regime":          regime_str,
        "n_neighbors":     len(selected),
        "regime_filtered": regime_filtered,
    }
    return selected, meta
