import numpy as np
from .metrics import get_score

# Methods with fully vectorized batch scoring (all scores computed in one numpy op)
_VECTORIZED = {"euclidean", "manhattan", "mse", "cosine", "pearson", "spearman", "mahalanobis"}

# Methods that need a pre-computed inverse covariance matrix
_NEEDS_INV_COV = {"mahalanobis"}


def _normalize_batch(query_win: np.ndarray,
                     all_cands: np.ndarray,
                     flatten: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalise query and all candidate windows using query statistics.

    query_win : (window_len,) or (window_len, n_features)
    all_cands : (N, window_len) or (N, window_len, n_features)
    flatten   : if True, flatten to (dim,) and (N, dim); if False keep 2D/3D shape.

    Returns q and c_all — flattened by default.
    """
    if query_win.ndim == 1:
        return query_win, all_cands  # 1D: no normalisation needed

    mean = query_win.mean(axis=0)                  # (n_features,)
    std  = query_win.std(axis=0)
    std  = np.where(std == 0, 1.0, std)

    q     = (query_win - mean) / std
    c_all = (all_cands  - mean) / std

    if flatten:
        return q.flatten(), c_all.reshape(len(all_cands), -1)
    return q, c_all


def _build_cand_matrix(series: np.ndarray,
                       window_len: int,
                       search_end: int) -> np.ndarray:
    """Stack all candidate windows: (N, window_len) or (N, window_len, n_features)."""
    return np.stack([series[i : i + window_len] for i in range(search_end)])


def _batch_scores(method: str, query_win: np.ndarray,
                  all_cands: np.ndarray,
                  inv_cov: np.ndarray = None) -> np.ndarray:
    """
    Vectorised similarity scores for every candidate at once.
    Returns shape (N,) float array — higher = more similar.
    """
    q, c = _normalize_batch(query_win, all_cands)

    if method == "euclidean":
        return 1.0 / (1.0 + np.sqrt(np.sum((c - q) ** 2, axis=1)))

    if method == "manhattan":
        return 1.0 / (1.0 + np.sum(np.abs(c - q), axis=1))

    if method == "mse":
        return 1.0 / (1.0 + np.mean((c - q) ** 2, axis=1))

    if method == "cosine":
        dots  = c @ q
        norms = np.linalg.norm(c, axis=1) * np.linalg.norm(q)
        return np.where(norms == 0, 0.0, dots / norms)

    if method == "pearson":
        # Pearson = cosine similarity of mean-centred vectors (identical ranking)
        q_c   = q - q.mean()
        c_c   = c - c.mean(axis=1, keepdims=True)
        num   = c_c @ q_c
        denom = np.linalg.norm(c_c, axis=1) * np.linalg.norm(q_c)
        return np.where(denom == 0, 0.0, num / denom)

    if method == "spearman":
        from scipy.stats import rankdata
        q_r   = rankdata(q)
        c_r   = np.apply_along_axis(rankdata, 1, c)
        n     = len(q)
        denom = n * (n ** 2 - 1)
        if denom == 0:
            return np.zeros(len(c))
        return 1.0 - 6.0 * ((q_r - c_r) ** 2).sum(axis=1) / denom

    if method == "mahalanobis":
        diff = c - q
        if inv_cov is not None:
            mah_sq = np.sum((diff @ inv_cov) * diff, axis=1)
        else:
            mah_sq = np.sum(diff ** 2, axis=1)
        return 1.0 / (1.0 + np.sqrt(np.maximum(mah_sq, 0.0)))

    raise ValueError(f"No vectorized implementation for '{method}'")


def _build_inv_cov(query_win: np.ndarray,
                   all_cands: np.ndarray) -> np.ndarray | None:
    """Inverse covariance matrix from normalised candidate windows (Mahalanobis)."""
    _, c = _normalize_batch(query_win, all_cands)
    cov  = np.cov(c.T)
    try:
        return np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
    except np.linalg.LinAlgError:
        return None


def _prepare_windows(query_win: np.ndarray,
                     cand_win: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-pair normalisation — used only for Kendall fallback."""
    if query_win.ndim == 1:
        return query_win, cand_win
    mean = query_win.mean(axis=0)
    std  = np.where(query_win.std(axis=0) == 0, 1.0, query_win.std(axis=0))
    return ((query_win - mean) / std).flatten(), ((cand_win - mean) / std).flatten()


def find_best_match(
    series: np.ndarray,
    window_len: int,
    forecast_len: int,
    method: str,
) -> tuple[int, float]:
    """Return index + score of the best analog match to the most-recent window."""
    query      = series[-window_len:]
    search_end = len(series) - window_len * 2 - forecast_len
    all_cands  = _build_cand_matrix(series, window_len, search_end)

    if method in _VECTORIZED:
        inv_cov    = _build_inv_cov(query, all_cands) if method in _NEEDS_INV_COV else None
        scores_arr = _batch_scores(method, query, all_cands, inv_cov)
        best_idx   = int(np.argmax(scores_arr))
        return best_idx, float(scores_arr[best_idx])

    # Kendall: no vectorized path, per-pair fallback
    best_score, best_idx = -np.inf, 0
    for i, cand in enumerate(all_cands):
        q, c = _prepare_windows(query, cand)
        score = get_score(method, q, c)
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx, best_score


def find_top_k_matches(
    series: np.ndarray,
    window_len: int,
    forecast_len: int,
    method: str,
    k: int,
    min_gap: int,
) -> list[tuple[int, float]]:
    """Return the top-k non-overlapping analog windows."""
    query      = series[-window_len:]
    search_end = len(series) - window_len * 2 - forecast_len
    all_cands  = _build_cand_matrix(series, window_len, search_end)

    if method in _VECTORIZED:
        inv_cov    = _build_inv_cov(query, all_cands) if method in _NEEDS_INV_COV else None
        scores_arr = _batch_scores(method, query, all_cands, inv_cov)
        scores     = sorted(zip(scores_arr.tolist(), range(search_end)), reverse=True)
    else:
        # Kendall: per-pair fallback
        scores = []
        for i, cand in enumerate(all_cands):
            q, c = _prepare_windows(query, cand)
            scores.append((get_score(method, q, c), i))
        scores.sort(reverse=True)

    selected: list[tuple[int, float]] = []
    for score, idx in scores:
        if not any(abs(idx - s_idx) < min_gap for s_idx, _ in selected):
            selected.append((idx, float(score)))
        if len(selected) == k:
            break

    return selected
