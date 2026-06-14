import numpy as np
from scipy.stats import rankdata, kendalltau


def spearman_score(query: np.ndarray, candidate: np.ndarray) -> float:
    n           = len(query)
    denominator = n * (n ** 2 - 1)
    if denominator == 0:
        return 0.0
    query_ranks = rankdata(query)
    cand_ranks  = rankdata(candidate)
    d2          = (query_ranks - cand_ranks) ** 2
    return 1 - (6 * d2.sum()) / denominator


def pearson_score(query: np.ndarray, candidate: np.ndarray) -> float:
    std_q = np.std(query)
    std_c = np.std(candidate)
    if std_q == 0 or std_c == 0:
        return 0.0
    cov = np.cov(query, candidate)[0][1]
    return cov / (std_q * std_c)


def cosine_score(query: np.ndarray, candidate: np.ndarray) -> float:
    nA = np.sqrt(np.sum(query ** 2))
    nB = np.sqrt(np.sum(candidate ** 2))
    if nA == 0 or nB == 0:
        return 0.0
    return np.dot(query, candidate) / (nA * nB)


def euclidean_score(query: np.ndarray, candidate: np.ndarray) -> float:
    dist = np.sqrt(np.sum((query - candidate) ** 2))
    return 1 / (1 + dist)


def mse_score(query: np.ndarray, candidate: np.ndarray) -> float:
    mse = np.mean((query - candidate) ** 2)
    return 1 / (1 + mse)


def kendall_score(query: np.ndarray, candidate: np.ndarray) -> float:
    tau, _ = kendalltau(query, candidate)
    return float(tau) if not np.isnan(tau) else 0.0


def manhattan_score(query: np.ndarray, candidate: np.ndarray) -> float:
    dist = np.sum(np.abs(query - candidate))
    return 1.0 / (1.0 + dist)



def mahalanobis_score(query: np.ndarray, candidate: np.ndarray,
                      inv_cov: np.ndarray = None) -> float:
    """Mahalanobis distance using pre-computed inverse covariance matrix.
    Falls back to Euclidean if inv_cov is not provided."""
    diff = query - candidate
    if inv_cov is not None:
        dist = float(np.sqrt(np.maximum(diff @ inv_cov @ diff, 0.0)))
    else:
        dist = float(np.sqrt(np.dot(diff, diff)))
    return 1.0 / (1.0 + dist)


_METRIC_MAP = {
    "spearman":    spearman_score,
    "pearson":     pearson_score,
    "cosine":      cosine_score,
    "euclidean":   euclidean_score,
    "kendall":     kendall_score,
    "manhattan":   manhattan_score,
    # mse + mahalanobis removed 2026-05-22 (mse = squared euclidean; mahalanobis coin-flip)
}


def get_score(method: str, query: np.ndarray, candidate: np.ndarray) -> float:
    fn = _METRIC_MAP.get(method.lower())
    if fn is None:
        raise ValueError(f"Unknown similarity method '{method}'. Choose from: {list(_METRIC_MAP)}")
    return fn(query, candidate)
