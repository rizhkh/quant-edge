# MU — Validation Analysis

_Generated 2026-06-13 13:59:11_  
_40 validation days, 2026-04-17 → 2026-06-12_
_Actual UP rate: 62%_

## ⚠️ OVERALL VERDICT: WEAK SIGNAL

**Recommended model: `kendall`**

Why:
- score 57.4 (below trust threshold of 60)

**Action:** Skip directional trades. Use cone bands for support/resistance only.

## ✅ RECENT VERDICT (last 5 days): TRUST

Best recent model: **`kendall`** (80% direction, score 59.7)

⚠️ **Recent verdict differs from overall** — the model may be decaying. Lean on the recent verdict for short-horizon trades.

## ✓ What's going right

- Top 3 by conservative score: kendall, euclidean, manhattan
- euclidean has 72% direction accuracy (40 days)

## ✗ What's going wrong

- ML models (47% avg dir) significantly underperform kNN methods (58% avg dir)
- cosine cone too narrow — only 40% in-band rate
- xgboost cone too narrow — only 40% in-band rate
- analog predicted UP 25% of the time but actual UP rate was 62% (directional bias)

## • Observations

- Tightest predictions: knn2 (7.27% avg err). Most reliable direction: euclidean (72%). Different models — pick by use case.

## Recommended usage

- **Direction call:** euclidean (72% correct, 40 days)
- **Tight price target:** knn2 (7.27% avg err)
- **Risk/cone bands:** kendall (55% in-band)

## Per-model stats

| Model | N | Avg Err % | Tight ≤3% | Dir Acc | In-Band | Score |
|-------|---|-----------|-----------|---------|---------|-------|
| kendall | 40 | 7.86% | 35% | 67% | 55% | 57.4 |
| euclidean | 40 | 8.20% | 28% | 72% | 47% | 55.8 |
| manhattan | 34 | 9.02% | 21% | 65% | 47% | 50.6 |
| knn2 | 33 | 7.27% | 27% | 64% | 30% | 49.0 |
| xgboost | 38 | 7.76% | 29% | 53% | 40% | 46.9 |
| spearman | 40 | 9.14% | 30% | 44% | 45% | 44.8 |
| pearson | 40 | 8.82% | 22% | 50% | 45% | 44.8 |
| cosine | 40 | 9.12% | 20% | 50% | 40% | 43.0 |
| lightgbm | 36 | 8.13% | 19% | 47% | 25% | 38.9 |
| randomforest | 34 | 8.61% | 18% | 41% | 35% | 38.0 |
| analog | 40 | 9.78% | 18% | 31% | — | 26.5 |
