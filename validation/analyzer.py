"""
Analyze forecast_validation_{T}.csv and produce a narrative report:
- Verdict (TRUST / WEAK / NONE)
- What's going right
- What's going wrong
- Observations
- Recommended usage per task

Returns structured findings + a Markdown rendering.
"""
import os
import re
import pandas as pd
from datetime import datetime
from validation.html_report import (
    _parse_cell, _err_pct, _dir_correct, _in_band, _direction_label, _MODELS
)


# ─── Thresholds ──────────────────────────────────────────────────────────────

MIN_N_TRUST     = 10        # samples needed for TRUST verdict
MIN_N_WEAK      = 5         # samples needed for any verdict at all
TRUST_SCORE     = 60        # conservative score
TRUST_DIR       = 60        # directional accuracy %
WEAK_SCORE      = 45
LOW_DIR_FLAG    = 50        # below this = "worse than coin flip"
NARROW_CONE     = 40        # in-band % below this = cone too narrow
TIGHT_THRESHOLD = 3.0       # error % considered "tight"


# ─── Per-model stat extraction ───────────────────────────────────────────────

def _model_stats(df: pd.DataFrame) -> list:
    rows = []
    for m in _MODELS:
        if m not in df.columns:
            continue
        cells = df[m].apply(_parse_cell)
        errs   = cells.apply(_err_pct).dropna()
        dirs   = cells.apply(_dir_correct).dropna()
        bands  = cells.apply(_in_band).dropna()
        labels = cells.apply(_direction_label).replace("", None).dropna()
        n = len(errs)
        if n == 0:
            continue
        avg_err   = float(errs.mean())
        tight_pct = float((errs <= TIGHT_THRESHOLD).mean() * 100)
        dir_pct   = float(dirs.mean() * 100) if len(dirs) else None
        band_pct  = float(bands.mean() * 100) if len(bands) else None
        score = (
            0.40 * (dir_pct or 0)
          + 0.30 * tight_pct
          + 0.20 * (band_pct or 0)
          + 0.10 * max(0, 100 - avg_err)
        )
        # Direction bias (what % of predictions are UP)
        up_pct = (labels == "UP").mean() * 100 if len(labels) else None

        rows.append({
            "model":     m,
            "n":         n,
            "avg_err":   round(avg_err, 2),
            "tight":     round(tight_pct, 1),
            "dir":       round(dir_pct, 1) if dir_pct is not None else None,
            "in_band":   round(band_pct, 1) if band_pct is not None else None,
            "score":     round(score, 1),
            "up_pct":    round(up_pct, 1) if up_pct is not None else None,
            "_dirs":     dirs,           # raw series for recency check
            "_errs":     errs,
        })
    return rows


def _rank1_counts(df: pd.DataFrame) -> dict:
    """Count how many times each model finished #1 in the per-row leaderboard column."""
    out = {}
    if "leaderboard" not in df.columns:
        return out
    for lb in df["leaderboard"].dropna():
        lines = [l.strip() for l in str(lb).split("\n") if l.strip()]
        if not lines:
            continue
        first = lines[0]
        parts = first.split(". ", 1)
        if len(parts) != 2:
            continue
        name = parts[1].split(" (")[0].strip()
        out[name] = out.get(name, 0) + 1
    return out


def _parse_close_from_actual(s) -> float | None:
    """Extract close from actual cell. Cell format: 'C: $70.92\\nO: ...' or plain number."""
    if pd.isna(s):
        return None
    s = str(s)
    m = re.search(r'C:\s*\$?([\d.]+)', s)
    if m:
        return float(m.group(1))
    try:
        return float(s.replace("$", "").strip())
    except Exception:
        return None


def _parse_price(s) -> float | None:
    """Strip $ from prior_close-style values."""
    if pd.isna(s):
        return None
    try:
        return float(str(s).replace("$", "").strip())
    except Exception:
        return None


def _actual_up_rate(df: pd.DataFrame) -> float | None:
    """% of validation days where actual closed above prior_close."""
    if "actual" not in df.columns or "prior_close" not in df.columns:
        return None
    actual = df["actual"].apply(_parse_close_from_actual)
    prior  = df["prior_close"].apply(_parse_price)
    valid  = pd.DataFrame({"a": actual, "p": prior}).dropna()
    if len(valid) == 0:
        return None
    return round((valid["a"] > valid["p"]).mean() * 100, 1)


def _recent_vs_overall(model_row: dict, recent_n: int = 5) -> dict | None:
    """Compare last `recent_n` direction outcomes vs overall."""
    dirs = model_row["_dirs"]
    if len(dirs) < max(recent_n + 3, 8):
        return None
    recent = dirs.tail(recent_n)
    older  = dirs.iloc[:-recent_n]
    if len(recent) == 0 or len(older) == 0:
        return None
    recent_pct = float(recent.mean() * 100)
    older_pct  = float(older.mean() * 100)
    return {"recent": round(recent_pct, 1), "older": round(older_pct, 1),
            "delta": round(recent_pct - older_pct, 1)}


def _recent_verdict(stats: list, recent_n: int = 5) -> dict | None:
    """
    Compute a separate verdict using ONLY the last `recent_n` validation rows.
    Returns dict with verdict, best_model, best_recent_dir, n_recent.
    """
    eligible = [s for s in stats if len(s["_dirs"]) >= recent_n]
    if not eligible:
        return None

    recent_rows = []
    for s in eligible:
        rdirs = s["_dirs"].tail(recent_n)
        rerrs = s["_errs"].tail(recent_n)
        if len(rdirs) == 0 or len(rerrs) == 0:
            continue
        recent_dir   = float(rdirs.mean() * 100)
        recent_err   = float(rerrs.mean())
        recent_tight = float((rerrs <= TIGHT_THRESHOLD).mean() * 100)
        score = 0.40 * recent_dir + 0.30 * recent_tight + 0.10 * max(0, 100 - recent_err)
        # No recent in-band; weight the rest accordingly (still 80% of original weights)
        recent_rows.append({
            "model": s["model"], "n": len(rdirs),
            "dir": round(recent_dir, 1), "err": round(recent_err, 2),
            "tight": round(recent_tight, 1), "score": round(score, 1),
        })

    if not recent_rows:
        return None
    recent_rows.sort(key=lambda r: -r["score"])
    best = recent_rows[0]

    # Recent verdict thresholds (relaxed since N is small)
    if best["dir"] >= 70 and best["score"] >= 50:
        verdict = "TRUST"
    elif best["dir"] >= 50:
        verdict = "WEAK SIGNAL"
    else:
        verdict = "NONE"

    return {
        "verdict":      verdict,
        "best_model":   best["model"],
        "best_dir":     best["dir"],
        "best_score":   best["score"],
        "n_recent":     best["n"],
        "all_below_50": all(r["dir"] < 50 for r in recent_rows),
    }


# Action lines per verdict — concrete instruction the user can act on
VERDICT_ACTIONS = {
    "TRUST":       "Act on direction call from recommended model. Size by confidence.",
    "WEAK SIGNAL": "Skip directional trades. Use cone bands for support/resistance only.",
    "NONE":        "Stay flat or wait for external signal (news/macro/peer move). Models add no edge here.",
}


def _consensus_agreement(df: pd.DataFrame, models: list) -> float | None:
    """% of rows where all listed models predicted the same direction."""
    if not models:
        return None
    dir_cols = []
    for m in models:
        if m not in df.columns:
            continue
        dir_cols.append(df[m].apply(_parse_cell).apply(_direction_label))
    if len(dir_cols) < 2:
        return None
    combined = pd.concat(dir_cols, axis=1).replace("", pd.NA).dropna()
    if len(combined) == 0:
        return None
    agree = combined.apply(lambda r: r.nunique() == 1, axis=1)
    return round(agree.mean() * 100, 1)


# ─── Analysis engine ─────────────────────────────────────────────────────────

def analyze(ticker: str) -> dict:
    csv_path = f"output/{ticker}/forecast_validation_{ticker}.csv"
    if not os.path.exists(csv_path):
        return {"ticker": ticker, "error": f"No validation file: {csv_path}"}

    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    if len(df) == 0:
        return {"ticker": ticker, "error": "Validation file is empty."}

    stats = _model_stats(df)
    if not stats:
        return {"ticker": ticker, "error": "No usable model data."}

    rank1 = _rank1_counts(df)
    actual_up_rate = _actual_up_rate(df)
    n_dates = len(df)
    date_min = str(df["date"].min())[:10] if "date" in df.columns else "?"
    date_max = str(df["date"].max())[:10] if "date" in df.columns else "?"

    # Sort by conservative score, by avg_err, by direction, by rank1 finishes
    # All ranking restricted to models with n >= MIN_N_WEAK (avoid recommending small-sample winners)
    eligible   = [s for s in stats if s["n"] >= MIN_N_WEAK]
    by_score   = sorted(eligible, key=lambda r: -r["score"])
    by_err     = sorted(eligible, key=lambda r: r["avg_err"])
    by_dir     = sorted([s for s in eligible if s["dir"] is not None],
                         key=lambda r: -r["dir"])

    best_score = by_score[0] if by_score else None
    best_err   = by_err[0]   if by_err   else None
    best_dir   = by_dir[0]   if by_dir   else None

    # === VERDICT ===
    verdict = "NONE"
    recommended = None
    reasons = []
    caveats = []

    # Pick best candidate that has direction data
    candidates = [s for s in by_score if s["dir"] is not None]
    if not candidates:
        candidates = list(by_score)
    candidate = candidates[0] if candidates else None

    if candidate:
        if (candidate["score"] >= TRUST_SCORE and
            candidate["n"] >= MIN_N_TRUST and
            (candidate["dir"] is None or candidate["dir"] >= TRUST_DIR)):
            verdict = "TRUST"
            recommended = candidate["model"]
            reasons.append(f"conservative score {candidate['score']:.1f}")
            if candidate["dir"] is not None:
                reasons.append(f"{candidate['dir']:.0f}% directional accuracy ({int(candidate['dir']*candidate['n']/100)}/{candidate['n']})")
            reasons.append(f"avg error {candidate['avg_err']:.2f}%")
            if candidate["in_band"] is not None and candidate["in_band"] >= 80:
                reasons.append(f"{candidate['in_band']:.0f}% in-band rate")
            if candidate["n"] < 20:
                caveats.append(f"only {candidate['n']} validation days — re-confirm at n≥25 in {25-candidate['n']} more weeks")
        elif (candidate["score"] >= WEAK_SCORE or
              (candidate["dir"] is not None and candidate["dir"] >= LOW_DIR_FLAG)):
            verdict = "WEAK SIGNAL"
            recommended = candidate["model"]
            reasons.append(f"score {candidate['score']:.1f} (below trust threshold of {TRUST_SCORE})")
            if candidate["n"] < MIN_N_TRUST:
                caveats.append(f"sample size n={candidate['n']} below trust floor of {MIN_N_TRUST}")
            if candidate["dir"] is not None and candidate["dir"] < TRUST_DIR:
                caveats.append(f"directional accuracy {candidate['dir']:.0f}% below {TRUST_DIR}% threshold")
        else:
            verdict = "NONE"
            recommended = None
            reasons.append(f"top model ({candidate['model']}) has score {candidate['score']:.1f}, below {WEAK_SCORE}")

    # All-models-bad override
    eligible_dirs = [s["dir"] for s in eligible if s["dir"] is not None]
    if eligible_dirs and max(eligible_dirs) < LOW_DIR_FLAG:
        verdict = "NONE"
        recommended = None
        reasons = [f"every model with sufficient data has direction accuracy < {LOW_DIR_FLAG}% (worse than coin flip)"]

    # === RECENT VERDICT (last 5 days) — catches active regime decay ===
    recent_v = _recent_verdict(stats, recent_n=5)

    # === BIAS-TRIGGERED DOWNGRADE ===
    # If the recommended model has strong directional bias (model's UP-rate
    # diverges from actual UP-rate by >=30 pts) AND recent dir <50%, downgrade.
    bias_downgraded = False
    if candidate and actual_up_rate is not None and recent_v:
        cand_stat = next((s for s in stats if s["model"] == candidate["model"]), None)
        if cand_stat and cand_stat["up_pct"] is not None:
            bias_gap = abs(cand_stat["up_pct"] - actual_up_rate)
            # Recent dir for the candidate model (its own last-5 history)
            cand_recent = float(cand_stat["_dirs"].tail(5).mean() * 100) if len(cand_stat["_dirs"]) >= 5 else None
            if bias_gap >= 30 and (cand_recent is not None and cand_recent < 50):
                tier = {"TRUST": "WEAK SIGNAL", "WEAK SIGNAL": "NONE", "NONE": "NONE"}
                old_verdict = verdict
                verdict = tier.get(verdict, verdict)
                if verdict != old_verdict:
                    bias_downgraded = True
                    caveats.append(
                        f"VERDICT DOWNGRADED ({old_verdict} → {verdict}): "
                        f"{candidate['model']} bias {bias_gap:.0f}pts off actual + recent dir {cand_recent:.0f}%"
                    )
                    if verdict == "NONE":
                        recommended = None

    # === WHAT'S RIGHT ===
    right = []
    # Top conservative models (all already filtered to n >= MIN_N_WEAK)
    top3 = by_score[:3]
    if top3 and top3[0]["score"] >= 50:
        names = ", ".join(s["model"] for s in top3)
        right.append(f"Top 3 by conservative score: {names}")
    # Strong direction
    strong_dir = [s for s in stats if s["dir"] is not None and s["dir"] >= 70 and s["n"] >= MIN_N_WEAK]
    if strong_dir:
        for s in strong_dir[:3]:
            right.append(f"{s['model']} has {s['dir']:.0f}% direction accuracy ({s['n']} days)")
    # Strong cone reliability
    strong_band = [s for s in stats if s["in_band"] is not None and s["in_band"] >= 80 and s["n"] >= MIN_N_WEAK]
    if strong_band:
        names = ", ".join(s["model"] for s in strong_band[:5])
        right.append(f"Cone reliable (≥80% in-band): {names}")
    # Tight predictions
    tight = [s for s in stats if s["tight"] >= 40 and s["n"] >= MIN_N_WEAK]
    if tight:
        for s in tight[:3]:
            right.append(f"{s['model']} keeps {s['tight']:.0f}% of predictions within 3% error")
    # Consensus
    top_models = [s["model"] for s in by_score[:3] if s["model"] in df.columns]
    consensus = _consensus_agreement(df, top_models)
    if consensus is not None and consensus >= 70:
        right.append(f"Top 3 models agree on direction {consensus:.0f}% of the time")
    # Rank-1 dominator
    if rank1:
        leader, count = max(rank1.items(), key=lambda x: x[1])
        if count >= max(3, n_dates * 0.4):
            right.append(f"{leader} finished #1 (lowest error) on {count}/{n_dates} days")

    # === WHAT'S WRONG ===
    wrong = []
    # All ML models below avg kNN
    knn_methods = ["spearman", "pearson", "cosine", "euclidean",
                   "kendall", "manhattan"]
    ml_methods = ["xgboost", "lightgbm", "randomforest"]
    knn_dirs = [s["dir"] for s in stats if s["model"] in knn_methods and s["dir"] is not None and s["n"] >= MIN_N_WEAK]
    ml_dirs  = [s["dir"] for s in stats if s["model"] in ml_methods and s["dir"] is not None and s["n"] >= MIN_N_WEAK]
    if knn_dirs and ml_dirs:
        avg_knn = sum(knn_dirs) / len(knn_dirs)
        avg_ml  = sum(ml_dirs) / len(ml_dirs)
        if avg_ml + 10 < avg_knn:
            wrong.append(f"ML models ({avg_ml:.0f}% avg dir) significantly underperform kNN methods ({avg_knn:.0f}% avg dir)")

    # Models with bad direction
    bad_dir = [s for s in stats if s["dir"] is not None and s["dir"] <= 30 and s["n"] >= MIN_N_WEAK]
    for s in bad_dir[:3]:
        wrong.append(f"{s['model']} direction accuracy {s['dir']:.0f}% — worse than fading the signal")

    # Cone too narrow
    narrow = [s for s in stats if s["in_band"] is not None and s["in_band"] < NARROW_CONE and s["n"] >= MIN_N_WEAK]
    for s in narrow[:2]:
        wrong.append(f"{s['model']} cone too narrow — only {s['in_band']:.0f}% in-band rate")

    # Direction bias
    if actual_up_rate is not None:
        for s in stats:
            if s["up_pct"] is None or s["n"] < MIN_N_WEAK:
                continue
            if abs(s["up_pct"] - actual_up_rate) >= 30:
                wrong.append(
                    f"{s['model']} predicted UP {s['up_pct']:.0f}% of the time but actual UP rate was {actual_up_rate:.0f}% (directional bias)"
                )
                break

    # Insufficient samples
    insufficient = [s for s in stats if s["n"] < MIN_N_WEAK]
    if insufficient:
        names = ", ".join(s["model"] for s in insufficient)
        wrong.append(f"Insufficient data (n<{MIN_N_WEAK}) for: {names}")

    # === OBSERVATIONS ===
    observations = []
    # Tight vs reliable mismatch
    if best_err and best_dir and best_err["model"] != best_dir["model"]:
        observations.append(
            f"Tightest predictions: {best_err['model']} ({best_err['avg_err']:.2f}% avg err). "
            f"Most reliable direction: {best_dir['model']} ({best_dir['dir']:.0f}%). Different models — pick by use case."
        )
    # Recent vs overall (model decay)
    for s in stats[:6]:
        rec = _recent_vs_overall(s)
        if rec and rec["delta"] <= -25:
            observations.append(f"{s['model']} decay detected: recent dir {rec['recent']:.0f}% vs overall {rec['older']:.0f}% (Δ {rec['delta']:+.0f})")
            break

    # === RECOMMENDED USAGE ===
    usage = {}
    if best_dir and best_dir["dir"] is not None and best_dir["dir"] >= TRUST_DIR:
        usage["Direction call"] = f"{best_dir['model']} ({best_dir['dir']:.0f}% correct, {best_dir['n']} days)"
    elif best_dir:
        usage["Direction call"] = f"{best_dir['model']} (only {best_dir['dir']:.0f}% — use with low confidence)"
    if best_err:
        usage["Tight price target"] = f"{best_err['model']} ({best_err['avg_err']:.2f}% avg err)"
    band_winner = max([s for s in stats if s["in_band"] is not None and s["n"] >= MIN_N_WEAK],
                      key=lambda r: r["in_band"], default=None)
    if band_winner:
        usage["Risk/cone bands"] = f"{band_winner['model']} ({band_winner['in_band']:.0f}% in-band)"
    avoid = [s["model"] for s in stats if s["dir"] is not None and s["dir"] <= 30 and s["n"] >= MIN_N_WEAK]
    if avoid:
        usage["Avoid"] = ", ".join(avoid)

    return {
        "ticker":       ticker,
        "n_dates":      n_dates,
        "date_min":     date_min,
        "date_max":     date_max,
        "actual_up_rate": actual_up_rate,
        "verdict":      verdict,
        "recommended":  recommended,
        "reasons":      reasons,
        "caveats":      caveats,
        "action":       VERDICT_ACTIONS.get(verdict, ""),
        "recent_verdict": recent_v,
        "bias_downgraded": bias_downgraded,
        "right":        right,
        "wrong":        wrong,
        "observations": observations,
        "usage":        usage,
        "stats":        [{k: v for k, v in s.items() if not k.startswith("_")} for s in stats],
        "top3":         [s["model"] for s in by_score[:3]],
    }


# ─── Markdown rendering ──────────────────────────────────────────────────────

def to_markdown(result: dict) -> str:
    if "error" in result:
        return f"# {result['ticker']} — Analysis Error\n\n{result['error']}\n"

    t = result["ticker"]
    lines = []
    lines.append(f"# {t} — Validation Analysis")
    lines.append("")
    lines.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_  ")
    lines.append(f"_{result['n_dates']} validation days, {result['date_min']} → {result['date_max']}_")
    if result["actual_up_rate"] is not None:
        lines.append(f"_Actual UP rate: {result['actual_up_rate']:.0f}%_")
    lines.append("")

    # Verdict
    badge = {"TRUST": "✅", "WEAK SIGNAL": "⚠️", "NONE": "🛑"}.get(result["verdict"], "")
    lines.append(f"## {badge} OVERALL VERDICT: {result['verdict']}")
    lines.append("")
    if result["recommended"]:
        lines.append(f"**Recommended model: `{result['recommended']}`**")
        lines.append("")
        if result["reasons"]:
            lines.append("Why:")
            for r in result["reasons"]:
                lines.append(f"- {r}")
            lines.append("")
    else:
        lines.append("**No model meets the trust threshold.** Don't rely on directional signals from this ticker right now.")
        lines.append("")
        for r in result["reasons"]:
            lines.append(f"- {r}")
        lines.append("")

    # Action line
    if result.get("action"):
        lines.append(f"**Action:** {result['action']}")
        lines.append("")

    # Recent verdict (last 5 days) — surfaces active regime decay
    rv = result.get("recent_verdict")
    if rv:
        rbadge = {"TRUST": "✅", "WEAK SIGNAL": "⚠️", "NONE": "🛑"}.get(rv["verdict"], "")
        lines.append(f"## {rbadge} RECENT VERDICT (last {rv['n_recent']} days): {rv['verdict']}")
        lines.append("")
        lines.append(f"Best recent model: **`{rv['best_model']}`** ({rv['best_dir']:.0f}% direction, score {rv['best_score']:.1f})")
        if rv["all_below_50"]:
            lines.append("")
            lines.append("⚠️ **All models below 50% direction in the last 5 days — active regime decay detected.**")
        if rv["verdict"] != result["verdict"]:
            lines.append("")
            lines.append(f"⚠️ **Recent verdict differs from overall** — the model may be decaying. "
                        f"Lean on the recent verdict for short-horizon trades.")
        lines.append("")

    if result["caveats"]:
        lines.append("Caveats:")
        for c in result["caveats"]:
            lines.append(f"- {c}")
        lines.append("")

    # What's right
    if result["right"]:
        lines.append("## ✓ What's going right")
        lines.append("")
        for r in result["right"]:
            lines.append(f"- {r}")
        lines.append("")

    # What's wrong
    if result["wrong"]:
        lines.append("## ✗ What's going wrong")
        lines.append("")
        for w in result["wrong"]:
            lines.append(f"- {w}")
        lines.append("")

    # Observations
    if result["observations"]:
        lines.append("## • Observations")
        lines.append("")
        for o in result["observations"]:
            lines.append(f"- {o}")
        lines.append("")

    # Recommended usage
    if result["usage"]:
        lines.append("## Recommended usage")
        lines.append("")
        for k, v in result["usage"].items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    # Stats table
    lines.append("## Per-model stats")
    lines.append("")
    lines.append("| Model | N | Avg Err % | Tight ≤3% | Dir Acc | In-Band | Score |")
    lines.append("|-------|---|-----------|-----------|---------|---------|-------|")
    for s in sorted(result["stats"], key=lambda r: -r["score"]):
        dir_s  = f"{s['dir']:.0f}%"     if s['dir']     is not None else "—"
        band_s = f"{s['in_band']:.0f}%" if s['in_band'] is not None else "—"
        lines.append(f"| {s['model']} | {s['n']} | {s['avg_err']:.2f}% | {s['tight']:.0f}% | {dir_s} | {band_s} | {s['score']:.1f} |")
    lines.append("")

    return "\n".join(lines)


def write_report(ticker: str) -> tuple[str, dict]:
    """Run analysis, write markdown to disk, return (path, result)."""
    result = analyze(ticker)
    md = to_markdown(result)
    out_dir = f"output/{ticker}/reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/analysis_{ticker}.md"
    with open(out_path, "w") as f:
        f.write(md)
    return out_path, result
