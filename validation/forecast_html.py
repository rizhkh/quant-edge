"""
Generate a self-contained HTML page for forecast_results_{T}.csv.
CSV stays as-is — this is just a friendlier reading layer.

Output: output/{T}/reports/forecast_{T}.html
"""
import os
import html
import json
from datetime import datetime
import pandas as pd


# (display_name, column_prefix, category)
MODELS = [
    ("analog",       "analog",     "analog"),
    ("knn",          "knn",        "knn"),
    ("spearman",     "spearman",   "knn"),
    ("pearson",      "pearson",    "knn"),
    ("cosine",       "cosine",     "knn"),
    ("euclidean",    "euclidean",  "knn"),
    ("kendall",      "kendall",    "knn"),
    ("manhattan",    "manhattan",  "knn"),
    ("xgboost",      "xgb",        "ml"),
    ("lightgbm",     "lgb",        "ml"),
    ("randomforest", "rf",         "ml"),
    ("knn2",         "knn2",       "knn2"),
]


def _f(v):
    """Try to coerce to float; return None on failure."""
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def _direction_for(row, prefix: str) -> str:
    val = row.get(f"{prefix}_direction")
    if pd.isna(val):
        return ""
    s = str(val).upper()
    if "UP" in s: return "UP"
    if "DOWN" in s: return "DOWN"
    return ""


def _color_dir(d: str) -> str:
    return "#22c55e" if d == "UP" else ("#ef4444" if d == "DOWN" else "#6b7280")


# Universe-wide high-conviction kNN methods — these 4 averaged 56-63% Day 1
# directional accuracy across n=146-168 forecasts per model. Used to compute
# the Day 1 trading-grade signal panel.
HIGH_CONVICTION_MODELS = ["pearson", "cosine", "manhattan", "spearman"]


def _high_conviction_signal(row) -> dict:
    """
    Day 1 high-conviction signal: fires when 3+ of {pearson, cosine, manhattan,
    spearman} agree on direction. Below 3-agreement → not actionable.
    """
    votes = {"UP": [], "DOWN": []}
    for prefix in HIGH_CONVICTION_MODELS:
        d = _direction_for(row, prefix)
        if d in ("UP", "DOWN"):
            votes[d].append(prefix)
    up_n   = len(votes["UP"])
    down_n = len(votes["DOWN"])
    total  = up_n + down_n
    if total == 0:
        return {"signal": "NO DATA", "agree_n": 0, "of": len(HIGH_CONVICTION_MODELS),
                "agreeing": [], "actionable": False}
    if up_n >= 3:
        return {"signal": "UP", "agree_n": up_n, "of": len(HIGH_CONVICTION_MODELS),
                "agreeing": votes["UP"], "actionable": True}
    if down_n >= 3:
        return {"signal": "DOWN", "agree_n": down_n, "of": len(HIGH_CONVICTION_MODELS),
                "agreeing": votes["DOWN"], "actionable": True}
    # Split — show what agreed but flag as not actionable
    leader = "UP" if up_n >= down_n else "DOWN"
    return {"signal": "SPLIT", "agree_n": max(up_n, down_n), "of": len(HIGH_CONVICTION_MODELS),
            "agreeing": votes[leader], "actionable": False}


def _direction_consensus(row, models: list) -> dict:
    up, down = 0, 0
    for _, prefix, _ in models:
        d = _direction_for(row, prefix)
        if d == "UP": up += 1
        elif d == "DOWN": down += 1
    total = up + down
    return {
        "up": up, "down": down, "total": total,
        "consensus": "UP" if up > down else ("DOWN" if down > up else "MIXED"),
        "agreement_pct": round(max(up, down) / total * 100, 1) if total else 0,
    }


def _trust_weighted_consensus(row, models: list, trust_map: dict) -> dict:
    """
    Weight each model's UP/DOWN vote by its validation dir accuracy.
    Models with dir <50% get vote weight = 0 (filtered out, not just downweighted).
    Models with no validation data fall back to weight 0.5 (small voice).
    Returns consensus + agreement % computed on weighted votes.
    """
    up_w, down_w, voters = 0.0, 0.0, []
    for name, prefix, _ in models:
        d = _direction_for(row, prefix)
        if d not in ("UP", "DOWN"): continue
        dir_acc = trust_map.get(name)  # may be None
        if dir_acc is None:
            weight = 0.5         # small voice when unknown
        elif dir_acc < 50:
            weight = 0.0         # filter out — coin-flip or worse
        else:
            weight = dir_acc / 100.0
        if weight <= 0: continue
        if d == "UP":   up_w += weight
        else:           down_w += weight
        voters.append({"model": name, "dir": d, "weight": round(weight, 2)})
    total_w = up_w + down_w
    if total_w == 0:
        return {"consensus": "MIXED", "up_weight": 0, "down_weight": 0,
                "agreement_pct": 0, "voters": voters, "n_voters": 0}
    return {
        "consensus": "UP" if up_w > down_w else ("DOWN" if down_w > up_w else "MIXED"),
        "up_weight":   round(up_w, 2),
        "down_weight": round(down_w, 2),
        "agreement_pct": round(max(up_w, down_w) / total_w * 100, 1),
        "voters":      voters,
        "n_voters":    len(voters),
    }


def _load_analyzer_data(ticker: str) -> dict:
    """Run analyzer.analyze() to get verdict + per-model dir accuracy. Empty dict if fails."""
    try:
        from validation.analyzer import analyze
        result = analyze(ticker)
        if "error" in result:
            return {}
        return result
    except Exception:
        return {}


def _trust_map_from_analyzer(analyzer_result: dict) -> dict:
    """Map model name -> dir accuracy %, from analyzer stats. Empty {} if unavailable."""
    if not analyzer_result or "stats" not in analyzer_result: return {}
    return {s["model"]: s["dir"] for s in analyzer_result["stats"] if s.get("dir") is not None}


def _top_models_from_analyzer(analyzer_result: dict, n: int = 6) -> list:
    """Return top-N model names by conservative score (analyzer's own ranking)."""
    if not analyzer_result or "stats" not in analyzer_result: return []
    sorted_stats = sorted(
        [s for s in analyzer_result["stats"] if s.get("score") is not None],
        key=lambda s: -s["score"],
    )
    return [s["model"] for s in sorted_stats[:n]]


def _broken_models_from_analyzer(analyzer_result: dict, n: int = 3) -> list:
    """
    Bottom-N model names by conservative score — to visually demote in the table.
    These are 'broken' for this ticker per the current validation data.
    """
    if not analyzer_result or "stats" not in analyzer_result: return []
    sorted_stats = sorted(
        [s for s in analyzer_result["stats"] if s.get("score") is not None],
        key=lambda s: s["score"],   # ascending — worst first
    )
    return [s["model"] for s in sorted_stats[:n]]


def _milestone_card(row, day_label: str, models_present: list, current_close: float | None) -> str:
    """One card summarizing model agreement + price range for one milestone day."""
    if row is None:
        return f"""<div class="milestone-card empty">
          <div class="milestone-day">{html.escape(day_label)}</div>
          <div class="empty-msg">no data</div>
        </div>"""

    medians = []
    lows = []
    highs = []
    for _, prefix, _ in models_present:
        col_med = "analog_price" if prefix == "analog" else f"{prefix}_median"
        m = _f(row.get(col_med))
        l = _f(row.get(f"{prefix}_low"))
        h = _f(row.get(f"{prefix}_high"))
        if m is not None: medians.append(m)
        if l is not None: lows.append(l)
        if h is not None: highs.append(h)

    if not medians:
        return f"""<div class="milestone-card empty">
          <div class="milestone-day">{html.escape(day_label)}</div>
          <div class="empty-msg">no medians</div>
        </div>"""

    cons = _direction_consensus(row, models_present)
    med_lo = min(medians)
    med_hi = max(medians)
    med_mid = sum(medians) / len(medians)
    band_lo = min(lows) if lows else med_lo
    band_hi = max(highs) if highs else med_hi
    date = str(row.get("date", ""))[:10]
    cons_color = _color_dir(cons["consensus"])

    move_str = ""
    if current_close and current_close > 0:
        move = (med_mid - current_close) / current_close * 100
        move_color = "#22c55e" if move >= 0 else "#ef4444"
        move_str = f"<div class='milestone-move' style='color:{move_color}'>{'+' if move >= 0 else ''}{move:.1f}% from now</div>"

    return f"""<div class="milestone-card">
      <div class="milestone-header">
        <div class="milestone-day">{html.escape(day_label)}</div>
        <div class="milestone-date">{html.escape(date)}</div>
      </div>
      <div class="milestone-consensus" style="color:{cons_color}">
        {html.escape(cons['consensus'])}
        <span class="agreement">{cons['up']}↑ {cons['down']}↓ · {cons['agreement_pct']}% agree</span>
      </div>
      <div class="milestone-price">${med_mid:.2f}</div>
      <div class="milestone-range">range ${med_lo:.2f} – ${med_hi:.2f}</div>
      <div class="milestone-band">cone ${band_lo:.2f} – ${band_hi:.2f}</div>
      {move_str}
    </div>"""


def _svg_chart(df: pd.DataFrame, models_present: list, current_close: float | None) -> str:
    """Simple inline SVG chart of model medians over time."""
    if len(df) == 0:
        return ""
    width, height = 1200, 320
    pad_l, pad_r, pad_t, pad_b = 50, 130, 20, 30

    # Collect medians per model
    series = []
    palette = ["#38bdf8", "#22c55e", "#eab308", "#a78bfa", "#fb923c",
               "#f472b6", "#34d399", "#facc15", "#94a3b8", "#fb7185",
               "#22d3ee", "#a3e635", "#c084fc", "#ec4899"]
    color_idx = 0
    for name, prefix, _ in models_present:
        col_med = "analog_price" if prefix == "analog" else f"{prefix}_median"
        if col_med not in df.columns:
            continue
        vals = df[col_med].apply(_f).tolist()
        if any(v is not None for v in vals):
            series.append({"name": name, "vals": vals, "color": palette[color_idx % len(palette)]})
            color_idx += 1

    if not series:
        return ""

    n = len(df)
    all_vals = [v for s in series for v in s["vals"] if v is not None]
    if current_close: all_vals.append(current_close)
    if not all_vals:
        return ""

    y_min, y_max = min(all_vals), max(all_vals)
    y_pad = (y_max - y_min) * 0.05 or 1
    y_min -= y_pad; y_max += y_pad

    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    def _x(i): return pad_l + (i / max(n - 1, 1)) * plot_w
    def _y(v): return pad_t + (1 - (v - y_min) / (y_max - y_min)) * plot_h

    # Y-axis labels
    y_labels = []
    for i in range(5):
        v = y_min + (y_max - y_min) * i / 4
        y = _y(v)
        y_labels.append(f"<text x='{pad_l - 6}' y='{y + 4}' fill='#64748b' font-size='10' text-anchor='end'>${v:.0f}</text>")
        y_labels.append(f"<line x1='{pad_l}' y1='{y}' x2='{width - pad_r}' y2='{y}' stroke='#1e293b' stroke-width='1'/>")

    # X-axis: show dates (first, ~25%, ~50%, ~75%, last)
    x_labels = []
    if "date" in df.columns:
        idx_pts = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        for i in idx_pts:
            d = str(df.iloc[i]["date"])[:10]
            x_labels.append(f"<text x='{_x(i)}' y='{height - pad_b + 16}' fill='#64748b' font-size='10' text-anchor='middle'>{d}</text>")

    # Current close horizontal reference line
    ref_line = ""
    if current_close is not None:
        y = _y(current_close)
        ref_line = (
            f"<line x1='{pad_l}' y1='{y}' x2='{width - pad_r}' y2='{y}' "
            f"stroke='#fbbf24' stroke-width='1' stroke-dasharray='4,4'/>"
            f"<text x='{width - pad_r + 6}' y='{y + 4}' fill='#fbbf24' font-size='10'>${current_close:.2f} now</text>"
        )

    # Series lines
    line_paths = []
    legend = []
    for i, s in enumerate(series):
        pts = []
        for j, v in enumerate(s["vals"]):
            if v is None: continue
            pts.append(f"{_x(j):.1f},{_y(v):.1f}")
        if pts:
            d = "M" + " L".join(pts)
            line_paths.append(f"<path d='{d}' stroke='{s['color']}' stroke-width='1.5' fill='none' opacity='0.85'/>")
            ly = pad_t + 14 + i * 18
            legend.append(
                f"<line x1='{width - pad_r + 8}' y1='{ly - 4}' x2='{width - pad_r + 22}' y2='{ly - 4}' stroke='{s['color']}' stroke-width='2'/>"
                f"<text x='{width - pad_r + 28}' y='{ly}' fill='#cbd5e1' font-size='10'>{s['name']}</text>"
            )

    return f"""<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" class="chart-svg">
      <rect width="{width}" height="{height}" fill="#0f172a" rx="8"/>
      {''.join(y_labels)}
      {''.join(x_labels)}
      {ref_line}
      {''.join(line_paths)}
      {''.join(legend)}
    </svg>"""


def _table_rows(df: pd.DataFrame, models_present: list, current_close: float | None,
                broken_set: set | None = None) -> str:
    broken_set = broken_set or set()
    rows_html = []
    for _, row in df.iterrows():
        date = str(row.get("date", ""))[:10]
        day  = row.get("day", "?")
        cons = _direction_consensus(row, models_present)
        cons_color = _color_dir(cons["consensus"])

        cells = [
            f"<td class='date-cell'><div>{html.escape(date)}</div><div class='day-num'>day {day}</div></td>",
            f"<td><span style='color:{cons_color};font-weight:600'>{cons['consensus']}</span><div class='small'>{cons['up']}↑ {cons['down']}↓ · {cons['agreement_pct']}%</div></td>",
        ]
        for name, prefix, _ in models_present:
            col_med = "analog_price" if prefix == "analog" else f"{prefix}_median"
            m = _f(row.get(col_med))
            l = _f(row.get(f"{prefix}_low"))
            h = _f(row.get(f"{prefix}_high"))
            d = _direction_for(row, prefix)
            if m is None:
                cells.append("<td class='muted'>—</td>"); continue
            d_color = _color_dir(d)
            range_str = ""
            if l is not None and h is not None:
                range_str = f"<div class='small'>${l:.2f} – ${h:.2f}</div>"
            move = None
            move_str = ""
            if current_close and current_close > 0:
                move = (m - current_close) / current_close * 100
                mv_color = "#22c55e" if move >= 0 else "#ef4444"
                move_str = f"<div class='small' style='color:{mv_color}'>{'+' if move >= 0 else ''}{move:.1f}%</div>"

            detail = {
                "model": name,
                "date": date,
                "day": int(day) if pd.notna(day) else None,
                "current_close": current_close,
                "median": m,
                "low_p90": l,
                "high_p20": h,
                "p90": _f(row.get(f"{prefix}_p90")),
                "p60": _f(row.get(f"{prefix}_p60")),
                "p20": _f(row.get(f"{prefix}_p20")),
                "pdcp90": _f(row.get(f"{prefix}_pdcp90")),
                "pdcp60": _f(row.get(f"{prefix}_pdcp60")),
                "pdcp20": _f(row.get(f"{prefix}_pdcp20")),
                "direction": d or None,
                "pct_move": move,
            }
            if prefix == "knn2":
                detail["vote_pct"] = _f(row.get("knn2_vote_pct"))
                detail["n_neighbors"] = _f(row.get("knn2_n_neighbors"))
                reg = row.get("knn2_regime")
                detail["regime"] = None if pd.isna(reg) else str(reg)
            if prefix == "analog":
                detail["pdcp"] = _f(row.get("analog_pdcp"))
            detail_attr = html.escape(json.dumps(detail), quote=True)
            broken_cls = " broken" if name in broken_set else ""
            cells.append(
                f"<td class='model-cell clickable{broken_cls}' data-direction='{d or 'NONE'}' "
                f"data-detail='{detail_attr}' title='Click for details'>"
                f"<div>${m:.2f} <span style='color:{d_color};font-size:10px;font-weight:600'>{d}</span></div>"
                f"{range_str}{move_str}</td>"
            )

        # data-direction = the consensus, used for filtering rows
        rows_html.append(
            f"<tr class='forecast-row' data-date='{date}' data-direction='{cons['consensus']}' "
            f"data-agreement='{cons['agreement_pct']}'>{''.join(cells)}</tr>"
        )
    return "\n".join(rows_html)


def _build_html(ticker: str, df: pd.DataFrame) -> str:
    if len(df) == 0:
        return f"<html><body><h1>{html.escape(ticker)}</h1><p>No forecast data.</p></body></html>"

    df = df.copy()
    # Filter to current/future forecast snapshot:
    # 1. Only keep dates >= today (drop old elapsed forecast rows with NaN data)
    # 2. For duplicate dates (same date appears in multiple forecast runs), keep the latest by last_updated
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        today = pd.Timestamp.now().normalize()
        df = df[df["date"] >= today].copy()
        if "last_updated" in df.columns:
            df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
            df = df.sort_values(["date", "last_updated"])
            df = df.drop_duplicates(subset=["date"], keep="last")
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("day").reset_index(drop=True)

    if len(df) == 0:
        return f"<html><body><h1>{html.escape(ticker)}</h1><p>No future forecast data — re-run main.py.</p></body></html>"

    # Determine current close (use first available)
    current_close = None
    if "current_close" in df.columns:
        for v in df["current_close"]:
            cc = _f(v)
            if cc is not None:
                current_close = cc; break

    last_updated = ""
    if "last_updated" in df.columns:
        last_updated = str(df.iloc[0].get("last_updated", ""))[:19]
    knn_method = ""
    if "knn_method" in df.columns:
        try: knn_method = str(df.iloc[0].get("knn_method") or "")
        except Exception: knn_method = ""

    n_days = len(df)
    date_min = str(df["date"].iloc[0]) if "date" in df.columns else "—"
    date_max = str(df["date"].iloc[-1]) if "date" in df.columns else "—"

    # Drop knn from main models since the winning method is reported separately
    models_present = [m for m in MODELS
                      if m[0] != "knn"
                      and (f"{m[1]}_median" in df.columns or m[1] == "analog" and "analog_price" in df.columns)]

    # === Load analyzer data (verdict, per-model dir accuracy, top models) ===
    analyzer_result = _load_analyzer_data(ticker)
    trust_map       = _trust_map_from_analyzer(analyzer_result)
    top_models      = _top_models_from_analyzer(analyzer_result, n=6)
    broken_models   = _broken_models_from_analyzer(analyzer_result, n=3)

    # Today's call (Day 1) — both naive and trust-weighted
    today_row    = df.iloc[0] if len(df) else None
    today_cons   = _direction_consensus(today_row, models_present)
    today_tw     = _trust_weighted_consensus(today_row, models_present, trust_map)
    today_color  = _color_dir(today_tw["consensus"]) if today_tw["n_voters"] > 0 else _color_dir(today_cons["consensus"])
    hc_signal    = _high_conviction_signal(today_row) if today_row is not None else {"signal": "NO DATA", "agree_n": 0, "of": 4, "agreeing": [], "actionable": False}

    # JSON for the JS top-N filter (data-driven from analyzer per ticker)
    import json
    if top_models:
        top_models_json = json.dumps(top_models)
    else:
        top_models_json = json.dumps(['kendall', 'spearman', 'pearson', 'cosine', 'euclidean', 'knn2'])

    # === Build recommended-model badge HTML (TRUST/WEAK/NONE + recommended model) ===
    if analyzer_result and analyzer_result.get("verdict"):
        v        = analyzer_result["verdict"]
        v_badge  = {"TRUST": "✅", "WEAK SIGNAL": "⚠️", "NONE": "🛑"}.get(v, "")
        v_color  = {"TRUST": "#22c55e", "WEAK SIGNAL": "#eab308", "NONE": "#ef4444"}.get(v, "#6b7280")
        rec      = analyzer_result.get("recommended") or "—"
        action   = analyzer_result.get("action") or ""
        n_d      = analyzer_result.get("n_dates", 0)
        # Recent verdict
        rv = analyzer_result.get("recent_verdict")
        recent_html = ""
        if rv:
            rv_badge = {"TRUST": "✅", "WEAK SIGNAL": "⚠️", "NONE": "🛑"}.get(rv["verdict"], "")
            rv_color = {"TRUST": "#22c55e", "WEAK SIGNAL": "#eab308", "NONE": "#ef4444"}.get(rv["verdict"], "#6b7280")
            differs  = rv["verdict"] != v
            differs_msg = '<div style="color:#fbbf24;font-size:11px;margin-top:4px">⚠ Recent differs from overall — trust the recent verdict for short-horizon trades</div>' if differs else ''
            recent_html = (
                f'<div style="border-left:3px solid {rv_color};padding-left:12px;margin-top:12px">'
                f'<div style="color:#94a3b8;font-size:11px;text-transform:uppercase">Recent verdict (last {rv["n_recent"]} days)</div>'
                f'<div style="color:{rv_color};font-weight:600">{rv_badge} {html.escape(rv["verdict"])} — best: <code>{html.escape(rv["best_model"])}</code> ({rv["best_dir"]:.0f}% dir)</div>'
                f'{differs_msg}'
                f'</div>'
            )
        recommended_panel_html = (
            f'<div class="recommended-panel" style="background:#1e293b;border:1px solid #334155;border-left:4px solid {v_color};border-radius:12px;padding:18px 20px;margin-bottom:20px">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:24px;flex-wrap:wrap">'
            f'<div>'
            f'<div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:.05em">Overall verdict (n={n_d})</div>'
            f'<div style="font-size:22px;font-weight:700;color:{v_color};margin-top:4px">{v_badge} {html.escape(v)}</div>'
            f'</div>'
            f'<div>'
            f'<div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:.05em">Recommended model</div>'
            f'<div style="font-size:22px;font-weight:700;color:#38bdf8;margin-top:4px">{html.escape(str(rec).upper())}</div>'
            f'</div>'
            f'</div>'
            f'<div style="margin-top:12px;color:#cbd5e1;font-size:13px"><strong>Action:</strong> {html.escape(action)}</div>'
            f'{recent_html}'
            f'</div>'
        )
    else:
        recommended_panel_html = (
            '<div class="recommended-panel" style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 20px;margin-bottom:20px;color:#94a3b8;font-size:13px">'
            'No analyzer data yet — run <code>analyze.py {ticker}</code> for a verdict + recommended model.'
            '</div>'
        ).replace('{ticker}', html.escape(ticker))

    # === High-conviction Day 1 signal panel ===
    # Based on universe-wide audit: pearson/cosine/manhattan/spearman hit 56-63% Day 1
    # directional accuracy (n=146-168 forecasts each). Trade only when 3+ agree.
    hc_sig = hc_signal["signal"]
    hc_actionable = hc_signal["actionable"]
    if hc_sig == "NO DATA":
        hc_panel_html = ""
    else:
        if hc_actionable:
            hc_color = "#22c55e" if hc_sig == "UP" else "#ef4444"
            hc_badge = "✅ TRADE SIGNAL"
            hc_bg    = "#0f1e2a"
            hc_border = hc_color
        else:
            hc_color = "#6b7280"
            hc_badge = "⚠ NOT ACTIONABLE — top kNN methods don't agree"
            hc_bg    = "#1e293b"
            hc_border = "#334155"
        hc_panel_html = (
            f'<div style="background:{hc_bg};border:2px solid {hc_border};border-radius:12px;'
            f'padding:18px 22px;margin-bottom:20px">'
              f'<div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:.08em;'
              f'margin-bottom:6px">Day 1 high-conviction signal · {hc_badge}</div>'
              f'<div style="font-size:32px;font-weight:700;color:{hc_color};line-height:1.1;margin-bottom:6px">'
                f'{hc_sig}'
              f'</div>'
              f'<div style="color:#94a3b8;font-size:13px">'
                f'{hc_signal["agree_n"]} of {hc_signal["of"]} top kNN methods agree '
                f'({", ".join(hc_signal["agreeing"]) if hc_signal["agreeing"] else "—"})'
              f'</div>'
              f'<div style="color:#64748b;font-size:11px;margin-top:8px">'
                f'Signal fires when ≥3 of pearson/cosine/manhattan/spearman agree at Day 1. '
                f'These 4 averaged 56-63% directional across the universe (n≈150 each). '
                f'Day 2-4 signals are weaker — wait for Day 5 data to accumulate.'
              f'</div>'
            f'</div>'
        )

    # Milestone cards
    milestones = [(label, day) for label, day in [("Day 1", 1), ("Day 5", 5), ("Day 10", 10), ("Day 20", 20), ("Day 30", 30)]]
    milestone_html = []
    for label, target_day in milestones:
        match = df[df["day"] == target_day]
        row = match.iloc[0] if len(match) else None
        milestone_html.append(_milestone_card(row, label, models_present, current_close))

    # knn2 regime/vote info if present
    knn2_panel = ""
    if "knn2_regime" in df.columns or "knn2_vote_pct" in df.columns:
        regime = today_row.get("knn2_regime") if today_row is not None else None
        vote   = _f(today_row.get("knn2_vote_pct")) if today_row is not None else None
        n_n    = today_row.get("knn2_n_neighbors") if today_row is not None else None
        if regime or vote is not None:
            vote_color = "#22c55e" if (vote or 0) >= 70 else ("#eab308" if (vote or 0) >= 55 else "#ef4444")
            vote_str = f"{vote:.1f}%" if vote is not None else "—"
            regime_str = html.escape(str(regime)) if regime and not pd.isna(regime) else "—"
            n_n_str = html.escape(str(int(n_n))) if n_n is not None and not pd.isna(n_n) else "—"
            knn2_panel = f"""
  <div class="panel knn2-panel">
    <div class="panel-header"><h2>knn2 regime + conviction (today)</h2></div>
    <div class="knn2-grid">
      <div><div class="kpi-label">Regime</div><div class="kpi-value">{regime_str}</div></div>
      <div><div class="kpi-label">Vote %</div><div class="kpi-value" style="color:{vote_color}">{vote_str}</div></div>
      <div><div class="kpi-label">Neighbors</div><div class="kpi-value">{n_n_str}</div></div>
    </div>
    <div class="hint">Vote ≥ 70% is the threshold for actionable signal.</div>
  </div>"""

    chart_svg = _svg_chart(df, models_present, current_close)

    broken_set = set(broken_models)
    table_header = (
        "<th data-col='date'>Date</th>"
        "<th data-col='consensus'>Consensus</th>"
        + "".join(
            f"<th data-col='{m[0]}' class='model-col{' broken' if m[0] in broken_set else ''}'"
            f"{' title=&quot;Bottom-3 by validation score — likely noise on this ticker&quot;' if m[0] in broken_set else ''}>"
            f"{html.escape(m[0])}{' ⚠' if m[0] in broken_set else ''}</th>"
            for m in models_present
        )
    )
    table_rows = _table_rows(df, models_present, current_close, broken_set)

    knn_method_html = f" · winning kNN method: <strong>{html.escape(knn_method)}</strong>" if knn_method else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html.escape(ticker)} — Forecast</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0; padding: 24px; line-height: 1.4;
  }}
  .container {{ max-width: 1500px; margin: 0 auto; }}
  header {{
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #334155;
  }}
  h1 {{ font-size: 32px; font-weight: 700; color: #f1f5f9; }}
  h1 .ticker {{ color: #38bdf8; }}
  .meta {{ color: #94a3b8; font-size: 13px; text-align: right; }}
  .meta .updated {{ color: #fbbf24; font-weight: 600; }}

  /* Today's call */
  .today {{
    background: #1e293b; border: 1px solid #334155; border-radius: 12px;
    padding: 24px; margin-bottom: 24px; display: grid;
    grid-template-columns: auto 1fr auto; gap: 32px; align-items: center;
  }}
  .today-label {{ color: #94a3b8; font-size: 11px; text-transform: uppercase;
                  letter-spacing: .05em; margin-bottom: 6px; }}
  .today-call {{ font-size: 36px; font-weight: 700; }}
  .today-detail {{ color: #cbd5e1; font-size: 14px; margin-top: 4px; }}
  .today-current {{ font-size: 28px; font-weight: 700; color: #fbbf24; }}

  /* Milestones */
  .milestones {{
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;
    margin-bottom: 24px;
  }}
  .milestone-card {{
    background: #1e293b; border: 1px solid #334155; border-radius: 12px;
    padding: 14px;
  }}
  .milestone-card.empty {{ opacity: 0.4; }}
  .milestone-header {{ display: flex; justify-content: space-between; margin-bottom: 8px; }}
  .milestone-day {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }}
  .milestone-date {{ font-size: 11px; color: #64748b; }}
  .milestone-consensus {{ font-size: 18px; font-weight: 700; margin-bottom: 4px; }}
  .milestone-consensus .agreement {{ font-size: 10px; color: #94a3b8; font-weight: 400; margin-left: 6px; }}
  .milestone-price {{ font-size: 22px; font-weight: 700; color: #f1f5f9; margin: 4px 0; }}
  .milestone-range, .milestone-band {{ font-size: 11px; color: #94a3b8; }}
  .milestone-move {{ font-size: 12px; font-weight: 600; margin-top: 4px; }}

  /* Panels */
  .panel {{
    background: #1e293b; border: 1px solid #334155; border-radius: 12px;
    overflow: hidden; margin-bottom: 24px;
  }}
  .panel-header {{ padding: 16px 20px; border-bottom: 1px solid #334155;
                    display: flex; justify-content: space-between; align-items: baseline; }}
  .panel-header h2 {{ font-size: 16px; font-weight: 600; color: #f1f5f9; }}

  /* Chart */
  .chart-wrap {{ padding: 12px; }}
  .chart-svg {{ width: 100%; height: auto; }}

  /* knn2 panel */
  .knn2-grid {{ display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 24px;
                 padding: 18px 20px; }}
  .kpi-label {{ color: #94a3b8; font-size: 11px; text-transform: uppercase;
                 letter-spacing: .05em; margin-bottom: 4px; }}
  .kpi-value {{ font-size: 18px; font-weight: 700; color: #f1f5f9; }}
  .hint {{ padding: 0 20px 16px; color: #64748b; font-size: 11px; }}

  /* Filter bar */
  .filter-bar {{
    display: flex; gap: 12px; padding: 16px 20px; border-bottom: 1px solid #334155;
    align-items: center; flex-wrap: wrap;
  }}
  .filter-bar input, .filter-bar select {{
    background: #0f172a; border: 1px solid #334155; border-radius: 6px;
    color: #e2e8f0; padding: 6px 10px; font-size: 13px; font-family: inherit;
  }}
  .filter-bar label {{ color: #94a3b8; font-size: 11px; text-transform: uppercase;
                        letter-spacing: .05em; }}
  .filter-bar button {{
    background: #334155; color: #e2e8f0; border: none; border-radius: 6px;
    padding: 6px 12px; font-size: 12px; cursor: pointer;
  }}
  .filter-bar button:hover {{ background: #475569; }}

  /* Table */
  .table-wrap {{ overflow-x: auto; max-height: 70vh; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #334155; white-space: nowrap; }}
  th {{ background: #0f172a; color: #94a3b8; font-size: 11px; text-transform: uppercase;
        letter-spacing: .05em; font-weight: 600; position: sticky; top: 0; z-index: 1; }}
  td.date-cell {{ font-family: 'SF Mono', Monaco, monospace; color: #e2e8f0; font-weight: 500; }}
  .day-num {{ color: #64748b; font-size: 10px; margin-top: 2px; }}
  .small {{ font-size: 10px; color: #94a3b8; margin-top: 2px; }}
  td.muted {{ color: #475569; text-align: center; }}
  .hide-col {{ display: none; }}
  td.clickable {{ cursor: pointer; transition: background .15s; }}
  td.clickable:hover {{ background: #1e293b; }}
  /* Broken models — bottom-3 by validation score for this ticker. Dimmed so they
     don't visually compete with high-conviction signals. Still readable on hover. */
  th.broken, td.broken {{ opacity: 0.35; }}
  th.broken {{ color: #ef4444 !important; }}
  td.broken:hover {{ opacity: 0.85; background: #1e293b; }}

  /* Modal */
  .modal-overlay {{
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,.7);
    z-index: 100; align-items: center; justify-content: center;
  }}
  .modal-overlay.open {{ display: flex; }}
  .modal {{
    background: #1e293b; border: 1px solid #334155; border-radius: 12px;
    padding: 24px; max-width: 480px; width: 90%; max-height: 85vh; overflow-y: auto;
    box-shadow: 0 20px 60px rgba(0,0,0,.5);
  }}
  .modal h3 {{ margin: 0 0 4px 0; color: #e2e8f0; font-size: 18px; }}
  .modal .modal-sub {{ color: #94a3b8; font-size: 12px; margin-bottom: 16px; }}
  .modal-close {{
    float: right; background: #334155; color: #e2e8f0; border: none; border-radius: 6px;
    padding: 4px 10px; font-size: 12px; cursor: pointer;
  }}
  .modal-close:hover {{ background: #475569; }}
  .kv-grid {{ display: grid; grid-template-columns: auto 1fr; gap: 6px 16px; font-size: 13px; }}
  .kv-grid .k {{ color: #94a3b8; }}
  .kv-grid .v {{ color: #e2e8f0; font-family: 'SF Mono', Monaco, monospace; }}
  .kv-section {{ margin-top: 16px; padding-top: 12px; border-top: 1px solid #334155; }}
  .kv-section-title {{ color: #64748b; font-size: 10px; text-transform: uppercase;
                       letter-spacing: .08em; margin-bottom: 8px; }}

  footer {{
    margin-top: 24px; padding-top: 16px; border-top: 1px solid #334155;
    color: #64748b; font-size: 11px; line-height: 1.5;
  }}
</style>
</head>
<body>
<div class="container">

  <header>
    <h1><span class="ticker">{html.escape(ticker)}</span> Forecast</h1>
    <div class="meta">
      <div class="updated">Last updated: {html.escape(last_updated)}</div>
      <div>{n_days} days · {date_min} → {date_max}{knn_method_html}</div>
    </div>
  </header>

  {recommended_panel_html}

  {hc_panel_html}

  <div class="today">
    <div>
      <div class="today-label">Today's call (Day 1) — TRUST-WEIGHTED</div>
      <div class="today-call" style="color:{today_color}">{today_tw['consensus'] if today_tw['n_voters'] > 0 else today_cons['consensus']}</div>
      <div class="today-detail">
        {('Trust-weighted: ' + str(today_tw['agreement_pct']) + '% agreement (' + str(today_tw['n_voters']) + ' models with dir≥50%)') if today_tw['n_voters'] > 0 else ('No validated models — falling back to naive vote: ' + str(today_cons['up']) + '↑ ' + str(today_cons['down']) + '↓')}
      </div>
      <div class="today-detail" style="color:#64748b;font-size:11px">
        Naive (all models, equal weight): {today_cons['consensus']} — {today_cons['up']}↑ {today_cons['down']}↓ ({today_cons['agreement_pct']}% agree)
      </div>
    </div>
    <div></div>
    <div>
      <div class="today-label">Current close</div>
      <div class="today-current">{('$%.2f' % current_close) if current_close else '—'}</div>
    </div>
  </div>

  <div class="milestones">
    {''.join(milestone_html)}
  </div>

  {knn2_panel}

  <div class="panel">
    <div class="panel-header">
      <h2>Forecast cone — model medians over time</h2>
      <div style="color:#94a3b8;font-size:12px">dashed line = current close</div>
    </div>
    <div class="chart-wrap">{chart_svg}</div>
  </div>

  <div class="panel">
    <div class="panel-header">
      <h2>All forecast days</h2>
    </div>
    <div class="filter-bar">
      <label>Date</label>
      <input type="text" id="date-filter" placeholder="2026-05" />
      <label>Direction</label>
      <select id="dir-filter">
        <option value="">all</option>
        <option value="UP">UP only</option>
        <option value="DOWN">DOWN only</option>
        <option value="MIXED">MIXED only</option>
      </select>
      <label>Min agreement %</label>
      <input type="number" id="agree-filter" min="0" max="100" placeholder="0" style="width: 70px" />
      <button id="show-all-models">Show all models</button>
      <button id="show-top-only">Show top 6 only</button>
      <span style="margin-left:auto;color:#64748b;font-size:12px" id="row-count"></span>
    </div>
    <div class="table-wrap">
      <table id="forecast-table">
        <thead><tr>{table_header}</tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
  </div>

  <footer>
    Forecast pulled from <code>forecast_results_{html.escape(ticker)}.csv</code>.
    Direction column shows each model's UP/DOWN call relative to current close.
    Consensus = majority of models with a direction. Agreement % = how many models agree with the consensus.
    Cone = min/max across all models' low/high bands.
    Click any model cell for full probability bands and pdcp details.
  </footer>

</div>

<div class="modal-overlay" id="cell-modal">
  <div class="modal">
    <button class="modal-close" id="modal-close">Close ✕</button>
    <h3 id="modal-title">—</h3>
    <div class="modal-sub" id="modal-sub">—</div>
    <div id="modal-body"></div>
  </div>
</div>

<script>
(function() {{
  const dateFilter  = document.getElementById('date-filter');
  const dirFilter   = document.getElementById('dir-filter');
  const agreeFilter = document.getElementById('agree-filter');
  const rows = document.querySelectorAll('.forecast-row');
  const rowCount = document.getElementById('row-count');

  // Top 6 models — derived from validation conservative leaderboard, generic default
  // Data-driven from analyzer.py (top 6 by conservative score for THIS ticker).
  // Fallback to a sensible default if analyzer hasn't run yet.
  const TOP_MODELS = {top_models_json};

  function applyFilters() {{
    const ds = (dateFilter.value || '').trim().toLowerCase();
    const dr = (dirFilter.value || '').trim();
    const ag = parseFloat(agreeFilter.value) || 0;
    let visible = 0;
    rows.forEach(r => {{
      const date = (r.dataset.date || '').toLowerCase();
      const dir  = r.dataset.direction || '';
      const agp  = parseFloat(r.dataset.agreement) || 0;
      const ok = (!ds || date.includes(ds)) &&
                 (!dr || dir === dr) &&
                 (agp >= ag);
      r.style.display = ok ? '' : 'none';
      if (ok) visible++;
    }});
    rowCount.textContent = visible + ' of ' + rows.length + ' rows';
  }}

  dateFilter.addEventListener('input', applyFilters);
  dirFilter.addEventListener('change', applyFilters);
  agreeFilter.addEventListener('input', applyFilters);

  document.getElementById('show-all-models').addEventListener('click', () => {{
    document.querySelectorAll('.model-col').forEach(th => th.classList.remove('hide-col'));
    document.querySelectorAll('tbody tr').forEach(r => {{
      const cells = r.querySelectorAll('td');
      cells.forEach((c, i) => {{ if (i >= 2) c.classList.remove('hide-col'); }});
    }});
  }});

  document.getElementById('show-top-only').addEventListener('click', () => {{
    const headers = document.querySelectorAll('th[data-col]');
    headers.forEach((th, i) => {{
      if (i < 2) return;
      const col = th.dataset.col;
      const keep = TOP_MODELS.includes(col);
      th.classList.toggle('hide-col', !keep);
      document.querySelectorAll('tbody tr').forEach(r => {{
        const cell = r.querySelectorAll('td')[i];
        if (cell) cell.classList.toggle('hide-col', !keep);
      }});
    }});
  }});

  applyFilters();

  // Cell click → modal
  const modal = document.getElementById('cell-modal');
  const modalTitle = document.getElementById('modal-title');
  const modalSub = document.getElementById('modal-sub');
  const modalBody = document.getElementById('modal-body');
  document.getElementById('modal-close').addEventListener('click', () => modal.classList.remove('open'));
  modal.addEventListener('click', (e) => {{ if (e.target === modal) modal.classList.remove('open'); }});
  document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') modal.classList.remove('open'); }});

  function fmt(v, prefix, suffix) {{
    if (v === null || v === undefined || isNaN(v)) return '—';
    return (prefix || '') + Number(v).toFixed(2) + (suffix || '');
  }}
  function row(k, v) {{
    return '<div class="k">' + k + '</div><div class="v">' + v + '</div>';
  }}
  function section(title, html) {{
    return '<div class="kv-section"><div class="kv-section-title">' + title + '</div>'
         + '<div class="kv-grid">' + html + '</div></div>';
  }}

  document.querySelectorAll('td.clickable').forEach(td => {{
    td.addEventListener('click', () => {{
      let d;
      try {{ d = JSON.parse(td.dataset.detail); }} catch (e) {{ return; }}
      modalTitle.textContent = d.model.toUpperCase() + ' — ' + d.date;
      const dayStr = d.day !== null ? ('Day ' + d.day) : '';
      const ccStr = d.current_close ? ('Current close: $' + Number(d.current_close).toFixed(2)) : '';
      modalSub.textContent = [dayStr, ccStr].filter(Boolean).join(' · ');

      const dirColor = d.direction === 'UP' ? '#22c55e' : (d.direction === 'DOWN' ? '#ef4444' : '#6b7280');
      const moveColor = (d.pct_move >= 0) ? '#22c55e' : '#ef4444';
      const moveStr = d.pct_move !== null && d.pct_move !== undefined
        ? '<span style="color:' + moveColor + '">' + (d.pct_move >= 0 ? '+' : '') + Number(d.pct_move).toFixed(2) + '%</span>'
        : '—';

      let html = '';
      html += section('Prediction', ''
        + row('Median', fmt(d.median, '$'))
        + row('Direction', '<span style="color:' + dirColor + ';font-weight:600">' + (d.direction || '—') + '</span>')
        + row('Move vs close', moveStr)
      );
      html += section('Cone (price levels)', ''
        + row('Low (p90 floor)', fmt(d.low_p90, '$'))
        + row('High (p20 ceil)', fmt(d.high_p20, '$'))
      );
      if (d.p90 !== null || d.p60 !== null || d.p20 !== null) {{
        html += section('Probability bands', ''
          + row('p90 (90% above)', fmt(d.p90, '$'))
          + row('p60 (60% above)', fmt(d.p60, '$'))
          + row('p20 (20% above)', fmt(d.p20, '$'))
        );
      }}
      if (d.pdcp90 !== null || d.pdcp60 !== null || d.pdcp20 !== null || d.pdcp !== undefined) {{
        let pdcpRows = '';
        if (d.pdcp !== undefined) pdcpRows += row('pdcp', fmt(d.pdcp, '$'));
        else {{
          pdcpRows += row('pdcp90 (conservative upside)', fmt(d.pdcp90, '$'));
          pdcpRows += row('pdcp60 (moderate upside)', fmt(d.pdcp60, '$'));
          pdcpRows += row('pdcp20 (stretch upside)', fmt(d.pdcp20, '$'));
        }}
        html += section('PDCP — upside targets (above current close)', pdcpRows);
      }}
      if (d.vote_pct !== undefined || d.regime !== undefined || d.n_neighbors !== undefined) {{
        let knn2Rows = '';
        if (d.vote_pct !== undefined && d.vote_pct !== null) knn2Rows += row('Vote %', Number(d.vote_pct).toFixed(1) + '%');
        if (d.n_neighbors !== undefined && d.n_neighbors !== null) knn2Rows += row('Neighbors', Number(d.n_neighbors).toFixed(0));
        if (d.regime) knn2Rows += row('Regime', d.regime);
        if (knn2Rows) html += section('knn2 details', knn2Rows);
      }}
      modalBody.innerHTML = html;
      modal.classList.add('open');
    }});
  }});
}})();
</script>
</body>
</html>"""


def generate_forecast_report(ticker: str) -> str | None:
    """Build HTML from forecast_results_{T}.csv. Returns output path or None."""
    csv_path = f"output/{ticker}/forecast_results_{ticker}.csv"
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if len(df) == 0:
        return None

    out_dir = f"output/{ticker}/reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/forecast_{ticker}.html"
    with open(out_path, "w") as f:
        f.write(_build_html(ticker, df))
    return out_path
