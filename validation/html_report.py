"""
Generate a self-contained HTML validation dashboard per ticker.
Reads forecast_validation_{T}.csv, writes output/{T}/reports/validate_{T}.html.
No external assets — everything is inlined for portability.
"""
import os
import re
import html
from datetime import datetime
import pandas as pd


_MODELS = ["analog", "spearman", "pearson", "cosine", "euclidean",
           "kendall", "manhattan", "xgboost", "lightgbm",
           "randomforest", "knn2"]


def _parse_cell(cell: str) -> dict:
    if not isinstance(cell, str):
        return {}
    out = {}
    for line in cell.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip().lower()] = v.strip()
    return out


def _err_pct(p: dict):
    m = re.search(r"\(([\d.]+)%\)", p.get("error", ""))
    return float(m.group(1)) if m else None


def _err_dollars(p: dict):
    m = re.search(r"\$([\d.]+)", p.get("error", ""))
    return float(m.group(1)) if m else None


def _dir_correct(p: dict):
    d = p.get("direction", "")
    if "CORRECT" in d:
        return 1
    if "MISS" in d:
        return 0
    return None


def _in_band(p: dict):
    r = p.get("range_result", "")
    if r == "IN BAND":
        return 1
    if r in ("ABOVE BAND", "BELOW BAND"):
        return 0
    return None


def _median_price(p: dict):
    m = re.search(r"\$([\d.]+)", p.get("median", "") or p.get("price", ""))
    return float(m.group(1)) if m else None


def _direction_label(p: dict):
    d = p.get("direction", "").upper()
    if "UP" in d:
        return "UP"
    if "DOWN" in d:
        return "DOWN"
    return ""


def _compute_model_stats(df: pd.DataFrame) -> list:
    rows = []
    for m in _MODELS:
        if m not in df.columns:
            continue
        cells = df[m].apply(_parse_cell)
        errs   = cells.apply(_err_pct).dropna()
        dirs   = cells.apply(_dir_correct).dropna()
        bands  = cells.apply(_in_band).dropna()
        n = len(errs)
        if n == 0:
            continue
        avg_err   = float(errs.mean())
        med_err   = float(errs.median())
        tight_pct = float((errs <= 3).mean() * 100)
        very_tight = float((errs <= 1.5).mean() * 100)
        dir_pct   = float(dirs.mean() * 100) if len(dirs) else None
        band_pct  = float(bands.mean() * 100) if len(bands) else None
        score = (
            0.40 * (dir_pct or 0)
          + 0.30 * tight_pct
          + 0.20 * (band_pct or 0)
          + 0.10 * max(0, 100 - avg_err)
        )
        rows.append({
            "model":      m,
            "n":          n,
            "avg_err":    round(avg_err, 2),
            "med_err":    round(med_err, 2),
            "tight":      round(tight_pct, 1),
            "very_tight": round(very_tight, 1),
            "dir":        round(dir_pct, 1) if dir_pct is not None else None,
            "in_band":    round(band_pct, 1) if band_pct is not None else None,
            "score":      round(score, 1),
        })
    return rows


def _kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f"<div class='kpi-sub'>{html.escape(sub)}</div>" if sub else ""
    return f"""
    <div class="kpi">
      <div class="kpi-label">{html.escape(label)}</div>
      <div class="kpi-value">{html.escape(value)}</div>
      {sub_html}
    </div>"""


def _color_for_dir(pct):
    if pct is None:
        return "#6b7280"
    if pct >= 70: return "#22c55e"
    if pct >= 50: return "#eab308"
    return "#ef4444"


def _color_for_err(pct):
    if pct is None:
        return "#6b7280"
    if pct <= 2:  return "#22c55e"
    if pct <= 5:  return "#eab308"
    return "#ef4444"


def _leaderboard_rows(stats: list, sort_key: str, ascending: bool) -> str:
    sorted_stats = sorted(stats, key=lambda r: (r[sort_key] is None, r[sort_key] if not ascending else -r[sort_key]), reverse=True)
    if ascending:
        sorted_stats = sorted(stats, key=lambda r: (r[sort_key] is None, r[sort_key] if r[sort_key] is not None else 999))
    out = []
    for i, r in enumerate(sorted_stats, 1):
        dir_s   = f"{r['dir']:.1f}%"     if r['dir']     is not None else "—"
        band_s  = f"{r['in_band']:.1f}%" if r['in_band'] is not None else "—"
        dir_clr  = _color_for_dir(r['dir'])
        err_clr  = _color_for_err(r['avg_err'])
        rank_class = "rank-gold" if i == 1 else ("rank-silver" if i == 2 else ("rank-bronze" if i == 3 else "rank-norm"))
        out.append(f"""
        <tr>
          <td class="{rank_class}">{i}</td>
          <td class="model">{html.escape(r['model'])}</td>
          <td>{r['n']}</td>
          <td><span style="color:{err_clr}">{r['avg_err']:.2f}%</span></td>
          <td>{r['tight']:.0f}%</td>
          <td><span style="color:{dir_clr}">{dir_s}</span></td>
          <td>{band_s}</td>
          <td><strong>{r['score']:.1f}</strong></td>
        </tr>""")
    return "\n".join(out)


def _per_day_rows(df: pd.DataFrame, ranked_models: list) -> str:
    out = []
    sorted_df = df.sort_values("date", ascending=False)
    top_models = ranked_models[:6]  # show top 6 columns to keep table readable

    for _, row in sorted_df.iterrows():
        date     = str(row.get("date", ""))[:10]
        day      = row.get("day", "?")
        actual   = row.get("actual", "")
        prior    = row.get("prior_close", "")

        # actual cell may be multi-line text "C: $X\nO:...". Extract close.
        actual_f = None
        if isinstance(actual, str):
            m = re.search(r'C:\s*\$?([\d.]+)', actual)
            if m:
                actual_f = float(m.group(1))
            else:
                try: actual_f = float(actual.replace("$", "").strip())
                except Exception: actual_f = None

        prior_f = None
        try:
            prior_f = float(str(prior).replace("$", "").strip())
        except Exception:
            prior_f = None

        if actual_f is not None and prior_f is not None and prior_f != 0:
            move_pct = (actual_f - prior_f) / prior_f * 100
            move_arrow = "▲" if move_pct >= 0 else "▼"
            move_color = "#22c55e" if move_pct >= 0 else "#ef4444"
            actual_str = f"${actual_f:.2f} <span style='color:{move_color};font-size:11px'>{move_arrow}{abs(move_pct):.1f}%</span>"
        else:
            actual_str = html.escape(str(actual)[:20])

        cells = [f"<td class='date-cell'>{date}<div class='day-num'>day {day}</div></td>",
                 f"<td>{actual_str}</td>"]

        for m in top_models:
            cell_text = row.get(m, "")
            parsed = _parse_cell(cell_text)
            if not parsed:
                cells.append("<td class='muted'>—</td>")
                continue
            med = _median_price(parsed)
            err_p = _err_pct(parsed)
            dir_p = _direction_label(parsed)
            dir_correct = _dir_correct(parsed)
            in_band_v   = _in_band(parsed)

            badges = []
            if dir_p:
                clr = "#22c55e" if dir_correct == 1 else ("#ef4444" if dir_correct == 0 else "#6b7280")
                badges.append(f"<span class='badge' style='background:{clr}20;color:{clr}'>{dir_p}</span>")
            if in_band_v == 1:
                badges.append("<span class='badge' style='background:#22c55e20;color:#22c55e'>IN</span>")
            elif in_band_v == 0:
                badges.append("<span class='badge' style='background:#ef444420;color:#ef4444'>OUT</span>")

            err_clr = _color_for_err(err_p)
            med_str = f"${med:.2f}" if med is not None else "—"
            err_str = f"<div class='err' style='color:{err_clr}'>{err_p:.1f}%</div>" if err_p is not None else ""
            cells.append(f"<td>{med_str}{err_str}<div class='badges'>{''.join(badges)}</div></td>")

        out.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(out)


def _per_day_header(ranked_models: list) -> str:
    cols = ["<th>Date</th>", "<th>Actual</th>"]
    for m in ranked_models[:6]:
        cols.append(f"<th>{html.escape(m)}</th>")
    return "<tr>" + "".join(cols) + "</tr>"


def _analysis_panel(ticker: str) -> str:
    """If analysis_{T}.md exists, regenerate structured analysis and render as HTML panel."""
    md_path = f"output/{ticker}/reports/analysis_{ticker}.md"
    if not os.path.exists(md_path):
        return ""
    try:
        from validation.analyzer import analyze
        result = analyze(ticker)
    except Exception:
        return ""
    if "error" in result:
        return ""

    verdict = result.get("verdict", "")
    badge_emoji = {"TRUST": "✅", "WEAK SIGNAL": "⚠️", "NONE": "🛑"}.get(verdict, "")
    badge_color = {"TRUST": "#22c55e", "WEAK SIGNAL": "#eab308", "NONE": "#ef4444"}.get(verdict, "#6b7280")
    rec = result.get("recommended") or "—"

    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in result.get("reasons", []))
    caveats = "".join(f"<li>{html.escape(c)}</li>" for c in result.get("caveats", []))
    right   = "".join(f"<li>{html.escape(r)}</li>" for r in result.get("right", []))
    wrong   = "".join(f"<li>{html.escape(w)}</li>" for w in result.get("wrong", []))
    obs     = "".join(f"<li>{html.escape(o)}</li>" for o in result.get("observations", []))
    usage_items = result.get("usage", {}) or {}
    usage   = "".join(
        f"<li><strong>{html.escape(k)}:</strong> {html.escape(v)}</li>"
        for k, v in usage_items.items()
    )

    reasons_html = f"<div class='analysis-block'><h4>Why</h4><ul>{reasons}</ul></div>" if reasons else ""
    caveats_html = f"<div class='analysis-block'><h4>Caveats</h4><ul>{caveats}</ul></div>" if caveats else ""
    right_html   = f"<div class='analysis-block'><h4 style='color:#22c55e'>✓ Going right</h4><ul>{right}</ul></div>" if right else ""
    wrong_html   = f"<div class='analysis-block'><h4 style='color:#ef4444'>✗ Going wrong</h4><ul>{wrong}</ul></div>" if wrong else ""
    obs_html     = f"<div class='analysis-block'><h4>• Observations</h4><ul>{obs}</ul></div>" if obs else ""
    usage_html   = f"<div class='analysis-block'><h4>Recommended usage</h4><ul>{usage}</ul></div>" if usage else ""

    return f"""
  <div class="analysis-panel">
    <div class="analysis-header">
      <div>
        <div class="analysis-verdict-label">VERDICT</div>
        <div class="analysis-verdict" style="color:{badge_color}">
          {badge_emoji} {html.escape(verdict)}
        </div>
      </div>
      <div class="analysis-rec">
        <div class="analysis-rec-label">Recommended model</div>
        <div class="analysis-rec-value">{html.escape(rec).upper()}</div>
      </div>
    </div>
    <div class="analysis-grid">
      {reasons_html}
      {caveats_html}
      {right_html}
      {wrong_html}
      {obs_html}
      {usage_html}
    </div>
  </div>
"""


def _build_html(ticker: str, df: pd.DataFrame) -> str:
    stats = _compute_model_stats(df)
    if not stats:
        return _empty_html(ticker)

    # Two leaderboards
    by_score = sorted(stats, key=lambda r: -r["score"])
    by_err   = sorted(stats, key=lambda r: r["avg_err"])

    best_model       = by_score[0]
    best_err_model   = by_err[0]
    n_days           = int(df.shape[0])
    date_min         = str(df["date"].min())[:10] if "date" in df.columns and len(df) else "—"
    date_max         = str(df["date"].max())[:10] if "date" in df.columns and len(df) else "—"
    last_updated     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build KPI cards
    avg_dir_all = [r["dir"] for r in stats if r["dir"] is not None]
    avg_dir = sum(avg_dir_all) / len(avg_dir_all) if avg_dir_all else 0

    kpis = "".join([
        _kpi_card("Best Conservative", best_model["model"].upper(),
                  f"score {best_model['score']:.1f} · {best_model['dir'] or 0:.0f}% dir · {best_model['avg_err']:.1f}% err"),
        _kpi_card("Tightest Predictions", best_err_model["model"].upper(),
                  f"avg err {best_err_model['avg_err']:.2f}% · {best_err_model['tight']:.0f}% within 3%"),
        _kpi_card("Validation Days", str(n_days),
                  f"{date_min} → {date_max}"),
        _kpi_card("Avg Direction Acc", f"{avg_dir:.1f}%",
                  f"across all models"),
    ])

    analysis_html = _analysis_panel(ticker)

    cons_table = _leaderboard_rows(stats, "score", ascending=False)
    err_table  = _leaderboard_rows(stats, "avg_err", ascending=True)

    ranked_models_for_perday = [r["model"] for r in by_score]
    per_day_header = _per_day_header(ranked_models_for_perday)
    per_day_rows   = _per_day_rows(df, ranked_models_for_perday)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html.escape(ticker)} — Validation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    padding: 24px;
    line-height: 1.4;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; }}
  header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid #334155;
  }}
  h1 {{
    font-size: 32px;
    font-weight: 700;
    color: #f1f5f9;
  }}
  h1 .ticker {{ color: #38bdf8; }}
  .meta {{ color: #94a3b8; font-size: 13px; text-align: right; }}
  .meta .updated {{ color: #fbbf24; font-weight: 600; }}

  .kpis {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
  }}
  .kpi {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px;
  }}
  .kpi-label {{
    color: #94a3b8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
  }}
  .kpi-value {{
    color: #f1f5f9;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 4px;
  }}
  .kpi-sub {{ color: #cbd5e1; font-size: 12px; }}

  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 32px;
  }}
  @media (max-width: 1100px) {{
    .grid-2 {{ grid-template-columns: 1fr; }}
    .kpis    {{ grid-template-columns: repeat(2, 1fr); }}
  }}

  .panel {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    overflow: hidden;
  }}
  .panel-header {{
    padding: 16px 20px;
    border-bottom: 1px solid #334155;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }}
  .panel-header h2 {{
    font-size: 16px;
    font-weight: 600;
    color: #f1f5f9;
  }}
  .panel-header .subtitle {{ color: #94a3b8; font-size: 12px; }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  th, td {{
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid #334155;
  }}
  th {{
    background: #0f172a;
    color: #94a3b8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
  }}
  tr:last-child td {{ border-bottom: none; }}
  td.model {{ font-weight: 600; color: #38bdf8; }}
  td.muted {{ color: #475569; text-align: center; }}

  .rank-gold   {{ color: #fbbf24; font-weight: 700; }}
  .rank-silver {{ color: #cbd5e1; font-weight: 700; }}
  .rank-bronze {{ color: #d97706; font-weight: 700; }}
  .rank-norm   {{ color: #64748b; }}

  .perday-panel {{ overflow-x: auto; }}
  .perday-panel table {{ min-width: 900px; }}
  td.date-cell {{
    font-family: 'SF Mono', Monaco, monospace;
    color: #e2e8f0;
    font-weight: 500;
  }}
  .day-num {{
    color: #64748b;
    font-size: 10px;
    margin-top: 2px;
  }}
  .err {{
    font-size: 11px;
    margin-top: 2px;
  }}
  .badges {{
    display: flex;
    gap: 4px;
    margin-top: 4px;
  }}
  .badge {{
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
  }}

  .analysis-panel {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 32px;
  }}
  .analysis-header {{
    display: flex;
    gap: 32px;
    align-items: center;
    padding-bottom: 16px;
    margin-bottom: 16px;
    border-bottom: 1px solid #334155;
  }}
  .analysis-verdict-label, .analysis-rec-label {{
    color: #94a3b8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
  }}
  .analysis-verdict {{
    font-size: 22px;
    font-weight: 700;
  }}
  .analysis-rec-value {{
    font-size: 22px;
    font-weight: 700;
    color: #38bdf8;
  }}
  .analysis-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px 24px;
  }}
  @media (max-width: 900px) {{
    .analysis-grid {{ grid-template-columns: 1fr; }}
    .analysis-header {{ flex-direction: column; gap: 12px; align-items: flex-start; }}
  }}
  .analysis-block h4 {{
    font-size: 12px;
    font-weight: 600;
    color: #cbd5e1;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
  }}
  .analysis-block ul {{
    list-style: none;
    padding-left: 0;
  }}
  .analysis-block li {{
    color: #e2e8f0;
    font-size: 13px;
    padding: 4px 0 4px 14px;
    position: relative;
    line-height: 1.45;
  }}
  .analysis-block li::before {{
    content: "•";
    color: #64748b;
    position: absolute;
    left: 0;
  }}

  footer {{
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid #334155;
    color: #64748b;
    font-size: 11px;
    line-height: 1.5;
  }}
</style>
</head>
<body>
<div class="container">

  <header>
    <h1><span class="ticker">{html.escape(ticker)}</span> Validation Dashboard</h1>
    <div class="meta">
      <div class="updated">Last updated: {last_updated}</div>
      <div>Range: {date_min} → {date_max} · {n_days} days</div>
    </div>
  </header>

  <div class="kpis">{kpis}</div>

  {analysis_html}

  <div class="grid-2">
    <div class="panel">
      <div class="panel-header">
        <h2>Conservative Leaderboard</h2>
        <div class="subtitle">40% dir · 30% tight · 20% in-band · 10% err</div>
      </div>
      <table>
        <thead>
          <tr><th>#</th><th>Model</th><th>N</th><th>Avg Err</th><th>Tight ≤3%</th><th>Dir Acc</th><th>In-Band</th><th>Score</th></tr>
        </thead>
        <tbody>{cons_table}</tbody>
      </table>
    </div>

    <div class="panel">
      <div class="panel-header">
        <h2>Production Leaderboard</h2>
        <div class="subtitle">sorted by avg error % (lower = better)</div>
      </div>
      <table>
        <thead>
          <tr><th>#</th><th>Model</th><th>N</th><th>Avg Err</th><th>Tight ≤3%</th><th>Dir Acc</th><th>In-Band</th><th>Score</th></tr>
        </thead>
        <tbody>{err_table}</tbody>
      </table>
    </div>
  </div>

  <div class="panel perday-panel">
    <div class="panel-header">
      <h2>Per-Day Predictions</h2>
      <div class="subtitle">most recent first · top 6 models by conservative score</div>
    </div>
    <table>
      <thead>{per_day_header}</thead>
      <tbody>{per_day_rows}</tbody>
    </table>
  </div>

  <footer>
    <strong>Conservative score formula:</strong>
    0.40 × directional_accuracy + 0.30 × pct_within_3%_error + 0.20 × in_band_rate + 0.10 × (100 − avg_err%).<br>
    Higher is better. Models with no cone (analog) are penalized on in-band component.
    Per-day badges: green = correct/in-band, red = miss/out-of-band.
  </footer>

</div>
</body>
</html>"""


def _empty_html(ticker: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{html.escape(ticker)} — No Data</title>
<style>body{{font-family:sans-serif;background:#0f172a;color:#e2e8f0;padding:48px;text-align:center}}</style>
</head><body>
<h1>{html.escape(ticker)}</h1>
<p>No validation data available yet.</p>
<p style="color:#64748b;font-size:12px">Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</body></html>"""


def generate_validation_report(ticker: str) -> str | None:
    """Build HTML report from forecast_validation_{T}.csv. Returns output path or None."""
    csv_path = f"output/{ticker}/forecast_validation_{ticker}.csv"
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    out_dir = f"output/{ticker}/reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/validate_{ticker}.html"

    html_str = _build_html(ticker, df)
    with open(out_path, "w") as f:
        f.write(html_str)
    return out_path
