import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


# ── Zig-zag pivot detector ─────────────────────────────────────────────────────

def detect_zigzag_pivots(prices: list, n_legs: int) -> list[tuple[int, float, str]]:
    """Return list of (index, price, 'HIGH'|'LOW') pivot tuples."""
    pivots = []
    for i in range(n_legs, len(prices) - n_legs):
        window = prices[i - n_legs : i + n_legs + 1]
        if prices[i] == max(window):
            pivots.append((i, prices[i], "HIGH"))
        elif prices[i] == min(window):
            pivots.append((i, prices[i], "LOW"))

    # Deduplicate: keep only alternating HIGH/LOW pivots
    clean = []
    for p in pivots:
        if not clean or clean[-1][2] != p[2]:
            clean.append(p)
        else:
            # keep the more extreme of two consecutive same-type pivots
            if p[2] == "HIGH" and p[1] > clean[-1][1]:
                clean[-1] = p
            elif p[2] == "LOW" and p[1] < clean[-1][1]:
                clean[-1] = p

    return clean


# ── Main plot function ─────────────────────────────────────────────────────────

def plot_forecast(
    df: pd.DataFrame,
    analog_result: dict,
    knn_result: dict,
    config: dict,
    save_path: str,
) -> None:
    DARK_BG   = "#1a1a2e"
    GRID_COL  = "#2a2a4a"
    WHITE     = "#e0e0e0"
    BLUE      = "#4a90d9"
    GREEN     = "#14d990"
    GREY_CONE = "#888888"

    show_all_paths = config.get("SHOW_ALL_PATHS", True)
    show_zigzag    = config.get("SHOW_ZIGZAG", True)
    zigzag_legs    = config.get("ZIGZAG_LEGS", 3)
    window_len     = config["WINDOW_LEN"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize         = (16, 9),
        gridspec_kw     = {"height_ratios": [4, 1]},
        facecolor       = DARK_BG,
    )
    for ax in (ax1, ax2):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.xaxis.label.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, alpha=0.4, linewidth=0.5)

    # ── Historical price (Panel 1) ─────────────────────────────────────────────
    # Show only last ~1 month of data (21 trading days) before today
    lookback_days = 21
    df_display = df.iloc[-lookback_days:].reset_index(drop=True)
    df_full_index = df.copy()

    history_x = np.arange(len(df_display))
    ax1.plot(history_x, df_display["Close"].values, color=WHITE, linewidth=0.8, zorder=2)

    # ── Matched segment shading ────────────────────────────────────────────────
    match_start_ts = analog_result["match_start"]
    match_end_ts   = analog_result["match_end"]
    ms_idx_full = df_full_index.index.get_loc(match_start_ts)
    me_idx_full = df_full_index.index.get_loc(match_end_ts)

    # Check if match is within display range, adjust for 1-month window
    df_start_full = len(df_full_index) - len(df_display)
    ms_idx = ms_idx_full - df_start_full
    me_idx = me_idx_full - df_start_full

    # Only draw match segment if it's visible in the display window
    if ms_idx >= 0 and me_idx < len(df_display):
        ax1.axvspan(ms_idx, me_idx, color=BLUE, alpha=0.20, zorder=1)
        ax1.text(
            (ms_idx + me_idx) / 2,
            ax1.get_ylim()[1] if ax1.get_ylim()[1] != 0 else df_display["Close"].max(),
            f"Best Match: {match_start_ts.date()} → {match_end_ts.date()} | Score: {analog_result['best_score']:.3f}",
            color=BLUE, fontsize=7, ha="center", va="top",
            fontfamily="monospace",
        )

    # ── Current window shading ─────────────────────────────────────────────────
    cw_start = len(df_display) - window_len
    if cw_start < 0:
        cw_start = 0
    ax1.axvspan(cw_start, len(df_display) - 1, color="#5050cc", alpha=0.18, zorder=1)
    ax1.text(
        (cw_start + len(df_display) - 1) / 2,
        df_display["Close"].iloc[cw_start:].min() * 0.98,
        "Current\nWindow",
        color="#8888ff", fontsize=6.5, ha="center", va="top", fontfamily="monospace",
    )

    # ── Today separator ────────────────────────────────────────────────────────
    today_x = len(df_display) - 1
    ax1.axvline(today_x, color=WHITE, linewidth=0.9, linestyle="--", alpha=0.5)
    ax1.text(today_x + 0.3, ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else df_display["Close"].min(),
             "Today", color=WHITE, fontsize=7, va="bottom", fontfamily="monospace")

    # ── Forecast x-axis positions ──────────────────────────────────────────────
    forecast_cone = knn_result["forecast_cone"]
    forecast_len  = len(forecast_cone)
    forecast_x    = np.arange(len(df_display), len(df_display) + forecast_len)

    # ── k-NN individual paths (faint) ─────────────────────────────────────────
    if show_all_paths:
        for path in knn_result["all_paths"]:
            ax1.plot(forecast_x, path, color=GREY_CONE, linewidth=0.6, alpha=0.25, zorder=3)

    # ── Forecast cone (10th–90th percentile) ──────────────────────────────────
    ax1.fill_between(
        forecast_x,
        forecast_cone["low"].values,
        forecast_cone["high"].values,
        color=GREY_CONE, alpha=0.25, zorder=4,
    )

    # ── k-NN median path ──────────────────────────────────────────────────────
    ax1.plot(
        forecast_x, forecast_cone["median"].values,
        color=WHITE, linewidth=1.4, linestyle="dotted", zorder=5, label="k-NN Median",
    )

    # ── Single analog forecast path ───────────────────────────────────────────
    analog_prices = analog_result["forecast_df"]["price"].values
    ax1.plot(
        forecast_x, analog_prices,
        color=GREEN, linewidth=1.2, linestyle="dotted", zorder=5, label="Analog Forecast",
    )

    # ── Zig-zag pivots on median path ─────────────────────────────────────────
    if show_zigzag:
        med_prices = list(forecast_cone["median"].values)
        pivots = detect_zigzag_pivots(med_prices, zigzag_legs)
        for j in range(len(pivots) - 1):
            p1, p2 = pivots[j], pivots[j + 1]
            seg_col = GREEN if p2[1] > p1[1] else "#e05050"
            ax1.plot(
                [forecast_x[p1[0]], forecast_x[p2[0]]],
                [p1[1], p2[1]],
                color=seg_col, linewidth=1.4, zorder=6,
            )

    # ── Price labels at milestones ────────────────────────────────────────────
    for day_label in [5, 10, 20]:
        if day_label <= forecast_len:
            row  = forecast_cone[forecast_cone["day"] == day_label].iloc[0]
            px   = forecast_x[day_label - 1]
            py   = row["median"]
            ax1.annotate(
                f"${py:.2f}",
                xy=(px, py), xytext=(px + 0.5, py),
                color=WHITE, fontsize=6.5, fontfamily="monospace",
                arrowprops=dict(arrowstyle="-", color=WHITE, lw=0.5),
            )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=GREEN,     linestyle="dotted", linewidth=1.2, label="Analog Forecast"),
        Line2D([0], [0], color=WHITE,     linestyle="dotted", linewidth=1.4, label="k-NN Median"),
        mpatches.Patch(facecolor=GREY_CONE, alpha=0.4,  label="k-NN Cone (10–90th %)"),
        mpatches.Patch(facecolor=BLUE,      alpha=0.35, label="Best Match Segment"),
        mpatches.Patch(facecolor="#5050cc", alpha=0.35, label="Current Window"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=7,
               facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=WHITE)

    ax1.set_title(
        f"HUT — Analog Pattern Forecast  |  Method: {config['SIMILARITY_METHOD'].capitalize()}  |  k={config['K']}",
        color=WHITE, fontsize=10, pad=8,
    )
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))

    # Set y-axis to show readable intervals (every $5)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(5))

    # ── Volume panel ──────────────────────────────────────────────────────────
    colors = [
        GREEN if c >= o else "#e05050"
        for c, o in zip(df_display["Close"].values, df_display["Open"].values)
    ]
    # Highlight matched segment in blue
    if ms_idx >= 0 and me_idx < len(df_display):
        for i in range(max(0, ms_idx), min(len(df_display), me_idx + 1)):
            colors[i] = BLUE

    ax2.bar(history_x, df_display["Volume"].values, color=colors, width=0.8, alpha=0.7)
    ax2.set_ylabel("Volume", color=WHITE, fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))

    # Shared x-axis tick labels (show dates at regular intervals)
    n_ticks = 8
    tick_positions = np.linspace(0, len(df_display) - 1, n_ticks, dtype=int)
    # Get dates from the original full dataframe
    df_display_orig = df.iloc[-lookback_days:]
    tick_labels    = [str(df_display_orig.index[i].date()) for i in tick_positions]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7, color=WHITE)
    ax1.set_xticks([])

    plt.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"> Plot saved to {save_path}")
