#!/usr/bin/env python3
"""
Robust test script to verify API calls and image creation for both ensemble scripts.
Creates the same images as the main scripts and saves CSV output for debugging.
"""

from __future__ import annotations

from datetime import timedelta, timezone

from io import StringIO

from pathlib import Path

import re

import matplotlib.colors as mcolors

import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter, DayLocator

from matplotlib.gridspec import GridSpec

import numpy as np

import pandas as pd

import requests

import seaborn as sns

from zoneinfo import ZoneInfo

# -----------------------------------------------------------------------------
# Matplotlib configuration
# -----------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.4,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "text.color": "#333333",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "grid.color": "#cccccc",
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

PROFESSIONAL_COLORS = {
    "control": "#1f77b4",        # Professional blue
    "mean": "#d62728",           # Professional red
    "median": "#2ca02c",         # Professional green
    "ensemble": "#7f7f7f",       # Neutral gray
    "ensemble_plume": "#B3D9FF", # Light blue for accumulation plumes
    "p25_75": "#ff7f0e",         # Orange
    "p10_90": "#ffbb78",         # Light orange
    "boxplot": "#4a4a4a",        # Dark gray
    "boxplot_edge": "#2a2a2a",   # Darker gray
}

# Location / ensemble settings
LATITUDE = 39.56406
LONGITUDE = -105.93513
ELEVATION_M = 3377.4
FORECAST_DAYS = 15
SLR_RATIO = 12.0  # Snow-to-liquid ratio (12:1)
LOCAL_TZ = ZoneInfo("America/Denver")

# Output directories
TEST_OUTPUT_DIR = Path("test_output")
TEST_CSV_DIR = TEST_OUTPUT_DIR / "csv_data"
TEST_IMAGES_DIR = TEST_OUTPUT_DIR / "images"

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def format_location_metadata(location_name: str, elevation_m: float) -> str:
    """Return formatted location string with elevation in feet."""
    elevation_ft = elevation_m * 3.28084
    return f"{location_name}, {elevation_ft:,.0f} ft"

def get_init_times(forecast_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get initialization time in both local and UTC.

    Args:
        forecast_df: DataFrame with tz-aware 'time' column.

    Returns:
        (init_local, init_utc)
    """
    init_local = forecast_df["time"].min()
    init_utc = init_local.astimezone(timezone.utc)
    return init_local, init_utc

def fetch_ensemble_data(
    latitude: float,
    longitude: float,
    models: list[str],
    forecast_days: int = FORECAST_DAYS,
) -> str:
    """Fetch ensemble data from Open-Meteo API using hourly precipitation."""
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "models": models,
        "hourly": "precipitation",
        "forecast_days": forecast_days,
        "precipitation_unit": "inch",
        "timezone": "America/Denver",
        "format": "csv",
        "elevation": ELEVATION_M,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.text

def parse_ensemble_csv(csv_data: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Parse CSV data into metadata and forecast dataframes."""
    sections = csv_data.split("\n\n")
    if len(sections) < 2:
        print(
            "Warning: CSV does not have the expected format with metadata and "
            "forecast sections"
        )
        return None, None

    metadata_df = pd.read_csv(StringIO(sections[0]), delimiter=",")
    forecast_lines = sections[1].strip().split("\n")
    headers = forecast_lines[0].split(",")
    data_rows = [
        line.split(",")
        for line in forecast_lines[1:]
        if line and len(line.split(",")) == len(headers)
    ]

    forecast_df = pd.DataFrame(data_rows, columns=headers)
    forecast_df["time"] = pd.to_datetime(forecast_df["time"])

    # Make 'time' tz-aware (local timezone from metadata or default to LOCAL_TZ)
    tz_name = metadata_df["timezone"].iat[0] if "timezone" in metadata_df.columns else None
    tz = ZoneInfo(tz_name) if tz_name else LOCAL_TZ
    forecast_df["time"] = forecast_df["time"].dt.tz_localize(tz)

    numeric_cols = [col for col in forecast_df.columns if col != "time"]
    for col in numeric_cols:
        forecast_df[col] = pd.to_numeric(forecast_df[col], errors="coerce")

    return metadata_df, forecast_df

def get_snowfall_columns(df: pd.DataFrame) -> tuple[list[str], str | None]:
    """
    Extract ensemble member and control columns from snowfall dataframe.

    Returns:
        (ensemble_cols, control_col)
    """
    pattern = re.compile(r"snowfall_member\d+")
    ensemble_cols = [c for c in df.columns if pattern.match(c)]
    control_candidates = [
        c
        for c in df.columns
        if "snowfall" in c.lower() and "member" not in c.lower() and c != "time"
    ]
    control_col = control_candidates[0] if control_candidates else None
    return ensemble_cols, control_col

def get_last_valid_time(forecast_df: pd.DataFrame) -> pd.Timestamp | None:
    """
    Find the last timestamp where at least one snowfall column has valid data.

    Args:
        forecast_df: DataFrame with 'time' column and snowfall columns.

    Returns:
        Last valid timestamp or None if no data is found.
    """
    ensemble_cols, control_col = get_snowfall_columns(forecast_df)
    all_data_cols = ensemble_cols + ([control_col] if control_col else [])

    if not all_data_cols:
        return None

    # Walk backward until we find a row with at least one non-NaN value.
    for idx in range(len(forecast_df) - 1, -1, -1):
        row_data = forecast_df.iloc[idx][all_data_cols]
        if not row_data.isna().all():
            return forecast_df.iloc[idx]["time"]

    return None

def convert_precipitation_to_snowfall(df: pd.DataFrame, model_prefix: str = "") -> pd.DataFrame:
    """Convert precipitation (liquid inches) to snowfall (inches) using SLR_RATIO."""
    precip_cols: list[str] = []
    precip_det_cols: list[str] = []

    for col in df.columns:
        if col == "time":
            continue
        name_lower = col.lower()
        if "precipitation_member" in name_lower:
            precip_cols.append(col)
        elif name_lower.startswith("precipitation") or (
            model_prefix and col.startswith(model_prefix) and "precipitation" in name_lower
        ):
            precip_det_cols.append(col)

    for col in precip_cols + precip_det_cols:
        if col in df.columns:
            snowfall_col = (
                col.replace("precipitation", "snowfall")
                .replace("Precipitation", "Snowfall")
            )
            df[snowfall_col] = df[col] * SLR_RATIO

    return df

def calculate_ensemble_stats(df: pd.DataFrame, model_prefix: str = "") -> pd.DataFrame | None:
    """Calculate ensemble statistics from snowfall member data."""
    ensemble_cols = [col for col in df.columns if "snowfall_member" in col.lower()]

    if not ensemble_cols:
        print("No ensemble member columns found")
        return None

    stats_df = pd.DataFrame({"time": df["time"]})
    det_cols = [
        col
        for col in df.columns
        if col.lower().startswith("snowfall") and "member" not in col.lower()
    ]
    if det_cols:
        stats_df["forecast"] = df[det_cols[0]]

    stats_df["mean"] = df[ensemble_cols].mean(axis=1)
    stats_df["median"] = df[ensemble_cols].median(axis=1)
    stats_df["min"] = df[ensemble_cols].min(axis=1)
    stats_df["max"] = df[ensemble_cols].max(axis=1)
    stats_df["p10"] = df[ensemble_cols].quantile(0.1, axis=1)
    stats_df["p25"] = df[ensemble_cols].quantile(0.25, axis=1)
    stats_df["p75"] = df[ensemble_cols].quantile(0.75, axis=1)
    stats_df["p90"] = df[ensemble_cols].quantile(0.9, axis=1)

    return stats_df

def extract_model_data(forecast_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Extract data for a specific model from combined forecast dataframe.

    Column format: precipitation_member01_ecmwf_aifs025 (inch)
                   precipitation_ecmwf_aifs025 (inch)
    """
    model_cols = ["time"]

    for col in forecast_df.columns:
        if col == "time":
            continue
        if model_name in col:
            model_cols.append(col)

    model_df = forecast_df[model_cols].copy()

    # Strip model suffix from column names for easier processing.
    rename_dict: dict[str, str] = {}
    for col in model_df.columns:
        if col == "time":
            continue
        new_col = col.replace(f"_{model_name}", "").replace(f"{model_name}_", "")
        rename_dict[col] = new_col

    model_df.rename(columns=rename_dict, inplace=True)

    return model_df

# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def _format_title(
    model_display: str,
    init_local: pd.Timestamp,
    variable_desc: str,
) -> tuple[str, str]:
    """Build the two-line title for a figure."""
    tz_abbr = init_local.strftime("%Z") if init_local.tzinfo else "MST"
    init_str = f"{init_local:%d %b %Y %H}{tz_abbr}"
    location_name = "North Peak (Keystone, CO)"
    location_metadata = format_location_metadata(location_name, ELEVATION_M)
    title_line1 = f"{model_display} – Collected {init_str} · {variable_desc}"
    title_line2 = f"{location_metadata} · SLR {int(SLR_RATIO)}:1"
    return title_line1, title_line2

def _add_footer(fig: plt.Figure, init_utc: pd.Timestamp) -> None:
    """Add a standardized data-source footer to a figure."""
    init_time_str = init_utc.strftime("%d %b %Y %H:%M UTC")
    fig.text(
        0.99,
        0.01,
        f"Data: Open-Meteo ensemble API, collected {init_time_str}",
        ha="right",
        va="bottom",
        fontsize=8,
        style="italic",
        color="#666666",
    )

# -----------------------------------------------------------------------------
# Plots: hourly and cumulative
# -----------------------------------------------------------------------------

def create_hourly_plot(
    forecast_df: pd.DataFrame,
    ensemble_stats: pd.DataFrame,
    model_name: str,
    model_display: str,
) -> bool:
    """Create hourly snowfall rate plot."""
    try:
        fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
        ensemble_cols, control_col = get_snowfall_columns(forecast_df)

        # Ensemble members (light grey)
        for col in ensemble_cols:
            ax.plot(
                forecast_df["time"],
                forecast_df[col],
                color=PROFESSIONAL_COLORS["ensemble"],
                alpha=0.08,
                linewidth=0.5,
                zorder=1,
            )

        # Control member (if present)
        if control_col:
            ax.plot(
                forecast_df["time"],
                forecast_df[control_col],
                color=PROFESSIONAL_COLORS["control"],
                linestyle="-",
                linewidth=2.5,
                label="Control",
                zorder=4,
            )

        # Ensemble mean
        ax.plot(
            ensemble_stats["time"],
            ensemble_stats["mean"],
            color=PROFESSIONAL_COLORS["mean"],
            linestyle="--",
            linewidth=2.5,
            label="Ensemble Mean",
            zorder=5,
        )

        # Percentile bands
        ax.fill_between(
            ensemble_stats["time"],
            ensemble_stats["p25"],
            ensemble_stats["p75"],
            color=PROFESSIONAL_COLORS["p25_75"],
            alpha=0.25,
            label="25–75th Percentile",
            zorder=2,
        )
        ax.fill_between(
            ensemble_stats["time"],
            ensemble_stats["p10"],
            ensemble_stats["p90"],
            color=PROFESSIONAL_COLORS["p10_90"],
            alpha=0.15,
            label="10–90th Percentile",
            zorder=2,
        )

        init_local, init_utc = get_init_times(forecast_df)
        title_line1, title_line2 = _format_title(
            model_display, init_local, "Hourly Snowfall Rate (inches)"
        )

        fig.suptitle(
            f"{title_line1}\n{title_line2}",
            fontsize=14,
            fontweight="bold",
            x=0.05,
            y=0.98,
            ha="left",
            va="top",
        )

        ax.set_ylabel("Snowfall (inches)", fontsize=10)
        ax.set_xlabel("", fontsize=10)

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            color="#999999",
            zorder=0,
        )
        ax.set_axisbelow(True)

        ax.legend(
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=False,
            fontsize=9,
            framealpha=0.95,
            edgecolor="#cccccc",
        )

        ax.xaxis.set_major_formatter(DateFormatter("%b %d\n%H:00"))
        ax.xaxis.set_major_locator(DayLocator(interval=1))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)

        ax.relim()
        ax.autoscale(axis="y", tight=True)
        _, ymax = ax.get_ylim()
        ax.set_ylim(bottom=0.0, top=ymax * 1.1 if ymax > 0 else 1.2)

        last_valid_time = get_last_valid_time(forecast_df)
        if last_valid_time is not None:
            padding = timedelta(hours=6)
            ax.set_xlim(left=forecast_df["time"].min(), right=last_valid_time + padding)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("#333333")

        _add_footer(fig, init_utc)

        filename = TEST_IMAGES_DIR / f"test_{model_name.replace('.', '_')}_hourly.png"
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"✓ Hourly plot saved: {filename}")
        return True
    except Exception as exc:  # debug aid
        print(f"✗ Hourly plot creation failed: {exc}")
        import traceback

        traceback.print_exc()
        return False

def create_cumulative_plot(
    forecast_df: pd.DataFrame,
    model_name: str,
    model_display: str,
) -> bool:
    """Create cumulative snowfall plot."""
    try:
        fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
        ensemble_cols, control_col = get_snowfall_columns(forecast_df)

        # Accumulate ensemble members
        accumulated = pd.concat(
            [forecast_df[col].cumsum().rename(f"{col}_acc") for col in ensemble_cols],
            axis=1,
        )
        forecast_df = forecast_df.join(accumulated)
        forecast_df["mean_acc"] = accumulated.mean(axis=1)

        if control_col:
            forecast_df["control_acc"] = forecast_df[control_col].cumsum()

        # Distinct colors for each plume
        cmap = plt.colormaps["tab20"]
        n_members = len(ensemble_cols)

        member_colors = []
        for i in range(n_members):
            base_color = cmap(i % 20)
            lightened = tuple(0.4 + 0.6 * c for c in base_color[:3])
            member_colors.append(lightened)

        for i, col in enumerate(ensemble_cols):
            ax.plot(
                forecast_df["time"],
                forecast_df[f"{col}_acc"],
                alpha=0.7,
                linewidth=1.0,
                color=member_colors[i],
                zorder=1,
            )

        if control_col:
            ax.plot(
                forecast_df["time"],
                forecast_df["control_acc"],
                color=PROFESSIONAL_COLORS["control"],
                linestyle="-",
                linewidth=3.0,
                label="Control",
                zorder=5,
            )

        ax.plot(
            forecast_df["time"],
            forecast_df["mean_acc"],
            color=PROFESSIONAL_COLORS["mean"],
            linestyle="--",
            linewidth=3.0,
            label="Ensemble Mean",
            zorder=4,
        )

        init_local, init_utc = get_init_times(forecast_df)
        title_line1, title_line2 = _format_title(
            model_display, init_local, "Snowfall Accumulation (inches)"
        )

        fig.suptitle(
            f"{title_line1}\n{title_line2}",
            fontsize=14,
            fontweight="bold",
            x=0.05,
            y=0.98,
            ha="left",
            va="top",
        )

        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Accumulated Snowfall (inches)", fontsize=10)

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            color="#999999",
            zorder=0,
        )
        ax.set_axisbelow(True)

        ax.legend(
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=False,
            fontsize=9,
            framealpha=0.95,
            edgecolor="#cccccc",
        )

        ax.xaxis.set_major_formatter(DateFormatter("%b %d\n%H:00"))
        ax.xaxis.set_major_locator(DayLocator(interval=1))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)

        ax.relim()
        ax.autoscale(axis="y", tight=True)
        _, ymax = ax.get_ylim()
        ax.set_ylim(bottom=0.0, top=ymax * 1.05 if ymax > 0 else 1.0)

        last_valid_time = get_last_valid_time(forecast_df)
        if last_valid_time is not None:
            padding = timedelta(hours=6)
            ax.set_xlim(left=forecast_df["time"].min(), right=last_valid_time + padding)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("#333333")

        _add_footer(fig, init_utc)

        filename = TEST_IMAGES_DIR / f"test_{model_name.replace('.', '_')}_cumulative.png"
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"✓ Cumulative plot saved: {filename}")
        return True
    except Exception as exc:  # debug aid
        print(f"✗ Cumulative plot creation failed: {exc}")
        import traceback

        traceback.print_exc()
        return False

# -----------------------------------------------------------------------------
# Heatmap + boxplot panel
# -----------------------------------------------------------------------------

def create_heatmap_panel(
    forecast_df: pd.DataFrame,
    model_name: str,
    model_display: str,
) -> bool:
    """
    Create a WeatherBell-style ensemble panel:
      - top: member vs time heatmap of 6-hour interval totals
      - bottom: boxplot with mean and control lines overlaid.
    """
    try:
        df = forecast_df.copy()

        df = df.set_index(df["time"])

        interval_hours = 6
        freq = f"{interval_hours}h"

        # Identify ensemble members and control
        member_re = re.compile(r"snowfall_member(\d+)")
        member_cols_with_idx: list[tuple[int, str]] = []

        for col in df.columns:
            match = member_re.match(col)
            if match:
                member_cols_with_idx.append((int(match.group(1)), col))

        member_cols_with_idx.sort(key=lambda x: x[0], reverse=True)
        member_cols = [c for _, c in member_cols_with_idx]

        _, control_col = get_snowfall_columns(df)

        if not member_cols:
            print("✗ Heatmap panel skipped: no snowfall_member columns found")
            return False

        last_valid_time_local = get_last_valid_time(forecast_df)

        if last_valid_time_local is not None:
            df = df[df.index <= last_valid_time_local]

        cols = member_cols + ([control_col] if control_col else [])

        interval_totals = df[cols].resample(freq, label="right", closed="right").sum()
        interval_totals = interval_totals.dropna(how="all")

        # Rename columns for display: m50, m49, ..., m01, c00
        rename: dict[str, str] = {}
        num_members = len(member_cols)

        for i, col in enumerate(member_cols):
            rename[col] = f"m{num_members - i:02d}"

        if control_col:
            rename[control_col] = "c00"

        interval_totals = interval_totals.rename(columns=rename)
        member_names = [v for v in rename.values() if v != "c00"]

        # Build heatmap matrix: rows = members/control, cols = intervals
        heatmap_data = interval_totals.transpose()

        if heatmap_data.empty or heatmap_data.isna().all().all():
            print("✗ Heatmap panel skipped: no valid data after grouping")
            return False

        interval_index = heatmap_data.columns
        interval_local = list(interval_index)
        interval_labels = [ts.strftime("%d-%H") for ts in interval_local]

        n_intervals = len(interval_index)

        # Avoid seaborn adding "time" as x-axis label
        heatmap_data.columns.name = None

        # Color configuration
        bounds = np.array(
            [
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,  # many bins < 1"
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.5,
                5.0,
                5.5,  # 1–5.5"
                5.75,
                6.00,
                6.25,
                6.50,
                6.75,
                7.00,  # 5.5–7"
                8.0,
                9.0,
                10.0,  # 7–10"
            ]
        )

        n_bins = len(bounds) - 1

        from matplotlib.colors import LinearSegmentedColormap, ListedColormap

        n_grey = 10
        n_blue = 9
        n_red = 6 + 3

        grey_ramp = LinearSegmentedColormap.from_list(
            "grey_ramp", ["#f5f5f5", "#8f8f8f"], N=n_grey
        )(np.linspace(0, 1, n_grey))

        blue_ramp = LinearSegmentedColormap.from_list(
            "blue_ramp", ["#e3f2fd", "#1976d2"], N=n_blue
        )(np.linspace(0, 1, n_blue))

        red_ramp = LinearSegmentedColormap.from_list(
            "red_ramp", ["#ffcdd2", "#b71c1c"], N=n_red
        )(np.linspace(0, 1, n_red))

        colors = np.vstack([grey_ramp, blue_ramp, red_ramp])
        assert colors.shape[0] == n_bins, "colors and bounds must align"

        cmap = ListedColormap(colors, name="snowfall_discrete")
        norm = mcolors.BoundaryNorm(bounds, ncolors=n_bins, clip=True)

        cbar_ticks = [0, 1, 3, 5.5, 7, 10]

        mask = (heatmap_data == 0) | (heatmap_data.isna())

        def format_inches(val: float) -> str:
            if pd.isna(val) or val == 0:
                return ""
            return f"{round(val):.0f}"

        annot_data = heatmap_data.map(format_inches)
        annot_data = annot_data.where(~mask, "")

        def get_text_color(val: float) -> str:
            if pd.isna(val) or val == 0:
                return ""
            clamped_val = min(max(val, 0.0), 10.0)
            rgba = cmap(norm(clamped_val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            return "white" if luminance < 0.5 else "black"

        text_colors = heatmap_data.map(get_text_color)
        text_colors = text_colors.where(~mask, "")

        # Stats for boxplot panel
        members_only = (
            interval_totals[member_names]
            if "c00" in interval_totals.columns
            else interval_totals
        )
        mean_int = members_only.mean(axis=1)
        control_int = interval_totals["c00"] if "c00" in interval_totals.columns else None

        # Layout
        fig = plt.figure(figsize=(16, 10), facecolor="white")
        gs = GridSpec(
            2,
            2,
            width_ratios=[20, 0.5],
            height_ratios=[3, 1],
            wspace=0.05,
            hspace=0.07,
        )
        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)

        ax_heat = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[:, 1])
        ax_ts = fig.add_subplot(gs[1, 0])

        init_local, init_utc = get_init_times(forecast_df)
        title_line1, title_line2 = _format_title(
            model_display, init_local, f"{interval_hours}-Hour Snowfall (inches)"
        )

        fig.suptitle(
            f"{title_line1}\n{title_line2}",
            x=0.02,
            y=0.97,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="#1a1a1a",
        )

        has_gt10 = np.nanmax(interval_totals.values) > 10.0

        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            norm=norm,
            annot=False,
            fmt="",
            mask=mask,
            cbar_ax=cax,
            cbar_kws={
                "label": "Snowfall (inches)",
                "shrink": 1.0,
                "extend": "max" if has_gt10 else "neither",
            },
            linewidths=0.15,
            linecolor="white",
            ax=ax_heat,
        )

        ax_heat.set_xlabel("")

        # Add numeric annotations
        for i, row_idx in enumerate(heatmap_data.index):
            for j, col_idx in enumerate(heatmap_data.columns):
                val = heatmap_data.loc[row_idx, col_idx]
                if pd.isna(val) or val == 0 or mask.loc[row_idx, col_idx]:
                    continue

                text = annot_data.loc[row_idx, col_idx]
                if not text:
                    continue

                text_color = text_colors.loc[row_idx, col_idx]
                ax_heat.text(
                    j + 0.5,
                    i + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                    weight="bold",
                )

        cbar = ax_heat.collections[0].colorbar
        cbar.set_ticks(cbar_ticks)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Snowfall (inches)", fontsize=10)

        ax_heat.set_yticks(np.arange(0.5, len(heatmap_data.index), 1))
        ax_heat.set_yticklabels(heatmap_data.index, rotation=0, fontsize=7)

        # Hour ticks at interval ends
        end_positions = np.arange(1, len(interval_index) + 1)

        major_pos: list[float] = []
        major_labels: list[str] = []
        minor_pos: list[float] = []

        for x_pos, ts_local in zip(end_positions, interval_local):
            h = ts_local.hour
            if h in (0, 12):
                major_pos.append(x_pos)
                major_labels.append(ts_local.strftime("%H"))
            elif h in (6, 18):
                minor_pos.append(x_pos)

        ax_heat.set_xticks(major_pos)
        ax_heat.set_xticklabels(major_labels, fontsize=8, rotation=0)
        ax_heat.set_xticks(minor_pos, minor=True)
        ax_heat.tick_params(axis="x", which="minor", length=3, width=0.6)
        ax_heat.tick_params(axis="x", which="major", length=4, width=0.9)

        ax_heat.set_xlim(0, n_intervals)

        for spine in ax_heat.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("#333333")

        # Bottom panel: boxplot + lines
        # Reset index first, then get the index column name
        df_for_melt = interval_totals[member_names].reset_index()
        # The index column name after reset_index is the original index name (or 'index' if unnamed)
        index_col_name = df_for_melt.columns[0]  # First column is the index
        
        melted = df_for_melt.melt(
            id_vars=index_col_name,
            value_vars=member_names,
            var_name="Member",
            value_name="Snowfall",
        )
        melted.rename(columns={index_col_name: "interval"}, inplace=True)
        melted.dropna(subset=["Snowfall"], inplace=True)

        boxplot_data: list[np.ndarray] = []
        positions = np.arange(1, n_intervals + 1)

        for interval_dt in interval_totals.index:
            interval_data = melted[melted["interval"] == interval_dt]["Snowfall"].values
            boxplot_data.append(interval_data if len(interval_data) > 0 else [])

        ax_ts.boxplot(
            boxplot_data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(
                facecolor=PROFESSIONAL_COLORS["boxplot"],
                edgecolor=PROFESSIONAL_COLORS["boxplot_edge"],
                linewidth=0.8,
                alpha=0.7,
            ),
            whiskerprops=dict(
                color=PROFESSIONAL_COLORS["boxplot_edge"], linewidth=0.8
            ),
            capprops=dict(color=PROFESSIONAL_COLORS["boxplot_edge"], linewidth=0.8),
            medianprops=dict(color="white", linewidth=1.2),
        )

        if control_int is not None:
            ax_ts.plot(
                positions,
                control_int.values,
                color=PROFESSIONAL_COLORS["control"],
                linestyle="-",
                linewidth=2,
                label="Control",
                zorder=5,
            )

        ax_ts.plot(
            positions,
            mean_int.values,
            color=PROFESSIONAL_COLORS["mean"],
            linestyle="--",
            linewidth=2,
            label="Ensemble Mean",
            zorder=5,
        )

        ax_ts.set_xlim(0, n_intervals)

        # Date ticks at 00 local
        day_ticks: list[float] = []
        day_labels: list[str] = []
        seen_dates: set = set()

        for pos, ts_local in zip(positions, interval_local):
            d = ts_local.date()
            if ts_local.hour == 0 and d not in seen_dates:
                seen_dates.add(d)
                day_ticks.append(pos)
                day_labels.append(ts_local.strftime("%d %b"))

        ax_ts.set_xticks(day_ticks)
        ax_ts.set_xticklabels(day_labels, fontsize=9)
        ax_ts.set_xlabel("")

        # Twin axis for 6-hour ticks on top, aligned with heatmap
        # Only show tick marks, no labels (labels are on heatmap x-axis)
        ax_ts_top = ax_ts.twiny()
        ax_ts_top.set_xlim(0, n_intervals)

        ax_ts_top.set_xticks(major_pos)
        ax_ts_top.set_xticklabels([''] * len(major_pos))  # No text, just ticks

        ax_ts_top.set_xticks(minor_pos, minor=True)

        ax_ts_top.tick_params(
            axis="x",
            which="major",
            length=4,
            width=0.9,
            direction="out",
            bottom=False,
            top=True,
            labelbottom=False,
            labeltop=False,  # No labels on top axis
        )

        ax_ts_top.tick_params(
            axis="x",
            which="minor",
            length=3,
            width=0.6,
            direction="out",
            bottom=False,
            top=True,
            labelbottom=False,
            labeltop=False,
        )

        ax_ts_top.set_xlabel("")

        for side, spine in ax_ts_top.spines.items():
            if side != "top":
                spine.set_visible(False)

        ax_ts.set_ylabel("Snowfall (inches)", fontsize=10)
        ax_ts.tick_params(axis="y", labelsize=9)

        ax_ts.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, color="#cccccc")
        ax_ts.set_axisbelow(True)

        for xi in range(n_intervals + 1):
            ax_ts.axvline(x=xi, color="lightgray", linestyle=":", linewidth=0.4, alpha=0.4)

        for tick in day_ticks:
            ax_ts.axvline(x=tick, color="#999999", linestyle="-", linewidth=0.8, alpha=0.7)

        ax_ts.legend(
            loc="upper right",
            frameon=True,
            fancybox=True,
            shadow=False,
            fontsize=9,
            framealpha=0.95,
            edgecolor="#cccccc",
        )

        for spine in ax_ts.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("#333333")

        ax_ts.relim()
        ax_ts.autoscale(axis="y", tight=True)
        ymin, ymax = ax_ts.get_ylim()
        ymax_padded = ymax * 1.05 if ymax > 0 else 1.0
        ax_ts.set_ylim(bottom=0.0, top=ymax_padded)

        # Local collection time footer (matches your original text)
        tz_abbr = init_local.strftime("%Z") if init_local.tzinfo else "MST"
        init_time_str = init_local.strftime("%d %b %Y %H:%M")
        fig.text(
            0.99,
            0.01,
            f"Data: Open-Meteo ensemble API, collected {init_time_str} {tz_abbr}. "
            "All data in local time.",
            ha="right",
            va="bottom",
            fontsize=8,
            style="italic",
            color="#666666",
        )

        filename = TEST_IMAGES_DIR / f"test_{model_name.replace('.', '_')}_panel.png"
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"✓ Heatmap panel saved: {filename}")
        return True
    except Exception as exc:  # debug aid
        print(f"✗ Heatmap panel creation failed: {exc}")
        import traceback

        traceback.print_exc()
        return False

# -----------------------------------------------------------------------------
# Test orchestration
# -----------------------------------------------------------------------------

def test_api_call_and_save_csv() -> tuple[dict[str, pd.DataFrame] | None, bool]:
    """Test API call for both models and save raw CSV data for debugging."""
    print("\n" + "=" * 70)
    print("Testing API call for both models (precipitation)")
    print("=" * 70)

    try:
        models = ["ecmwf_aifs025", "ecmwf_ifs025"]
        csv_data = fetch_ensemble_data(LATITUDE, LONGITUDE, models)
        print("✓ API call successful")

        csv_file = TEST_CSV_DIR / "combined_raw.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        csv_file.write_text(csv_data)
        print(f"✓ Raw CSV saved: {csv_file}")

        metadata_df, forecast_df = parse_ensemble_csv(csv_data)

        if metadata_df is None or forecast_df is None:
            print("✗ Failed to parse CSV data")
            return None, False

        metadata_file = TEST_CSV_DIR / "combined_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        print(f"✓ Metadata saved: {metadata_file}")

        forecast_file = TEST_CSV_DIR / "combined_forecast.csv"
        forecast_df.to_csv(forecast_file, index=False)
        print(f"✓ Forecast data saved: {forecast_file}")

        model_data: dict[str, pd.DataFrame] = {}

        for model_name in models:
            model_df = extract_model_data(forecast_df, model_name)
            model_df = convert_precipitation_to_snowfall(model_df)

            model_file = TEST_CSV_DIR / f"{model_name.replace('.', '_')}_forecast.csv"
            model_df.to_csv(model_file, index=False)
            print(f"✓ {model_name} forecast saved: {model_file}")

            model_data[model_name] = model_df

        return model_data, True
    except requests.exceptions.RequestException as exc:
        print(f"✗ API request failed: {exc}")
        return None, False
    except Exception as exc:
        print(f"✗ Error: {exc}")
        import traceback

        traceback.print_exc()
        return None, False

def test_model(
    forecast_df: pd.DataFrame | None,
    model_name: str,
    model_display: str,
) -> dict[str, bool]:
    """Run all plots/tests for a single model."""
    print("\n" + "#" * 70)
    print(f"Testing Model: {model_display} ({model_name})")
    print("#" * 70)

    results: dict[str, bool] = {}

    if forecast_df is None:
        print(f"\n✗ Skipping image creation for {model_name} (no data)")
        return results

    ensemble_cols, control_col = get_snowfall_columns(forecast_df)

    print("\nData Analysis:")
    print(f"  - Rows: {len(forecast_df)}")
    print(f"  - Ensemble members: {len(ensemble_cols)}")
    print(f"  - Control column: {control_col if control_col else 'None'}")
    print(f"  - Time range: {forecast_df['time'].min()} to {forecast_df['time'].max()}")

    if ensemble_cols:
        sample_col = ensemble_cols[0]
        sample_data = forecast_df[sample_col]
        nan_count = sample_data.isna().sum()
        total_count = len(sample_data)
        has_data = not sample_data.isna().all()

        print(f"  - Sample column ({sample_col}):")
        print(f"    - NaN values: {nan_count}/{total_count} ({100 * nan_count / total_count:.1f}%)")
        print(f"    - Has valid data: {has_data}")

        if has_data:
            print(
                f"    - Min: {sample_data.min():.3f}, "
                f"Max: {sample_data.max():.3f}, "
                f"Mean: {sample_data.mean():.3f}"
            )

    ensemble_stats = calculate_ensemble_stats(forecast_df)
    if ensemble_stats is None:
        print(f"\n✗ Skipping image creation for {model_name} (stats calculation failed)")
        return results

    print(f"\nCreating images for {model_name}...")
    results["hourly"] = create_hourly_plot(forecast_df, ensemble_stats, model_name, model_display)
    results["cumulative"] = create_cumulative_plot(forecast_df, model_name, model_display)
    results["heatmap"] = create_heatmap_panel(forecast_df, model_name, model_display)

    return results

def main() -> int:
    """Run tests for both ensemble models."""
    print("=" * 70)
    print("Ensemble Script Test Suite - Robust Version")
    print("Using precipitation with 12:1 SLR conversion to snowfall")
    print("=" * 70)
    print(f"Output directory: {TEST_OUTPUT_DIR.absolute()}")
    print(f"CSV data: {TEST_CSV_DIR.absolute()}")
    print(f"Images: {TEST_IMAGES_DIR.absolute()}")

    model_data, api_success = test_api_call_and_save_csv()

    if not api_success or model_data is None:
        print("✗ API call failed - cannot proceed with image creation")
        return 1

    models = [
        ("ecmwf_ifs025", "ECMWF IFS 0.25°"),
        ("ecmwf_aifs025", "ECMWF AIFS 0.25°"),
    ]

    all_results: dict[str, dict[str, bool]] = {}

    for model_name, model_display in models:
        forecast_df = model_data.get(model_name)
        all_results[model_name] = test_model(forecast_df, model_name, model_display)

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        for test_name, success in results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  {status}: {test_name}")

    total_tests = sum(len(r) for r in all_results.values())
    total_passed = sum(sum(r.values()) for r in all_results.values())

    print("\n" + "=" * 70)
    print(f"Overall: {total_passed}/{total_tests} tests passed")
    print("=" * 70)

    if total_passed == total_tests:
        print("✓ All tests passed!")
        return 0

    print(
        "✗ Some tests failed - check CSV files in test_output/csv_data/ for details"
    )
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
