#!/usr/bin/env python3
"""
Plot precomputed UMAP/PCA coordinates colored by yield (or another numeric property).

Examples:
    python3 replotter_with_yields.py
    python3 replotter_with_yields.py --algorithm PCA --value-column Yield
"""

import argparse
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CSV_PATH = "/Users/eriktrebilcock/Downloads/md_michael_acceptor_UMAP_coords_yields.csv"
OUTPUT_DIR = "/Users/eriktrebilcock/Downloads/umap_yield_plots"
ALGORITHM = "UMAP"
VALUE_COLUMNS = ["Yield"]

LIGHT = "#c8ddf0"
MID = "#4c87c7"
DARK = "#0b2c6b"

CMAP = LinearSegmentedColormap.from_list("gradient", [LIGHT, MID, DARK], N=256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot precomputed coordinates from a CSV and color points by yield/property."
    )
    parser.add_argument("--csv", default=CSV_PATH, help="Input CSV file.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for PNG outputs.")
    parser.add_argument(
        "--algorithm",
        default=ALGORITHM,
        help="Preferred coordinate prefix, e.g. UMAP, PCA, or TSNE.",
    )
    parser.add_argument(
        "--value-column",
        action="append",
        dest="value_columns",
        help="Numeric column to color by. Can be supplied more than once.",
    )
    parser.add_argument("--vmin", type=float, default=0.0, help="Minimum color scale value.")
    parser.add_argument("--vmax", type=float, default=100.0, help="Maximum color scale value.")
    return parser.parse_args()


def detect_coordinate_columns(df: pd.DataFrame, algorithm: str) -> tuple[str, str]:
    algorithm = algorithm.upper()
    candidate_pairs = [
        (f"{algorithm}1", f"{algorithm}2"),
        ("UMAP1", "UMAP2"),
        ("PCA1", "PCA2"),
        ("TSNE1", "TSNE2"),
    ]

    for col1, col2 in candidate_pairs:
        if col1 in df.columns and col2 in df.columns:
            return col1, col2

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]

    raise ValueError(
        "Could not find two coordinate columns. Expected columns such as "
        f"'{algorithm}1'/'{algorithm}2', 'UMAP1'/'UMAP2', or at least two numeric columns."
    )


def scale_point_size(values: pd.Series) -> pd.Series:
    clipped = np.clip(values, 0, 100)
    return 20 + (clipped / 100) * 120


def plot_with_value(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    output_dir: str,
    vmin: float,
    vmax: float,
) -> None:
    value_numeric = pd.to_numeric(df[value_col], errors="coerce")
    coord_mask = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    mask = coord_mask & value_numeric.notna()

    if not coord_mask.any():
        raise ValueError(f"No valid coordinate rows were found for columns '{x_col}' and '{y_col}'.")

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(
        pd.to_numeric(df.loc[coord_mask, x_col], errors="coerce"),
        pd.to_numeric(df.loc[coord_mask, y_col], errors="coerce"),
        color="#9b9898",
        s=35,
        alpha=0.4,
        edgecolors="none",
        zorder=1,
    )

    if mask.any():
        sizes = scale_point_size(value_numeric.loc[mask])
        sc = ax.scatter(
            pd.to_numeric(df.loc[mask, x_col], errors="coerce"),
            pd.to_numeric(df.loc[mask, y_col], errors="coerce"),
            c=value_numeric.loc[mask],
            cmap=CMAP,
            vmin=vmin,
            vmax=vmax,
            s=sizes,
            edgecolors="k",
            linewidths=0.15,
            zorder=3,
        )
        plt.colorbar(sc, ax=ax)
    else:
        print(f"Skipping colored overlay for '{value_col}' because no numeric values were found.")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col}/{y_col} colored by {value_col}")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    safe_name = value_col.replace(" ", "_")
    out_path = os.path.join(output_dir, f"{x_col}_{y_col}_{safe_name}.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found at:\n{args.csv}")

    df = pd.read_csv(args.csv)
    print("CSV columns detected:", df.columns.tolist())

    x_col, y_col = detect_coordinate_columns(df, args.algorithm)
    print(f"Using coordinate columns: {x_col}, {y_col}")

    value_columns = args.value_columns or VALUE_COLUMNS
    missing = [col for col in value_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required value columns in CSV: {missing}")

    for col in value_columns:
        plot_with_value(df, x_col, y_col, col, args.output_dir, args.vmin, args.vmax)


if __name__ == "__main__":
    main()
