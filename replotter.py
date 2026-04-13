#!/usr/bin/env python3

"""
Plot precomputed 2D coordinates from a CSV file and color points by cluster.

Examples:
    python3 replotter.py
    python3 replotter.py --csv /path/to/data.csv --algorithm PCA --palette Set2
    python3 replotter.py --palette '#1b9e77,#d95f02,#7570b3'
"""

import argparse
import os

import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import get_cmap

# Default settings. These can also be overridden with command-line arguments.
CSV_PATH = "/Users/eriktrebilcock/Downloads/md_michael_acceptor_UMAP_coords.csv"
PNG_OUT = "/Users/eriktrebilcock/Downloads/umap_clusters_matplotlib.png"
N_CLUSTERS = 11
ALGORITHM = "UMAP" # "UMAP", or "PCA"
COLOR_PALETTE = "tab20"
FIGURE_SIZE = (6.2, 7.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot precomputed UMAP/PCA/t-SNE coordinates from a CSV file."
    )
    parser.add_argument("--csv", default=CSV_PATH, help="Input CSV file with 2D coordinates.")
    parser.add_argument("--out", default=PNG_OUT, help="Output PNG file path.")
    parser.add_argument(
        "--clusters",
        type=int,
        default=N_CLUSTERS,
        help="Number of clusters to compute with hierarchical clustering.",
    )
    parser.add_argument(
        "--algorithm",
        default=ALGORITHM,
        help="Preferred coordinate prefix, e.g. UMAP, PCA, or TSNE.",
    )
    parser.add_argument(
        "--palette",
        default=COLOR_PALETTE,
        help=(
            "Matplotlib palette name such as 'tab20', 'Set2', 'viridis', "
            "or a comma-separated list of colors such as '#1f77b4,#ff7f0e,#2ca02c'."
        ),
    )
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


def build_palette(palette_spec: str, n_colors: int) -> list:
    colors = [color.strip() for color in palette_spec.split(",") if color.strip()]
    if len(colors) > 1:
        invalid = [color for color in colors if not mcolors.is_color_like(color)]
        if invalid:
            raise ValueError(f"Invalid color values in palette list: {invalid}")
        return [colors[i % len(colors)] for i in range(n_colors)]

    cmap = get_cmap(palette_spec, n_colors)
    return [cmap(i) for i in range(n_colors)]


def add_clusters(df: pd.DataFrame, x_col: str, y_col: str, n_clusters: int) -> pd.DataFrame:
    coords = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce")
    invalid_rows = coords.isna().any(axis=1)
    if invalid_rows.any():
        dropped = int(invalid_rows.sum())
        print(f"Dropping {dropped} rows with missing or non-numeric coordinates.")
        df = df.loc[~invalid_rows].copy()
        coords = coords.loc[~invalid_rows]
    else:
        df = df.copy()

    if len(coords) == 0:
        raise ValueError("No valid coordinate rows were found in the CSV.")

    if n_clusters < 1:
        raise ValueError("N_CLUSTERS must be at least 1.")

    if len(coords) == 1:
        df["Cluster"] = 1
        return df

    effective_clusters = min(n_clusters, len(coords))
    if effective_clusters != n_clusters:
        print(
            f"Requested {n_clusters} clusters, but only {len(coords)} valid points were found. "
            f"Using {effective_clusters} clusters instead."
        )

    linkage_matrix = linkage(coords.to_numpy(), method="ward")
    df["Cluster"] = fcluster(linkage_matrix, t=effective_clusters, criterion="maxclust")
    return df


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found at:\n{args.csv}")

    df = pd.read_csv(args.csv)
    print("CSV columns:", df.columns.tolist())

    x_col, y_col = detect_coordinate_columns(df, args.algorithm)
    print(f"Using coordinate columns: {x_col}, {y_col}")

    df = add_clusters(df, x_col, y_col, args.clusters)

    with_clusters_path = args.csv.replace(".csv", "_with_clusters.csv")
    df.to_csv(with_clusters_path, index=False)
    print(f"Saved CSV with clusters to:\n{with_clusters_path}")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    clusters = sorted(df["Cluster"].unique())
    palette = build_palette(args.palette, len(clusters))

    for color, cl in zip(palette, clusters):
        mask = df["Cluster"] == cl
        ax.scatter(
            df.loc[mask, x_col],
            df.loc[mask, y_col],
            s=35,
            color=color,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.8,
            label=f"Cluster {cl}",
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{args.algorithm.upper()} Projection with {len(clusters)} Clusters")
    ax.legend(loc="best", fontsize=8, markerscale=1.0, frameon=False)
    plt.tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"Saved plot to:\n{args.out}")
    print(f"Palette used: {args.palette}")


if __name__ == "__main__":
    main()
