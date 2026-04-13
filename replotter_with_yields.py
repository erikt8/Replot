#!/usr/bin/env python3
"""
Plot precomputed UMAP coordinates colored by yields (or other property).

in terminal run:
pip install numpy pandas matplotlib


Add in input and output file paths below
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Replot UMAP data with custom colors/gradients
# Edit this
light = "#c8ddf0"
mid   = "#4c87c7"
dark  = "#0b2c6b"

cmap = LinearSegmentedColormap.from_list(
    "gradient",
    [light, mid, dark],
    N=256
)


def scale_point_size(y):
    """
    Map yield values to point sizes.
    - 0–30 → 20–40
    - 30–70 → 40–80
    - 70–100 → 80–140

    Delete this if you want all points to be the same size
    """
    y = np.clip(y, 0, 100)


    return 20 + (y / 100) * 120


def plot_umap_with_yield(df: pd.DataFrame,
                         yield_col: str,
                         output_dir: str,
                         vmin: float = 0.0,
                         vmax: float = 100.0) -> None:

    y_numeric = pd.to_numeric(df[yield_col], errors="coerce")
    fig, ax = plt.subplots(figsize=(7.5, 7))

    ax.scatter(
        df["UMAP1"],
        df["UMAP2"],
        color="#9b9898",     # darker neutral grey
        s=35,                # larger background size
        alpha=0.4,
        edgecolors="none",
        zorder=1,
    )
    mask = y_numeric.notna()

    if mask.any():

        sizes = scale_point_size(y_numeric[mask])

        sc = ax.scatter(
            df.loc[mask, "UMAP1"],
            df.loc[mask, "UMAP2"],
            c=y_numeric[mask],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=sizes,
            edgecolors="k",
            linewidths=0.15,
            zorder=3,
        )

        cbar = plt.colorbar(sc, ax=ax)
        '''cbar.set_label(f"{yield_col} (%)")'''

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"UMAP colored by {yield_col}")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    safe_name = yield_col.replace(" ", "_")
    out_path = os.path.join(output_dir, f"umap_{safe_name}.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    #Add in proper .csv input file path
    csv_path = "/Users/eriktrebilcock/Downloads/md_michael_acceptor_UMAP_coords_yields.csv"
    #Edit output file path and file name
    output_dir = "/Users/eriktrebilcock/Downloads/umap_yield_plots"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at:\n{csv_path}")

    df = pd.read_csv(csv_path)
    print("CSV columns detected:", df.columns.tolist())

    required_cols = ["UMAP1", "UMAP2", "Yield"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    for col in ["Yield"]:
        plot_umap_with_yield(df, yield_col=col, output_dir=output_dir)


if __name__ == "__main__":
    main()