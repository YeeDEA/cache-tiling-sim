"""Render docs/heatmap.png from the CSV written by scripts/sweep_heatmap.sh."""
import csv
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless backend; pyplot is imported inside main() after this



def main(src: str, dst: str) -> None:
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(src, newline="")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, bits, label in zip(axes, ("64", "4"), ("fp64 (8 B)", "packed int4 (0.5 B)")):
        sub = [r for r in rows if r["bits"] == bits]
        tiles = sorted({int(r["tile"]) for r in sub})
        pads = sorted({int(r["pad"]) for r in sub})
        grid = np.zeros((len(tiles), len(pads)))
        for r in sub:
            grid[tiles.index(int(r["tile"])), pads.index(int(r["pad"]))] = float(r["miss_rate"])
        im = ax.imshow(np.log10(np.maximum(grid, 1e-4)), cmap="magma_r", aspect="auto", vmin=-2, vmax=2)  # shared scale
        for i in range(len(tiles)):
            for j in range(len(pads)):
                ax.text(j, i, f"{grid[i, j]:.3g}", ha="center", va="center", fontsize=7, color="white" if grid[i, j] > 1 else "black")
        ax.set_xticks(range(len(pads)), pads)
        ax.set_yticks(range(len(tiles)), tiles)
        ax.set_xlabel("row padding (elements)")
        ax.set_ylabel("tile T")
        ax.set_title(f"tiled matmul miss rate %, N=256, {label}")
        fig.colorbar(im, ax=ax, label="log10(miss rate %)")
    fig.suptitle("32 KB / 8-way / 64 B L1, LRU")
    fig.tight_layout()
    fig.savefig(dst, dpi=130)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])