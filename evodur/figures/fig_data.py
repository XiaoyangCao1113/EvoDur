"""Data figure: distribution of incident durations in each benchmark (empirical survival curves).

    python -m evodur.figures.fig_data --out paper_out/figs

Needs all six datasets (the Beijing data are not public).
"""
import argparse
from pathlib import Path

import numpy as np

from evodur.data import load  # noqa: E402
from evodur.figures import style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

style.apply()
SETS = [("ring4", "Beijing 4th Ring"), ("citywide", "Beijing citywide"), ("ny511early", "NY/NJ/CT 511"),
        ("sydney", "Sydney"), ("seattle", "Seattle"), ("sanfrancisco", "San Francisco")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs")
    style.FIG_DIR = Path(ap.parse_args().out)
    fig, ax = plt.subplots(figsize=(style.WIDTH["onehalf"], 2.3))
    grid = np.geomspace(1, 1440, 400)
    for k, (name, title) in enumerate(SETS):
        t = np.sort(load(name, 1.0).t)
        surv = 1 - np.searchsorted(t, grid, side="right") / len(t)
        st = style.series_style(k)
        if k == 5:  # slot 5 is a second green next to slot 2: take the purple slot instead
            st["color"] = style.SERIES[6]
        ax.plot(grid, surv, label=f"{title} ({np.median(t):.0f} min)", **st)
    ax.axhline(0.5, color=style.INK2, lw=0.5, ls=(0, (1, 2)), zorder=0.5)
    ax.set_xscale("log")
    ax.set_xlim(1, 1440)
    ax.set_ylim(0, 1.0)
    ticks = [1, 5, 15, 60, 240, 1440]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1", "5", "15", "60", "240", "1440"])
    ax.minorticks_off()
    ax.set_xlabel("Duration $t$ (min, log scale)")
    ax.set_ylabel("Share lasting longer than $t$")
    ax.legend(title="Dataset (median duration)", loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=2.4, alignment="left")
    fig.tight_layout()
    style.save(fig, "fig_data")
    print("wrote fig_data")


if __name__ == "__main__":
    main()
