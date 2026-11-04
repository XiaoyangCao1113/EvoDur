"""Shared style for the paper figures (Transportation Research Part C).

- Sizes: Elsevier single column 90 mm, 1.5 column 140 mm, double column 190 mm; "text" = 165 mm, the text width of the
  preprint manuscript (figures set at this width print at their nominal font size).
- Fonts: serif (STIX, close to Times) at 8 pt, maths in STIX.
- Colours: categorical slots in fixed order (validated: CVD and normal-vision
  separation pass in light mode). Three slots are below 3:1 contrast on white,
  so every series also gets its own line style and a legend (print / CVD).
- Output: vector PDF for the manuscript and a 300 dpi PNG for previews.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4
WIDTH = {"single": 90 * MM, "onehalf": 140 * MM, "double": 190 * MM, "text": 165 * MM}

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DASHES = ["-", (0, (5, 2)), (0, (1.2, 1.4)), (0, (6, 2, 1.2, 2)), (0, (3, 1.5)), (0, (8, 2, 1.2, 2, 1.2, 2)), "-", (0, (5, 2))]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
# Fixed colour per method across all figures (colour follows the entity, never its rank).
MODEL_COLOR = {
    "EvoDur (ours)": SERIES[0], "Weibull MDN": SERIES[1], "Neural Fine-Gray": SERIES[2], "DeepHit": SERIES[3],
    "Neural CR mixture": SERIES[4], "Linear CR mixture": SERIES[6], "Linear AFT": SERIES[5], "Neural CR mixture, 2 or 4 per class": SERIES[7], "Separate models": INK2, "Neural stage-wise model": INK2,
}

FIG_DIR = Path("figs")  # output directory of save(); the entry points set it (make_paper.sh: <out>/figs)


def apply():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.3, "lines.markersize": 4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_label(ax, letter):
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")


def series_style(k):
    return dict(color=SERIES[k % len(SERIES)], linestyle=DASHES[k % len(DASHES)])


def check_layout(fig, name, tol=1.0):
    """Warn about text that overlaps other text, legends that cover data, and anything cut at the canvas edge.
    Coordinates are display pixels after a draw; `tol` pixels of touching are allowed."""
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hidden = set()  # tick labels outside the view limits are not drawn
    for ax in fig.axes:
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - 1e-9 * abs(hi - lo) <= tk.get_loc() <= hi + 1e-9 * abs(hi - lo)):
                    hidden.update({id(tk.label1), id(tk.label2)})
    texts = []
    for t in fig.findobj(Text):
        if id(t) in hidden:
            continue
        if t.get_visible() and t.get_text().strip():
            try:
                bb = t.get_window_extent(r)
            except Exception:
                continue
            if bb.width > 0 and bb.height > 0:
                texts.append((t.get_text().strip()[:30], bb))
    problems = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i][1], texts[j][1]
            ov_w = min(a.x1, b.x1) - max(a.x0, b.x0)
            ov_h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ov_w > tol and ov_h > tol and texts[i][0] != texts[j][0]:
                problems.append(f"text overlap: '{texts[i][0]}' / '{texts[j][0]}'")
    for leg in fig.findobj(Legend):
        if leg.axes is None:
            continue
        lb = leg.get_window_extent(r)
        for ln in leg.axes.get_lines():
            if not ln.get_visible() or ln.get_label().startswith("_"):
                continue
            xy = ln.axes.transData.transform(np.column_stack(ln.get_data()).astype(float))
            inside = (xy[:, 0] > lb.x0 + tol) & (xy[:, 0] < lb.x1 - tol) & (xy[:, 1] > lb.y0 + tol) & (xy[:, 1] < lb.y1 - tol)
            if inside.any():
                problems.append(f"legend covers data of '{ln.get_label()[:30]}'")
    for ax in fig.axes:
        ab = ax.get_window_extent(r)
        for ln in ax.get_lines():
            if not ln.get_visible() or not ln.get_clip_on() or len(ln.get_xdata()) < 3:
                continue
            xy = ln.get_transform().transform(np.column_stack(ln.get_data()).astype(float))
            xy = xy[np.isfinite(xy).all(1)]
            out = (xy[:, 0] < ab.x0 - tol) | (xy[:, 0] > ab.x1 + tol) | (xy[:, 1] < ab.y0 - tol) | (xy[:, 1] > ab.y1 + tol)
            if out.any():
                problems.append(f"curve cut at the axes edge: axes {fig.axes.index(ax)} '{ax.get_title()[:20]}', line '{ln.get_label()[:30]}' ({out.mean():.0%} of points)")
    from matplotlib.patches import FancyArrowPatch

    # savefig(bbox_inches="tight") keeps only fig.get_tightbbox(); arrows outside it are cut
    tb = fig.get_tightbbox(r).transformed(fig.dpi_scale_trans)
    from matplotlib.text import Annotation

    arrows = list(fig.findobj(FancyArrowPatch)) + [a.arrow_patch for a in fig.findobj(Annotation) if a.arrow_patch is not None]
    for arr in arrows:
        try:
            bb = arr.get_window_extent(r)
        except Exception:
            continue
        if bb.x0 < tb.x0 - tol or bb.y0 < tb.y0 - tol or bb.x1 > tb.x1 + tol or bb.y1 > tb.y1 + tol:
            problems.append("arrow outside the saved area (would be cut)")
    for msg in sorted(set(problems)):
        print(f"[layout] {name}: {msg}")
    return problems


def save(fig, name):
    check_layout(fig, name)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
