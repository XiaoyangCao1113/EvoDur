"""Figures: (a) P(c | t) curves (clearance-method evidence) per dataset;
(b) case study of evidence updating for one 4th Ring incident.

    python -m evodur.figures.fig_evidence method --out paper_out/figs     # fig_method_curves (the figure of the paper)
    python -m evodur.figures.fig_evidence case --out paper_out/figs       # fig_case_update (one 4th Ring incident)

The curves are fitted once (small networks, about 10 min on a CPU for all six datasets, needs the data) and cached in
<out>/../cache/method_curves_cache.npz, so a change of the layout needs no refit.
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from evodur.data import encoder  # noqa: E402
from evodur.evidence import ArrivalEvidence, MethodEvidence, fit_evidence, fit_mixaft, load_with_stages  # noqa: E402
from evodur.figures import style  # noqa: E402

style.apply()

torch.set_num_threads(2)
CACHE = Path("cache")  # set in __main__
INK, INK2 = style.INK, style.INK2
BEIJING_METHODS = ["Parties cleared, no congestion", "Police cleared, no congestion", "Parties cleared, congestion", "Police cleared, congestion"]
BEIJING_SHORT = ["Parties, no congestion", "Police, no congestion", "Parties, congestion", "Police, congestion"]  # names on the bands
TAU = 1.0


def split(data, frac=0.85, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data.t))
    n = int(frac * len(idx))
    return idx[:n], idx[n:]

BANDS = ["#9dc3e6", "#f6c28b", "#a9d8b8", "#e9a9c4", "#c6bde8"]  # light, colour-blind safe; neighbours differ in hue and lightness


def stack_panel(ax, tg, P, t, labels, ticks):
    """Stacked area of P(c | t) over the duration, the class names written on the bands; the data density is a thin strip
    below the zero line."""
    P = P / P.sum(1, keepdims=True)
    x0, x1 = float(tg[0]), float(tg[-1])
    cum = np.vstack([np.zeros(len(tg)), np.cumsum(P, 1).T])
    lx = np.log(tg)
    foot = -0.075
    for k in range(P.shape[1]):
        ax.fill_between(tg, cum[k], cum[k + 1], color=BANDS[k % len(BANDS)], lw=0, zorder=2)
        ax.plot(tg, cum[k + 1], color="white", lw=0.7, zorder=3)
    bins = np.geomspace(x0, x1, 41)
    h, _ = np.histogram(t, bins)
    ax.bar(bins[:-1], 0.06 * h / h.max(), bottom=foot, width=np.diff(bins), align="edge", color="#b9b8b3", lw=0, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(x0, x1)
    ax.set_ylim(foot, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.spines["left"].set_bounds(0, 1)
    ax.spines["bottom"].set_position(("data", foot))
    ax.grid(False)
    tk = []
    for v in ticks:
        if x0 <= v <= x1 and (not tk or np.log10(v / tk[-1]) >= 0.17 * np.log10(x1 / x0)):
            tk.append(v)
    ax.set_xticks(tk)
    ax.set_xticklabels([str(v) for v in tk])
    ax.minorticks_off()


def label_bands(ax, tg, P, labels):
    """Names on the bands, at the place where the band is thickest over the width of the text (call after the layout)."""
    P = P / P.sum(1, keepdims=True)
    x0, x1 = float(tg[0]), float(tg[-1])
    cum = np.vstack([np.zeros(len(tg)), np.cumsum(P, 1).T])
    lx = np.log(tg)
    foot = ax.get_ylim()[0]
    fig = ax.figure
    fig.canvas.draw()
    ax_w = ax.get_window_extent().width
    ax_h = ax.get_window_extent().height
    thin = []
    for k, lab in enumerate(labels):
        done = False
        for cand in (lab, textwrap.fill(lab, max(8, len(lab) // 2 + 4))):
            txt = ax.text(0, 0, cand, fontsize=7, color=INK, ha="center", va="center", zorder=4, linespacing=1.0)
            bb = txt.get_window_extent()
            tw = bb.width / ax_w * (lx[-1] - lx[0])  # text width in log-x units
            th = bb.height / ax_h * (1 - foot)       # text height in y units
            best, bi = -1, 0
            yc = (cum[k] + cum[k + 1]) / 2
            for i in range(len(tg)):
                lo_, hi_ = lx[i] - tw / 2, lx[i] + tw / 2
                if lo_ < lx[0] + 0.03 * (lx[-1] - lx[0]) or hi_ > lx[-1] - 0.02 * (lx[-1] - lx[0]):
                    continue
                m = (lx >= lo_) & (lx <= hi_)
                v = min((yc[i] - th / 2 - cum[k][m]).min(), (cum[k + 1][m] - yc[i] - th / 2).min())  # room around the text
                if v > best:
                    best, bi = v, i
            if best >= 0.004:
                txt.set_position((tg[bi], (cum[k] + cum[k + 1])[bi] / 2))
                done = True
                break
            txt.remove()
        if not done:
            thin.append(k)
    # too thin for the text: the name stands right of the panel, joined to the band at the right edge
    items = sorted(((cum[k, -1] + cum[k + 1, -1]) / 2, k) for k in thin)[::-1]  # from the top; each name sits at or below its band
    ys, floor = [], 1.0
    for y, k in items:
        lab = textwrap.fill(labels[k], 15)
        h = (lab.count("\n") + 1) * 8.0 * fig.dpi / 72 / ax_h * (1 - foot)  # height of the text in y units
        yt = min(y, floor - h / 2)
        ys.append((yt, y, lab, h))
        floor = yt - h / 2 - 0.005
    for yt, y, lab, h in ys:
        ax.annotate(lab, (x1, y), (x1 * 1.12, yt), fontsize=6.5, color=INK2, ha="left", va="center", annotation_clip=False,
                    linespacing=1.0, arrowprops=dict(arrowstyle="-", color="#9a9994", lw=0.5, shrinkA=1, shrinkB=1, relpos=(0, 0.5)))


def method_curves():
    sets = [("ring4", "Beijing 4th Ring", BEIJING_SHORT),
            ("citywide", "Beijing citywide", BEIJING_SHORT),
            ("ny511early", "NY/NJ/CT 511", ["No further detail", "Property damage", "Injuries", "Severe"]),
            ("sydney", "Sydney", ["Emergency only", "Emergency + road agency", "Tow truck", "Road agency only", "Unknown/other"]),
            ("seattle", "Seattle", ["Report written", "Other report", "Assistance, no action", "Not found, canceled", "Citation, arrest"]),
            ("sanfrancisco", "San Francisco", ["Report taken", "Handled", "None, unable to locate", "Canceled", "Other"])]
    cache = CACHE / "method_curves_cache.npz"  # fitted curves, so layout changes need no refit
    CACHE.mkdir(parents=True, exist_ok=True)
    store = dict(np.load(cache)) if cache.exists() else {}
    for name, title, labels in sets:
        if f"{name}_P" in store:
            continue
        data, _ = load_with_stages(name, TAU)
        tr, va = split(data)
        enc = encoder(data).fit(data.frame.iloc[tr])
        Xtr, Xva = (enc.transform(data.frame.iloc[i]).astype(np.float32) for i in (tr, va))
        C = int(data.c.max()) + 1
        ev = fit_evidence(lambda d: MethodEvidence(d, C), Xtr, data.c[tr], data.t[tr], Xva, data.c[va], data.t[va],
                          seed=0, batch=1024)
        lo, hi = np.quantile(data.t, [0.01, 0.99])
        tg = np.geomspace(max(lo, 1.5), hi, 100).astype(np.float32)
        # average over the held-out incidents' x of P(c | t, x)
        Xs = torch.as_tensor(Xva[:2000])
        P = np.zeros((len(tg), C))
        with torch.no_grad():
            for i, tv in enumerate(tg):
                tt = torch.full((len(Xs),), float(tv))
                for k in range(C):
                    P[i, k] = ev.logp(Xs, torch.full((len(Xs),), k, dtype=torch.long), tt).exp().mean().item()
        store[f"{name}_tg"], store[f"{name}_P"], store[f"{name}_t"] = tg, P, data.t
        np.savez_compressed(cache, **store)
    fig, axes = plt.subplots(2, 3, figsize=(style.WIDTH["text"], 4.0))
    ticks = [2, 5, 10, 20, 30, 60, 120, 240, 600]
    for j, (ax, (name, title, labels)) in enumerate(zip(axes.ravel(), sets)):
        stack_panel(ax, store[f"{name}_tg"], store[f"{name}_P"], store[f"{name}_t"], labels, ticks)
        ax.set_title(f"({'abcdef'[j]}) {title}", loc="left", color=INK)
    for ax in axes[1]:
        ax.set_xlabel("Duration $t$ (min)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$P(c \mid t)$")
    fig.subplots_adjust(left=0.065, right=0.905, top=0.94, bottom=0.12, wspace=0.62, hspace=0.45)
    for ax, (name, _, labels) in zip(axes.ravel(), sets):
        label_bands(ax, store[f"{name}_tg"], store[f"{name}_P"], labels)
    style.save(fig, "fig_method_curves")
    print("wrote fig_method_curves")


def case_study():
    data, r = load_with_stages("ring4", TAU)
    tr, va = split(data)
    enc = encoder(data).fit(data.frame.iloc[tr])
    Xtr, Xva = (enc.transform(data.frame.iloc[i]).astype(np.float32) for i in (tr, va))
    t, c = data.t, data.c
    tau = np.full(len(t), TAU, np.float32)
    prior = fit_mixaft(Xtr, t[tr], tau[tr], Xva, t[va], tau[va], seed=0)
    ok = (r > 0) & (r < t)
    a_tr, a_va = tr[ok[tr]], va[ok[va]]
    ev_a = fit_evidence(ArrivalEvidence, enc.transform(data.frame.iloc[a_tr]).astype(np.float32), r[a_tr], t[a_tr],
                        enc.transform(data.frame.iloc[a_va]).astype(np.float32), r[a_va], t[a_va], seed=0)
    ev_c = fit_evidence(lambda d: MethodEvidence(d, 4), Xtr, c[tr], t[tr], Xva, c[va], t[va], seed=0)
    # a held-out incident with a long duration, police handling with congestion
    cand = [i for i in va if ok[i] and c[i] == 3 and 50 < t[i] < 90]
    i = cand[0]
    x = torch.as_tensor(enc.transform(data.frame.iloc[[i]]).astype(np.float32))
    g = np.geomspace(1.01, 400, 600).astype(np.float32)
    G = torch.as_tensor(g)
    Xg = x.expand(len(g), -1)
    with torch.no_grad():
        lf = prior.log_f(Xg, G).numpy()
        la = ev_a.logp(Xg, torch.full((len(g),), float(r[i])), G).numpy()
        lc = ev_c.logp(Xg, torch.full((len(g),), int(c[i]), dtype=torch.long), G).numpy()

    def norm(logd, lower=TAU):
        """density of log t (t f(t)), normalised on the grid; zero below the known lower bound"""
        d = np.exp(logd - logd.max()) * (g > lower) * g
        return d / np.trapezoid(d, np.log(g))

    s_el = 45.0
    states = [
        ("Call (prior)", norm(lf)),
        (f"Police arrived at {r[i]:.0f} min", norm(lf + la, r[i])),
        ("+ method: police, congestion", norm(lf + la + lc, r[i])),
        (f"+ still going at {s_el:.0f} min", norm(lf + la + lc, s_el)),
    ]
    fig, ax = plt.subplots(figsize=(style.WIDTH["single"], 2.4))
    for k, (lab, d) in enumerate(states):
        ax.plot(g, d, label=lab, **style.series_style(k))
    ax.axvline(t[i], color=INK, lw=1.0, ls="--")
    ax.annotate(f"observed\n{t[i]:.0f} min", xy=(t[i], ax.get_ylim()[1] * 0.55), xytext=(t[i] * 2.2, ax.get_ylim()[1] * 0.7),
                fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xscale("log")
    ax.set_xlim(1, 400)
    ax.set_xlabel("Duration $t$ (min)")
    ax.set_ylabel(r"Predictive density of $\ln t$")
    ax.legend(loc="upper left", handlelength=2.6)
    fig.tight_layout()
    style.save(fig, "fig_case_update")
    print("wrote fig_case_update; incident", int(i), "t", round(float(t[i]), 1), "r", round(float(r[i]), 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all", choices=["all", "case", "method"])
    ap.add_argument("--out", default="figs")
    ap.add_argument("--cache", default=None, help="directory of the fitted-curves cache (default: <out>/../cache)")
    a = ap.parse_args()
    which = a.which
    style.FIG_DIR = Path(a.out)
    CACHE = Path(a.cache) if a.cache else Path(a.out).resolve().parent / "cache"
    if which in ("all", "case"):
        case_study()
    if which in ("all", "method"):
        method_curves()
