"""Summary, table and figure of the interpretation analysis (evodur.interpret).

    python -m evodur.analysis.interpret_tables --results results --out paper_out/tables --figs paper_out/figs

Reads interpret_{data}_f{k}.csv of every dataset present and writes
  interpret_summary.csv   mean over folds with the minimum and maximum over folds, for every number
                          (permutations are averaged within a fold first)
  tab_interpret.tex       per dataset, the effect on the median and on P(T > cut min) of (A) call-time variables, for EvoDur
                          and for the linear log-logistic AFT, and (B) of reported label classes and of the police arrival
  fig_interpret_{data}.pdf  (a) change in P(T > cut) per call-time variable, (b) per piece of response information,
                          (c) importance of groups of call-time variables at the call and after the response information

The effects are counterfactual: the variable is set to one level, then to the other, for every held-out incident
(an association within the data, not a causal effect). "Major" is a duration above `--cut` minutes (default 120).
"""
from __future__ import annotations

import argparse
import glob
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from evodur.figures import style

DATASETS = {"ring4": "Beijing 4th Ring", "citywide": "Beijing citywide", "ny511early": "NY/NJ/CT 511",
            "sydney": "Sydney", "seattle": "Seattle", "sanfrancisco": "San Francisco"}
KEYS = ["data", "section", "variable", "contrast", "model", "metric"]


def load(R):
    parts = []
    for d in DATASETS:
        for f in sorted(glob.glob(str(Path(R) / f"interpret_{d}_f[0-9].csv"))):
            parts.append(pd.read_csv(f))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    for k in ("variable", "contrast", "model"):
        df[k] = df[k].fillna("")
    return df


def summarize(df):
    """Mean and min / max over folds. Permutations are first averaged within a fold."""
    imp = df["section"] == "importance"
    perm = (df[imp].groupby(["data", "fold", "section", "variable", "model", "metric"], sort=False)["value"].mean()
            .reset_index().assign(contrast="mean over permutations"))
    df = pd.concat([df[~imp], perm], ignore_index=True)
    g = df.groupby(KEYS, sort=False)["value"]
    return g.agg(mean="mean", min="min", max="max", n_folds="count").reset_index()


def pick(S, d, section, variable, model, metric, contrast=None):
    m = (S.data == d) & (S.section == section) & (S.variable == variable) & (S.model == model) & (S.metric == metric)
    if contrast is not None:
        m &= S.contrast == contrast
    s = S[m]
    return s.iloc[0] if len(s) else None


def _pp(x):
    """Signed percentage points with one decimal; a value that rounds to zero is written 0.0."""
    v = round(100 * x, 1)
    return "0.0" if v == 0 else f"{v:+.1f}"


def _tex(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def table(S, cut):
    from evodur.analysis.paper_results import typeset_numbers

    pm = f"p{cut}_diff"
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         rf" & \multicolumn{{2}}{{c}}{{EvoDur}} & \multicolumn{{2}}{{c}}{{Linear AFT}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         rf"Change & Median ratio & $\Delta P(T>{cut})$ (pp) & Time ratio & $\Delta P(T>{cut})$ (pp) \\", r"\midrule"]

    def cell(v, f):
        return "--" if v is None else (_pp(v) if f is None else f.format(v))

    first = True
    for d, title in DATASETS.items():
        D = S[S.data == d]
        if D.empty:
            continue
        if not first:
            L.append(r"\addlinespace")
        first = False
        L.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{title}}}}} \\")
        calls = D[(D.section == "call_effect") & (D.model == "evodur") & (D.metric == pm)]
        if len(calls):
            L.append(r"\emph{Call-time variable (counterfactual)} & & & & \\")
        for v in calls.variable:
            g = lambda model, metric: pick(S, d, "call_effect", v, model, metric)
            e_ratio, e_p, a_ratio, a_p = g("evodur", "median_ratio"), g("evodur", pm), g("aft", "time_ratio_exact"), g("aft", pm)
            L.append(f"\\quad {_tex(v)} & {e_ratio['mean']:.2f} & {_pp(e_p['mean'])} & "
                     f"{cell(None if a_ratio is None else a_ratio['mean'], '{:.2f}')} & "
                     f"{cell(None if a_p is None else a_p['mean'], None)} \\\\")
        labs = D[(D.section == "label_effect") & (D.metric == pm) & D.contrast.str.startswith("report")]
        if len(labs):
            L.append(r"\emph{Label reported (EvoDur, relative to the call)} & & & & \\")
        for v in labs.variable:
            e_ratio, e_p = (pick(S, d, "label_effect", v, "evodur", m, "report this class for every test incident")
                            for m in ("median_ratio", pm))
            L.append(f"\\quad {_tex(v)} & {e_ratio['mean']:.2f} & {_pp(e_p['mean'])} & -- & -- \\\\")
        arr = D[(D.section == "arrival_effect") & (D.metric == pm)]
        if len(arr):
            L.append(r"\emph{Police arrival (EvoDur, relative to the call)} & & & & \\")
        for v in arr.variable:
            e_ratio, e_p, rm = (pick(S, d, "arrival_effect", v, "evodur", m) for m in ("median_ratio", pm, "r_minutes"))
            L.append(f"\\quad Arrival after {rm['mean']:.0f} min ({v.split(' after ')[-1].split(' ')[0]} percentile) & "
                     f"{e_ratio['mean']:.2f} & {_pp(e_p['mean'])} & -- & -- \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return typeset_numbers("\n".join(L)) + "\n"


def _wrap(s, w=30):
    return "\n".join(textwrap.wrap(s, w))


def figure(S, d, cut, fig_dir):
    from evodur.figures import style
    import matplotlib.pyplot as plt

    style.FIG_DIR = Path(fig_dir)
    style.apply()
    D = S[S.data == d]
    pm = f"p{cut}_diff"
    sc = 100.0

    def err(row):
        return [[sc * (row["mean"] - row["min"])], [sc * (row["max"] - row["mean"])]]

    calls = [v for v in D[(D.section == "call_effect") & (D.model == "evodur") & (D.metric == pm)].variable]
    labs = [v for v in D[(D.section == "label_effect") & (D.metric == pm) & D.contrast.str.startswith("report")].variable]
    arrs = [v for v in D[(D.section == "arrival_effect") & (D.metric == pm)].variable]
    imps = D[(D.section == "importance") & (D.model == "call")].set_index("variable")["mean"].sort_values(ascending=False)
    resp_state = "arrival+label" if (D[(D.section == "importance")].model == "arrival+label").any() else "label"
    ncol = max(len(calls), len(labs) + len(arrs), len(imps))
    fig, axs = plt.subplots(1, 3, figsize=(style.WIDTH["text"], 0.36 * ncol + 2.3),
                            gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})
    ax = axs[0]
    if calls:
        y = np.arange(len(calls))[::-1]
        for k, (m, name, mk) in enumerate((("evodur", "EvoDur", "o"), ("aft", "Linear AFT", "s"))):
            rows = [pick(S, d, "call_effect", v, m, pm) for v in calls]
            ax.errorbar([sc * r["mean"] for r in rows], y + (0.12 if k == 0 else -0.12), fmt=mk, ms=3.5, lw=0.7, capsize=1.5,
                        xerr=np.hstack([err(r) for r in rows]), label=name, color=style.SERIES[k], mfc=style.SERIES[k])
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap(v) for v in calls])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    else:
        ax.text(0.5, 0.5, "no interpretable\ncall-time variable", ha="center", va="center", transform=ax.transAxes, color=style.INK2)
        ax.set_yticks([])
    ax.axvline(0, color=style.INK2, lw=0.6)
    ax.set_xlabel(f"Change in $P(T>{cut}$ min) (pp)")
    ax.set_title("(a) Call-time variable", loc="left")

    ax = axs[1]
    rows = [(v, "label") for v in labs] + [(v, "arrival") for v in arrs]
    y = np.arange(len(rows))[::-1]
    for yy, (v, kind) in zip(y, rows):
        if kind == "label":
            r1 = pick(S, d, "label_effect", v, "evodur", pm, "report this class for every test incident")
            r2 = pick(S, d, "label_effect", v, "evodur", pm, "incidents of this class (factual)")
            ax.errorbar([sc * r1["mean"]], [yy + 0.1], xerr=err(r1), fmt="o", ms=3.5, lw=0.7, capsize=1.5, color=style.SERIES[0])
            if r2 is not None:
                ax.errorbar([sc * r2["mean"]], [yy - 0.14], xerr=err(r2), fmt="D", ms=3, lw=0.7, capsize=1.5, color=style.SERIES[0],
                            mfc="white")
        else:
            r1 = pick(S, d, "arrival_effect", v, "evodur", pm)
            ax.errorbar([sc * r1["mean"]], [yy], xerr=err(r1), fmt="o", ms=3.5, lw=0.7, capsize=1.5, color=style.SERIES[2])
    labels = [_wrap(v) for v, k in rows if k == "label"] + \
             [_wrap(f"Police arrive after {pick(S, d, 'arrival_effect', v, 'evodur', 'r_minutes')['mean']:.0f} min "
                    f"({'fast' if '25th' in v else 'slow'})") for v, k in rows if k == "arrival"]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if labs and arrs:
        ax.axhline(len(arrs) - 0.5, color=style.GRID, lw=0.8)
    ax.axvline(0, color=style.INK2, lw=0.6)
    ax.set_xlabel(f"Change in $P(T>{cut}$ min) (pp)")
    ax.set_title("(b) Response information", loc="left")
    ax.plot([], [], "o", ms=3.5, color=style.SERIES[0], label="Reported for every incident")
    ax.plot([], [], "D", ms=3, mfc="white", color=style.SERIES[0], label="Incidents of that class")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)

    ax = axs[2]
    y = np.arange(len(imps))[::-1]
    resp = "arrival and label" if resp_state == "arrival+label" else "label"
    series = (("call", "At the call", style.SERIES[0], 0.26),
              (resp_state + ", prior only", f"After {resp}, call-time variables\nscrambled in the prior only", style.SERIES[3], 0.0),
              (resp_state, f"After {resp}, scrambled\nin all modules", style.SERIES[1], -0.26))
    for st, name, col, off in series:
        rows = [pick(S, d, "importance", g, st, "logp_drop") for g in imps.index]
        ax.barh(y + off, [r["mean"] for r in rows], height=0.24, color=col,
                xerr=np.hstack([[[r["mean"] - r["min"]], [r["max"] - r["mean"]]] for r in rows]),
                error_kw=dict(lw=0.6, capsize=1.2, ecolor=style.INK2), label=name)
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(g, 24) for g in imps.index])
    ax.axvline(0, color=style.INK2, lw=0.6)
    ax.set_xlabel("Drop in log score when permuted")
    ax.set_title("(c) Reliance on call-time variables", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    for a in axs:
        a.grid(axis="y", visible=False)
        a.tick_params(axis="y", length=0)
    fig.suptitle(DATASETS[d], fontsize=8, fontweight="bold", y=1.0)
    fig.tight_layout()
    style.save(fig, f"fig_interpret_{d}")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- paper figure
# Call-time variables that describe the incident itself (time-of-day contrasts and NY/NJ/CT, which records no
# incident attribute at the call, are left out of the paper figure; they stay in interpret_summary.csv).
PAPER_VARS = {
    "ring4": ["Injury or fatality vs property damage only", "Rollover vs collision between vehicles",
              "Truck involved vs no truck", "Rain vs no rain"],
    "citywide": ["Injury or fatality vs property damage only", "Rollover vs collision between vehicles",
                 "Truck involved vs no truck", "Rain vs no rain"],
    "sydney": ["Pedestrian vs car (second party)", "Truck vs car (first vehicle)", "Heavy vs moderate traffic volume"],
    "seattle": ["High (1) vs low (5) priority", "911 call vs officer on view"],
    "sanfrancisco": ["Call type: injury accident vs non-injury accident", "Priority A vs priority C",
                     "Started by an officer (on view) vs called in"],
}
FACTUAL = "incidents of this class (factual)"


def _rows_call(S):
    rows = []
    for d, vs in PAPER_VARS.items():
        for v in vs:
            e = pick(S, d, "call_effect", v, "evodur", "median_ratio")
            a = pick(S, d, "call_effect", v, "aft", "median_ratio")
            if e is not None:
                rows.append((d, v, e, a))
    return rows


def _rows_label(S):
    rows = []
    for d in DATASETS:
        sub = S[(S.data == d) & (S.section == "label_effect") & (S.metric == "median_ratio") & (S.contrast == FACTUAL)]
        for _, r in sub.sort_values("mean").iterrows():
            rows.append((d, r.variable, r, None))
    return rows


def _dot_panel(ax, rows, label_fn, show_aft):
    """One row per effect, grouped by dataset: group header (bold, left of the axis), alternate groups shaded."""
    y, ticks, labels, heads, bands = 0.0, [], [], [], []
    prev = None
    for d, v, e, a in rows:
        if d != prev:
            if prev is not None:
                bands[-1][1] = y - 0.2
                y += 0.4
            bands.append([y - 0.5, None])
            heads.append((y, DATASETS[d]))
            y += 1.0
            prev = d
        ax.plot([e["min"], e["max"]], [y, y], color=style.SERIES[0], lw=1.3, solid_capstyle="butt", zorder=3)
        ax.plot(e["mean"], y, "o", color=style.SERIES[0], ms=5.5, zorder=4)
        if show_aft and a is not None:
            ax.plot(a["mean"], y, "o", mfc="none", mec=style.INK2, mew=1.0, ms=5.5, zorder=5)
        ticks.append(y)
        labels.append(label_fn(v))
        y += 1.0
    bands[-1][1] = y - 0.5
    for k, (y0, y1) in enumerate(bands):
        if k % 2 == 0:
            ax.axhspan(y0, y1, color="#f3f2ee", lw=0, zorder=0)
    for yh, name in heads:
        ax.annotate(name, (0.0, yh), xycoords=ax.get_yaxis_transform(), xytext=(-5.5, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7, fontweight="bold", color=style.INK)
    ax.axvline(1.0, color=style.INK, lw=0.8, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.5, -0.5)
    ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)


CALL_LABEL = {
    "Injury or fatality vs property damage only": "Injury or fatality (vs property damage)",
    "Rollover vs collision between vehicles": "Rollover (vs collision between vehicles)",
    "Truck involved vs no truck": "Truck involved",
    "Rain vs no rain": "Rain in the call hour",
    "Pedestrian vs car (second party)": "Pedestrian involved (vs car)",
    "Truck vs car (first vehicle)": "Truck (vs car)",
    "Heavy vs moderate traffic volume": "Heavy traffic (vs moderate)",
    "High (1) vs low (5) priority": "Priority 1 (vs 5)",
    "911 call vs officer on view": "911 call (vs officer on view)",
    "Call type: injury accident vs non-injury accident": "Injury accident (vs non-injury)",
    "Priority A vs priority C": "Priority A (vs C)",
    "Started by an officer (on view) vs called in": "Officer on view (vs called in)",
}


LABEL_SHORT = {
    "Severe (road closed, overturn, fire, investigation)": "Severe (closure, overturn, fire)",
    "Other (citation, arrest, advised, ...)": "Other (citation, arrest, ...)",
    "No disposition, unable to locate or gone on arrival": "None, unable to locate, gone",
}


def paper_figure(S, fig_dir, name="fig_interpret"):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullFormatter
    style.FIG_DIR = Path(fig_dir)
    style.apply()
    rc, rl = _rows_call(S), _rows_label(S)
    fig, axs = plt.subplots(1, 2, figsize=(style.WIDTH["text"], 4.7))
    _dot_panel(axs[0], rc, lambda v: CALL_LABEL.get(v, v), True)
    _dot_panel(axs[1], rl, lambda v: LABEL_SHORT.get(v, v), False)
    for ax, rows, show_aft, ticks in ((axs[0], rc, True, [0.5, 0.7, 1, 1.4, 2]), (axs[1], rl, False, [0.5, 0.7, 1, 1.4, 2, 3])):
        vals = [v for _, _, e, a in rows for v in (e["min"], e["max"], *([a["mean"]] if show_aft and a is not None else []))]
        lo, hi = min(vals), max(vals)
        ax.set_xlim(lo / 1.12, hi * 1.12)
        ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if lo / 1.12 <= t <= hi * 1.12]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.minorticks_off()
        ax.set_xticklabels([f"{t:g}" for t in ticks if lo / 1.12 <= t <= hi * 1.12])
        ax.grid(axis="x", color="white", lw=0.8, zorder=0.5)
    axs[0].set_xlabel("Median duration ratio, with / without")
    axs[1].set_xlabel("Median duration ratio, after / before")
    axs[0].set_title("(a) Call-time attributes", loc="left", fontweight="bold")
    axs[1].set_title("(b) Response label", loc="left", fontweight="bold")
    h = [Line2D([], [], color=style.SERIES[0], marker="o", ms=5, lw=1.3, label="EvoDur (dot: mean; line: range over folds)"),
         Line2D([], [], color=style.INK2, marker="o", mfc="none", ms=5, lw=0, label="Linear AFT model (time ratio)")]
    fig.legend(handles=h, loc="lower center", ncol=2)
    fig.tight_layout(w_pad=1.5, rect=(0, 0.06, 1, 1))
    style.save(fig, name)
    plt.close(fig)


def paper_table(S, cut=120):
    """Appendix table: change in P(T > cut) for the call-time attributes (EvoDur and linear AFT) and the labels."""
    m = f"p{cut}_diff"
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"Dataset & Attribute (vs reference) & Median ratio & \multicolumn{2}{c}{Change in $P(\text{major})$ (pp)} \\",
         r"\cmidrule(lr){4-5}", r" & & EvoDur & EvoDur & Linear AFT \\", r"\midrule"]
    prev = None
    for d, v, e, a in _rows_call(S):
        pe, pa = pick(S, d, "call_effect", v, "evodur", m), pick(S, d, "call_effect", v, "aft", m)
        L.append(f"{DATASETS[d] if d != prev else ''} & {_tex(CALL_LABEL.get(v, v))} & {e['mean']:.2f} & "
                 f"{_pp(pe['mean'])} & {_pp(pa['mean'])} \\\\")
        prev = d
    L += [r"\bottomrule", r"\end{tabular}"]
    t1 = "\n".join(L) + "\n"
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"Dataset & Label & Median ratio & \multicolumn{2}{c}{$P(\text{major})$ (\%)} \\", r"\cmidrule(lr){4-5}",
         r" & & after / before & At the call & With the label \\", r"\midrule"]
    prev = None
    for d, v, e, _ in _rows_label(S):
        r0 = pick(S, d, "label_effect", v, "evodur", f"p{cut}_ref", FACTUAL)
        r1 = pick(S, d, "label_effect", v, "evodur", f"p{cut}_new", FACTUAL)
        L.append(f"{DATASETS[d] if d != prev else ''} & {_tex(v)} & {e['mean']:.2f} & {100 * r0['mean']:.1f} & "
                 f"{100 * r1['mean']:.1f} \\\\")
        prev = d
    L += [r"\bottomrule", r"\end{tabular}"]
    t2 = "\n".join(L) + "\n"
    return t1, t2


def importance_table(S):
    """Drop in log score when one group of call-time variables is permuted in the prior, at the call and after the
    response information (arrival and label, or the label alone where no arrival is recorded)."""
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule", "Variable group & " + " & ".join(DATASETS.values()) + r" \\", r"\midrule"]
    groups = [g for g in S[S.section == "importance"].variable.unique()]
    for st, title in (("call", "At the call"), ("after", "After arrival and label (label only: NY/NJ/CT, Sydney)")):
        L.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{title}}}}} \\")
        for g in groups:
            cells = []
            for d in DATASETS:
                if st == "call":
                    r = pick(S, d, "importance", g, "call", "logp_drop")
                else:
                    r = pick(S, d, "importance", g, "arrival+label, prior only", "logp_drop")
                    r = r if r is not None else pick(S, d, "importance", g, "label, prior only", "logp_drop")
                cells.append("--" if r is None else f"{1000 * r['mean']:.0f}")
            L.append(_tex(g) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="directory with interpret_{data}_f{k}.csv")
    ap.add_argument("--out", required=True, help="directory for interpret_summary.csv and tab_interpret.tex")
    ap.add_argument("--figs", default=None, help="directory for the figures (default: <out>/../figs)")
    ap.add_argument("--cut", type=int, default=120, choices=(30, 60, 120), help="a duration above this many minutes is 'major'")
    a = ap.parse_args()
    df = load(a.results)
    if df is None:
        print("no interpret_*.csv in", a.results)
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = summarize(df)
    S.to_csv(out / "interpret_summary.csv", index=False)
    (out / "tab_interpret.tex").write_text(table(S, a.cut))
    figs = a.figs or out.resolve().parent / "figs"
    for d in DATASETS:
        if (S.data == d).any():
            figure(S, d, a.cut, figs)
    paper_figure(S, figs)
    t1, t2 = paper_table(S, a.cut)
    (out / "tab_interpret_call.tex").write_text(t1)
    (out / "tab_interpret_label.tex").write_text(t2)
    (out / "tab_importance.tex").write_text(importance_table(S))
    pd.set_option("display.width", 220)
    nf = df.groupby("data").fold.nunique().to_dict()
    print("folds per dataset:", nf)


if __name__ == "__main__":
    main()
