"""Tables and figures of the paper that are computed from the result files of run.py and evidence.py.

    python -m evodur.analysis.paper_results --results results --out paper_out

Reads folds_{data}_tau1_{tag}.csv (run.py), evidence_states_{data}_{tag}.csv (evidence.py) and the per-incident
files for the tags final_f*, finalnfg_f*, finalens_f*, finalensnfg_f*, lc{frac}_f*, ... Missing files are
skipped, so it can run on partial results. The results directory is only read. Writes LaTeX tables to <out>/tables/,
figures to <out>/figs/ and the derived single-network EvoDur files (rule: lowest validation NLL among the four prior
variants) to <out>/_derived/.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evodur.figures import style

import matplotlib.pyplot as plt  # noqa: E402

style.apply()
DERIVED = None  # directory of the derived files; set by main()
DATASETS = {"ring4": "Beijing 4th Ring", "citywide": "Beijing citywide", "ny511early": "NY/NJ/CT 511",
            "sydney": "Sydney", "seattle": "Seattle", "sanfrancisco": "San Francisco"}
LANDMARKS = [10, 20, 30, 45, 60, 90, 120]
OURS = "EvoDur"
VARIANTS = ["Evidence-DeepMix-8", "Evidence-DeepMix-8-het", "Evidence-DeepMix-16", "Evidence-DeepMix-16-het"]
# display name, in table order; models without a label term have no "label known" entry
NAMES = {
    OURS: "EvoDur (ours)", "WeibullMDN-sel": "Weibull MDN", "NeuralFineGray": "Neural Fine-Gray", "DeepHit-CR": "DeepHit",
    "DeepCAFT": "Neural CR mixture", "DeepCAFT-M": "Neural CR mixture, 2 or 4 per class", "Li2015": "Linear CR mixture", "AFT": "Linear AFT", "XGBoost-AFT": "XGBoost-AFT",
    "RSF": "RSF", "DeepSurv": "DeepSurv",
}
USES_LABEL = {OURS, *VARIANTS, "DeepHit-CR", "DeepCAFT", "DeepCAFT-M", "Li2015", "NeuralFineGray", "Evidence-AFT"}
# The learning curves and the tau = 15 runs use the Weibull MDN with a fixed K = 4.
FIXED_K = {"WeibullMDN-4": "Weibull MDN ($K=4$)"}


def disp(m):
    return NAMES.get(m, FIXED_K.get(m, m))


def fig_label(m):
    """Display name for matplotlib, which does not turn the LaTeX "--" into an en dash."""
    return disp(m).replace("--", "\u2013")
ABLATION = {
    OURS: "EvoDur (ours; selected on validation)", "Evidence-DeepMix-8": "$K=8$, shared shapes",
    "Evidence-DeepMix-8-het": "$K=8$, shapes depend on $\\x$", "Evidence-DeepMix-16": "$K=16$, shared shapes",
    "Evidence-DeepMix-16-het": "$K=16$, shapes depend on $\\x$", "Evidence-AFT": "Linear prior ($K=1$), vertical",
    "DeepAFT": "Neural prior, $K=1$", "DeepMix-2": "Neural prior, $K=2$", "DeepMix-4": "Neural prior, $K=4$",
    "DeepMix-8-constpi": "Neural prior, $K=8$, constant weights", "Mix-linear-8": "Linear prior, $K=8$",
    "DeepCAFT": "Neural CR mixture (horizontal)", "Li2015": "Linear CR mixture (horizontal)",
}


def read(results, pattern):
    # *_finalinc_* runs only store per-incident scores for the incident-level tests; keep them out of the tables.
    # Derived files (the validation-selected EvoDur) live in the DERIVED directory (<out>/_derived), not in results.
    found = glob.glob(str(Path(results) / pattern)) + (glob.glob(str(Path(DERIVED) / pattern)) if DERIVED else [])
    files = sorted(f for f in found if "finalinc" not in Path(f).name)
    if not files:
        return None
    # Optional result files with the tags *xgb* or *bl3* (a separate run of the listed models) take precedence over
    # the rows of these models in the other files; the job list of the release does not write such files.
    parts = []
    for f in files:
        df = pd.read_csv(f)
        if "model" in df:
            for key, redo in RERUNS.items():
                if not any(key in Path(g).name for g in files):
                    continue
                hit = df.model.map(base_name).isin(redo)
                df = df[hit] if key in Path(f).name else df[~hit]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


RERUNS = {"xgb": {"XGBoost-AFT"}, "bl3": {"WeibullMDN-4", "WeibullMDN-sel", "DeepHit-CR", "DeepSurv"}}


def base_name(m):
    return m.split("-ens")[0]


def fmt(v, best=None, second=None, nd=3):
    if pd.isna(v):
        return "--"
    s = f"{v:.{nd}f}"
    if best is not None and np.isclose(v, best):
        return rf"\textbf{{{s}}}"
    if second is not None and np.isclose(v, second):
        return rf"\underline{{{s}}}"
    return s


def top2(vals, higher=True):
    v = sorted({round(x, 6) for x in vals if pd.notna(x)}, reverse=higher)
    return (v + [None, None])[:2]


# ------------------------------------------------------------------ tables
def main_table(R, block=None):
    """Log score at the call and with the label known, all six datasets; single models then 5-seed ensembles.
    block="single" or "ens" gives only that block, without the block heading (Tables main and main_ens of the paper)."""
    single, ens = {}, {}
    for d in DATASETS:
        single[d] = read(R, f"folds_{d}_tau1_final*_f[0-9].csv")
        if single[d] is not None:
            single[d] = single[d][~single[d].model.str.contains("-ens")]
        ens[d] = read(R, f"folds_{d}_tau1_final*ens*_f[0-9].csv")
    if all(v is None for v in single.values()):
        return None
    lines = [r"\begin{tabular}{l" + "rr" * len(DATASETS) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for n in DATASETS.values()) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DATASETS))),
             "Model & " + " & ".join(["Call", "Label"] * len(DATASETS)) + r" \\", r"\midrule"]
    for title, src in (("Single networks", single), ("Ensembles of five seeds", ens)):
        if block is not None and title[0] != {"single": "S", "ens": "E"}[block]:
            continue
        means = {}
        for d, df in src.items():
            if df is None:
                continue
            g = df.assign(base=df.model.map(base_name)).groupby("base")
            means[d] = pd.DataFrame({"call": g.report_logp.mean(), "known": g.known_logp.mean()})
            means[d].loc[~means[d].index.isin(USES_LABEL), "known"] = np.nan
        if not means:
            continue
        models = [m for m in NAMES if any(m in v.index and pd.notna(v.loc[m, "call"]) for v in means.values())]
        means = {d: v.loc[[m for m in models if m in v.index]] for d, v in means.items()}
        if block is None:
            lines.append(rf"\multicolumn{{{1 + 2 * len(DATASETS)}}}{{l}}{{\emph{{{title}}}}} \\")
        bests = {d: (top2(v.call), top2(v.known)) for d, v in means.items()}
        for m in models:
            cells = []
            for d in DATASETS:
                if d not in means or m not in means[d].index:
                    cells += ["--", "--"]
                    continue
                (b1, s1), (b2, s2) = bests[d]
                cells += [fmt(means[d].loc[m, "call"], b1, s1), fmt(means[d].loc[m, "known"], b2, s2)]
            lines.append(NAMES[m] + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if block is None and title.startswith("Single") else "")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(x for x in lines if x)


def paired_wins(R):
    """Ours minus the best other model, per fold (single networks): mean difference and wins."""
    rows = []
    for d in DATASETS:
        df = read(R, f"folds_{d}_tau1_final*_f[0-9].csv")
        if df is None or OURS not in set(df.model):
            continue
        df = df[~df.model.str.contains("-ens")]
        piv = {k: df.pivot_table(index=["repeat", "fold"], columns="model", values=k) for k in ("report_logp", "known_logp")}
        for k, p in piv.items():
            others = [m for m in p.columns if m in NAMES and m != OURS and (k == "report_logp" or m in USES_LABEL)]
            best = p[others].mean().idxmax()
            diff = (p[OURS] - p[best]).dropna()
            rows.append({"data": d, "state": "call" if k == "report_logp" else "label", "best_other": NAMES.get(best, best),
                         "diff": diff.mean(), "sd": diff.std(), "wins": int((diff > 0).sum()), "folds": len(diff)})
    return pd.DataFrame(rows)


def point_summary_table(R):
    """Point, ranking and interval measures at the call: EvoDur against the best other model for each measure, and
    EvoDur's absolute error once evidence has arrived (single networks, means over folds)."""
    rows = []
    for d, title in DATASETS.items():
        df = read(R, f"folds_{d}_tau1_final*_f[0-9].csv")
        if df is None:
            continue
        g = df[~df.model.str.contains("-ens")].groupby("model")[["IBS", "C_index", "MAE", "cover80"]].mean()
        g = g.loc[[m for m in NAMES if m in g.index]]
        o = g.drop(OURS)
        e = read(R, f"evidence_states_{d}_final_f[0-9].csv")
        st = "arrival+method-chain" if e is not None and (e.state == "arrival+method-chain").any() else "method"
        mae_ev = e[e.state == st].evidence_mae.mean() if e is not None else np.nan
        cov_other = o.cover80.iloc[(o.cover80 - 0.8).abs().argmax()]
        rows.append([title, f"{g.loc[OURS, 'IBS']:.4f}", f"{o.IBS.min():.4f}", f"{g.loc[OURS, 'C_index']:.3f}",
                     f"{o.C_index.max():.3f}", f"{g.loc[OURS, 'MAE']:.1f}", f"{o.MAE.min():.1f}",
                     f"{mae_ev:.1f}" + ("" if st != "method" else "$^\\dagger$"),
                     f"{100 * g.loc[OURS, 'cover80']:.1f}", f"{100 * o.cover80.min():.1f}--{100 * o.cover80.max():.1f}"])
    if not rows:
        return None
    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{IBS} & \multicolumn{2}{c}{C-index} & \multicolumn{3}{c}{MAE of the median (min)} & \multicolumn{2}{c}{Coverage of 80\% interval (\%)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
             r"Dataset & EvoDur & Best other & EvoDur & Best other & EvoDur & Best other & EvoDur, updated & EvoDur & Others (range) \\",
             r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def point_known_table(R):
    """Point, ranking and interval measures once the label is known: EvoDur against the best of the label-using
    baselines for each measure (single networks, means over folds)."""
    rows = []
    for d, title in DATASETS.items():
        df = read(R, f"folds_{d}_tau1_pointk*_f[0-9].csv")
        if df is None or OURS not in set(df.model):
            continue
        cap = read(R, f"folds_{d}_tau1_finalcap_f[0-9].csv")  # the capacity-matched mixture-form model
        if cap is not None and "IBS_known" in cap:
            df = pd.concat([df, cap[cap.model == "DeepCAFT-M"]], ignore_index=True)
        g = df.groupby("model")[["IBS_known", "C_index_known", "MAE_known", "PI80_cover_known"]].mean()
        o = g.drop(OURS)
        e = g.loc[OURS]
        rows.append([title, f"{e.IBS_known:.4f}", f"{o.IBS_known.min():.4f}", f"{e.C_index_known:.3f}",
                     f"{o.C_index_known.max():.3f}", f"{e.MAE_known:.1f}", f"{o.MAE_known.min():.1f}",
                     f"{100 * e.PI80_cover_known:.1f}",
                     f"{100 * o.PI80_cover_known.min():.1f}--{100 * o.PI80_cover_known.max():.1f}"])
    if not rows:
        return None
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{IBS} & \multicolumn{2}{c}{C-index} & \multicolumn{2}{c}{MAE of the median (min)} & \multicolumn{2}{c}{Coverage of 80\% interval (\%)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"Dataset & EvoDur & Best other & EvoDur & Best other & EvoDur & Best other & EvoDur & Others (range) \\",
             r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


POINT_COLS = ["MAE", "MAPE", "MedAE", "PI80_width", "PI80_cover"]


def point_minutes_data(R):
    """Point accuracy (min, %), 80% interval width and coverage of the predictive median, per dataset, model and state
    (call / label known); means over repeats and folds, single networks. The label-known numbers of the models in the
    pointk runs come from those runs (as in point_known_table), the others from the final runs."""
    rows = []
    for d in DATASETS:
        fin = read(R, f"folds_{d}_tau1_final*_f[0-9].csv")
        if fin is None:
            continue
        fin = fin[~fin.model.str.contains("-ens")]
        pk = read(R, f"folds_{d}_tau1_pointk*_f[0-9].csv")
        g_call = fin.groupby("model")[POINT_COLS].mean()
        known = [c + "_known" for c in POINT_COLS if c + "_known" in fin]
        g_fin = fin.groupby("model")[known].mean().dropna(how="all")
        g_pk = pk.groupby("model")[[c for c in known if c in pk]].mean() if pk is not None else g_fin.iloc[:0]
        g_known = pd.concat([g_pk, g_fin.drop(index=g_pk.index, errors="ignore")])
        for m in g_call.index:
            r = {"data": d, "model": m, "state": "call", **g_call.loc[m].to_dict()}
            rows.append(r)
            if m in g_known.index and m in USES_LABEL | {OURS} | set(VARIANTS) | {"DeepCAFT-M"}:
                rows.append({"data": d, "model": m, "state": "label",
                             **{c: g_known.loc[m].get(c + "_known", np.nan) for c in POINT_COLS}})
    return pd.DataFrame(rows)


def point_minutes_table(R, csv_path=None):
    """Point accuracy in minutes: MAE and MAPE of the median for EvoDur, the best other model on that metric and the
    linear AFT, at the call and with the label known, plus 80% interval width and coverage. Writes all models to csv."""
    df = point_minutes_data(R)
    if df.empty or OURS not in set(df.model):
        return None
    if csv_path is not None:
        out = df.copy()
        out["model_name"] = out.model.map(lambda m: NAMES.get(m, FIXED_K.get(m, m)))
        out["MAPE"] = 100 * out.MAPE
        out["PI80_cover"] = 100 * out.PI80_cover
        out.rename(columns={"MAE": "MAE_min", "MAPE": "MAPE_pct", "MedAE": "MedAE_min", "PI80_width": "PI80_width_min",
                            "PI80_cover": "PI80_cover_pct"}).round(3).to_csv(csv_path, index=False)
    # linear AFT: with the label known it is the linear prior with the same evidence update (Evidence-AFT)
    aft = {"call": "AFT", "label": "Evidence-AFT"}
    out_rows = {"MAE": [], "MAPE": [], "PI": []}
    for d, title in DATASETS.items():
        cell = {k: [title] for k in out_rows}
        for st in ("call", "label"):
            g = df[(df.data == d) & (df.state == st)].set_index("model")
            if OURS not in g.index:
                for k in ("MAE", "MAPE"):
                    cell[k] += ["--"] * 4
                cell["PI"] += ["--"] * 4
                continue
            cand = g.loc[[m for m in g.index if m in NAMES and m != OURS]]
            for k, scale in (("MAE", 1), ("MAPE", 100)):
                best = cand[k].idxmin()
                a = g[k].get(aft[st], np.nan)
                vals = [scale * g.loc[OURS, k], scale * cand.loc[best, k], NAMES[best].split(",")[0], scale * a]
                lo = min(v for v in (vals[0], vals[1], vals[3]) if pd.notna(v))
                f = lambda v: "--" if pd.isna(v) else (rf"\textbf{{{v:.1f}}}" if np.isclose(v, lo) else f"{v:.1f}")
                cell[k] += [f(vals[0]), f(vals[1]), vals[2], f(vals[3])]
            w, c = g["PI80_width"], g["PI80_cover"]
            cell["PI"] += [f"{w[OURS]:.1f}", "--" if pd.isna(w.get(aft[st], np.nan)) else f"{w[aft[st]]:.1f}",
                           f"{100 * c[OURS]:.1f}", "--" if pd.isna(c.get(aft[st], np.nan)) else f"{100 * c[aft[st]]:.1f}"]
        for k in out_rows:
            out_rows[k].append(cell[k])
    head = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
            r" & \multicolumn{4}{c}{At the call} & \multicolumn{4}{c}{Label known} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}"]
    sub = r"Dataset & EvoDur & Best other & Model & Linear AFT & EvoDur & Best other & Model & Linear AFT \\"
    pi_head = (r"Dataset & \multicolumn{2}{c}{Width (min)} & \multicolumn{2}{c}{Coverage (\%)} & "
               r"\multicolumn{2}{c}{Width (min)} & \multicolumn{2}{c}{Coverage (\%)} \\" + "\n"
               r" & EvoDur & AFT & EvoDur & AFT & EvoDur & AFT & EvoDur & AFT \\")
    lines = head + [sub]
    for title, key in (("MAE of the median (min)", "MAE"), ("MAPE of the median (\\%)", "MAPE")):
        lines += [r"\midrule", rf"\multicolumn{{9}}{{l}}{{\emph{{{title}}}}} \\"]
        lines += [" & ".join(r) + r" \\" for r in out_rows[key]]
    lines += [r"\midrule", r"\multicolumn{9}{l}{\emph{80\% prediction interval (10th to 90th percentile)}} \\", pi_head]
    # the interval block has its own column layout (no model-name columns): reuse the nine-column grid
    pi = []
    for r in out_rows["PI"]:
        pi.append(" & ".join([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]]) + r" \\")
    lines += pi + [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def other_metrics_table(R, d):
    df = read(R, f"folds_{d}_tau1_final*_f[0-9].csv")
    if df is None:
        return None
    df = df[~df.model.str.contains("-ens")]
    cols = [("IBS", "IBS", False), ("C_index", "C-index", True), ("MAE", "MAE (min)", False),
            ("MedAE", "MedAE (min)", False), ("cover80", "Cover. 80\\%", None), ("PI80_width", "Width 80\\% (min)", False)]
    cols = [c for c in cols if c[0] in df]
    g = df.groupby("model")[[c[0] for c in cols]].mean()
    g = g.loc[[m for m in NAMES if m in g.index]]
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             "Model & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for m in g.index:
        cells = []
        for k, _, hi in cols:
            v = g[k]
            if hi is None:  # coverage: closest to nominal
                dist = (v - 0.8).abs()
                b = v[dist == dist.min()].iloc[0]
                cells.append(fmt(g.loc[m, k], b, None))
            else:
                b, s = top2(v, hi)
                nd = 1 if "MAE" in k or "width" in k.lower() else 4 if k == "IBS" else 3
                cells.append(fmt(g.loc[m, k], b, s, nd))
        lines.append(NAMES[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def ablation_table(R):
    rows = {}
    for d in DATASETS:
        parts = [read(R, f"folds_{d}_tau1_{tg}_f[0-9].csv") for tg in ("final", "variants", "finalevo")]
        parts = [p_ for p_ in parts if p_ is not None]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        base = df[df.model == "AFT"].set_index(["repeat", "fold"])
        for m in ABLATION:
            g = df[df.model == m].set_index(["repeat", "fold"])
            if len(g):
                rows.setdefault(m, {})[d] = ((g.report_logp - base.report_logp).mean(),
                                             (g.known_logp - base.report_logp).mean() if m in USES_LABEL else np.nan)
    if not rows:
        return None
    lines = [r"\begin{tabular}{l" + "rr" * len(DATASETS) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for n in DATASETS.values()) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DATASETS))),
             "Variant & " + " & ".join(["Call", "Label"] * len(DATASETS)) + r" \\", r"\midrule"]
    for m, lab in ABLATION.items():
        if m in rows:
            cells = []
            for d in DATASETS:
                a, b = rows[m].get(d, (np.nan, np.nan))
                cells += [fmt(1000 * a, nd=0) if pd.notna(a) else "--", fmt(1000 * b, nd=0) if pd.notna(b) else "--"]
            lines.append(lab + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


STATE_NAMES = {"report": "Call", "method": "Label", "arrival": "Police arrival", "arrival+method": "Arrival and label",
               "arrival+method-chain": "Arrival and label"}


def state_label(s):
    if s in STATE_NAMES:
        return STATE_NAMES[s]
    if s.startswith("elapsed"):
        v = s[len("elapsed"):]
        return f"Open at {v.split('+')[0]} min" + (" and label" if "+method" in v else "")
    return s


def states_table(R, tag="final"):
    """Evidence update minus a separate model fitted for the state (log score x 1000, wins / folds)."""
    out = {}
    for d in DATASETS:
        e = read(R, f"evidence_states_{d}_{tag}_f[0-9].csv")
        if e is None:
            continue
        e = e.assign(diff=e.evidence_logp - e.separate_model_logp)
        for col, key in (("separate_linear_logp", "dlin"), ("separate_xgb_logp", "dxgb")):
            e[key] = e.evidence_logp - e[col] if col in e else np.nan
        g = e.groupby("state").agg(diff=("diff", "mean"), wins=("diff", lambda x: int((x > 0).sum())), k=("diff", "size"),
                                   n=("n", "mean"), dlin=("dlin", "mean"), dxgb=("dxgb", "mean"))
        g = g[g.k == g.k.max()]  # states whose runs are still incomplete wait until every fold is in
        out[d] = g[g.n >= 100]  # same rule as the dynamic figure: at least 100 test incidents per fold
    if not out:
        return None
    order = ["arrival", "method", "arrival+method-chain"] + [f"elapsed{s}{p}" for p in ("", "+method") for s in LANDMARKS]
    states = [s for s in order if any(s in v.index for v in out.values())]
    lines = [r"\begin{tabular}{l" + "r" * len(DATASETS) + "}", r"\toprule",
             "Information state & " + " & ".join(DATASETS.values()) + r" \\", r"\midrule"]
    for s in states:
        cells = []
        for d in DATASETS:
            if d in out and s in out[d].index:
                r = out[d].loc[s]
                f = lambda v: "--" if pd.isna(v) else f"{1000 * v:+.0f}"
                cells.append(" / ".join([f(r["diff"]), f(r.dlin), f(r.dxgb)]))
            else:
                cells.append("--")
        lines.append(state_label(s) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def open_table(R, tag="final"):
    """Main-text summary of states_table: the update minus the neural stage-wise model refitted on the incidents still
    open, with the label known (log score x 1000, wins / folds)."""
    out = {}
    for d in DATASETS:
        e = read(R, f"evidence_states_{d}_{tag}_f[0-9].csv")
        if e is None:
            continue
        e = e.assign(diff=e.evidence_logp - e.separate_model_logp)
        g = e.groupby("state").agg(diff=("diff", "mean"), wins=("diff", lambda x: int((x > 0).sum())), k=("diff", "size"),
                                   n=("n", "mean"))
        out[d] = g[(g.k == g.k.max()) & (g.n >= 100)]
    if not out:
        return None
    states = ["method"] + [f"elapsed{s}+method" for s in (10, 30, 60, 90, 120)]
    lines = [r"\begin{tabular}{l" + "r" * len(DATASETS) + "}", r"\toprule",
             "Information state & " + " & ".join(DATASETS.values()) + r" \\", r"\midrule"]
    for s in states:
        if not any(s in v.index for v in out.values()):
            continue
        cells = []
        for d in DATASETS:
            if d in out and s in out[d].index:
                r = out[d].loc[s]
                cells.append(f"{fmt_signed(1000 * r['diff'])} ({int(r.wins)}/{int(r.k)})")
            else:
                cells.append("--")
        name = "Label" if s == "method" else f"Label, open {s[len('elapsed'):].split('+')[0]} min"
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def fmt_signed(v):
    v = int(round(v))
    return "0" if v == 0 else (f"$+${v}" if v > 0 else f"$-${-v}")


ARRIVAL_SETS = ("ring4", "citywide", "seattle", "sanfrancisco")  # datasets that record the police arrival time
STAGE = [("separate_model_logp", "Neural"), ("separate_linear_logp", "Linear AFT"), ("separate_xgb_logp", "XGBoost-AFT")]


def arrival_table(R):
    """States with police arrival: update minus each stage-wise model (5-fold CV). Arrival and label uses the
    chain-rule label module."""
    rows = []
    for d in ARRIVAL_SETS:
        cv = read(R, f"evidence_states_{d}_final_f[0-9].csv")
        if cv is None:
            continue
        for st, lab in (("arrival", "Police arrival"), ("arrival+method-chain", "Arrival and label")):
            e = cv[cv.state == st]
            if len(e) < 5:
                continue
            cells = []
            for col, _ in STAGE:
                if col not in e or e[col].isna().any():
                    cells.append("--")
                    continue
                dd = e.evidence_logp - e[col]
                cells.append(f"{1000 * dd.mean():+.0f} ({int((dd > 0).sum())}/{len(dd)})")
            rows.append((DATASETS[d], lab, cells))
    if not rows:
        return None
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             "Dataset & Information state & " + " & ".join(n for _, n in STAGE) + r" \\", r"\midrule"]
    prev = None
    for name, lab, cells in rows:
        lines.append((name if name != prev else "") + " & " + lab + " & " + " & ".join(cells) + r" \\")
        prev = name
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


SENS = {"Evidence-DeepMix-8": "EvoDur ($K=8$, shared shapes)", "WeibullMDN-4": "Weibull MDN ($K=4$)", "DeepHit-CR": "DeepHit",
        "DeepCAFT": "Neural CR mixture", "Li2015": "Linear CR mixture", "AFT": "Linear AFT", "XGBoost-AFT": "XGBoost-AFT"}


def sens_table(R):
    """Log score with tau = 15 min (records shorter than 15 min removed), single networks."""
    rows = {}
    for d in DATASETS:
        df = read(R, f"folds_{d}_tau15_sens*_f[0-9].csv")
        if df is None:
            continue
        g = df.groupby("model")[["report_logp", "known_logp"]].mean()
        for m in SENS:
            if m in g.index:
                rows.setdefault(m, {})[d] = (g.loc[m, "report_logp"], g.loc[m, "known_logp"] if m in USES_LABEL else np.nan)
    if not rows:
        return None
    sets = {d: n for d, n in DATASETS.items() if any(d in r for r in rows.values())}
    lines = [r"\begin{tabular}{l" + "rr" * len(sets) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for n in sets.values()) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(sets))),
             "Model & " + " & ".join(["Call", "Label"] * len(sets)) + r" \\", r"\midrule"]
    best = {d: (top2([r[d][0] for r in rows.values() if d in r]), top2([r[d][1] for r in rows.values() if d in r]))
            for d in sets}
    for m in SENS:
        if m not in rows:
            continue
        cells = []
        for d in sets:
            for j in (0, 1):
                v = rows[m].get(d, (np.nan, np.nan))[j]
                if pd.isna(v):
                    cells.append("--")
                    continue
                t = f"{v:.3f}"
                if round(v, 6) == best[d][j][0]:
                    t = rf"\textbf{{{t}}}"
                elif round(v, 6) == best[d][j][1]:
                    t = rf"\underline{{{t}}}"
                cells.append(t)
        lines.append(SENS[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def color(m):
    return style.MODEL_COLOR.get(NAMES.get(m, "Weibull MDN" if m in FIXED_K else m), style.INK2)


def line_style(m):
    keys = list(style.MODEL_COLOR)
    name = NAMES.get(m, "Weibull MDN" if m in FIXED_K else m)
    k = keys.index(name) if name in keys else 6
    markers = {"EvoDur (ours)": "o", "Weibull MDN": "D", "Neural Fine-Gray": "v", "DeepHit": "P", "Neural CR mixture": "X",
               "Neural CR mixture, 2 or 4 per class": "s", "Linear CR mixture": "^", "Linear AFT": "h"}
    return dict(color=style.MODEL_COLOR.get(name, style.INK2), linestyle=style.DASHES[k % len(style.DASHES)],
                marker=markers.get(name, "o"))


def nb_ci(x, ratio=0.25, level=0.95):
    """Mean and half-width of a Nadeau-Bengio corrected CI over folds (test/train size ratio `ratio`)."""
    from scipy import stats

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), np.nan
    se = x.std(ddof=1) * np.sqrt(1 / len(x) + ratio)
    return x.mean(), stats.t.ppf(0.5 + level / 2, len(x) - 1) * se


def fig_dynamic(R, known=False):
    """Incidents still open at s (and, if `known`, with the label known): log score of each model minus EvoDur,
    with 95% corrected confidence intervals over folds. Single networks."""
    fig, grid = plt.subplots(2, 3, figsize=(style.WIDTH["text"], 4.6))
    axes = list(grid.ravel()[:len(DATASETS)])  # a figure legend below the panels, so all six panels are data
    models = (["NeuralFineGray", "DeepCAFT", "DeepCAFT-M", "Li2015"] if known
              else ["WeibullMDN-sel", "NeuralFineGray", "DeepCAFT", "DeepCAFT-M", "Li2015"])
    series = models + ["sep"]
    offs = dict(zip(series, np.linspace(-0.06, 0.06, len(series))))  # dodge in log10(s)
    any_data = False
    for ax, (d, title) in zip(axes, DATASETS.items()):
        df = read(R, f"folds_{d}_tau1_final*_f[0-9].csv")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xticks([10, 30, 60, 120])
        ax.set_xticklabels(["10", "30", "60", "120"])
        ax.minorticks_off()
        ax.axhline(0, color=style.SERIES[0], lw=1.2, zorder=1)
        if df is None or "lm10_logp" not in df:
            continue
        any_data = True
        df = df[~df.model.str.contains("-ens")].set_index(["repeat", "fold"])
        col = "lm{}_known_logp" if known else "lm{}_logp"
        ours = df[df.model == OURS]
        # show an elapsed time only if at least 100 test incidents per fold are still open (fixed rule)
        lms = [s_ for s_ in LANDMARKS if ours[f"lm{s_}_n"].mean() >= 100]
        ax.set_xlim(8, max(lms) * 1.25)
        ax.set_xticks([v for v in (10, 30, 60, 120) if v <= max(lms)])
        ax.set_xticklabels([str(v) for v in (10, 30, 60, 120) if v <= max(lms)])
        for m in series:
            xs = np.array(lms, float)
            if m == "sep":
                e = read(R, f"evidence_states_{d}_final_f[0-9].csv")
                if e is None:
                    continue
                st = "elapsed{}+method" if known else "elapsed{}"
                vals = [nb_ci((e[e.state == st.format(s_)].separate_model_logp - e[e.state == st.format(s_)].evidence_logp).values)
                        for s_ in lms]
                sty = dict(color=style.INK2, linestyle="none", marker="s")
                lab = "Neural stage-wise model"
            else:
                g = df[df.model == m]
                if not len(g) or col.format(10) not in g:
                    continue
                vals = [nb_ci((g[col.format(s_)] - ours[col.format(s_)]).dropna().values) for s_ in lms]
                sty = {**line_style(m), "linestyle": "none"}
                lab = fig_label(m)
            mu, hw = (np.array(v) for v in zip(*vals))
            colr = sty["color"]
            if m == "sep":
                line = dict(color=colr, linestyle=(0, (4, 1.5)), marker="s")
            else:
                line = {**line_style(m)}
            line.pop("marker", None)
            ax.plot(xs, mu, lw=1.3, label=lab, **line)
            ok = np.isfinite(hw)
            ax.fill_between(xs[ok], (mu - hw)[ok], (mu + hw)[ok], color=colr, alpha=0.08, lw=0)
        ax.set_xlabel("Elapsed time $s$ (min)")
    if not any_data:
        plt.close(fig)
        return None
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Log score minus EvoDur")
    for ax in axes[:3]:
        ax.set_xlabel("")
    h, lab = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D

    h = [Line2D([], [], color=style.SERIES[0], lw=1.2)] + h
    lab = ["EvoDur (ours) = 0"] + lab
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, handlelength=2.8, columnspacing=1.6)
    fig.tight_layout(h_pad=1.2, w_pad=1.0, rect=(0, 0.1, 1, 1))
    name = "fig_dynamic_known" if known else "fig_dynamic"
    style.save(fig, name)
    plt.close(fig)
    return name


def fig_learning(R, known=False):
    """Learning curves as the difference EvoDur minus each competitor (log score per incident; above zero EvoDur is better),
    one line per competitor, each panel with its own vertical range."""
    sets = {d: t for d, t in DATASETS.items() if read(R, f"folds_{d}_tau1_lc*0.1_f[0-9].csv") is not None}
    fig, grid = plt.subplots(2, 3, figsize=(style.WIDTH["text"], 3.6))
    axes = list(grid.ravel()[:len(sets)])
    for ax in grid.ravel()[len(sets):]:
        ax.axis("off")
    fracs = [0.1, 0.25, 0.5, 1.0]
    # at the call: the strongest call-time models; with the label known: the models that use the label
    models = ["DeepCAFT-M", "NeuralFineGray", "DeepCAFT"] if known else ["WeibullMDN-4", "DeepHit-CR", "DeepCAFT"]
    score = "known_logp" if known else "report_logp"
    any_data = False
    for ax, (d, title) in zip(axes, sets.items()):
        ax.set_title(title)
        ax.axhline(0, color=style.INK, lw=0.8, zorder=1)
        pts = {m: [] for m in models}
        for f in fracs:
            df = read(R, f"folds_{d}_tau1_lc*{f}_f[0-9].csv") if f < 1 else read(R, f"folds_{d}_tau1_final*_f[0-9].csv")
            if df is not None:
                df = df[~df.model.str.contains("-ens")]
            if df is None:
                continue
            if f == 1:
                df = df[df.repeat == 0]
            df = df.set_index(["repeat", "fold"])
            ours = df[df.model == OURS][score]
            for m in models:
                g = df[df.model == m][score]
                if len(g) and len(ours):
                    pts[m].append((f, (ours - g).dropna().mean()))  # x axis: fraction of the training data
        for m in models:
            if pts[m]:
                any_data = True
                xs, ys = zip(*pts[m])
                ax.plot(xs, ys, ms=4.5, label=fig_label(m), zorder=3, **line_style(m))
        ax.set_xscale("log")
        ax.set_xlim(0.08, 1.25)
        ax.set_xticks(fracs)
        ax.set_xticklabels(["10%", "25%", "50%", "100%"])
        ax.minorticks_off()
        ax.set_xlabel("Share of training data")
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -0.1 * (hi - lo)), max(hi, 0.1 * (hi - lo)))  # keep the zero line inside, with a margin
        ax.grid(axis="x", visible=False)
    if not any_data:
        plt.close(fig)
        return None
    fig.supylabel(("Log score with the label known" if known else "Log score at the call") + ": EvoDur minus competitor", fontsize=8, x=0.005)
    for ax in axes[:3]:
        ax.set_xlabel("")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False, handlelength=2.2, columnspacing=1.6, handletextpad=0.5)
    fig.tight_layout(h_pad=1.0, w_pad=1.0, rect=(0.015, 0.07, 1, 1))
    name = "fig_learning_known" if known else "fig_learning"
    style.save(fig, name)
    plt.close(fig)
    return name


def fig_pit(R):
    """PIT histograms at the call for the 5-seed ensembles (a calibrated model gives a flat histogram)."""
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(style.WIDTH["text"], 1.8), sharey=True)
    models = [OURS, "WeibullMDN-sel", "DeepHit-CR"]
    any_data = False
    for ax, (d, title) in zip(axes, DATASETS.items()):
        ax.set_title(title)
        ax.axhline(1, color=style.INK2, lw=0.6, ls=":")
        df = read(R, f"folds_{d}_tau1_final*ens_f[0-9].csv")
        if df is None or "pit_h0" not in df:
            continue
        any_data = True
        mids = (np.arange(10) + 0.5) / 10
        for m in models:
            g = df[df.model.map(base_name) == m]
            if len(g):
                h = g[[f"pit_h{i}" for i in range(10)]].mean().to_numpy()
                h = h / h.sum() * 10
                ax.step(np.r_[0, mids + 0.05], np.r_[h[0], h], where="pre", label=fig_label(m), **{**line_style(m), "marker": None})
        ax.set_xlim(0, 1)
        ax.set_ylim(0.85, 1.15)  # same fixed range in every panel; tight enough to show the tail deviations
        ax.set_xlabel("PIT value")
    if not any_data:
        plt.close(fig)
        return None
    axes[0].set_ylabel("Density")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), bbox_to_anchor=(0.5, -0.08), handlelength=2.6)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    style.save(fig, "fig_pit")
    plt.close(fig)
    return "fig_pit"


def coverage_table(R):
    """Coverage of central 50/80/90% intervals at the call (5-seed ensembles)."""
    rows = {}
    for d in DATASETS:
        df = read(R, f"folds_{d}_tau1_final*ens*_f[0-9].csv")
        if df is None or "cover80" not in df:
            continue
        g = df.assign(base=df.model.map(base_name)).groupby("base")[["cover50", "cover80", "cover90"]].mean()
        for m in g.index:
            if m in NAMES:
                rows.setdefault(m, {})[d] = g.loc[m]
    if not rows:
        return None
    lines = [r"\begin{tabular}{l" + "rrr" * len(DATASETS) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{n}}}" for n in DATASETS.values()) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(DATASETS))),
             "Model & " + " & ".join(["50\\%", "80\\%", "90\\%"] * len(DATASETS)) + r" \\", r"\midrule"]
    for m in [m for m in NAMES if m in rows]:
        cells = []
        for d in DATASETS:
            v = rows[m].get(d)
            cells += ["--"] * 3 if v is None else [f"{100 * v[k]:.1f}" for k in ("cover50", "cover80", "cover90")]
        lines.append(NAMES[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


OPS_STATES = ["report", "arrival", "arrival+method-chain", "elapsed60+method"]


def ops_table(R):
    """Operational measures of EvoDur's forecast in each information state (value of each piece of information)."""
    blocks = []
    for d, title in DATASETS.items():
        e = read(R, f"evidence_states_{d}_final_f[0-9].csv")
        if e is None or "evidence_mae" not in e:
            continue
        g = e.groupby("state").mean(numeric_only=True)
        lines = []
        for st in [x for x in OPS_STATES if x in g.index]:
            r = g.loc[st]
            cells = [f"{r['evidence_mae']:.1f}", f"{r['evidence_width80']:.0f}", f"{100 * r['evidence_cover80']:.0f}",
                     f"{r['evidence_auc60']:.3f}" if pd.notna(r.get("evidence_auc60")) else "--"]
            lines.append(state_label(st) + " & " + " & ".join(cells) + r" \\")
        blocks.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{title}}}}} \\" + "\n" + "\n".join(lines))
    if not blocks:
        return None
    head = [r"\begin{tabular}{lcccc}", r"\toprule",
            r"State & MAE (min) & Width of 80\% interval (min) & Coverage of 80\% interval (\%) & AUC for $T > 60$ min \\",
            r"\midrule"]
    return "\n".join(head + ["\n\\midrule\n".join(blocks), r"\bottomrule", r"\end{tabular}"])


def incident_tests(R):
    """Paired tests on per-incident log scores (pooled over folds): ours minus each comparator."""
    from scipy import stats

    rows = []
    for d in DATASETS:
        zs = []
        for k in range(5):
            fi, fe = Path(R) / f"incidents_{d}_tau1_finalinc_f{k}.npz", Path(DERIVED) / f"incidents_{d}_tau1_evo_f{k}.npz"
            fc = Path(R) / f"incidents_{d}_tau1_finalcap_f{k}.npz"  # capacity-matched baselines (C23)
            if fi.exists() and fe.exists():
                z = dict(np.load(fi))
                z.update(dict(np.load(fe)))
                if fc.exists():
                    z.update(dict(np.load(fc)))
                fb = Path(R) / f"incidents_{d}_tau1_finalbl3_f{k}.npz"  # separate run of these baselines, if present
                if fb.exists():
                    z.update(dict(np.load(fb)))
                zs.append(z)
        if not zs:
            continue
        for state, others in (("report", ["WeibullMDN-sel", "DeepCAFT", "DeepCAFT-M", "DeepHit-CR", "Li2015", "AFT"]),
                              ("known", ["DeepCAFT", "DeepCAFT-M", "DeepHit-CR", "Li2015"])):
            for o in others:
                diff = np.concatenate([z[k] - z[k.replace(OURS, o)] for z in zs for k in z
                                       if k.startswith("0|") and k.endswith(f"|{OURS}|{state}") and k.replace(OURS, o) in z])  # repeat 0: each incident once
                if len(diff):
                    t_, p_ = stats.ttest_1samp(diff, 0)
                    rows.append({"data": d, "state": state, "vs": NAMES.get(o, o), "n": len(diff), "mean": diff.mean(),
                                 "se": diff.std(ddof=1) / np.sqrt(len(diff)), "t": t_, "p": p_,
                                 "share_better": float(np.mean(diff > 0))})
        es = [np.load(f) for f in sorted(glob.glob(str(Path(R) / f"evidence_incidents_{d}_final_f[0-9]_fold[0-9].npz")))]
        for st in [x for x in OPS_STATES if x != "report"] + ["elapsed90", "elapsed120"]:  # at the call both are the prior
            parts = [z[f"{st}|evidence"] - z[f"{st}|separate"] for z in es if f"{st}|evidence" in z.files]
            diff = np.concatenate(parts) if parts else []
            if len(diff):
                t_, p_ = stats.ttest_1samp(diff, 0)
                rows.append({"data": d, "state": st, "vs": "Separate model", "n": len(diff), "mean": diff.mean(),
                             "se": diff.std(ddof=1) / np.sqrt(len(diff)), "t": t_, "p": p_, "share_better": float(np.mean(diff > 0))})
    return pd.DataFrame(rows)


def incident_table(it):
    """Paired tests at the call and with the label known: mean difference per incident (EvoDur minus comparator) and
    p-value, one row per comparator and dataset."""
    it = it[it.state.isin(["report", "known"])]
    comps = list(dict.fromkeys(it.vs))
    lines = [r"\begin{tabular}{ll" + "r" * len(DATASETS) + "}", r"\toprule",
             "State & Comparator & " + " & ".join(DATASETS.values()) + r" \\", r"\midrule"]
    for state, title in (("report", "At the call"), ("known", "Label known")):
        rows = [c for c in comps if ((it.state == state) & (it.vs == c)).any()]
        for j, c in enumerate(rows):
            cells = []
            for d in DATASETS:
                r = it[(it.data == d) & (it.state == state) & (it.vs == c)]
                if r.empty:
                    cells.append("--")
                    continue
                r = r.iloc[0]
                p = "$<$0.001" if r.p < 0.001 else f"{r.p:.3f}"
                cells.append(f"{r['mean']:+.4f} ({p})")
            lines.append(f"{title if j == 0 else ''} & {c} & " + " & ".join(cells) + r" \\")
        if state == "report":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def derive_evodur(R):
    """Single-network EvoDur: in each (repeat, fold), the prior variant with the lowest validation NLL
    (rule fixed before the results). Reads R, writes rows and per-incident scores to DERIVED."""
    out = Path(DERIVED)
    out.mkdir(parents=True, exist_ok=True)
    for d in DATASETS:
        for k in range(5):
            parts = [Path(R) / f"folds_{d}_tau1_final_f{k}.csv", Path(R) / f"folds_{d}_tau1_variants_f{k}.csv"]
            if not all(p.exists() for p in parts):
                continue
            df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
            df = df[df.model.isin(VARIANTS)]
            rows, inc = [], {}
            for (rep, fold), g in df.groupby(["repeat", "fold"]):
                if set(g.model) != set(VARIANTS):
                    continue
                best = g.loc[g.val_nll.idxmin()].copy()
                best["chosen"], best["model"] = best["model"], OURS
                rows.append(best)
                src = {m: Path(R) / f"incidents_{d}_tau1_{'finalinc' if m == 'Evidence-DeepMix-8' else 'variants'}_f{k}.npz"
                       for m in VARIANTS}[best["chosen"]]
                if src.exists():
                    z = np.load(src)
                    for st in ("report", "known"):
                        key = f"{rep}|{fold}|{best['chosen']}|{st}"
                        if key in z.files:
                            inc[f"{rep}|{fold}|{OURS}|{st}"] = z[key]
            if rows:
                pd.DataFrame(rows).to_csv(out / f"folds_{d}_tau1_finalevo_f{k}.csv", index=False)
            if inc:
                np.savez_compressed(out / f"incidents_{d}_tau1_evo_f{k}.npz", **inc)


GAIN_STATES = [("report_logp", "At the\ncall"), ("known_logp", "Label\nknown"), ("lm30_known_logp", "Open\n30 min"),
               ("lm60_known_logp", "Open\n60 min"), ("lm120_known_logp", "Open\n120 min")]


def fig_gain(R):
    """EvoDur minus the best other model in each information state (single networks, mean over folds). One small
    panel per dataset: that dataset in blue, the other five in light gray behind it, on a shared vertical scale.
    From the label onwards only models that use the label compete; the open-at-s states also know the label.
    Points from fewer than 100 open test incidents per fold are drawn hollow."""
    curves = {}
    for d, title in DATASETS.items():
        df = read(R, f"folds_{d}_tau1_final*_f[0-9].csv")
        if df is None or OURS not in set(df.model):
            continue
        g = df[~df.model.str.contains("-ens")].groupby("model").mean(numeric_only=True)
        ys, thin = [], []
        for col, _ in GAIN_STATES:
            pool = [m for m in NAMES if m in g.index and m != OURS and (col == "report_logp" or m in USES_LABEL)]
            ys.append(g.loc[OURS, col] - g.loc[pool, col].max() if col in g else np.nan)
            ncol = col.replace("_known_logp", "_n")
            thin.append(ncol.startswith("lm") and ncol in g and g.loc[OURS, ncol] < 100)
        curves[title] = (np.array(ys), np.array(thin))
    fig, grid = plt.subplots(2, 3, figsize=(style.WIDTH["text"], 3.5), sharex=True, sharey=True)
    xs = np.arange(len(GAIN_STATES))
    blue = style.SERIES[0]
    for ax, (title, (ys, thin)) in zip(grid.ravel(), curves.items()):
        for other, (yo, _) in curves.items():
            if other != title:
                ax.plot(xs, yo, color="#d4d3cd", lw=0.9, zorder=1)
        ax.plot(xs, ys, color=blue, lw=1.8, zorder=3)
        ax.plot(xs[~thin], ys[~thin], ls="none", marker="o", ms=3.8, color=blue, mec="white", mew=0.6, zorder=4)
        ax.plot(xs[thin], ys[thin], ls="none", marker="o", ms=3.8, color=blue, mfc="white", mew=1.1, zorder=4)
        ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(0, 5), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7, color=style.INK)
        ax.set_title(title, fontsize=8)
        ax.axhline(0, color=style.INK2, lw=0.6, zorder=2)
        ax.grid(axis="x", visible=False)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Call", "Label", "30", "60", "120"])
        ax.set_xlim(-0.35, len(xs) - 0.65)
    ax.set_ylim(-0.01, 0.125)
    fig.supylabel("EvoDur minus best other model (log score per incident)", fontsize=8, x=0.005)
    grid[1, 1].set_xlabel("Information state: at the call, label known, still open at 30, 60 or 120 min")
    fig.tight_layout(h_pad=0.8, w_pad=0.8, rect=(0.015, 0, 1, 1))
    style.save(fig, "fig_gain")
    plt.close(fig)
    return "fig_gain"


def fig_summary(R, datasets=None):
    """Forest plot of fold-level log-score differences with corrected 95% intervals (Nadeau and Bengio).
    (a) at the call vs the best other model with a density, (b) label known vs the best other label-using model,
    (c) after police arrival vs the neural stage-wise model (open marker) and vs the better stage-wise model of earlier
    work, linear AFT or XGBoost-AFT (filled marker)."""
    datasets = datasets or DATASETS
    panels = [("At the call", "against the best other model"), ("Label known", "against the best competing-risk model"),
              ("Police arrival", "against stage-wise models")]
    rows = {p_[0]: [] for p_ in panels}
    for d, title in datasets.items():
        df = read(R, f"folds_{d}_tau1_final*_f[0-9].csv")
        if df is not None and OURS in set(df.model):
            df = df[~df.model.str.contains("-ens")]
            for ttl, col, pool in (("At the call", "report_logp", None), ("Label known", "known_logp", USES_LABEL)):
                piv = df.pivot_table(index=["repeat", "fold"], columns="model", values=col)
                others = [m for m in piv.columns if m != OURS and m in NAMES and (pool is None or m in pool)
                          and piv[m].notna().all()]
                best = max(others, key=lambda m: piv[m].mean())
                rows[ttl].append((title, [(piv[OURS] - piv[best]).to_numpy(), "filled", NAMES[best]]))
        e = read(R, f"evidence_states_{d}_final_f[0-9].csv")
        if e is not None and (e.state == "arrival").any():
            a = e[e.state == "arrival"]
            lin, xgb = a.separate_linear_logp.mean(), a.separate_xgb_logp.mean()
            old = a.separate_linear_logp if lin >= xgb else a.separate_xgb_logp
            rows["Police arrival"].append((title, [(a.evidence_logp - a.separate_model_logp).to_numpy(), "open", "neural"],
                                           [(a.evidence_logp - old).to_numpy(), "filled", "earlier"]))
    if not any(rows.values()):
        return None
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 3, figsize=(style.WIDTH["text"], 0.30 * len(datasets) + 1.25))
    names = list(datasets.values())
    blue, gray, light = style.SERIES[0], "#8a8984", "#f3f2ee"
    shape = {"neural": "o", "earlier": "D"}  # panel (c): which stage-wise model the marker compares with
    for k_, (ax, (ttl, sub)) in enumerate(zip(axes, panels)):
        for y0 in range(0, len(names), 2):  # alternating row shading
            ax.axhspan(y0 - 0.5, y0 + 0.5, color=light, lw=0, zorder=0)
        ax.axvline(0, color=style.INK, lw=0.8, zorder=1)
        for title, *marks in rows[ttl]:
            y0 = names.index(title)
            for j, (x, kind, who) in enumerate(marks):
                y = y0 + (0.2 * (2 * j - 1) if len(marks) > 1 else 0)
                m, hw = nb_ci(x)
                sig = m - hw > 0 or m + hw < 0  # the interval excludes zero
                col = blue if sig else gray
                ax.errorbar([m], [y], xerr=[[hw], [hw]], fmt=shape.get(who, "o"), ms=4.6 if who != "earlier" else 4.2, color=col,
                            elinewidth=1.1, capsize=0, mfc=col if sig else "white", mew=1.0, zorder=3)
        if ttl == "Police arrival":  # datasets without a recorded arrival time have no row by design
            have = {t_ for t_, *_ in rows[ttl]}
            lo, hi = ax.get_xlim()
            for y0, title in enumerate(names):
                if title not in have:
                    ax.text(0.5, y0, "n/a", transform=ax.get_yaxis_transform(), fontsize=7, color="#a9a8a3", va="center", ha="center")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names if ax is axes[0] else [])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_title(f"({'abc'[k_]}) {ttl}", fontsize=8, loc="left", pad=12, fontweight="bold")
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=7, color=style.INK2, va="bottom")
        ax.set_xlabel("Log score difference")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="white", lw=0.8, zorder=0.5)
        for sp in ("left",):
            ax.spines[sp].set_visible(False)
    h = [Line2D([], [], marker="o", ls="", color=blue, label="95% interval excludes 0"),
         Line2D([], [], marker="o", ls="", color=gray, mfc="white", label="includes 0"),
         Line2D([], [], marker="o", ls="", color=style.INK2, label="(c) neural stage-wise model"),
         Line2D([], [], marker="D", ls="", ms=3.8, color=style.INK2, label="(c) linear or XGBoost stage-wise")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7, handletextpad=0.2, columnspacing=1.6, frameon=False)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.07, 1, 1))
    style.save(fig, "fig_summary")
    plt.close(fig)
    # which competitor each marker refers to, for the caption
    print({ttl: [(t, [m[2] for m in ms]) for t, *ms in rows[ttl]] for ttl in rows})
    return "fig_summary"


def cost_table(R, datasets=None):
    """Mean training time per fold (s) on the cluster CPUs; EvoDur counts all four prior variants."""
    datasets = datasets or DATASETS
    cols = {}
    for d, title in datasets.items():
        fs = [f for tg in ("final", "variants", "finalnfg", "finalcap", "finalbl3") for f in glob.glob(str(Path(R) / f"folds_{d}_tau1_{tg}_f[0-9].csv"))]
        if not fs:
            continue
        redo, parts = RERUNS["bl3"], []
        has_rerun = any("finalbl3" in Path(f).name for f in fs)  # separate run of the baselines, if present
        for f in fs:  # its training times replace those of the same models
            df = pd.read_csv(f)
            hit = df.model.isin(redo)
            parts.append(df[hit] if "finalbl3" in Path(f).name else (df[~hit] if has_rerun else df))
        df = pd.concat(parts, ignore_index=True)
        df = df[~df.model.str.contains("-ens")]
        sec = df.groupby("model").seconds.mean()
        col = {OURS: sec[VARIANTS].sum() if all(v in sec for v in VARIANTS) else np.nan}
        for m in ("WeibullMDN-sel", "NeuralFineGray", "DeepHit-CR", "DeepCAFT", "DeepCAFT-M", "Li2015", "AFT", "XGBoost-AFT", "RSF", "DeepSurv"):
            col[m] = sec.get(m, np.nan)
        cols[title] = col
    if not cols:
        return None
    t = pd.DataFrame(cols)
    lines = [r"\begin{tabular}{l" + "r" * t.shape[1] + "}", r"\toprule", "Model & " + " & ".join(t.columns) + r" \\", r"\midrule"]
    for m, row in t.iterrows():
        cells = ["--" if pd.isna(v) else (f"{v:.0f}" if v >= 10 else f"{v:.1f}") for v in row]
        lines.append(NAMES.get(m, m) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def typeset_numbers(tex):
    """A true minus sign for negative numbers, and a plain 0 for a difference that rounds to zero."""
    import re

    tex = re.sub(r"(?<![\w.$])[+-]0(?![.\d])", "0", tex)
    return re.sub(r"(?<![\w$-])-(?=\d)", "$-$", tex)


def main():
    global DERIVED
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="directory with the result files of run.py / evidence.py (read only)")
    ap.add_argument("--out", required=True, help="output directory (tables/, figs/, _derived/ are created inside)")
    a = ap.parse_args()
    R, out = a.results, Path(a.out)
    TAB = out / "tables"
    TAB.mkdir(parents=True, exist_ok=True)
    DERIVED = out / "_derived"
    style.FIG_DIR = out / "figs"
    derive_evodur(R)
    outputs = {"tab_main": main_table(R), "tab_main_single": main_table(R, "single"), "tab_main_ens": main_table(R, "ens"),
               "tab_ablation": ablation_table(R), "tab_states": states_table(R), "tab_open": open_table(R), "tab_point": point_summary_table(R),
               "tab_point_known": point_known_table(R), "tab_point_minutes": point_minutes_table(R, TAB / "point_minutes_all.csv"),
               "tab_arrival": arrival_table(R), "tab_sens": sens_table(R), "tab_coverage": coverage_table(R),
               "tab_cost": cost_table(R)}  # tab_own, tab_ops, tab_stage_minutes, tab_delay: ops_analysis.py; tab_class: application.py
    for d in DATASETS:
        outputs[f"tab_metrics_{d}"] = other_metrics_table(R, d)
    for k, v in outputs.items():
        if v:
            v = typeset_numbers(v)
            (TAB / f"{k}.tex").write_text(v + "\n")
            print("wrote", k)
    it = incident_tests(R)
    if len(it):
        it.to_csv(TAB / "incident_tests.csv", index=False)
        (TAB / "tab_incident.tex").write_text(typeset_numbers(incident_table(it)) + "\n")
        print(it.round(4).to_string())
    w = paired_wins(R)
    if len(w):
        w.to_csv(TAB / "paired_wins.csv", index=False)
        print(w.round(4).to_string())
    for f in (fig_dynamic(R), fig_dynamic(R, known=True), fig_learning(R), fig_learning(R, known=True), fig_pit(R), fig_summary(R), fig_gain(R)):
        if f:
            print("wrote", f)


if __name__ == "__main__":
    main()
