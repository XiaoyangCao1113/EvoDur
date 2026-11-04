"""Operational analyses from the information-state runs with --save-detail (tag _ops_f<k>).

(1) Forecast quality in minutes in each information state, for EvoDur and the stage-wise models
    (neural, linear AFT, XGBoost-AFT): MAE of the median, median and mean 80% interval width, coverage,
    Brier score and AUC for P(T > 60 min), log score.
(2) Label delay: at elapsed time s, with the police on scene, the label is used only if it was reported
    by s, i.e. r + d <= s for a delay d after arrival. The stage-wise neural model gets the same rule
    (its label model if the label is in, its no-label model otherwise).

Usage: python -m evodur.analysis.ops_analysis --results <dir> --out <dir>
(writes ops_minutes.csv, ops_delay.csv, ops_same_incidents.csv and the tables tab_own, tab_stage_minutes, tab_ops, tab_delay)
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATASETS = {"ring4": "Beijing 4th Ring", "citywide": "Beijing citywide", "seattle": "Seattle", "sanfrancisco": "San Francisco"}
STATES = ["report", "arrival", "method", "arrival+method-chain"]
WHO = ["evidence", "separate", "linear", "xgb"]
DELAYS = [0, 10, 20, 30, 60, np.inf]
LANDMARKS = [30, 45, 60, 90, 120]


def load(R, ds):
    out = {}
    for f in sorted(glob.glob(str(Path(R) / f"evidence_detail_{ds}_ops_f[0-9]_fold[0-9].npz"))):
        k = int(f.split("_fold")[-1].split(".")[0])
        out[k] = dict(np.load(f))
    return out


def measures(z, state, who):
    t, low = z[f"{state}|t"], z[f"{state}|low"]
    q10, q50, q90, p60 = (z[f"{state}|{who}|{q}"] for q in ("q10", "q50", "q90", "p60"))
    m = {"lp": float(np.mean(z[f"{state}|{who}|lp"])), "mae": float(np.mean(np.abs(q50 - t))),
         "width_med": float(np.median(q90 - q10)), "width_mean": float(np.mean(q90 - q10)),
         "cover80": float(np.mean((t > q10) & (t <= q90))), "n": len(t),
         # interval score of the central 80% interval (Gneiting and Raftery, 2007) and quantile loss of the 90% quantile
         "iscore": float(np.mean((q90 - q10) + 10 * np.clip(q10 - t, 0, None) + 10 * np.clip(t - q90, 0, None))),
         "pin90": float(np.mean(np.maximum(0.9 * (t - q90), -0.1 * (t - q90))))}
    u = low < 60
    y = (t[u] > 60).astype(float)
    if u.sum() and 0 < y.mean() < 1:
        m["brier60"] = float(np.mean((p60[u] - y) ** 2))
        m["auc60"] = float(roc_auc_score(y, p60[u]))
    return m


def minutes_table(R):
    rows = []
    for ds in DATASETS:
        for k, z in load(R, ds).items():
            for st in STATES:
                for who in WHO:
                    if f"{st}|{who}|q50" in z:
                        rows.append({"data": ds, "fold": k, "state": st, "who": who, **measures(z, st, who)})
    return pd.DataFrame(rows)


def delay_table(R):
    rows = []
    for ds in DATASETS:
        for k, z in load(R, ds).items():
            for s in LANDMARKS:
                a, b = f"arrival+elapsed{s}", f"arrival+elapsed{s}+method"
                if f"{a}|idx" not in z or f"{b}|idx" not in z:
                    continue
                assert np.array_equal(z[f"{a}|idx"], z[f"{b}|idx"]), (ds, k, s)
                t, r = z[f"{a}|t"], z[f"{a}|r"]
                for d in DELAYS:
                    use = (r + d) <= s
                    for who in ("evidence", "separate"):
                        lp = np.where(use, z[f"{b}|{who}|lp"], z[f"{a}|{who}|lp"])
                        q50 = np.where(use, z[f"{b}|{who}|q50"], z[f"{a}|{who}|q50"])
                        rows.append({"data": ds, "fold": k, "s": s, "delay": d, "who": who, "share_label": float(use.mean()),
                                     "lp": float(lp.mean()), "mae": float(np.mean(np.abs(q50 - t))), "n": len(t)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- tables for the paper
def _subset(z, state, who, keep_idx):
    """Measures of one model in one state, restricted to the incidents in keep_idx."""
    m = np.isin(z[f"{state}|idx"], keep_idx)
    zz = {f"{state}|{k}": z[f"{state}|{k}"][m] for k in ("t", "low")}
    zz.update({f"{state}|{who}|{q}": z[f"{state}|{who}|{q}"][m] for q in ("q10", "q50", "q90", "p60", "lp")})
    return measures(zz, state, who)


def paper_tables(R, out):
    """Same incidents in every row: those with a recorded arrival time between the call and the end."""
    rows = []
    for ds in DATASETS:
        for k, z in load(R, ds).items():
            keep = z["arrival|idx"]
            for st in ("report", "method", "arrival", "arrival+method-chain"):
                for who in WHO:
                    if f"{st}|{who}|q50" in z:
                        rows.append({"data": ds, "fold": k, "state": st, "who": who, **_subset(z, st, who, keep)})
            for who in WHO:
                st = "arrival+elapsed60+method"
                if f"{st}|{who}|q50" in z:
                    rows.append({"data": ds, "fold": k, "state": st, "who": who, **measures(z, st, who)})
    df = pd.DataFrame(rows)
    df.to_csv(Path(out) / "ops_same_incidents.csv", index=False)
    g = df.groupby(["data", "state", "who"]).mean(numeric_only=True)
    name = {"report": "Call", "method": "Label", "arrival": "Police arrival", "arrival+method-chain": "Arrival and label",
            "arrival+elapsed60+method": "Open at 60 min, arrival and label"}
    pct = lambda a, b: f"({100 * (b - a) / a:+.0f}\\%)".replace("+", "$+$").replace("-", "$-$")

    # main text: EvoDur's own forecast at the call, after arrival, after arrival and label
    L = [r"\begin{tabular}{lccccccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{MAE (min)} & \multicolumn{3}{c}{Width of 80\% interval (min)} & \multicolumn{3}{c}{Coverage (\%)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         r"Dataset & Call & Arrival & Arrival, label & Call & Arrival & Arrival, label & Call & Arrival & Arrival, label \\", r"\midrule"]
    for ds, nm in DATASETS.items():
        e = lambda st, col: g.loc[(ds, st, "evidence"), col]
        c0, c1, c2 = (e(s, "mae") for s in ("report", "arrival", "arrival+method-chain"))
        w0, w1, w2 = (e(s, "width_med") for s in ("report", "arrival", "arrival+method-chain"))
        v = [100 * e(s, "cover80") for s in ("report", "arrival", "arrival+method-chain")]
        L.append(f"{nm} & {c0:.1f} & {c1:.1f} {pct(c0, c1)} & {c2:.1f} {pct(c0, c2)} & {w0:.0f} & {w1:.0f} {pct(w0, w1)} & "
                 f"{w2:.0f} {pct(w0, w2)} & {v[0]:.0f} & {v[1]:.0f} & {v[2]:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (Path(out) / "tab_own.tex").write_text("\n".join(L) + "\n")

    # main text: after arrival and label, EvoDur against the stage-wise models (bold: best per dataset and measure)
    who_nm = [("evidence", "EvoDur"), ("separate", "Neural stage-wise"), ("linear", "Linear AFT stage-wise"), ("xgb", "XGBoost-AFT stage-wise")]
    cols = [("iscore", "{:.0f}", min), ("pin90", "{:.1f}", min), ("cover80", "{:.0f}", None), ("width_med", "{:.0f}", None), ("mae", "{:.1f}", min)]
    L = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"Dataset & Model & Interval score & 90\% quantile loss & Coverage (\%) & Width (min) & MAE (min) \\", r"\midrule"]
    for ds, nm in DATASETS.items():
        sub = g.loc[(ds, "arrival+method-chain")]
        for i, (w, wn) in enumerate(who_nm):
            cells = []
            for c, f, best in cols:
                v = sub.loc[w, c] * (100 if c == "cover80" else 1)
                txt = f.format(v)
                if best is not None and f.format(sub.loc[w, c]) == f.format(sub[c].min()):
                    txt = rf"\textbf{{{txt}}}"
                cells.append(txt)
            L.append((nm if i == 0 else "") + f" & {wn} & " + " & ".join(cells) + r" \\")
        if ds != list(DATASETS)[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    (Path(out) / "tab_stage_minutes.tex").write_text("\n".join(L) + "\n")

    # appendix: EvoDur in every state (replaces the earlier operational table)
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"State & MAE (min) & Width of 80\% interval (min) & Coverage (\%) & Brier score, $T > 60$ min & AUC, $T > 60$ min \\",
         r"\midrule"]
    for ds, nm in DATASETS.items():
        L.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{nm}}}}} \\")
        for st in ("report", "method", "arrival", "arrival+method-chain", "arrival+elapsed60+method"):
            r_ = g.loc[(ds, st, "evidence")]
            b = "--" if pd.isna(r_.get("brier60")) else f"{r_.brier60:.3f}"
            a = "--" if pd.isna(r_.get("auc60")) else f"{r_.auc60:.3f}"
            L.append(f"{name[st]} & {r_.mae:.1f} & {r_.width_med:.0f} & {100 * r_.cover80:.0f} & {b} & {a} \\\\")
        if ds != list(DATASETS)[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    (Path(out) / "tab_ops.tex").write_text("\n".join(L) + "\n")

    # appendix: label delay
    sg = lambda v, n: ("$-$" if round(v, n) < 0 else "") + f"{abs(v):.{n}f}"
    dt = delay_table(R)
    dt["delay"] = dt["delay"].replace(np.inf, -1)
    d = dt.groupby(["data", "s", "delay", "who"])[["share_label", "lp", "mae"]].mean().unstack("who")
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Dataset & Open at & Delay (min) & With label (\%) & Gain from label ($\times 10^{-3}$) & Gain from label (min) & EvoDur $-$ neural stage-wise ($\times 10^{-3}$) \\",
         r"\midrule"]
    for ds, nm in DATASETS.items():
        for s in (30, 60):
            if (ds, s, -1) not in d.index:
                continue
            base = d.loc[(ds, s, -1)]
            first = True
            for dl in (0, 10, 30, -1):
                if (ds, s, dl) not in d.index:
                    continue
                r_ = d.loc[(ds, s, dl)]
                gl = 1000 * (r_[("lp", "evidence")] - base[("lp", "evidence")])
                gm = base[("mae", "evidence")] - r_[("mae", "evidence")]
                es = 1000 * (r_[("lp", "evidence")] - r_[("lp", "separate")])
                L.append(f"{nm if (first and s == 30) else ''} & {f'{s} min' if first else ''} & "
                         f"{'never' if dl == -1 else int(dl)} & {100 * r_[('share_label', 'evidence')]:.0f} & "
                         f"{sg(gl, 1)} & {sg(gm, 2)} & {sg(es, 1)} \\\\")
                first = False
        if ds != list(DATASETS)[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    txt = "\n".join(L) + "\n"
    (Path(out) / "tab_delay.tex").write_text(txt)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mt = minutes_table(a.results)
    mt.to_csv(out / "ops_minutes.csv", index=False)
    dt = delay_table(a.results)
    dt.to_csv(out / "ops_delay.csv", index=False)
    pd.set_option("display.width", 220)
    g = mt.groupby(["data", "state", "who"])[["lp", "mae", "width_med", "width_mean", "cover80", "brier60", "auc60"]].mean()
    print(g.round(3).to_string())
    d = dt.groupby(["data", "s", "delay", "who"])[["share_label", "lp", "mae"]].mean().unstack("who")
    print(d.round(3).to_string())
    paper_tables(a.results, out)  # tab_own, tab_stage_minutes, tab_ops, tab_delay


if __name__ == "__main__":
    main()
