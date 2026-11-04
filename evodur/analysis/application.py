"""Use in incident management, from the information-state runs with --save-detail (tag _ops2_f<k>).

(1) Incident class: MUTCD Section 6I.01 classes minor (< 30 min), intermediate (30 min to 2 h) and major (> 2 h).
    Ranked probability score (RPS) of the forecast class probabilities, and Brier score / AUC for "major".
(2) Cost-loss decision: act (e.g. sign a long diversion) when P(major) > C/L. Relative economic value
    V = (E_clim - E) / (E_clim - E_perfect) (Richardson, 2000), per fold, for a range of C/L.
(3) A worked incident: the held-out major Seattle incident of fold 0 (with an arrival time) at the median of
    EvoDur's change in P(major) from the call to arrival and label.

Usage: python -m evodur.analysis.application --results <dir> --out <dir> [--figs <dir>]
(writes app_class.csv, app_value.csv, app_example.csv, tab_class.tex and the figures fig_value, fig_value_beijing)
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATASETS = {"ring4": "Beijing 4th Ring", "citywide": "Beijing citywide", "seattle": "Seattle", "sanfrancisco": "San Francisco"}
STATES = {"report": "Call", "arrival": "Police arrival", "arrival+method-chain": "Arrival and label"}
WHO = {"evidence": "EvoDur", "separate": "Neural stage-wise", "linear": "Linear AFT stage-wise", "xgb": "XGBoost-AFT stage-wise"}
RATIOS = np.round(np.arange(1, 100) / 100, 2)  # 0.01 ... 0.99


def load(R, ds):
    return {int(f.split("_fold")[-1][0]): dict(np.load(f))
            for f in sorted(glob.glob(str(Path(R) / f"evidence_detail_{ds}_ops2_f[0-9]_fold[0-9].npz")))}


def subset(z, st, keep):
    m = np.isin(z[f"{st}|idx"], keep)
    return m


def class_scores(t, p30, p120):
    """RPS over the three ordered classes and Brier/AUC for major."""
    c1, c2 = 1 - p30, 1 - p120  # cumulative P(class <= minor), P(class <= intermediate)
    o1, o2 = (t <= 30).astype(float), (t <= 120).astype(float)
    rps = ((c1 - o1) ** 2 + (c2 - o2) ** 2) / 2
    y = (t > 120).astype(float)
    return {"rps": rps.mean(), "brier_major": np.mean((p120 - y) ** 2),
            "auc_major": roc_auc_score(y, p120) if 0 < y.mean() < 1 else np.nan, "share_major": y.mean()}


def value(p, y, r):
    """Relative economic value of acting when p > r, with cost C = r and loss L = 1."""
    act = p > r
    e = np.mean(act * r + (~act) * y)
    e_clim = min(r, y.mean())
    e_perf = y.mean() * r
    return (e_clim - e) / (e_clim - e_perf)


def worked_example(R, out):
    out = Path(out)
    z = load(R, "seattle")[0]
    keep = z["arrival|idx"]
    states = ["report", "arrival", "arrival+method-chain", "arrival+elapsed60+method"]
    t = z["arrival|t"]
    major = np.where(t > 120)[0]
    ids = z["arrival|idx"][major]
    pos = lambda st: {i: j for j, i in enumerate(z[f"{st}|idx"])}
    pr, pl = pos("report"), pos("arrival+method-chain")
    d = np.array([z["arrival+method-chain|evidence|p120"][pl[i]] - z["report|evidence|p120"][pr[i]] for i in ids])
    pick = ids[np.argsort(d)[len(d) // 2]]
    rows = []
    for st in states:
        p = pos(st)
        if pick not in p:
            continue
        j = p[pick]
        rows.append({"state": st, "t": float(z[f"{st}|t"][j]), "r": float(z[f"{st}|r"][j]) if f"{st}|r" in z else np.nan,
                     **{f"{w}_{q}": float(z[f"{st}|{w}|{q}"][j]) for w in ("evidence", "xgb") if f"{st}|{w}|q50" in z
                        for q in ("q10", "q50", "q90", "p30", "p120")}})
    df = pd.DataFrame(rows)
    df["incident"] = int(pick)  # row of the incident in the loaded Seattle data (used by tables_extra.example_table)
    df.to_csv(out / "app_example.csv", index=False)
    print("worked example: incident", int(pick))
    print(df.round(3).to_string())
    return df


# ----------------------------------------------------------------------------- outputs for the paper
def paper_outputs(R, tab_dir, fig_name="fig_value", fig_dir=None):
    from evodur.figures import style
    import matplotlib.pyplot as plt

    if fig_dir is not None:
        style.FIG_DIR = Path(fig_dir)

    rows, vrows = [], []
    for ds in DATASETS:
        for k, z in load(R, ds).items():
            keep = z["arrival|idx"]
            for st in STATES:
                m = subset(z, st, keep)
                t = z[f"{st}|t"][m]
                for w in WHO:
                    if f"{st}|{w}|p120" not in z:
                        continue
                    p30, p120 = z[f"{st}|{w}|p30"][m], z[f"{st}|{w}|p120"][m]
                    rows.append({"data": ds, "fold": k, "state": st, "who": w, **class_scores(t, p30, p120)})
                    if w == "evidence":
                        y = (t > 120).astype(float)
                        vrows += [{"data": ds, "fold": k, "state": st, "ratio": r, "value": value(p120, y, r)} for r in RATIOS]
    cs, vs = pd.DataFrame(rows), pd.DataFrame(vrows)
    g = cs.groupby(["data", "state", "who"]).mean(numeric_only=True)

    # table: class forecast, EvoDur in three states and the stage-wise models after arrival and label
    L = [r"\begin{tabular}{lcccccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{EvoDur} & \multicolumn{3}{c}{Stage-wise, arrival and label} & \multicolumn{2}{c}{AUC, major (EvoDur)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
         r"Dataset & Call & Arrival & Arrival, label & Neural & Linear AFT & XGBoost-AFT & Call & Arrival, label \\", r"\midrule"]
    for ds, nm in DATASETS.items():
        e = lambda st, w="evidence", c="rps": g.loc[(ds, st, w), c]
        best = min(e("arrival+method-chain", w) for w in WHO)
        f = lambda v: rf"\textbf{{{v:.3f}}}" if f"{v:.3f}" == f"{best:.3f}" else f"{v:.3f}"
        L.append(f"{nm} ({100 * e('report', c='share_major'):.0f}\\% major) & {e('report'):.3f} & {e('arrival'):.3f} & "
                 f"{f(e('arrival+method-chain'))} & " + " & ".join(f(e("arrival+method-chain", w)) for w in ("separate", "linear", "xgb"))
                 + f" & {e('report', c='auc_major'):.2f} & {e('arrival+method-chain', c='auc_major'):.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (Path(tab_dir) / "tab_class.tex").write_text("\n".join(L) + "\n")

    # figure: relative economic value of EvoDur in three states
    style.apply()
    v = vs.groupby(["data", "state", "ratio"]).value.mean()
    # vertical range fitted to each pair of datasets
    for name, sets, ylim in ((fig_name, ("seattle", "sanfrancisco"), (-0.02, 0.66)),
                             (fig_name + "_beijing", ("ring4", "citywide"), (-0.02, 0.52))):
        fig, axs = plt.subplots(1, 2, figsize=(style.WIDTH["onehalf"], 2.3), sharey=True)
        for ax, ds in zip(axs, sets):
            for k, st in enumerate(STATES):
                y = v.loc[(ds, st)]
                ax.plot(y.index, y.values, label=STATES[st], **style.series_style(k))
            ax.axhline(0, color=style.INK2, lw=0.6)
            ax.set_title(DATASETS[ds])
            ax.set_xlabel("Cost of acting / loss avoided ($C/L$)")
            if name.endswith("_beijing"):  # about 3% of the incidents are major, so the value peaks at small C/L: log axis
                ax.set_xscale("log")
                ax.set_xlim(0.01, 1)
                ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1])
                ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
                ax.minorticks_off()
            else:
                ax.set_xlim(0, 1)
        axs[0].set_ylim(*ylim)
        axs[0].set_ylabel("Relative economic value")
        axs[1].legend(loc="upper right")
        fig.tight_layout()
        style.save(fig, name)
        plt.close(fig)
    return cs, vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=".", help="directory for the tables and csv files")
    ap.add_argument("--figs", default=None, help="directory for fig_value and fig_value_beijing (default: <out>/../figs)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, vrows = [], []
    for ds in DATASETS:
        for k, z in load(a.results, ds).items():
            keep = z["arrival|idx"]  # the same incidents in every state
            for st in STATES:
                m = subset(z, st, keep)
                t = z[f"{st}|t"][m]
                for w in WHO:
                    if f"{st}|{w}|p120" not in z:
                        continue
                    p30, p120 = z[f"{st}|{w}|p30"][m], z[f"{st}|{w}|p120"][m]
                    rows.append({"data": ds, "fold": k, "state": st, "who": w, **class_scores(t, p30, p120)})
                    y = (t > 120).astype(float)
                    for r in RATIOS:
                        vrows.append({"data": ds, "fold": k, "state": st, "who": w, "ratio": r, "value": value(p120, y, r)})
    cs, vs = pd.DataFrame(rows), pd.DataFrame(vrows)
    cs.to_csv(out / "app_class.csv", index=False)
    vs.to_csv(out / "app_value.csv", index=False)
    pd.set_option("display.width", 220)
    print(cs.groupby(["data", "state", "who"])[["rps", "brier_major", "auc_major", "share_major"]].mean().round(4).to_string())
    print(vs.groupby(["data", "state", "who", "ratio"]).value.mean().unstack("ratio").round(3).to_string())
    worked_example(a.results, out)
    paper_outputs(a.results, out, fig_dir=a.figs or out.resolve().parent / "figs")  # tab_class, fig_value, fig_value_beijing


if __name__ == "__main__":
    main()
