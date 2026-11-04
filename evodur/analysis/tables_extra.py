"""Tables masked and example of the paper.

    python -m evodur.analysis.tables_extra --results results --out paper_out/tables

tab_masked   EvoDur minus the masked-input single network (run by masked.py), log score x 1000 per incident, with the
             number of folds in which EvoDur is better. EvoDur's score is `evidence_logp` of evidence_states_{d}_final_f*.csv;
             the network's is `masked_logp` of masked_states_{d}_masked_f*.csv and masked_states_{d}_maskedg_f*.csv
             (uniform and group-weighted state sampling). In each fold and state the network is the scheme with the
             higher validation log score (`masked_val_logp`).
tab_example  Forecasts of one held-out Seattle incident in four information states. The numbers come from the
             --save-detail files of evidence.py (tag _ops2), the incident is chosen by a fixed rule in
             application.worked_example; the description of the call and the label come from the Seattle data, so this table
             needs the data (EVODUR_DATA) and is skipped without it.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from evodur.analysis.application import worked_example

DATASETS = {"ring4": "Beijing 4th Ring", "citywide": "Beijing citywide", "ny511early": "NY/NJ/CT 511",
            "sydney": "Sydney", "seattle": "Seattle", "sanfrancisco": "San Francisco"}
STATES = [("report", "Call"), ("method", "Label"), ("arrival", "Police arrival"), ("arrival+method-chain", "Arrival and label"),
          ("elapsed30+method", "Label, open 30 min"), ("elapsed60+method", "Label, open 60 min"),
          ("elapsed120+method", "Label, open 120 min")]
SCHEMES = ("masked", "maskedg")


def _read(R, pattern):
    files = sorted(glob.glob(str(Path(R) / pattern)))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def masked_cell(R, d):
    """{state: 'mean difference x 1000 (wins/folds)'} for dataset d (names of the result files)."""
    ev = _read(R, f"evidence_states_{d}_final_f[0-9].csv")
    sch = {s: _read(R, f"masked_states_{d}_{s}_f[0-9].csv") for s in SCHEMES}
    if ev is None or any(v is None for v in sch.values()):
        return {}
    ev = ev.set_index(["state", "fold"]).evidence_logp
    both = pd.concat([m.assign(scheme=s) for s, m in sch.items()]).set_index(["state", "fold"])
    out = {}
    for state, _ in STATES:
        if state not in both.index.get_level_values(0):
            continue
        b = both.loc[state]
        pick = b.sort_values("masked_val_logp").groupby(level="fold").tail(1).masked_logp  # higher validation score
        folds = ev.loc[state].index.intersection(pick.index)
        best = (ev.loc[state].loc[folds] - pick.loc[folds]).to_numpy() if len(folds) else None
        if best is not None:
            v = int(round(1000 * best.mean()))
            sign = f"$+${v}" if v > 0 else (f"$-${-v}" if v < 0 else "0")
            out[state] = f"{sign} ({int((best > 0).sum())}/{len(best)})"
    return out


def masked_table(R):
    cells = {d: masked_cell(R, d) for d in DATASETS}
    if not any(cells.values()):
        return None
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule", "Information state & " + " & ".join(DATASETS.values()) + r" \\", r"\midrule"]
    for state, name in STATES:
        lines.append(name + " & " + " & ".join(cells[d].get(state, "--") for d in DATASETS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- example
CALL_TEXT = {"obs - mvc - with injuries (includes hit and run)": "collision with injuries"}


def _prob(v):
    return f"{v:.2f}"


def example_table(R, out):
    """Table of the worked Seattle example. Needs the Seattle data for the call description and the label."""
    from evodur.evidence import load_with_stages
    from evodur.public_data import SEATTLE_METHODS

    df = worked_example(R, out)  # also writes app_example.csv
    data, _ = load_with_stages("seattle", 1.0)
    i = int(df.incident.iloc[0])
    row, c = data.frame.iloc[i], int(data.c[i])
    assert abs(float(data.t[i]) - float(df.t.iloc[0])) < 1e-2  # same incident as in the detail files (float32)
    r = df.set_index("state")
    call = CALL_TEXT.get(row["initial"], row["initial"])
    info = {"report": ("0 min", f"Call: {call}, priority {row['priority']}"),
            "arrival": (f"{r.loc['arrival', 'r']:.0f} min", "Police arrive"),
            "arrival+method-chain": ("After arrival", f"Label: {SEATTLE_METHODS[c].split(' (')[0]}"),
            "arrival+elapsed60+method": ("60 min", "Still open")}
    L = [r"\begin{tabular}{llcccccc}", r"\toprule", r" & & \multicolumn{3}{c}{EvoDur} & \multicolumn{2}{c}{XGBoost-AFT stage-wise} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
         r"Time & Information & Median & 80\% interval & $P(\text{major})$ & 80\% interval & $P(\text{major})$ \\", r"\midrule"]
    for st, (when, what) in info.items():
        e = r.loc[st]
        xg = "-- & --" if pd.isna(e.xgb_q10) else f"{e.xgb_q10:.0f}--{e.xgb_q90:.0f} & {_prob(e.xgb_p120)}"
        L.append(f"{when} & {what} & {e.evidence_q50:.0f} & {e.evidence_q10:.0f}--{e.evidence_q90:.0f} & {_prob(e.evidence_p120)} & {xg} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True, help="directory of the .tex tables")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = masked_table(a.results)
    if t:
        (out / "tab_masked.tex").write_text(t + "\n")
        print("wrote tab_masked")
    try:
        t = example_table(a.results, out)
    except FileNotFoundError as e:
        print("skipped tab_example (needs the Seattle data):", e)
        t = None
    if t:
        (out / "tab_example.tex").write_text(t + "\n")
        print("wrote tab_example")


if __name__ == "__main__":
    main()
