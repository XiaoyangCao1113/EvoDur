#!/usr/bin/env python
"""Write the job list of the paper: one shell command per line, for any subset of the six datasets.

    python scripts/make_jobs.py                           # all six datasets, the groups the paper uses -> stdout
    python scripts/make_jobs.py seattle sanfrancisco -o jobs_public.txt
    python scripts/make_jobs.py ring4 --groups final,evidence --threads 2

The commands are the ones that produced the results of the paper (flags, models and tags as in the original cluster job
lists). Every dataset needs five fold jobs per group, so one command is one fold of one group; the commands are
independent and write to results/ (override with EVODUR_RESULTS). Run them with `scripts/run_sequential.sh jobs.txt` or as
a SLURM array (`scripts/slurm_array.sbatch.example`).

Per-dataset settings: 4th Ring Road (`ring4`) uses batch size 256 and three repetitions of the five folds in the cross-validated
runs; all other datasets use batch size 1024 and one repetition. The learning-curve and masked-network runs use the batch
size of the dataset.

Groups (what the results are used for):
  final       Tables main, ablation, states, point, coverage, cost, figures summary/gain/dynamic/learning
              (tags final, variants, finalcap, finalnfg, finalens, finalevoens, finalcapens, finalensnfg, finalinc, pointk, pointkaft, pointknfg)
              and the evidence-update experiment (evidence.py, tag _final)
  learning    learning curves (tags lc, lccap, lcevo with 10, 25 and 50% of the training data)
  sens        tau = 15 min, records shorter than 15 min removed (tags sens, sensxgb); not reported in the paper and
              not in the default groups
  interpret   interpretation analysis (interpret.py: counterfactual effects of call-time variables and of the label /
              arrival, permutation importance), one job per dataset and fold
"""
from __future__ import annotations

import argparse
import sys

DATASETS = ["ring4", "citywide", "ny511early", "sydney", "seattle", "sanfrancisco"]
WITH_ARRIVAL = {"ring4", "citywide", "seattle", "sanfrancisco"}
BATCH = {"ring4": 256}  # all others 1024
REPEATS = {"ring4": 3}  # repetitions of the five folds in the final runs; all others 1
FOLDS = range(5)
LM = "--landmarks 10,20,30,45,60,90,120"
MAIN = ("AFT,Li2015,DeepAFT,DeepCAFT,DeepMix-2,DeepMix-4,DeepMix-8,DeepMix-16,DeepMix-8-constpi,Mix-linear-8,"
        "Evidence-AFT,Evidence-DeepMix-8,DeepHit-CR,WeibullMDN-4,XGBoost-AFT,RSF,DeepSurv")
SENS = ("AFT,Li2015,DeepAFT,DeepCAFT,DeepMix-4,DeepMix-8,DeepMix-4-constpi,Evidence-AFT,Evidence-DeepMix-4,Evidence-DeepMix-8,"
        "DeepHit-CR,WeibullMDN-4,XGBoost-AFT,RSF,DeepSurv")
VARIANTS = "Evidence-DeepMix-8-het,Evidence-DeepMix-16,Evidence-DeepMix-16-het"
FRACS = ["0.1", "0.25", "0.5"]


def run_final(d, k, threads):
    """run.py, tau = 1, one fold (all repetitions of it)."""
    b, r = BATCH.get(d, 1024), REPEATS.get(d, 1)
    base = f"python -m evodur.run --data {d} --tau 1 --folds 5 --repeats {r} --batch {b} --threads {threads} --only-folds {k}"
    ens = "--n-seeds 5"
    # (tag, extra flags, models)
    spec = [("final", LM, MAIN), ("finalcap", LM, "DeepCAFT-M,WeibullMDN-sel"), ("variants", LM, VARIANTS),
            ("pointk", LM, "Li2015,DeepCAFT,DeepHit-CR,EvoDur"), ("pointkaft", LM, "Evidence-AFT"), ("finalnfg", LM, "NeuralFineGray"),
            ("pointknfg", LM, "NeuralFineGray"), ("finalcapens", f"{LM} {ens}", "DeepCAFT-M,WeibullMDN-sel"),
            ("finalens", f"{LM} {ens}", "Evidence-DeepMix-8,WeibullMDN-4,DeepHit-CR,DeepCAFT"),
            ("finalevoens", f"{LM} {ens}", "EvoDur"), ("finalensnfg", f"{LM} {ens}", "NeuralFineGray"),
            ("finalinc", "", "AFT,Li2015,DeepCAFT,DeepHit-CR,WeibullMDN-4,Evidence-DeepMix-8")]
    return [f"{base} {x + ' ' if x else ''}--models {m} --tag {t}_f{k}" for t, x, m in spec]


def evidence_final(d, k, threads):
    return [f"python -m evodur.evidence --data {d} --tau 1 --folds {k} --batch {BATCH.get(d, 1024)} --threads {threads} --tag _final_f{k}"]


def learning(d, k, threads):
    b = BATCH.get(d, 1024)
    base = f"python -m evodur.run --data {d} --tau 1 --folds 5 --repeats 1 --batch {b} --threads {threads} --only-folds {k}"
    out = []
    for f in FRACS:
        out += [f"{base} --train-frac {f} --models AFT,WeibullMDN-4,DeepHit-CR,DeepCAFT,Evidence-DeepMix-8 --tag lc{f}_f{k}",
                f"{base} --train-frac {f} --models DeepCAFT-M,NeuralFineGray --tag lccap{f}_f{k}",
                f"{base} --train-frac {f} --models EvoDur --tag lcevo{f}_f{k}"]
    return out


def sens(d, k, threads):
    b, r = BATCH.get(d, 1024), REPEATS.get(d, 1)
    base = f"python -m evodur.run --data {d} --tau 15 --folds 5 --repeats {r} --batch {b} --threads {threads} --only-folds {k}"
    return [f"{base} --models {SENS} --tag sens_f{k}", f"{base} --models AFT,XGBoost-AFT --tag sensxgb_f{k}"]


def ops(d, k, threads):
    # batch 256 for all datasets, as in the original runs
    return [f"python -m evodur.evidence --data {d} --tau 1 --folds {k} --batch 256 --threads {threads} --tag {t}_f{k} --save-detail"
            for t in ("_ops", "_ops2")]


def masked(d, k, threads):
    b = BATCH.get(d, 1024)
    base = f"python -m evodur.masked --data {d} --tau 1 --folds {k} --batch {b} --threads {threads}"
    return [f"{base} --tag _masked_f{k}", f"{base} --tag _maskedg_f{k} --mix group --skip-weibull"]


def interpret(d, k, threads):
    return [f"python -m evodur.interpret --data {d} --tau 1 --folds {k} --batch {BATCH.get(d, 1024)} --threads {threads}"]


GROUPS = ["final", "learning", "sens", "ops", "masked", "interpret"]
DEFAULT_GROUPS = [g for g in GROUPS if g != "sens"]


def jobs(d, groups, threads):
    out = []
    for g in groups:
        for k in FOLDS:
            if g == "final":
                out += run_final(d, k, threads) + evidence_final(d, k, threads)
            elif g == "learning":
                out += learning(d, k, threads)
            elif g == "sens":
                out += sens(d, k, threads)
            elif g == "ops" and d in WITH_ARRIVAL:
                out += ops(d, k, threads)
            elif g == "masked":
                out += masked(d, k, threads)
            elif g == "interpret":
                out += interpret(d, k, threads)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("datasets", nargs="*", default=DATASETS, help=f"subset of {DATASETS} (default: all)")
    ap.add_argument("--groups", default=",".join(DEFAULT_GROUPS), help=f"comma-separated subset of {GROUPS}")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads per job (the original runs used 4)")
    ap.add_argument("-o", "--out", default="-", help="output file (default: stdout)")
    a = ap.parse_args()
    bad = [d for d in a.datasets if d not in DATASETS] + [g for g in a.groups.split(",") if g not in GROUPS]
    if bad:
        sys.exit(f"unknown dataset or group: {bad}")
    lines = [c for d in a.datasets for c in jobs(d, a.groups.split(","), a.threads)]
    text = "\n".join(lines) + "\n"
    if a.out == "-":
        sys.stdout.write(text)
    else:
        with open(a.out, "w") as fh:
            fh.write(text)
        print(f"wrote {len(lines)} commands to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
