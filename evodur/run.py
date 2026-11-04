"""Paired cross-validation of the competing-risk mixture family.

    python -m evodur.run --data ring4 --tau 1 --folds 5 --repeats 3
    python -m evodur.run --data citywide --tau 1 --folds 5 --repeats 1

Per fold and model it records the held-out
  report_logp : mean log f(t | x, T>tau)          (clearance method unknown)
  known_logp  : mean log f(t | x, c, T>tau)       (method known; = report for models without c)
  joint_logp  : mean log p(c, t | x, T>tau)       (only models that use c)
  MAPE / MAE / MedAE of the predictive median (report time), C-index
and, for learned emissions, the estimated P(c | z).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch

from evodur.data import RESULTS, encoder, load
from evodur import baselines as BL
from evodur.model import Config, CRMixture, EvidenceModel

OUT = RESULTS
GRID = np.unique(np.concatenate([np.linspace(0.5, 180, 360), np.geomspace(180, 1440, 80)]))


def specs(tau, C=4):
    lin = dict(neural=False, tau=tau)
    deep = dict(neural=True, tau=tau)
    return {
        "AFT": Config(K=1, emission="none", **lin),
        "Li2015": Config(K=C, emission="identity", **lin),
        "Mix-linear-4": Config(K=4, emission="none", **lin),
        "Latent-linear-4": Config(K=4, emission="learned", **lin),
        "DeepAFT": Config(K=1, emission="none", **deep),
        "DeepCAFT": Config(K=C, emission="identity", **deep),
        # mixture form with as many components as EvoDur: two or four log-logistic components per label class
        "DeepCAFT-2pc": Config(K=2 * C, emission="block", **deep),
        "DeepCAFT-4pc": Config(K=4 * C, emission="block", **deep),
        "DeepCAFT-M": ("select", ["DeepCAFT-2pc", "DeepCAFT-4pc"]),
        "DeepMix-4": Config(K=4, emission="none", **deep),
        "LatentDeepCAFT-4": Config(K=4, emission="learned", **deep),
        "LatentDeepCAFT-6": Config(K=6, emission="learned", **deep),
        "CtxDeepCAFT-4": Config(K=4, emission="contextual", **deep),
        "CtxDeepCAFT-2": Config(K=2, emission="contextual", **deep),
        "Ctx-linear-4": Config(K=4, emission="contextual", **lin),
        "DeepMix-4+aux": Config(K=4, emission="aux", **deep),
        "DeepMix-4-constpi": Config(K=4, emission="none", const_pi=True, **deep),
        "Mix-linear-4-constpi": Config(K=4, emission="none", const_pi=True, **lin),
        "DeepMix-8": Config(K=8, emission="none", **deep),
        "DeepMix-2": Config(K=2, emission="none", **deep),
        "DeepMix-16": Config(K=16, emission="none", **deep),
        "DeepMix-8-constpi": Config(K=8, emission="none", const_pi=True, **deep),
        "Mix-linear-8": Config(K=8, emission="none", **lin),
        # Evidence factorization: duration model fitted on t only, P(c | x, t) for updates.
        "Evidence-AFT": ("evidence", Config(K=1, emission="none", **lin)),
        "Evidence-DeepMix-4": ("evidence", Config(K=4, emission="none", **deep)),
        "Evidence-DeepMix-4-constpi": ("evidence", Config(K=4, emission="none", const_pi=True, **deep)),
        "Evidence-DeepMix-8": ("evidence", Config(K=8, emission="none", **deep)),
        # covariate-dependent component shapes, and more components
        "Evidence-DeepMix-8-het": ("evidence", Config(K=8, emission="none", het_scale=True, **deep)),
        "Evidence-DeepMix-16": ("evidence", Config(K=16, emission="none", **deep)),
        "Evidence-DeepMix-16-het": ("evidence", Config(K=16, emission="none", het_scale=True, **deep)),
        # EvoDur: the variant above with the best validation likelihood (seed 0), rule fixed before the results
        "EvoDur": ("select", ["Evidence-DeepMix-8", "Evidence-DeepMix-8-het", "Evidence-DeepMix-16", "Evidence-DeepMix-16-het"]),
        # Baselines from the literature (same folds, same features).
        "DeepHit-CR": ("baseline", lambda tau, batch: BL.DeepHitCR(tau, C, batch=batch)),
        "WeibullMDN-4": ("baseline", lambda tau, batch: BL.WeibullMDN(tau, 4, batch=batch)),
        "WeibullMDN-8": ("baseline", lambda tau, batch: BL.WeibullMDN(tau, 8, batch=batch)),
        "WeibullMDN-16": ("baseline", lambda tau, batch: BL.WeibullMDN(tau, 16, batch=batch)),
        # the same selection budget as EvoDur: number of components chosen on the validation set
        "WeibullMDN-sel": ("select", ["WeibullMDN-4", "WeibullMDN-8", "WeibullMDN-16"]),
        "XGBoost-AFT": ("baseline", lambda tau, batch: BL.XGBoostAFT(tau)),
        "RSF": ("baseline", lambda tau, batch: BL.RSF(tau)),
        "DeepSurv": ("baseline", lambda tau, batch: BL.DeepSurv(tau, batch=batch)),
        "NeuralFineGray": ("baseline", lambda tau, batch: __import__("evodur.nfg_baseline", fromlist=["x"]).NeuralFineGrayCR(tau, C, batch=batch)),
        "CtxDeepCAFT-6": Config(K=6, emission="contextual", **deep),
    }


def quantile_from_surv(S, grid, q):
    idx = (S <= 1 - q).argmax(1)
    idx[(S > 1 - q).all(1)] = len(grid) - 1
    return grid[idx]


def median_from_surv(S, grid):
    return quantile_from_surv(S, grid, 0.5)


def point_metrics(S, grid, tt, suffix=""):
    """Point, ranking and interval measures of the forecast with survival curves S (rows: incidents) on grid."""
    from lifelines.utils import concordance_index
    med = median_from_surv(S, grid)
    q10, q90 = quantile_from_surv(S, grid, 0.1), quantile_from_surv(S, grid, 0.9)
    m = grid <= 240
    alive = (tt[:, None] > grid[None, m]).astype(float)
    ibs = float(np.trapezoid(((alive - S[:, m]) ** 2).mean(0), grid[m]) / (grid[m][-1] - grid[m][0]))
    out = {"IBS": ibs, "PI80_cover": float(np.mean((tt >= q10) & (tt <= q90))), "PI80_width": float(np.median(q90 - q10)),
           "MAPE": float(np.mean(np.abs(med - tt) / tt)), "MAE": float(np.mean(np.abs(med - tt))),
           "MedAE": float(np.median(np.abs(med - tt))), "C_index": float(concordance_index(tt, med))}
    return {k + suffix: v for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ring4")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--models", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n-seeds", type=int, default=1, help="seed ensemble size")
    ap.add_argument("--only-folds", default="", help="comma-separated fold indices to run")
    ap.add_argument("--split", default="cv", choices=["cv", "temporal"])
    ap.add_argument("--train-frac", type=float, default=1.0, help="subsample the training part (learning curves)")
    ap.add_argument("--landmarks", default="", help="elapsed times s (min) for dynamic evaluation, e.g. 10,20,30,60")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    torch.set_num_threads(args.threads)

    from lifelines.utils import concordance_index
    from sklearn.model_selection import StratifiedKFold, train_test_split

    data = load(args.data, min_minutes=args.tau)
    if args.split == "temporal":
        # test years lie outside the training range, so calendar year is not used as a feature here
        data.numeric = [v for v in data.numeric if v != "year"]
    t, c = data.t, data.c
    print(f"{args.data}: n={len(t)}, tau={args.tau}, method shares={np.bincount(c) / len(c)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.data}_tau{int(args.tau)}" + (f"_{args.tag}" if args.tag else "")
    rows, emis = [], []
    incidents = {}  # per-incident log scores, for incident-level paired tests
    C = int(c.max()) + 1
    all_specs = specs(args.tau, C)
    for cfg_ in all_specs.values():
        if not (isinstance(cfg_, tuple) and cfg_[0] in ("baseline", "select")):
            (cfg_[1] if isinstance(cfg_, tuple) else cfg_).batch = args.batch
    names = [m for m in all_specs if not args.models or m in args.models.split(",")]
    grid = GRID[GRID > args.tau] if args.tau > 0 else GRID

    from evodur.splits import iter_splits

    for rep, fold, tr_, va, te in iter_splits(data, args.split, args.folds, args.repeats, args.train_frac, args.only_folds):
        enc = encoder(data).fit(data.frame.iloc[tr_])
        Xtr, Xva, Xte = (enc.transform(data.frame.iloc[i]).astype(np.float32) for i in (tr_, va, te))
        for name in names:
            spec = all_specs[name]
            t0 = time.perf_counter()

            def fit_one(seed):
                if isinstance(spec, tuple) and spec[0] == "baseline":
                    mm = spec[1](args.tau, args.batch).fit(Xtr, t[tr_], c[tr_], Xva, t[va], c[va], seed=seed)
                    return mm, isinstance(mm, BL.DeepHitCR) or type(mm).__name__ == "NeuralFineGrayCR", None
                if isinstance(spec, tuple):
                    mm = EvidenceModel(spec[1], n_methods=C, batch=args.batch).fit(Xtr, t[tr_], c[tr_], Xva, t[va], c[va], seed=seed)
                    return mm, True, spec[1]
                mm = CRMixture(spec, n_methods=C).fit(Xtr, t[tr_], c[tr_], Xva, t[va], c[va], seed=seed)
                return mm, spec.emission != "none", spec

            # Seed ensemble (equal-weight mixture of the predictive distributions) when --n-seeds > 1.
            chosen = None
            if isinstance(spec, tuple) and spec[0] == "select":
                cands = {}
                for v in spec[1]:
                    spec = all_specs[v]
                    cands[v] = fit_one(rep * 10 + fold)
                chosen = min(cands, key=lambda v: cands[v][0].val_nll)
                spec = all_specs[chosen]
                fitted = [cands[chosen]] + [fit_one(rep * 10 + fold + 1000 * s) for s in range(1, args.n_seeds)]
            else:
                fitted = [fit_one(rep * 10 + fold + 1000 * s) for s in range(args.n_seeds)]
            m, uses_c, cfg = fitted[0]
            ens = [f[0] for f in fitted]

            def logmeanexp(arrs):
                if any(a is None for a in arrs):
                    return None
                return np.logaddexp.reduce(np.stack(arrs), axis=0) - np.log(len(arrs))

            rep_lp = logmeanexp([mm.log_density(Xte, t[te]) for mm in ens])
            kn_lp = logmeanexp([mm.log_density(Xte, t[te], c[te]) for mm in ens]) if uses_c else rep_lp
            if rep_lp is None:
                rep_lp = kn_lp = np.full(len(te), np.nan)
            S = np.mean([mm.survival(Xte, grid) for mm in ens], axis=0)
            med = median_from_surv(S, grid)
            q10, q90 = quantile_from_surv(S, grid, 0.1), quantile_from_surv(S, grid, 0.9)
            ibs_mask = grid <= 240
            alive = (t[te][:, None] > grid[None, ibs_mask]).astype(float)
            ibs = float(np.trapezoid(((alive - S[:, ibs_mask]) ** 2).mean(0), grid[ibs_mask]) / (grid[ibs_mask][-1] - grid[ibs_mask][0]))
            # calibration: PIT = F(t_i | x_i) and central-interval coverage at several levels
            Si = np.array([np.interp(t[te][i], grid, S[i]) for i in range(len(te))])
            pit = np.clip(1 - Si, 0, 1)
            calib = {f"cover{int(100 * L)}": float(np.mean(np.abs(pit - 0.5) <= L / 2)) for L in (0.5, 0.8, 0.9)}
            calib.update({f"pit_h{j}": float(np.mean((pit >= j / 10) & (pit < (j + 1) / 10 + (j == 9) * 1e-9))) for j in range(10)})
            row = {"data": args.data, "tau": args.tau, "split": args.split, "train_frac": args.train_frac, **calib,
                   "n_train": len(tr_), "repeat": rep, "fold": fold,
                   "model": name + (f"-ens{args.n_seeds}" if args.n_seeds > 1 else ""), "chosen": chosen,
                   "report_logp": float(rep_lp.mean()), "known_logp": float(kn_lp.mean()),
                   "val_nll": m.val_nll, "IBS": ibs,
                   "PI80_cover": float(np.mean((t[te] >= q10) & (t[te] <= q90))), "PI80_width": float(np.median(q90 - q10)),
                   "MAPE": float(np.mean(np.abs(med - t[te]) / t[te])), "MAE": float(np.mean(np.abs(med - t[te]))),
                   "MedAE": float(np.median(np.abs(med - t[te]))), "C_index": float(concordance_index(t[te], med)),
                   "seconds": time.perf_counter() - t0}
            # Dynamic prediction: the incident is still going at s. Conditional density of the
            # ensemble = sum_m f_m(t) / sum_m S_m(s) (the ensemble is an equal-weight mixture).
            for s_lm in [float(v) for v in args.landmarks.split(",") if v]:
                idx = np.flatnonzero(t[te] > s_lm)
                if len(idx) < 20:
                    continue
                Xs, ts = Xte[idx], t[te][idx]
                for lab, cc in (("", None), ("_known", c[te][idx] if uses_c else None)):
                    if lab and cc is None:
                        continue
                    dens = [mm.log_density(Xs, ts, cc) if cc is not None else mm.log_density(Xs, ts) for mm in ens]
                    if any(dd is None for dd in dens):
                        continue
                    Ss = [mm.survival(Xs, np.array([s_lm]), cc)[:, 0] if cc is not None else mm.survival(Xs, np.array([s_lm]))[:, 0]
                          for mm in ens]
                    lp = logmeanexp(dens) - np.log(np.clip(np.mean(Ss, axis=0), 1e-12, None))
                    row[f"lm{int(s_lm)}{lab}_logp"] = float(np.mean(lp))
                row[f"lm{int(s_lm)}_n"] = len(idx)
            if uses_c:
                # the same measures once the label is known, for every model that uses it
                S_k = np.mean([mm.survival(Xte, grid, c[te]) for mm in ens], axis=0)
                row.update(point_metrics(S_k, grid, t[te], "_known"))
            if isinstance(m, EvidenceModel) and len(ens) == 1:
                row["joint_logp"] = float(m.joint_logp(Xte, t[te], c[te]).mean())
            elif uses_c and isinstance(m, CRMixture) and len(ens) == 1:
                with torch.no_grad():
                    row["joint_logp"] = float(m._joint_ll(m.net, torch.as_tensor(Xte), torch.log(torch.as_tensor(t[te], dtype=torch.float32)),
                                                          torch.as_tensor(c[te])).mean())
            if cfg is not None and cfg.emission in ("learned", "contextual"):
                emis.append({"repeat": rep, "fold": fold, "model": name, "P_c_given_z": m.emission_matrix().round(3).tolist(),
                             "type_share": m.type_weights(Xte).mean(0).numpy().round(3).tolist(),
                             "alpha": m.net.log_alpha.detach().exp().numpy().round(2).tolist(),
                             "beta": m.net.log_beta.detach().exp().numpy().round(2).tolist()})
            rows.append(row)
            mname = row["model"]
            incidents[f"{rep}|{fold}|test_index"] = te.astype(np.int32)
            incidents[f"{rep}|{fold}|{mname}|report"] = rep_lp.astype(np.float32)
            incidents[f"{rep}|{fold}|{mname}|known"] = kn_lp.astype(np.float32)
            print(f"[r{rep}f{fold}] {name:18s} report={row['report_logp']:.4f} known={row['known_logp']:.4f} "
                  f"MAPE={row['MAPE']:.3f} C={row['C_index']:.3f} ({row['seconds']:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"folds_{tag}.csv", index=False)
        np.savez_compressed(OUT / f"incidents_{tag}.npz", **incidents)
        with (OUT / f"emission_{tag}.json").open("w") as fh:
            json.dump(emis, fh)
    df = pd.DataFrame(rows)
    cols = [k for k in ("report_logp", "known_logp", "joint_logp", "MAPE", "MAE", "C_index", "IBS", "PI80_cover", "seconds") if k in df.columns]
    summ = df.groupby("model")[cols].mean()
    if (df.model == "AFT").any():
        base = df[df.model == "AFT"].set_index(["repeat", "fold"])["report_logp"]
        diff = {mname: (g.set_index(["repeat", "fold"])["report_logp"] - base) for mname, g in df.groupby("model")}
        summ["report_vs_AFT_mean"] = {k: v.mean() for k, v in diff.items()}
        summ["report_vs_AFT_sd"] = {k: v.std() for k, v in diff.items()}
    summ.to_csv(OUT / f"summary_{tag}.csv")
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
