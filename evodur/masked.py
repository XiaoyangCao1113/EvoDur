"""Two extra baselines for the evidence experiment (same data, folds, states and test incidents as evidence.py).

1. Masked-input single network: ONE MixAFT (K=8) whose input is the call-time features x plus optional evidence
   [arrival known, log r], [label known, one-hot c], [elapsed known, log s]. It is trained once with random masking:
   every step each incident is put in one information state drawn uniformly from the states it can be in, and the
   loss is the truncated log-likelihood log f(t | inputs) - log S(lower | inputs) with the state's lower bound.
   The one network is then scored in every state evidence.py evaluates (column masked_logp).
2. Matched prior: the Weibull MDN (K=4) replaces the MixAFT prior and is combined with EvoDur's label module by the
   same Bayes update (columns weibull_prior_logp, weibull_evidence_logp).

    python -m evodur.masked --data ring4 --tau 1 --folds 0 --batch 256 --tag _masked_f0
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import torch

from evodur.data import encoder
from evodur.evidence import (OUT, T_MAX, ArrivalEvidence, MethodEvidence, MethodGivenArrival, MixAFT, fit_evidence,
                               load_with_stages, posterior_logp, summary_metrics, train)

S_LIST = (10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0)  # elapsed-time states of evidence.py
S_ARR = (20.0, 30.0, 45.0, 60.0, 90.0, 120.0)  # arrival + elapsed states


def make_specs(has_r, s_arr):
    """(name, arrival known, label known, elapsed bound s or None) for every information state of evidence.py."""
    sp = [("report", 0, 0, None), ("method", 0, 1, None)]
    if has_r:
        sp += [("arrival", 1, 0, None), ("arrival+method", 1, 1, None)]
        sp += [(f"arrival+elapsed{int(s)}" + ("+method" if lab else ""), 1, lab, s) for s in s_arr for lab in (0, 1)]
    sp += [(f"elapsed{int(s)}" + ("+method" if lab else ""), 0, lab, s) for s in S_LIST for lab in (0, 1)]
    return sp


class Pool:
    """Incidents of one split with, for each information state, whether the incident can be in it, its lower bound,
    and a builder of the masked-network input."""

    def __init__(self, X, t, r, ok_r, oh, specs, tau):
        self.X, self.oh = torch.as_tensor(X), torch.as_tensor(oh)
        self.t = torch.as_tensor(t, dtype=torch.float32)
        self.logr = torch.as_tensor(np.where(ok_r, np.log(np.clip(np.where(ok_r, r, 1.0), 0.1, None)), 0.0), dtype=torch.float32)[:, None]
        n = len(t)
        self.avail = np.zeros((n, len(specs)), bool)
        self.lower = np.full((n, len(specs)), tau, np.float32)
        for j, (_, a, lab, s) in enumerate(specs):
            ok = ok_r.copy() if a else np.ones(n, bool)
            if s is None:
                if a:
                    self.lower[:, j] = np.maximum(np.where(ok_r, r, tau), tau)
            else:
                with np.errstate(invalid="ignore"):
                    ok &= t > s
                    if a:
                        ok &= r < s
                self.lower[:, j] = s
            self.avail[:, j] = ok
        self.avail_t, self.lower_t = torch.as_tensor(self.avail), torch.as_tensor(self.lower)
        self.flag = [torch.tensor([float(sp_[k]) for sp_ in specs]) for k in (1, 2)]
        self.has_s = torch.tensor([float(sp_[3] is not None) for sp_ in specs])
        self.log_s = torch.tensor([float(np.log(sp_[3])) if sp_[3] is not None else 0.0 for sp_ in specs])

    def feats(self, i, j):
        """Input rows for incidents i in states j (index tensors of equal length)."""
        a, lab, e = self.flag[0][j][:, None], self.flag[1][j][:, None], self.has_s[j][:, None]
        return torch.cat([self.X[i], a, a * self.logr[i], lab, lab * self.oh[i], e, e * self.log_s[j][:, None]], 1)


def state_weights(specs, mix):
    """Sampling weight of each state: 'uniform' over available states, or 'group', where the elapsed-time states of
    one evidence combination together weigh as much as the state without elapsed time."""
    w = np.ones(len(specs), np.float32)
    if mix == "group":
        for j, (_, a, lab, s) in enumerate(specs):
            if s is not None:
                w[j] = 1.0 / sum(1 for (_, a2, l2, s2) in specs if s2 is not None and (a2, l2) == (a, lab))
    return torch.as_tensor(w)


def fit_masked(P, V, seed=0, n_init=2, batch=256, K=8, max_epochs=400, W=None):
    """One MixAFT trained with random masking; validation loss = expected loss under the training state mix
    (each validation incident weighted over its available states by W)."""
    W = torch.ones(P.avail_t.shape[1]) if W is None else W
    Wv = V.avail_t.float() * W
    iv, jv = V.avail_t.nonzero(as_tuple=True)
    wv = Wv[iv, jv] / Wv.sum(1)[iv]
    d = P.feats(torch.arange(2), torch.zeros(2, dtype=torch.long)).shape[1]
    best = None
    for k in range(n_init):
        torch.manual_seed(seed * 10 + k)
        m = MixAFT(d, K)

        def loss_fn(b):
            j = torch.multinomial(P.avail_t[b].float() * W, 1)[:, 0]
            return -m.loglik(P.feats(b, j), P.t[b], P.lower_t[b, j]).mean()

        def val_fn(ch=100000):
            tot = 0.0
            for s in range(0, len(iv), ch):
                i, j = iv[s:s + ch], jv[s:s + ch]
                tot += float((wv[s:s + ch] * m.loglik(V.feats(i, j), V.t[i], V.lower_t[i, j])).sum())
            return -tot / len(V.t)

        v = train(m.parameters(), loss_fn, val_fn, len(P.t), seed=seed * 10 + k, modules=(m,), batch=batch, max_epochs=max_epochs)
        if best is None or v < best[0]:
            best = (v, m)
    return best[1]


class WeibullPrior:
    """Adapter: a fitted WeibullMDN as a prior for posterior_logp (log density on (tau, inf); the truncation
    constant is per row and cancels in the normalisation)."""

    def __init__(self, w):
        self.w = w
        w.net.eval()

    def log_f(self, X, t):
        return self.w._ll(X, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ring4")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-epochs", type=int, default=400, help="masked network only (smoke tests)")
    ap.add_argument("--skip-masked", action="store_true")
    ap.add_argument("--skip-weibull", action="store_true")
    ap.add_argument("--mix", default="uniform", choices=["uniform", "group"], help="state sampling of the masked network")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    from evodur.baselines import WeibullMDN
    from evodur.splits import iter_splits

    data, r = load_with_stages(args.data, args.tau)
    t, c = data.t, data.c
    C = int(c.max()) + 1
    has_r = r is not None
    ok_r = ((r > 0) & (r < t)) if has_r else np.zeros(len(t), bool)
    r_ = r if has_r else np.zeros(len(t))
    print(f"{args.data}: n={len(t)}, C={C}, arrival available for {ok_r.mean():.3f}", flush=True)
    B, tau = args.batch, args.tau
    oh = np.eye(C, dtype=np.float32)[c]
    rows = []
    for _, fold, tr_, va, te in iter_splits(data, "cv", 5, 1, 1.0, args.folds):
        enc = encoder(data).fit(data.frame.iloc[tr_])
        Xtr, Xva, Xte = (enc.transform(data.frame.iloc[i]).astype(np.float32) for i in (tr_, va, te))
        res = {}  # state -> dict of columns
        lp_inc = {}

        if not args.skip_masked:
            # arrival + elapsed states are used when evidence.py evaluates them (>= 50 train / val / test incidents)
            def n_in(idx, s):
                with np.errstate(invalid="ignore"):
                    return int((ok_r[idx] & (r_[idx] < s) & (t[idx] > s)).sum())

            s_arr = tuple(s for s in S_ARR if has_r and min(n_in(tr_, s), n_in(va, s), n_in(te, s)) >= 50)
            specs = make_specs(has_r, s_arr)
            P, V, E = (Pool(X_, t[i], r_[i], ok_r[i], oh[i], specs, tau) for X_, i in ((Xtr, tr_), (Xva, va), (Xte, te)))
            m = fit_masked(P, V, seed=fold, batch=B, max_epochs=args.max_epochs, W=state_weights(specs, args.mix))
            for j, (name, _, _, _) in enumerate(specs):
                sel = E.avail_t[:, j].nonzero(as_tuple=True)[0]
                if len(sel) == 0:
                    continue
                jj = torch.full((len(sel),), j, dtype=torch.long)
                Xq, tt, low = E.feats(sel, jj), E.t[sel], E.lower_t[sel, j]
                with torch.no_grad():
                    lp = m.loglik(Xq, tt, low, upper=T_MAX).numpy()
                    _, sm = posterior_logp(m, [], Xq.numpy(), tt.numpy(), low.numpy(), summary=True)
                # the same state on the validation incidents, to choose between the two sampling schemes
                sel_v = V.avail_t[:, j].nonzero(as_tuple=True)[0]
                with torch.no_grad():
                    jv = torch.full((len(sel_v),), j, dtype=torch.long)
                    lp_v = m.loglik(V.feats(sel_v, jv), V.t[sel_v], V.lower_t[sel_v, j], upper=T_MAX).numpy()
                row = {"n": len(sel), "masked_logp": float(lp.mean()),
                       "masked_val_logp": float(lp_v.mean()) if len(sel_v) else float("nan")}
                row.update({f"masked_{k}": v for k, v in summary_metrics(sm, tt.numpy(), low.numpy()).items()})
                for nm in (name, "arrival+method-chain") if name == "arrival+method" else (name,):
                    res.setdefault(nm, {}).update(row)
                    lp_inc[f"{nm}|masked"] = lp.astype(np.float32)
                    lp_inc[f"{nm}|idx"] = te[sel.numpy()].astype(np.int64)
                print(name, row, flush=True)

        if not args.skip_weibull:
            ev_met = fit_evidence(lambda d: MethodEvidence(d, C), Xtr, c[tr_], t[tr_], Xva, c[va], t[va], seed=fold, batch=B)
            E_met = lambda cc: ((lambda Xq, a, tq: ev_met.logp(Xq, torch.as_tensor(a, dtype=torch.long), tq)), cc)
            wb = WeibullPrior(WeibullMDN(tau, 4, batch=B).fit(Xtr, t[tr_], None, Xva, t[va], None, seed=fold))
            tau_te = np.full(len(te), tau, np.float32)
            lp = posterior_logp(wb, [], Xte, t[te], tau_te)
            res.setdefault("report", {}).update({"n": len(te), "weibull_prior_logp": float(lp.mean()), "weibull_evidence_logp": float(lp.mean())})
            lp_inc["report|weibull"] = lp.astype(np.float32)
            lp = posterior_logp(wb, [E_met(c[te])], Xte, t[te], tau_te)
            res.setdefault("method", {}).update({"n": len(te), "weibull_evidence_logp": float(lp.mean())})
            lp_inc["method|weibull"] = lp.astype(np.float32)
            if has_r:
                a_tr, a_va, te_r = tr_[ok_r[tr_]], va[ok_r[va]], te[ok_r[te]]
                ev_arr = fit_evidence(ArrivalEvidence, Xtr[ok_r[tr_]], r[a_tr], t[a_tr], Xva[ok_r[va]], r[a_va], t[a_va], seed=fold, batch=B)
                E_arr = lambda rr: ((lambda Xq, a, tq: ev_arr.logp(Xq, torch.as_tensor(a, dtype=torch.float32), tq)), rr)
                occ = []
                if not ok_r.all():
                    ev_occ = fit_evidence(lambda d: MethodEvidence(d, 2), Xtr, ok_r[tr_].astype(np.int64), t[tr_], Xva,
                                          ok_r[va].astype(np.int64), t[va], seed=fold, batch=B)
                    occ = [((lambda Xq, a, tq: ev_occ.logp(Xq, torch.as_tensor(a, dtype=torch.long), tq)), np.ones(len(te_r), np.int64))]
                cr = lambda idx: np.column_stack([c[idx], r[idx]]).astype(np.float32)
                ev_met_r = fit_evidence(lambda d: MethodGivenArrival(d, C), Xtr[ok_r[tr_]], cr(a_tr), t[a_tr], Xva[ok_r[va]], cr(a_va),
                                        t[a_va], seed=fold, batch=B)
                E_metr = ((lambda Xq, a, tq: ev_met_r.logp(Xq, torch.as_tensor(a, dtype=torch.float32), tq)), cr(te_r))
                lp = posterior_logp(wb, [E_arr(r[te_r]), E_metr] + occ, Xte[ok_r[te]], t[te_r], np.maximum(r[te_r], tau))
                res.setdefault("arrival+method-chain", {}).update({"n": len(te_r), "weibull_evidence_logp": float(lp.mean())})
                lp_inc["arrival+method-chain|weibull"] = lp.astype(np.float32)
            for k_, v_ in res.items():
                if "weibull_evidence_logp" in v_:
                    print(k_, v_["weibull_evidence_logp"], flush=True)

        for state, d in res.items():
            rows.append({"data": args.data, "split": "cv", "n_train": len(tr_), "fold": fold, "state": state, **d})
        np.savez_compressed(OUT / f"masked_incidents_{args.data}{args.tag}_fold{fold}.npz", **lp_inc)
        pd.DataFrame(rows).to_csv(OUT / f"masked_states_{args.data}{args.tag}.csv", index=False)
    print(pd.DataFrame(rows).drop(columns=["data", "split"]).groupby("state").mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
