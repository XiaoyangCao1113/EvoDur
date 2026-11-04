"""Evidence updating for incident duration.

Prior (report time): f(t | x), a neural mixture of log-logistic AFTs.
Each piece of information e that arrives later has a likelihood p(e | t, x);
the prediction is updated by Bayes' rule

    f(t | x, e_1, ..., e_m)  proportional to  f(t | x) * prod_j p(e_j | t, x).

Evidence types used here:
  elapsed : the accident is still going at s minutes, p = 1{t > s}
  arrival : police arrived r minutes after the call; modelled through
            u = r / t in (0, 1), logit(u) ~ Normal(mu(x, t), sigma(x, t))
  method  : recorded clearance method c, P(c | x, t)

Baseline for every information state: a separate discriminative model that
takes the available information as input and is left-truncated at what is
already known (e.g. T > r after arrival), as in sequential prediction.

    python -m evodur.evidence --data ring4
"""
from __future__ import annotations

import argparse
import copy
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evodur.data import DATA, RESULTS, encoder, load

OUT = RESULTS
sp = nn.functional.softplus
T_MAX = 1440.0


# ----------------------------------------------------------------------------- data

def load_with_stages(name, min_minutes=1.0):
    """Same rows as evodur.data.load, plus the response time r (call -> arrival).
    Public datasets have no arrival time: r is None."""
    data = load(name, min_minutes)
    if name in ("seattle", "sanfrancisco"):  # dispatch records with the first-arrival time
        return data, data.frame["_arrival"].to_numpy(float)
    if name in ("sydney", "ny511early"):
        return data, None
    if name == "ring4":
        raw = pd.read_excel(DATA / "raw_data.xlsx")
        t_all = raw["totaltimem"].to_numpy(float)
        r_all = ((raw["D_ARRIVE_TIME"] - raw["D_CALL_TIME"]).dt.total_seconds() / 60).to_numpy()
        c_ok = raw["TREAT ？"].isin([1, 2, 31, 32]).to_numpy()
    else:
        raw = pd.read_excel(DATA / "raw_data_extended_stata.xlsx", sheet_name="Sheet1")

        def ts(col):
            return pd.to_datetime(raw[col].astype(str).str.replace(r"^2018([-/])", r"2008\1", regex=True), format="mixed")

        call = ts("D_CALL_TIME")
        t_all = ((ts("D_DONE_TIME") - call).dt.total_seconds() / 60.0).to_numpy()
        r_all = ((ts("D_ARRIVE_TIME") - call).dt.total_seconds() / 60.0).to_numpy()
        ok = np.isfinite(t_all) & (t_all > 0)
        t_all, r_all = t_all[ok], r_all[ok]
        c_ok = raw.loc[ok, "TREAT"].isin([1, 2, 31, 32]).to_numpy()
    keep = (t_all > min_minutes) & c_ok
    r = r_all[keep]
    assert len(r) == len(data.t) and np.allclose(t_all[keep], data.t)
    return data, r


# ----------------------------------------------------------------------------- networks

CUTS = (30, 60, 120, 240)  # P(T > c min) saved with --save-detail; 30 and 120 min bound the MUTCD incident classes

def mlp(d, out, hidden=(64, 64), dropout=0.1):
    layers, prev = [], d
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    return nn.Sequential(*layers, nn.Linear(prev, out))


def train(net_params, loss_fn, val_fn, n, lr=2e-3, wd=1e-4, batch=256, max_epochs=400, patience=25, seed=0, modules=()):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net_params, lr=lr, weight_decay=wd)
    best, state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        for m in modules:
            m.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            loss = loss_fn(perm[s:s + batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        for m in modules:
            m.eval()
        with torch.no_grad():
            v = float(val_fn())
        if v < best - 1e-5:
            best, bad, state = v, 0, [copy.deepcopy(m.state_dict()) for m in modules]
        else:
            bad += 1
            if bad >= patience:
                break
    for m, st in zip(modules, state):
        m.load_state_dict(st)
        m.eval()
    return best


class MixAFT(nn.Module):
    """Neural mixture of K log-logistic AFTs with per-row left truncation.
    Used as the report-time prior and as the per-state discriminative baseline."""

    def __init__(self, d, K=4, linear=False):
        super().__init__()
        self.K = K
        self.lin = nn.Linear(d, 2 * K)
        self.deep = (lambda X: 0.0) if linear else mlp(d, 2 * K)
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(np.geomspace(15, 60, K), dtype=torch.float32)))
        self.log_beta = nn.Parameter(torch.ones(K))

    def comps(self, X, log_t):
        o = self.lin(X) + self.deep(X)
        h, log_w = o[:, :self.K], torch.log_softmax(o[:, self.K:], 1)
        beta = torch.exp(self.log_beta)
        u = beta * (log_t[:, None] + h - self.log_alpha)
        return log_w, self.log_beta - log_t[:, None] + u - 2 * sp(u), -sp(u)  # log w, log f_k, log S_k

    def log_f(self, X, t):
        log_w, lf, _ = self.comps(X, torch.log(t))
        return torch.logsumexp(log_w + lf, 1)

    def log_S(self, X, t):
        log_w, _, lS = self.comps(X, torch.log(t))
        return torch.logsumexp(log_w + lS, 1)

    def loglik(self, X, t, lower, upper=None):
        """log f(t | x, lower < T), or on (lower, upper] when an upper limit is given."""
        if upper is None:
            return self.log_f(X, t) - self.log_S(X, lower)
        lS_lo, lS_up = self.log_S(X, lower), self.log_S(X, torch.full_like(lower, float(upper)))
        return self.log_f(X, t) - lS_lo - torch.log1p(-torch.exp(lS_up - lS_lo).clamp(max=1 - 1e-7))


def fit_mixaft(X, t, lower, Xv, tv, lowerv, K=8, seed=0, n_init=2, batch=256, linear=False):
    X, Xv = (torch.as_tensor(a, dtype=torch.float32) for a in (X, Xv))
    t, tv, lower, lowerv = (torch.as_tensor(a, dtype=torch.float32) for a in (t, tv, lower, lowerv))
    best = None
    for i in range(n_init):
        torch.manual_seed(seed * 10 + i)
        m = MixAFT(X.shape[1], K, linear=linear)
        v = train(m.parameters(), lambda b: -m.loglik(X[b], t[b], lower[b]).mean(), lambda: -m.loglik(Xv, tv, lowerv).mean(),
                  len(t), seed=seed * 10 + i, modules=(m,), batch=batch)
        if best is None or v < best[0]:
            best = (v, m)
    return best[1]


class XGBoostAFTTrunc:
    """XGBoost-AFT for one information state: log-logistic model of the excess duration T - lower (+0.1 min),
    where lower is the per-row truncation point (arrival time or elapsed time); as in baselines.XGBoostAFT."""

    def fit(self, X, t, lower, Xv, tv, lowerv, seed=0):
        import xgboost as xgb

        y, yv = t - lower + 0.1, tv - lowerv + 0.1
        d = xgb.DMatrix(X, label_lower_bound=y, label_upper_bound=y)
        dv = xgb.DMatrix(Xv, label_lower_bound=yv, label_upper_bound=yv)
        best = None
        for sigma in (0.3, 0.4, 0.5, 0.8):
            for depth in (3, 5):
                params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "logistic",
                          "aft_loss_distribution_scale": sigma, "learning_rate": 0.05, "max_depth": depth,
                          "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "seed": seed, "nthread": 4,
                          "base_score": float(np.median(y))}
                m = xgb.train(params, d, 2000, evals=[(dv, "val")], early_stopping_rounds=50, verbose_eval=False)
                self.model, self.sigma, self.best_iter = m, sigma, m.best_iteration
                v = -float(np.mean(self.loglik(Xv, tv, lowerv)))
                if np.isfinite(v) and (best is None or v < best[0]):
                    best = (v, m, sigma, m.best_iteration)
        _, self.model, self.sigma, self.best_iter = best
        return self

    def loglik(self, X, t, lower, upper=None):
        import xgboost as xgb

        mu = np.log(self.model.predict(xgb.DMatrix(np.asarray(X)), iteration_range=(0, self.best_iter + 1)))
        y = np.asarray(t) - np.asarray(lower) + 0.1
        z = (np.log(y) - mu) / self.sigma
        ll = -np.log(self.sigma * y) + z - 2 * np.logaddexp(0, z)
        if upper is not None:  # renormalise on (lower, upper]: log P(Y <= upper - lower + 0.1)
            z_up = (np.log(upper - np.asarray(lower) + 0.1) - mu) / self.sigma
            ll = ll + np.logaddexp(0, -z_up)
        return ll


def fit_stage(X, t, lower, Xv, tv, lowerv, seed=0, batch=256):
    """Separate models for one information state: the neural mixture (same architecture as the prior), and the
    stage-wise models of earlier work, a linear log-logistic AFT and XGBoost-AFT."""
    return {"separate": fit_mixaft(X, t, lower, Xv, tv, lowerv, seed=seed, batch=batch),
            "linear": fit_mixaft(X, t, lower, Xv, tv, lowerv, K=1, seed=seed, batch=batch, linear=True),
            "xgb": XGBoostAFTTrunc().fit(X, t, lower, Xv, tv, lowerv, seed=seed)}


class ArrivalEvidence(nn.Module):
    """p(r | t, x) for the response time r < t: y = logit(r / t) follows a
    mixture of M normals whose parameters depend on (x, log t)."""

    def __init__(self, d, M=5):
        super().__init__()
        self.M = M
        self.net = mlp(d + 2, 3 * M)

    def logp(self, X, r, t):
        z = (torch.log(t)[:, None] - 3.4) / 0.7
        o = self.net(torch.cat([X, z, z ** 2], 1))
        log_w = torch.log_softmax(o[:, :self.M], 1)
        mu, log_sig = o[:, self.M:2 * self.M], o[:, 2 * self.M:].clamp(-4, 3)
        u = (r / t).clamp(1e-4, 1 - 1e-4)
        y = (torch.log(u) - torch.log1p(-u))[:, None]
        comp = -0.5 * ((y - mu) / torch.exp(log_sig)) ** 2 - log_sig - 0.5 * np.log(2 * np.pi)
        return torch.logsumexp(log_w + comp, 1) - torch.log(u * (1 - u)) - torch.log(t)


class MethodEvidence(nn.Module):
    """P(c | t, x)."""

    def __init__(self, d, C=4):
        # C is set from the data in main() via functools.partial
        super().__init__()
        self.net = mlp(d + 2, C)

    def logp(self, X, c, t):
        z = (torch.log(t)[:, None] - 3.4) / 0.7
        return torch.log_softmax(self.net(torch.cat([X, z, z ** 2], 1)), 1).gather(1, c[:, None]).squeeze(1)


class MethodGivenArrival(nn.Module):
    """P(c | t, r, x): the label module when the arrival time r is also known (chain rule,
    p(r, c | t, x) = p(r | t, x) P(c | t, r, x)), so the two pieces need not be independent given t.
    The evidence value is a pair (c, r) per row."""

    def __init__(self, d, C=4):
        super().__init__()
        self.net = mlp(d + 4, C)

    def logp(self, X, e, t):
        c, r = e[:, 0].long(), e[:, 1]
        z = (torch.log(t)[:, None] - 3.4) / 0.7
        u = (r / t).clamp(1e-4, 1 - 1e-4)[:, None]
        y = (torch.log(u) - torch.log1p(-u)) / 3.0
        lr = (torch.log(r.clamp(min=0.1))[:, None] - 2.4) / 0.8
        return torch.log_softmax(self.net(torch.cat([X, z, z ** 2, y, lr], 1)), 1).gather(1, c[:, None]).squeeze(1)


def fit_evidence(cls, X, e, t, Xv, ev, tv, seed=0, batch=256):
    X, Xv, t, tv = (torch.as_tensor(a, dtype=torch.float32) for a in (X, Xv, t, tv))
    discrete = np.issubdtype(np.asarray(e).dtype, np.integer)
    e = torch.as_tensor(e, dtype=torch.long if discrete else torch.float32)
    ev = torch.as_tensor(ev, dtype=torch.long if discrete else torch.float32)
    torch.manual_seed(seed)
    m = cls(X.shape[1])
    train(m.parameters(), lambda b: -m.logp(X[b], e[b], t[b]).mean(), lambda: -m.logp(Xv, ev, tv).mean(), len(t),
          seed=seed, modules=(m,), batch=batch)
    return m


# ----------------------------------------------------------------------------- posterior

@torch.no_grad()
def posterior_logp(prior, evidences, X, t, lower, n_grid=600, summary=False, chunk=1000):
    """log f(t | x, evidence) for each row, computed in blocks of `chunk` rows to bound memory.
    Each piece of evidence is a pair (fn, values): fn(X_rep, values_rep, t) -> log p(e | t, x), with one value
    per row (e.g. the label or the arrival time). A bare function (X_rep, t) -> log p is also accepted; it is
    then evaluated on all rows at once. `lower` is the known lower bound of T (elapsed time, arrival or tau)."""
    X = np.asarray(X, dtype=np.float32)
    t, lower = np.asarray(t, dtype=np.float32), np.asarray(lower, dtype=np.float32)
    if any(callable(e) for e in evidences):
        chunk = len(t)
    outs = [_posterior_block(prior, [e if callable(e) else (e[0], np.asarray(e[1])[k:k + chunk]) for e in evidences],
                             X[k:k + chunk], t[k:k + chunk], lower[k:k + chunk], n_grid, summary)
            for k in range(0, len(t), chunk)]
    if not summary:
        return np.concatenate(outs)
    return (np.concatenate([o[0] for o in outs]),
            {key: np.concatenate([o[1][key] for o in outs]) for key in outs[0][1]})


def _posterior_block(prior, evidences, X, t, lower, n_grid, summary):
    X = torch.as_tensor(X, dtype=torch.float32)
    t = torch.as_tensor(t, dtype=torch.float32)
    lower = torch.as_tensor(lower, dtype=torch.float32)
    n, d = X.shape

    def ev_logp(e, Xq, tq, rep):
        if callable(e):
            return e(Xq, tq)
        fn, vals = e
        return fn(Xq, np.repeat(vals, rep, axis=0), tq)

    with torch.no_grad():
        # numerator at the observed t
        num = prior.log_f(X, t)
        for e in evidences:
            num = num + ev_logp(e, X, t, 1)
        # normaliser: integrate over t in (lower, T_MAX] on a per-row log grid
        u = torch.linspace(0, 1, n_grid)
        lg = torch.log(lower)[:, None] + (np.log(T_MAX) - torch.log(lower))[:, None] * u[None, :]
        lg = lg[:, 1:] - 0.5 * (lg[:, 1:] - lg[:, :-1])  # midpoints
        dlog = (np.log(T_MAX) - torch.log(lower)) / (n_grid - 1)
        tg = torch.exp(lg)
        Xr = X[:, None, :].expand(n, tg.shape[1], d).reshape(-1, d)
        tf = tg.reshape(-1)
        integrand = prior.log_f(Xr, tf) + torch.log(tf)
        for e in evidences:
            integrand = integrand + ev_logp(e, Xr, tf, tg.shape[1])
        integrand = integrand.reshape(n, -1)
        log_norm = torch.logsumexp(integrand, 1) + torch.log(dlog)
        if not summary:
            return (num - log_norm).numpy()
        # predictive summaries from the same grid: quantiles (interpolated in log t) and P(T > 60 min)
        w = torch.softmax(integrand, 1).numpy()
        cdf = np.cumsum(w, 1)
        lgn = lg.numpy()
        q = {}
        for p_ in (0.1, 0.5, 0.9):
            j = np.clip((cdf < p_).sum(1), 1, cdf.shape[1] - 1)
            r_ = np.arange(n)
            c0, c1 = cdf[r_, j - 1], cdf[r_, j]
            frac = np.clip((p_ - c0) / np.maximum(c1 - c0, 1e-12), 0, 1)
            q[p_] = np.exp(lgn[r_, j - 1] + frac * (lgn[r_, j] - lgn[r_, j - 1]))
        out = {"q10": q[0.1], "q50": q[0.5], "q90": q[0.9]}
        out.update({f"p{c}": (w * (tg.numpy() > float(c))).sum(1) for c in CUTS})
        return (num - log_norm).numpy(), out


def xgb_summary(model, X, low):
    """Predictive summaries of an XGBoostAFTTrunc model on T in (low, T_MAX], in closed form.
    The model is log-logistic for Y = T - low + 0.1, so S(y) = 1 / (1 + exp((log y - mu) / sigma)); the summaries are
    conditional on Y > 0.1 (i.e. T > low) and Y <= T_MAX - low + 0.1, the same range the grid summaries use.
    (loglik() itself is only truncated above, as trained; the mass of Y < 0.1 is negligible.)"""
    import xgboost as xgb

    low = np.asarray(low, np.float64)
    mu = np.log(model.model.predict(xgb.DMatrix(np.asarray(X)), iteration_range=(0, model.best_iter + 1))).astype(np.float64)
    sg = model.sigma
    S = lambda y: 1.0 / (1.0 + np.exp(np.clip((np.log(y) - mu) / sg, -700, 700)))
    s0, s1 = S(np.full_like(low, 0.1)), S(T_MAX - low + 0.1)
    out = {}
    for p_, key in ((0.1, "q10"), (0.5, "q50"), (0.9, "q90")):
        sv = np.clip(s0 - p_ * (s0 - s1), 1e-300, 1 - 1e-16)
        y = np.exp(mu + sg * np.log1p(-sv) - sg * np.log(sv))  # S^-1: z = log(1 / S - 1)
        out[key] = np.clip(y + low - 0.1, low, T_MAX)
    for c in CUTS:
        pc = np.where(low >= c, 1.0, (S(np.maximum(c - low, 0.0) + 0.1) - s1) / np.maximum(s0 - s1, 1e-300))
        out[f"p{c}"] = np.clip(pc, 0.0, 1.0)
    return out


def summary_metrics(sm, t, low, thr=60.0, median_width=False):
    """Point, interval and threshold measures of a predictive distribution (see posterior_logp)."""
    from sklearn.metrics import roc_auc_score

    out = {"mae": float(np.mean(np.abs(sm["q50"] - t))), "width80": float(np.mean(sm["q90"] - sm["q10"])),
           "cover80": float(np.mean((t > sm["q10"]) & (t <= sm["q90"])))}
    if median_width:
        out["width80med"] = float(np.median(sm["q90"] - sm["q10"]))
    m = low < thr  # P(T > 60) is informative only while T > 60 is still uncertain
    y = (t[m] > thr).astype(float)
    if m.sum() > 0 and 0 < y.mean() < 1:
        out["brier60"] = float(np.mean((sm["p60"][m] - y) ** 2))
        out["auc60"] = float(roc_auc_score(y, sm["p60"][m]))
    return out


# ----------------------------------------------------------------------------- experiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ring4")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--split", default="cv", choices=["cv", "temporal"])
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--combined-only", action="store_true", help="only the states that combine arrival and elapsed time")
    ap.add_argument("--save-detail", action="store_true", help="also save per-incident predictive summaries of all models "
                    "(evidence_detail_*.npz) and summary metrics of the linear / XGBoost stage-wise models")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    from sklearn.model_selection import StratifiedKFold, train_test_split

    data, r = load_with_stages(args.data, args.tau)
    if args.split == "temporal":
        # test years lie outside the training range, so calendar year is not used as a feature here
        data.numeric = [v for v in data.numeric if v != "year"]
    t, c = data.t, data.c
    C = int(c.max()) + 1
    has_r = r is not None
    ok_r = ((r > 0) & (r < t)) if has_r else np.zeros(len(t), bool)
    print(f"{args.data}: n={len(t)}, C={C}, arrival available for {ok_r.mean():.3f}", flush=True)
    rows = []
    from evodur.splits import iter_splits

    for _, fold, tr_, va, te in iter_splits(data, args.split, 5, 1, args.train_frac, args.folds):
        enc = encoder(data).fit(data.frame.iloc[tr_])
        Xtr, Xva, Xte = (enc.transform(data.frame.iloc[i]).astype(np.float32) for i in (tr_, va, te))
        tau = np.full(len(t), args.tau, dtype=np.float32)
        B = args.batch
        oh = lambda cc: np.eye(C, dtype=np.float32)[cc]

        prior = fit_mixaft(Xtr, t[tr_], tau[tr_], Xva, t[va], tau[va], seed=fold, batch=B)
        ev_met = fit_evidence(lambda d: MethodEvidence(d, C), Xtr, c[tr_], t[tr_], Xva, c[va], t[va], seed=fold, batch=B)
        only = args.combined_only
        base = {} if only else {"method": fit_stage(np.hstack([Xtr, oh(c[tr_])]), t[tr_], tau[tr_], np.hstack([Xva, oh(c[va])]), t[va], tau[va], seed=fold, batch=B)}
        E_met = lambda cc: ((lambda Xq, a, tq: ev_met.logp(Xq, torch.as_tensor(a, dtype=torch.long), tq)), cc)

        def bl(model, X, tt, low):
            with torch.no_grad():
                # on (low, T_MAX], the support the update is normalised on, so both are scored on the same range
                return model.loglik(torch.as_tensor(X), torch.as_tensor(tt, dtype=torch.float32), torch.as_tensor(low, dtype=torch.float32),
                                    upper=T_MAX).numpy()

        res = {}

        stage_lp = {}
        detail = {}  # state -> {"meta": (idx, r), "who": {name: (summary dict, per-incident log score)}}; only with --save-detail

        def run_state(name, evs, X, tt, low, models, Xb, idx=None):
            """Score the update and the separate models for one information state, with predictive summaries."""
            if not isinstance(models, dict):
                models = {"separate": models}
            model = models["separate"]
            ev_lp, ev_s = posterior_logp(prior, evs, X, tt, low, summary=True)
            _, bl_s = posterior_logp(model, [], Xb, tt, low, summary=True)
            res[name] = (ev_lp, bl(model, Xb, tt, low), ev_s, bl_s, np.asarray(tt), np.asarray(low))
            stage_lp[name] = {k: (bl(m, Xb, tt, low) if k == "linear" else m.loglik(Xb, tt, low, upper=T_MAX))
                              for k, m in models.items() if k != "separate"}
            if args.save_detail:
                ridx = np.asarray(idx)
                r_ = np.full(len(ridx), np.nan, np.float32)
                if has_r:
                    r_[ok_r[ridx]] = r[ridx][ok_r[ridx]]
                whos = {"evidence": (ev_s, res[name][0]), "separate": (bl_s, res[name][1])}
                for k, m in models.items():
                    if k == "linear":
                        whos[k] = (posterior_logp(m, [], Xb, tt, low, summary=True)[1], stage_lp[name][k])
                    elif k == "xgb":
                        whos[k] = (xgb_summary(m, Xb, low), stage_lp[name][k])
                detail[name] = {"meta": (ridx, r_), "who": whos}

        run_state("report", [], Xte, t[te], tau[te], prior, Xte, te)
        if not only:
            run_state("method", [E_met(c[te])], Xte, t[te], tau[te], base["method"], np.hstack([Xte, oh(c[te])]), te)
        if has_r:
            a_tr, a_va = tr_[ok_r[tr_]], va[ok_r[va]]
            ev_arr = fit_evidence(ArrivalEvidence, Xtr[ok_r[tr_]], r[a_tr], t[a_tr], Xva[ok_r[va]], r[a_va], t[a_va], seed=fold, batch=B)
            lr_ = lambda idx: np.log(np.clip(r[idx], 0.1, None)).astype(np.float32)[:, None]
            if not only:
                base["arrival"] = fit_stage(np.hstack([Xtr[ok_r[tr_]], lr_(a_tr)]), t[a_tr], np.maximum(r[a_tr], args.tau),
                                            np.hstack([Xva[ok_r[va]], lr_(a_va)]), t[a_va], np.maximum(r[a_va], args.tau), seed=fold, batch=B)
                base["arrival+method"] = fit_stage(np.hstack([Xtr[ok_r[tr_]], lr_(a_tr), oh(c[a_tr])]), t[a_tr], np.maximum(r[a_tr], args.tau),
                                                   np.hstack([Xva[ok_r[va]], lr_(a_va), oh(c[a_va])]), t[a_va], np.maximum(r[a_va], args.tau), seed=fold, batch=B)
            te_r = te[ok_r[te]]
            Xr_te = Xte[ok_r[te]]
            E_arr = lambda rr: ((lambda Xq, a, tq: ev_arr.logp(Xq, torch.as_tensor(a, dtype=torch.float32), tq)), rr)
            # that an arrival was recorded at all is evidence too (dispatch data record no arrival for many short
            # incidents): P(arrival recorded | t, x), fitted on all training incidents
            occ = lambda n: []
            if not ok_r.all():
                ev_occ = fit_evidence(lambda d: MethodEvidence(d, 2), Xtr, ok_r[tr_].astype(np.int64), t[tr_], Xva,
                                      ok_r[va].astype(np.int64), t[va], seed=fold, batch=B)
                occ = lambda n: [((lambda Xq, a, tq: ev_occ.logp(Xq, torch.as_tensor(a, dtype=torch.long), tq)), np.ones(n, np.int64))]
            E_occ = occ(len(te_r))
            low_r = np.maximum(r[te_r], args.tau)
            if not only:
                run_state("arrival", [E_arr(r[te_r])] + E_occ, Xr_te, t[te_r], low_r, base["arrival"], np.hstack([Xr_te, lr_(te_r)]), te_r)
                run_state("arrival+method", [E_arr(r[te_r]), E_met(c[te_r])] + E_occ, Xr_te, t[te_r], low_r, base["arrival+method"],
                          np.hstack([Xr_te, lr_(te_r), oh(c[te_r])]), te_r)
            # chain rule: the label module also sees the arrival time (no independence assumption)
            cr = lambda idx: np.column_stack([c[idx], r[idx]]).astype(np.float32)
            ev_met_r = fit_evidence(lambda d: MethodGivenArrival(d, C), Xtr[ok_r[tr_]], cr(a_tr), t[a_tr], Xva[ok_r[va]], cr(a_va),
                                    t[a_va], seed=fold, batch=B)
            E_metr = lambda idx: ((lambda Xq, a, tq: ev_met_r.logp(Xq, torch.as_tensor(a, dtype=torch.float32), tq)), cr(idx))
            if not only:
                run_state("arrival+method-chain", [E_arr(r[te_r]), E_metr(te_r)] + E_occ, Xr_te, t[te_r], low_r, base["arrival+method"],
                          np.hstack([Xr_te, lr_(te_r), oh(c[te_r])]), te_r)
            # evidence combined: the police are on scene (r < s) and the incident is still open at s. The stage-wise
            # models are fitted on the training incidents in exactly that state.
            with np.errstate(invalid="ignore"):
                in_state = lambda idx: ok_r[idx] & (r[idx] < s) & (t[idx] > s)
                for s in (20.0, 30.0, 45.0, 60.0, 90.0, 120.0):
                    mtr, mva, mte = in_state(tr_), in_state(va), in_state(te)
                    if min(mtr.sum(), mva.sum(), mte.sum()) < 50:
                        continue
                    i_tr, i_va, i_te = tr_[mtr], va[mva], te[mte]
                    fs = lambda X_, i_, lab: np.hstack([X_, lr_(i_)] + ([oh(c[i_])] if lab else []))
                    lo_tr, lo_va, lo_te = (np.full(m.sum(), s, np.float32) for m in (mtr, mva, mte))
                    for lab in (False, True):
                        models = fit_stage(fs(Xtr[mtr], i_tr, lab), t[i_tr], lo_tr, fs(Xva[mva], i_va, lab), t[i_va], lo_va,
                                           seed=fold, batch=B)
                        evs = [E_arr(r[i_te])] + ([E_metr(i_te)] if lab else []) + occ(len(i_te))
                        run_state(f"arrival+elapsed{int(s)}" + ("+method" if lab else ""), evs, Xte[mte], t[i_te], lo_te,
                                  models, fs(Xte[mte], i_te, lab), i_te)
        for s in (() if only else (10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0)):
            m_tr, m_va = t[tr_] > s, t[va] > s
            lm = fit_stage(Xtr[m_tr], t[tr_][m_tr], np.full(m_tr.sum(), s, np.float32), Xva[m_va], t[va][m_va],
                            np.full(m_va.sum(), s, np.float32), seed=fold, batch=B)
            lmc = fit_stage(np.hstack([Xtr[m_tr], oh(c[tr_][m_tr])]), t[tr_][m_tr], np.full(m_tr.sum(), s, np.float32),
                             np.hstack([Xva[m_va], oh(c[va][m_va])]), t[va][m_va], np.full(m_va.sum(), s, np.float32), seed=fold, batch=B)
            idx = te[t[te] > s]
            lo = np.full(len(idx), s, np.float32)
            Xs = Xte[t[te] > s]
            run_state(f"elapsed{int(s)}", [], Xs, t[idx], lo, lm, Xs, idx)
            run_state(f"elapsed{int(s)}+method", [E_met(c[idx])], Xs, t[idx], lo, lmc, np.hstack([Xs, oh(c[idx])]), idx)
        per_incident, det_arr = {}, {}
        for state, (ev_lp, bl_lp, ev_s, bl_s, tt, low) in res.items():
            row = {"data": args.data, "split": args.split, "train_frac": args.train_frac, "n_train": len(tr_),
                   "fold": fold, "state": state, "n": len(ev_lp),
                   "evidence_logp": float(np.mean(ev_lp)), "separate_model_logp": float(np.mean(bl_lp))}
            for k, lp in stage_lp.get(state, {}).items():
                row[f"separate_{k}_logp"] = float(np.mean(lp))
                per_incident[f"{state}|separate_{k}"] = np.asarray(lp, np.float32)
            for who, sm in (("evidence", ev_s), ("separate", bl_s)):
                row.update({f"{who}_{k}": v for k, v in summary_metrics(sm, tt, low, median_width=args.save_detail).items()})
            if args.save_detail:
                d = detail[state]
                ridx, r_ = d["meta"]
                det_arr.update({f"{state}|t": np.asarray(tt, np.float32), f"{state}|low": np.asarray(low, np.float32),
                                f"{state}|r": r_, f"{state}|idx": ridx.astype(np.int64)})
                for who, (sm, lp) in d["who"].items():
                    det_arr[f"{state}|{who}|lp"] = np.asarray(lp, np.float32)
                    det_arr.update({f"{state}|{who}|{k}": np.asarray(sm[k], np.float32) for k in ("q10", "q50", "q90")})
                    det_arr.update({f"{state}|{who}|p{c}": np.clip(sm[f"p{c}"], 0.0, 1.0).astype(np.float32) for c in CUTS})  # grid sums can round to 1 + 1e-7
                    if who in ("linear", "xgb"):
                        row.update({f"separate_{who}_{k}": v for k, v in summary_metrics(sm, tt, low, median_width=True).items()})
            rows.append(row)
            per_incident[f"{state}|evidence"] = ev_lp.astype(np.float32)
            per_incident[f"{state}|separate"] = bl_lp.astype(np.float32)
            print(rows[-1], flush=True)
        np.savez_compressed(OUT / f"evidence_incidents_{args.data}{args.tag}_fold{fold}.npz", **per_incident)
        if args.save_detail:
            np.savez_compressed(OUT / f"evidence_detail_{args.data}{args.tag}_fold{fold}.npz", **det_arr)
        pd.DataFrame(rows).to_csv(OUT / f"evidence_states_{args.data}{args.tag}.csv", index=False)
    df = pd.DataFrame(rows)
    summ = df.groupby("state")[["n", "evidence_logp", "separate_model_logp"]].mean()
    summ["evidence_minus_separate"] = summ["evidence_logp"] - summ["separate_model_logp"]
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
