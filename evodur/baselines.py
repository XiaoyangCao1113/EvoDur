"""Baselines with the same interface as the DeepCAFT models:
fit(X, t, c, X_val, t_val, c_val, seed), log_density(X, t, c=None),
survival(X, grid, c=None). log_density returns None when the model has no
continuous density (RSF, DeepSurv).

All durations are left-truncated at tau (only T > tau observed).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn


def _mlp(d, out, hidden=(64, 64), dropout=0.1):
    layers, prev = [], d
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    return nn.Sequential(*layers, nn.Linear(prev, out))


# same budget as the mixture models and EvoDur (model.Config): best of three initialisations, 400 epochs, patience 25
N_INIT = 3


def _best_of(make_net, fit_net, seed):
    """Train N_INIT networks from different initialisations and keep the one with the best validation loss."""
    best = None
    for i in range(N_INIT):
        torch.manual_seed(seed * 100 + i)
        net = make_net()
        v = fit_net(net, seed * 100 + i)
        if best is None or v < best[0]:
            best = (v, net)
    return best[1], best[0]


def _train(net, loss_fn, n, val_fn, lr=2e-3, wd=1e-4, batch=256, max_epochs=400, patience=25, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best, state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        net.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            loss = loss_fn(perm[s:s + batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(val_fn())
        if v < best - 1e-5:
            best, bad, state = v, 0, copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    net.eval()
    return best


class DeepHitCR:
    """DeepHit (Lee et al., 2018) for competing risks: softmax over (cause, time
    bin), with a last open bin [edge_B, inf) per cause. Trained on the
    likelihood only (no ranking loss) so that its log-likelihood is comparable.
    The density is piecewise constant within bins; the open bin spreads its
    mass uniformly over the next 24 h."""

    TAIL = 1440.0

    def __init__(self, tau, n_methods, n_bins=20, batch=256):
        self.tau, self.C, self.B, self.batch = tau, n_methods, n_bins, batch

    def fit(self, X, t, c, X_val, t_val, c_val, seed=0):
        lo = max(self.tau, 0.5)
        qs = np.quantile(t, np.linspace(0, 0.99, self.B + 1)[1:])
        self.edges = np.unique(np.concatenate([[lo], qs]))
        self.B = len(self.edges)  # bins: B-1 closed + 1 open
        self.width = np.concatenate([np.diff(self.edges), [self.TAIL]])
        X, Xv = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(X_val, dtype=torch.float32)
        cell, cell_v = self._cell(t, c), self._cell(t_val, c_val)
        ce = nn.CrossEntropyLoss()
        self.net, self.val_nll = _best_of(
            lambda: _mlp(X.shape[1], self.C * self.B),
            lambda net, sd: _train(net, lambda b: ce(net(X[b]), cell[b]), len(t),
                                   lambda: ce(net(Xv), cell_v), lr=1e-3, batch=self.batch, seed=sd), seed)
        return self

    def _bin(self, t):
        return np.clip(np.searchsorted(self.edges, t, side="right") - 1, 0, self.B - 1)

    def _cell(self, t, c):
        return torch.as_tensor(c * self.B + self._bin(t), dtype=torch.long)

    @torch.no_grad()
    def _pmf(self, X):
        p = torch.softmax(self.net(torch.as_tensor(X, dtype=torch.float32)), dim=1).numpy()
        return p.reshape(len(X), self.C, self.B)

    def log_density(self, X, t, c=None):
        pmf = self._pmf(X)
        b, n = self._bin(t), np.arange(len(t))
        if c is None:
            dens = pmf.sum(1)[n, b] / self.width[b]
        else:
            dens = pmf[n, c, b] / pmf[n, c, :].sum(1) / self.width[b]
        return np.log(dens + 1e-12)

    def survival(self, X, grid, c=None):
        pmf = self._pmf(X)
        n = np.arange(len(X))
        m = pmf.sum(1) if c is None else pmf[n, c, :] / pmf[n, c, :].sum(1, keepdims=True)
        ends = np.concatenate([self.edges, [self.edges[-1] + self.TAIL]])
        S_edges = np.concatenate([np.ones((len(X), 1)), 1 - np.cumsum(m, axis=1)], axis=1).clip(0, 1)
        out = np.empty((len(X), len(grid)))
        for j, g in enumerate(grid):
            if g <= ends[0]:
                out[:, j] = 1.0
            elif g >= ends[-1]:
                out[:, j] = 0.0
            else:
                k = np.searchsorted(ends, g, side="right") - 1
                w = (g - ends[k]) / (ends[k + 1] - ends[k])
                out[:, j] = S_edges[:, k] * (1 - w) + S_edges[:, k + 1] * w
        return out


class WeibullMDN:
    """Mixture density network with K Weibull components (in the spirit of
    Kidando et al., 2024): weights, scales and shapes all depend on x."""

    def __init__(self, tau, K=4, batch=256):
        self.tau, self.K, self.batch = tau, K, batch

    def _parts(self, X, t):
        o = self.net(X)
        K = self.K
        log_w = torch.log_softmax(o[:, :K], 1)
        log_lam = o[:, K:2 * K] + float(np.log(30.0))
        k = nn.functional.softplus(o[:, 2 * K:]) + 0.3
        tt = t[:, None]
        z = (tt / torch.exp(log_lam)) ** k
        log_f = torch.log(k) - log_lam + (k - 1) * (torch.log(tt) - log_lam) - z
        log_S = -z
        if self.tau > 0:
            ztau = (self.tau / torch.exp(log_lam)) ** k
            norm = torch.logsumexp(log_w - ztau, 1)
        else:
            norm = torch.zeros(len(t))
        return log_w, log_f, log_S, norm, log_lam, k

    def _ll(self, X, t):
        log_w, log_f, _, norm, _, _ = self._parts(X, t)
        return torch.logsumexp(log_w + log_f, 1) - norm

    def fit(self, X, t, c, X_val, t_val, c_val, seed=0):
        X, Xv = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(X_val, dtype=torch.float32)
        tt, tv = torch.as_tensor(t, dtype=torch.float32), torch.as_tensor(t_val, dtype=torch.float32)
        def fit_net(net, sd):
            self.net = net  # _ll reads self.net
            return _train(net, lambda b: -self._ll(X[b], tt[b]).mean(), len(t),
                          lambda: -self._ll(Xv, tv).mean(), batch=self.batch, seed=sd)
        self.net, self.val_nll = _best_of(lambda: _mlp(X.shape[1], 3 * self.K), fit_net, seed)
        return self

    @torch.no_grad()
    def log_density(self, X, t, c=None):
        return self._ll(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(t, dtype=torch.float32)).numpy()

    @torch.no_grad()
    def survival(self, X, grid, c=None):
        X = torch.as_tensor(X, dtype=torch.float32)
        log_w, _, _, norm, log_lam, k = self._parts(X, torch.full((len(X),), float(grid[0])))
        g = torch.as_tensor(grid, dtype=torch.float32)
        z = (g[None, None, :] / torch.exp(log_lam)[:, :, None]) ** k[:, :, None]
        S = (torch.exp(log_w)[:, :, None] * torch.exp(-z)).sum(1) / torch.exp(norm)[:, None]
        return S.clamp(0, 1).numpy()


class XGBoostAFT:
    """XGBoost with the survival:aft objective, log-logistic errors (Barnwal et al., 2022).
    XGBoost has no entry time, so it models the excess duration T - tau
    (plus 0.1 min), which gives a proper distribution on T > tau."""

    def __init__(self, tau, sigma=1.0):
        self.tau, self.sigma = tau, sigma

    def fit(self, X, t, c, X_val, t_val, c_val, seed=0):
        import xgboost as xgb

        y, yv = t - self.tau + 0.1, t_val - self.tau + 0.1
        d = xgb.DMatrix(X, label_lower_bound=y, label_upper_bound=y)
        dv = xgb.DMatrix(X_val, label_lower_bound=yv, label_upper_bound=yv)
        best = None
        for sigma in (0.3, 0.4, 0.5, 0.8):
            for depth in (3, 5):
                params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "logistic",
                          "aft_loss_distribution_scale": sigma, "learning_rate": 0.05, "max_depth": depth,
                          "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "seed": seed, "nthread": 4,
                          # start from the median duration; the default (0.5) is far off and the small-scale
                          # fits then fail to converge within the round limit
                          "base_score": float(np.median(y))}
                m = xgb.train(params, d, 2000, evals=[(dv, "val")], early_stopping_rounds=50, verbose_eval=False)
                self.model, self.sigma, self.best_iter = m, sigma, m.best_iteration
                v = -float(self.log_density(X_val, t_val).mean())
                if not np.isfinite(v):
                    continue
                if best is None or v < best[0]:
                    best = (v, m, sigma, m.best_iteration)
        self.val_nll, self.model, self.sigma, self.best_iter = best
        return self

    def _mu(self, X):
        import xgboost as xgb

        return np.log(self.model.predict(xgb.DMatrix(X), iteration_range=(0, self.best_iter + 1)))

    def log_density(self, X, t, c=None):
        mu, s = self._mu(X), self.sigma
        y = t - self.tau + 0.1
        z = (np.log(y) - mu) / s
        return -np.log(s * y) + z - 2 * np.logaddexp(0, z)

    def survival(self, X, grid, c=None):
        mu, s = self._mu(X), self.sigma
        y = np.clip(grid - self.tau + 0.1, 1e-6, None)
        return 1 / (1 + np.exp((np.log(y)[None, :] - mu[:, None]) / s))


class RSF:
    def __init__(self, tau, seed=0):
        self.tau, self.seed = tau, seed

    def fit(self, X, t, c, X_val, t_val, c_val, seed=0):
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv

        if len(t) > 20000:  # RSF memory grows with n; fit on a 20k subsample
            sub = np.random.default_rng(seed).choice(len(t), 20000, replace=False)
            X, t = X[sub], t[sub]
        if len(np.unique(t)) > 10000:  # RSF memory also grows with the number of unique times (Sydney: all unique);
            # round each time up to the next point of run.GRID: leaf survival at every evaluation grid point is unchanged
            snap = np.append(np.unique(np.concatenate([np.linspace(0.5, 180, 360), np.geomspace(180, 1440, 80)])), 1441.0)
            t = snap[np.minimum(np.searchsorted(snap, t, side="left"), len(snap) - 1)]
        best = None
        for leaf in (10, 30):
            m = RandomSurvivalForest(n_estimators=200, min_samples_leaf=leaf, max_features="sqrt", n_jobs=4,
                                     random_state=seed, low_memory=False)
            m.fit(X, Surv.from_arrays(np.ones(len(t), bool), t))
            self.model = m
            grid = np.linspace(max(self.tau, 1), 240, 120)
            S = self.survival(X_val, grid)
            alive = (t_val[:, None] > grid[None, :]).astype(float)
            ibs = float(((alive - S) ** 2).mean())
            if best is None or ibs < best[0]:
                best = (ibs, m)
        self.val_nll, self.model = best
        return self

    def log_density(self, X, t, c=None):
        return None

    def survival(self, X, grid, c=None):
        fns = self.model.predict_survival_function(X, return_array=True)
        times = self.model.unique_times_
        idx = np.searchsorted(times, grid, side="right") - 1
        return np.where(idx[None, :] >= 0, fns[:, np.clip(idx, 0, None)], 1.0)


class DeepSurv:
    """Cox model with an MLP log-risk (Katzman et al., 2018); Breslow baseline."""

    def __init__(self, tau, batch=256):
        self.tau, self.batch = tau, batch

    @staticmethod
    def _cox(g, t):
        order = torch.argsort(t, descending=True)
        g = g[order]
        return -(g - torch.logcumsumexp(g, 0)).mean()

    def fit(self, X, t, c, X_val, t_val, c_val, seed=0):
        X, Xv = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(X_val, dtype=torch.float32)
        tt, tv = torch.as_tensor(t, dtype=torch.float32), torch.as_tensor(t_val, dtype=torch.float32)
        self.net, self.val_nll = _best_of(
            lambda: _mlp(X.shape[1], 1),
            lambda net, sd: _train(net, lambda b: self._cox(net(X[b]).squeeze(-1), tt[b]), len(t),
                                   lambda: self._cox(net(Xv).squeeze(-1), tv), batch=self.batch, seed=sd), seed)
        with torch.no_grad():
            g = self.net(X).squeeze(-1).numpy()
        order = np.argsort(t)
        ts, eg = t[order], np.exp(g[order])
        H = np.cumsum(1.0 / np.cumsum(eg[::-1])[::-1])
        self.ts, self.H = ts, H
        return self

    def log_density(self, X, t, c=None):
        return None

    def survival(self, X, grid, c=None):
        with torch.no_grad():
            g = self.net(torch.as_tensor(X, dtype=torch.float32)).squeeze(-1).numpy()
        H0 = np.concatenate([[0.0], self.H])[np.searchsorted(self.ts, grid, side="right")]
        return np.exp(-np.outer(np.exp(g), H0))
