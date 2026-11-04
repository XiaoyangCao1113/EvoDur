"""Neural Fine-Gray (Jeanselme et al., 2023, CHIL; arXiv:2305.06703) as a baseline.

The network is the authors' NeuralFineGrayTorch (third_party/nfg_torch.py).
Joint model: p(c = k, t | x) = beta_k(x) f_k(t | x), with S_k = exp(-H_k) and
H_k(t) = t * g_k(x, t) from a positive (monotone) network, so f_k = h_k S_k
with h_k = dH_k/dt from autograd. All events are observed here (no censoring).

Changes needed for our setting (the network itself is unchanged):
- left truncation at tau: the likelihood is divided by sum_k beta_k S_k(tau);
- time is mapped to u in [0, 1] for the network, either linearly (u = t / 1440)
  or on the log scale (u = ln t / ln 1440, needs tau >= 1); densities get the
  Jacobian du/dt. The scale is chosen on validation data together with the
  network size.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from evodur.third_party.nfg_torch import NeuralFineGrayTorch

T_SCALE = 1440.0


class NeuralFineGrayCR:
    def __init__(self, tau, n_methods, batch=256, layers=(64, 64), layers_surv=(64,), dropout=0.1,
                 lr=1e-3, weight_decay=1e-4, patience=25, max_epochs=400):
        self.tau, self.C, self.batch = tau, n_methods, batch
        self.hp = dict(layers=list(layers), layers_surv=list(layers_surv), dropout=dropout)
        self.lr, self.wd, self.patience, self.max_epochs = lr, weight_decay, patience, max_epochs

    # ------------------------------------------------------------------ pieces
    def _all(self, X, u):
        """log beta (n,K), log S_k(u) (n,K), log h_k(u) (n,K) in the u scale."""
        log_sr, log_b, tau_t = self.net(X, u)
        hs = []
        for k in range(self.C):
            (g,) = torch.autograd.grad(-log_sr[:, k].sum(), tau_t, create_graph=self.net.training, retain_graph=True)
            hs.append(g[:, 0].clamp_min(1e-10))
        return log_b, log_sr, torch.log(torch.stack(hs, 1))

    def _log_trunc(self, X):
        if self.tau <= 0:
            return torch.zeros(len(X), dtype=torch.float64)
        log_sr, log_b, _ = self.net(X, torch.full((len(X),), float(self._u([self.tau])[0]), dtype=torch.float64))
        return torch.logsumexp(log_b + log_sr, 1)

    def _joint(self, X, u, c, log_jac):
        """log p(c, t | x, T > tau) in the t scale."""
        log_b, log_sr, log_h = self._all(X, u)
        n = torch.arange(len(c))
        return log_b[n, c] + log_sr[n, c] + log_h[n, c] + log_jac - self._log_trunc(X)

    # ------------------------------------------------------------------ fit
    GRID = (
        dict(layers=[100, 100, 100], layers_surv=[100], dropout=0.0, lr=1e-3, scale="log"),  # authors' default sizes
        dict(layers=[64, 64], layers_surv=[64], dropout=0.1, lr=1e-3, scale="log"),
        dict(layers=[64, 64], layers_surv=[64], dropout=0.1, lr=3e-3, scale="log"),
        dict(layers=[100, 100, 100], layers_surv=[100], dropout=0.0, lr=1e-3, scale="linear"),
    )
    scale = "linear"

    def _u(self, t):
        t = np.asarray(t, dtype=float)
        return t / T_SCALE if self.scale == "linear" else np.log(t) / np.log(T_SCALE)

    def _log_jac(self, t):
        """log du/dt."""
        t = np.asarray(t, dtype=float)
        return np.full(len(t), -np.log(T_SCALE)) if self.scale == "linear" else -np.log(t) - np.log(np.log(T_SCALE))

    def fit(self, X, t, c, X_val, t_val, c_val, seed=0):
        """Small grid (GRID), chosen on validation likelihood."""
        best = None
        for g in self.GRID:
            if g["scale"] == "log" and self.tau < 1:
                continue
            self.hp = dict(layers=g["layers"], layers_surv=g["layers_surv"], dropout=g["dropout"])
            self.lr, self.scale = g["lr"], g["scale"]
            self._fit_one(X, t, c, X_val, t_val, c_val, seed)
            if best is None or self.val_nll < best[0]:
                best = (self.val_nll, copy.deepcopy(self.net), dict(g))
        self.val_nll, self.net, self.chosen = best
        self.scale = self.chosen["scale"]
        return self

    def _fit_one(self, X, t, c, X_val, t_val, c_val, seed=0):
        torch.manual_seed(seed)
        X, Xv = (torch.as_tensor(a, dtype=torch.float64) for a in (X, X_val))
        u, uv = (torch.as_tensor(self._u(a), dtype=torch.float64) for a in (t, t_val))
        lj, ljv = (torch.as_tensor(self._log_jac(a), dtype=torch.float64) for a in (t, t_val))
        c, cv = (torch.as_tensor(a, dtype=torch.long) for a in (c, c_val))
        self.net = NeuralFineGrayTorch(X.shape[1], risks=self.C, **self.hp).double()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, state, bad = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(len(u))
            for s in range(0, len(u), self.batch):
                b = perm[s:s + self.batch]
                if len(b) < 2:
                    continue
                loss = -self._joint(X[b], u[b], c[b], lj[b]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            v = float(-self._joint(Xv, uv, cv, ljv).mean().detach())
            if v < best - 1e-5:
                best, bad, state = v, 0, copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(state)
        self.net.eval()
        self.val_nll = best
        return self

    # ------------------------------------------------------------------ predict
    def log_density(self, X, t, c=None):
        X = torch.as_tensor(X, dtype=torch.float64)
        u = torch.as_tensor(self._u(t), dtype=torch.float64)
        log_b, log_sr, log_h = self._all(X, u)
        if c is None:  # report time: marginal over methods
            lp = torch.logsumexp(log_b + log_sr + log_h, 1) - self._log_trunc(X)
        else:  # method known: f_c(t) / S_c(tau)
            n = torch.arange(len(c))
            cc = torch.as_tensor(c, dtype=torch.long)
            lp = log_sr[n, cc] + log_h[n, cc]
            if self.tau > 0:
                ls_tau, _, _ = self.net(X, torch.full((len(X),), float(self._u([self.tau])[0]), dtype=torch.float64))
                lp = lp - ls_tau[n, cc]
        return (lp.detach().numpy() + self._log_jac(t))

    @torch.no_grad()
    def survival(self, X, grid, c=None):
        X = torch.as_tensor(X, dtype=torch.float64)
        n = len(X)
        out = np.empty((n, len(grid)))
        if self.tau > 0:
            ls_tau, lb, _ = self.net(X, torch.full((n,), float(self._u([self.tau])[0]), dtype=torch.float64))
        for j, g in enumerate(grid):
            if g < max(self.tau, 1e-9):
                out[:, j] = 1.0
                continue
            log_sr, log_b, _ = self.net(X, torch.full((n,), float(self._u([g])[0]), dtype=torch.float64))
            if c is None:
                S = (log_b + log_sr).exp().sum(1)
                if self.tau > 0:
                    S = S / (lb + ls_tau).exp().sum(1)
            else:
                cc = torch.as_tensor(c, dtype=torch.long)
                S = log_sr[torch.arange(n), cc].exp()
                if self.tau > 0:
                    S = S / ls_tau[torch.arange(n), cc].exp()
            out[:, j] = S.clamp(0, 1).numpy()
        return out
