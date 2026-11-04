"""Neural competing-risk mixture of log-logistic AFT models.

    p(c, t | x) = sum_z pi_z(x) P(c | z) f_z(t | x)

f_z is a log-logistic AFT density with location h_z(x) and baseline
(alpha_z, beta_z): S_z(t|x) = 1 / (1 + (t e^{h_z(x)} / alpha_z)^{beta_z}).
pi_z(x) is a softmax over K types. c is the recorded clearance method.

emission="identity": z = c (K = number of methods). This is the competing-risk
                     mixture of Li et al. (2015); with a neural network it is
                     DeepCAFT.
emission="learned":  c is an observation of the latent type z with a learned
                     K x C matrix P(c|z) (latent DeepCAFT).
emission="contextual": log P(c | z, x) = log_softmax(e_z + w(x)). w(x) absorbs
                     how the recording context (unit, location, time ...)
                     shifts the recorded method; e_z is what the latent type
                     adds. With e_z constant the model is a plain mixture plus
                     an independent classifier for c.
emission="aux":      control for "contextual": same c head but e_z fixed at 0,
                     so c depends on x only (mixture + auxiliary classifier).
emission="block":    each method has its own K / C components (z belongs to
                     method z // (K / C)), so P(c | x) is the summed weight of
                     its block and f(t | c, x) is a mixture of K / C
                     log-logistic components: the mixture form with the same
                     flexibility as a K-component duration model.
emission="none":     c is not used; plain latent mixture.
K = 1:               single AFT (no mixture).
neural=False:        h_z and pi_z are linear in x.

Durations may be left-truncated at tau (only T > tau observed); the
likelihood is then divided by P(T > tau | x) = sum_z pi_z S_z(tau).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

sp = nn.functional.softplus


def loglogistic(log_t, h, log_alpha, log_beta):
    """log f(t) and log S(t); broadcasting over the last axis (types)."""
    beta = torch.exp(log_beta)
    u = beta * (log_t + h - log_alpha)
    log_S = -sp(u)
    log_f = log_beta - log_t + u - 2.0 * sp(u)
    return log_f, log_S


class Net(nn.Module):
    def __init__(self, d, K, C, neural, hidden, dropout, emission, init_seed, init_mode=0, const_pi=False, het_scale=False):
        super().__init__()
        self.K, self.C, self.emission, self.const_pi, self.het_scale = K, C, emission, const_pi, het_scale
        if het_scale:
            self.lin_b = nn.Linear(d, K)
            nn.init.zeros_(self.lin_b.weight)
            nn.init.zeros_(self.lin_b.bias)
        self.lin_h = nn.Linear(d, K)
        self.lin_pi = nn.Linear(d, K)
        self.neural = neural
        if neural:
            layers, prev = [], d
            for hdim in hidden:
                layers += [nn.Linear(prev, hdim), nn.ReLU(), nn.Dropout(dropout)]
                prev = hdim
            self.body = nn.Sequential(*layers)
            self.deep_h = nn.Linear(prev, K)
            self.deep_pi = nn.Linear(prev, K)
            if het_scale:
                self.deep_b = nn.Linear(prev, K)
                nn.init.zeros_(self.deep_b.weight)
                nn.init.zeros_(self.deep_b.bias)
            if emission in ("contextual", "aux"):
                self.deep_c = nn.Linear(prev, C)
        if emission in ("contextual", "aux"):
            self.lin_c = nn.Linear(d, C)
        g = torch.Generator().manual_seed(init_seed)
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(np.geomspace(15.0, 60.0, K) if K > 1 else [30.0], dtype=torch.float32))
                                      + 0.1 * torch.randn(K, generator=g))
        self.log_beta = nn.Parameter(torch.full((K,), 1.0))
        if emission in ("learned", "contextual"):
            # init_mode 0 with K = C starts from the identity (the Li 2015 /
            # DeepCAFT solution); other inits start near "c independent of z"
            # (the plain mixture). The best init is chosen on validation data,
            # so the model can end up anywhere between the two special cases.
            if K == C and init_mode == 0:
                init = torch.log(0.8 * torch.eye(K) + 0.2 / C)
            else:
                init = torch.zeros(K, C)
            self.emit_logits = nn.Parameter(init + 0.3 * torch.randn(K, C, generator=g))

    def forward(self, x):
        """h_z(x), log pi_z(x), and the context logits w(x) for c (or None)."""
        h, logit = self.lin_h(x), self.lin_pi(x)
        wc = self.lin_c(x) if self.emission in ("contextual", "aux") else None
        b = self.lin_b(x) if self.het_scale else None
        if self.neural:
            z = self.body(x)
            h, logit = h + self.deep_h(z), logit + self.deep_pi(z)
            if self.het_scale:
                b = b + self.deep_b(z)
        # log shape of each component: shared, or shifted by a bounded function of x (het_scale)
        self.lb = self.log_beta[None, :] + (1.5 * torch.tanh(b / 1.5) if b is not None else 0.0)
        if self.lb.shape[0] != len(x):
            self.lb = self.lb.expand(len(x), -1)
        if self.const_pi:
            # mixture weights do not depend on x: the components only make the
            # baseline distribution more flexible (control for misspecification)
            logit = self.lin_pi.bias.expand(len(x), -1)
            if wc is not None:
                wc = wc + self.deep_c(z)
        return h, torch.log_softmax(logit, dim=-1), wc

    def log_emit(self, wc=None):
        """log P(c | z [, x]), shape (n or 1, K, C)."""
        if self.emission == "identity":
            return torch.where(torch.eye(self.K, self.C, dtype=torch.bool), 0.0, -1e4)[None]
        if self.emission == "block":
            own = (torch.arange(self.K)[:, None] // (self.K // self.C)) == torch.arange(self.C)[None, :]
            return torch.where(own, 0.0, -1e4)[None]
        if self.emission == "learned":
            return torch.log_softmax(self.emit_logits, dim=1)[None]
        if self.emission == "contextual":
            return torch.log_softmax(self.emit_logits[None] + wc[:, None, :], dim=2)
        if self.emission == "aux":
            return torch.log_softmax(wc, dim=1)[:, None, :].expand(-1, self.K, -1)
        return torch.zeros(1, self.K, self.C)


@dataclass
class Config:
    K: int = 4
    emission: str = "identity"  # identity | learned | none
    neural: bool = True
    hidden: tuple = (64, 64)
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch: int = 256
    max_epochs: int = 400
    patience: int = 25
    tau: float = 0.0  # left-truncation point (minutes); 0 = none
    n_init: int = 3
    const_pi: bool = False
    het_scale: bool = False  # component shapes depend on x


class CRMixture:
    def __init__(self, cfg: Config, n_methods: int):
        self.cfg, self.C = cfg, n_methods
        if cfg.emission == "identity":
            assert cfg.K == n_methods, "identity emission needs K = number of methods"
        if cfg.emission == "block":
            assert cfg.K % n_methods == 0, "block emission needs K = m x number of methods"

    # ---------------------------------------------------------------- likelihood
    def _parts(self, net, X, log_t):
        h, log_pi, wc = net(X)
        log_f, log_S = loglogistic(log_t[:, None], h, net.log_alpha[None, :], net.lb)
        if self.cfg.tau > 0:
            _, log_Stau = loglogistic(torch.full_like(log_t, float(np.log(self.cfg.tau)))[:, None], h,
                                      net.log_alpha[None, :], net.lb)
            log_norm = torch.logsumexp(log_pi + log_Stau, dim=1)
        else:
            log_norm = torch.zeros(len(log_t))
        return log_f, log_S, log_pi, log_norm, wc

    @staticmethod
    def _emit_at(net, wc, c):
        """log P(c_i | z, x_i), shape (n, K)."""
        E = net.log_emit(wc)
        E = E.expand(len(c), -1, -1)
        return E.gather(2, c[:, None, None].expand(-1, E.shape[1], 1)).squeeze(2)

    def _joint_ll(self, net, X, log_t, c):
        """log p(c, t | x, T > tau) per row."""
        log_f, _, log_pi, log_norm, wc = self._parts(net, X, log_t)
        e = self._emit_at(net, wc, c)  # (n, K)
        return torch.logsumexp(log_pi + e + log_f, dim=1) - log_norm

    # ---------------------------------------------------------------- training
    def fit(self, X, t, c, X_val, t_val, c_val, seed=0):
        cfg = self.cfg
        X, X_val = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(X_val, dtype=torch.float32)
        lt, lt_val = torch.log(torch.as_tensor(t, dtype=torch.float32)), torch.log(torch.as_tensor(t_val, dtype=torch.float32))
        c, c_val = torch.as_tensor(c, dtype=torch.long), torch.as_tensor(c_val, dtype=torch.long)
        best_v, best_net = np.inf, None
        for i in range(cfg.n_init):
            torch.manual_seed(seed * 100 + i)
            net = Net(X.shape[1], cfg.K, self.C, cfg.neural, cfg.hidden, cfg.dropout, cfg.emission, seed * 100 + i, init_mode=i, const_pi=cfg.const_pi,
                      het_scale=cfg.het_scale)
            dist = [p for n, p in net.named_parameters() if n in ("log_alpha", "log_beta", "emit_logits")]
            rest = [p for n, p in net.named_parameters() if n not in ("log_alpha", "log_beta", "emit_logits")]
            opt = torch.optim.Adam([{"params": rest, "weight_decay": cfg.weight_decay},
                                    {"params": dist, "weight_decay": 0.0}], lr=cfg.lr)
            v_best, state, bad = np.inf, None, 0
            n = len(lt)
            for _ in range(cfg.max_epochs):
                net.train()
                perm = torch.randperm(n)
                for s in range(0, n, cfg.batch):
                    b = perm[s:s + cfg.batch]
                    loss = -self._joint_ll(net, X[b], lt[b], c[b]).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(-self._joint_ll(net, X_val, lt_val, c_val).mean())
                if v < v_best - 1e-5:
                    v_best, bad, state = v, 0, copy.deepcopy(net.state_dict())
                else:
                    bad += 1
                    if bad >= cfg.patience:
                        break
            net.load_state_dict(state)
            if v_best < best_v:
                best_v, best_net = v_best, net
        self.net, self.val_nll = best_net.eval(), best_v
        return self

    # ---------------------------------------------------------------- prediction
    @torch.no_grad()
    def type_weights(self, X, c=None):
        """P(z | x) at report time, or P(z | x, c) once the method is known."""
        X = torch.as_tensor(X, dtype=torch.float32)
        _, log_pi, wc = self.net(X)
        if c is None:
            return log_pi.exp()
        w = log_pi + self._emit_at(self.net, wc, torch.as_tensor(c, dtype=torch.long))
        return torch.softmax(w, dim=1)

    @torch.no_grad()
    def log_density(self, X, t, c=None):
        """log f(t | x [, c], T > tau)."""
        X = torch.as_tensor(X, dtype=torch.float32)
        lt = torch.log(torch.as_tensor(t, dtype=torch.float32))
        log_f, _, log_pi, _, _ = self._parts(self.net, X, lt)
        log_w = torch.log(self.type_weights(X, c) + 1e-30)
        num = torch.logsumexp(log_w + log_f, dim=1)
        if self.cfg.tau > 0:
            h, _, _ = self.net(X)
            _, log_Stau = loglogistic(torch.full_like(lt, float(np.log(self.cfg.tau)))[:, None], h,
                                      self.net.log_alpha[None, :], self.net.lb)
            num = num - torch.logsumexp(log_w + log_Stau, dim=1)
        return num.numpy()

    @torch.no_grad()
    def survival(self, X, grid, c=None):
        """S(t | x [, c], T > tau) on a time grid, shape (n, G)."""
        X = torch.as_tensor(X, dtype=torch.float32)
        h, _, _ = self.net(X)
        w = self.type_weights(X, c)
        lg = torch.log(torch.as_tensor(grid, dtype=torch.float32))
        _, log_S = loglogistic(lg[None, None, :], h[:, :, None], self.net.log_alpha[None, :, None], self.net.lb[:, :, None])
        S = (w[:, :, None] * log_S.exp()).sum(1)
        if self.cfg.tau > 0:
            _, log_Stau = loglogistic(torch.tensor(float(np.log(self.cfg.tau))), h, self.net.log_alpha[None, :], self.net.lb)
            S = S / (w * log_Stau.exp()).sum(1, keepdim=True)
        return S.clamp(0, 1).numpy()

    @torch.no_grad()
    def emission_matrix(self):
        """P(c|z) for "learned"; for "contextual", softmax(e_z) (type effect at w = 0)."""
        if self.cfg.emission == "contextual":
            return torch.softmax(self.net.emit_logits, dim=1).numpy()
        return self.net.log_emit()[0].exp().numpy()


class EvidenceModel:
    """Evidence factorization p(c, t | x) = f(t | x) P(c | x, t).

    The duration model f(t | x) is fitted on t only, so the report-time
    prediction does not depend on the clearance labels. The label model
    P(c | x, t) treats the recorded clearance method as evidence about the
    duration (this also covers labels that are consequences of a long
    duration, such as "congestion"). Once c is recorded, the prediction is
    updated by Bayes' rule:
        f(t | x, c) = f(t | x) P(c | x, t) / int f(s | x) P(c | x, s) ds.
    """

    T_MAX = 1440.0

    def __init__(self, dur_cfg: Config, n_methods: int, hidden=(64, 64), dropout=0.1, lr=2e-3, weight_decay=1e-4,
                 batch=256, max_epochs=400, patience=25, n_init=3):
        self.dur = CRMixture(dur_cfg, n_methods)
        self.cfg = dur_cfg
        self.C = n_methods
        self.hp = dict(hidden=hidden, dropout=dropout, lr=lr, weight_decay=weight_decay, batch=batch,
                       max_epochs=max_epochs, patience=patience, n_init=n_init)
        lo = max(dur_cfg.tau, 0.5)
        self.grid = np.geomspace(lo, self.T_MAX, 400)

    def _feat(self, X, log_t):
        z = (log_t - self.lt_mean) / self.lt_sd
        return torch.cat([X, z[:, None], (z ** 2)[:, None]], dim=1)

    def fit(self, X, t, c, X_val, t_val, c_val, seed=0):
        self.dur.fit(X, t, c, X_val, t_val, c_val, seed=seed)
        hp = self.hp
        X, X_val = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(X_val, dtype=torch.float32)
        lt, lt_val = torch.log(torch.as_tensor(t, dtype=torch.float32)), torch.log(torch.as_tensor(t_val, dtype=torch.float32))
        c, c_val = torch.as_tensor(c, dtype=torch.long), torch.as_tensor(c_val, dtype=torch.long)
        self.lt_mean, self.lt_sd = float(lt.mean()), float(lt.std())
        F, F_val = self._feat(X, lt), self._feat(X_val, lt_val)
        ce = nn.CrossEntropyLoss()
        # same budget as the prior: best of n_init initialisations on the validation cross-entropy
        best_all, best_cls = np.inf, None
        for i in range(hp["n_init"]):
            torch.manual_seed(seed * 100 + i)
            layers, prev = [], X.shape[1] + 2
            for hdim in hp["hidden"]:
                layers += [nn.Linear(prev, hdim), nn.ReLU(), nn.Dropout(hp["dropout"])]
                prev = hdim
            cls = nn.Sequential(*layers, nn.Linear(prev, self.C))
            opt = torch.optim.Adam(cls.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
            best, state, bad = np.inf, None, 0
            for _ in range(hp["max_epochs"]):
                cls.train()
                perm = torch.randperm(len(c))
                for s in range(0, len(c), hp["batch"]):
                    b = perm[s:s + hp["batch"]]
                    loss = ce(cls(F[b]), c[b])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                cls.eval()
                with torch.no_grad():
                    v = float(ce(cls(F_val), c_val))
                if v < best - 1e-5:
                    best, bad, state = v, 0, copy.deepcopy(cls.state_dict())
                else:
                    bad += 1
                    if bad >= hp["patience"]:
                        break
            cls.load_state_dict(state)
            cls.eval()
            if best < best_all:
                best_all, best_cls = best, cls
        self.cls = best_cls
        self.val_nll = self.dur.val_nll
        self.val_ce = best_all
        return self

    @torch.no_grad()
    def _label_logp(self, X, log_t):
        return torch.log_softmax(self.cls(self._feat(X, log_t)), dim=1)

    @torch.no_grad()
    def _update_terms(self, X, c, chunk=2000):
        """Bin masses of f(.|x) on the grid and P(c|x, t) at bin midpoints; returns
        (mass (n,G), pc (n,G)) with the last bin holding the tail mass."""
        X = np.asarray(X, dtype=np.float32)
        S = self.dur.survival(X, self.grid)  # (n, G)
        mass = np.concatenate([S[:, :-1] - S[:, 1:], S[:, -1:]], axis=1)
        mid = np.concatenate([np.sqrt(self.grid[:-1] * self.grid[1:]), [self.grid[-1] * 1.2]])
        lmid = torch.log(torch.as_tensor(mid, dtype=torch.float32))
        G = len(mid)
        pcs = []
        for s in range(0, len(X), chunk):
            Xb = torch.as_tensor(X[s:s + chunk])
            n = len(Xb)
            Xr = Xb[:, None, :].expand(n, G, Xb.shape[1]).reshape(n * G, -1)
            lp = self._label_logp(Xr, lmid.repeat(n))  # (n*G, C)
            cb = torch.as_tensor(c[s:s + chunk], dtype=torch.long)
            pcs.append(lp.reshape(n, G, -1).gather(2, cb[:, None, None].expand(n, G, 1)).squeeze(2).exp().numpy())
        return np.clip(mass, 0, None), np.concatenate(pcs)

    def type_weights(self, X, c=None):
        return self.dur.type_weights(X)

    def log_density(self, X, t, c=None):
        lf = self.dur.log_density(X, t)
        if c is None:
            return lf
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        lpc = self._label_logp(Xt, torch.log(torch.as_tensor(t, dtype=torch.float32)))
        lpc = lpc.gather(1, torch.as_tensor(c, dtype=torch.long)[:, None]).squeeze(1).numpy()
        mass, pc = self._update_terms(X, c)
        return lf + lpc - np.log((mass * pc).sum(1) + 1e-30)

    def survival(self, X, grid, c=None):
        if c is None:
            return self.dur.survival(X, grid)
        mass, pc = self._update_terms(X, c)
        w = mass * pc
        z = w.sum(1, keepdims=True) + 1e-30
        tail = np.concatenate([np.cumsum(w[:, ::-1], axis=1)[:, ::-1], np.zeros((len(w), 1))], axis=1) / z
        # S(q | x, c): the bins above q, plus the part of q's own bin above q (prior mass weighted by P(c | x, t) of
        # that bin, as in the normaliser); q's bin is i with grid_i <= q < grid_{i+1}, the last bin is open
        grid = np.asarray(grid, dtype=float)
        i = np.clip(np.searchsorted(self.grid, grid, side="right") - 1, 0, len(self.grid) - 1)
        S_q = self.dur.survival(X, np.maximum(grid, self.grid[0]))
        S_hi = np.concatenate([self.dur.survival(X, self.grid[1:]), np.zeros((len(w), 1))], axis=1)[:, i]
        out = tail[:, i + 1] + np.clip(S_q - S_hi, 0, None) * pc[:, i] / z
        out[:, grid < self.grid[0]] = 1.0
        return np.clip(out, 0.0, 1.0)

    def joint_logp(self, X, t, c):
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        lpc = self._label_logp(Xt, torch.log(torch.as_tensor(t, dtype=torch.float32)))
        return self.dur.log_density(X, t) + lpc.gather(1, torch.as_tensor(c, dtype=torch.long)[:, None]).squeeze(1).numpy()
