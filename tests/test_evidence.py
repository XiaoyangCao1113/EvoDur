import numpy as np
import torch

from evodur.data import encoder
from evodur.evidence import (T_MAX, ArrivalEvidence, MethodEvidence, MethodGivenArrival, fit_evidence, fit_mixaft,
                             posterior_logp)

TAU = 1.0


def _setup(data):
    X = encoder(data).fit_transform(data.frame).astype(np.float32)
    t, c = data.t, data.c
    tr, va, te = np.arange(400), np.arange(400, 500), np.arange(500, len(t))
    lo = np.full(len(t), TAU, np.float32)
    torch.manual_seed(0)
    prior = fit_mixaft(X[tr], t[tr], lo[tr], X[va], t[va], lo[va], K=4, n_init=1)
    return X, t, c, tr, va, te, lo, prior


def test_no_evidence_is_the_prior(data):
    X, t, c, tr, va, te, lo, prior = _setup(data)
    post = posterior_logp(prior, [], X[te], t[te], lo[te], n_grid=2000)
    with torch.no_grad():
        direct = prior.loglik(torch.as_tensor(X[te]), torch.as_tensor(t[te], dtype=torch.float32), torch.as_tensor(lo[te])).numpy()
    assert np.isfinite(post).all() and np.allclose(post, direct, atol=5e-2)


def test_posterior_normalises(data):
    X, t, c, tr, va, te, lo, prior = _setup(data)
    ev = fit_evidence(lambda d: MethodEvidence(d, 4), X[tr], c[tr], t[tr], X[va], c[va], t[va])
    g = np.geomspace(TAU * 1.0005, T_MAX, 3000).astype(np.float32)
    for k in te[:2]:
        Xg = np.repeat(X[[k]], len(g), axis=0)
        lp = posterior_logp(prior, [(lambda Xq, e, tq: ev.logp(Xq, torch.as_tensor(e).long(), tq), np.full(len(g), c[k]))], Xg, g, np.full(len(g), TAU, np.float32), n_grid=1200)
        mass = np.trapezoid(np.exp(lp) * g, np.log(g))
        assert abs(mass - 1.0) < 5e-2


def test_label_probabilities_sum_to_one(data):
    X, t, c, tr, va, te, lo, prior = _setup(data)
    ev = fit_evidence(lambda d: MethodEvidence(d, 4), X[tr], c[tr], t[tr], X[va], c[va], t[va])
    Xt, tt = torch.as_tensor(X[te]), torch.as_tensor(t[te], dtype=torch.float32)
    with torch.no_grad():
        p = sum(ev.logp(Xt, torch.full((len(te),), k, dtype=torch.long), tt).exp() for k in range(4))
    assert torch.allclose(p, torch.ones_like(p), atol=1e-5)


def test_arrival_density_integrates_to_one(data):
    X, t, c, tr, va, te, lo, prior = _setup(data)
    r = data.frame["_arrival"].to_numpy(float)
    ok = (r > 0) & (r < t)
    a_tr, a_va = tr[ok[tr]], va[ok[va]]
    ev = fit_evidence(ArrivalEvidence, X[a_tr], r[a_tr], t[a_tr], X[a_va], r[a_va], t[a_va])
    tt = 40.0
    rg = np.linspace(1e-3, tt - 1e-3, 20000).astype(np.float32)
    with torch.no_grad():
        d = ev.logp(torch.as_tensor(np.repeat(X[[te[0]]], len(rg), 0)), torch.as_tensor(rg), torch.full((len(rg),), tt)).exp().numpy()
    assert abs(np.trapezoid(d, rg) - 1.0) < 2e-2


def test_updates_and_summaries(data):
    X, t, c, tr, va, te, lo, prior = _setup(data)
    r = data.frame["_arrival"].to_numpy(float)
    ok = (r > 0) & (r < t)
    a_tr, a_va, a_te = tr[ok[tr]], va[ok[va]], te[ok[te]]
    ev_a = fit_evidence(ArrivalEvidence, X[a_tr], r[a_tr], t[a_tr], X[a_va], r[a_va], t[a_va])
    cr = lambda idx: np.column_stack([c[idx], r[idx]]).astype(np.float32)
    ev_cr = fit_evidence(lambda d: MethodGivenArrival(d, 4), X[a_tr], cr(a_tr), t[a_tr], X[a_va], cr(a_va), t[a_va])
    low = np.maximum(r[a_te], TAU).astype(np.float32)
    evs = [(lambda Xq, e, tq: ev_a.logp(Xq, torch.as_tensor(e), tq), r[a_te].astype(np.float32)),
           (lambda Xq, e, tq: ev_cr.logp(Xq, torch.as_tensor(e), tq), cr(a_te))]
    lp, s = posterior_logp(prior, evs, X[a_te], t[a_te], low, summary=True)
    assert np.isfinite(lp).all()
    assert np.all(s["q10"] <= s["q50"]) and np.all(s["q50"] <= s["q90"]) and np.all(s["q10"] >= low * 0.999)
    assert np.all((s["p60"] >= 0) & (s["p60"] <= 1 + 1e-5))
    # an incident still open at 60 min has P(T > 60) = 1
    _, s60 = posterior_logp(prior, [], X[te[:5]], t[te[:5]], np.full(5, 60.0, np.float32), summary=True)
    assert np.allclose(s60["p60"], 1.0, atol=1e-3)
