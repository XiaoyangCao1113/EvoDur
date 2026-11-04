import numpy as np
import pandas as pd
import torch

from evodur.analysis.interpret_tables import summarize
from evodur.data import encoder
from evodur.evidence import MethodEvidence, fit_evidence, fit_mixaft, posterior_logp
from evodur.interpret import PriorOnPermuted, apply_level, check_levels, forecast, on_original

TAU = 1.0


def _fit(data):
    X = encoder(data).fit_transform(data.frame).astype(np.float32)
    t, c = data.t, data.c
    tr, va, te = np.arange(400), np.arange(400, 500), np.arange(500, len(t))
    lo = np.full(len(t), TAU, np.float32)
    torch.manual_seed(0)
    prior = fit_mixaft(X[tr], t[tr], lo[tr], X[va], t[va], lo[va], K=4, n_init=1)
    ev = fit_evidence(lambda d: MethodEvidence(d, 4), X[tr], c[tr], t[tr], X[va], c[va], t[va])
    return X, t, c, te, lo, prior, ev


def test_prior_only_permutation_reduces_to_the_update_when_nothing_is_permuted(data):
    X, t, c, te, lo, prior, ev = _fit(data)
    e = ((lambda Xq, a, tq: ev.logp(Xq, torch.as_tensor(a, dtype=torch.long), tq)), c[te])
    plain = posterior_logp(prior, [e], X[te], t[te], lo[te])
    d = X.shape[1]
    split = posterior_logp(PriorOnPermuted(prior, d), [on_original(e, d)], np.hstack([X[te], X[te]]), t[te], lo[te])
    assert np.allclose(plain, split, atol=1e-4)


def test_reporting_a_longer_class_raises_the_forecast(data):
    X, t, c, te, lo, prior, ev = _fit(data)
    n = len(te)
    s = {k: forecast(prior, [((lambda Xq, a, tq: ev.logp(Xq, torch.as_tensor(a, dtype=torch.long), tq)), np.full(n, k, np.int64))],
                     X[te], t[te], lo[te]) for k in (0, 3)}
    assert s[3]["median"].mean() > s[0]["median"].mean() and s[3]["p60"].mean() > s[0]["p60"].mean()


def test_counterfactual_levels_and_summary(data):
    f = data.frame.iloc[:5]
    g = apply_level(f, {"zone": "b", "weekday": 1})
    assert (g.zone == "b").all() and (g.weekday == 1).all() and f.zone.iloc[0] == data.frame.zone.iloc[0]
    assert check_levels(("x", {"zone": "a"}, {"zone": "b"}), data.frame) is None
    assert "zone='zz'" in check_levels(("x", {"zone": "a"}, {"zone": "zz"}), data.frame)
    rows = [dict(data="ring4", fold=k, section="importance", variable="Location", contrast=f"perm{p}", model="call",
                 metric="logp_drop", value=float(k + p), n=1) for k in (0, 1) for p in (0, 1)]
    S = summarize(pd.DataFrame(rows))
    r = S.iloc[0]
    assert len(S) == 1 and r["mean"] == 1.0 and r["min"] == 0.5 and r["max"] == 1.5 and r["n_folds"] == 2
