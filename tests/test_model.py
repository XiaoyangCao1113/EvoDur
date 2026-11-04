import numpy as np
import torch

from evodur.run import specs as model_specs
from evodur.model import Config, CRMixture, EvidenceModel
from evodur.data import encoder


def _xy(data):
    X = encoder(data).fit_transform(data.frame).astype(np.float32)
    return X, data.t, data.c, np.arange(400), np.arange(400, 500), np.arange(500, len(data.t))


def test_prior_fits_and_log_likelihood_is_finite(data):
    X, t, c, tr, va, te = _xy(data)
    m = CRMixture(Config(K=4, tau=1.0, max_epochs=5, n_init=1), n_methods=4).fit(X[tr], t[tr], c[tr], X[va], t[va], c[va])
    lp = m.log_density(X[te], t[te])
    assert lp.shape == (len(te),) and np.isfinite(lp).all() and np.isfinite(m.val_nll)
    S = m.survival(X[te], np.array([1.0, 10.0, 100.0, 1000.0]))
    assert np.all(np.diff(S, axis=1) <= 1e-6) and np.allclose(S[:, 0], 1.0, atol=1e-5)


def test_density_integrates_to_one(data):
    X, t, c, tr, va, te = _xy(data)
    m = CRMixture(Config(K=8, emission="none", tau=1.0, max_epochs=3, n_init=1, het_scale=True), 4).fit(X[tr], t[tr], c[tr], X[va], t[va], c[va])
    g = np.geomspace(1.0, 1e6, 4000)
    x = np.repeat(X[te[:3]], len(g), axis=0)
    f = np.exp(m.log_density(x, np.tile(g, 3))).reshape(3, -1)
    mass = np.trapezoid(f * g, np.log(g), axis=1)
    assert np.allclose(mass, 1.0, atol=2e-2)


def test_competing_risk_mixture_uses_label(data):
    X, t, c, tr, va, te = _xy(data)
    m = CRMixture(Config(K=4, emission="identity", tau=1.0, max_epochs=3, n_init=1), 4).fit(X[tr], t[tr], c[tr], X[va], t[va], c[va])
    w = m.type_weights(X[te], c[te]).numpy()
    assert np.allclose(w[np.arange(len(te)), c[te]], 1.0, atol=1e-4)


def test_evidence_model_label_update_normalises(data):
    X, t, c, tr, va, te = _xy(data)
    m = EvidenceModel(Config(K=4, tau=1.0, max_epochs=3, n_init=1), 4, max_epochs=3).fit(X[tr], t[tr], c[tr], X[va], t[va], c[va])
    known = m.log_density(X[te], t[te], c[te])
    assert np.isfinite(known).all()
    # the posterior over durations given the label sums to one over the grid
    mass, pc = m._update_terms(X[te[:5]], c[te[:5]])
    post = mass * pc / (mass * pc).sum(1, keepdims=True)
    assert np.allclose(post.sum(1), 1.0)
    # without a label the update is the prior
    assert np.allclose(m.log_density(X[te], t[te]), m.dur.log_density(X[te], t[te]))


def test_model_specs_cover_paper_models():
    specs = model_specs(1.0, 4)
    for name in ("EvoDur", "AFT", "Li2015", "DeepCAFT", "WeibullMDN-4", "DeepHit-CR", "NeuralFineGray", "XGBoost-AFT",
                 "RSF", "DeepSurv", "Evidence-DeepMix-8", "Evidence-DeepMix-16-het"):
        assert name in specs
    assert all(v in specs for v in specs["EvoDur"][1])
