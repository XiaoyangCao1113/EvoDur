"""Time one Bayes update of EvoDur on one CPU core (the cost quoted in the paper's appendix on computational cost).

    python scripts/time_update.py [--d 60] [--n 2000]

The cost of an update does not depend on the fitted weights, so the modules are built with the paper's architecture
(prior with K = 8 components, label, arrival and label-given-arrival modules) and random weights. Each update evaluates
every module on a grid of 600 durations and returns the log density of the observed duration and the predictive
summaries (quantiles and exceedance probabilities).
"""
import argparse
import time

import numpy as np
import torch

from evodur.evidence import ArrivalEvidence, MethodEvidence, MethodGivenArrival, MixAFT, posterior_logp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=60, help="number of encoded call-time features")
    ap.add_argument("--n", type=int, default=2000, help="incidents per timing run")
    ap.add_argument("--classes", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    prior, lab, arr, lab_arr = (MixAFT(a.d, K=8), MethodEvidence(a.d, C=a.classes), ArrivalEvidence(a.d),
                                MethodGivenArrival(a.d, C=a.classes))
    for m in (prior, lab, arr, lab_arr):
        m.eval()
    X = rng.normal(size=(a.n, a.d)).astype(np.float32)
    t = rng.lognormal(3.5, 0.8, a.n).astype(np.float32) + 2
    r = (t * rng.uniform(0.05, 0.6, a.n)).astype(np.float32)
    c = rng.integers(0, a.classes, a.n)
    T = torch.as_tensor
    lab_f = lambda X_, v, t_: lab.logp(X_, T(v).long(), t_)
    arr_f = lambda X_, v, t_: arr.logp(X_, T(v, dtype=torch.float32), t_)
    lab_arr_f = lambda X_, v, t_: lab_arr.logp(X_, T(v, dtype=torch.float32), t_)
    states = {
        "call": ([], np.ones(a.n, np.float32)),
        "label": ([(lab_f, c)], np.ones(a.n, np.float32)),
        "arrival and label": ([(arr_f, r), (lab_arr_f, np.stack([c, r], 1).astype(np.float32))], r),
    }
    for name, (ev, lower) in states.items():
        posterior_logp(prior, [(f, v[:200]) for f, v in ev], X[:200], t[:200], lower[:200], summary=True)  # warm-up
        t0 = time.perf_counter()
        posterior_logp(prior, ev, X, t, lower, summary=True)
        batch_ms = 1000 * (time.perf_counter() - t0) / a.n
        t0 = time.perf_counter()
        for i in range(200):
            posterior_logp(prior, [(f, v[i:i + 1]) for f, v in ev], X[i:i + 1], t[i:i + 1], lower[i:i + 1], summary=True)
        single_ms = 1000 * (time.perf_counter() - t0) / 200
        print(f"{name:18s} {single_ms:5.2f} ms for one incident alone, {batch_ms:5.2f} ms per incident in batches")


if __name__ == "__main__":
    main()
