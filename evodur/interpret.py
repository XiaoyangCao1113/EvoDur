"""Interpretation of EvoDur: how call-time variables and each piece of response information move the forecast.

For one dataset and given folds (same splits, encoder, seeds and networks as evidence.py: the call-time prior with
8 components and shared shapes, the label module, and, where an arrival time exists, the arrival module, the
arrival-recorded classifier and the label-given-arrival classifier) it computes, on the held-out incidents:

  A. call_effect     Counterfactual marginal effects of 4-6 interpretable call-time variables. The variable is set to
                     level a for every test incident, then to level b (raw frame, then encoded); we report the mean
                     predicted median, P(T > 30/60/120 min), their differences b - a and the ratio of mean medians,
                     for EvoDur's prior and for the linear log-logistic AFT fitted on the same encoded features
                     (its time ratio exp(-(h_b - h_a)) is exact and also reported).
  B. label_effect    Counterfactual label: report class c for every test incident and update the prior; mean median and
                     P(T > cut) after the update, minus the call-time value; also restricted to incidents whose true
                     class is c (the factual effect).
     arrival_effect  Counterfactual arrival at the 25th and 75th percentile of the training arrival times (all test
                     incidents; the update truncates at r), minus the call-time values.
     importance      Permutation importance of groups of call-time variables: the raw columns of a group are permuted
                     jointly (before the encoder), the drop in mean test log score is measured at the call and in the
                     state "arrival + label" ("label" where the data have no arrival time). The same permutation is used
                     in the states and all modules see the permuted features. A second version scrambles the features
                     for the prior only (the evidence modules keep the true ones; model "..., prior only"): permuting
                     them everywhere also pits the permuted features against the observed arrival and label, which the
                     modules p(r | t, x), P(c | t, x) penalize on top of the information lost. The evaluation incidents
                     are those that reach the response state (recorded arrival), so all states use the same incidents.

The test set is subsampled to --max-test incidents (fixed seed) to bound the run time of the grid integrations.
Output: interpret_{data}{tag}_f{k}.csv in $EVODUR_RESULTS, long format
(data, fold, section, variable, contrast, model, metric, value, n).

    python -m evodur.interpret --data ring4 --tau 1 --folds 0 --batch 256 --threads 4
    python -m evodur.interpret --data sydney --folds 0 --max-epochs 3 --max-test 2000   # smoke test
"""
from __future__ import annotations

import argparse
import functools
import time
import warnings

import numpy as np
import pandas as pd
import torch

from evodur import evidence as ev
from evodur.data import encoder
from evodur.evidence import (CUTS, OUT, ArrivalEvidence, MethodEvidence, MethodGivenArrival, fit_evidence, fit_mixaft,
                               load_with_stages, posterior_logp)
from evodur.public_data import SF_METHODS, SYDNEY_METHODS, SEATTLE_METHODS

LABEL_NAMES = {
    "ring4": ["Parties cleared, no congestion", "Police cleared, no congestion", "Parties cleared, congestion",
              "Police cleared, congestion"],
    "ny511early": ["Accident, no further detail", "Accident with property damage", "Accident with injuries",
                   "Severe (road closed, overturn, fire, investigation)"],
    "sydney": [s[0].upper() + s[1:] for s in SYDNEY_METHODS],
    "seattle": [s[0].upper() + s[1:] for s in SEATTLE_METHODS],
    "sanfrancisco": [s[0].upper() + s[1:] for s in SF_METHODS],
}
LABEL_NAMES["citywide"] = LABEL_NAMES["ring4"]

# ----------------------------------------------------------------------------- variables of the analysis


def _hour(h):
    return {"hour_sin": float(np.sin(2 * np.pi * h / 24)), "hour_cos": float(np.cos(2 * np.pi * h / 24))}


def _month(m):
    return {"month_sin": float(np.sin(2 * np.pi * m / 12)), "month_cos": float(np.cos(2 * np.pi * m / 12))}


def _mode(s, exclude=()):
    s = s[~s.isin(exclude)]
    return s.value_counts().index[0]


def call_specs(name, frame):
    """[(plain-English label, level a (reference), level b)]; a level is a dict {raw column: value}.
    The reference is the first level of the label, 'b vs a'. Levels are taken from the raw frame of the dataset."""
    time_ = [("Weekday (Wed) vs weekend (Sun)", {"weekday": 0, "dow": "6"}, {"weekday": 1, "dow": "2"}),
             ("Evening rush (17:00) vs night (03:00)", _hour(3), _hour(17))]
    if name in ("ring4", "citywide"):
        truck = int(_mode(frame["truck"], exclude=(0,)))  # ring4 codes the trucks 0 / 1 / 2, citywide 0 / 1
        coll = _mode(frame["acc_type"], exclude=[v for v in frame["acc_type"].unique() if v not in ("9086", "9087")])
        return [("Injury or fatality vs property damage only", {"severity": "9083"}, {"severity": "9084"}),
                ("Truck involved vs no truck", {"truck": 0}, {"truck": truck}),
                ("Bus involved vs no bus", {"bus": 0}, {"bus": 1}),
                ("Rollover vs collision between vehicles", {"acc_type": coll}, {"acc_type": "9092"}),
                ("Rain vs no rain", {"rain": 0}, {"rain": 1}),
                ("Peak hour vs off-peak", {"peak": 0}, {"peak": 1})]
    if name == "sydney":
        return [("Truck vs car (first vehicle)", {"primary_vehicle": "car"}, {"primary_vehicle": "truck"}),
                ("Pedestrian vs car (second party)", {"secondary_vehicle": "car"}, {"secondary_vehicle": "pedestrian"}),
                ("Heavy vs moderate traffic volume", {"traffic_volume": "moderate"}, {"traffic_volume": "heavy"}),
                ("Three lanes vs one lane", {"lanes": "1"}, {"lanes": "3"})] + time_
    if name == "seattle":
        return [("High (1) vs low (5) priority", {"priority": "5"}, {"priority": "1"}),
                ("911 call vs officer on view", {"source": "onview"}, {"source": "911"}),
                ("Call type: injuries vs non-injury, blocking",
                 {"initial": "obs - mvc - non injury, blocking"}, {"initial": "obs - mvc - with injuries (includes hit and run)"})] + time_
    if name == "sanfrancisco":
        return [("Priority A vs priority C", {"priority": "C"}, {"priority": "A"}),
                ("Started by an officer (on view) vs called in", {"onview": "N"}, {"onview": "Y"}),
                ("Call type: injury accident vs non-injury accident", {"original": "518"}, {"original": "519"})] + time_
    if name == "ny511early":  # no incident-specific variable at the call: only time and season are interpretable
        return time_ + [("Winter (Jan) vs summer (Jul)", _month(7), _month(1))]
    raise KeyError(name)


# groups of call-time variables for the permutation importance (raw column names of data.py / public_data.py)
GROUPS = {
    "Time of day": ["hour_sin", "hour_cos", "peak"],
    "Day and season": ["weekday", "dow", "month_sin", "month_cos", "year"],
    "Location": ["lon", "lat", "dist", "region", "area", "county", "state", "precinct", "sector", "beat", "neighborhood",
                 "district", "supervisor"],
    "Road and traffic": ["road_type", "road_number", "direction", "facility", "street", "lanes", "traffic_volume"],
    "Vehicles involved": ["vehicles", "taxi", "bus", "truck", "primary_vehicle", "secondary_vehicle"],
    "Severity and incident type": ["severity", "acc_type", "initial", "original"],
    "Priority, call source, agency": ["priority", "source", "onview", "sensitive", "org"],
    "Weather": ["temp_c", "visibility_km", "wind_ms", "rain", "snow", "fog_haze", "precip_recent"],
}


def groups_for(data):
    used = set(data.numeric) | set(data.binary) | set(data.categorical)
    out = {g: [c for c in cols if c in used] for g, cols in GROUPS.items()}
    out = {g: cols for g, cols in out.items() if cols}
    left = used - {c for cols in out.values() for c in cols}
    assert not left, f"features without a group: {sorted(left)}"
    return out


# ----------------------------------------------------------------------------- helpers


class Rows:
    def __init__(self, data, fold):
        self.data, self.fold, self.rows = data, fold, []

    def add(self, section, variable, contrast, model, metric, value, n=np.nan):
        self.rows.append({"data": self.data, "fold": self.fold, "section": section, "variable": variable,
                          "contrast": contrast, "model": model, "metric": metric, "value": float(value), "n": n})

    def frame(self):
        return pd.DataFrame(self.rows)


def forecast(model, evs, X, t, low):
    """Predictive summaries (median, P(T > c)) of every test incident, per-incident arrays."""
    _, s = posterior_logp(model, evs, X, t, low, summary=True)
    out = {"median": np.asarray(s["q50"], float)}
    out.update({f"p{c}": np.clip(np.asarray(s[f"p{c}"], float), 0.0, 1.0) for c in CUTS if c in (30, 60, 120)})
    return out


def _summary_rows(rows, section, variable, contrast, model, ref, new, n, suffix=""):
    """Mean median and mean P(T > c) after (`new`) and before (`ref`), and their difference / ratio."""
    ma, mb = ref["median"].mean(), new["median"].mean()
    rows.add(section, variable, contrast, model, f"median_ref{suffix}", ma, n)
    rows.add(section, variable, contrast, model, f"median_new{suffix}", mb, n)
    rows.add(section, variable, contrast, model, f"median_diff{suffix}", mb - ma, n)
    rows.add(section, variable, contrast, model, f"median_ratio{suffix}", mb / ma, n)
    for c in (30, 60, 120):
        pa, pb = ref[f"p{c}"].mean(), new[f"p{c}"].mean()
        rows.add(section, variable, contrast, model, f"p{c}_ref{suffix}", pa, n)
        rows.add(section, variable, contrast, model, f"p{c}_new{suffix}", pb, n)
        rows.add(section, variable, contrast, model, f"p{c}_diff{suffix}", pb - pa, n)


def _subset(s, m):
    return {k: v[m] for k, v in s.items()}


def apply_level(frame, level):
    f = frame.copy()
    for col, val in level.items():
        f[col] = val
    return f


def check_levels(spec, train_frame, min_count=20):
    """A categorical / binary level must occur at least `min_count` times in the training part, otherwise the encoder
    pools it into one 'infrequent' level (and a level absent from training is unseen)."""
    for lev in (spec[1], spec[2]):
        for col, val in lev.items():
            if col in ("hour_sin", "hour_cos", "month_sin", "month_cos"):
                continue
            n = int((train_frame[col] == val).sum())
            if n < min_count:
                return f"{col}={val!r} occurs {n} times in training"
    return None


class PriorOnPermuted:
    """Prior that reads the second half of a concatenated feature matrix [original | permuted]; the evidence modules read
    the first half (see on_original). Used to scramble the call-time variables for the prior only."""

    def __init__(self, prior, d):
        self.prior, self.d = prior, d

    def log_f(self, X, t):
        return self.prior.log_f(X[:, self.d:], t)


def on_original(e, d):
    fn, vals = e
    return (lambda Xq, a, tq: fn(Xq[:, :d], a, tq)), vals


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ring4")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-test", type=int, default=10000, help="test incidents used for the counterfactuals and the importance (random subsample)")
    ap.add_argument("--n-perm", type=int, default=3, help="permutations per group of variables")
    ap.add_argument("--max-epochs", type=int, default=400, help="training epochs of all networks (smoke tests only)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.max_epochs != 400:
        ev.train = functools.partial(ev.train, max_epochs=args.max_epochs)  # fit_mixaft / fit_evidence resolve train() here
    from evodur.splits import iter_splits

    data, r = load_with_stages(args.data, args.tau)
    t, c = data.t, data.c
    C = int(c.max()) + 1
    has_r = r is not None
    ok_r = ((r > 0) & (r < t)) if has_r else np.zeros(len(t), bool)
    print(f"{args.data}: n={len(t)}, C={C}, arrival available for {ok_r.mean():.3f}", flush=True)
    names = LABEL_NAMES[args.data]
    assert len(names) == C
    B = args.batch
    tau = np.full(len(t), args.tau, dtype=np.float32)
    groups = groups_for(data)

    for _, fold, tr_, va, te in iter_splits(data, "cv", 5, 1, 1.0, args.folds):
        t0 = time.perf_counter()
        R = Rows(args.data, fold)
        path = OUT / f"interpret_{args.data}{args.tag}_f{fold}.csv"
        enc = encoder(data).fit(data.frame.iloc[tr_])
        Xtr, Xva = (enc.transform(data.frame.iloc[i]).astype(np.float32) for i in (tr_, va))

        # ---- the networks, fitted exactly as in evidence.py
        prior = fit_mixaft(Xtr, t[tr_], tau[tr_], Xva, t[va], tau[va], seed=fold, batch=B)
        lin = fit_mixaft(Xtr, t[tr_], tau[tr_], Xva, t[va], tau[va], K=1, seed=fold, batch=B, linear=True)
        ev_met = fit_evidence(lambda d: MethodEvidence(d, C), Xtr, c[tr_], t[tr_], Xva, c[va], t[va], seed=fold, batch=B)
        E_met = lambda cc: ((lambda Xq, a, tq: ev_met.logp(Xq, torch.as_tensor(a, dtype=torch.long), tq)), cc)
        occ = lambda n: []
        if has_r:
            a_tr, a_va = tr_[ok_r[tr_]], va[ok_r[va]]
            ev_arr = fit_evidence(ArrivalEvidence, Xtr[ok_r[tr_]], r[a_tr], t[a_tr], Xva[ok_r[va]], r[a_va], t[a_va], seed=fold, batch=B)
            E_arr = lambda rr: ((lambda Xq, a, tq: ev_arr.logp(Xq, torch.as_tensor(a, dtype=torch.float32), tq)), rr)
            if not ok_r.all():  # that an arrival was recorded at all is evidence too
                ev_occ = fit_evidence(lambda d: MethodEvidence(d, 2), Xtr, ok_r[tr_].astype(np.int64), t[tr_], Xva,
                                      ok_r[va].astype(np.int64), t[va], seed=fold, batch=B)
                occ = lambda n: [((lambda Xq, a, tq: ev_occ.logp(Xq, torch.as_tensor(a, dtype=torch.long), tq)), np.ones(n, np.int64))]
            cr = lambda idx: np.column_stack([c[idx], r[idx]]).astype(np.float32)
            ev_met_r = fit_evidence(lambda d: MethodGivenArrival(d, C), Xtr[ok_r[tr_]], cr(a_tr), t[a_tr], Xva[ok_r[va]], cr(a_va),
                                    t[a_va], seed=fold, batch=B)
            E_metr = lambda idx: ((lambda Xq, a, tq: ev_met_r.logp(Xq, torch.as_tensor(a, dtype=torch.float32), tq)), cr(idx))
        print(f"fold {fold}: fitted in {time.perf_counter() - t0:.0f}s", flush=True)

        # ---- test incidents (subsample)
        sub = np.sort(np.random.default_rng(fold).choice(te, min(args.max_test, len(te)), replace=False))
        n = len(sub)
        frame = data.frame.iloc[sub].reset_index(drop=True)
        X0 = enc.transform(frame).astype(np.float32)
        t_s, c_s, low0 = t[sub], c[sub], tau[sub]
        R.add("meta", "test incidents", "", "", "n", n, n)
        R.add("meta", "test incidents", "", "", "share_arrival_recorded", ok_r[sub].mean() if has_r else 0.0, n)

        # ---- baseline at the call
        call = {"evodur": forecast(prior, [], X0, t_s, low0), "aft": forecast(lin, [], X0, t_s, low0)}
        for m, s in call.items():
            R.add("baseline", "call", "", m, "mean_median", s["median"].mean(), n)
            for cut in (30, 60, 120):
                R.add("baseline", "call", "", m, f"mean_p{cut}", s[f"p{cut}"].mean(), n)

        # ---- A. call-time marginal effects
        t1 = time.perf_counter()
        for spec in call_specs(args.data, data.frame):
            label, la, lb = spec
            why = check_levels(spec, data.frame.iloc[tr_])
            if why:
                print(f"  skip '{label}': {why}", flush=True)
                continue
            Xa, Xb = (enc.transform(apply_level(frame, lv)).astype(np.float32) for lv in (la, lb))
            contrast = " vs ".join(", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in lv.items()) for lv in (lb, la))
            for m, model in (("evodur", prior), ("aft", lin)):
                sa, sb = forecast(model, [], Xa, t_s, low0), forecast(model, [], Xb, t_s, low0)
                _summary_rows(R, "call_effect", label, contrast, m, sa, sb, n)
                if m == "aft":  # exact time ratio of the linear model: median is proportional to exp(-h(x))
                    with torch.no_grad():
                        dh = (model.lin(torch.as_tensor(Xb))[:, 0] - model.lin(torch.as_tensor(Xa))[:, 0]).numpy()
                    R.add("call_effect", label, contrast, m, "time_ratio_exact", float(np.exp(-dh).mean()), n)
        R.frame().to_csv(path, index=False)
        print(f"fold {fold}: call-time effects {time.perf_counter() - t1:.0f}s", flush=True)

        # ---- B(i). counterfactual label, and the factual effect on the incidents of that class
        t1 = time.perf_counter()
        ref = call["evodur"]
        for k in range(C):
            s = forecast(prior, [E_met(np.full(n, k, np.int64))], X0, t_s, low0)
            _summary_rows(R, "label_effect", names[k], "report this class for every test incident", "evodur", ref, s, n)
            m_k = c_s == k
            if m_k.sum() > 0:
                _summary_rows(R, "label_effect", names[k], "incidents of this class (factual)", "evodur", _subset(ref, m_k),
                              _subset(s, m_k), int(m_k.sum()), suffix="")
            R.add("label_effect", names[k], "", "", "share_of_test", m_k.mean(), n)
        R.frame().to_csv(path, index=False)
        print(f"fold {fold}: label effects {time.perf_counter() - t1:.0f}s", flush=True)

        # ---- B(ii). counterfactual arrival time
        t1 = time.perf_counter()
        if has_r:
            for q in (25, 75):
                rq = float(np.percentile(r[tr_][ok_r[tr_]], q))
                s = forecast(prior, [E_arr(np.full(n, rq, np.float32))] + occ(n), X0, t_s, np.full(n, max(rq, args.tau), np.float32))
                _summary_rows(R, "arrival_effect", f"Police arrive after {q}th percentile of response times", f"r={rq:.1f} min",
                              "evodur", ref, s, n)
                R.add("arrival_effect", f"Police arrive after {q}th percentile of response times", f"r={rq:.1f} min", "evodur",
                      "r_minutes", rq, n)
            R.frame().to_csv(path, index=False)
            print(f"fold {fold}: arrival effects {time.perf_counter() - t1:.0f}s", flush=True)

        # ---- B(iii). permutation importance of call-time groups, at the call and after the response information
        t1 = time.perf_counter()
        keep = ok_r[sub] if has_r else np.ones(n, bool)  # incidents that reach the response state
        idx = sub[keep]
        fr, t_e, tau_e = frame[keep].reset_index(drop=True), t_s[keep], low0[keep]
        ne = len(idx)
        state = "arrival+label" if has_r else "label"
        if has_r:
            low_r = np.maximum(r[idx], args.tau).astype(np.float32)
            evs = lambda: [E_arr(r[idx]), E_metr(idx)] + occ(ne)
        else:
            low_r = tau_e
            evs = lambda: [E_met(c[idx])]

        Xe = enc.transform(fr).astype(np.float32)
        d_in = Xe.shape[1]
        prior_perm = PriorOnPermuted(prior, d_in)

        def scores(X):
            """Mean log score at the call, after the response information with every module reading the permuted
            features, and after it with only the prior reading them (the evidence modules see the true features)."""
            return (float(posterior_logp(prior, [], X, t_e, tau_e).mean()),
                    float(posterior_logp(prior, evs(), X, t_e, low_r).mean()),
                    float(posterior_logp(prior_perm, [on_original(e, d_in) for e in evs()], np.hstack([Xe, X]), t_e, low_r).mean()))

        base = scores(Xe)
        state_p = state + ", prior only"
        R.add("importance_base", "all", "", "call", "logp", base[0], ne)
        R.add("importance_base", "all", "", state, "logp", base[1], ne)
        R.add("importance_base", "all", "", state_p, "logp", base[2], ne)
        rng = np.random.default_rng(1000 + fold)
        for g, cols in groups.items():
            for k in range(args.n_perm):
                perm = rng.permutation(ne)  # the same permutation in both states
                fp = fr.copy()
                for col in cols:
                    fp[col] = fr[col].to_numpy()[perm]
                sc = scores(enc.transform(fp).astype(np.float32))
                R.add("importance", g, f"perm{k}", "call", "logp_drop", base[0] - sc[0], ne)
                R.add("importance", g, f"perm{k}", state, "logp_drop", base[1] - sc[1], ne)
                R.add("importance", g, f"perm{k}", state_p, "logp_drop", base[2] - sc[2], ne)
            d = [x for x in R.rows if x["section"] == "importance" and x["variable"] == g]
            print(f"  importance {g:30s} call {np.mean([x['value'] for x in d if x['model'] == 'call']):+.4f}   "
                  f"{state} {np.mean([x['value'] for x in d if x['model'] == state]):+.4f}   "
                  f"prior only {np.mean([x['value'] for x in d if x['model'] == state_p]):+.4f}", flush=True)
        R.frame().to_csv(path, index=False)
        print(f"fold {fold}: importance {time.perf_counter() - t1:.0f}s; total {time.perf_counter() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
