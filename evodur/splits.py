"""Train / validation / test splits shared by run.py and evidence.py.

cv       : stratified K-fold by the label;
           15% of the training part is held out for early stopping.
temporal : the most recent 20% of incidents (by call time) are the test set;
           the latest 15% of the remaining incidents form the validation set.
           `repeats` then only changes the training seeds.
train_frac < 1 subsamples the training part (validation and test unchanged),
for learning curves.
"""
from __future__ import annotations

import numpy as np


def iter_splits(data, mode="cv", folds=5, repeats=1, train_frac=1.0, only_folds=""):
    from sklearn.model_selection import StratifiedKFold, train_test_split

    t, c = data.t, data.c
    keep = set(only_folds.split(",")) if only_folds else None
    for rep in range(repeats):
        if mode == "cv":
            skf = StratifiedKFold(folds, shuffle=True, random_state=100 + rep)
            parts = []
            for fold, (tr, te) in enumerate(skf.split(np.zeros(len(t)), c)):
                tr_, va = train_test_split(tr, test_size=0.15, random_state=rep * 10 + fold, stratify=c[tr])
                parts.append((fold, tr_, va, te))
        elif mode == "temporal":
            order = np.argsort(data.frame["call_time"].to_numpy(), kind="stable")
            n_te = int(round(0.2 * len(t)))
            te, pool = order[-n_te:], order[:-n_te]
            n_va = int(round(0.15 * len(pool)))
            parts = [(0, pool[:-n_va], pool[-n_va:], te)]
        else:
            raise KeyError(mode)
        for fold, tr_, va, te in parts:
            if keep is not None and str(fold) not in keep:
                continue
            if train_frac < 1.0:
                rng = np.random.default_rng(rep * 10 + fold + 7)
                tr_ = np.sort(rng.choice(tr_, max(int(round(train_frac * len(tr_))), 50), replace=False))
            yield rep, fold, tr_, va, te
