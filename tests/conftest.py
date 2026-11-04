import numpy as np
import pandas as pd
import pytest
import torch

from evodur.data import Data


def synthetic_data(n=600, seed=0, arrival=True):
    """Small synthetic benchmark with the structure of the dispatch data: call-time features, a duration t that
    depends on them, a label c that depends on t, and an arrival time r < t for most incidents."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    zone = rng.choice(["a", "b", "c"], size=n)
    t = np.exp(3.2 + 0.4 * x1 + 0.3 * (zone == "b") + 0.45 * rng.logistic(size=n)).clip(1.2, 1400)
    c = np.digitize(np.log(t) + 0.5 * rng.normal(size=n), [2.8, 3.3, 3.8])
    frame = pd.DataFrame({"x1": x1, "x2": x2, "weekday": rng.integers(0, 2, n), "zone": zone,
                          "call_time": pd.Timestamp("2020-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 3e5, n)), "min")})
    if arrival:
        r = t * rng.beta(2, 5, size=n)
        r[rng.random(n) < 0.1] = np.nan  # some incidents have no recorded arrival
        frame["_arrival"] = r
    return Data(frame, t, c.astype(int), "synthetic", ["x1", "x2"], ["weekday"], ["zone"])


@pytest.fixture
def data():
    torch.manual_seed(0)
    return synthetic_data()
