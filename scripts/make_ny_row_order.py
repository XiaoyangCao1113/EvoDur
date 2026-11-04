#!/usr/bin/env python
"""(Maintainers) Record the row order of the NY 511 snapshot used in the paper as expected/ny511early_row_order.npz.

    python scripts/make_ny_row_order.py data/public/ny511_events/ny511_accident_events.csv

The portal returns the events in no stable order, and the folds depend on row position. This stores, for each row of the
ny511early benchmark in the order of the snapshot, its rank in the canonical sort of evodur.public_data (integers only).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evodur import public_data as P

raw = pd.read_csv(sys.argv[1])
raw = raw[~raw["event_type"].str.lower().str.contains("cleared", na=False)]
start = pd.to_datetime(raw["create_time"], errors="coerce")
end = pd.to_datetime(raw["close_time"], errors="coerce")
t = ((end - start).dt.total_seconds() / 60).to_numpy()
ok = start.dt.year.between(2010, 2015).to_numpy() & np.isfinite(t) & (t > 0)
raw = raw[ok].reset_index(drop=True)  # rows of ny511early in the order of the snapshot
canon = raw.sort_values([c for c in P.NY_KEY if c in raw], kind="stable", na_position="last").index.to_numpy()
rank = np.empty(len(canon), np.uint32)
rank[canon] = np.arange(len(canon), dtype=np.uint32)
P.NY_ORDER_FILE.parent.mkdir(exist_ok=True)
np.savez_compressed(P.NY_ORDER_FILE, rank=rank)
print("wrote", P.NY_ORDER_FILE, len(rank), "rows")
