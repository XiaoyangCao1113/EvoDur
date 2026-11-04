import numpy as np
import pandas as pd

from evodur import public_data
from evodur.public_data import load_public
from evodur.splits import iter_splits


def _write_ny(root, n=6):
    start = pd.date_range("2012-05-01 08:00", periods=n, freq="5h")
    p = root / "public" / "ny511_events"
    p.mkdir(parents=True)
    pd.DataFrame({"event_type": ["accident", "accident with property damage", "injury accident", "road closed due to accident",
                                 "accident cleared", "accident"],
                  "create_time": start, "close_time": start + pd.Timedelta("45min"), "longitude": -74.0, "latitude": 40.7,
                  "state": "NY", "county": "Kings", "facility_name": "I-278", "direction": "east",
                  "organization_name": "New York State Department of Transportation"}
                 ).to_csv(p / "ny511_accident_events.csv", index=False)


def test_ny511_column_contract(tmp_path, capsys):
    _write_ny(tmp_path)
    f, t, c, num, binary, cat, names = load_public("ny511early", tmp_path, 1.0)
    assert len(t) == 5 and np.allclose(t, 45) and sorted(c) == [0, 0, 1, 2, 3]  # the "cleared" record is dropped
    for col in num + binary + cat:
        assert col in f and f[col].notna().all(), col
    assert len(names) == 4


def test_ny511_row_order_is_independent_of_file_order(tmp_path):
    """The portal returns the rows in no stable order; the loader must give the same table for any file order."""
    _write_ny(tmp_path)
    a = load_public("ny511early", tmp_path, 1.0)
    path = tmp_path / "public" / "ny511_events" / "ny511_accident_events.csv"
    pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
    b = load_public("ny511early", tmp_path, 1.0)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


def test_pinned_window_and_excluded_ids():
    col = "when"
    raw = pd.DataFrame({col: ["2009-06-02 03:54:41", "2009-06-02 03:54:42", "2020-01-01 00:00:00", "2026-10-23 23:33:47",
                              "2026-10-23 23:33:48", "2026-10-24 00:00:00"],
                        "cad_event_number": ["1", "2", "2026000316760", "4", "5", "6"]})
    out = public_data._pin(raw, col, "seattle")
    assert list(out["cad_event_number"]) == ["2", "4"]  # before the window, after it, and the excluded late id are dropped


def test_splits_are_disjoint(data):
    seen = []
    for rep, fold, tr, va, te in iter_splits(data, "cv", 5, 1):
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
        seen.append(te)
    assert np.array_equal(np.sort(np.concatenate(seen)), np.arange(len(data.t)))
    (_, _, tr, va, te), = iter_splits(data, "temporal")
    order = data.frame["call_time"].to_numpy()
    assert order[tr].max() <= order[va].min() and order[va].max() <= order[te].min()
    (_, _, tr_small, _, _), = iter_splits(data, "cv", 5, 1, 0.5, "0")
    (_, _, tr_full, _, _), = iter_splits(data, "cv", 5, 1, 1.0, "0")
    assert set(tr_small) < set(tr_full)
