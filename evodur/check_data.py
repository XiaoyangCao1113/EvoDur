"""Check that the data on disk are the data of the paper.

    python -m evodur.check_data                     # all datasets found in <data>/ (default ./data or $EVODUR_DATA)
    python -m evodur.check_data seattle sydney
    python -m evodur.check_data --write expected/data_checks.json     # (maintainers) record the values of the current data

For each dataset the script loads it exactly as the experiments do (`evodur.data.load(name, 1.0)`), prints the number of
incidents and two SHA-256 checksums of the processed records, and compares them with expected/data_checks.json (recorded
from the snapshot used in the paper):

  content_hash  the records sorted by key: call time, longitude, latitude, label class and duration (minutes, 3 decimals).
                A match means the same incidents with the same durations and labels.
  order_hash    the same lines in the order of the loaded table. The cross-validation folds are drawn by row position,
                so only when this matches too do the folds, and hence the numbers, agree with the paper.

A mismatch in the number of incidents or in content_hash means the portal changed records inside the pinned window
(late corrections of closing times, removed calls); the results then differ slightly from the paper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evodur.data import DATA, load

DATASETS = ["ring4", "citywide", "ny511early", "sydney", "seattle", "sanfrancisco"]
EXPECTED = Path(__file__).resolve().parents[1] / "expected" / "data_checks.json"
FILES = {"ring4": "raw_data.xlsx", "citywide": "raw_data_extended_stata.xlsx",
         "ny511early": "public/ny511_events/ny511_accident_events.csv",
         "sydney": "public/sydney_rti/extracted/Excel Files/Final Road Traffic Incidents.xlsx",
         "seattle": "public/seattle_cad/seattle_mv_collisions.csv", "sanfrancisco": "public/sf_cad/sf_traffic_accidents.csv"}


def summarize(name: str) -> dict:
    d = load(name, 1.0)
    f = d.frame
    when = f["call_time"].dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy()
    keys = [f"{w}|{lo:.5f}|{la:.5f}|{c}|{t:.3f}" for w, lo, la, c, t in zip(when, f["lon"], f["lat"], d.c, d.t)]
    sha = lambda lines: hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return {"n": int(len(d.t)), "content_hash": sha(sorted(keys)), "order_hash": sha(keys),
            "first_call": str(f["call_time"].min()), "last_call": str(f["call_time"].max()),
            "median_minutes": round(float(np.median(d.t)), 3), "class_counts": np.bincount(d.c).tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", default=DATASETS)
    ap.add_argument("--write", default=None, help="write the computed values to this json file instead of comparing")
    a = ap.parse_args()
    exp = json.loads(EXPECTED.read_text()) if EXPECTED.exists() else {}
    got, bad = {}, 0
    for rel, want in exp.get("_files", {}).items():  # raw files that are fixed (the Sydney release, the weather file)
        f = DATA / rel
        if f.exists() and not a.write:
            have = hashlib.sha256(f.read_bytes()).hexdigest()
            print(f"file {rel}: {'OK' if have == want else 'DIFFERENT (sha256 ' + have[:12] + ')'}")
            bad += have != want
    for name in a.names:
        if not (DATA / FILES[name]).exists():
            print(f"{name:13s} not found ({DATA / FILES[name]})")
            continue
        s = got[name] = summarize(name)
        line = f"{name:13s} n={s['n']:>7,d}  content {s['content_hash'][:12]}  order {s['order_hash'][:12]}"
        e = exp.get(name)
        if a.write or e is None:
            print(line + ("" if a.write else "  (no expected values recorded)"))
            continue
        ok_content = (s["n"], s["content_hash"]) == (e["n"], e["content_hash"])
        ok_order = s["order_hash"] == e["order_hash"]
        status = "OK" if ok_content and ok_order else ("same incidents, different row order (folds will differ)" if ok_content else "MISMATCH")
        bad += status != "OK"
        print(f"{line}  expected n={e['n']:,d}  -> {status}")
    if a.write:
        Path(a.write).parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(Path(a.write).read_text()) if Path(a.write).exists() else {}
        old.update(got)
        Path(a.write).write_text(json.dumps(old, indent=1, sort_keys=True) + "\n")
        print("wrote", a.write)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
