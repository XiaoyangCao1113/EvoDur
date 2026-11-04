import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _jobs(*args):
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_jobs.py"), *args], capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()


def test_make_jobs_counts_and_settings():
    ring, ny = _jobs("ring4"), _jobs("ny511early")
    assert len(ring) == 135 and len(ny) == 125  # ny511early has no arrival time, so no operational runs
    assert all("--batch 256" in c and "--repeats 3" in c for c in ring if " evodur.run " in c and "--tag final_" in c)
    assert all("--batch 1024" in c for c in ny if "--tag final_" in c)
    assert len(set(ring)) == len(ring)


def test_every_job_names_a_known_model():
    from evodur.run import specs

    known = set(specs(1.0, 4))
    for line in _jobs():
        if " evodur.run " in line:
            models = line.split("--models ")[1].split(" --")[0].split(",")
            assert set(models) <= known, line


def test_expected_data_values_are_recorded():
    exp = json.loads((ROOT / "expected" / "data_checks.json").read_text())
    for name in ("ring4", "citywide", "ny511early", "sydney", "seattle", "sanfrancisco"):
        assert exp[name]["n"] > 0 and len(exp[name]["content_hash"]) == 64
