import sys

import pandas as pd

import evodur.run as run


def test_run_main_on_synthetic(monkeypatch, tmp_path, data):
    monkeypatch.setattr(run, "load", lambda name, min_minutes: data)
    monkeypatch.setattr(run, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run", "--data", "syn", "--tau", "1", "--only-folds", "0", "--threads", "1",
                                      "--landmarks", "30", "--models", "AFT,Evidence-AFT,DeepHit-CR", "--tag", "smoke"])
    run.main()
    df = pd.read_csv(tmp_path / "folds_syn_tau1_smoke.csv")
    assert set(df.model) == {"AFT", "Evidence-AFT", "DeepHit-CR"}
    assert df[["report_logp", "known_logp", "lm30_logp"]].notna().all().all()
    assert (tmp_path / "incidents_syn_tau1_smoke.npz").exists()
