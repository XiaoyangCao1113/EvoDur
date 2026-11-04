"""Datasets of the paper: two Beijing datasets (2008, not public) and four public ones (see public_data.py).

ring4    : 4th Ring Road main lanes, <data>/raw_data.xlsx (5,388 accidents)
citywide : whole city, <data>/raw_data_extended_stata.xlsx (40,120 accidents).
           Its timestamps say 2018; the data are from 2008, so the year is corrected before weather matching.
The data directory is <repo>/data, or $EVODUR_DATA.

Both get the same report-time features and the weather of the reporting hour.
Clearance method c: TREAT 1 -> 0 (self, no congestion), 2 -> 1 (police, no
congestion), 31 -> 2 (self, congestion), 32 -> 3 (police, congestion).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("EVODUR_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("EVODUR_RESULTS", ROOT / "results"))
from evodur.weather import attach_weather  # noqa: E402

METHOD_MAP = {1: 0, 2: 1, 31: 2, 32: 3}
METHOD_NAMES = ["self/no-cong", "police/no-cong", "self/cong", "police/cong"]
NUMERIC = ["lon", "lat", "dist", "hour_sin", "hour_cos", "month_sin", "month_cos", "vehicles", "temp_c", "visibility_km", "wind_ms"]
BINARY = ["taxi", "bus", "truck", "weekday", "peak", "rain", "snow", "fog_haze", "precip_recent"]
CATEGORICAL = ["region", "severity", "acc_type", "direction"]

# The Beijing files are in Chinese: the direction values below and some column names in load() are those of the raw files.
_DIR = {"南向北": "S2N", "北向南": "N2S", "东向西": "E2W", "西向东": "W2E"}


def _direction(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    out = pd.Series("other", index=s.index)
    for k, v in _DIR.items():
        out[s.str.startswith(k)] = v
    return out


@dataclass
class Data:
    frame: pd.DataFrame  # features (raw, not yet encoded)
    t: np.ndarray  # duration, minutes
    c: np.ndarray  # clearance method 0..3
    name: str
    numeric: list
    binary: list
    categorical: list


def load(name: str, min_minutes: float = 0.0) -> Data:
    if name in ("sydney", "ny511early", "seattle", "sanfrancisco"):
        from evodur.public_data import load_public

        f, t, c, numeric, binary, categorical, _ = load_public(name, DATA, min_minutes)
        return Data(f, t, c, name, numeric, binary, categorical)
    if name == "ring4":
        raw = pd.read_excel(DATA / "raw_data.xlsx")
        call = pd.to_datetime(raw["D_CALL_TIME"])
        t = raw["totaltimem"].to_numpy(float)
        f = pd.DataFrame({
            "lon": raw["N_TELE_X"], "lat": raw["N_TELE_Y"], "dist": raw["Dis_Center ？"],
            "vehicles": raw["vehiclenumber"], "taxi": raw["taxi"], "bus": raw["bus"], "truck": raw["truck"],
            "weekday": raw["weekday"], "peak": raw["peakhour（是否在高峰期）"],
            "region": raw["N_REGION ？"].astype(str), "severity": raw["N_GRADE ？"].replace({0: 9083}).astype(str),
            "acc_type": raw["N_CHARACTER ？"].astype(str), "direction": _direction(raw["S_DIRECTION"]),
        })
        c = raw["TREAT ？"].map(METHOD_MAP).to_numpy()
        extra_cat = []
    elif name == "citywide":
        raw = pd.read_excel(DATA / "raw_data_extended_stata.xlsx", sheet_name="Sheet1")

        def ts(col):
            # Year written as 2018 but the data are from 2008. Records of
            # 29 Feb are stored as text ("2018/2/29 ...") because the date does
            # not exist in 2018, and their precomputed durations are missing.
            return pd.to_datetime(raw[col].astype(str).str.replace(r"^2018([-/])", r"2008\1", regex=True), format="mixed")

        call = ts("D_CALL_TIME")
        t_all = ((ts("D_DONE_TIME") - call).dt.total_seconds() / 60.0).to_numpy()
        ok = np.isfinite(t_all) & (t_all > 0)
        raw, call, t = raw[ok].reset_index(drop=True), call[ok].reset_index(drop=True), t_all[ok]
        f = pd.DataFrame({
            "lon": raw["N_TELE_X"], "lat": raw["N_TELE_Y"], "dist": raw["Dis_Center"],
            "vehicles": raw["vehiclenumber"], "taxi": raw["taxi"], "bus": raw["bus"], "truck": raw["truck"],
            "weekday": raw["weekday"], "peak": raw["peakhour"],
            "region": raw["N_REGION"].astype(str), "severity": raw["N_GRADE"].replace({0: 9083}).astype(str),
            "acc_type": raw["N_CHARACTER"].astype(str), "direction": _direction(raw["S_DIRECTION"].fillna("")),
            "road_type": raw["roadtype"].astype(str), "road_number": raw["roadnumber"].astype(str),
        })
        c = raw["TREAT"].map(METHOD_MAP).to_numpy()
        extra_cat = ["road_type", "road_number"]
    else:
        raise KeyError(name)
    f["hour_sin"] = np.sin(2 * np.pi * call.dt.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * call.dt.hour / 24)
    f["month_sin"] = np.sin(2 * np.pi * call.dt.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * call.dt.month / 12)
    f["call_time"] = call
    f = attach_weather(f, time_col="call_time")
    for col in ("temp_c", "visibility_km", "wind_ms"):
        f[col] = f[col].fillna(f[col].median())
    for col in ("rain", "snow", "fog_haze", "precip_recent"):
        f[col] = f[col].fillna(0).astype(int)
    keep = (t > min_minutes) & ~np.isnan(c.astype(float))
    f, t, c = f[keep].reset_index(drop=True), t[keep], c[keep].astype(int)
    return Data(f, t, c, name, NUMERIC, BINARY, CATEGORICAL + extra_cat)


def encoder(data: Data, min_frequency: int = 20):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer([
        ("num", StandardScaler(), data.numeric),
        ("bin", "passthrough", data.binary),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency, sparse_output=False), data.categorical),
    ])
