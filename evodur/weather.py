"""Hourly weather for Beijing 2008 and matching to accident call times.

Source: NOAA ISD global-hourly, station 54511099999 (ZBAA, Beijing Capital
International Airport), file <data>/external/weather/noaa_isd_54511_2008.csv,
downloaded by `python -m evodur.download_data weather` from
https://noaa-global-hourly-pds.s3.amazonaws.com/2008/54511099999.csv

Weather is taken from the half-hourly METAR reports (REM field). ISD times are
UTC; accident times are Beijing time (UTC+8).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("EVODUR_DATA", Path(__file__).resolve().parents[1] / "data"))
WEATHER_DIR = DATA / "external" / "weather"
ISD_PATH = WEATHER_DIR / "noaa_isd_54511_2008.csv"
HOURLY_PATH = WEATHER_DIR / "beijing_weather_2008_hourly.csv"

WEATHER_COLS = ("temp_c", "visibility_km", "wind_ms", "rain", "snow", "fog_haze", "precip_recent")

_PRECIP_RAIN = re.compile(r"(?<![A-Z])[-+]?(?:VC)?(?:SH|TS|FZ)?(?:RA|DZ)\b|(?<![A-Z])[-+]?TSRA\b|(?<![A-Z])[-+]?SHRA\b")
_PRECIP_SNOW = re.compile(r"(?<![A-Z])[-+]?(?:SH|BL|DR)?(?:SN|SG|PL|GS|GR)\b|RASN|SNRA")
_FOG_HAZE = re.compile(r"\b(?:FG|BR|HZ|FU|DU|SA)\b|\bMIFG\b|\bBCFG\b")


def _parse_metar(text: str) -> dict:
    out = {"temp_c": np.nan, "visibility_km": np.nan, "wind_ms": np.nan, "rain": 0, "snow": 0, "fog_haze": 0}
    body = text.split(" NOSIG")[0].split(" BECMG")[0].split(" TEMPO")[0]
    m = re.search(r"\b(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?MPS\b", body)
    if m:
        out["wind_ms"] = float(m.group(2))
    if "CAVOK" in body:
        out["visibility_km"] = 10.0
    else:
        m = re.search(r"MPS(?: \d{3}V\d{3})? (\d{4})\b", body)
        if m:
            out["visibility_km"] = min(int(m.group(1)), 9999) / 1000.0
    m = re.search(r"\b(M?\d{2})/(M?\d{2})?\b", body)
    if m:
        out["temp_c"] = float(m.group(1).replace("M", "-"))
    # Strip the recent-weather group (RE..) so it is not counted as current weather.
    current = re.sub(r"\bRE[A-Z]+\b", "", body)
    out["rain"] = int(bool(_PRECIP_RAIN.search(current)))
    out["snow"] = int(bool(_PRECIP_SNOW.search(current)))
    out["fog_haze"] = int(bool(_FOG_HAZE.search(current)))
    return out


def build_hourly(isd_path: Path = ISD_PATH, out_path: Path = HOURLY_PATH) -> pd.DataFrame:
    raw = pd.read_csv(isd_path, low_memory=False)
    raw["metar"] = raw["REM"].astype(str).str.extract(r"METAR (ZBAA [^;]*)")[0]
    raw = raw.dropna(subset=["metar"]).copy()
    raw["time_utc"] = pd.to_datetime(raw["DATE"])
    parsed = pd.DataFrame([_parse_metar(t) for t in raw["metar"]], index=raw.index)
    obs = pd.concat([raw[["time_utc"]], parsed], axis=1)
    obs["time_local"] = obs["time_utc"] + np.timedelta64(8, "h")
    obs = obs.drop_duplicates("time_local").sort_values("time_local").set_index("time_local")

    hourly = obs.drop(columns=["time_utc"]).resample("1h").agg(
        {"temp_c": "mean", "visibility_km": "min", "wind_ms": "mean", "rain": "max", "snow": "max", "fog_haze": "max"}
    )
    hourly[["temp_c", "visibility_km", "wind_ms"]] = hourly[["temp_c", "visibility_km", "wind_ms"]].interpolate(limit=3)
    hourly[["rain", "snow", "fog_haze"]] = hourly[["rain", "snow", "fog_haze"]].fillna(0).astype(int)
    # Any rain or snow in the current or previous 2 hours (wet road surface proxy).
    precip = (hourly["rain"] | hourly["snow"]).astype(int)
    hourly["precip_recent"] = precip.rolling(3, min_periods=1).max().astype(int)
    hourly.index.name = "hour_local"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out_path, float_format="%.2f")
    return hourly


def load_hourly() -> pd.DataFrame:
    if not HOURLY_PATH.exists():
        return build_hourly()
    return pd.read_csv(HOURLY_PATH, parse_dates=["hour_local"]).set_index("hour_local")


def attach_weather(df: pd.DataFrame, time_col: str = "call_time") -> pd.DataFrame:
    """Add weather of the clock hour containing each call time (known at report time)."""
    hourly = load_hourly()
    key = pd.to_datetime(df[time_col]).dt.floor("h")
    matched = hourly.reindex(key.values)
    matched.index = df.index
    out = df.copy()
    for c in WEATHER_COLS:
        out[c] = matched[c].values
    return out


if __name__ == "__main__":
    h = build_hourly()
    print(h.describe().T)
    print(f"Wrote {HOURLY_PATH}")
