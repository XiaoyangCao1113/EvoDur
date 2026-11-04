"""Public benchmark datasets (download with `python -m evodur.download_data`).

sydney       : Sydney road traffic incidents 2017-2022 (Anna, Bisht & Chand, 2023, Data in Brief 51:109769;
               Mendeley Data doi:10.17632/cgnx2cs665, version 5, CC BY 4.0). Crashes only. Post-hoc label c = attending groups.
ny511early   : 511 NY events (data.ny.gov, ah74-pg4w), accidents 2010-2015, NY/NJ/CT. Post-hoc label c = event type.
seattle      : Seattle Police Department call data (data.seattle.gov, 33kz-ixgy), final call type
               'TRAFFIC - MV COLLISION INVESTIGATION'.
sanfrancisco : San Francisco law enforcement dispatched calls for service (data.sf.gov, 2zdj-bwza), vehicle accidents.

Durations above 24 h are dropped. The Socrata portals keep adding records, so the loaders (and the download queries) are
pinned to the call-time window of the snapshot used in the paper (WINDOWS below).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Call-time window (inclusive) of the snapshot used in the paper: first and last call time in the downloaded file.
# ny511early is selected by calendar year (2010-2015) in load_ny511.
WINDOWS = {
    "seattle": ("2009-06-02 03:54:42", "2026-10-23 23:33:47"),
    "sanfrancisco": ("2014-12-31 23:18:13", "2026-10-26 23:33:34"),
}
# Records that the portals added after the snapshot (27 Oct 2026) but dated inside the window: late-arriving calls.
# (column, ids) per dataset; the download queries exclude them too.
EXCLUDE = {
    "seattle": ("cad_event_number", ["2026000316760"]),
    "sanfrancisco": ("cad_number", ["262993680", "262992897", "262993594", "262993485"]),
}


def _pin(raw, col, name):
    """Keep the rows whose call time lies in the pinned window and drop the excluded late records
    (a no-op on the snapshot of the paper)."""
    lo, hi = (pd.Timestamp(v) for v in WINDOWS[name])
    when = pd.to_datetime(raw[col])
    key, ids = EXCLUDE[name]
    return raw[(when >= lo) & (when <= hi) & ~raw[key].astype(str).isin(ids)].reset_index(drop=True)


SYDNEY_METHODS = ["emergency only", "emergency + road agency", "tow truck", "road agency only", "unknown/other"]
NY_METHODS = ["accident", "property damage", "injury", "severe (closure/overturn/fire/investigation)"]


def _cyclic(f, when):
    f["hour_sin"] = np.sin(2 * np.pi * (when.dt.hour + when.dt.minute / 60) / 24)
    f["hour_cos"] = np.cos(2 * np.pi * (when.dt.hour + when.dt.minute / 60) / 24)
    f["month_sin"] = np.sin(2 * np.pi * when.dt.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * when.dt.month / 12)
    f["weekday"] = (when.dt.dayofweek < 5).astype(int)
    f["dow"] = when.dt.dayofweek.astype(str)
    f["year"] = when.dt.year.astype(float)
    return f


def _top(s, k, other="other"):
    s = s.fillna(other).astype(str).str.strip().str.lower()
    keep = s.value_counts().index[:k]
    return s.where(s.isin(keep), other)


def _sydney_group(g: str) -> int:
    g = str(g).lower()
    if g == "unknown":
        return 4
    emerg = "emergency" in g
    agency = any(k in g for k in ("transport for nsw", "rms", "motorway crew"))
    if "tow truck" in g:
        return 2
    if emerg and agency:
        return 1
    if emerg:
        return 0
    if agency:
        return 3
    return 4


def load_sydney(root, category="Crash"):
    raw = pd.read_excel(root / "public" / "sydney_rti" / "extracted" / "Excel Files" / "Final Road Traffic Incidents.xlsx")
    raw = raw[raw["Main_Category"] == category].reset_index(drop=True)
    start = pd.to_datetime(raw["Start_Time"], format="%d-%m-%Y %H:%M")
    t = raw["Duration_in_Minutes"].to_numpy(float)
    lanes = raw["Actual_Number_of_Lanes "].astype(str).replace({"99": "unknown"})
    f = pd.DataFrame({
        "lon": raw["Longitude"], "lat": raw["Latitude"],
        "primary_vehicle": _top(raw["Primary_Vehicle"], 10), "secondary_vehicle": _top(raw["Secondary_Vehicle"], 10),
        "direction": _top(raw["Direction"], 10), "traffic_volume": _top(raw["Traffic_Volume"], 5),
        "lanes": lanes, "area": raw["SA3_NAME21"].astype(str), "street": _top(raw["Main_Street"], 50),
    })
    f = _cyclic(f, start)
    f["call_time"] = start
    c = raw["Attending_Groups"].map(_sydney_group).to_numpy()
    numeric = ["lon", "lat", "hour_sin", "hour_cos", "month_sin", "month_cos", "year"]
    binary = ["weekday"]
    categorical = ["primary_vehicle", "secondary_vehicle", "direction", "traffic_volume", "lanes", "area", "street", "dow"]
    return f, t, c, numeric, binary, categorical, SYDNEY_METHODS


def _ny_type(s: str) -> int:
    s = str(s).lower()
    if any(k in s for k in ("road closed", "overturn", "fire", "investigation", "closed")):
        return 3
    if "injur" in s or "pedestrian" in s:
        return 2
    if "property damage" in s:
        return 1
    return 0


NY_ORDER_FILE = Path(__file__).resolve().parents[1] / "expected" / "ny511early_row_order.npz"
NY_KEY = ["event_type", "organization_name", "facility_name", "direction", "city", "county", "state", "create_time", "close_time",
          "event_description", "responding_organization_id", "latitude", "longitude", "georeference"]


def _ny_row_order(raw):
    """Positions that put the rows of `raw` (reset index) in the order of the paper's snapshot.

    The portal returns the 511 NY events in no stable order, and the cross-validation folds are drawn by row position, so
    the rows are first sorted canonically (all columns, ties are identical rows) and then permuted with the rank of each
    snapshot row in that order, stored in expected/ny511early_row_order.npz (integers only, no data). Without the file the
    canonical order is used and the folds, hence the numbers, differ slightly from the paper."""
    canon = raw.sort_values([c for c in NY_KEY if c in raw], kind="stable", na_position="last").index.to_numpy()
    if NY_ORDER_FILE.exists():
        rank = np.load(NY_ORDER_FILE)["rank"]
        if len(rank) == len(canon):
            return canon[rank]
        print(f"WARNING ny511early: {len(canon)} rows, but {NY_ORDER_FILE.name} is for {len(rank)}; using the canonical row "
              "order, so the folds differ from the paper", file=sys.stderr)
    else:
        print(f"WARNING {NY_ORDER_FILE} not found; using the canonical row order for ny511early", file=sys.stderr)
    return canon


def load_ny511(root, years=(2016, 2019)):
    raw = pd.read_csv(root / "public" / "ny511_events" / "ny511_accident_events.csv")
    raw = raw[~raw["event_type"].str.lower().str.contains("cleared", na=False)]
    start = pd.to_datetime(raw["create_time"], errors="coerce")
    end = pd.to_datetime(raw["close_time"], errors="coerce")
    t = ((end - start).dt.total_seconds() / 60).to_numpy()
    ok = start.dt.year.between(*years).to_numpy() & np.isfinite(t) & (t > 0)
    raw, start, t = raw[ok].reset_index(drop=True), start[ok].reset_index(drop=True), t[ok]
    if years == (2010, 2015):  # the benchmark of the paper
        o = _ny_row_order(raw)
        raw, start, t = raw.iloc[o].reset_index(drop=True), start.iloc[o].reset_index(drop=True), t[o]
    org = raw["organization_name"].fillna("").str.lower()
    org = org.str.replace(r"new york state department of transportation", "nysdot", regex=True)
    org = org.str.replace(r"new jersey turnpike authority|nj turnpike auth\.?", "nj turnpike", regex=True)
    f = pd.DataFrame({
        "lon": raw["longitude"], "lat": raw["latitude"], "state": _top(raw["state"], 3),
        "county": _top(raw["county"], 40), "facility": _top(raw["facility_name"], 60),
        "direction": _top(raw["direction"], 8), "org": _top(org, 25),
    })
    f = _cyclic(f, start)
    f["call_time"] = start
    c = raw["event_type"].map(_ny_type).to_numpy()
    numeric = ["lon", "lat", "hour_sin", "hour_cos", "month_sin", "month_cos", "year"]
    binary = ["weekday"]
    categorical = ["state", "county", "facility", "direction", "org", "dow"]
    return f, t, c, numeric, binary, categorical, NY_METHODS


SEATTLE_METHODS = ["report written (no arrest)", "other or follow-up report", "assistance or no action",
                   "unable to locate or canceled", "citation or arrest"]
SF_METHODS = ["report taken", "handled", "no disposition, unable to locate or gone on arrival",
              "canceled or no merit", "other (citation, arrest, advised, ...)"]

_SEATTLE_CLASS = {
    "REPORT WRITTEN (NO ARREST)": 0,
    "OTHER REPORT MADE": 1, "FOLLOW-UP REPORT MADE": 1, "OLD REPORT WITH EVIDENCE": 1, "STREET CHECK WRITTEN": 1,
    "UNABLE TO LOCATE INCIDENT OR COMPLAINANT": 3, "RESPONDING UNIT(S) CANCELLED BY RADIO": 3, "CANCELLED EVENT": 3,
    "DUPLICATE EVENT": 3, "BAD LOCATION": 3, "EXTRA UNIT": 3,
    "CITATION": 4, "PHYSICAL ARREST MADE": 4, "OLD CRIMINAL CITATION": 4,
}  # every other clearance description (assistance rendered, no police action, radio broadcast, warning, ...) -> 2
_SF_CLASS = {"REP": 0, "HAN": 1, "ND": 2, "UTL": 2, "GOA": 2, "CAN": 3, "NOM": 3}  # CIT, ARR, ADV, ABA, ... and missing -> 4


def load_seattle(root):
    """Seattle Police Department call data (data.seattle.gov, 33kz-ixgy), final call type
    'TRAFFIC - MV COLLISION INVESTIGATION' (public domain), one row per event.
    Duration = call queued to the last SPD unit back in service; `_arrival` = call to first SPD unit at scene
    (evidence, never a feature)."""
    raw = pd.read_csv(root / "public" / "seattle_cad" / "seattle_mv_collisions.csv", low_memory=False)
    raw = _pin(raw, "cad_event_original_time_queued", "seattle")
    start = pd.to_datetime(raw["cad_event_original_time_queued"])
    end = pd.to_datetime(raw["last_spd_call_sign_in_service_time"])
    arr = pd.to_datetime(raw["first_spd_call_sign_at_scene_time"])
    t = ((end - start).dt.total_seconds() / 60).to_numpy()
    lon, lat = raw["dispatch_longitude"].to_numpy(float), raw["dispatch_latitude"].to_numpy(float)
    swap = (lon > 0) & (lat < 0)  # a few hundred rows have longitude and latitude exchanged
    lon, lat = np.where(swap, lat, lon), np.where(swap, lon, lat)
    bad = ~((lon > -123) & (lon < -121.5) & (lat > 47) & (lat < 48))  # placeholders (-1, 0) -> missing
    lon, lat = np.where(bad, np.nan, lon), np.where(bad, np.nan, lat)
    f = pd.DataFrame({
        "lon": lon, "lat": lat, "precinct": raw["dispatch_precinct"].astype(str), "sector": raw["dispatch_sector"].astype(str),
        "beat": _top(raw["dispatch_beat"], 90), "neighborhood": _top(raw["dispatch_neighborhood"], 60),
        "priority": raw["priority"].astype(str), "initial": _top(raw["initial_call_type"], 30),
        "source": _top(raw["call_type"], 4),
    })
    f = _cyclic(f, start)
    f["call_time"] = start
    f["_arrival"] = ((arr - start).dt.total_seconds() / 60).to_numpy()
    c = raw["cad_event_clearance_description"].map(lambda d: _SEATTLE_CLASS.get(str(d).strip().upper(), 2)).to_numpy()
    numeric = ["lon", "lat", "hour_sin", "hour_cos", "month_sin", "month_cos", "year"]
    binary = ["weekday"]
    categorical = ["precinct", "sector", "beat", "neighborhood", "priority", "initial", "source", "dow"]
    return f, t, c, numeric, binary, categorical, SEATTLE_METHODS


def load_sanfrancisco(root):
    """San Francisco law enforcement dispatched calls for service, closed (data.sf.gov, 2zdj-bwza), final call type
    in {INJURY VEH ACCIDENT, VEH ACCIDENT, H&R VEH ACCIDENT, H&R INJURY ACCIDENT} (Open Data Commons PDDL).
    Duration = received to closed; `_arrival` = received to on scene (evidence, never a feature)."""
    raw = pd.read_csv(root / "public" / "sf_cad" / "sf_traffic_accidents.csv", low_memory=False)
    raw = _pin(raw, "received_datetime", "sanfrancisco")
    start = pd.to_datetime(raw["received_datetime"])
    end = pd.to_datetime(raw["close_datetime"])
    arr = pd.to_datetime(raw["onscene_datetime"])
    t = ((end - start).dt.total_seconds() / 60).to_numpy()
    xy = raw["intersection_point"].astype(str).str.extract(r"POINT \((-?[\d.]+) (-?[\d.]+)\)").astype(float)
    f = pd.DataFrame({
        "lon": xy[0].to_numpy(), "lat": xy[1].to_numpy(), "district": raw["police_district"].astype(str),
        "neighborhood": _top(raw["analysis_neighborhood"], 45), "supervisor": raw["supervisor_district"].astype(str),
        "priority": raw["priority_original"].astype(str), "original": _top(raw["call_type_original"], 30),
        "onview": raw["onview_flag"].astype(str), "sensitive": raw["sensitive_call"].astype(str),
    })
    f = _cyclic(f, start)
    f["call_time"] = start
    f["_arrival"] = ((arr - start).dt.total_seconds() / 60).to_numpy()
    c = raw["disposition"].map(lambda d: _SF_CLASS.get(str(d).strip().upper(), 4)).to_numpy()
    numeric = ["lon", "lat", "hour_sin", "hour_cos", "month_sin", "month_cos", "year"]
    binary = ["weekday"]
    categorical = ["district", "neighborhood", "supervisor", "priority", "original", "onview", "sensitive", "dow"]
    return f, t, c, numeric, binary, categorical, SF_METHODS


PUBLIC = ("sydney", "ny511early", "seattle", "sanfrancisco")


def load_public(name, root, min_minutes):
    """`root` is the data directory (default <repo>/data, override with EVODUR_DATA)."""
    loaders = {"sydney": load_sydney, "ny511early": lambda r: load_ny511(r, years=(2010, 2015)),
               "seattle": load_seattle, "sanfrancisco": load_sanfrancisco}
    f, t, c, numeric, binary, categorical, names = loaders[name](root)
    keep = np.isfinite(t) & (t > min_minutes) & (t <= 1440) & f[["lon", "lat"]].notna().all(1).to_numpy()
    return f[keep].reset_index(drop=True), t[keep], c[keep].astype(int), numeric, binary, categorical, names
