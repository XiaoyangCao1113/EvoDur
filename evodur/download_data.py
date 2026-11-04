"""Download the four public datasets (and the NOAA weather file of the Beijing data) into <data>/.

    python -m evodur.download_data                    # sydney ny511 seattle sanfrancisco
    python -m evodur.download_data seattle weather    # a subset
    python -m evodur.download_data --test             # a few rows from each portal, to check URLs and filters

<data> is ./data in the repository, or $EVODUR_DATA. The files are written where evodur.public_data expects them:

    public/sydney_rti/sydney_rti_v5.zip, public/sydney_rti/extracted/...        Mendeley Data cgnx2cs665, version 5
    public/ny511_events/ny511_accident_events.csv                               data.ny.gov ah74-pg4w
    public/seattle_cad/seattle_mv_collisions.csv                                data.seattle.gov 33kz-ixgy
    public/sf_cad/sf_traffic_accidents.csv                                      data.sf.gov 2zdj-bwza
    external/weather/noaa_isd_54511_2008.csv                                    NOAA ISD, Beijing Capital Airport, 2008

The three Socrata portals keep adding records. Each query is therefore pinned to the call-time window of the snapshot used in
the paper (WINDOWS in evodur.public_data; NY 511: calls from 2010 to 2015, the years of the `ny511early` benchmark), and the
few records that the portals added after the snapshot but dated inside the window are excluded by id (EXCLUDE). Seattle and
San Francisco are written sorted by event number, as in the snapshot; the NY 511 rows are fetched ordered by the portal's row
id (the portal has no stable default order) and put into the snapshot's order by the loader (public_data._ny_row_order).
Run `python -m evodur.check_data` afterwards to compare row counts and checksums with the paper's snapshot.
"""
from __future__ import annotations

import argparse
import io
import os
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from evodur.public_data import EXCLUDE, WINDOWS

DATA = Path(os.environ.get("EVODUR_DATA", Path(__file__).resolve().parents[1] / "data"))
PAGE = 50000
SYDNEY_URL = "https://data.mendeley.com/public-api/zip/cgnx2cs665/download/5"
WEATHER_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/2008/54511099999.csv"
NY = "https://data.ny.gov/resource/ah74-pg4w.csv"
SEATTLE = "https://data.seattle.gov/resource/33kz-ixgy.csv"
SF = "https://data.sf.gov/resource/2zdj-bwza.csv"


def _iso(ts):
    return ts.replace(" ", "T")


# ---------------------------------------------------------------------------------------------- queries (SoQL)
NY_WHERE = ("lower(event_type) like '%accident%' AND create_time >= '2010-01-01T00:00:00' "
            "AND create_time < '2016-01-01T00:00:00'")


def _not_in(name):
    col, ids = EXCLUDE[name]
    return f"{col} not in ({','.join(repr(i) for i in ids)})"


SEATTLE_WHERE = ("final_call_type='TRAFFIC - MV COLLISION INVESTIGATION' AND "
                 f"cad_event_original_time_queued >= '{_iso(WINDOWS['seattle'][0])}' AND "
                 f"cad_event_original_time_queued <= '{_iso(WINDOWS['seattle'][1])}' AND {_not_in('seattle')}")
SEATTLE_SELECT = ("cad_event_number,cad_event_original_time_queued,first_spd_call_sign_at_scene_time,cad_event_arrived_time,"
                  "last_spd_call_sign_in_service_time,cad_event_clearance_description,call_type,priority,initial_call_type,"
                  "dispatch_precinct,dispatch_sector,dispatch_beat,dispatch_neighborhood,dispatch_longitude,dispatch_latitude")
SF_WHERE = ("call_type_final_desc in ('INJURY VEH ACCIDENT','VEH ACCIDENT','H&R VEH ACCIDENT','H&R INJURY ACCIDENT') AND "
            f"received_datetime >= '{_iso(WINDOWS['sanfrancisco'][0])}' AND received_datetime <= '{_iso(WINDOWS['sanfrancisco'][1])}' "
            f"AND {_not_in('sanfrancisco')}")
SF_SELECT = ("cad_number,received_datetime,onscene_datetime,close_datetime,call_type_original,call_type_original_desc,"
             "call_type_final_desc,priority_original,disposition,onview_flag,sensitive_call,police_district,"
             "analysis_neighborhood,supervisor_district,intersection_point")


def socrata(base, where, select=None, order=None, limit=None):
    """All rows of a SoQL query as strings (paged, with retries)."""
    parts, off = [], 0
    while True:
        params = {"$where": where, "$limit": PAGE if limit is None else limit, "$offset": off}
        if select:
            params["$select"] = select
        if order:
            params["$order"] = order
        url = f"{base}?{urllib.parse.urlencode(params)}"
        for _ in range(4):
            try:
                df = pd.read_csv(io.BytesIO(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()),
                                 low_memory=False, dtype=str)
                break
            except Exception as e:  # network hiccup: wait and retry
                print("retry:", e, flush=True)
                time.sleep(10)
        else:
            raise RuntimeError(f"query failed: {url}")
        print(f"  {base.split('/')[-1]} offset {off}: {len(df)} rows", flush=True)
        if df.empty or limit is not None:
            if not df.empty:
                parts.append(df)
            break
        parts.append(df)
        off += PAGE
        time.sleep(2)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------------------------- downloads
UA = {"User-Agent": "Mozilla/5.0 (evodur data download; python-urllib)"}  # Mendeley refuses the default Python agent


def _fetch(url, path):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(path, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)


def _probe(url):
    """Fetch the first 100 bytes of a large file (to test that the URL works)."""
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-99"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return f"HTTP {r.status}, {r.headers.get('Content-Type')}, first bytes {r.read(4)!r} ({url})"


def get_sydney(test=False):
    d = DATA / "public" / "sydney_rti"
    d.mkdir(parents=True, exist_ok=True)
    if test:
        print("sydney:", _probe(SYDNEY_URL))
        return
    zp = d / "sydney_rti_v5.zip"
    _fetch(SYDNEY_URL, zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(d / "extracted")
    print("sydney:", zp)


def get_ny511(test=False):
    df = socrata(NY, NY_WHERE, order=":id", limit=3 if test else None)
    if test:
        print(df[["event_type", "create_time", "close_time"]].to_string())
        return
    out = DATA / "public" / "ny511_events"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ny511_accident_events.csv", index=False)  # in :id order; the loader fixes the row order
    print("ny511:", len(df), "rows")


def get_seattle(test=False):
    df = socrata(SEATTLE, SEATTLE_WHERE, SEATTLE_SELECT, "cad_event_number,call_sign_dispatch_id", limit=3 if test else None)
    if test:
        print(df[["cad_event_number", "cad_event_original_time_queued"]].to_string())
        return
    # one row per dispatched unit in the source; the event-level fields are repeated, keep the first row of each event
    df = df.drop_duplicates("cad_event_number")
    df = df.sort_values("cad_event_number", kind="stable").reset_index(drop=True)
    out = DATA / "public" / "seattle_cad"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "seattle_mv_collisions.csv", index=False)
    print("seattle:", len(df), "events")


def get_sf(test=False):
    df = socrata(SF, SF_WHERE, SF_SELECT, "cad_number", limit=3 if test else None)
    if test:
        print(df[["cad_number", "received_datetime"]].to_string())
        return
    df = df.sort_values("cad_number", kind="stable").reset_index(drop=True)
    out = DATA / "public" / "sf_cad"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sf_traffic_accidents.csv", index=False)
    print("sanfrancisco:", len(df), "rows")


def get_weather(test=False):
    if test:
        print("weather:", _probe(WEATHER_URL))
        return
    out = DATA / "external" / "weather"
    out.mkdir(parents=True, exist_ok=True)
    _fetch(WEATHER_URL, out / "noaa_isd_54511_2008.csv")
    print("weather:", out / "noaa_isd_54511_2008.csv")


SOURCES = {"sydney": get_sydney, "ny511": get_ny511, "seattle": get_seattle, "sanfrancisco": get_sf, "weather": get_weather}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", default=["sydney", "ny511", "seattle", "sanfrancisco"], help=f"subset of {list(SOURCES)}")
    ap.add_argument("--test", action="store_true", help="fetch three rows (or a HEAD request) per source and print them")
    a = ap.parse_args()
    for n in a.names:
        if n not in SOURCES:
            raise SystemExit(f"unknown source {n}; choose from {list(SOURCES)}")
        print(f"== {n}")
        SOURCES[n](test=a.test)


if __name__ == "__main__":
    main()
