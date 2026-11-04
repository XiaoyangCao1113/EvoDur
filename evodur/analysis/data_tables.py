"""Descriptive tables for the data section: call-time variables of the Beijing data, the label classes of every
benchmark with their durations, and the call-time variables of the public data.

    python -m evodur.analysis.data_tables --out paper_out/tables

Needs all six datasets (the Beijing data are not public).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evodur.data import load  # noqa: E402
from evodur.evidence import load_with_stages  # noqa: E402
from evodur.public_data import SEATTLE_METHODS, SF_METHODS, SYDNEY_METHODS  # noqa: E402

SETS = {"ring4": "Beijing 4th Ring", "citywide": "Beijing citywide", "ny511early": "NY/NJ/CT 511",
        "sydney": "Sydney", "seattle": "Seattle", "sanfrancisco": "San Francisco"}
LABELS = {
    "ring4": ["Parties cleared, no congestion", "Police cleared, no congestion", "Parties cleared, congestion",
              "Police cleared, congestion"],
    "ny511early": ["Accident, no further detail", "Accident with property damage", "Accident with injuries",
                   "Severe (road closed, overturn, fire, investigation)"],
    "sydney": [s[0].upper() + s[1:] for s in SYDNEY_METHODS],
    "seattle": [s[0].upper() + s[1:] for s in SEATTLE_METHODS],
    "sanfrancisco": [s[0].upper() + s[1:] for s in SF_METHODS],
}
LABELS["citywide"] = LABELS["ring4"]
SEVERITY = {("9083",): "Property damage only", ("9084", "9085"): "Injury or fatality"}
# the two Beijing files code rear-end and other vehicle-vehicle collisions differently (9086/9087), so they are merged
ACC_TYPE = {("9086", "9087"): "Collision between vehicles", ("9089",): "With pedestrian or non-motor vehicle",
            ("9090",): "With fixed object", ("9092",): "Rollover", ("9093",): "Fire"}


def pct(x):
    return f"{100 * x:.1f}"


def beijing_table():
    d = {k: load(k, 1.0).frame for k in ("ring4", "citywide")}
    rows = []

    def add(group, name, desc, fn):
        rows.append((group, name, desc, *[fn(d[k]) for k in d]))

    add("Time", "Weekday", "1 if Monday to Friday", lambda f: pct(f.weekday.mean()))
    add("Time", "Peak hour", "1 if 7:00--9:00 or 17:00--19:00", lambda f: pct(f.peak.mean()))
    add("Time", "Hour, month", "Hour of day and month (sine and cosine)", lambda f: "--")
    add("Location", "Longitude, latitude", "Coordinates of the incident", lambda f: "--")
    add("Location", "Distance to center", "Distance to Tiananmen (km), mean", lambda f: f"{f.dist.mean():.1f}")
    add("Location", "Region", "Administrative region (handling unit)", lambda f: f"{f.region.nunique()} levels")
    add("Location", "Direction", "Direction of travel", lambda f: f"{f.direction.nunique()} levels")
    add("Location", "Road type", "1 if expressway, 0 if arterial", lambda f: pct((f.road_type == "1").mean()) if "road_type" in f else "--")
    add("Location", "Road number", "Ring road, expressway or named arterial", lambda f: f"{f.road_number.nunique()} levels" if "road_number" in f else "--")
    add("Vehicles", "Two vehicles or fewer", "1 if one or two vehicles are involved", lambda f: pct(f.vehicles.mean()))
    add("Vehicles", "Taxi", "1 if a taxi is involved", lambda f: pct(f.taxi.mean()))
    add("Vehicles", "Bus", "1 if a bus is involved", lambda f: pct(f.bus.mean()))
    add("Vehicles", "Truck", "1 if a truck is involved", lambda f: pct(f.truck.mean()))
    for code, lab in SEVERITY.items():
        add("Severity", lab, "", lambda f, c=code: pct(f.severity.isin(c).mean()))
    for code, lab in ACC_TYPE.items():
        add("Incident type", lab, "", lambda f, c=code: pct(f.acc_type.isin(c).mean()))
    add("Weather", "Temperature", "Mean of the call hour ($^\\circ$C), mean", lambda f: f"{f.temp_c.mean():.1f}")
    add("Weather", "Visibility", "Lowest in the call hour (km), mean", lambda f: f"{f.visibility_km.mean():.1f}")
    add("Weather", "Wind speed", "Mean of the call hour (m/s), mean", lambda f: f"{f.wind_ms.mean():.1f}")
    add("Weather", "Rain, snow, fog or haze", "1 if reported in the call hour", lambda f: f"{pct(f.rain.mean())}, {pct(f.snow.mean())}, {pct(f.fog_haze.mean())}")
    add("Weather", "Recent precipitation", "1 if precipitation in the call hour or the two hours before", lambda f: pct(f.precip_recent.mean()))
    lines = [r"\begin{tabular}{llp{6.2cm}rr}", r"\toprule",
             r"Group & Variable & Description & 4th Ring & Citywide \\", r"\midrule"]
    prev = None
    for g, n, desc, a, b in rows:
        if prev is not None and g != prev:
            lines.append(r"\addlinespace")
        lines.append(f"{g if g != prev else ''} & {n} & {desc} & {a} & {b} \\\\")
        prev = g
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def label_table():
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Dataset & Label class & Share (\%) & Median (min) & Mean (min) \\", r"\midrule"]
    for k, title in SETS.items():
        if k in ("ring4", "citywide"):
            dd = load(k, 1.0)
            t, c = dd.t, dd.c
        else:
            dd, _ = load_with_stages(k, 1.0)
            t, c = dd.t, dd.c
        if k != "ring4":
            lines.append(r"\addlinespace")
        for j, lab in enumerate(LABELS[k]):
            m = c == j
            lines.append(f"{title if j == 0 else ''} & {lab} & {pct(m.mean())} & {np.median(t[m]):.1f} & {t[m].mean():.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# call-time variables of the public data: (column, name, description); pooled columns keep their k most frequent levels
PUBLIC_VARS = {
    "ny511early": [("state", "State", "State of the event"),
                   ("county", "County", "County, 39 most frequent"),
                   ("facility", "Facility", "Named road, 60 most frequent"),
                   ("direction", "Direction", "Direction of travel, 8 most frequent"),
                   ("org", "Reporting organization", "Agency that created the event, 25 most frequent")],
    "sydney": [("primary_vehicle", "Main vehicle", "Type of the main vehicle involved"),
               ("secondary_vehicle", "Second vehicle", "Type of the second vehicle involved, 8 most frequent"),
               ("direction", "Direction", "Direction of travel"),
               ("traffic_volume", "Traffic volume", "Traffic volume as recorded"),
               ("lanes", "Lanes", "Number of lanes at the location"),
               ("area", "Area", "Statistical area (SA3)"),
               ("street", "Street", "Main street, 50 most frequent")],
    "seattle": [("precinct", "Precinct", "Police precinct of the dispatch"),
                ("sector", "Sector", "Police sector, named by a phonetic letter"),
                ("beat", "Beat", "Patrol beat"),
                ("neighborhood", "Neighborhood", "Neighborhood, 60 most frequent"),
                ("priority", "Priority", "Priority assigned by the dispatch system"),
                ("initial", "Initial call type", "How the call was first classified, 30 most frequent"),
                ("source", "Call source", "How the call was received, 4 most frequent")],
    "sanfrancisco": [("district", "Police district", "Police district of the call"),
                     ("neighborhood", "Neighborhood", "Analysis neighborhood, 41 most frequent"),
                     ("supervisor", "Supervisor district", "Supervisor district"),
                     ("priority", "Priority", "Priority level first set by the dispatcher"),
                     ("original", "Original call type", "Call type code first set by the dispatcher, 30 most frequent"),
                     ("onview", "On-view flag", "N: 911 call, Y: started by an officer, HSOC: 311 case")],
}
LEVEL_NAME = {"ny": "NY", "nj": "NJ", "ct": "CT", "nys thruway authority cad": "NYS Thruway",
              "i-90 - nys thruway": "I-90 Thruway", "i-87 - nys thruway": "I-87 Thruway",
              "nysdot - region 8": "NYSDOT Region 8", "nysdot - region 11": "NYSDOT Region 11",
              "crash urgent": "Urgent", "crash hs": "Highway safety", "crash service": "Service",
              "south of market": "South of Market", "financial district/south beach": "Financial District/South Beach",
              "bayview hunters point": "Bayview Hunters Point", "slu/cascade": "SLU/Cascade",
              "northbound": "Northbound", "m4 motorway": "M4 Motorway", "multiple vehicles": "Multiple vehicles",
              "telephone other, not 911": "Other telephone", "onview": "On view", "HSOC": "HSOC"}
CODES = ("geoid", "initial_seattle")  # long codes: only the shares are informative


def _levels(s, col):
    col = "initial_seattle" if col == "initial" and s.astype(str).str.contains(" - ").any() else col
    """Number of levels and the three largest (the pooled level is named 'other' and reported as well)."""
    share = s.astype(str).value_counts(normalize=True)
    named = [v for v in share.index if v != "other"][:3]
    tail = f"; other {pct(share['other'])}" if share.get("other", 0) >= 0.0005 else ""
    if col in CODES:
        return str(s.nunique()), f"largest level {pct(share[named[0]])}{tail}"

    def nice(v):
        if v in LEVEL_NAME:
            return LEVEL_NAME[v]
        if col == "priority":
            return v.replace("Priority ", "")
        if col == "supervisor":
            return v[:-2] if v.endswith(".0") else v
        if v.isupper() and v.isalpha() and len(v) > 2:  # names stored in capitals (precincts, sectors, districts)
            return v.title()
        return v if v.isupper() or v[:1].isdigit() else v.title()
    return str(s.nunique()), ", ".join(f"{nice(v)} {pct(share[v])}" for v in named) + tail


def public_table():
    lines = [r"\begin{tabular}{l>{\raggedright\arraybackslash}p{4.2cm}r>{\raggedright\arraybackslash}p{7.3cm}}", r"\toprule",
             r"Variable & Description & Levels & Largest levels (\% of incidents) \\", r"\midrule"]
    for k in PUBLIC_VARS:
        d, _ = load_with_stages(k, 1.0)
        f = d.frame
        if k != "ny511early":
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{SETS[k]}}} ({len(f):,} incidents)}} \\".replace(",", "{,}"))
        for col, name, desc in PUBLIC_VARS[k]:
            n, top = _levels(f[col], col)
            lines.append(f"{name} & {desc} & {n} & {top} \\\\")
    lines += [r"\addlinespace", r"\multicolumn{4}{l}{\textit{All four}} \\",
              r"Longitude, latitude & Coordinates of the call & -- & -- \\",
              r"Hour, month & Hour of day and month (sine and cosine) & -- & -- \\",
              r"Weekday & 1 if Monday to Friday & 2 & NY/NJ/CT, Sydney, Seattle, San Francisco: " + ", ".join(
                  pct(load_with_stages(k, 1.0)[0].frame.weekday.mean()) for k in PUBLIC_VARS) + r" \\",
              r"Day of week, year & Day of week; calendar year & 7; 6--17 & -- \\",
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tables")
    TAB = Path(ap.parse_args().out)
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "tab_beijing_vars.tex").write_text(beijing_table() + "\n")
    (TAB / "tab_labels.tex").write_text(label_table() + "\n")
    (TAB / "tab_public_vars.tex").write_text(public_table() + "\n")
    print("wrote tab_beijing_vars, tab_labels, tab_public_vars")


if __name__ == "__main__":
    main()
