"""Teaser figure (Fig. 1), drawn as an SVG by hand and printed to PDF with headless Chrome.

One incident as the response unfolds: the scene on the road (a crash closes one lane, traffic queues in the other,
a tow truck clears it), the events above it with the information each brings, and the predicted duration below
(sideways densities, 80% band and median) narrowing at each piece of information; durations already passed are
ruled out, and the bounds of the minor / intermediate / major incident classes are marked on the right.

    python -m evodur.figures.teaser_svg <out>/figs                        # writes <out>/figs/fig_teaser.svg
    node evodur/figures/render_svg.js <out>/figs/fig_teaser.svg <out>/figs/fig_teaser   # SVG -> PDF + PNG (Node.js + puppeteer)

Densities are illustrative shapes, not fitted output. Icons: Material Design Icons (Pictogrammers, Apache 2.0),
see evodur/figures/icons/. Font: STIX (as in the other figures), embedded from matplotlib's copy.
"""
import re
from pathlib import Path

import matplotlib
import numpy as np

HERE = Path(__file__).resolve().parent
FONTS = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTE, LINE = "#1d1d1b", "#5c5b57", "#8a8984", "#dcdbd5"
T_TRUE, R_ARR, S_LAB, S_NOW = 70.0, 9.0, 20.0, 45.0

# ---------------------------------------------------------------- illustrative model (same as before)
g = np.geomspace(2, 600, 1500)
lg = np.log(g)


def loglogistic(mu, s):
    z = (lg - mu) / s
    return np.exp(-z) / (s * (1 + np.exp(-z)) ** 2)


prior = 0.55 * loglogistic(np.log(22), 0.55) + 0.45 * loglogistic(np.log(55), 0.6)
lik_r = 1 / (1 + np.exp(-(lg - np.log(24)) / 0.3))
lik_c = loglogistic(np.log(85), 0.45) / loglogistic(np.log(28), 0.9)
lik_c /= lik_c.max()


def state(s):
    d = prior.copy()
    if s >= R_ARR:
        d = d * lik_r
    if s >= S_LAB:
        d = d * lik_c
    return d * (g > s)


def quantiles(d, qs=(0.1, 0.5, 0.9)):
    c = np.cumsum(d)
    c /= c[-1]
    return [g[np.searchsorted(c, q)] for q in qs]


def stage_color(s):
    return BLUE if s < R_ARR else ORANGE if s < S_LAB else GREEN


# ---------------------------------------------------------------- SVG helpers
out = []
W, H = 1050, 512


def add(x):
    out.append(x)


def pts(xs, ys):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def text(x, y, s, size=14.9, color=INK, anchor="start", weight="normal", style="normal", extra=""):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size * 1.2:.1f}" fill="{color}" text-anchor="{anchor}" '
        f'font-weight="{weight}" font-style="{style}" {extra}>{s}</text>')


def icon(name, cx, cy, size, color, rot=0):
    d = re.search(r' d="([^"]+)"', (HERE / "icons" / f"{name}.svg").read_text()).group(1)
    k = size / 24
    add(f'<path d="{d}" fill="{color}" transform="translate({cx:.2f},{cy:.2f}) rotate({rot}) '
        f'translate({-size / 2:.2f},{-size / 2:.2f}) scale({k:.4f})"/>')


def hatched(shape, x0, y0, x1, y1):
    """Fill a shape with light diagonal hatching, drawn as clipped lines (a tiled pattern leaves seams in some PDF viewers)."""
    k = sum(o.startswith("<clipPath") for o in defs)
    defs.append(f'<clipPath id="hc{k}">{shape}</clipPath>')
    add(shape.replace("/>", ' fill="#f3f2ee"/>'))
    lines = "".join(f'<line x1="{c:.1f}" y1="{y0:.1f}" x2="{c - (y1 - y0):.1f}" y2="{y1:.1f}"/>'
                    for c in np.arange(x0, x1 + (y1 - y0), 9.9))
    add(f'<g clip-path="url(#hc{k})" stroke="#e2e1db" stroke-width="2">{lines}</g>')


def badge(name, cx, cy, color, r=15):
    add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="#ffffff" stroke="{color}" stroke-width="2.2"/>')
    icon(name, cx, cy, r * 1.15, color)


# ---------------------------------------------------------------- the incident, as it unfolds
X0, X1, SMIN, SMAX = 78, 930, -2.0, 80.0
Y0, Y1, TMIN, TMAX = 460, 194, 3.0, 500.0
sx = lambda s: X0 + (X1 - X0) * (np.asarray(s) - SMIN) / (SMAX - SMIN)
ty = lambda t: Y0 - (Y0 - Y1) * (np.log(np.asarray(t)) - np.log(TMIN)) / (np.log(TMAX) - np.log(TMIN))

defs = [f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{GREEN}"/></marker>']
for nm, col in (("b", BLUE), ("o", ORANGE), ("g", GREEN), ("y", YELLOW)):
    defs.append(f'<linearGradient id="h{nm}" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="{col}" stop-opacity="0.55"/>'
                f'<stop offset="1" stop-color="{col}" stop-opacity="0.12"/></linearGradient>')
    defs.append(f'<linearGradient id="v{nm}" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="{col}" stop-opacity="0.45"/>'
                f'<stop offset="1" stop-color="{col}" stop-opacity="0.04"/></linearGradient>')
gid = {BLUE: "b", ORANGE: "o", GREEN: "g", YELLOW: "y"}

# ruled-out region below the "now" line t = s
ss = np.linspace(TMIN, SMAX, 200)
hatched(f'<polygon points="{pts(sx(ss), ty(ss))} {sx(SMAX):.1f},{Y0} {sx(TMIN):.1f},{Y0}"/>', X0, Y1, X1, Y0)
add(f'<polyline points="{pts(sx(ss), ty(ss))}" fill="none" stroke="{MUTE}" stroke-width="1.1"/>')
ang = np.degrees(np.arctan2(ty(58) - ty(56), sx(58) - sx(56)))  # follow the slope of the line t = s
text(sx(57), ty(44), "now", size=13.8, color=INK2, anchor="middle", style="italic",
     extra=f'transform="rotate({ang:.1f} {sx(57):.1f} {ty(44):.1f})"')
text(sx(33), ty(8.2), "already passed:", size=13.8, color=INK2, anchor="middle")
text(sx(33), ty(8.2) + 18, "ruled out", size=13.8, color=INK2, anchor="middle")

# light guides from each event down to the plot
events = [(0, "phone-alert", "Call", "location, type", BLUE), (R_ARR, "police-badge", "Police arrive", "arrival time", ORANGE),
          (S_LAB, "clipboard-text-outline", "Clearance method", "label: towing", GREEN),
          (S_NOW, "clock-outline", "Still open", "elapsed time", INK2), (T_TRUE, "flag-checkered", "Cleared", "70 min", INK)]
BY = 82  # event badges
for s, *_ in events:
    add(f'<line x1="{sx(s):.1f}" y1="{BY + 14}" x2="{sx(s):.1f}" y2="{Y0}" stroke="{LINE}" stroke-width="1" stroke-dasharray="2 3"/>')

# bounds of the incident classes (minor < 30 min, intermediate, major > 2 h), with a key on the right
for t in (30, 120):
    add(f'<line x1="{X0}" y1="{ty(t):.1f}" x2="{X1}" y2="{ty(t):.1f}" stroke="{MUTE}" stroke-width="1" stroke-dasharray="1.5 3.5"/>')
text(X1 + 12, Y1 - 4, "Incident class", size=12.6, color=INK2)
for lo, hi, name, rng, shade in ((TMIN, 30, "minor", "under 30 min", "#d9d8d2"), (30, 120, "intermediate", "30 min to 2 h", "#a9a8a2"),
                                 (120, TMAX, "major", "over 2 h", "#6f6e69")):
    add(f'<line x1="{X1 + 12}" y1="{ty(lo) - 1.5:.1f}" x2="{X1 + 12}" y2="{ty(hi) + 1.5:.1f}" stroke="{shade}" stroke-width="4"/>')
    ym = ty(np.sqrt(lo * hi))
    text(X1 + 22, ym - 2, name, size=13.8)
    text(X1 + 22, ym + 15, rng, size=12.6, color=INK2)

# 80% band and median, colored by stage
s_grid = np.linspace(0, T_TRUE, 400)
Q = np.array([quantiles(state(s)) for s in s_grid])
for lo, hi in ((0, R_ARR), (R_ARR, S_LAB), (S_LAB, T_TRUE)):
    m = (s_grid >= lo) & (s_grid <= hi)
    col = stage_color(lo)
    xs = sx(s_grid[m])
    add(f'<polygon points="{pts(xs, ty(Q[m, 2]))} {pts(xs[::-1], ty(Q[m, 0])[::-1])}" fill="{col}" fill-opacity="0.11"/>')
    add(f'<polyline points="{pts(xs, ty(Q[m, 1]))}" fill="none" stroke="{col}" stroke-width="2.6" stroke-linecap="round"/>')

text(sx(33), ty(Q[np.searchsorted(s_grid, 33), 2]) + 17, "80% interval", size=12.6, color=INK2, anchor="middle", style="italic")
text(sx(60), ty(Q[np.searchsorted(s_grid, 60), 1]) + 14, "median", size=12.6, color=INK2, anchor="middle", style="italic")

# sideways densities at the moments a forecast is issued
for s in (0, R_ARR, S_LAB, S_NOW):
    d = state(s)
    d = d / np.trapezoid(d, lg)
    col = stage_color(s)
    thr = 0.08 * d.max()
    keep = d > thr
    w = 54 * (d[keep] - thr) / (d.max() - thr)  # width in px
    xs, ys = sx(s) + w, ty(g[keep])
    add(f'<polygon points="{sx(s):.2f},{ys[0]:.2f} {pts(xs, ys)} {sx(s):.2f},{ys[-1]:.2f}" fill="url(#h{gid[col]})"/>')
    add(f'<polyline points="{pts(xs, ys)}" fill="none" stroke="{col}" stroke-width="1.6"/>')
    add(f'<line x1="{sx(s):.2f}" y1="{ys[0]:.2f}" x2="{sx(s):.2f}" y2="{ys[-1]:.2f}" stroke="{col}" stroke-width="1.6"/>')

# true duration
add(f'<line x1="{X0}" y1="{ty(T_TRUE):.1f}" x2="{X1}" y2="{ty(T_TRUE):.1f}" stroke="{INK}" stroke-width="1.1" stroke-dasharray="5 4"/>')
text(sx(33), ty(T_TRUE) + 18, "true duration, 70 min", size=13.8, anchor="middle")
add(f'<circle cx="{sx(T_TRUE):.1f}" cy="{ty(T_TRUE):.1f}" r="5" fill="#ffffff" stroke="{INK}" stroke-width="1.8"/>')

# axes: light ticks only
add(f'<line x1="{X0}" y1="{Y0}" x2="{X1}" y2="{Y0}" stroke="{INK2}" stroke-width="1"/>')
add(f'<line x1="{X0}" y1="{Y1 - 6}" x2="{X0}" y2="{Y0}" stroke="{INK2}" stroke-width="1"/>')
for t in (5, 10, 20, 50, 100, 200, 500):
    add(f'<line x1="{X0 - 4}" y1="{ty(t):.1f}" x2="{X0}" y2="{ty(t):.1f}" stroke="{INK2}" stroke-width="1"/>')
    text(X0 - 8, ty(t) + 4, str(t), size=13.8, color=INK2, anchor="end")
for s in range(0, 80, 10):
    add(f'<line x1="{sx(s):.1f}" y1="{Y0}" x2="{sx(s):.1f}" y2="{Y0 + 4}" stroke="{INK2}" stroke-width="1"/>')
    text(sx(s), Y0 + 18, str(s), size=13.8, color=INK2, anchor="middle")
text((X0 + X1) / 2, Y0 + 40, "Time since the call (min)", size=16.1, anchor="middle")
text(26, (Y0 + Y1) / 2, "Predicted duration (min, log scale)", size=16.1, anchor="middle",
     extra=f'transform="rotate(-90 26 {(Y0 + Y1) / 2})"')

# the road along the same time axis: the crash blocks one lane and traffic queues in the other until the tow truck
# clears it; vehicles take the color of the stage they belong to
RT, RB = 110, 174
RM = (RT + RB) / 2
LU, LL = RM - 16, RM + 16  # open lane, blocked lane
ROAD, CAR = "#55544e", "#cfcec8"
add(f'<rect x="{X0 - 6}" y="{RT}" width="{X1 - X0 + 12}" height="{RB - RT}" fill="{ROAD}"/>')
for y in (RT + 3, RB - 3):
    add(f'<line x1="{X0 - 6}" y1="{y}" x2="{X1 + 6}" y2="{y}" stroke="#ffffff" stroke-opacity="0.75" stroke-width="1.2"/>')
add(f'<line x1="{X0 - 2}" y1="{RM}" x2="{X1 + 6}" y2="{RM}" stroke="#ffffff" stroke-width="1.4" stroke-dasharray="9 7"/>')
CARS = ["car-side", "car-hatchback", "car-estate", "car-side", "van-utility", "car-hatchback", "car-side", "car-estate"]
GREYS = ["#cfcec8", "#bdbcb6", "#dcdbd5"]
q, k = 3.2, 0  # queue in the open lane: it builds after the crash and is densest while the lane is closed
while q < 66.5:
    icon(CARS[k % len(CARS)], sx(q), LU, 26, GREYS[k % 3])
    q, k = q + 3.9 + 3.0 * np.exp(-(q - 3) / 12), k + 1
for q, k in ((74.8, 1), (79.2, 4)):  # free flow once cleared
    icon(CARS[k], sx(q), LU, 26, GREYS[k % 3])
icon("alert", sx(-1.1), LL, 15, YELLOW)
icon("car-side", sx(1.5), LL + 1, 30, BLUE, rot=16)  # the crashed car
kc = 30 / 24  # police car: a car with a light bar on its roof
icon("car-side", sx(9.6), LL + 1, 30, ORANGE)
add(f'<rect x="{sx(9.6) - 3 * kc:.1f}" y="{LL + 1 - 7.7 * kc:.1f}" width="{4 * kc:.1f}" height="{1.7 * kc:.1f}" rx="0.8" fill="#ffffff"/>')
for q in np.arange(15.5, 64, 7):  # cones close the lane
    icon("traffic-cone", sx(q), LL + 2, 17, YELLOW)
xt = sx(73.6)  # the tow truck pulls the crashed car away
icon("tow-truck", xt, LL, 36, GREEN)
icon("car-side", xt - 16.5 - 12, LL + 1, 28, BLUE, rot=-10)
icon("car-estate", sx(79.2), LL, 28, GREYS[0])
for s, ic, lab, info, col in events:
    badge(ic, sx(s), BY, col, r=14)
    text(sx(s), BY - 37, lab, size=13.6, anchor="middle")
    text(sx(s), BY - 21, info, size=12.6, color=INK2, anchor="middle", style="italic")

# ---------------------------------------------------------------- write
font = lambda fam, f, w, st: (f'@font-face{{font-family:"{fam}";src:url("file://{FONTS / f}");font-weight:{w};font-style:{st};}}')
css = (font("STIXT", "STIXGeneral.ttf", "normal", "normal") + font("STIXT", "STIXGeneralBol.ttf", "bold", "normal")
       + font("STIXT", "STIXGeneralItalic.ttf", "normal", "italic") + "text{font-family:STIXT;text-rendering:geometricPrecision;font-kerning:normal;}")
TOP = 24  # empty band above the event labels, cropped
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H - TOP}" viewBox="0 {TOP} {W} {H - TOP}">'
       f'<style>{css}</style><defs>{"".join(defs)}</defs><rect y="{TOP}" width="{W}" height="{H - TOP}" fill="#ffffff"/>' + "".join(out) + "</svg>")
import sys

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figs")
out.mkdir(parents=True, exist_ok=True)
dst = out / "fig_teaser.svg"
dst.write_text(svg)
print("wrote", dst)
