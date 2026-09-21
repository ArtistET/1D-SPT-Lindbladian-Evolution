#!/usr/bin/env python3
"""Plot trajectory means and standard errors as a dependency-free SVG."""

import csv
import math
import sys
from pathlib import Path


if len(sys.argv) != 3:
    raise SystemExit("usage: python3 plot_trajectory_results.py INPUT.csv OUTPUT.svg")

input_path, output_path = sys.argv[1:]
with open(input_path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
slope_path = Path(input_path).with_name(Path(input_path).stem + "_slopes.csv")
with slope_path.open(newline="", encoding="utf-8") as handle:
    slope_rows = list(csv.DictReader(handle))

sample_count = max(int(row["samples"]) for row in rows)
available_times = sorted({float(row["time"]) for row in rows if int(row["samples"]) == sample_count})
selected_times = tuple(time for time in available_times if math.isclose(time % 2, 0.0, abs_tol=1e-10))
selected = [
    row for row in rows
    if int(row["samples"]) == sample_count
    and any(math.isclose(float(row["time"]), time, abs_tol=1e-10) for time in selected_times)
]
ratios = sorted({float(row["tR_over_tD"]) for row in selected})
if len(selected) != len(selected_times) * len(ratios):
    raise SystemExit("input CSV does not contain a complete time/ratio grid")

benchmark = {
    "odd": {0.98: 0.0216377265205618, 0.99: 0.029488338463743383, 1.0: 0.038673304161676754,
            1.01: 0.048620819656861526, 1.02: 0.05851569991410938},
    "even": {0.98: 0.06025785025440545, 0.99: 0.04996164346582242, 1.0: 0.039813015019188644,
             1.01: 0.030530533098452002, 1.02: 0.022696555464249635},
}

# Existing N=10, Dmax=100 DMRG checkpoints (bond-unconverged); no matching U=1 data exists.
u0_benchmark = {
    "odd": {0.98: 0.019105032165657555, 0.99: 0.019391160426581115, 1.0: 0.019679175437066633,
            1.01: 0.01996615795171945, 1.02: 0.02025063167811967},
    "even": {0.98: 0.020428925873017912, 0.99: 0.020137634864220634, 1.0: 0.019852264237019832,
             1.01: 0.019572602503736065, 1.02: 0.019289489602627254},
}
u0_d200_center = {"odd": 0.05233055680636908, "even": 0.05438395563875983}
u0_slopes = {
    parity: abs((values[0.99] - values[1.01]) / (1 / 0.99 - 1 / 1.01))
    for parity, values in u0_benchmark.items()
}

for parity in ("odd", "even"):
    differences = []
    for row in selected:
        if math.isclose(float(row["time"]), 0.0, abs_tol=1e-10):
            differences.append(abs(float(row[f"{parity}_mean"]) - benchmark[parity][float(row["tD"])]))
    if max(differences) > 1e-8:
        raise SystemExit(f"{parity} t=0 values do not reproduce the ground-state benchmark")

width, height = 2000, 700
panel_width, panel_height = 520, 440
panel_lefts = (75, 710, 1370)
panel_top = 100
colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7A3E9D")


def svg_text(x, y, value, *, size=14, anchor="middle", weight="normal", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{weight}" fill="#17202a"{transform}>{value}</text>')


dt = float(rows[0].get("dt", 0.05))
parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    '<rect width="100%" height="100%" fill="#ffffff"/>',
    svg_text(width / 2, 34, f"Quantum trajectories: N=10, U=10, M={sample_count}, dt={dt:g}", size=21, weight="bold"),
    svg_text(width / 2, 59, "Points: trajectory mean; bars: standard error; crosses: U=10 GS; dashed: U=0 GS D100 (unconverged)", size=13),
]

x_min, x_max = min(ratios), max(ratios)
for panel_index, parity in enumerate(("odd", "even")):
    left = panel_lefts[panel_index]
    right = left + panel_width
    bottom = panel_top + panel_height
    y_values = []
    for row in selected:
        mean = float(row[f"{parity}_mean"])
        error = float(row[f"{parity}_stderr"])
        y_values.extend((mean - error, mean + error))
    y_values.extend(benchmark[parity].values())
    y_values.extend(u0_benchmark[parity].values())
    y_values.append(u0_d200_center[parity])
    y_min, y_max = min(y_values), max(y_values)
    padding = 0.07 * (y_max - y_min)
    y_min -= padding
    y_max += padding

    sx = lambda value: left + (value - x_min) / (x_max - x_min) * panel_width
    sy = lambda value: bottom - (value - y_min) / (y_max - y_min) * panel_height

    parts.append(f'<rect x="{left}" y="{panel_top}" width="{panel_width}" height="{panel_height}" fill="none" stroke="#4b5563"/>')
    for tick in range(6):
        value = y_min + tick * (y_max - y_min) / 5
        y = sy(value)
        parts.append(f'<line x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#d9dee5"/>')
        parts.append(svg_text(left - 10, y + 5, f"{value:.3f}", size=12, anchor="end"))
    for ratio in ratios:
        x = sx(ratio)
        parts.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{panel_top}" y2="{bottom}" stroke="#edf0f4"/>')
        parts.append(svg_text(x, bottom + 23, f"{ratio:.4f}".rstrip("0"), size=12))

    for time_index, time in enumerate(selected_times):
        time_rows = sorted(
            (row for row in selected if math.isclose(float(row["time"]), time, abs_tol=1e-10)),
            key=lambda row: float(row["tR_over_tD"]),
        )
        points = []
        for row in time_rows:
            x = sx(float(row["tR_over_tD"]))
            mean = float(row[f"{parity}_mean"])
            error = float(row[f"{parity}_stderr"])
            y, y_low, y_high = sy(mean), sy(mean - error), sy(mean + error)
            points.append((x, y))
            parts.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{y_low:.2f}" y2="{y_high:.2f}" stroke="{colors[time_index]}"/>')
            parts.append(f'<line x1="{x-3:.2f}" x2="{x+3:.2f}" y1="{y_low:.2f}" y2="{y_low:.2f}" stroke="{colors[time_index]}"/>')
            parts.append(f'<line x1="{x-3:.2f}" x2="{x+3:.2f}" y1="{y_high:.2f}" y2="{y_high:.2f}" stroke="{colors[time_index]}"/>')
        point_string = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(f'<polyline points="{point_string}" fill="none" stroke="{colors[time_index]}" stroke-width="2"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{colors[time_index]}" stroke="#ffffff" stroke-width="0.8"/>')

    for tD, value in benchmark[parity].items():
        x, y = sx(1 / tD), sy(value)
        parts.append(f'<path d="M{x-5:.2f},{y-5:.2f} L{x+5:.2f},{y+5:.2f} M{x-5:.2f},{y+5:.2f} L{x+5:.2f},{y-5:.2f}" stroke="#111827" stroke-width="2"/>')

    u0_points = " ".join(
        f"{sx(1 / tD):.2f},{sy(value):.2f}"
        for tD, value in sorted(u0_benchmark[parity].items(), key=lambda item: 1 / item[0])
    )
    parts.append(f'<polyline points="{u0_points}" fill="none" stroke="#6b7280" stroke-width="2" stroke-dasharray="7 5"/>')
    d200_x, d200_y = sx(1.0), sy(u0_d200_center[parity])
    parts.append(f'<rect x="{d200_x-5:.2f}" y="{d200_y-5:.2f}" width="10" height="10" fill="#ffffff" stroke="#6b7280" stroke-width="2"/>')

    parts.append(svg_text((left + right) / 2, panel_top - 16, f"{parity.capitalize()} string order", size=17, weight="bold"))
    parts.append(svg_text((left + right) / 2, bottom + 51, "t_R / t_D", size=15))
    parts.append(svg_text(left - 64, (panel_top + bottom) / 2, "String order", size=15, rotate=-90))

# Central finite-difference slopes versus time. A logarithmic axis makes the
# two-order-of-magnitude relaxation visible; error bars that include zero are
# clipped at the displayed lower bound.
slope_rows = [row for row in slope_rows if int(row["samples"]) == sample_count]
slope_times = sorted({float(row["time"]) for row in slope_rows})
if slope_times != available_times:
    raise SystemExit("slope CSV does not match the trajectory time grid")
left = panel_lefts[2]
right = left + panel_width
bottom = panel_top + panel_height
slope_y_min, slope_y_max = 1e-3, 2.0
sx = lambda value: left + (value - slope_times[0]) / (slope_times[-1] - slope_times[0]) * panel_width
sy = lambda value: bottom - (math.log10(max(value, slope_y_min)) - math.log10(slope_y_min)) / (
    math.log10(slope_y_max) - math.log10(slope_y_min)
) * panel_height
parts.append(f'<rect x="{left}" y="{panel_top}" width="{panel_width}" height="{panel_height}" fill="none" stroke="#4b5563"/>')
for value in (1e-3, 1e-2, 1e-1, 1.0):
    y = sy(value)
    parts.append(f'<line x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#d9dee5"/>')
    parts.append(svg_text(left - 10, y + 5, f"{value:g}", size=12, anchor="end"))
for time in selected_times:
    x = sx(time)
    parts.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{panel_top}" y2="{bottom}" stroke="#edf0f4"/>')
    parts.append(svg_text(x, bottom + 23, f"{time:g}", size=12))

slope_colors = {"odd": "#0072B2", "even": "#D55E00"}
for parity in ("odd", "even"):
    points = []
    for row in slope_rows:
        time = float(row["time"])
        value = abs(float(row[f"{parity}_slope"]))
        error = float(row[f"{parity}_slope_stderr"])
        x, y = sx(time), sy(value)
        points.append((x, y))
        if math.isclose(time, round(time), abs_tol=1e-10):
            y_low = sy(max(value - error, slope_y_min))
            y_high = sy(min(value + error, slope_y_max))
            parts.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{y_low:.2f}" y2="{y_high:.2f}" stroke="{slope_colors[parity]}" opacity="0.65"/>')
    point_string = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    parts.append(f'<polyline points="{point_string}" fill="none" stroke="{slope_colors[parity]}" stroke-width="2"/>')
    for row in slope_rows:
        time = float(row["time"])
        if math.isclose(time, round(time), abs_tol=1e-10):
            value = abs(float(row[f"{parity}_slope"]))
            parts.append(f'<circle cx="{sx(time):.2f}" cy="{sy(value):.2f}" r="3.2" fill="{slope_colors[parity]}"/>')
    y_ref = sy(u0_slopes[parity])
    parts.append(f'<line x1="{left}" x2="{right}" y1="{y_ref:.2f}" y2="{y_ref:.2f}" stroke="{slope_colors[parity]}" stroke-width="2" stroke-dasharray="7 5"/>')

parts.append(svg_text((left + right) / 2, panel_top - 16, "Absolute central slopes", size=17, weight="bold"))
parts.append(svg_text((left + right) / 2, bottom + 51, "Evolution time", size=15))
parts.append(svg_text(left - 58, (panel_top + bottom) / 2, "|dSO/d(t_R/t_D)| (log)", size=15, rotate=-90))
parts.append(svg_text((left + right) / 2, 665, "Error bars including zero are clipped at 10⁻³", size=11))

legend_y = 605
legend_start = 75
for index, time in enumerate(selected_times):
    x = legend_start + index * 105
    parts.append(f'<line x1="{x}" x2="{x+24}" y1="{legend_y}" y2="{legend_y}" stroke="{colors[index]}" stroke-width="3"/>')
    parts.append(f'<circle cx="{x+12}" cy="{legend_y}" r="3.5" fill="{colors[index]}"/>')
    parts.append(svg_text(x + 31, legend_y + 5, f"t={time:g}", size=12, anchor="start"))
parts.append('<path d="M850,600 L860,610 M850,610 L860,600" stroke="#111827" stroke-width="2"/>')
parts.append(svg_text(867, legend_y + 5, "U=10 GS", size=12, anchor="start"))
parts.append(f'<line x1="970" x2="994" y1="{legend_y}" y2="{legend_y}" stroke="#6b7280" stroke-width="2" stroke-dasharray="7 5"/>')
parts.append(svg_text(1001, legend_y + 5, "U=0 GS D100", size=12, anchor="start"))
parts.append(f'<rect x="1135" y="600" width="10" height="10" fill="#ffffff" stroke="#6b7280" stroke-width="2"/>')
parts.append(svg_text(1152, legend_y + 5, "U=0 GS D200 center", size=12, anchor="start"))
parts.append(f'<line x1="1390" x2="1414" y1="{legend_y}" y2="{legend_y}" stroke="{slope_colors["odd"]}" stroke-width="2"/>')
parts.append(svg_text(1421, legend_y + 5, "|odd slope|", size=12, anchor="start"))
parts.append(f'<line x1="1530" x2="1554" y1="{legend_y}" y2="{legend_y}" stroke="{slope_colors["even"]}" stroke-width="2"/>')
parts.append(svg_text(1561, legend_y + 5, "|even slope|", size=12, anchor="start"))
parts.append(f'<line x1="1680" x2="1704" y1="{legend_y}" y2="{legend_y}" stroke="{slope_colors["odd"]}" stroke-width="2" stroke-dasharray="7 5"/>')
parts.append(svg_text(1711, legend_y + 5, "U=0 odd", size=12, anchor="start"))
parts.append(f'<line x1="1810" x2="1834" y1="{legend_y}" y2="{legend_y}" stroke="{slope_colors["even"]}" stroke-width="2" stroke-dasharray="7 5"/>')
parts.append(svg_text(1841, legend_y + 5, "U=0 even", size=12, anchor="start"))
parts.append('</svg>')

Path(output_path).write_text("\n".join(parts), encoding="utf-8")
print(f"Wrote {output_path}; M={sample_count}; t=0 benchmark verified within 1e-8")
