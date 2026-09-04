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

sample_count = max(int(row["samples"]) for row in rows)
selected_times = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
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

for parity in ("odd", "even"):
    differences = []
    for row in selected:
        if math.isclose(float(row["time"]), 0.0, abs_tol=1e-10):
            differences.append(abs(float(row[f"{parity}_mean"]) - benchmark[parity][float(row["tD"])]))
    if max(differences) > 1e-8:
        raise SystemExit(f"{parity} t=0 values do not reproduce the ground-state benchmark")

width, height = 1400, 650
panel_width, panel_height = 570, 440
panel_lefts = (90, 790)
panel_top = 100
colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def svg_text(x, y, value, *, size=14, anchor="middle", weight="normal", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{weight}" fill="#17202a"{transform}>{value}</text>')


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    '<rect width="100%" height="100%" fill="#ffffff"/>',
    svg_text(width / 2, 34, f"Quantum trajectories: N=10, U=10, M={sample_count}, dt=0.05", size=21, weight="bold"),
    svg_text(width / 2, 59, "Points: trajectory mean; bars: standard error; crosses: previous ground-state benchmark", size=13),
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

    parts.append(svg_text((left + right) / 2, panel_top - 16, f"{parity.capitalize()} string order", size=17, weight="bold"))
    parts.append(svg_text((left + right) / 2, bottom + 51, "t_R / t_D", size=15))
    parts.append(svg_text(left - 64, (panel_top + bottom) / 2, "String order", size=15, rotate=-90))

legend_y = 605
legend_start = 220
for index, time in enumerate(selected_times):
    x = legend_start + index * 132
    parts.append(f'<line x1="{x}" x2="{x+24}" y1="{legend_y}" y2="{legend_y}" stroke="{colors[index]}" stroke-width="3"/>')
    parts.append(f'<circle cx="{x+12}" cy="{legend_y}" r="3.5" fill="{colors[index]}"/>')
    parts.append(svg_text(x + 31, legend_y + 5, f"t={time:g}", size=12, anchor="start"))
parts.append('<path d="M1034,600 L1044,610 M1034,610 L1044,600" stroke="#111827" stroke-width="2"/>')
parts.append(svg_text(1051, legend_y + 5, "GS benchmark", size=12, anchor="start"))
parts.append('</svg>')

Path(output_path).write_text("\n".join(parts), encoding="utf-8")
print(f"Wrote {output_path}; M={sample_count}; t=0 benchmark verified within 1e-8")
