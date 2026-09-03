#!/usr/bin/env python3
import csv
import math
import sys

import matplotlib.pyplot as plt


if len(sys.argv) != 3:
    raise SystemExit("usage: python plot_trajectory_results.py INPUT.csv OUTPUT.png")

input_path, output_path = sys.argv[1:]
with open(input_path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

sample_count = max(int(row["samples"]) for row in rows)
selected_times = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
selected = [
    row for row in rows
    if int(row["samples"]) == sample_count
    and any(math.isclose(float(row["time"]), time, abs_tol=1e-10) for time in selected_times)
]

benchmark = {
    "odd": {0.98: 0.0216377265205618, 0.99: 0.029488338463743383, 1.0: 0.038673304161676754,
            1.01: 0.048620819656861526, 1.02: 0.05851569991410938},
    "even": {0.98: 0.06025785025440545, 0.99: 0.04996164346582242, 1.0: 0.039813015019188644,
             1.01: 0.030530533098452002, 1.02: 0.022696555464249635},
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
for axis, parity in zip(axes, ("odd", "even")):
    for time in sorted(selected_times):
        time_rows = sorted(
            (row for row in selected if math.isclose(float(row["time"]), time, abs_tol=1e-10)),
            key=lambda row: float(row["tR_over_tD"]),
        )
        x = [float(row["tR_over_tD"]) for row in time_rows]
        y = [float(row[f"{parity}_mean"]) for row in time_rows]
        error = [float(row[f"{parity}_stderr"]) for row in time_rows]
        axis.errorbar(x, y, yerr=error, marker="o", markersize=4, capsize=2, linewidth=1.2, label=f"t={time:g}")

    benchmark_points = sorted((1 / tD, value) for tD, value in benchmark[parity].items())
    axis.scatter([point[0] for point in benchmark_points], [point[1] for point in benchmark_points],
                 marker="x", s=55, color="black", linewidths=1.4, label="previous GS benchmark")
    axis.set_title(f"{parity.capitalize()} string order")
    axis.set_xlabel(r"$t_R/t_D$")
    axis.set_ylabel("SO")
    axis.grid(alpha=0.25)

axes[1].legend(fontsize=8, ncol=2)
fig.suptitle(f"Quantum trajectories: N=10, U=10, M={sample_count}, dt=0.05")
fig.tight_layout()
fig.savefig(output_path, dpi=220, bbox_inches="tight")
print(f"Wrote {output_path}")
