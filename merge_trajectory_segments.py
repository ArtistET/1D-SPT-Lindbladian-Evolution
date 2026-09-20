#!/usr/bin/env python3
"""Merge consecutive trajectory-summary CSV segments."""

import csv
import math
import sys
from pathlib import Path


if len(sys.argv) < 4:
    raise SystemExit("usage: python3 merge_trajectory_segments.py OUTPUT.csv SEGMENT.csv SEGMENT.csv [...]")

output = Path(sys.argv[1])
segments = [Path(path) for path in sys.argv[2:]]


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def common_samples(groups):
    return set.intersection(*({int(row["samples"]) for row in rows} for rows in groups))


rows_by_segment = [read(path) for path in segments]
samples = common_samples(rows_by_segment)
merged = []
jump_offsets = {}
previous_end = None
for index, rows in enumerate(rows_by_segment):
    start = min(float(row["time"]) for row in rows)
    end = max(float(row["time"]) for row in rows)
    if previous_end is not None and not math.isclose(start, previous_end, abs_tol=1e-10):
        raise SystemExit(f"non-contiguous segments: {previous_end} -> {start}")
    for row in rows:
        sample_count = int(row["samples"])
        time = float(row["time"])
        if sample_count not in samples or (index and math.isclose(time, start, abs_tol=1e-10)):
            continue
        item = dict(row)
        key = (row["tD"], row["samples"])
        item["mean_cumulative_jumps"] = f'{float(row["mean_cumulative_jumps"]) + jump_offsets.get(key, 0.0):.16g}'
        merged.append(item)
    for row in rows:
        if int(row["samples"]) in samples and math.isclose(float(row["time"]), end, abs_tol=1e-10):
            key = (row["tD"], row["samples"])
            jump_offsets[key] = jump_offsets.get(key, 0.0) + float(row["mean_cumulative_jumps"])
    previous_end = end

merged.sort(key=lambda row: (float(row["tD"]), int(row["samples"]), float(row["time"])))
fieldnames = list(merged[0])
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(merged)

expected_times = sorted({float(row["time"]) for row in merged})
for key in {(row["tD"], row["samples"]) for row in merged}:
    times = [float(row["time"]) for row in merged if (row["tD"], row["samples"]) == key]
    if len(times) != len(expected_times) or any(
        not math.isclose(actual, expected, abs_tol=1e-10)
        for actual, expected in zip(times, expected_times)
    ):
        raise SystemExit(f"incomplete merged time grid for {key}")

slope_groups = [read(path.with_name(path.stem + "_slopes.csv")) for path in segments]
slope_samples = common_samples(slope_groups)
slope_rows = []
previous_end = None
for index, rows in enumerate(slope_groups):
    start = min(float(row["time"]) for row in rows)
    end = max(float(row["time"]) for row in rows)
    if previous_end is not None and not math.isclose(start, previous_end, abs_tol=1e-10):
        raise SystemExit(f"non-contiguous slope segments: {previous_end} -> {start}")
    slope_rows.extend(
        row for row in rows
        if int(row["samples"]) in slope_samples
        and not (index and math.isclose(float(row["time"]), start, abs_tol=1e-10))
    )
    previous_end = end

slope_rows.sort(key=lambda row: (int(row["samples"]), float(row["time"])))
slope_output = output.with_name(output.stem + "_slopes.csv")
with slope_output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(slope_rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(slope_rows)

expected_slope_times = sorted({float(row["time"]) for row in slope_rows})
for sample_count in slope_samples:
    times = [float(row["time"]) for row in slope_rows if int(row["samples"]) == sample_count]
    if len(times) != len(expected_slope_times) or any(
        not math.isclose(actual, expected, abs_tol=1e-10)
        for actual, expected in zip(times, expected_slope_times)
    ):
        raise SystemExit(f"incomplete merged slope grid for M={sample_count}")

print(
    f"Wrote {output} and {slope_output}; samples={sorted(samples)}; "
    f"times={expected_times[0]:g}:0.1:{expected_times[-1]:g}"
)
