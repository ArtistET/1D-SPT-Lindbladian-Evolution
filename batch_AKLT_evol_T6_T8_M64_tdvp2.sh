#!/bin/bash
set -euo pipefail

mkdir -p sbatches/out sbatches/err log
manifest="log/trajectory_T6_T8_M64_submission_$(date +%Y%m%d_%H%M%S).log"

for tD in 0.98 0.99 1.0 1.01 1.02
do
    job_label=${tD/./}
    job_id=$(sbatch --parsable --array=1-64%20 --exclude=k18r3n06 \
        -c 4 --mem=8G -t 12:00:00 --job-name="tdvp2_T8_M64_tD${job_label}" \
        sub_evol.sh true true 6.0 10 100 20 0.1 0.2 1.0 "$tD" 0.0 \
        0.1 0.1 0.1 0.1 100 100 20 10.0 0.025 80 1 array \
        260903 1e-8 true 4)
    printf 'tD=%s job=%s trajectories=1:64 loadt=6.0 final_t=8.0\n' "$tD" "$job_id" | tee -a "$manifest"
done

printf 'manifest=%s\n' "$manifest"
