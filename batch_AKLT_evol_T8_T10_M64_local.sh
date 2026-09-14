#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PARALLEL_JOBS=${PARALLEL_JOBS:-4}
export JULIA_BIN=${JULIA_BIN:-$HOME/.local/bin/julia}
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

cd "$PROJECT_DIR"
mkdir -p log
manifest="log/trajectory_T8_T10_M64_local_$(date +%Y%m%d_%H%M%S).log"

for tD in 0.98 0.99 1.0 1.01 1.02
do
    run_dir="trajectory_evolution/N10_t(0.1,0.2)_tR1.0_tD${tD}_J0.0_U10.0_I10.1_I20.1_IR0.1_ID0.1/Dmax100_dt0.025_tdvp2_seed260903"
    for trajectory_id in $(seq 1 64)
    do
        result="$run_dir/results/T8.0_to_T10.0_traj${trajectory_id}-${trajectory_id}.jld2"
        checkpoint="$run_dir/checkpoints/T10.0/trajectory_${trajectory_id}.jld2"
        if [[ -s "$result" && -s "$checkpoint" ]]
        then
            printf 'skip tD=%s trajectory=%s (already complete)\n' "$tD" "$trajectory_id" | tee -a "$manifest"
            continue
        fi
        while (( $(jobs -pr | wc -l) >= PARALLEL_JOBS ))
        do
            wait -n
        done
        printf 'start tD=%s trajectory=%s\n' "$tD" "$trajectory_id" | tee -a "$manifest"
        ./sub_evol.sh true true 8.0 10 100 20 0.1 0.2 1.0 "$tD" 0.0 \
            0.1 0.1 0.1 0.1 100 100 20 10.0 0.025 80 1 "$trajectory_id" \
            260903 1e-8 true 4 &
    done
done

wait
printf 'all T=8->10 trajectories finished\n' | tee -a "$manifest"
