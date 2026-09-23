#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAJECTORIES=${TRAJECTORIES:-4}
PARALLEL_JOBS=${PARALLEL_JOBS:-4}
NICE_LEVEL=${NICE_LEVEL:-10}
export JULIA_BIN=${JULIA_BIN:-$HOME/.local/bin/julia}
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

cd "$PROJECT_DIR"
mkdir -p log
manifest="log/trajectory_T12_D400_QN_M${TRAJECTORIES}_local_$(date +%Y%m%d_%H%M%S).log"

for tD in 0.98 0.99 1.0 1.01 1.02
do
    state="ground_states/N10_t(0.1,0.2)_tR1.0_tD${tD}_J0.0_U10.0/Dmax400/Dstep50/AKLT_N10_t(0.1,0.2)_tR1.0_tD${tD}_J0.0_U10.0_Dmax400.jld2"
    [[ -s "$state" && -f "${state%.jld2}.complete" ]] || {
        printf 'incomplete D400 ground state for tD=%s\n' "$tD" >&2
        exit 1
    }
done

for tD in 0.98 0.99 1.0 1.01 1.02
do
    run_dir="trajectory_evolution/N10_t(0.1,0.2)_tR1.0_tD${tD}_J0.0_U10.0_I10.1_I20.1_IR0.1_ID0.1/Dmax400_dt0.025_tdvp2_seed260903"
    for trajectory_id in $(seq 1 "$TRAJECTORIES")
    do
        result="$run_dir/results/T0.0_to_T12.0_traj${trajectory_id}-${trajectory_id}.jld2"
        checkpoint="$run_dir/checkpoints/T12.0/trajectory_${trajectory_id}.jld2"
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
        nice -n "$NICE_LEVEL" ./sub_evol.sh true false 0.0 10 400 50 \
            0.1 0.2 1.0 "$tD" 0.0 0.1 0.1 0.1 0.1 \
            400 400 50 10.0 0.025 480 1 "$trajectory_id" \
            260903 1e-8 true 4 &
    done
done

wait
printf 'all D400 T=12 trajectories finished\n' | tee -a "$manifest"
