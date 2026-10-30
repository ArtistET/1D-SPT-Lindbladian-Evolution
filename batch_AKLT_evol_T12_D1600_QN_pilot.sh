#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_DIR"
export JULIA_BIN=${JULIA_BIN:-$HOME/.local/bin/julia}
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p log
manifest="log/trajectory_T12_D1600_QN_pilot_$(date +%Y%m%d_%H%M%S).log"

for td in 0.98 1.0 1.02; do
    gs="ground_states/N10_t(0.1,0.2)_tR1.0_tD${td}_J0.0_U10.0/Dmax400/Dstep50/AKLT_N10_t(0.1,0.2)_tR1.0_tD${td}_J0.0_U10.0_Dmax400.jld2"
    [[ -s "$gs" && -f "${gs%.jld2}.complete" ]] || { echo "missing complete D400 QN GS: $td" >&2; exit 1; }
done

for td in 0.98 1.0 1.02; do
    run="trajectory_evolution/N10_t(0.1,0.2)_tR1.0_tD${td}_J0.0_U10.0_I10.1_I20.1_IR0.1_ID0.1/Dmax1600_dt0.025_tdvp2_seed260903"
    for id in 1 2 3 4; do
        result="$run/results/T0.0_to_T12.0_traj${id}-${id}.jld2"
        checkpoint="$run/checkpoints/T12.0/trajectory_${id}.jld2"
        if [[ -s "$result" && -s "$checkpoint" ]]; then
            echo "skip existing tD=$td id=$id" | tee -a "$manifest"
            continue
        fi
        if [[ -e "$result" || -e "$checkpoint" ]]; then
            echo "partial pair requires inspection: tD=$td id=$id" | tee -a "$manifest" >&2
            exit 1
        fi
        while (( $(jobs -pr | wc -l) >= 3 )); do wait -n; done
        echo "start tD=$td id=$id Dload=400 Dmax=1600" | tee -a "$manifest"
        nice -n 10 ./sub_evol.sh true false 0.0 10 1600 50 \
            0.1 0.2 1.0 "$td" 0.0 0.1 0.1 0.1 0.1 \
            400 400 50 10.0 0.025 480 1 "$id" \
            260903 1e-8 true 4 &
    done
done
wait
echo "D1600 pilot launcher finished" | tee -a "$manifest"
