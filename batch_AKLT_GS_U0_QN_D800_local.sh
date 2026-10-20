#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JULIA_BIN=${JULIA_BIN:-$HOME/.local/bin/julia}
NICE_LEVEL=${NICE_LEVEL:-10}
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

cd "$PROJECT_DIR"
mkdir -p log
manifest="log/AKLT_GS_U0_QN_D800_launcher_$(date +%Y%m%d_%H%M%S).log"

for tD in 1.0 0.99 1.01 0.98 1.02
do
    state="ground_states/N10_t(0.1,0.2)_tR1.0_tD${tD}_J0.0_U0.0/Dmax800/Dstep100/AKLT_N10_t(0.1,0.2)_tR1.0_tD${tD}_J0.0_U0.0_Dmax800.jld2"
    marker="${state%.jld2}.complete"
    if [[ -s "$state" && -f "$marker" ]]
    then
        printf 'skip tD=%s (already complete)\n' "$tD" | tee -a "$manifest"
        continue
    fi

    if [[ "$tD" == "1.0" ]]
    then
        Dload=400
        Dstepload=50
    else
        Dload=0
        Dstepload=0
    fi
    log="log/AKLT_GS_U0_QN_D800_tD${tD}_$(date +%Y%m%d_%H%M%S).log"
    printf 'start tD=%s Dload=%s log=%s\n' "$tD" "$Dload" "$log" | tee -a "$manifest"
    nice -n "$NICE_LEVEL" "$JULIA_BIN" --project=. AKLT_GS_U0_QN.jl \
        "$tD" 800 "$Dload" "$Dstepload" 2>&1 | tee "$log"
    touch "$marker"
    printf 'complete tD=%s\n' "$tD" | tee -a "$manifest"
done

printf 'all U=0 QN D800 ground states complete\n' | tee -a "$manifest"
