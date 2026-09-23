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
manifest="log/AKLT_GS_QN_D400_launcher_$(date +%Y%m%d_%H%M%S).log"

# Resume partial D400 states in place. New points start from the completed QN
# D200/Dstep40 states, while D400/Dstep50 keeps the historical files intact.
for tD in 1.0 0.99 1.01 0.98 1.02
do
    state="ground_states/N10_t(0.1,0.2)_tR1.0_tD${tD}_J0.0_U10.0/Dmax400/Dstep50/AKLT_N10_t(0.1,0.2)_tR1.0_tD${tD}_J0.0_U10.0_Dmax400.jld2"
    marker="${state%.jld2}.complete"
    if [[ -s "$state" && -f "$marker" ]]
    then
        printf 'skip tD=%s (already complete)\n' "$tD" | tee -a "$manifest"
        continue
    fi

    if [[ -s "$state" ]]
    then
        Dload=400
        Dstepload=50
    else
        Dload=200
        Dstepload=40
    fi

    log="log/AKLT_GS_QN_D400_tD${tD}_$(date +%Y%m%d_%H%M%S).log"
    printf 'start tD=%s Dload=%s Dstepload=%s log=%s\n' "$tD" "$Dload" "$Dstepload" "$log" | tee -a "$manifest"
    nice -n "$NICE_LEVEL" "$JULIA_BIN" --project=. AKLT_GS.jl \
        --load true -N 10 --Dmax 400 --Dstep 50 \
        --t1 0.1 --t2 0.2 --tR 1.0 --tD "$tD" -J 0.0 \
        --initD 200 --Dload "$Dload" --Dstepload "$Dstepload" -U 10.0 \
        2>&1 | tee "$log"
    touch "$marker"
    printf 'complete tD=%s\n' "$tD" | tee -a "$manifest"
done

printf 'all QN Dmax=400 ground states complete\n' | tee -a "$manifest"
