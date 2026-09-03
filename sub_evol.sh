#!/bin/bash
#SBATCH --job-name=AKLT_evol
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --partition=kshcnormal02
#SBATCH -o sbatches/out/%x_%j.out
#SBATCH -e sbatches/err/%x_%j.err

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

LOG_DIR=log
mkdir -p "$LOG_DIR"

# 轨迹之间用多个 Slurm 作业并行；单条轨迹内部只让 BLAS 使用已申请的核。
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

load=${1:-true}
loadsl=${2:-false}
loadt=${3:-0.0}
N=${4:-10}
Dmax=${5:-400}
Dstep=${6:-50}
t1=${7:-0.1}
t2=${8:-0.2}
tR=${9:-1.0}
tD=${10:-2.0}
J=${11:-0.0}
I1=${12:-0.1}
I2=${13:-0.1}
IR=${14:-0.1}
ID=${15:-0.1}
initD=${16:-10}
Dload=${17:-400}
Dstepload=${18:-50}
U=${19:-10.0}
dt=${20:-0.01}
tsmax=${21:-20}
ntraj=${22:-1}
traj_start=${23:-1}
seed=${24:-1234}
cutoff=${25:-1e-8}
save_traj=${26:-false}

LOG_FILE="$LOG_DIR/AKLT_evol_${CURRENT_TIME}_N${N}_Dmax${Dmax}_t1_${t1}_t2_${t2}_tR${tR}_tD${tD}_J${J}_U${U}_I1${I1}_I2${I2}_IR${IR}_ID${ID}_loadt${loadt}_dt${dt}_ts${tsmax}_traj${traj_start}n${ntraj}.log"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "Slurm job=${SLURM_JOB_ID:-interactive} host=$(hostname) started=$(date --iso-8601=seconds)"

julia --project=/public/home/rdcha/Workspace/1D-SPT-Lindbladian-Evolution \
  /public/home/rdcha/Workspace/1D-SPT-Lindbladian-Evolution/AKLT_evolution.jl \
  --load "$load" --loadsl "$loadsl" --loadt "$loadt" -N "$N" \
  --Dmax "$Dmax" --Dstep "$Dstep" \
  --t1 "$t1" --t2 "$t2" \
  --tR "$tR" --tD "$tD" \
  -J "$J" \
  --I1 "$I1" --I2 "$I2" --IR "$IR" --ID "$ID" \
  --initD "$initD" \
  --Dload "$Dload" --Dstepload "$Dstepload" -U "$U" \
  --dt "$dt" --tsmax "$tsmax" \
  --ntraj "$ntraj" --traj-start "$traj_start" --seed "$seed" \
  --cutoff "$cutoff" --save-traj "$save_traj"
