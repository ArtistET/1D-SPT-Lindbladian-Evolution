#!/bin/bash

mkdir -p sbatches/out sbatches/err log

# ============ 可修改参数（顶层）============
# 耗散强度
I1=0.1
I2=0.1
IR=0.1
ID=0.1
# 加载的基态/切片的 bond dimension 信息
# 注意：Dload/Dsload 必须与要加载文件的 Dmax/Dstep 对得上
#   - load=true,  loadsl=false : 加载基态文件 (其 Dmax/Dstep)
#   - load=true,  loadsl=true  : 加载演化切片文件 (其 Dmax/Dstep)
Dload=400
Dsload=50
# 加载/初始化控制
load=true       # 是否加载初始态
loadsl=false    # 是否从某个演化切片继续
loadt=0.0       # 加载切片对应的时刻

# 时间步进
dt=0.01
tsmax=3
# 每个作业顺序计算的轨迹数。多个独立作业可用不同 traj_start 并行提交。
ntraj=1
traj_start=1
njobs=1
seed=1234
cutoff=1e-8
save_traj=false

# ============ 其余信息（与 GS 批量脚本保持一致）============
Dmax=400
Dstep=50
t1=0.1
t2=0.2
tR=1.0
J=0.0

# ============ 循环提交 Slurm 作业 ============
for N in 10
do
    for U in 10
    do
        for tD in 0.98 
        do
            for ((job_index=0; job_index<njobs; job_index++))
            do
                this_traj_start=$((traj_start + job_index*ntraj))
                echo "Submitting trajectory job for N=$N tD=$tD U=$U Dmax=$Dmax Dstep=$Dstep, loading D=$Dload Dstep=$Dsload, I=($I1,$I2,$IR,$ID), dt=$dt tsmax=$tsmax loadt=$loadt traj_start=$this_traj_start ntraj=$ntraj"
                # initD=Dload 是有意的：不加载时从 checkpoint 对应的键维设置开始 DMRG。
                sbatch sub_evol.sh $load $loadsl $loadt $N $Dmax $Dstep $t1 $t2 $tR $tD $J $I1 $I2 $IR $ID $Dload $Dload $Dsload $U $dt $tsmax $ntraj $this_traj_start $seed $cutoff $save_traj
            done
        done
    done
done
