# 项目协作与实验状态

本文件记录后续代理必须继承的工作约定、已经完成的改动、数值验证结果和当前风险。不要把“程序能运行”误写成“物理结果已收敛”。

## 工作目录、服务器与 Git

- 唯一本地工作目录：`/home/et2333/Workspace/1D-SPT-Lindbladian-Evolution`。
- 曙光昆山目录：`/public/home/rdcha/Workspace/1D-SPT-Lindbladian-Evolution`，SSH 别名为 `Shuguang_Kunshan`。
- 远程仓库：`git@github.com:ArtistET/1D-SPT-Lindbladian-Evolution.git`，主分支为 `main`。
- 当前实验结果基线：`213f9ed add N10 trajectory convergence results`；更新本文件后的最新提交以 `git log -1` 为准。
- 服务器上的 `AKLT_GS.jl` 有用户未提交修改，必须保留；拉取前先检查和暂存，拉取后恢复，不得顺手提交或覆盖。
- `ground_states/`、`trajectory_evolution/`、`log/` 和 `sbatches/` 是大文件或运行产物，不进入 Git。正式展示用的小型汇总和图片放在 `experiment_results/`。
- 不要在本地启动大规模张量网络计算。提交服务器任务前，先检查现有结果、`squeue`、`sacct` 和日志，避免重复计算。
- 所有服务器任务必须把 Julia 标准输出和错误写入项目 `log/`；同时保留提交 manifest 和 `sbatches/out|err`，不能只依赖终端输出。

## 当前文件职责

- `AKLT_evolution.jl`：当前量子轨迹/Monte Carlo wave-function 实现。
- `AKLT_evolution_old.jl`：原 MPDO/Kraus 全分支求和实现，仅作对照，不要删除。
- `AKLT_GS.jl`：哈密顿量、基态 DMRG、string-order 算符等公共定义；新生成基态默认使用 QN 守恒。
- `QUANTUM_TRAJECTORY_ALGORITHM.md`：从 Lindblad 方程到代码实现的详细推导。
- `test_trajectory.jl`：小系统构件与概率检查。
- `sub_evol.sh`：单个 Slurm 作业入口及详细日志记录。
- `batch_AKLT_evol_sub.sh`：按互不重叠的轨迹编号并行提交。
- `analyze_trajectory_results.jl`：合并分批或部分完成的 JLD2 结果并输出不同样本数的均值、标准误、键维和跳跃统计。
- `plot_trajectory_results.py`：只依赖 Python 标准库，生成 SVG；同时校验 `t=0` 是否复现旧基态 benchmark。

## 已完成的算法改造

旧代码直接演化密度矩阵 MPO，每步需要构造并相加所有 `K_i ρ K_i†`，键维和内存迅速增长。当前入口改为多条纯态 MPS 量子轨迹：

1. 每步用两张密度关联矩阵计算全部跳跃概率。
2. 每条轨迹只随机选择 no-jump 或一个 jump 分支。
3. 对归一化后的 MPS 测量奇、偶 string order。
4. 最后对独立轨迹求均值和标准误。

本模型仍有 `16N` 个物理跳跃通道：每晶胞四类 bond、两个方向、两种自旋。它们没有被物理合并。内存优化来自：通道平时只保存元数据；只为被选中的跳跃临时构造 MPO；`K0` 中成对的 `L†L` 用数算符恒等式合并到一个 `OpSum`。

String order 的正式测量采用预构造 SO MPO 与 MPS 的直接收缩。每次运行前仍保留旧的逐段 `apply` 路径作为初态一致性检查；该检查成本相对演化很小，不要无故删除。

每条轨迹使用 `MersenneTwister(seed + trajectory_id - 1)`，不同轨迹相互独立。这不是 MCMC，不需要 thinning。不同参数点复用相同轨迹编号和随机种子属于 common-random-number 配对，可降低曲线差分噪声；统计分析时不要错误地把不同参数点当完全独立样本。

结果每完成一条轨迹就保存一次，并记录 `completed_trajectories`。作业在轨迹中间超时后，已完成的行仍可合并；同一轨迹编号若被 rescue 作业重算，分析器必须避免重复计数。

## 已做数值 benchmark

测试模型取 `N=10, U=10, tR=1`，其余哈密顿量参数为 `t1=0.1, t2=0.2, J=0`，耗散振幅 `I1=I2=IR=ID=0.1`。

- `Dmax=100, dt=0.05` 与 `Dmax=150, dt=0.05` 在 `T=0.1` 的差异：odd 约 `1.9e-7`，even 约 `6.0e-7`。
- `Dmax=100` 下 `dt=0.05` 与 `dt=0.025` 在 `T=0.1` 的差异：odd 约 `7.2e-7`，even 约 `1.7e-6`。
- `Dmax=60` 与 `Dmax=100` 相差约 `2.7e-4`，不应使用 `Dmax=60`。
- 短时 benchmark 峰值内存约：`Dmax=100` 为 3.6 GB，`Dmax=150` 为 6.2 GB。正式测试按每作业 4 CPU、8 GB 提交。
- 这些 benchmark 主要覆盖从基态出发、跳跃前的短时间区间，不能替代跳跃后的传播器验证。

## N=10、U=10 的 64 样本实验

参数：

```text
tD = 0.98, 0.99, 1.0, 1.01, 1.02
tR/tD = 1/tD
Dload = Dmax = 100
Dstepload = Dstep = 20
dt = 0.05
tsmax = 10
T = 0.5
seed = 260903
cutoff = 1e-8
4 trajectories/job, 4 CPU/job, 8 GB/job
```

结果文件：

- `experiment_results/trajectory_N10_U10_M64.csv`
- `experiment_results/trajectory_N10_U10_M64.svg`
- `experiment_results/trajectory_N10_U10_M64.png`

CSV 同时包含 `M=4,8,16,32,64` 的累计样本统计。五个参数点的 `t=0` SO 与旧基态 benchmark 相差小于 `1e-8`，初始化和两种 SO 测量路径通过检查。

64 样本仍未统计收敛：

- `M=32 -> 64` 最大均值变化：odd `2.45e-3`，even `1.17e-3`。
- `M=64` 最大标准误：odd `1.61e-3`，even `1.04e-3`。
- 按当前最坏方差和 `1/sqrt(M)` 粗略外推，标准误到 `1e-3`、`7.5e-4`、`5e-4` 分别约需 170、300、670 条轨迹。
- 后续若只看定性曲线，建议至少验证到 `M=256`；用于斜率拟合建议从 `M=512` 起，并在 `128/256/512` 逐级检查。这个数目是外推，不是已经验证的收敛结论。

到 `T=0.5` 时平均每条轨迹只有 `0.28125` 次跳跃。中心数值斜率从初态到 `T=0.5` 大致为：odd `-0.9565 -> -0.9184`，even `0.9715 -> 0.9837`。当前曲线仍接近初态，尚未显示向 `U<1.7` 平缓曲线演化；更长时间实验前必须先解决下面的传播器风险。

记录的最大 MPS 键维始终达到 `Dmax=100`，而初态本身已经是 D=100，因此这批数据不能说明真实所需键维是否下降，也不能排除截断误差。

## 运行状态与日志

M=64 扩展批次的 40 个作业全部 `COMPLETED`，耗时约 47–51 分钟/作业。首批 40 个作业中有一个 `120794088` 因节点性能异常在 3 小时超时；其中轨迹 1、2 已完成，轨迹 3、4 后由 `120806544` 和 `120806553` 分别在约 15 分钟内补齐。汇总文件已经包含完整的轨迹编号 1:64。

对应提交记录：

- `log/trajectory_convergence_submission_152883b.log`
- `log/trajectory_convergence_submission_M64_b071f0c.log`

详细输出位于：

- `log/AKLT_evol_*.log`
- `sbatches/out/AKLT_evol_<jobid>.out`
- `sbatches/err/AKLT_evol_<jobid>.err`

判断任务成功时必须同时检查 Slurm 状态、日志末尾、`completed_trajectories` 和结果文件；`sub_evol.sh` 使用 `tee`，目前没有 `set -o pipefail`，只看 Slurm 的退出状态不够可靠。

## 当前最高优先级风险

### 1. 跳跃后的固定能量平移失效

当前 `K0` 使用初态能量

```text
E_ref = <psi_initial|H|psi_initial>
```

消除广延基态能量导致的一阶 Euler 范数误差。它在跳跃前有效，但跳跃会改变轨迹能量。日志 `120794088` 中一次跳跃后的两个 no-jump 步出现：

```text
branch_norm = 1.2366163360
branch_norm = 1.2466448904
```

而对应的一阶期望约为 `1 - p_jump ~= 0.9763`。归一化不能消除由此造成的状态方向误差。因此现有图只能作为流程测试，不能作为正式长时间物理结果。

下一轮长计算前，应先比较并验证以下最小方案之一：每步按当前态更新能量平移，或改用更稳定/更高阶的非厄米 no-jump 传播器。修复后必须专门做“发生跳跃后的 dt 收敛”测试。

### 2. 旧基态没有 QN 分块

`AKLT_GS.jl` 已支持 `conserve_qns=true`，但本次加载的历史 D=100 基态没有 QN block，日志会明确警告。因此本次实验没有得到 QN 的内存收益。若后续正式扩展 N，应重新 DMRG 生成 QN 基态，不要继续依赖旧非 QN checkpoint。

### 3. 正式实验尚未确定的参数

尚未完成：传播器修复后的时间步验证、长时间尺度、`Dmax` 收敛、QN 基态基准、以及 256–512 样本收敛。规划下一步时应先解决传播器，再决定扩大 `N`、`T` 或样本数；不要直接照搬当前批处理参数启动昂贵任务。

## 后续代理的最小检查清单

1. 开始前运行 `git status`，并检查服务器是否已有同参数结果或正在运行的任务。
2. 不覆盖服务器上用户的 `AKLT_GS.jl` 修改。
3. 修改量子轨迹算法时同步更新 `QUANTUM_TRAJECTORY_ALGORITHM.md`。
4. 先做单轨迹、小 `T`、强制发生跳跃后的数值检查，再提交批量任务。
5. 检查 `p_jump <= 1`、`branch_norm`、负概率警告、最大键维、计时和峰值内存。
6. 样本合并按 trajectory id，不能简单平均各作业的 stderr。
7. 任何展示图必须保留 `t=0` benchmark 和误差棒，并明确样本数、`dt`、`Dmax` 与是否收敛。
