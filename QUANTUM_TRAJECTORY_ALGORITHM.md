# AKLT Lindblad 演化中的量子轨迹算法

本文解释 `AKLT_evolution.jl` 使用的 Monte Carlo wave-function（MCWF，也称 quantum-jump 或 quantum-trajectory）算法，包括它如何从 Lindblad 方程导出、代码如何实现、近似误差来自哪里，以及它与本项目原有 string-order 拟合的关系。

## 1. 为什么从 MPDO 改成量子轨迹

旧程序直接把密度矩阵写成 MPO：

$$
\rho(t)=\text{MPDO}.
$$

一个离散时间步需要计算 no-jump 分支和全部 jump 分支，再把它们求和：

$$
\rho(t+\Delta t)
=K_0\rho K_0^\dagger
+\sum_{\mu=1}^{N_{\rm ch}}K_\mu\rho K_\mu^\dagger.
$$

本模型共有 $N_{\rm ch}=16N$ 个跳跃通道。对于 $N=28$，就是 448 个分支。每个分支都是一次 MPO–MPO 作用，求和后还会增加 operator-space entanglement，因此时间和内存都很容易失控。

量子轨迹不显式保存 $\rho$，而是保存许多纯态 MPS：

$$
|\psi^{(1)}(t)\rangle,\ldots,|\psi^{(M)}(t)\rangle.
$$

密度矩阵和观测量由轨迹平均恢复：

$$
\rho(t)\approx \frac{1}{M}\sum_{m=1}^{M}
|\psi^{(m)}(t)\rangle\langle\psi^{(m)}(t)|,
$$

$$
\langle O\rangle_t
\approx \frac{1}{M}\sum_{m=1}^{M}
\langle\psi^{(m)}(t)|O|\psi^{(m)}(t)\rangle.
$$

每条轨迹的一个时间步只走一个随机分支，所以不再同时生成、保存和求和全部 MPDO 分支。代价是结果带有统计误差，需要通过增加轨迹数收敛。

## 2. Lindblad 方程

项目中的主方程为

$$
\frac{d\rho}{dt}
=-i[H,\rho]
+\sum_\mu\left(
L_\mu\rho L_\mu^\dagger
-\frac12L_\mu^\dagger L_\mu\rho
-\frac12\rho L_\mu^\dagger L_\mu
\right).
$$

其中：

- $H$ 是 `AKLT_GS.jl` 中 `system_ham` 生成的梯子哈密顿量；
- $L_\mu$ 是一个带方向、带自旋的耗散 hopping channel；
- 下标 $\mu$ 同时包含几何键、方向和自旋。

对从 source 站点 $b$ 跳到 target 站点 $a$ 的自旋 $\sigma$ 通道，代码使用

$$
L_{a\leftarrow b,\sigma}
=I_{ab}\,c_{a\sigma}^\dagger c_{b\sigma},
$$

其中 $I_{ab}$ 是 `I1`、`I2`、`IR` 或 `ID`。

注意代码参数 $I$ 是跳跃算符的振幅，所以耗散率中出现的是 $I^2$。

## 3. 一阶 jump 抽样与离散 Kraus 形式

在一阶时间步 $\Delta t$ 下定义

$$
K_0
=1-iH\Delta t
-\frac{\Delta t}{2}\sum_\mu L_\mu^\dagger L_\mu,
$$

$$
K_\mu=\sqrt{\Delta t}\,L_\mu.
$$

展开 no-jump 分支：

$$
\begin{aligned}
K_0\rho K_0^\dagger
&=\left(1-iH\Delta t-\frac{\Delta t}{2}\sum_\mu L_\mu^\dagger L_\mu\right)
\rho
\left(1+iH\Delta t-\frac{\Delta t}{2}\sum_\mu L_\mu^\dagger L_\mu\right)
\\
&=\rho-i\Delta t[H,\rho]
-\frac{\Delta t}{2}\sum_\mu
\left(L_\mu^\dagger L_\mu\rho+\rho L_\mu^\dagger L_\mu\right)
+O(\Delta t^2).
\end{aligned}
$$

每个 jump 分支为

$$
K_\mu\rho K_\mu^\dagger
=\Delta t\,L_\mu\rho L_\mu^\dagger.
$$

将全部分支相加：

$$
K_0\rho K_0^\dagger+\sum_\mu K_\mu\rho K_\mu^\dagger
=\rho+\Delta t\,\mathcal L(\rho)+O(\Delta t^2),
$$

这就是 Lindblad 方程的一阶离散形式。

这段推导仍用于确定每一步的 jump 概率，但当前程序不再把一阶 Euler 的 $K_0$ 作用到 MPS。no-jump 状态改由第 4.2–4.3 节的非厄米有效哈密顿量指数传播；因此当前方法是“一阶 jump 事件抽样 + TDVP no-jump 传播”的离散 MCWF。

## 4. 从 Kraus 分支到随机纯态演化

假设当前状态为归一化纯态 $|\psi\rangle$。第 $\mu$ 个跳跃的概率为

$$
p_\mu
=\langle\psi|K_\mu^\dagger K_\mu|\psi\rangle
=\Delta t\,\langle\psi|L_\mu^\dagger L_\mu|\psi\rangle.
$$

总跳跃概率为

$$
p_{\rm jump}=\sum_\mu p_\mu,
$$

no-jump 概率在一阶 MCWF 中取

$$
p_0=1-p_{\rm jump}.
$$

必须满足

$$
0\le p_{\rm jump}\le1.
$$

若 $p_{\rm jump}>1$，说明 $\Delta t$ 太大，代码会报错并要求减小时间步。

产生一个均匀随机数 $r\in[0,1)$：

- 若 $r\ge p_{\rm jump}$，选择 no-jump；
- 若 $r<p_{\rm jump}$，根据各 $p_\mu$ 所占区间选择一个 jump channel。

条件态分别为

$$
|\psi_0'\rangle
=\frac{e^{-iH_{\rm eff}\Delta t}|\psi\rangle}
{\|e^{-iH_{\rm eff}\Delta t}|\psi\rangle\|},
$$

$$
|\psi_\mu'\rangle
=\frac{K_\mu|\psi\rangle}{\|K_\mu|\psi\rangle\|}.
$$

对随机分支取系综平均，就在 $O(\Delta t)$ 精度上恢复 Lindblad 演化。

### 4.1 为什么仍用 $p_0=1-p_{\rm jump}$ 抽样

有效哈密顿量的指数传播满足

$$
\|e^{-iH_{\rm eff}\Delta t}\psi\|^2
=1-p_{\rm jump}+O(\Delta t^2).
$$

程序在每个离散时间步开头计算

$$
p_0=1-\sum_\mu p_\mu
$$

并且每步至多发生一次 jump。这是一阶 jump-time 离散；实际指数传播后的 no-jump 范数与 $p_0$ 只要求在 $O(\Delta t^2)$ 内一致。代码把实际范数保存为 `branch_weights`，并要求它与 $1-p_{\rm jump}$ 的绝对差不超过 `0.01`。该阈值是防止传播器明显失效的保护条件，不是时间步已经收敛的证明。

### 4.2 有效哈密顿量与能量平移

把 Lindblad 方程写成

$$
\frac{d\rho}{dt}
=-i\left(H_{\rm eff}\rho-\rho H_{\rm eff}^\dagger\right)
+\sum_\mu L_\mu\rho L_\mu^\dagger,
$$

其中

$$
H_{\rm eff}=H-\frac{i}{2}\sum_\mu L_\mu^\dagger L_\mu.
$$

第一项就是“已知本步没有发生 jump”时的非归一化条件演化。$H_{\rm eff}$ 的反厄米部分使态范数下降，该范数损失与 jump 总概率相对应。

当前代码实际构造

$$
\widetilde H_{\rm eff}=H_{\rm eff}-E_{\rm shift}I,
$$

其中 $E_{\rm shift}$ 是实数，当前取本次运行所加载初态的能量期望值，并在该运行段中保持不变；代码变量 `H_eff` 实际保存的是这里的 $\widetilde H_{\rm eff}$。它只选择能量零点，因为

$$
e^{-i\widetilde H_{\rm eff}\Delta t}
=e^{+iE_{\rm shift}\Delta t}e^{-iH_{\rm eff}\Delta t}.
$$

右侧多出的因子只是全局相位；态的范数、jump 概率、归一化后的观测量和密度矩阵 channel 都不变。从主方程看也有 $[H-E_{\rm shift}I,\rho]=[H,\rho]$。这个结论要求 $E_{\rm shift}$ 为实数且乘完整 Hilbert 空间上的恒等算符；复数平移或非恒等算符会改变物理。

保留该平移的数值动机是把 Hamiltonian 的谱中心移近零，减少 Krylov 指数传播需要处理的无物理意义大相位。它不是 MCWF 正确性所必需；设为零应在 TDVP 容差内给出相同结果。

早期代码对一阶 Euler 算符使用相同想法，但有限步长下

$$
1-i\Delta t(H_{\rm eff}-E_{\rm shift}I)
$$

并不等于全局相位乘以 $1-i\Delta tH_{\rm eff}$，两者从 $O(\Delta t^2)$ 开始不同。一次 jump 改变轨迹能量后，固定初态平移曾导致 no-jump 范数约为 `1.24`，而一阶期望约为 `0.98`；归一化不能修复这种状态方向误差。这正是生产算法从 Euler 改为指数 TDVP 的原因。

### 4.3 两站点非厄米 TDVP

程序使用 ITensor 的两站点 TDVP 近似

$$
|\widetilde\psi(t+\Delta t)\rangle
=e^{-i\widetilde H_{\rm eff}\Delta t}|\psi(t)\rangle.
$$

调用的关键设置为：

```julia
nojump_state = tdvp(H_eff, -1im * dt, psi;
    nsite=2, maxdim=maxdim, cutoff=cutoff, normalize=false,
    updater_kwargs=(;
        ishermitian=false, tol=1e-8,
        krylovdim=15, maxiter=30, eager=true))
```

- `nsite=2` 允许 no-jump 演化增加 MPS 键维；单站点 TDVP 不能增加已有键空间；
- `ishermitian=false` 是因为 $H_{\rm eff}$ 含反厄米耗散项；
- `normalize=false` 保留未归一化范数，供生存概率诊断使用；
- `maxdim` 和 `cutoff` 控制两站点分裂时的 MPS 截断；
- Krylov 参数控制局域非厄米指数作用的精度。

取得 `branch_weights` 后才对 no-jump 态归一化。TDVP 消除了旧 Euler 中由广延 Hamiltonian 能量造成的大型多项式截断误差，但不会消除有限 `dt` 的 jump-time 离散误差、MPS 截断误差或 TDVP/Krylov 投影误差，所以仍必须做跳跃后的时间步和键维收敛检查。

## 5. 本模型为什么有 $16N$ 个 channel

系统有 $2N$ 个 Electron site，组成两腿梯子。几何键数为：

| 类型 | 键数 |
|---|---:|
| 第一条腿 | $N$ |
| 第二条腿 | $N$ |
| rung | $N$ |
| diagonal | $N$ |
| 合计 | $4N$ |

每条几何键又有：

- 两个方向：$a\leftarrow b$ 与 $b\leftarrow a$；
- 两种自旋：$\uparrow$ 与 $\downarrow$。

所以

$$
N_{\rm ch}=4N\times2\times2=16N.
$$

例如：

- $N=10$：160 个 channel；
- $N=28$：448 个 channel；
- $N=40$：640 个 channel。

这 160 个 channel 在物理上没有被合并。新程序只是不再提前构造并保存 160 个 MPO；`JumpChannel` 只保存 target、source、自旋算符名、强度和标签。

对应代码：

```julia
struct JumpChannel
    target::Int
    source::Int
    create_op::String
    destroy_op::String
    number_op::String
    rate::Float64
    label::String
end
```

`create_jump_channels` 调用 `add_bond_channels!`，为每条几何键生成四个 channel。

## 6. $H_{\rm eff}$ 中 $\sum L_\mu^\dagger L_\mu$ 的严格化简

对

$$
L_{a\leftarrow b}=I c_a^\dagger c_b,
$$

有

$$
\begin{aligned}
L_{a\leftarrow b}^\dagger L_{a\leftarrow b}
&=I^2c_b^\dagger c_a c_a^\dagger c_b\\
&=I^2c_b^\dagger(1-n_a)c_b\\
&=I^2n_b(1-n_a).
\end{aligned}
$$

反方向满足

$$
L_{b\leftarrow a}^\dagger L_{b\leftarrow a}
=I^2n_a(1-n_b).
$$

两个方向之和为

$$
I^2\left(n_a+n_b-2n_an_b\right).
$$

该等式只用于构造 $H_{\rm eff}$ 中本来就要求和的 $\sum L_\mu^\dagger L_\mu$。jump 分支中的两个方向仍然是独立随机事件。

代码把 Hamiltonian、固定的实数能量平移和化简后的耗散项一次性放进同一个 `OpSum`：

```julia
os = hamiltonian + (-energy_shift, "Id", 1)

os += -0.5im * coefficient, number_op, a
os += -0.5im * coefficient, number_op, b
os += 1im * coefficient, number_op, a, number_op, b
```

最后只调用一次：

```julia
H_eff = MPO(os, sites)
```

这避免了逐个构造 $L_\mu^\dagger L_\mu$ MPO，再做通用 MPO 加法和分解。`coefficient` 在这里是跳跃振幅的平方，不包含 `dt`；时间步由随后 `tdvp(H_eff, -1im * dt, ...)` 的传播时间给出。

`create_effective_hamiltonian` 依赖每条键的四个 channel 顺序，因此代码显式断言：

- up 正反方向互换；
- down 正反方向互换；
- up/down 使用同一 target/source；
- 四个 channel 强度相同。

未来若修改 channel 排列，错误会立即暴露，而不会静默生成错误的 $H_{\rm eff}$。

## 7. 不逐个作用 jump MPO，直接计算全部概率

直接计算

$$
p_\mu=\|\sqrt{\Delta t}L_\mu\psi\|^2
$$

需要为全部 $16N$ 个 channel 做 MPO–MPS 作用，仍然太慢。

利用第 6 节的恒等式：

$$
p_{a\leftarrow b,\sigma}
=\Delta t\,I_{ab}^2
\left(
\langle n_{b\sigma}\rangle
-\langle n_{b\sigma}n_{a\sigma}\rangle
\right).
$$

代码只计算两张关联矩阵：

```julia
correlations = Dict(
    "Nup" => correlation_matrix(psi, "Nup", "Nup"),
    "Ndn" => correlation_matrix(psi, "Ndn", "Ndn"),
)
```

然后从矩阵中读取每个 channel 所需的 occupation 和 joint occupation：

```julia
occupation = real(corr[source, source])
joint_occupation = real(corr[source, target])
probability = dt * rate^2 * (occupation - joint_occupation)
```

由于 MPS 截断和浮点误差可能产生很小的负数，代码将负概率 clip 到 0；若负值小于 `-1e-9`，会输出警告。

严格为零的 channel 不会被选择。选择循环使用严格不等号，并在没有找到 channel 时显式报错，避免落入错误的最后一个 channel。

## 8. 单个时间步在代码中的流程

`trajectory_step` 的逻辑可写成：

```text
输入：归一化 MPS ψ、H_eff、channel 元数据、随机数发生器

1. 用 Nup/Ndn 关联矩阵计算所有 pμ
2. pjump = sum(pμ)
3. 检查 0 ≤ pjump ≤ 1
4. 产生一个随机数 r
5. 若 r ≥ pjump：
       ψ ← TDVP(H_eff, -i dt) ψ
       记录未归一化 no-jump 范数
       检查该范数与 1-pjump 的差不超过 0.01
       normalize!(ψ)
   否则：
       按累计概率选择唯一 channel μ
       临时构造 Kμ = sqrt(dt)Lμ
       ψ ← Kμ ψ
       检查显式范数与解析 pμ 是否一致
       normalize!(ψ)
6. 返回新 ψ、事件编号、pjump 和分支范数
```

对应的核心代码是：

```julia
jump_weights = jump_probabilities(psi, dt, channels)
total_jump_probability = sum(jump_weights)
draw = rand(rng)

if draw >= total_jump_probability
    nojump_state = tdvp(H_eff, -1im * dt, psi;
        nsite=2, maxdim=maxdim, cutoff=cutoff, normalize=false,
        updater_kwargs=(;
            ishermitian=false, tol=1e-8,
            krylovdim=15, maxiter=30, eager=true))
    nojump_weight = real(inner(nojump_state, nojump_state))
    abs(nojump_weight - (1 - total_jump_probability)) <= 0.01 || error(...)
    normalize!(nojump_state)
else
    # 按累计 pμ 选择一个 channel
    jump_state = apply(create_jump_operator(...), psi; ...)
    normalize!(jump_state)
end
```

每一步恰好只调用一次 `rand(rng)`。断点续算的随机流恢复依赖这个约定。jump 分支仍在时间步起点按一阶概率选择，并且每步至多一次 jump；TDVP 只替换 no-jump 状态传播，不会把整个离散 MCWF 自动提升为高阶 jump 算法。

## 9. string-order observable 的测量

项目只需要 odd 和 even 两个 string-order observable。对一条纯态轨迹，直接计算

$$
C_{\rm odd}^{(m)}(t)
=-\langle\psi^{(m)}(t)|O_{\rm odd}|\psi^{(m)}(t)\rangle,
$$

$$
C_{\rm even}^{(m)}(t)
=-\langle\psi^{(m)}(t)|O_{\rm even}|\psi^{(m)}(t)\rangle.
$$

代码先把 head、body 和 tail 组合成一次构造的 SO MPO，然后使用 MPS–MPO–MPS 直接收缩：

```julia
C_odd = -inner(psi', SO_odd, psi)
C_even = -inner(psi', SO_even, psi)
```

这样不会为了测量再生成一个 `SO * psi` 的长期中间态。

程序启动时仍保留独立检查：

- 一条路径使用完整 SO MPO 直接收缩；
- 一条路径依次 apply head/body/tail；
- apply 路径作为高精度参考，不截断局域 head/body 的中间结果，只在 tail 使用不低于初态键维的 `maxdim`；
- 若结果不满足 `rtol=1e-5, atol=1e-7`，程序停止；实际差值仍会完整写入日志。

旧 MPDO 程序也保留初态的 `apply` 测量，并与 MPO–MPO 直接收缩比较。

## 10. 轨迹平均与统计误差

若共有 $M$ 条轨迹，程序保存样本均值：

$$
\overline C(t)=\frac1M\sum_{m=1}^{M}C^{(m)}(t).
$$

无偏样本方差为

$$
s^2(t)=\frac{1}{M-1}
\sum_{m=1}^{M}\left(C^{(m)}(t)-\overline C(t)\right)^2.
$$

均值标准误差为

$$
\operatorname{SE}[\overline C(t)]
=\frac{s(t)}{\sqrt M}.
$$

结果文件中的对应字段为：

- `C_odd_samples`、`C_even_samples`：每条轨迹的原始样本；
- `SO_odd_mean`、`SO_even_mean`：轨迹均值；
- `SO_odd_stderr`、`SO_even_stderr`：均值标准误差；
- `jump_indices`：0 表示 no-jump，其余值表示 channel 编号；
- `total_jump_probabilities`：每一步的 $p_{\rm jump}$；
- `branch_weights`：所选分支归一化之前的范数平方；
- `bond_dimensions`：每条轨迹在每个测量时刻的 MPS 最大键维；
- `measurement_seconds`、`evolution_seconds`、`checkpoint_seconds`：分段计时。

当 $M=1$ 时无法估计样本方差，所以标准误差保存为 `NaN`。

统计误差通常按

$$
\operatorname{SE}\propto M^{-1/2}
$$

下降。把误差缩小 10 倍通常需要约 100 倍轨迹数。

### 10.1 它是不是 MCMC，是否需要 thinning

单条量子轨迹沿时间确实是 Markov 随机过程：下一步只依赖当前 MPS 和一个新随机数。但本程序对固定时刻 $t$ 的估计量不是从一条平衡态 MCMC 链连续取样，而是从相同初态出发、使用不同 `trajectory_id` 和随机流的多条独立轨迹取样。因此：

- 固定时刻不同轨迹之间不需要 burn-in 或 thinning；
- 同一条轨迹在相邻时间的观测值当然相关，但这些点分别估计不同物理时刻，不要求彼此独立；
- 只有将来试图用“一条很长的稳态轨迹的时间平均”代替轨迹系综平均时，才必须估计积分自相关时间，并据此选择采样间隔和有效样本数。

收敛实验使用独立轨迹，并比较 $M=4,8,16,32,64,\ldots$ 的嵌套样本均值和标准误差，不通过跳过时间点制造表面上的独立样本。

## 11. 初始化、加载和 checkpoint

新版保留原来的主要命令行参数，并新增：

| 参数 | 含义 |
|---|---|
| `--ntraj` | 当前作业顺序计算的轨迹数 |
| `--traj-start` | 当前作业第一条轨迹的全局编号 |
| `--seed` | 基础随机种子 |
| `--cutoff` | MPS 截断阈值 |
| `--save-traj` | 是否保存每条轨迹末态 MPS |
| `--measure-every` | 每隔多少个演化步测量一次 SO；不改变实际演化步长 |

三种初始化模式为：

1. `load=true, loadsl=false`：加载已有基态 MPS；
2. `load=false, loadsl=false`：重新 DMRG 并从新基态开始；
3. `load=true, loadsl=true`：加载新版量子轨迹 checkpoint。

旧程序保存的是混态 MPDO slice。一般混态不能唯一还原成一条纯态轨迹，所以新版 `loadsl=true` 不能加载旧 MPDO slice。若需要继续旧 MPDO 演化，应运行 `AKLT_evolution_old.jl`。

checkpoint 的随机数恢复方式为：

```julia
rng = MersenneTwister(seed + trajectory_id - 1)
for _ in 1:completed_steps
    rand(rng)
end
```

由于每步只消耗一个随机数，续算得到的后续随机序列与不中断运行一致。

## 12. QN 守恒

新生成的 Electron site 使用：

```julia
siteinds("Electron", 2N; conserve_qns=true)
```

系统 Hamiltonian 和 hopping jump 都保持总粒子数与总 $S_z$，因此可以利用 QN block-sparse 张量降低内存和计算量。

旧的非 QN 基态文件仍可加载，但它的 Index 本身没有 QN 信息，加载后不会自动变成 block-sparse。程序会输出警告。要获得 QN 优势，必须用 `load=false` 重新生成基态。

## 13. Slurm 并行方式

不同轨迹彼此独立，最自然的并行方式是多个 Slurm 作业，而不是在一个进程里同时保存多个大 MPS。

`batch_AKLT_evol_sub.sh` 中：

- `ntraj`：每个作业内顺序计算多少条轨迹；
- `njobs`：提交多少个独立作业；
- `traj_start`：第一批的起始轨迹编号。

第 `job_index` 个作业使用

$$
\text{this\_traj\_start}
=\text{traj\_start}+\text{job\_index}\times\text{ntraj}.
$$

因此不同作业不会使用相同的轨迹编号或随机流。

`sub_evol.sh` 将完整 stdout/stderr 同时写入：

- Slurm 的 `sbatches/out`、`sbatches/err`；
- 项目 `log/AKLT_evol_...log`。

日志开头记录 job ID、节点名和开始时间。

## 14. 主要误差来源与收敛检查

量子轨迹结果至少有五类误差。

### 14.1 时间离散误差

当前 no-jump 使用双站点非厄米 TDVP 近似指数传播，不再使用一阶 Euler $K_0$。时间离散误差主要有三部分：

1. jump 概率在时间步起点按 $p_\mu=\Delta t\langle L_\mu^\dagger L_\mu\rangle$ 计算，并且每步至多发生一次 jump；
2. TDVP 将非厄米演化投影到有限键维 MPS 流形；
3. 局域 Krylov 指数作用和两站点分裂带来容差及截断误差。

因此，指数 TDVP 消除了旧 Euler 的大型 Hamiltonian 多项式误差，但整个离散 MCWF 仍不是有限 `dt` 下的精确 Lindblad channel。`branch_weights` 检查只能发现明显异常，不能替代收敛测试。

至少比较：

$$
\Delta t,\quad \Delta t/2,\quad \Delta t/4.
$$

比较时应使用相同物理总时间，而不是相同步数。

### 14.2 轨迹统计误差

至少比较不同 $M$，例如：

$$
M=16,32,64,128,\ldots
$$

直到两个 SO 的误差条小于要区分的物理信号。

### 14.3 MPS 截断误差

比较不同 `Dmax` 和 `cutoff`，例如：

$$
D=100,200,400,
$$

以及

$$
\text{cutoff}=10^{-7},10^{-8},10^{-9}.
$$

### 14.4 初态误差

初始 DMRG 的能量、键维和 QN 设置必须收敛。非 QN 旧基态可用于兼容测试，但不适合评估 QN 版本的最终资源占用。

### 14.5 有限尺寸误差

最终物理结论仍需对 $N\to\infty$ 外推，而不能由单个 $N$ 的轨迹结果决定。

## 15. 与历史 notebook 中基态拟合的关系

仓库当前保存的是 `plot.ipynb`，没有名为 `colab_plot.ipynb` 的文件；若本地或 Colab 中另有这个名字，它应是仓库 notebook 的外部副本。历史 notebook 的 “1D SPT project” 分析包含以下工作：

- 对 $U=0,10,32,100,316,1000$ 绘制 odd/even SO 随 $t_R/t_D$ 的变化；
- 使用过 $N=10,14,20,28,40$；
- 临界附近使用 $t_D/t_R=0.98,0.99,0.995,0.998,0.999,1,1.001,1.002,1.005,1.01,1.02$；
- 对 $U=10$ 使用 $\Delta=0.04,0.02,0.01,0.004$ 的中心差分；
- 先对有限差分斜率做 $\Delta\to0$ 外推，再做 $1/N\to0$ 外推。

中心差分可写成

$$
s_O(N,U,\Delta)
=\frac{O(1+\Delta/2)-O(1-\Delta/2)}{\Delta},
$$

其中 $O$ 可以是 odd 或 even string order。

notebook 当前主要拟合逆斜率：

$$
\frac{1}{s_O(N,U,\Delta)}
=a_N+b_N\Delta,
$$

先由 $a_N$ 得到 $\Delta\to0$ 的有限尺寸斜率，再拟合

$$
a_N=A+B/N,
$$

最后取 $1/A$ 作为热力学极限斜率估计。

这个方法在确定性 DMRG 数据上可以作为探索性指标，但用于量子轨迹数据时需要格外小心：

1. 当斜率接近 0 时，取倒数会强烈放大统计噪声；
2. odd/even 斜率符号相反，跨零时逆斜率可能发散；
3. 不同 $\Delta$ 使用独立轨迹时，差分方差会叠加；
4. 再做一次 $1/N$ 外推会继续放大前一级拟合误差。

对演化数据，更稳妥的顺序是：

1. 对每个 $(U,N,t_D/t_R,t)$ 先得到轨迹均值和标准误；
2. 检查 $M$、$D$、`cutoff`、$\Delta t$ 收敛；
3. 使用带误差权重的中心差分或局域多项式，先拟合斜率本身而不是立即取倒数；
4. 做 $\Delta\to0$ 外推并保留协方差；
5. 最后做 $1/N\to0$ 外推；
6. 比较不同 $U$ 下斜率的量级和置信区间。

“无相变区域斜率比有相变区域小约两个数量级”可以作为当前数值观察，但 $U\approx1.7$ 应继续视为待收敛验证的估计，而不是预先固定的临界值。

若要降低中心差分的轨迹噪声，可以让 $1-\Delta/2$ 和 $1+\Delta/2$ 两侧使用相同的轨迹编号与基础随机种子，即 common random numbers。两侧随机涨落可能部分抵消，但必须通过重复种子组验证误差估计没有被低估。

## 16. 当前实现没有做的事情

当前版本有意没有加入：

- 连续时间 waiting-time MCWF；
- 在单个离散步内抽取 jump 的精确发生时刻；
- 自适应 `dt` 或自动 TDVP 局部误差控制；
- 对演化结果自动执行 $\Delta\to0$ 和 $N\to\infty$ 拟合；
- 在单个 Julia 进程中并行保存多条大 MPS。

当前选择是边界清楚的一阶 jump 抽样配合稳定的 TDVP no-jump 传播。更高阶或连续时间算法只有在现有 `dt` 收敛测试无法以合理成本达到目标误差时才需要加入。

## 17. 已完成的验证

本地 $N=4$ QN 回归测试覆盖：

- channel 总数；
- 每个非零 jump channel 的解析概率与显式 MPO 范数；
- 化简后的 $H_{\rm eff}$ 与一阶展开的一致性；
- 总跳跃概率和归一化；
- 一步随机演化；
- 显式跳跃后的 TDVP no-jump 生存范数；
- 跳跃后固定总时间的 $\Delta t$ 与 $\Delta t/2$ 态重合；
- 非零 SO 状态上的 MPS 直接测量、MPS apply、MPDO apply 和 MPO–MPO inner。

最终结果为 76 项全部通过。

还完成了真实 $N=10,D=100$、非 QN 基态上的传播器检查：

- `dt=0.025` 的单站点和双站点 TDVP 单步分别约为 111 秒和 242 秒；
- 两者的 no-jump 范数差约 $5\times10^{-6}$；
- 双站点方案虽然更慢，但允许演化产生新的键维，因此用于生产计算；
- 强制 jump 后比较 `dt=0.05,0.025,0.0125`，最大生存概率误差依次约为 $2.70\times10^{-4}$、$6.74\times10^{-5}$、$1.68\times10^{-5}$；
- `dt=0.025` 与 `0.0125` 的最终 odd/even SO 差都约为 $1.8\times10^{-7}$，态 infidelity 约为 $1.8\times10^{-6}$。

这些结果支持当前 `dt=0.025` 的短时、跳跃后传播精度，但不能替代长时间、不同 `Dmax`、不同系统尺寸和轨迹样本数的独立收敛检查。

## 18. 生产计算的验收顺序

正式结果按以下顺序验收：

1. 用强制发生 jump 的轨迹比较 $\Delta t,\Delta t/2,\Delta t/4$；
2. 检查每步 `p_jump <= 1`、no-jump 范数、负概率、NaN 和日志完整性；
3. 比较不同 `Dmax`、`cutoff`，并记录实际最大 MPS 键维；
4. 逐级增加独立轨迹数，检查均值变化和标准误，而不只看误差棒外观；
5. 确认目标时间上的 SO 或临界点附近斜率已与初态产生统计显著差异；
6. 再扩展系统尺寸，并最终接入 $\Delta\to0$、$N\to\infty$ 分析。

## 19. 当前算法与前两种实现的区别

| 实现 | 保存的状态 | no-jump 传播 | 每步处理的分支 | 主要问题或代价 |
|---|---|---|---:|---|
| 旧 MPDO/Kraus | 密度矩阵 MPO | 一阶 Kraus | 全部 $1+16N$ 个分支并求和 | operator-space 键维和内存快速增长 |
| 早期量子轨迹 | 纯态 MPS | 一阶 Euler $K_0$ | 随机选择一个分支 | 跳跃后固定能量平移可产生很大的有限步长状态误差 |
| 当前量子轨迹 | 纯态 MPS | 双站点非厄米 TDVP 指数传播 | 随机选择一个分支 | 仍有一阶 jump-time 离散、TDVP/MPS 截断和统计误差 |

三种实现目标都是同一个 Lindblad 方程。当前方案没有改变 $16N$ 个物理 jump channel，也没有把不同方向或自旋合并；它只改变了状态表示、每步选择分支的方式，以及 no-jump 条件态的数值传播器。相对于早期量子轨迹，最关键的改变是用

$$
e^{-iH_{\rm eff}\Delta t}|\psi\rangle
$$

替代一阶多项式 $K_0|\psi\rangle$。因此实数 $E_{\rm shift}I$ 恢复为严格的全局相位规范，而不再通过有限阶多项式影响状态方向。

## 20. N=10、U=10、T=4 的生产结果

传播器修复后完成了五个近临界参数点的长时间实验：

```text
tD = 0.98, 0.99, 1.0, 1.01, 1.02
tR = 1
dt = 0.025, tsmax = 160, T = 4
measure_every = 4
Dload = Dmax = 100
seed = 260903
M = 256 trajectories per tD
```

中心斜率从初态的 odd `-0.9565283970`、even `+0.9714583645`，演化到 `T=4` 的

```text
odd  = -0.4747839721 ± 0.0261204430
even = +0.4952725948 ± 0.0277745704
```

相对初态的变化分别约为 `18.4` 和 `17.1` 个 `T=4` 标准误，已经满足“演化态与初始基态的中心斜率存在可见差异”的目标。`M=128 -> 256` 时，`T=4` 五点均值的最大变化为 odd `5.11e-4`、even `1.71e-4`；但若遍历所有记录时刻，最大变化仍为 odd `2.27e-3`、even `1.70e-3`，因此只能把 `M=256` 视为足够回答当前定性问题，不能宣称整条时间曲线严格统计收敛。

所有轨迹记录的最大键维都是 `100`，且加载的历史初态本身就是非 QN、最大键维 `100` 的 MPS，所以仍不能排除 `Dmax` 截断误差。五个参数点在 `T=4` 的平均累计 jump 数为 `1.771875`。

完整结果和说明位于：

- `experiment_results/trajectory_N10_U10_T4_M256_tdvp2.csv`
- `experiment_results/trajectory_N10_U10_T4_M256_tdvp2_slopes.csv`
- `experiment_results/trajectory_N10_U10_T4_M256_tdvp2.svg`
- `experiment_results/trajectory_N10_U10_T4_M256_tdvp2.png`
- `experiment_results/trajectory_N10_U10_T4_M256_tdvp2_report.md`

## 参考资料

- J. Dalibard, Y. Castin, and K. Mølmer, *Wave-function approach to dissipative processes in quantum optics*, Physical Review Letters **68**, 580 (1992).
- R. Dum, P. Zoller, and H. Ritsch, *Monte Carlo simulation of the atomic master equation for spontaneous emission*, Physical Review A **45**, 4879 (1992).
- H. J. Carmichael, *An Open Systems Approach to Quantum Optics*, Springer (1993).
- ITensor MPS/MPO documentation: <https://docs.itensor.org/ITensorMPS/stable/MPSandMPO.html>
- ITensor Electron site documentation: <https://docs.itensor.org/ITensorMPS/stable/IncludedSiteTypes.html>
