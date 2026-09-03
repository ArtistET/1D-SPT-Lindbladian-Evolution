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

## 3. 从 Lindblad 方程到离散 Kraus 形式

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
=\frac{K_0|\psi\rangle}{\|K_0|\psi\rangle\|},
$$

$$
|\psi_\mu'\rangle
=\frac{K_\mu|\psi\rangle}{\|K_\mu|\psi\rangle\|}.
$$

对随机分支取系综平均，就在 $O(\Delta t)$ 精度上恢复 Lindblad 演化。

### 4.1 为什么不直接把 $\|K_0\psi\|^2$ 当作抽样概率

形式上有

$$
\|K_0\psi\|^2
=1-p_{\rm jump}+O(\Delta t^2).
$$

但 Euler 形式的 $K_0=1-iH\Delta t+\cdots$ 含有

$$
\Delta t^2\langle H^2\rangle.
$$

对广延哈密顿量，体系越大，$\langle H^2\rangle$ 越大。在本项目的 $N=10,D=100,\Delta t=0.01$ 基准中，直接计算的 Kraus 总范数约为 2.108，而不是接近 1。它主要是有限步长的 Hamiltonian Euler 误差，不能用来重新缩放物理跳跃率。

因此代码直接用一阶正确的

$$
p_0=1-\sum_\mu p_\mu
$$

抽样，同时把实际分支范数保存为 `branch_weights`，作为时间步和截断误差诊断，而不是把它当概率。

### 4.2 为什么在 $K_0$ 中平移初态能量

连续时间演化允许给哈密顿量减去任意实常数：

$$
H\longrightarrow H-E_{\rm ref}I.
$$

它只给未归一化 no-jump 状态增加整体相位，归一化后的物理状态和所有观测量不变。但是一阶 Euler 近似并不精确保持这种等价性；广延的基态能量会使无物理意义的 $E_{\rm ref}^2\Delta t^2$ 项变大，并相对压低耗散修正。

程序因此取

$$
E_{\rm ref}=\langle\psi(0)|H|\psi(0)\rangle
$$

并实际构造

$$
K_0=1-i\Delta t(H-E_{\rm ref}I)-\frac{\Delta t}{2}\sum_\mu L_\mu^\dagger L_\mu.
$$

这不替代时间步收敛检查，但对从基态出发的短时间演化可显著减小 Euler 误差。`energy_shift` 会写入运行日志；续算时它由所加载的第一条轨迹状态重新估计。

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

## 6. $K_0$ 中 $\sum L_\mu^\dagger L_\mu$ 的严格化简

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

该等式只用于构造 $K_0$ 中本来就要求和的 $\sum L_\mu^\dagger L_\mu$。jump 分支中的两个方向仍然是独立随机事件。

代码把 Hamiltonian、单位算符和化简后的耗散项一次性放进同一个 `OpSum`：

```julia
os = (-1im * dt) * hamiltonian + (1.0, "Id", 1)

os += -0.5 * coefficient, number_op, a
os += -0.5 * coefficient, number_op, b
os += coefficient, number_op, a, number_op, b
```

最后只调用一次：

```julia
K0 = MPO(os, sites)
```

这避免了把已经构造好的多个 MPO 再做通用 MPO 加法和分解。曙光上的 $N=10$ 测试中，旧路径运行约 13 分 51 秒仍未构造完成；单 OpSum 路径约 25.8 秒完成。

`create_nojump_operator` 依赖每条键的四个 channel 顺序，因此代码现在显式断言：

- up 正反方向互换；
- down 正反方向互换；
- up/down 使用同一 target/source；
- 四个 channel 强度相同。

未来若修改 channel 排列，错误会立即暴露，而不会静默生成错误的 $K_0$。

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
输入：归一化 MPS ψ、K0、channel 元数据、随机数发生器

1. 用 Nup/Ndn 关联矩阵计算所有 pμ
2. pjump = sum(pμ)
3. 检查 0 ≤ pjump ≤ 1
4. 产生一个随机数 r
5. 若 r ≥ pjump：
       ψ ← K0 ψ
       记录未归一化分支范数
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
    nojump_state = apply(K0, psi; cutoff=cutoff, maxdim=maxdim)
    normalize!(nojump_state)
else
    # 按累计 pμ 选择一个 channel
    jump_state = apply(create_jump_operator(...), psi; ...)
    normalize!(jump_state)
end
```

每一步恰好只调用一次 `rand(rng)`。断点续算的随机流恢复依赖这个约定。

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
- 两条路径现在使用相同的 `cutoff` 和 `maxdim`；
- 若结果不满足 `rtol=1e-6, atol=1e-9`，程序停止。

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

## 11. 初始化、加载和 checkpoint

新版保留原来的主要命令行参数，并新增：

| 参数 | 含义 |
|---|---|
| `--ntraj` | 当前作业顺序计算的轨迹数 |
| `--traj-start` | 当前作业第一条轨迹的全局编号 |
| `--seed` | 基础随机种子 |
| `--cutoff` | MPS 截断阈值 |
| `--save-traj` | 是否保存每条轨迹末态 MPS |

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

当前 no-jump 使用一阶 Euler $K_0$，单步误差为 $O(\Delta t^2)$，固定总时间下的全局误差通常为 $O(\Delta t)$。

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

## 15. 与 `colab_plot.ipynb` 中基态拟合的关系

notebook 的 “1D SPT project” 已包含以下工作：

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

- 自动合并不同 Slurm 作业产生的结果文件；
- 对演化结果自动执行 $\Delta\to0$ 和 $N\to\infty$ 拟合；
- 用 TDVP 替代 Euler no-jump；
- 在单个 Julia 进程中并行保存多条大 MPS。

原因是这些功能不影响量子轨迹主算法的正确运行，而且应在第一批生产数据确认文件规模、轨迹方差和单步资源后再决定具体形式。

## 17. 已完成的验证

曙光计算节点上的 $N=4$ QN 回归测试覆盖：

- channel 总数；
- 每个非零 jump channel 的解析概率与显式 MPO 范数；
- 化简前后 $K_0$ 作用结果；
- 总跳跃概率和归一化；
- 一步随机演化；
- 非零 SO 状态上的 MPS 直接测量、MPS apply、MPDO apply 和 MPO–MPO inner。

最终结果为 74 项全部通过。

还完成了 $N=10,D=100$ 的旧非 QN 基态基准：

- 基态加载约 8.4 秒；
- $K_0$ 构造约 25.8 秒；
- 两张关联矩阵和全部 jump 概率约 83.2 秒；
- 完整一步约 586.7 秒；
- 峰值内存约 3.32 GB。

这些数字说明新算法已经大幅降低分支数量和固定构造成本，但不能替代 $N=28/40,D=400$ 的 QN 生产前试跑。

## 18. 推荐的第一批生产检查

建议按以下顺序推进，而不是立即提交大量轨迹：

1. 重新生成一个较小尺寸的 QN 基态；
2. 用 `ntraj=1, tsmax=1` 测单步时间和峰值内存；
3. 用同一初态比较 $\Delta t$ 与 $\Delta t/2$；
4. 用 $M=8$ 或 $16$ 估计两个 SO 的轨迹方差；
5. 根据目标误差反推生产轨迹数；
6. 再扩展到 $N=28/40$；
7. 最后把演化数据接入 notebook 的 $\Delta\to0$、$N\to\infty$ 流程。

## 参考资料

- J. Dalibard, Y. Castin, and K. Mølmer, *Wave-function approach to dissipative processes in quantum optics*, Physical Review Letters **68**, 580 (1992).
- R. Dum, P. Zoller, and H. Ritsch, *Monte Carlo simulation of the atomic master equation for spontaneous emission*, Physical Review A **45**, 4879 (1992).
- H. J. Carmichael, *An Open Systems Approach to Quantum Optics*, Springer (1993).
- ITensor MPS/MPO documentation: <https://docs.itensor.org/ITensorMPS/stable/MPSandMPO.html>
- ITensor Electron site documentation: <https://docs.itensor.org/ITensorMPS/stable/IncludedSiteTypes.html>
