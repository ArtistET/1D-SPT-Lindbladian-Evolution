"""Reconcile the five-D paired export and generate the stopping-point report."""
import csv
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiment_results'
DS = (400, 600, 800, 1200, 1600)
TDS = (.98, 1., 1.02)
TIMES = (2, 4, 6, 10, 12)
PREFIX = 'N10_U10_D1600'

def read(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))

def write(name, value):
    p = (OUT / name).resolve()
    assert p.is_relative_to(ROOT)
    p.write_text(value, encoding='utf-8')

def se(v):
    return st.stdev(v) / math.sqrt(len(v))

raw = read(PREFIX + '_trajectory_timecurves.csv')
audit = read(PREFIX + '_file_audit.csv')
pairs = read(PREFIX + '_paired_timecurves.csv')
assert (len(raw), len(audit), len(pairs)) == (7260, 60, 7260)
data = {(int(r['D']), float(r['tD']), int(r['id']), round(float(r['time']), 1)): r for r in raw}
assert len(data) == len(raw)
for d in DS:
    for td in TDS:
        for i in range(1, 5):
            assert all((d, td, i, k / 10) in data for k in range(121))
for r in raw:
    assert all(math.isfinite(float(r[k])) for k in ('odd', 'even', 'bond'))
    assert 1 <= int(r['bond']) <= int(r['D'])

def values(d, td, t, parity):
    return [float(data[d, td, i, t][parity]) for i in range(1, 5)]

def slopes(d, t, parity):
    return [(a-b)/.04 for a,b in zip(values(d,1.02,t,parity), values(d,.98,t,parity))]

indexed = {}
for r in pairs:
    lo, hi, td, t = int(r['Dlo']), int(r['Dhi']), float(r['tD']), round(float(r['time']),1)
    parity = r['parity']
    x, y = values(lo,td,t,parity), values(hi,td,t,parity)
    dif = [b-a for a,b in zip(x,y)]
    for key, val in [('lo_mean',st.mean(x)),('hi_mean',st.mean(y)),('paired_delta',st.mean(dif)),('paired_SE',se(dif))]:
        assert abs(float(r[key])-val) < 1e-14, (r,key,val)
    indexed[lo,hi,td,parity,t] = r
assert len(indexed) == 7260
old = read('N10_U10_D1200_paired_timecurves.csv')
for r in old:
    k = (int(r['Dlo']),int(r['Dhi']),float(r['tD']),r['parity'],round(float(r['time']),1))
    for field in ('lo_mean','hi_mean','paired_delta','paired_SE'):
        assert abs(float(r[field])-float(indexed[k][field])) < 1e-14

slope_lines = ['Dlo,Dhi,parity,time,lo_mean,lo_SE,hi_mean,hi_SE,paired_delta,paired_SE']
for hi in DS:
    for lo in DS:
        if lo >= hi:
            continue
        for parity in ('odd','even'):
            for k in range(121):
                t=k/10
                x,y=slopes(lo,t,parity),slopes(hi,t,parity)
                dif=[b-a for a,b in zip(x,y)]
                slope_lines.append(','.join(map(str,(lo,hi,parity,t,st.mean(x),se(x),st.mean(y),se(y),st.mean(dif),se(dif)))))
write(PREFIX+'_paired_slopes.csv','\n'.join(slope_lines)+'\n')

lines = ['# N=10、U=10 SO 曲线可信度验证：D1600 停止点正式报告',
'','核验日期：2026-11-10（北京时间）。',
'', '**结论：本轮已批准的计算与核验已完成，但传播键维收敛尚未得到证明。** D1600 相对 D1200 的 T12 六项配对 SO 均继续升高，幅度为 0.000359–0.000576，相对 D1600 均值约 2.0%–3.1%；12 条末态最大键维和统一中间切口维数均为1600。按既定硬上限停止增加 D，交付本报告后等待讨论，不自动开始更小步长、更多样本、更长时间或更高初态键维计算。',
'', '## 实验设计与逐文件核验',
'', 'tD=0.98/1.00/1.02，每点 id=1:4；五档传播 D=400/600/800/1200/1600 均加载同一对应参数的 QN D400 初态。N=10，U=10，t1=0.1，t2=0.2，tR=1，J=0，I1=I2=IR=ID=0.1；dt=0.025，480步至T=12，每4步测量一次，共121个0:0.1:12时间点；seed=260903，cutoff=1e-8，Dstep=Dstepload=50。D800初态检查是独立实验，未混入传播对照。',
'', '扩展 `validate_d600_pilot.jl` 支持1600，以远端新文件 `validate_bond_pilot_20261110.jl 1600` 运行，退出码0；旧未跟踪脚本保留。五档共60对结果/checkpoint（120个JLD2）通过参数、完整轨迹数、网格、480步记录、复SO有限性、通道编号0:160、概率、分支权重、键维、20站点QN和范数检查，末态键维与结果记录一致。五档初始SO相符至1e-10。Checkpoint只含保存态，参数核对来自对应结果文件与单轨迹日志。',
'', 'D1600最大范数偏差1.11e-16；跳跃概率范围0.0111372–0.0119716，分支权重范围4.37241e-5–0.988497。12份单轨迹日志均包含顺序且无重复的1:480步、一次最终保存、QN初态和单线程记录，未检出ERROR、NaN、OOM、负概率或生存概率越界错误；原批处理正常结束，无剩余演化进程。D1200/D1600的12×480=5760个随机分支编号完全相同。',
'', '新导出的7260行全时间配对表，逐行用独立Python标准库统计从7260行单轨迹测量表复算均值和配对标准误，差异小于1e-14；低四档的4356行与已发布D1200配对表一致至1e-14。传回的三份原始CSV与远端SHA-256一致。日志留存于 `log/validate_d1600_pilot_20261110.log` 和 `log/validate_d1600_logs_20261110.json`。',
'', '## T12：五档 SO 水平与最后一档变化',
'', '表中SO为复SO的实部系综均值；差值标准误按同id逐条相减后计算 `sd(差值)/sqrt(4)`，不是把两档独立标准误相加；±均为一个标准误。',
'', '| tD | parity | D400 | D600 | D800 | D1200 | D1600 | D1600−D1200 ± SE | 相对D1600 |',
'|---:|:---|---:|---:|---:|---:|---:|---:|---:|']
for td in TDS:
    for parity in ('odd','even'):
        means=[st.mean(values(d,td,12,parity)) for d in DS]
        r=indexed[1200,1600,td,parity,12]
        delta,err=float(r['paired_delta']),float(r['paired_SE'])
        lines.append(f'| {td:.2f} | {parity} | '+' | '.join(f'{v:.8f}' for v in means)+f' | +{delta:.8f} ± {err:.8f} | {delta/means[-1]:.2%} |')
lines += ['', '六项变化方向一致，但因参数点、parity和时间共享随机数，不能作为六个独立检验相乘。相对上一档D1200−D800的0.000526–0.000804，最后一档变化有所减小但仍有约2%–3%的水平漂移；它只是有限档位敏感性，不能当作D→∞残差上界，也不作未经验证的1/D外推。',
'', '## 时间分辨的 SO 与斜率',
'', '下表给出各指定时刻六项D1600−D1200配对均值差的范围；完整121点及全部十种D配对见数据文件。',
'', '| T | 六项差值最小值 | 六项差值最大值 |', '|---:|---:|---:|']
for t in TIMES:
    v=[float(indexed[1200,1600,td,p,t]['paired_delta']) for td in TDS for p in ('odd','even')]
    lines.append(f'| {t} | {min(v):+.9g} | {max(v):+.9g} |')
lines += ['', '本pilot斜率定义为 `[SO(tD=1.02)−SO(tD=0.98)]/0.04`，是对tD的端点差商；旧M64图使用0.99/1.01两点对tR/tD的中心差商，两者不能直接混用。每条id先计算端点差商，再计算系综均值和同id档位差的标准误。',
'', '| T | parity | D400斜率 | D600斜率 | D800斜率 | D1200斜率 | D1600斜率 ± SE | D1600−D1200 ± SE |',
'|---:|:---|---:|---:|---:|---:|---:|---:|']
for t in TIMES:
    for parity in ('odd','even'):
        vv=[slopes(d,t,parity) for d in DS]
        dif=[b-a for a,b in zip(vv[-2],vv[-1])]
        lines.append(f'| {t} | {parity} | '+' | '.join(f'{st.mean(v):+.7f}' for v in vv[:-1])+f' | {st.mean(vv[-1]):+.7f} ± {se(vv[-1]):.7f} | {st.mean(dif):+.7f} ± {se(dif):.7f} |')
lines += ['', 'T12最后一档斜率差odd为+0.000838±0.001698，even为+0.003836±0.002079；M4的差分不显著或较小不构成斜率等价性证明。SO水平仍随D变化，不能仅凭斜率差较小宣布整条曲线收敛。',
'', '## 末态键维与保留 Schmidt 尾部',
'', '全部五档的12条末态均达到对应最大键维；在统一切口10|11正交化后测得如下范围。',
'', '| D | 末态最大键维范围 | 切口10\|11维数范围 | 最小保留奇异值范围 | 最小10个保留值平方和范围 |',
'|---:|---:|---:|---:|---:|']
for d in DS:
    rr=[r for r in audit if int(r['D'])==d]
    def interval(k,fmt):
        v=[float(r[k]) for r in rr]
        return format(min(v),fmt)+'–'+format(max(v),fmt)
    lines.append(f'| {d} | {interval("final_bond",".0f")} | {interval("center_dim",".0f")} | {interval("min_retained_s",".3g")} | {interval("retained_tail10_sq",".3g")} |')
lines += ['', '**保留谱不是丢弃权重。** 这些量描述保存态中仍保留的尾部，不包含被截断的D以上权重，不能给出SO误差上界；达到上限也不单独证明误差大小。与持续的配对SO漂移合看，现有证据不足以确认传播收敛。全时间单轨迹CSV同时保留各时刻实际最大键维，逐文件尾部数值见audit表。',
'', '## 初态、统计、步长与参考线误差',
'', '| 误差来源 | 已有证据 | 尚未解决的限制 |',
'|:---|:---|:---|',
'| 传播键维 | 同一D400初态、同id五档对照，D1600仍有2.0%–3.1%的T12水平变化 | 无D→∞误差界，尚未证明SO或斜率收敛 |',
'| U10初态 | 五点D400/D800共10个态已核验；最大odd/even初态变化9.51e-6/4.74e-5，能量降低1.33e-5–1.52e-5 | D800仍触上限；未固定传播D比较两种初态，无法约束长时初态误差 |',
'| 独立轨迹统计 | D400五点M64已完成；M32→64末态均值最大变化0.000451/0.001083 | 全时间最大变化0.004569/0.003184；高D只有三点M4，不能借用D400的M64精度 |',
'| 时间平台 | D400 M64的T11–12较平缓 | T10→12 odd配对斜率仍变化2.39个SE；不显著不等于平台，尚不能确认稳态或弛豫时间 |',
'| 步长/传播器 | N3精确密度矩阵/TDVP基准82/82 PASS；dt减半改善短时离散误差 | 不能替代N10、T12的dt=0.025与0.0125系综比较 |',
'| U0参考 | 自由费米子相关矩阵行列式给出五点精确SO，中心odd/even=0.020646803789 | 旧D100/D200/D800 DMRG参考不应视为精确；U0基态不是耗散态SO的上下界，无可靠U1五点参考 |',
'', '已完成的D800初态检查详见 [初态报告](N10_U10_QN_D800_initial_state_report.md)，本轮未重复计算。初态差较小只能说明初始SO的档位敏感性，不能据此断言其长时间影响也小。',
'', 'N3基准在一上一下粒子的36维扇区检验跳跃概率、有效哈密顿量和密度矩阵/精确指数；T=0.2时dt减半使初态与强制跳跃后SO离散误差约减半，单步TDVP相对精确指数的态infidelity约3.15e-6；这些验证排除了部分实现错误，但不能给出N10长时误差界。相关源文件为 `test_exact_lindblad_small.jl` 与 `exact_u0_string_order.jl`，此前结果记录于Windows项目AGENTS.md。',
'', 'D400 M64的T12对tR/tD中心斜率odd=+0.010215±0.011735、even=+0.032659±0.011937；T≈11–12仍只是候选平台。旧D100和D400初态不同，二者末态SO差不能全部归因于传播截断。本轮同初态五档试验消除了这项混淆，但未解决有限步长和统计误差。单轨迹复SO并不保证虚部为零，本次有限性核验包含实部与虚部，汇总沿用实部统计约定。',
'', '精确U0参考消除了该参考的MPS键维近似问题，却不能证明动力学应趋向U0基态；曲线交叉本身不构成算法错误证据。N=10有限系统、有限时间结果也不能证明热力学极限下存在或不存在相变。',
'', '## 交付与停止决定',
'', 'D1600传播与U10 D800初态两批均已完成并处理；此次停止是遵守本轮D1600硬上限，并非判定已经收敛。下一步的精度目标、固定传播D的初态对照、步长与样本验证须等待用户讨论决定；本轮不启动D>1600、dt0.0125、M128/256、T14/16或额外初态计算。报告提交推送、Windows AGENTS同步后删除n10-so监控。',
'', '- [全时间SO配对表：7260行](N10_U10_D1600_paired_timecurves.csv)',
'- [单轨迹时间与键维：7260行](N10_U10_D1600_trajectory_timecurves.csv)',
'- [全时间斜率配对表：2420行](N10_U10_D1600_paired_slopes.csv)',
'- [逐文件末态与保留尾部：60行](N10_U10_D1600_file_audit.csv)',
'- [前一档D1200报告](N10_U10_D1200_QN_paired_pilot_report.md)',
'', '## 附表：指定时刻的五档SO均值',
'', '| T | tD | parity | D400 | D600 | D800 | D1200 | D1600 |',
'|---:|---:|:---|---:|---:|---:|---:|---:|']
for t in TIMES:
    for td in TDS:
        for parity in ('odd','even'):
            lines.append(f'| {t} | {td:.2f} | {parity} | '+' | '.join(f'{st.mean(values(d,td,t,parity)):.8f}' for d in DS)+' |')
write(PREFIX+'_QN_paired_pilot_report.md','\n'.join(lines)+'\n')
print('PASS: 7260 paired rows independently recomputed; 4356 previous rows unchanged; 60 file audits; 2420 slope rows; report saved.')
