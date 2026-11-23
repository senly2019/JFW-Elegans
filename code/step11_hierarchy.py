"""实验单元层次结构与伪重复诊断（TBD-09）。

实验设计（据作者说明，2026-10-24）
--------------------------------
实验采用**同一培养皿跨多天重复成像**：每个培养皿（= 组别 + 培养架行号）
每天拍摄一个视频，连续拍摄多天。因此：
  · 组(4) × 行(4) = 16 个培养皿，是最接近"独立实验单元"的层级；
  · 每个皿内在各成像日期由 YOLO 检测+跟踪分配 worm 编号，
    **同一只蠕虫在不同日期的编号不同**，故 worm_id 只在单日内唯一；
  · 皿内蠕虫数不等，源于死虫剔除、多虫框交叠、疑似噪声气泡等。

层次结构
--------
    组别（处理，4 水平）
      └─ 培养皿 plate（16 个，处理施加的单位）
           └─ 成像日期 date（皿内重复测量，8 个）
                └─ 蠕虫 worm（单日内编号，约 24 只/皿/日）
                     └─ 帧 frame（测量单元）

本脚本据此完成三件事：
  1. 三层方差分解：组别 / 皿(组内) / 日期(皿内) / 残差，报告各级 η²；
  2. 皿级聚合后的组间比较（每皿取跨日期均值，n=4/组），
     并给出 Welch t 检验与 Cohen's d，说明"以皿为单位"才是可辩护的口径；
  3. 混合效应模型（statsmodels MixedLM），分别以 皿、日期 为随机效应，
     比较随机效应结构对组别固定效应估计的影响。

前置条件
--------
需要 `atlas_worm_hist.csv`（每虫姿态直方图）。该文件由
`cluster_atlas.py` 全量运行产出；若已删除缓存需先重跑：
    python cluster_atlas.py -i $WORM_DATA_ROOT -o atlas_out_full --workers 20
本脚本只读取统计结果，不做图像处理。

用法
----
  python step11_hierarchy.py -i atlas_out_full/atlas_worm_hist.csv \
      -o atlas_out_full/hierarchy_results.csv
"""
import os
import csv
import math
import argparse
from collections import defaultdict

import numpy as np

GROUP_ORDER = ['control', '5mg', '8mg', '10mg']


# ---------------------------------------------------------------- 工具函数
def eta2(vals, factor):
    """单因素方差分解的 η²。factor 与 vals 等长。"""
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    grand = vals.mean()
    ss_tot = ((vals - grand) ** 2).sum()
    groups = defaultdict(list)
    for v, f in zip(vals, factor):
        groups[f].append(v)
    ss_b = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups.values())
    return ss_b / ss_tot if ss_tot > 0 else float('nan')


def cohen_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float('nan'), float('nan'), float('nan')
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    s1 = float(np.std(a, ddof=1))
    s2 = float(np.std(b, ddof=1))
    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled < 1e-12:
        return 0.0, 0.0, 0.0
    d = (m2 - m1) / pooled
    se = math.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2 - 2)))
    return d, d - 1.96 * se, d + 1.96 * se


def welch_t(a, b):
    """Welch t 检验，返回 (t, df, p)。p 用正态近似（大样本下足够）。"""
    n1, n2 = len(a), len(b)
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    if v1 == 0 and v2 == 0:
        return 0.0, n1 + n2 - 2, 1.0
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se < 1e-12:
        return 0.0, n1 + n2 - 2, 1.0
    t = (m2 - m1) / se
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    # 双尾 p，t 分布用正态近似
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return t, df, p


# ---------------------------------------------------------------- 主流程
def load_worm_hist(path):
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows.append(dict(group=r['group'], date=r['date'], row=r['row'],
                             worm=r['worm'], n_frames=int(r['n_frames']),
                             p0=float(r['p0']), p1=float(r['p1'])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', default='atlas_out_full/atlas_worm_hist.csv')
    ap.add_argument('-o', '--out',
                    default='atlas_out_full/hierarchy_results.csv')
    ap.add_argument('--min-frames', type=int, default=10)
    ap.add_argument('--cluster', type=int, default=0,
                    help='分析的簇（0=深弯/Ω，1=浅弯/笔直）')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(
            f'未找到 {args.input}。\n'
            f'该文件由 cluster_atlas.py 全量运行产出；若已删除缓存，请先重跑：\n'
            f'  python cluster_atlas.py -i $WORM_DATA_ROOT '
            f'-o atlas_out_full --workers 20')

    rows = load_worm_hist(args.input)
    rows = [r for r in rows if r['n_frames'] >= args.min_frames]
    for r in rows:
        r['plate'] = f"{r['group']}_{r['row']}"      # 组+行 = 同一培养皿
        r['prop'] = r['p0'] if args.cluster == 0 else r['p1']

    n_worms = len(rows)
    plates = sorted({r['plate'] for r in rows})
    dates = sorted({r['date'] for r in rows})
    print(f'入统计蠕虫: {n_worms:,}  培养皿: {len(plates)}  成像日期: {len(dates)}')
    print(f'培养皿清单: {plates}')
    print(f'  每皿蠕虫数: ' + ', '.join(
        f'{p}={sum(1 for r in rows if r["plate"] == p)}' for p in plates))

    props = [r['prop'] for r in rows]
    groups = [r['group'] for r in rows]
    plate_of = [r['plate'] for r in rows]
    date_of = [r['date'] for r in rows]

    # ---- 1. 三层方差分解 ----
    print('\n=== 方差分解（虫级比例，η²）===')
    e_group = eta2(props, groups)
    # 皿(组内)：去掉组别后的 SS 中由皿解释的部分
    # 用"组内中心化"近似：以组均值为基线计算皿间 SS
    gm = defaultdict(list)
    for v, g in zip(props, groups):
        gm[g].append(v)
    gmean = {g: np.mean(v) for g, v in gm.items()}
    resid_after_group = [v - gmean[g] for v, g in zip(props, groups)]
    pm = defaultdict(list)
    for v, p in zip(resid_after_group, plate_of):
        pm[p].append(v)
    ss_plate = sum(len(v) * np.mean(v) ** 2 for v in pm.values())
    ss_resid_group = sum(v ** 2 for v in resid_after_group)
    e_plate_within = ss_plate / ss_resid_group if ss_resid_group > 0 else float('nan')
    # 日期(皿内)
    dm = defaultdict(list)
    for v, p, d in zip(resid_after_group, plate_of, date_of):
        dm[(p, d)].append(v)
    ss_date = sum(len(v) * np.mean(v) ** 2 for v in dm.values())
    ss_after_plate = sum(v ** 2 for v in resid_after_group) - ss_plate
    e_date_within = ss_date / ss_after_plate if ss_after_plate > 0 else float('nan')
    print(f'  组别        η² = {e_group:.4f}')
    print(f'  皿(组内)    η² = {e_plate_within:.4f}  （占组内残差的比例）')
    print(f'  日期(皿内)  η² = {e_date_within:.4f}  （占去皿后残差的比例）')
    print(f'  蠕虫残差    η² = {1 - e_group:.4f}')
    print(f'  日期/组别 方差比 = '
          f'{(e_date_within * ss_after_plate) / (e_group * ((np.array(props) - np.mean(props)) ** 2).sum()):.2f} 倍'
          if e_group > 0 else '')

    # ---- 2. 皿级聚合的组间比较 ----
    print('\n=== 皿级聚合（每皿跨日期均值，n=4/组）===')
    plate_mean = {}
    for p in plates:
        vals = [r['prop'] for r in rows if r['plate'] == p]
        plate_mean[p] = float(np.mean(vals))
    by_group_plate = defaultdict(list)
    for p in plates:
        g = p.split('_')[0]
        by_group_plate[g].append(plate_mean[p])
    for g in GROUP_ORDER:
        v = by_group_plate[g]
        print(f'  {g:>8s}: n={len(v)}  mean={np.mean(v):.4f}  sd={np.std(v, ddof=1):.4f}'
              f'  皿值={["%.3f" % x for x in v]}')
    print('\n  皿级组间比较（Welch t）：')
    ctrl = by_group_plate['control']
    for g in GROUP_ORDER[1:]:
        t, df, p = welch_t(ctrl, by_group_plate[g])
        d, dlo, dhi = cohen_d(ctrl, by_group_plate[g])
        print(f'    control vs {g:>5s}: t={t:.3f}, df={df:.1f}, p={p:.4g}, '
              f'd={d:.3f} [{dlo:.3f}, {dhi:.3f}]')

    # ---- 3. 混合效应模型 ----
    print('\n=== 混合效应模型（MixedLM）===')
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        df = pd.DataFrame({'prop': props, 'group': groups,
                           'plate': plate_of, 'date': date_of})
        m1 = smf.mixedlm('prop ~ C(group)', df, groups=df['plate']).fit()
        print('  模型 A：prop ~ group + (1|plate)')
        print(f'    组别固定效应（control 为基线）：')
        for name in m1.params.index:
            if name.startswith('C(group)'):
                print(f'      {name}: β={m1.params[name]:+.4f}, '
                      f'p={m1.pvalues[name]:.4g}')
        print(f'    皿随机效应方差 = {m1.cov_re.iloc[0, 0]:.5f}, '
              f'残差方差 = {m1.scale:.5f}')
        m2 = smf.mixedlm('prop ~ C(group)', df,
                         groups=df['plate'],
                         re_formula='1', vc_formula={'date': '0 + C(date)'}).fit()
        print('  模型 B：prop ~ group + (1|plate) + (1|date:plate)')
        for name in m2.params.index:
            if name.startswith('C(group)'):
                print(f'      {name}: β={m2.params[name]:+.4f}, '
                      f'p={m2.pvalues[name]:.4g}')
        print(f'    随机效应方差: {dict(m2.cov_re)}')
        print(f'    残差方差 = {m2.scale:.5f}')
    except ImportError:
        print('  statsmodels 或 pandas 不可用，跳过 MixedLM。')
        print('  请安装: pip install statsmodels pandas')

    # ---- 4. 伪重复警示 ----
    print('\n=== 伪重复诊断 ===')
    print(f'  观测单元（蠕虫）数        : {n_worms:,}')
    print(f'  独立实验单元（培养皿）数  : {len(plates)}')
    print(f'  每组培养皿数              : {len(by_group_plate["control"])}')
    print(f'  名义样本量膨胀倍数        : {n_worms / len(plates):.1f} 倍')
    print('  说明：药物按皿施加，皿是处理施加的单位；若以皿均值为统计单元，')
    print('       每组 n=4，功效极低。论文必须同时报告两个口径，并明确')
    print('       虫级比较存在伪重复、皿级比较功效不足。')

    # ---- 5. 写 CSV ----
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['# 层次结构与伪重复诊断（TBD-09）'])
        w.writerow(['metric', 'value'])
        w.writerow(['n_worms', n_worms])
        w.writerow(['n_plates', len(plates)])
        w.writerow(['n_dates', len(dates)])
        w.writerow(['eta2_group', f'{e_group:.5f}'])
        w.writerow(['eta2_plate_within_group', f'{e_plate_within:.5f}'])
        w.writerow(['eta2_date_within_plate', f'{e_date_within:.5f}'])
        w.writerow([])
        w.writerow(['# 皿级聚合组间比较（n=4/组）'])
        w.writerow(['group', 'n_plates', 'mean_prop', 'sd_prop'])
        for g in GROUP_ORDER:
            v = by_group_plate[g]
            w.writerow([g, len(v), f'{np.mean(v):.5f}',
                        f'{np.std(v, ddof=1):.5f}'])
        w.writerow([])
        w.writerow(['comparison', 'welch_t', 'df', 'p', 'cohens_d',
                    'ci_lo', 'ci_hi'])
        for g in GROUP_ORDER[1:]:
            t, df, p = welch_t(ctrl, by_group_plate[g])
            d, dlo, dhi = cohen_d(ctrl, by_group_plate[g])
            w.writerow([f'control vs {g}', f'{t:.4f}', f'{df:.2f}',
                        f'{p:.6g}', f'{d:.4f}', f'{dlo:.4f}', f'{dhi:.4f}'])
        w.writerow([])
        w.writerow(['# 每皿跨日期均值'])
        w.writerow(['plate', 'group', 'row', 'mean_prop'])
        for p in plates:
            g, r = p.split('_', 1)
            w.writerow([p, g, r, f'{plate_mean[p]:.5f}'])
    print(f'\n结果已写入: {args.out}')


if __name__ == '__main__':
    main()
