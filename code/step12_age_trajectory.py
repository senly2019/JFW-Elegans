"""姿态的年龄维度：簇占比随成像日期的演变与组间轨迹差异（step12_age_trajectory.py）。

动机（2026-10-24）
----------------
实验为寿命/活力监测，同一培养皿跨多天重复成像，全量覆盖约 21 天（完整寿命周期）。
因此"成像日期"本质上是**年龄（时序）维度**，而非独立批次。据此本脚本检验：
  1. 各簇占比是否随成像日期系统变化（姿态的年龄演变）；
  2. 该演变轨迹是否存在组间差异（药物是否改变衰老轨迹）；
  3. 各组在各日期的样本量变化（生存差异的间接指标）。

若第 1 项成立而既有分析只做跨日期的简单组间平均，则年龄演变会被
"组间年龄构成不同"稀释——这可能是简单组间比较效应偏弱的原因之一。

前置条件
--------
需要 `atlas_worm_hist.csv`（每虫姿态直方图，含 date 列）。

用法
----
  python step12_age_trajectory.py -i atlas_out_full/atlas_worm_hist.csv \
      -o atlas_out_full/age_trajectory.csv
"""
import os
import csv
import math
import argparse
from collections import defaultdict

GROUP_ORDER = ['control', '5mg', '8mg', '10mg']


def lin_trend(xs, ys):
    """最小二乘直线趋势，返回 (slope, intercept, r, se_slope, n)。"""
    n = len(xs)
    nan5 = (float('nan'),) * 5
    if n < 3:
        return nan5
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx < 1e-12:
        return nan5
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    sst = sum((y - my) ** 2 for y in ys)
    r = sxy / math.sqrt(sxx * sst) if sst > 1e-12 else float('nan')
    # 斜率标准误
    s2 = sum(e ** 2 for e in resid) / (n - 2) if n > 2 else float('nan')
    se = math.sqrt(s2 / sxx) if s2 == s2 and sxx > 0 else float('nan')
    return slope, intercept, r, se, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', default='atlas_out_full/atlas_worm_hist.csv')
    ap.add_argument('-o', '--out', default='atlas_out_full/age_trajectory.csv')
    ap.add_argument('--min-frames', type=int, default=10)
    ap.add_argument('--cluster', type=int, default=0, help='0=深弯/Ω，1=浅弯/笔直')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f'未找到 {args.input}')

    rows = []
    with open(args.input, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if int(r['n_frames']) < args.min_frames:
                continue
            rows.append(dict(group=r['group'], date=r['date'],
                             prop=float(r['p%d' % args.cluster])))
    dates = sorted({r['date'] for r in rows})
    # 只保留每组都有观测的日期，避免生存缺失导致的伪趋势
    cnt = defaultdict(int)
    for r in rows:
        cnt[(r['group'], r['date'])] += 1
    complete = [d for d in dates
                if all(cnt[(g, d)] > 0 for g in GROUP_ORDER)]
    print(f'全部日期 {len(dates)} 个；各组均有观测的日期 {len(complete)} 个: {complete}')
    if len(complete) < 3:
        raise SystemExit('有效日期不足 3 个，无法估计趋势')

    # ---- 1. 各日期各组均值与样本量 ----
    print('\n=== 各成像日期的簇占比（均值，括号内为虫数）===')
    per = defaultdict(list)
    for r in rows:
        per[(r['group'], r['date'])].append(r['prop'])
    print('  date     ' + ''.join(f'{g:>18s}' for g in GROUP_ORDER))
    for d in complete:
        cells = []
        for g in GROUP_ORDER:
            v = per[(g, d)]
            cells.append(f'{sum(v)/len(v):.3f}(n={len(v):>3d})')
        print(f'  {d}  ' + ''.join(f'{c:>18s}' for c in cells))

    # ---- 2. 组内年龄趋势 ----
    print('\n=== 组内趋势（簇占比 ~ 日期序号）===')
    xs = list(range(len(complete)))
    trends = {}
    for g in GROUP_ORDER:
        ys = [sum(per[(g, d)]) / len(per[(g, d)]) for d in complete]
        slope, intercept, r, se, n = lin_trend(xs, ys)
        trends[g] = (slope, se, ys)
        t = slope / se if se and se == se else float('nan')
        p = math.erfc(abs(t) / math.sqrt(2)) if t == t else float('nan')
        print(f'  {g:>6s}: 斜率={slope:+.5f}/日  r={r:+.4f}  '
              f't={t:+.3f}  p={p:.4g}  首末 Δ={ys[-1]-ys[0]:+.4f}')

    # ---- 3. 组间斜率差异（control 为参照）----
    print('\n=== 组间轨迹差异（斜率差，control 为参照）===')
    s_ref, se_ref, ys_ref = trends['control']
    for g in GROUP_ORDER[1:]:
        s_g, se_g, ys_g = trends[g]
        diff = s_g - s_ref
        se_diff = math.sqrt(se_ref ** 2 + se_g ** 2)
        t = diff / se_diff if se_diff > 0 else float('nan')
        p = math.erfc(abs(t) / math.sqrt(2)) if t == t else float('nan')
        print(f'  {g:>6s} − control: 斜率差={diff:+.5f}/日  t={t:+.3f}  p={p:.4g}'
              f'{"  *" if p < 0.05 else ""}')

    # ---- 4. 生存差异的间接指标：各日期样本量比 ----
    print('\n=== 各组样本量随日期的变化（生存/保留的间接指标）===')
    base = {g: cnt[(g, complete[0])] for g in GROUP_ORDER}
    print('  date     ' + ''.join(f'{g:>12s}' for g in GROUP_ORDER))
    for d in complete:
        cells = [f'{cnt[(g,d)]/base[g]*100:5.1f}%' for g in GROUP_ORDER]
        print(f'  {d}  ' + ''.join(f'{c:>12s}' for c in cells))
    print('  （以各组首个日期的虫数为 100%）')
    late = complete[-1]
    print(f'  末期 {late} 相对首期: ' + ', '.join(
        f'{g}={cnt[(g,late)]/base[g]*100:.0f}%' for g in GROUP_ORDER))

    # ---- 5. 写 CSV ----
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['# 姿态的年龄维度分析（TBD-12）'])
        w.writerow(['complete_dates'] + complete)
        w.writerow([])
        w.writerow(['# 各日期各组簇占比'])
        w.writerow(['date'] + GROUP_ORDER)
        for d in complete:
            w.writerow([d] + [f'{sum(per[(g,d)])/len(per[(g,d)]):.5f}'
                              for g in GROUP_ORDER])
        w.writerow([])
        w.writerow(['# 各日期各组虫数'])
        w.writerow(['date'] + GROUP_ORDER)
        for d in complete:
            w.writerow([d] + [cnt[(g, d)] for g in GROUP_ORDER])
        w.writerow([])
        w.writerow(['# 组内年龄趋势'])
        w.writerow(['group', 'slope_per_day', 'pearson_r', 'se_slope',
                    't', 'p', 'delta_first_last'])
        for g in GROUP_ORDER:
            s, se, ys = trends[g]
            slope, intercept, r, se2, n = lin_trend(xs, ys)
            t = s / se if se and se == se else float('nan')
            p = math.erfc(abs(t) / math.sqrt(2)) if t == t else float('nan')
            w.writerow([g, f'{s:.6f}', f'{r:.4f}', f'{se:.6f}',
                        f'{t:.4f}' if t == t else '',
                        f'{p:.6g}' if p == p else '',
                        f'{ys[-1]-ys[0]:.5f}'])
        w.writerow([])
        w.writerow(['# 组间斜率差异（control 为参照）'])
        w.writerow(['comparison', 'slope_diff', 'se_diff', 't', 'p'])
        for g in GROUP_ORDER[1:]:
            s_g, se_g, _ = trends[g]
            diff = s_g - s_ref
            se_diff = math.sqrt(se_ref ** 2 + se_g ** 2)
            t = diff / se_diff if se_diff > 0 else float('nan')
            p = math.erfc(abs(t) / math.sqrt(2)) if t == t else float('nan')
            w.writerow([f'{g} - control', f'{diff:.6f}', f'{se_diff:.6f}',
                        f'{t:.4f}' if t == t else '',
                        f'{p:.6g}' if p == p else ''])
    print(f'\n结果已写入: {args.out}')
    print('\n解读要点：')
    print('  · 若各组斜率均为负，说明该簇占比随年龄下降（姿态的年龄演变）；')
    print('  · 若给药组斜率显著大于 control，提示药物改变了衰老轨迹；')
    print('  · 样本量占比下降提示生存差异，组间比较需警惕幸存者偏差。')


if __name__ == '__main__':
    main()
