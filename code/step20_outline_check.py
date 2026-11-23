"""step20_outline_check.py — 诊断 OUTLINE 管线的高轮廓系数（0.93）来自何处。

背景
----
`step15` 的预注册判定里，S1（所有组合最大轮廓系数 < 0.35）**不成立**：
OUTLINE 管线在 k-means k=2 上给出 silhouette = 0.9345（ward k=2 为 0.9181），
而其余三条管线只有 0.12–0.31。这个数必须解释，否则不能写"连续"。

三种可能，必须分开：
  A 技术伪迹：小簇由离群帧驱动（轮廓提取退化、尺度未归一化、闭合失败）
  B 真实的二分类：两簇对应"伸直 vs 弯曲"（若如此，则与人工二值 κ=0.211 一致，
    结论应改为"一条主导轴 + 轴上二分"，而非"连续"）
  C 其他结构：与弯曲无关但与成像批次相关（技术性）

判别手段
--------
1. 簇大小与轮廓贡献：小簇（<10%）几乎一定是离群驱动。
2. 两簇在 KAPPA 管线的弯曲量上是否分离（bbi / max_kappa）——B 会分离，A/C 不会。
3. 两簇与操作化二值标签（笔直 vs 弯曲）的吻合度（κ）——B 会明显 >0。
4. 两簇的组别 / 日期分布是否极不均衡——C 会（技术伪迹常与批次绑定）。
5. 两簇在傅里叶第 0 阶（尺度项）上的差异——A（尺度伪迹）会极大。
6. 每簇输出 5 个代表帧路径，供肉眼复核。

用法（ele 环境，约 1 分钟）
------------------------
  python step20_outline_check.py -c atlas_out_full/indep/indep_cache.npz \
      -o atlas_out_full/indep/outline_check.txt

产物
----
  outline_check.txt   诊断报告（含结论与代表帧列表）
"""

import os
import sys
import argparse
import numpy as np

GROUP_ORDER = ['control', '5mg', '8mg', '10mg']


def main():
    ap = argparse.ArgumentParser(description='诊断 OUTLINE 管线的高轮廓系数')
    ap.add_argument('-c', '--cache', default='atlas_out_full/indep/indep_cache.npz')
    ap.add_argument('-o', '--out', default='atlas_out_full/indep/outline_check.txt')
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--n-sil', type=int, default=3000, help='轮廓子样本')
    ap.add_argument('--n-show', type=int, default=6, help='每簇代表帧数')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score, cohen_kappa_score

    z = np.load(args.cache, allow_pickle=True)
    need = ['OUTLINE', 'KAPPA', 'filenames']
    for n in need:
        if n not in z:
            raise SystemExit(f'缓存缺少 {n}')
    X = z['OUTLINE'].astype(np.float64)
    K = z['KAPPA'].astype(np.float64)
    files = [str(x).replace('\\', '/') for x in z['filenames']]
    groups = z['groups'] if 'groups' in z else None
    dates = z['dates'] if 'dates' in z else None
    n = len(X)
    print(f'帧数 {n:,}；OUTLINE {X.shape[1]} 维；KAPPA {K.shape[1]} 维')

    lines = []
    def P(s=''):
        print(s)
        lines.append(s)

    P('=' * 68)
    P('OUTLINE 管线高轮廓系数诊断')
    P('=' * 68)
    P(f'缓存：{args.cache}    帧数 {n:,}')

    # ---- 1. k=2 聚类 ----
    km = KMeans(n_clusters=args.k, n_init=20, random_state=args.seed).fit(X)
    lab = km.labels_
    cnt = np.bincount(lab, minlength=args.k)
    order = np.argsort(-cnt)
    P()
    P('-' * 68)
    P('① 簇大小（按降序）')
    P('-' * 68)
    for c in order:
        P(f'  cluster {c}: {cnt[c]:7,} 帧 ({cnt[c] / n * 100:5.2f}%)')
    small = cnt.min() / n
    P(f'  最小簇占比 = {small * 100:.2f}%'
      f'{"  <-- 低于 10%，高度可疑为离群驱动" if small < 0.10 else ""}')

    # ---- 2. 轮廓贡献 ----
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(n, min(args.n_sil, n), replace=False)
    sil_all = silhouette_samples(X[idx], lab[idx])
    P()
    P('-' * 68)
    P('② 轮廓系数分解（子样本 %d）' % len(idx))
    P('-' * 68)
    P(f'  整体 silhouette = {silhouette_score(X[idx], lab[idx]):.4f}')
    for c in range(args.k):
        m = lab[idx] == c
        if m.any():
            P(f'  cluster {c}: n={m.sum():5d}  mean={sil_all[m].mean():+.4f}  '
              f'min={sil_all[m].min():+.4f}  max={sil_all[m].max():+.4f}')

    # ---- 3. 与弯曲量是否分离 ----
    bbi = K.mean(axis=1)
    maxk = K.max(axis=1)
    P()
    P('-' * 68)
    P('③ 两簇在弯曲量上的差异（若分离 -> 可能是真实的"直 vs 弯"）')
    P('-' * 68)
    P(f'{"cluster":>8s} {"n":>7s} {"bbi mean":>9s} {"bbi sd":>8s} '
      f'{"max_kappa":>10s} {"fourier0":>9s}')
    stats = {}
    for c in range(args.k):
        m = lab == c
        stats[c] = dict(n=int(m.sum()), bbi=float(bbi[m].mean()),
                        bbis=float(bbi[m].std()), maxk=float(maxk[m].mean()),
                        f0=float(X[m, 0].mean()))
        P(f'{c:>8d} {m.sum():>7d} {bbi[m].mean():>9.4f} {bbi[m].std():>8.4f} '
          f'{maxk[m].mean():>10.4f} {X[m, 0].mean():>9.5f}')
    d_bbi = abs(stats[0]['bbi'] - stats[1]['bbi']) if args.k == 2 else float('nan')
    d_f0 = abs(stats[0]['f0'] - stats[1]['f0']) if args.k == 2 else float('nan')
    ratio_f0 = (max(stats[0]['f0'], stats[1]['f0']) /
                max(min(stats[0]['f0'], stats[1]['f0']), 1e-9)) if args.k == 2 else float('nan')
    P(f'  Δbbi = {d_bbi:.4f}   Δfourier0 = {d_f0:.5f}  '
      f'尺度比 = {ratio_f0:.2f}×')
    P('  判读：Δbbi 明显（>0.03）→ 与弯曲程度相关；尺度比 >2 → 尺度伪迹；'
      '两者都小 → 技术性差异')

    # ---- 4. 与操作化二值标签吻合 ----
    P()
    P('-' * 68)
    P('④ 与"笔直 vs 弯曲"二值标签的吻合度')
    P('-' * 68)
    # 用 bbi<0.16 复现笔直判据（与 pseudo-label 规则一致）
    is_straight = (bbi < 0.16).astype(int)
    # 把簇映射成"更直的那簇 = 1"
    c_straight = 0 if stats[0]['bbi'] < stats[1]['bbi'] else 1
    pred = (lab == c_straight).astype(int)
    kappa_bin = float(cohen_kappa_score(is_straight, pred))
    agree = float((is_straight == pred).mean())
    P(f'  以 bbi<0.16 定义"笔直"：笔直帧 {is_straight.sum():,} '
      f'({is_straight.mean() * 100:.1f}%)')
    P(f'  簇 -> 二值的 κ = {kappa_bin:+.4f}，一致率 {agree:.3f}')
    P('  判读：κ > 0.3 → 该簇结构就是"直 vs 弯"（与人工二值 κ=0.211 同向）')

    # ---- 5. 组别 / 日期分布 ----
    P()
    P('-' * 68)
    P('⑤ 两簇的组别与日期分布（技术伪迹常与批次绑定）')
    P('-' * 68)
    for axis, vals, name in (('group', groups, '组别'), ('date', dates, '日期')):
        if vals is None:
            continue
        vals = np.asarray(vals).astype(str)
        uniq = sorted(set(vals))[:8]
        P(f'  [{name}] 各簇内的构成占比（列和 = 100%）')
        header = ' ' * 12 + ''.join(f'{u:>9s}' for u in uniq)
        P(header)
        for c in range(args.k):
            inner = vals[lab == c]
            if len(inner) == 0:
                continue
            P(f'  cluster {c:<2d} ' +
              ''.join(f'{(inner == u).mean() * 100:>8.1f}%' for u in uniq))
        P(f'  全体基线  ' +
          ''.join(f'{(vals == u).mean() * 100:>8.1f}%' for u in uniq))
        dev = 0.0
        for c in range(args.k):
            inner = vals[lab == c]
            for u in uniq:
                base = (vals == u).mean()
                if base > 0:
                    dev = max(dev, (inner == u).mean() / base)
        P(f'  最大占比偏离 = {dev:.2f}×'
          f'{"  (>3× → 与批次强关联，倾向技术伪迹)" if dev > 3 else ""}')

    # ---- 6. 代表帧 ----
    P()
    P('-' * 68)
    P('⑥ 每簇代表帧（供肉眼复核；建议每簇看 3–5 张）')
    P('-' * 68)
    for c in range(args.k):
        m = np.flatnonzero(lab == c)
        if len(m) == 0:
            continue
        center = km.cluster_centers_[c]
        d = np.linalg.norm(X[m] - center, axis=1)
        pick = m[np.argsort(d)[:args.n_show]]
        P(f'  cluster {c}（n={len(m):,}，bbi 均值 {stats[c]["bbi"]:.4f}）：')
        for i in pick:
            P(f'    {files[i]}   bbi={bbi[i]:.4f}')

    # ---- 结论 ----
    P()
    P('=' * 68)
    P('结论')
    P('=' * 68)
    tags = []
    if small < 0.10:
        tags.append('离群驱动（小簇 <10%）')
    if ratio_f0 > 2.0:
        tags.append('尺度伪迹（第 0 阶相差 >2 倍）')
    if d_bbi > 0.03:
        tags.append('与弯曲程度相关')
    if kappa_bin > 0.30:
        tags.append('等价于"直 vs 弯"二值结构')
    if not tags:
        tags.append('与弯曲、尺度、批次均无明显关联，需肉眼复核代表帧')
    P('  指向：' + '；'.join(tags))
    P()
    if '等价于"直 vs 弯"二值结构' in tags and '离群驱动（小簇 <10%）' not in tags:
        P('  → 建议改写结论为：「存在一条主导轴，且该轴在 OUTLINE 表示上呈现二分，')
        P('     该二分与人工二值判断同向；但不支持三类结构」。这不否定"无三类"，')
        P('     只把"连续"改为"单轴二分 + 轴上连续"。')
    elif '离群驱动（小簇 <10%）' in tags or '尺度伪迹（第 0 阶相差 >2 倍）' in tags:
        P('  → 该 0.93 应判为技术伪迹：在正文与 S3 中说明 OUTLINE 管线因轮廓退化/')
        P('     尺度未归一化而被单列剔除，其高轮廓系数不作为结构证据；')
        P('     S1 改为"在三条可用管线上成立"。')
    else:
        P('  → 需看代表帧后再判定；在判定前不得以该数值支持或否定"连续"。')

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n报告已写出：{args.out}')


if __name__ == '__main__':
    main()
