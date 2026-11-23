"""step17_posture_profile.py — 段内姿态指纹：一条视频（一只虫的帧序列）内姿态的分布与动态。

为什么用连续轴而不是三类占比
--------------------------
原始设计（step9）的指纹是三类占比 p_正弦 / p_Ω / p_笔直。但三类体系未通过人工验证
（4.3 节），用三类占比描述视频会把已被否证的分类重新引入结论。因此本脚本把指纹
改写在经标定的连续弯曲深度轴（BDI）上：

  水平  mean / median BDI                  这条虫整体弯得深不深
  离散  sd / IQR / p5–p95                  姿态波动多大
  动态  mean|ΔBDI|、lag-1 自相关           段内姿态变化快慢 / 记忆性
  转移  跨切点次数率、最长停留帧数          在"深弯区/浅弯区"之间的切换
  占比  frac_above_cut                     替代 p_Ω，但不依赖类别

统计口径
--------
  · 每只虫 = 一条视频段（帧序列按文件名中的帧号排序，非按目录顺序）；
  · 虫级为统计单元；若按皿/组比较，须同时报告皿级口径（伪重复见 4.5 节）；
  · 帧数不足 min_frames 的虫剔除并计数；
  · 切点用 BDI 上 k=2 的 k-means 分界（与 4.4 节同口径），不额外引入阈值。

用法（ele 环境）
--------------
  python step17_posture_profile.py -c atlas_out_full/kappa_cache.npz \
      -o atlas_out_full/profile --min-frames 50

产物
----
  profile_worm.csv     每虫的指纹（水平/离散/动态/转移/占比）
  profile_group.csv    每组的虫级均值与皿级均值（附伪重复提示）
  profile_reliability.txt  分半信度（水平与占比两类指标）
  fig11_posture_profile.png  四联图
"""

import os
import re
import csv
import argparse
import numpy as np

GROUP_ORDER = ['control', '5mg', '8mg', '10mg']
FEATURE_NAMES = ['bbi', 'max_kappa', 'n_high_regions', 'mid_dominance',
                 'peak_midness', 'kappa_std', 'periodicity_peaks']


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'  已写 {path}（{len(rows)} 行）')


def frame_no(rel):
    """从文件名解析帧号：.../worm12_045.png -> 45。解析失败返回 -1。"""
    stem = os.path.splitext(os.path.basename(str(rel)))[0]
    m = re.search(r'_(\d+)$', stem)
    return int(m.group(1)) if m else -1


def build_bdi(kappa, feats, n_fit=50000, seed=0):
    """BDI = κ(s) 第一主成分，符号固定为与 bbi 正相关，归一到 0–1。"""
    from sklearn.decomposition import PCA
    rng = np.random.RandomState(seed)
    fit_idx = (rng.choice(len(kappa), n_fit, replace=False)
               if len(kappa) > n_fit else np.arange(len(kappa)))
    p = PCA(n_components=1, random_state=seed).fit(kappa[fit_idx])
    z = p.transform(kappa)[:, 0]
    bbi = feats[:, FEATURE_NAMES.index('bbi')]
    r = float(np.corrcoef(z[fit_idx], bbi[fit_idx])[0, 1])
    if r < 0:
        z = -z
    z = (z - np.percentile(z, 0.1)) / max(
        np.percentile(z, 99.9) - np.percentile(z, 0.1), 1e-9)
    return np.clip(z, 0, 1).astype(np.float32), float(p.explained_variance_ratio_[0])


def km_cut(bdi, seed=0):
    """k=2 分界（与 4.4 同口径）。"""
    from sklearn.cluster import KMeans
    x = bdi if len(bdi) <= 200000 else np.random.RandomState(seed).choice(
        bdi, 200000, replace=False)
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(x[:, None])
    c = np.sort(km.cluster_centers_[:, 0])
    return float((c[0] + c[1]) / 2)


def profile_one(v, cut):
    """单只虫的指纹。v 为按帧号排序后的 BDI 序列。"""
    n = len(v)
    d = np.diff(v)
    above = v > cut
    # 跨切点次数与最长停留
    cross = int(np.sum(above[1:] != above[:-1]))
    dwell, cur = 0, 0
    for a in above:
        cur = cur + 1 if a else 0
        dwell = max(dwell, cur)
    acf1 = float(np.corrcoef(v[:-1], v[1:])[0, 1]) if n > 3 and np.std(v) > 1e-9 else float('nan')
    return dict(
        n_frames=n,
        bdi_mean=round(float(v.mean()), 4),
        bdi_median=round(float(np.median(v)), 4),
        bdi_sd=round(float(v.std(ddof=1)), 4) if n > 1 else float('nan'),
        bdi_iqr=round(float(np.percentile(v, 75) - np.percentile(v, 25)), 4),
        bdi_p5=round(float(np.percentile(v, 5)), 4),
        bdi_p95=round(float(np.percentile(v, 95)), 4),
        frac_above_cut=round(float(above.mean()), 4),
        acf1=round(acf1, 4) if acf1 == acf1 else '',
        mean_abs_delta=round(float(np.abs(d).mean()), 5) if n > 1 else float('nan'),
        cross_rate=round(cross / max(n - 1, 1), 4),
        max_dwell_above=int(dwell),
    )


def group_summary(rows):
    """虫级与皿级两种口径（皿级才可辩护，见 4.5）。"""
    out = []
    for g in GROUP_ORDER:
        sub = [r for r in rows if r['group'] == g]
        if not sub:
            continue
        worm_means = [r['bdi_mean'] for r in sub]
        plates = {}
        for r in sub:
            plates.setdefault((r['date'], r['row']), []).append(r['bdi_mean'])
        plate_means = [float(np.mean(v)) for v in plates.values()]
        out.append(dict(
            group=g, n_worms=len(sub), n_plates=len(plates),
            worm_mean=round(float(np.mean(worm_means)), 4),
            worm_sd=round(float(np.std(worm_means, ddof=1)) if len(worm_means) > 1 else float('nan'), 4),
            plate_mean=round(float(np.mean(plate_means)), 4),
            plate_sd=round(float(np.std(plate_means, ddof=1)) if len(plate_means) > 1 else float('nan'), 4),
            frac_above=round(float(np.mean([r['frac_above_cut'] for r in sub])), 4),
            mom=round(float(np.mean([r['mean_abs_delta'] for r in sub])), 5),
        ))
    return out


def split_half_reliability(profiles_by_worm):
    """时间分半信度：前半段与后半段的指纹指标相关（Pearson + Spearman–Brown）。"""
    a_mean, b_mean, a_frac, b_frac = [], [], [], []
    for _, rec in profiles_by_worm.items():
        if rec['n_frames'] < 20:
            continue
        a_mean.append(rec['first_mean'])
        b_mean.append(rec['second_mean'])
        a_frac.append(rec['first_frac'])
        b_frac.append(rec['second_frac'])
    out = {}
    for name, (x, y) in (('bdi_mean', (a_mean, b_mean)),
                         ('frac_above_cut', (a_frac, b_frac))):
        if len(x) < 10:
            out[name] = (float('nan'), float('nan'), len(x))
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        sb = 2 * r / (1 + r) if r > -1 else float('nan')
        out[name] = (round(r, 4), round(sb, 4), len(x))
    return out


def main():
    ap = argparse.ArgumentParser(description='段内姿态指纹（连续轴版）')
    ap.add_argument('-c', '--cache', default='atlas_out_full/kappa_cache.npz')
    ap.add_argument('-o', '--outdir', default='atlas_out_full/profile')
    ap.add_argument('--min-frames', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    z = np.load(args.cache, allow_pickle=True)
    kappa = z['kappa'].astype(np.float32)
    feats = np.asarray(z['features'], dtype=np.float32)
    files = np.asarray([str(f) for f in z['filenames']])
    groups, dates = z['groups'], z['dates']
    rows_w, worms = z['rows'], z['worms']
    print(f'[载入] {len(kappa):,} 帧')

    bdi, var = build_bdi(kappa, feats, seed=args.seed)
    cut = km_cut(bdi, seed=args.seed)
    print(f'[轴] PC1 方差 {var:.4f}；k=2 切点 {cut:.4f}')

    # 按 (组,日期,行,虫) 聚合，并按帧号排序
    key = np.asarray([f'{g}|{d}|{r}|{w}' for g, d, r, w in
                      zip(groups, dates, rows_w, worms)])
    # 按 (虫, 帧号) 二级排序：先保证同一只虫的帧连续，再保证段内按帧序
    fno = np.asarray([frame_no(f) for f in files], dtype=int)
    order = np.lexsort((fno, key))          # 主键 key，次键 fno
    key_s, bdi_s = key[order], bdi[order]

    uk, start = np.unique(key_s, return_index=True)
    end = np.r_[start[1:], len(key_s)]
    profiles, by_worm = [], {}
    dropped = 0
    for k, s, e in zip(uk, start, end):
        v = bdi_s[s:e]
        if len(v) < args.min_frames:
            dropped += 1
            continue
        g, d, r, w = k.split('|')
        rec = profile_one(v, cut)
        h = len(v) // 2
        by_worm[k] = dict(rec, first_mean=float(v[:h].mean()),
                          second_mean=float(v[h:].mean()),
                          first_frac=float((v[:h] > cut).mean()),
                          second_frac=float((v[h:] > cut).mean()))
        profiles.append(dict(group=g, date=d, row=r, worm=w, **rec))
    print(f'[虫] 入统计 {len(profiles):,}，剔除 {dropped}（帧数 < {args.min_frames}）')

    write_csv(os.path.join(args.outdir, 'profile_worm.csv'), profiles,
              ['group', 'date', 'row', 'worm', 'n_frames',
               'bdi_mean', 'bdi_median', 'bdi_sd', 'bdi_iqr', 'bdi_p5', 'bdi_p95',
               'frac_above_cut', 'acf1', 'mean_abs_delta', 'cross_rate',
               'max_dwell_above'])
    gs = group_summary(profiles)
    write_csv(os.path.join(args.outdir, 'profile_group.csv'), gs)

    # 信度
    rel = split_half_reliability(by_worm)
    lines = ['段内姿态指纹：时间分半信度', '=' * 46]
    for name, (r, sb, n) in rel.items():
        lines.append(f'{name}: r={r}, Spearman–Brown={sb}（{n} 只）')
    # 动态指标的量级（回答"段内变化有多大"）
    mom = [r['mean_abs_delta'] for r in profiles]
    acf = [r['acf1'] for r in profiles if r['acf1'] != '']
    lines.append('')
    lines.append(f'段内变化量 mean|ΔBDI| = {np.mean(mom):.5f}'
                 f'（BDI 值域 0–1）；lag-1 自相关 = {np.mean(acf):.4f}')
    lines.append('解读：|ΔBDI| 很小且自相关接近 1 时，表示段内姿态近乎不变，'
                 '"指纹"主要由水平项承担，动态项信息有限。')
    txt = '\n'.join(lines)
    with open(os.path.join(args.outdir, 'profile_reliability.txt'), 'w',
              encoding='utf-8') as f:
        f.write(txt + '\n')
    print('\n' + txt)

    make_figure(args.outdir, profiles, gs, bdi_s, key_s, cut, rel)


def make_figure(outdir, profiles, gs, bdi_s, key_s, cut, rel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 11,
                         'figure.dpi': 300})
    COLORS = {'control': '#4D4D4D', '5mg': '#0072B2', '8mg': '#D55E00',
              '10mg': '#009E73'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # A: 三条示例虫的段内轨迹
    ax = axes[0, 0]
    uk = np.unique(key_s)
    picked = uk[::max(1, len(uk) // 4)][:4]
    for k, c in zip(picked, ['#0072B2', '#D55E00', '#009E73', '#CC79A7']):
        v = bdi_s[key_s == k]
        ax.plot(np.arange(len(v)), v, lw=1.2, color=c,
                label=f'{k.split("|")[3]} ({len(v)}f)')
    ax.axhline(cut, ls='--', color='#888', lw=1)
    ax.set_xlabel('Frame index within video')
    ax.set_ylabel('Bend-depth index')
    ax.set_title('A. Within-video posture trajectories')
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # B: 每虫均值分布（按组）
    ax = axes[0, 1]
    data = [[r['bdi_mean'] for r in profiles if r['group'] == g]
            for g in GROUP_ORDER if any(r['group'] == g for r in profiles)]
    names = [g for g in GROUP_ORDER if any(r['group'] == g for r in profiles)]
    if data:
        bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                        widths=0.55)
        # matplotlib 3.9+ 移除了 boxplot(labels=...)，改用 tick_labels；
        # 这里用 set_xticks/set_xticklabels 写法，所有版本通用
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names)
        for patch, g in zip(bp['boxes'], names):
            patch.set_facecolor(COLORS[g])
            patch.set_alpha(0.55)
    ax.set_ylabel('Per-worm mean BDI')
    ax.set_title('B. Profile level by group')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # C: 动态项分布
    ax = axes[1, 0]
    mom = [r['mean_abs_delta'] for r in profiles]
    ax.hist(mom, bins=50, color='#0072B2', alpha=0.85)
    ax.set_xlabel('mean |ΔBDI| per frame')
    ax.set_ylabel('Number of worms')
    ax.set_title('C. Within-video change is small')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # D: 信度
    ax = axes[1, 1]
    ax.axis('off')
    rows = [(k, f'r={v[0]}, SB={v[1]} (n={v[2]})') for k, v in rel.items()]
    ax.text(0.02, 0.9, 'Split-half reliability\n\n' +
            '\n'.join(f'{a:18s} {b}' for a, b in rows),
            va='top', fontsize=10, family='DejaVu Sans')
    ax.set_title('D. Reliability of profile metrics')

    fig.suptitle('Per-video posture profile on a continuous axis', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(outdir, 'fig11_posture_profile.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'图已保存：{out}')


if __name__ == '__main__':
    main()
