"""step18b_rater_profile.py — 单个标注者的"判据画像"：TA 到底在看什么？

背景
----
step18 只回答"两个人一致不一致"。但当人–机 κ 接近 0 时，有两种完全相反的
解释，必须分清：

  (a) 标注者没有按骨架曲率在看（看的是粗细、亮度、是否出框……）
      -> 问题在标注，重标/加训可救；
  (b) 我们的七个曲率特征根本没编码人眼看到的东西
      -> 问题在特征，论文的"连续轴"结论要改。

本脚本用一个可判读的指标把两者分开：对每个标注者，算"人判弯/直"对每个
骨架特征的**单变量 AUC**。若某标注者的 bbi-AUC 接近 0.5，说明他的判断
与弯曲深度无关；若接近 0.8 以上，说明他就是按曲率在判。

同时给出各作答码的 bbi 均值（应当单调：直 < Ω ≈ 正弦），以及与旧标注者
（默认 human_labels.csv，C2 那批 210 帧）的并排对照。

判读经验值
  标注者 bbi-AUC ≥ 0.70  -> 曲率驱动，可作金标准
  0.60–0.70             -> 弱曲率驱动，混杂其他线索
  ≤ 0.60                -> 基本不是在看曲率（或图像不足以支撑该判断）

用法
----
  # 单份作答
  python step18b_rater_profile.py -l annotation_v2/labels_A.csv

  # 两份一起，并与旧标注者对照
  python step18b_rater_profile.py -l annotation_v2/labels_A.csv ^
        annotation_v2/labels_B.csv --ref human_labels.csv
"""

import os
import csv
import argparse
import numpy as np

FEATURE_NAMES = ['bbi', 'max_kappa', 'n_high_regions', 'mid_dominance',
                 'peak_midness', 'kappa_std', 'periodicity_peaks']
CODE_NAME = {1: 'Straight', 2: 'Omega', 3: 'Sine'}


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load_feats(cache_path):
    """只取 features 与 filenames，不读曲率矩阵（省内存）。"""
    z = np.load(cache_path, allow_pickle=True)
    files = [str(x).replace('\\', '/') for x in z['filenames']]
    feats = np.asarray(z['features'], dtype=np.float32)
    return {f: i for i, f in enumerate(files)}, feats


def auc(score, label):
    """Mann-Whitney AUC；label=1 为正类。"""
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def collect_pairs(rows, idx, feats, path_col, code_col='label_code'):
    pairs = []
    miss = 0
    for r in rows:
        p = (r.get(path_col) or '').replace('\\', '/')
        raw = (r.get(code_col) or '').strip()
        if not raw:
            continue
        try:
            c = int(float(raw))
        except ValueError:
            continue
        if c not in (1, 2, 3):
            continue
        if p not in idx:
            miss += 1
            continue
        pairs.append((c, feats[idx[p]]))
    return pairs, miss


def profile(tag, pairs, out_lines):
    def P(s=''):
        print(s)
        out_lines.append(s)

    P('=' * 70)
    P(f'{tag}   n={len(pairs)}')
    P('=' * 70)
    if not pairs:
        P('  [跳过] 无可匹配样本')
        return {}
    cnt = {c: sum(1 for p in pairs if p[0] == c) for c in (1, 2, 3)}
    P('  作答分布：' + '  '.join(f'{CODE_NAME[c]}={cnt[c]} ({cnt[c]/len(pairs)*100:.1f}%)'
                                for c in (1, 2, 3)))
    P('  各码的 bbi（应单调：Straight 最低）：')
    for c in (1, 2, 3):
        v = np.array([p[1][0] for p in pairs if p[0] == c], dtype=float)
        if len(v):
            P(f'    {CODE_NAME[c]:9s} n={len(v):3d}  均值 {v.mean():.4f}  '
              f'中位 {np.median(v):.4f}  标准差 {v.std():.4f}')
    lab = [0 if p[0] == 1 else 1 for p in pairs]
    a_bbi = auc([p[1][0] for p in pairs], lab)
    P(f'  二值(弯/直) 的 bbi-AUC = {a_bbi:.3f}')
    P('  各特征二值 AUC：' + '  '.join(
        f'{FEATURE_NAMES[j]}={auc([p[1][j] for p in pairs], lab):.3f}'
        for j in range(len(FEATURE_NAMES))))
    if not (a_bbi == a_bbi):
        verdict = '无法判定'
    elif a_bbi >= 0.70:
        verdict = '曲率驱动（可作金标准）'
    elif a_bbi >= 0.60:
        verdict = '弱曲率驱动（混杂其他线索）'
    else:
        verdict = '基本不是在看弯曲深度'
    P(f'  -> 判读：{verdict}')
    return dict(bbi_auc=a_bbi, n=len(pairs), **{f'cnt{c}': cnt[c] for c in (1, 2, 3)})


def main():
    ap = argparse.ArgumentParser(description='标注者判据画像（单变量 AUC 剖面）')
    ap.add_argument('-l', '--labels', nargs='+', required=True,
                    help='一份或多份作答 CSV')
    ap.add_argument('--ref', default='human_labels.csv',
                    help='参照标注（旧 C2 那批），不存在则跳过')
    ap.add_argument('-c', '--cache', default='atlas_out_full/kappa_cache.npz')
    ap.add_argument('-o', '--out', default='annotation_v2/rater_profile.txt')
    args = ap.parse_args()

    idx, feats = load_feats(args.cache)
    print(f'缓存载入：{len(idx)} 帧 × {feats.shape[1]} 特征')

    lines = []
    results = {}
    if args.ref and os.path.exists(args.ref):
        rows = read_csv(args.ref)
        pairs, miss = collect_pairs(rows, idx, feats, 'filename')
        results['REF'] = profile(f'参照标注者（{args.ref}）', pairs, lines)
    else:
        print(f'[提示] 参照文件 {args.ref} 不存在，跳过对照')

    for lp in args.labels:
        rows = read_csv(lp)
        pairs, miss = collect_pairs(rows, idx, feats, 'rel_path')
        tag = os.path.basename(lp) + (f'（{miss} 条路径不在缓存中）' if miss else '')
        results[os.path.basename(lp)] = profile(tag, pairs, lines)
        lines.append('')

    lines.append('=' * 70)
    lines.append('汇总')
    lines.append('=' * 70)
    for k, v in results.items():
        if v:
            lines.append(f"  {k:28s} n={v['n']:3d}  bbi-AUC={v['bbi_auc']:.3f}  "
                         f"直{v['cnt1']}/Ω{v['cnt2']}/正弦{v['cnt3']}")
    txt = '\n'.join(lines)
    print('\n' + txt.split('汇总')[-1])
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(txt + '\n')
        print(f'\n已写出：{args.out}')


if __name__ == '__main__':
    main()
