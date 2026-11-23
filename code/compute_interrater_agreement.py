"""人工标注 vs 伪标签的一致性计算（论文用）。

输出：
  · 总体一致率 + Cohen's kappa（3 类名义）+ bootstrap 95% CI
  · 各类的人-机一致率、精确率/召回率（以人工标注为参照）
  · 3×3 混淆矩阵（行=人工，列=伪标签）
  · 按浓度组分层的一致率（检验是否存在系统性偏差）
  · 一张可直接放论文的柱状图（各类一致率 + kappa 区间）

用法（ele 环境）：
  python compute_interrater_agreement.py -b human_annotation_batch.csv \
      -o human_labels.csv --fig human_agreement.png
"""
import os
import csv
import random
import argparse
from collections import defaultdict

import numpy as np

CLASS_NAMES = {0: 'Sinusoidal', 1: 'Omega', 2: 'Straight'}
CODE_TO_LABEL = {1: 2, 2: 1, 3: 0}     # GUI 键值 -> 标签编码


def cohen_kappa(a, b, n_cls=3):
    """名义 kappa。a/b 为标签编码序列。"""
    n = len(a)
    if n == 0:
        return float('nan')
    cm = np.zeros((n_cls, n_cls), dtype=float)
    for x, y in zip(a, b):
        cm[x][y] += 1
    po = np.trace(cm) / n
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = float((row * col).sum()) / (n * n)
    if abs(1 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--batch', required=True, help='human_annotation_batch.csv')
    ap.add_argument('-o', '--labels', required=True, help='human_labels.csv')
    ap.add_argument('--fig', default='human_agreement.png')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    pseudo = {}
    for r in csv.DictReader(open(args.batch, encoding='utf-8-sig')):
        pseudo[r['filename']] = int(r['pseudo_label'])

    human = {}
    for r in csv.DictReader(open(args.labels, encoding='utf-8-sig')):
        human[r['filename']] = CODE_TO_LABEL[int(r['label_code'])]

    common = [f for f in pseudo if f in human]
    if not common:
        print('没有可配对的标注！')
        return
    H = [human[f] for f in common]
    P = [pseudo[f] for f in common]
    n = len(common)
    print('=' * 70)
    print(f'人工标注 vs 操作化伪标签：配对 {n} 帧'
          f'（批次共 {len(pseudo)} 帧，已标 {len(human)} 帧）')
    print('=' * 70)

    acc = sum(1 for h, p in zip(H, P) if h == p) / n
    kappa = cohen_kappa(H, P)
    # bootstrap CI
    boots = []
    idx = np.arange(n)
    for _ in range(args.boot):
        b = np.random.choice(idx, size=n, replace=True)
        boots.append(cohen_kappa([H[i] for i in b], [P[i] for i in b]))
    boots = np.array([x for x in boots if x == x])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f'\n总体一致率 (agreement) = {acc:.3f} ({acc*100:.1f}%)')
    print(f'Cohen\'s kappa = {kappa:.3f}  bootstrap 95% CI = [{lo:.3f}, {hi:.3f}]')
    interp = ('excellent' if kappa >= 0.8 else
              'substantial' if kappa >= 0.6 else
              'moderate' if kappa >= 0.4 else 'fair' if kappa >= 0.2 else 'slight')
    print(f'  Landis-Koch 解读：{interp}')

    print('\n各类（以人工标注为参照）：')
    print(f"{'类别':<12s}{'支持数':>7s}{'一致率':>9s}{'精确率':>9s}{'召回率':>9s}{'F1':>8s}")
    for c in range(3):
        idx_c = [i for i in range(n) if H[i] == c]
        if not idx_c:
            continue
        agree = sum(1 for i in idx_c if P[i] == c) / len(idx_c)
        tp = agree * len(idx_c)
        fp = sum(1 for i in range(n) if P[i] == c and H[i] != c)
        fn = len(idx_c) - tp
        pr, rc, f1 = prf(tp, fp, fn)
        print(f'{CLASS_NAMES[c]:<12s}{len(idx_c):>7d}{agree:>9.3f}{pr:>9.3f}{rc:>9.3f}{f1:>8.3f}')

    print('\n混淆矩阵（行=人工标注，列=伪标签）：')
    print(f"{'':<12s}" + ''.join(f'{CLASS_NAMES[c]:>12s}' for c in range(3)))
    cm = np.zeros((3, 3), dtype=int)
    for h, p in zip(H, P):
        cm[h][p] += 1
    for c in range(3):
        print(f'{CLASS_NAMES[c]:<12s}' + ''.join(f'{cm[c][k]:>12d}' for k in range(3)))

    # 按组分层
    groups = defaultdict(list)
    for f in common:
        g = f.replace('\\', '/').split('/')[0]
        groups[g].append((human[f], pseudo[f]))
    print('\n按浓度组分层的一致率：')
    for g in sorted(groups):
        arr = groups[g]
        a = sum(1 for h, p in arr if h == p) / len(arr)
        k = cohen_kappa([x[0] for x in arr], [x[1] for x in arr])
        print(f'  {g:>10s}: n={len(arr):>4d}  agreement={a:.3f}  kappa={k:.3f}')

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        agrees = []
        for c in range(3):
            idx_c = [i for i in range(n) if H[i] == c]
            agrees.append(sum(1 for i in idx_c if P[i] == c) / len(idx_c)
                          if idx_c else 0.0)
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        xs = np.arange(3)
        bars = ax.bar(xs, agrees, 0.5, color=['#2196F3', '#4CAF50', '#F44336'])
        for b, v in zip(bars, agrees):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f'{v:.3f}',
                    ha='center', fontsize=11, fontweight='bold')
        ax.axhline(acc, ls='--', lw=1.2, color='#555',
                   label=f'overall agreement = {acc:.3f}')
        ax.set_xticks(xs)
        ax.set_xticklabels([CLASS_NAMES[c] for c in range(3)], fontsize=11)
        ax.set_ylabel('Human–pseudo agreement', fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.set_title(f'Human agreement with operational labels  '
                     f'(kappa = {kappa:.3f}, 95% CI [{lo:.3f}, {hi:.3f}])',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(args.fig, dpi=300, bbox_inches='tight')
        print(f'\n图已保存：{args.fig}')
    except Exception as e:
        print(f'（画图跳过：{e}）')

    print('\n论文可用表述：')
    print(f'  "在类别均衡抽取的 {n} 帧样本上，人工标注与操作化标签的一致率为 '
          f'{acc*100:.1f}%（Cohen\'s kappa = {kappa:.3f}, 95% CI '
          f'[{lo:.3f}, {hi:.3f}]），表明所定义的姿态类别在形态学上具有'
          f'可辨识的一致性。"')


if __name__ == '__main__':
    main()
