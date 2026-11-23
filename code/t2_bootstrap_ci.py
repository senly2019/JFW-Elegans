# -*- coding: utf-8 -*-
"""表 2 / 表 7 κ 值的 bootstrap 95% CI（与 t2_postdrop_three_rater.py 完全同口径）。

- 主数据（表 2）：annotation_v2_postdrop，唯一项（is_repeat==0）；
  Fleiss / 两两 / 人–机 / 人内重测 κ 全部给 percentile bootstrap CI。
- 跨数据集（表 7）：annotation_wc，labels_R1/R2 vs key.csv 规则类。
- 2000 次有放回 item 级重抽样，seed=42，取 2.5%/97.5% 分位。
"""
import csv
import itertools
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 2000
SEED = 42
CODE_TO_CLASS = {1: 2, 2: 1, 3: 0}   # 原始作答码 -> 类空间（0 正弦/1 Ω/2 直）


def bin_of(c):
    """类空间 -> 二值：2(直)=0，其余=1(弯)。"""
    return 0 if c == 2 else 1


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def cohen_kappa(a, b, n_cat):
    cm = np.zeros((n_cat, n_cat))
    for x, y in zip(a, b):
        cm[int(x), int(y)] += 1
    N = cm.sum()
    po = np.trace(cm) / N
    pe = (cm.sum(0) * cm.sum(1)).sum() / (N * N)
    return (po - pe) / (1 - pe)


def fleiss_kappa(rows, n_cat):
    mats = np.asarray([np.bincount(np.asarray(r, dtype=int), minlength=n_cat)
                       for r in rows], dtype=float)
    n_raters = len(rows[0])
    P = ((mats ** 2).sum(1) - n_raters) / (n_raters * (n_raters - 1))
    pj = mats.sum(0) / mats.sum()
    Pe = float((pj ** 2).sum())
    return (P.mean() - Pe) / (1 - Pe)


def boot_ci(fn, units, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(units)
    idx = rng.integers(0, n, (n_boot, n))
    vals = []
    for row in idx:
        v = fn([units[i] for i in row])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


def main():
    print('=== 表 2：主数据（annotation_v2_postdrop，唯一项）===')
    d = os.path.join(HERE, 'annotation_v2_postdrop')
    key = {r['item_id']: r for r in read_csv(os.path.join(d, 'key.csv'))}
    R = {}
    for w in 'ABC':
        for r in read_csv(os.path.join(d, f'labels_{w}.csv')):
            R.setdefault(w, {})[r['item_id']] = int(r['label_code'])
    uniq = [i for i in key if int(key[i]['is_repeat']) == 0]

    # Fleiss（三人共同三类子集）
    com3 = [i for i in uniq if all(R[w].get(i) in (1, 2, 3) for w in 'ABC')]
    rows3 = [tuple(CODE_TO_CLASS[R[w][i]] for w in 'ABC') for i in com3]
    rows2 = [tuple(bin_of(c) for c in r) for r in rows3]
    print(f'Fleiss 三类 κ = {fleiss_kappa(rows3, 3):+.3f}  '
          f'CI {tuple(map(lambda v: f"{v:+.3f}", boot_ci(lambda us: fleiss_kappa(us, 3), rows3)))}'
          f'  (n={len(com3)})')
    print(f'Fleiss 二值 κ = {fleiss_kappa(rows2, 2):+.3f}  '
          f'CI {tuple(map(lambda v: f"{v:+.3f}", boot_ci(lambda us: fleiss_kappa(us, 2), rows2)))}'
          f'  (n={len(com3)})')

    # 两两
    for x, y in itertools.combinations('ABC', 2):
        com = [i for i in uniq
               if R[x].get(i) in (1, 2, 3) and R[y].get(i) in (1, 2, 3)]
        a3 = [(CODE_TO_CLASS[R[x][i]], CODE_TO_CLASS[R[y][i]]) for i in com]
        a2 = [(bin_of(CODE_TO_CLASS[R[x][i]]),
               bin_of(CODE_TO_CLASS[R[y][i]])) for i in com]
        k3 = cohen_kappa([u[0] for u in a3], [u[1] for u in a3], 3)
        k2 = cohen_kappa([u[0] for u in a2], [u[1] for u in a2], 2)
        c3 = boot_ci(lambda us: cohen_kappa([u[0] for u in us],
                                            [u[1] for u in us], 3), a3)
        c2 = boot_ci(lambda us: cohen_kappa([u[0] for u in us],
                                            [u[1] for u in us], 2), a2)
        print(f'{x}-{y} (n={len(com)}): 三类 κ={k3:+.3f} CI '
              f'{tuple(map(lambda v: f"{v:+.3f}", c3))} | '
              f'二值 κ={k2:+.3f} CI {tuple(map(lambda v: f"{v:+.3f}", c2))}')

    # 人–机（pseudo_label）
    for w in 'ABC':
        sel = [i for i in uniq
               if R[w].get(i) in (1, 2, 3) and int(key[i]['pseudo_label']) >= 0]
        a3 = [(CODE_TO_CLASS[R[w][i]], int(key[i]['pseudo_label']))
              for i in sel]
        a2 = [(bin_of(CODE_TO_CLASS[R[w][i]]),
               bin_of(int(key[i]['pseudo_label']))) for i in sel]
        k3 = cohen_kappa([u[0] for u in a3], [u[1] for u in a3], 3)
        k2 = cohen_kappa([u[0] for u in a2], [u[1] for u in a2], 2)
        c3 = boot_ci(lambda us: cohen_kappa([u[0] for u in us],
                                            [u[1] for u in us], 3), a3)
        c2 = boot_ci(lambda us: cohen_kappa([u[0] for u in us],
                                            [u[1] for u in us], 2), a2)
        print(f'人–机 {w} (n={len(sel)}): 三类 κ={k3:+.3f} CI '
              f'{tuple(map(lambda v: f"{v:+.3f}", c3))} | '
              f'二值 κ={k2:+.3f} CI {tuple(map(lambda v: f"{v:+.3f}", c2))}')

    # 人内重测（隐藏重复项 vs 对应唯一项）
    path2uniq = {}
    for i in uniq:
        path2uniq.setdefault(key[i]['rel_path'], i)
    for w in 'ABC':
        pairs = []
        for i in key:
            if int(key[i]['is_repeat']) != 1:
                continue
            u = path2uniq.get(key[i]['rel_path'])
            if u is None:
                continue
            c1, c2 = R[w].get(i), R[w].get(u)
            if c1 in (1, 2, 3) and c2 in (1, 2, 3):
                pairs.append((CODE_TO_CLASS[c1], CODE_TO_CLASS[c2]))
        if len(pairs) < 10:
            print(f'{w}: 重测可配对不足（{len(pairs)}）')
            continue
        k3 = cohen_kappa([p[0] for p in pairs], [p[1] for p in pairs], 3)
        c3 = boot_ci(lambda us: cohen_kappa([u[0] for u in us],
                                            [u[1] for u in us], 3), pairs)
        print(f'{w} 重测 三类 κ={k3:+.3f} CI '
              f'{tuple(map(lambda v: f"{v:+.3f}", c3))}  (n={len(pairs)})')

    print()
    print('=== 表 7：跨数据集（annotation_wc）===')
    wc = os.path.join(HERE, 'annotation_wc')
    kwc = {r['item_id']: r for r in read_csv(os.path.join(wc, 'key.csv'))}
    H = {}
    for tag in ('R1', 'R2'):
        H[tag] = {r['item_id']: r.get('label_code', '').strip()
                  for r in read_csv(os.path.join(wc, f'labels_{tag}.csv'))}
    H2R = {'1': 2, '2': 1, '3': 0}
    for tag in ('R1', 'R2'):
        com = [i for i in kwc if i in H[tag] and H[tag][i] in H2R]
        a3 = [(H2R[H[tag][i]], int(kwc[i]['rule_label'])) for i in com]
        k3 = cohen_kappa([u[0] for u in a3], [u[1] for u in a3], 3)
        c3 = boot_ci(lambda us: cohen_kappa([u[0] for u in us],
                                            [u[1] for u in us], 3), a3)
        print(f'人–规则 {tag} 三类 κ={k3:+.3f} CI '
              f'{tuple(map(lambda v: f"{v:+.3f}", c3))}  (n={len(com)})')
    com = [i for i in kwc if i in H['R1'] and i in H['R2']
           and H['R1'][i] in H2R and H['R2'][i] in H2R]
    a3 = [(H2R[H['R1'][i]], H2R[H['R2'][i]]) for i in com]
    k3 = cohen_kappa([u[0] for u in a3], [u[1] for u in a3], 3)
    c3 = boot_ci(lambda us: cohen_kappa([u[0] for u in us],
                                        [u[1] for u in us], 3), a3)
    print(f'人–人 三类 κ={k3:+.3f} CI '
          f'{tuple(map(lambda v: f"{v:+.3f}", c3))}  (n={len(com)})')


if __name__ == '__main__':
    main()
