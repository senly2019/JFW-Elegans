"""make_annotation_batch_v2.py — 双标注者 + 隐藏重复 的人工标注批次生成器。

为什么需要 v2
-------------
现有 210 帧标注是**单人、无重复**，因此无法估计标注噪声，"类别不可感知"与
"标注者失误"两种解释在数据上不可区分（详见 人工标注有效性_风险与补实验方案.md）。
v2 通过三件事把它变成可判读的实验：
  ① 两名标注者各标一遍（人–人一致性）；
  ② 隐藏重复 60 帧（同一人前后各标一次 → 标注噪声上限）；
  ③ 按弯曲深度分层、刻意纳入中间地带（避免只测到易例）。

设计
----
  分层：组（4）× bbi 十分位（10）= 40 格，每格约 6 帧 → 250 帧
  重复：从 250 帧中抽 60 帧，以新编号混入（标注者不知哪条是重复）
  输出：items_A.csv / items_B.csv（仅编号与路径，无标签，顺序不同）
        key.csv（真实映射：编号 → 路径、是否重复、组/日期/行/虫、bbi）
        exemplars/（每类 5 张候选示例，供开标前统一口径）

标注者只需在 Excel 里填三列：
  label_code  1=笔直  2=Ω 形（单深弯）  3=正弦形（多波浪/其他）
  confidence  1=不确定  2=一般  3=确定
  seconds     该帧看了多少秒（大致即可）

用法（ele 环境）
--------------
  python make_annotation_batch_v2.py -c atlas_out_full/kappa_cache.npz \
      --root $WORM_DATA_ROOT -o annotation_v2
"""

import os
import csv
import shutil
import argparse
import numpy as np

from wormpath import get_data_root

FEATURE_NAMES = ['bbi', 'max_kappa', 'n_high_regions', 'mid_dominance',
                 'peak_midness', 'kappa_std', 'periodicity_peaks']
GROUP_ORDER = ['control', '5mg', '8mg', '10mg']


def write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'  已写 {path}（{len(rows)} 行）')


def pseudo_label(feats):
    """0=正弦（兜底）1=Ω 2=笔直，与论文 2.4 判据同口径。"""
    i_bbi = FEATURE_NAMES.index('bbi')
    i_mk = FEATURE_NAMES.index('max_kappa')
    i_nr = FEATURE_NAMES.index('n_high_regions')
    lab = np.zeros(len(feats), dtype=int)
    lab[(feats[:, i_nr] <= 1) & (feats[:, i_mk] >= 0.35)] = 1
    lab[feats[:, i_bbi] < 0.16] = 2
    return lab


def main():
    ap = argparse.ArgumentParser(description='双标注者+重复 的标注批次生成')
    ap.add_argument('-c', '--cache', default='atlas_out_full/kappa_cache.npz')
    ap.add_argument('--root', default=get_data_root())
    ap.add_argument('-o', '--outdir', default='annotation_v2')
    ap.add_argument('--n-frames', type=int, default=250)
    ap.add_argument('--n-repeat', type=int, default=60)
    ap.add_argument('--n-exemplar', type=int, default=5, help='每类示例张数')
    ap.add_argument('--n-bins', type=int, default=10, help='bbi 分层数')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    z = np.load(args.cache, allow_pickle=True)
    feats = np.asarray(z['features'], dtype=np.float32)
    files = np.asarray([str(f) for f in z['filenames']])
    groups = z['groups']
    dates, rows_w, worms = z['dates'], z['rows'], z['worms']
    n = len(feats)
    print(f'[载入] {n:,} 帧')

    bbi = feats[:, FEATURE_NAMES.index('bbi')]
    lab = pseudo_label(feats)
    rng = np.random.RandomState(args.seed)

    # ---- 分层抽样：组 × bbi 十分位 ----
    cells = {}
    for g in GROUP_ORDER:
        m = groups == g
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        qs = np.quantile(bbi[idx], np.linspace(0, 1, args.n_bins + 1))
        for b in range(args.n_bins):
            lo, hi = qs[b], qs[b + 1]
            sel = idx[(bbi[idx] >= lo) & (bbi[idx] <= hi if b == args.n_bins - 1
                                          else bbi[idx] < hi)]
            if len(sel):
                cells[(g, b)] = sel
    per_cell = max(1, args.n_frames // max(len(cells), 1))
    picked = []
    for key in sorted(cells):
        sel = cells[key]
        k = min(per_cell, len(sel))
        picked += list(rng.choice(sel, k, replace=False))
    picked = np.array(sorted(set(picked)))
    # 补足/截断到目标帧数
    if len(picked) < args.n_frames:
        rest = np.setdiff1d(np.arange(n), picked)
        extra = rng.choice(rest, min(args.n_frames - len(picked), len(rest)),
                           replace=False)
        picked = np.sort(np.concatenate([picked, extra]))
    picked = picked[:args.n_frames]
    print(f'[抽样] {len(picked)} 帧（{len(cells)} 格，每格约 {per_cell}）')

    # 检查图像是否可读（只查存在性，不读像素，快）
    ok = []
    for i in picked:
        p = os.path.join(args.root, files[i].replace('\\', '/'))
        if os.path.exists(p):
            ok.append(i)
    if len(ok) < len(picked):
        print(f'[警告] {len(picked) - len(ok)} 帧图像不存在，已剔除')
    picked = np.array(ok)
    print(f'[可用] {len(picked)} 帧')

    # ---- 重复项：从已抽帧中再取 n_repeat，作为新编号混入 ----
    rep_src = rng.choice(picked, min(args.n_repeat, len(picked)), replace=False)

    # ---- 组装 item 列表 ----
    items = []          # (item_id, base_index, is_repeat)
    for j, i in enumerate(picked):
        items.append((f'F{j + 1:03d}', int(i), 0))
    for j, i in enumerate(rep_src):
        items.append((f'R{j + 1:03d}', int(i), 1))
    print(f'[批次] 唯一 {len(picked)} + 重复 {len(rep_src)} = {len(items)} 条/人')

    key_rows = []
    for iid, i, is_rep in items:
        key_rows.append(dict(
            item_id=iid, rel_path=files[i].replace('\\', '/'),
            is_repeat=is_rep, group=groups[i], date=dates[i],
            row=rows_w[i], worm=worms[i],
            bbi=round(float(bbi[i]), 4), pseudo_label=int(lab[i])))
    write_csv(os.path.join(args.outdir, 'key.csv'), key_rows,
              ['item_id', 'rel_path', 'is_repeat', 'group', 'date', 'row',
               'worm', 'bbi', 'pseudo_label'])

    # ---- 两名标注者的题面（顺序不同，均不含标签）----
    for name, seed in (('A', 1), ('B', 2)):
        order = np.random.RandomState(args.seed + seed).permutation(len(items))
        rows = [dict(item_id=items[k][0], rel_path=files[items[k][1]].replace('\\', '/'),
                     label_code='', confidence='', seconds='') for k in order]
        write_csv(os.path.join(args.outdir, f'items_{name}.csv'), rows,
                  ['item_id', 'rel_path', 'label_code', 'confidence', 'seconds'])

    # ---- 示例候选（每类取极端值，供开标前统一口径）----
    ex_dir = os.path.join(args.outdir, 'exemplars')
    os.makedirs(ex_dir, exist_ok=True)
    i_bbi = FEATURE_NAMES.index('bbi')
    i_mk = FEATURE_NAMES.index('max_kappa')
    i_nr = FEATURE_NAMES.index('n_high_regions')
    picks = {
        'straight': np.argsort(bbi)[:200],
        'omega': np.argsort(-(feats[:, i_mk] * (feats[:, i_nr] <= 1)))[:200],
        'sine': np.argsort(-feats[:, i_nr])[:200],
    }
    made = 0
    for cls, cand in picks.items():
        got = 0
        for i in cand:
            src = os.path.join(args.root, files[i].replace('\\', '/'))
            if not os.path.exists(src):
                continue
            dst = os.path.join(ex_dir, f'exemplar_{cls}_{got + 1:02d}.png')
            try:
                shutil.copyfile(src, dst)
                got += 1
                made += 1
            except Exception:
                continue
            if got >= args.n_exemplar:
                break
        print(f'  示例 {cls}: {got} 张')
    print(f'[示例] 共 {made} 张 → {ex_dir}')

    print('\n下一步：')
    print('  1. 先把 exemplars/ 里的示例过一遍，剔掉不像的，再给标注者看')
    print('  2. 两名标注者分别填 items_A.csv / items_B.csv 的 label_code 三列')
    print('  3. 汇总时用 key.csv 还原真实路径与是否重复（key.csv 不要给标注者）')


if __name__ == '__main__':
    main()
