"""step18d_verdict_audit.py — 预注册判定的敏感性复核。

为什么需要
---------
step18 按预注册规则给出 R1（人–人二值 κ=0.641 ≥0.60，人–机二值 κ=0.277 ≤0.30）。
但这一判定有三处可能被审稿人打穿，必须自己先量化：

  (1) **0 的处置**：标注者 B 在 310 条里按了 35 次 0（图像不可用），A 一次没按。
      而人–人 κ 只在双方都给 1/2/3 的 221 个唯一项上计算——即"以 B 的可用性判断
      为条件"做了筛选。若 B 的 0 偏严，κ 被抬高；若偏松，则丢掉了真该看的难图。
      本脚本给出三种处置（剔除 / 计入弯曲 / 计入笔直）下的 κ。
  (2) **人–机 κ 的对比基数不同**：A 从未按 0，其人–机 κ 在 250 项上算；
      B 在 221 项上算。两者不可直接平均。本脚本强制在共同子集上重算。
  (3) **判定对"用哪个标注者"敏感**：A 单独 = R1，B 单独落在中间地带。
      预注册规则写的是"两人平均"，本脚本把单人/平均/共同子集三种都列出来。

另外把 0/非0 的一致性、以及各标注者的弯曲深度判别力（bbi-AUC）一并输出，
供论文 Methods 引用。

用法
----
  python step18d_verdict_audit.py -d annotation_v2
"""

import os
import csv
import argparse
import numpy as np

CODE_TO_CLASS = {1: 2, 2: 1, 3: 0}      # 作答码 -> 内部码（2=直 1=Ω 0=正弦）
CLASS_NAME = {0: 'Sinusoidal', 1: 'Omega', 2: 'Straight'}
FN = ['bbi', 'max_kappa', 'n_high_regions', 'mid_dominance',
      'peak_midness', 'kappa_std', 'periodicity_peaks']


def read_csv(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def kappa(a, b, n):
    a = np.asarray(a, int); b = np.asarray(b, int)
    cm = np.zeros((n, n))
    for x, y in zip(a, b):
        if 0 <= x < n and 0 <= y < n:
            cm[x, y] += 1
    N = cm.sum()
    if N == 0:
        return float('nan'), 0
    po = np.trace(cm) / N
    pe = (cm.sum(0) * cm.sum(1)).sum() / (N * N)
    return float((po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else 1.0), int(N)


def auc(score, label):
    s = np.asarray(score, float); y = np.asarray(label, int)
    m = np.isfinite(s); s, y = s[m], y[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sm = np.zeros(len(u)); np.add.at(sm, inv, ranks); ranks = (sm / cnt)[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dir', default='annotation_v2')
    ap.add_argument('-c', '--cache', default='atlas_out_full/kappa_cache.npz')
    ap.add_argument('-o', '--out', default='annotation_v2/verdict_audit.txt')
    args = ap.parse_args()

    lines = []

    def P(s=''):
        print(s)
        lines.append(s)

    key = {r['item_id']: r for r in read_csv(os.path.join(args.dir, 'key.csv'))}
    uniq = [i for i in key if key[i]['is_repeat'] == '0']
    A = {r['item_id']: int(r['label_code']) for r in
         read_csv(os.path.join(args.dir, 'labels_A.csv')) if r['label_code'].strip()}
    B = {r['item_id']: int(r['label_code']) for r in
         read_csv(os.path.join(args.dir, 'labels_B.csv')) if r['label_code'].strip()}

    z = np.load(args.cache, allow_pickle=True)
    files = [str(x).replace('\\', '/') for x in z['filenames']]
    feats = np.asarray(z['features'], np.float32)
    idx = {f: i for i, f in enumerate(files)}
    kappa_mat = z['kappa'].astype(np.float32)
    pc1 = kappa_mat  # 占位，下面统一 PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1, random_state=0).fit(kappa_mat)
    bdi_arr = pca.transform(kappa_mat)[:, 0]
    i_bbi = 0
    corr = float(np.corrcoef(bdi_arr, feats[:, i_bbi].astype(float))[0, 1])
    if corr < 0:
        bdi_arr = -bdi_arr

    P('=' * 74)
    P('预注册判定敏感性复核（step18d）')
    P('=' * 74)

    # ---------- 0. 0 与 9 的分布 ----------
    P('\n[0] 不可用(0) 与 无法归类(9) 的作答分布')
    for nm, ans in (('A', A), ('B', B)):
        u = [i for i in uniq if i in ans]
        n0 = sum(1 for i in u if ans[i] == 0)
        n9 = sum(1 for i in u if ans[i] == 9)
        P(f'  {nm}: 唯一项 {len(u)} 条中  0 = {n0}（{n0/len(u)*100:.1f}%）  '
          f'9 = {n9}（{n9/len(u)*100:.1f}%）')
    # 0 的一致性（A 恒为非 0，天然为 0）
    both0 = [i for i in uniq if A.get(i) == 0 and B.get(i) == 0]
    P(f'  双方都按 0 的唯一项：{len(both0)}；'
      f'"是否不可用"这一判定本身两人零交集' if not both0 else '')
    # B 的 0 对应图的客观体检
    aud = {r['rel_path'].replace('\\', '/'): r['flags']
           for r in read_csv(os.path.join(args.dir, 'unusable_audit.csv'))}
    b0 = [i for i in uniq if B.get(i) == 0]
    flagged = sum(1 for i in b0 if aud.get(key[i]['rel_path'], 'ok') != 'ok')
    P(f'  B 判 0 的 {len(b0)} 个唯一项里，客观体检也有缺陷的仅 {flagged} 个；'
      f'其余 {len(b0)-flagged} 个是 B 自己判定不可用')
    if b0:
        bs = [feats[idx[key[i]['rel_path']]][0] for i in b0
              if key[i]['rel_path'] in idx]
        ns = [feats[idx[key[i]['rel_path']]][0] for i in uniq
              if i in B and B[i] != 0 and key[i]['rel_path'] in idx]
        P(f'    B 判 0 的 bbi 均值 {np.mean(bs):.3f} vs 判非 0 的 {np.mean(ns):.3f}'
          f'（差 {np.mean(bs)-np.mean(ns):+.3f}）——若接近 0，说明 0 与弯曲无关，是另一套标准')

    # ---------- 1. 人–人，三种 0 处置 ----------
    P('\n[1] 人–人一致性：0 的三种处置')
    common = [i for i in uniq if A.get(i) in (1, 2, 3) and B.get(i) in (1, 2, 3)]
    ca = [CODE_TO_CLASS[A[i]] for i in common]
    cb = [CODE_TO_CLASS[B[i]] for i in common]
    k3, n3 = kappa(ca, cb, 3)
    k2, n2 = kappa([0 if c == 2 else 1 for c in ca],
                   [0 if c == 2 else 1 for c in cb], 2)
    P(f'  剔除 B 的 0（n={len(common)}）：三类 κ={k3:+.3f}  二值 κ={k2:+.3f}  '
      f'一致率 {np.mean([x == y for x, y in zip(ca, cb)]):.3f}')
    # 计入弯曲：0 -> 当作弯曲（内部 0 正弦）
    ca2 = [CODE_TO_CLASS.get(A[i], 0) for i in uniq]
    cb2 = [CODE_TO_CLASS.get(B[i], 0) for i in uniq]
    k2b, n2b = kappa([0 if c == 2 else 1 for c in ca2],
                     [0 if c == 2 else 1 for c in cb2], 2)
    k3b, _ = kappa(ca2, cb2, 3)
    P(f'  把 B 的 0 计入弯曲（n={n2b}）：二值 κ={k2b:+.3f}  三类 κ={k3b:+.3f}')
    # 计入笔直
    ca3 = [CODE_TO_CLASS.get(A[i], 2) for i in uniq]
    cb3 = [CODE_TO_CLASS.get(B[i], 2) for i in uniq]
    k2c, _ = kappa([0 if c == 2 else 1 for c in ca3],
                   [0 if c == 2 else 1 for c in cb3], 2)
    P(f'  把 B 的 0 计入笔直：二值 κ={k2c:+.3f}')
    ok60 = min(k2, k2b, k2c) >= 0.60
    P(f'  -> 三种处置的区间：{min(k2, k2b, k2c):+.3f} ~ {max(k2, k2b, k2c):+.3f}'
      f'（{"全部达到 0.60，判定对 0 的处置不敏感" if ok60 else "跨越 0.60，判定对 0 的处置敏感"}）')

    # ---------- 2. 人–机，共同子集 ----------
    P('\n[2] 人–机一致性：强制在同一共同子集（n=%d）上重算' % len(common))
    hm = {}
    for nm, ans in (('A', A), ('B', B)):
        hu = [CODE_TO_CLASS[ans[i]] for i in common]
        ps = [int(key[i]['pseudo_label']) for i in common]
        ka, _ = kappa(hu, ps, 3)
        kb, _ = kappa([0 if c == 2 else 1 for c in hu],
                      [0 if c == 2 else 1 for c in ps], 2)
        P(f'  {nm}: 三类 κ={ka:+.3f}   二值 κ={kb:+.3f}')
        hm[nm] = kb
    mean_hm = np.mean([hm['A'], hm['B']])
    P(f'  两人平均二值 κ = {mean_hm:+.3f}'
      f'（{"≤0.30" if mean_hm <= 0.30 else "0.30–0.50 中间地带"}）')
    P(f'  单人落点：A {hm["A"]:+.3f} → '
      f'{"≤0.30（R1 区）" if hm["A"] <= 0.30 else "中间地带"}；'
      f'B {hm["B"]:+.3f} → {"≤0.30（R1 区）" if hm["B"] <= 0.30 else "中间地带"}')

    # ---------- 3. BDI 效标 ----------
    P('\n[3] BDI 效标（共同子集）')
    for nm, ans in (('A', A), ('B', B)):
        sc = [bdi_arr[idx[key[i]['rel_path']]] for i in common
              if key[i]['rel_path'] in idx]
        lb = [0 if CODE_TO_CLASS[ans[i]] == 2 else 1 for i in common
              if key[i]['rel_path'] in idx]
        P(f'  {nm}: AUC = {auc(sc, lb):.3f}（弯 {sum(lb)} / 直 {len(lb)-sum(lb)}）')
    # bbi 判别力
    P('\n  各标注者"弯/直"的 bbi-AUC（独立于 BDI）：')
    for nm, ans in (('A', A), ('B', B)):
        sc = [feats[idx[key[i]['rel_path']]][0] for i in common if key[i]['rel_path'] in idx]
        lb = [0 if CODE_TO_CLASS[ans[i]] == 2 else 1 for i in common
              if key[i]['rel_path'] in idx]
        P(f'    {nm}: {auc(sc, lb):.3f}')

    # ---------- 4. 预注册判定 ----------
    P('\n[4] 预注册判定（按 step18 的操作化：人–机取两人平均）')
    P(f'  人–人二值 κ = {k2:+.3f}   人–机二值 κ = {mean_hm:+.3f}   '
      f'人内重复二值 κ = (A .880 / B .944)')
    if k2 >= 0.6 and mean_hm <= 0.30:
        P('  判定：R1 — 轴真实存在、操作化规则未捕捉；4.3 结论维持并可加强')
    elif k2 >= 0.6 and mean_hm >= 0.50:
        P('  判定：R2 — 仅三类层级过度切分，二值可用')
    elif k2 < 0.40:
        P('  判定：R3 — 标注不可靠；4.3 整节撤回')
    else:
        P('  判定：R4 — 证据不足；4.3 降级为"未能验证"')
    P('  敏感性：若只用 B，人–机二值 κ=+0.374 落在 0.30–0.50 中间地带（R4）；'
      '若只用 A，则为 R1。')
    P('  因此论文必须同时报告两人数值与平均，不能只报有利于结论的那一个。')

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n已写出：{args.out}')


if __name__ == '__main__':
    main()
