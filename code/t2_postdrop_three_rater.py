"""t2_postdrop_three_rater.py —— 论文 3.2 表 2/表 3 的三人标注删后复算（T2 回填专用）。

背景
----
论文表 2/表 3 的删前数字（Fleiss κ 0.536/0.702、两两三对、判 Ω 比例、
分歧结构、0-处置敏感性）来自一次性脚本 _tmp_three.py / _tmp_fleiss.py，
跑在 annotation_v2/（三人 A/B/C，310 项）。噪声剔除后标注批次变为
annotation_v2_postdrop/（295 项），表 2/表 3 需按同口径复算。
step18_agreement_v2.py 按设计只报 A/B 两人的成对一致性，不含三人 Fleiss，
因此本脚本单独补齐三人侧的全部数字。

口径（与删前完全一致，勿改）
----
- 作答码：1=笔直、2=Ω、3=正弦；0=图像不可用；9=无法归类。
- 二值化：笔直(1) -> 直，其余(2/3) -> 弯。
- 涉及多人的统计一律限制在"唯一项"（is_repeat=0）上；
  三人 Fleiss 限制在三人均给出三类的共同唯一项上。
- 表 3 分歧结构按"两两共同有效项"逐对拆分。
- 人内重测：隐藏重复项 vs 其对应唯一项，逐人配对。
- 人–机：与 key.csv 的 pseudo_label（操作化伪标签）比，限唯一项。
- R1 预注册规则（与 step18_agreement_v2.py 一致）：
  人–人二值 κ ≥ 0.6 且 人–机二值 κ ≤ 0.30（人内二值 κ ≥ 0.5 为前提）。

用法
----
  python t2_postdrop_three_rater.py -d annotation_v2_postdrop
  python t2_postdrop_three_rater.py -d annotation_v2_postdrop \
      -o annotation_v2_postdrop/agreement_three_rater.txt
"""

import os
import csv
import argparse
import itertools
import numpy as np

# GUI 码 -> 语义
CODE_NAME = {0: '不可用', 1: '笔直', 2: 'Omega', 3: '正弦', 9: '无法归类'}
# 三类内部码（与 step18 一致）：0=正弦, 1=Ω, 2=直
CODE_TO_CLASS = {1: 2, 2: 1, 3: 0}
CLASS_NAME = {0: '正弦', 1: 'Ω', 2: '直'}


# ------------------------------------------------------------
# 读入
# ------------------------------------------------------------

def read_csv(path):
    if not os.path.exists(path):
        raise SystemExit(f'缺少文件：{path}')
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load_key(path):
    key = {}
    for r in read_csv(path):
        key[r['item_id']] = dict(
            rel_path=r['rel_path'].replace('\\', '/'),
            is_repeat=int(r['is_repeat']),
            group=r.get('group', ''),
            pseudo_label=int(r['pseudo_label']) if r.get('pseudo_label') else -1,
        )
    return key


def load_rater(dirpath, who):
    """返回 {item_id: code}，code ∈ {0,1,2,3,9}；无效作答不计入。"""
    p = os.path.join(dirpath, f'labels_{who}.csv')
    if not os.path.exists(p):
        p = os.path.join(dirpath, f'items_{who}.csv')
    ans = {}
    for r in read_csv(p):
        c = (r.get('label_code') or '').strip()
        if c in ('0', '1', '2', '3', '9'):
            ans[r['item_id']] = int(c)
    return ans


# ------------------------------------------------------------
# κ 统计
# ------------------------------------------------------------

def cohen_kappa(a, b, n_cat):
    """a/b 为内部码数组。返回 (kappa, n, 一致率)。"""
    cm = np.zeros((n_cat, n_cat), dtype=int)
    for x, y in zip(a, b):
        cm[x, y] += 1
    n = int(cm.sum())
    if n == 0:
        return float('nan'), 0, float('nan')
    po = np.trace(cm) / n
    pe = float((cm.sum(0) * cm.sum(1)).sum()) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else float('nan')
    return k, n, po


def fleiss_kappa(rows, n_cat):
    """rows: 每行为一名目上全体标注者的内部码列表；评人数 = len(rows[0])。"""
    mats = []
    for r in rows:
        mats.append(np.bincount(np.asarray(r, dtype=int), minlength=n_cat))
    mats = np.asarray(mats, dtype=float)
    n_raters = len(rows[0])          # 每个名目上的评人数（勿与类别数混淆）
    P = (mats ** 2).sum(1) - n_raters
    P /= n_raters * (n_raters - 1)
    Pbar = P.mean()
    pj = mats.sum(0) / mats.sum()
    Pe = float((pj ** 2).sum())
    return (Pbar - Pe) / (1 - Pe) if Pe < 1 else float('nan')


def bin_of(code):
    """内部码 -> 二值：2(直)=0，其余=1(弯)。"""
    return 0 if code == 2 else 1


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='三人标注删后复算（表 2/表 3 回填）')
    ap.add_argument('-d', '--dir', default='annotation_v2_postdrop',
                    help='标注批次目录（默认删后 annotation_v2_postdrop）')
    ap.add_argument('-o', '--out', default=None,
                    help='报告路径（默认 <dir>/agreement_three_rater.txt）')
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.dir, 'agreement_three_rater.txt')

    key = load_key(os.path.join(args.dir, 'key.csv'))
    R = {w: load_rater(args.dir, w) for w in ('A', 'B', 'C')}
    uniq = [i for i in key if key[i]['is_repeat'] == 0]

    lines = []
    def P(s=''):
        print(s)
        lines.append(s)

    P('=' * 62)
    P('三人标注一致性（表 2 / 表 3 删后复算；口径与删前 _tmp_three 一致）')
    P('=' * 62)
    P(f'批次目录：{args.dir}')
    P(f'题面：{len(key)} 项（唯一 {len(uniq)}，'
      f'隐藏重复 {len(key) - len(uniq)}）')

    # ---- ① 每人作答概况（唯一项） ----
    P('\n' + '-' * 62)
    P('① 每名标注者作答概况（唯一项；对应论文 3.2 覆盖面段）')
    P('-' * 62)
    for w in ('A', 'B', 'C'):
        a = R[w]
        u3 = [i for i in uniq if a.get(i) in (1, 2, 3)]
        n0 = sum(1 for i in uniq if a.get(i) == 0)
        n9 = sum(1 for i in uniq if a.get(i) == 9)
        dist = {c: sum(1 for i in u3 if a[i] == c) for c in (1, 2, 3)}
        om = dist[2] / len(u3) * 100 if u3 else float('nan')
        P(f'  {w}: 有效三类 {len(u3)}  判0 {n0}  判9 {n9}  '
          f'直/Ω/正弦 = {dist[1]}/{dist[2]}/{dist[3]}  判Ω {om:.1f}%')

    # ---- ② 两两一致性（唯一项，双方三类有效） ----
    P('\n' + '-' * 62)
    P('② 两两一致性（唯一项，双方均给出三类）')
    P('-' * 62)
    pair_common = {}
    for x, y in itertools.combinations('ABC', 2):
        com = [i for i in uniq
               if R[x].get(i) in (1, 2, 3) and R[y].get(i) in (1, 2, 3)]
        pair_common[(x, y)] = com
        ca = [CODE_TO_CLASS[R[x][i]] for i in com]
        cb = [CODE_TO_CLASS[R[y][i]] for i in com]
        k3, n, po3 = cohen_kappa(ca, cb, 3)
        k2, _, po2 = cohen_kappa([bin_of(c) for c in ca],
                                 [bin_of(c) for c in cb], 2)
        P(f'  {x}-{y}: n={n}  三类 κ={k3:+.3f}（一致率 {po3:.3f}）  '
          f'二值 κ={k2:+.3f}（一致率 {po2:.3f}）')

    # ---- ③ 三人共同子集 Fleiss（表 2 前两行） ----
    P('\n' + '-' * 62)
    P('③ 三人共同子集（三人均给出三类）：Fleiss κ')
    P('-' * 62)
    com3 = [i for i in uniq if all(R[w].get(i) in (1, 2, 3) for w in 'ABC')]
    rows3 = [[CODE_TO_CLASS[R[w][i]] for w in 'ABC'] for i in com3]
    kf3 = fleiss_kappa(rows3, 3)
    kf2 = fleiss_kappa([[bin_of(c) for c in r] for r in rows3], 2)
    P(f'  三人共同唯一项：{len(com3)}')
    P(f'  Fleiss κ（三类，三人）= {kf3:+.3f}')
    P(f'  Fleiss κ（二值，三人）= {kf2:+.3f}')
    for (x, y), com in pair_common.items():
        com_c = [i for i in com3]  # 共同子集上两两二值
        ca = [bin_of(CODE_TO_CLASS[R[x][i]]) for i in com_c]
        cb = [bin_of(CODE_TO_CLASS[R[y][i]]) for i in com_c]
        k2, n, _ = cohen_kappa(ca, cb, 2)
        P(f'  共同子集上 {x}-{y} 二值 κ={k2:+.3f}（n={n}）')

    # ---- ④ 表 3 分歧结构（两两共同有效项逐对拆分） ----
    P('\n' + '-' * 62)
    P('④ 表 3 分歧结构（两两共同有效项；Ω↔正弦互换 vs 跨弯／直）')
    P('-' * 62)
    tot_d = tot_om = tot_cross = 0
    for (x, y), com in pair_common.items():
        d = om_swap = cross = 0
        for i in com:
            a, b = R[x][i], R[y][i]
            if a == b:
                continue
            d += 1
            if {a, b} == {2, 3}:
                om_swap += 1
            elif 1 in (a, b):
                cross += 1
        pct = om_swap / d * 100 if d else float('nan')
        P(f'  {x}-{y}: 分歧 {d}  Ω↔正弦 {om_swap}（{pct:.0f}%）  '
          f'跨弯／直 {cross}')
        tot_d += d; tot_om += om_swap; tot_cross += cross
    if tot_d:
        P(f'  合计: 分歧 {tot_d}  Ω↔正弦 {tot_om}'
          f'（{tot_om / tot_d * 100:.0f}%）  跨弯／直 {tot_cross}'
          f'（{tot_cross / tot_d * 100:.0f}%）')

    # ---- ⑤ 人内重测（隐藏重复项） ----
    P('\n' + '-' * 62)
    P('⑤ 人内重测：隐藏重复项 vs 对应唯一项')
    P('-' * 62)
    path2uniq = {}
    for i in uniq:
        path2uniq.setdefault(key[i]['rel_path'], i)
    self_k2 = {}
    for w in ('A', 'B', 'C'):
        pairs = []
        for r in key:
            if key[r]['is_repeat'] != 1:
                continue
            u = path2uniq.get(key[r]['rel_path'])
            if u is None:
                continue
            c1, c2 = R[w].get(r), R[w].get(u)
            if c1 in (1, 2, 3) and c2 in (1, 2, 3):
                pairs.append((CODE_TO_CLASS[c1], CODE_TO_CLASS[c2]))
        if len(pairs) < 10:
            P(f'  {w}: [跳过] 可配对重复项不足（{len(pairs)}）')
            continue
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        k3, n, po3 = cohen_kappa(x, y, 3)
        k2, _, _ = cohen_kappa([bin_of(c) for c in x],
                               [bin_of(c) for c in y], 2)
        self_k2[w] = k2
        P(f'  {w}: n={n}  三类自一致 κ={k3:+.3f}（一致率 {po3:.3f}）  '
          f'二值自一致 κ={k2:+.3f}')

    # ---- ⑥ 人–机（key.csv 的 pseudo_label） ----
    P('\n' + '-' * 62)
    P('⑥ 人–机一致性（对照 key.csv 操作化伪标签，唯一项）')
    P('-' * 62)
    mach_k2 = {}
    for w in ('A', 'B', 'C'):
        sel = [i for i in uniq
               if R[w].get(i) in (1, 2, 3) and key[i]['pseudo_label'] >= 0]
        if len(sel) < 10:
            P(f'  {w}: [跳过] 有效 {len(sel)} 条')
            continue
        hu = [CODE_TO_CLASS[R[w][i]] for i in sel]
        ps = [key[i]['pseudo_label'] for i in sel]
        k3, n, _ = cohen_kappa(hu, ps, 3)
        k2, _, _ = cohen_kappa([bin_of(c) for c in hu],
                               [bin_of(c) for c in ps], 2)
        mach_k2[w] = k2
        P(f'  {w}: n={n}  三类 κ={k3:+.3f}  二值 κ={k2:+.3f}')
    if mach_k2:
        vals = list(mach_k2.values())
        P(f'  人–机二值 κ 三人均值 = {np.mean(vals):+.3f}'
          f'（范围 {min(vals):+.3f} 至 {max(vals):+.3f}）')

    # ---- ⑦ 0-处置敏感性（两两二值 κ） ----
    P('\n' + '-' * 62)
    P('⑦ “图像不可用(0)”处置敏感性（两两二值 κ）')
    P('-' * 62)
    for (x, y), com in pair_common.items():
        res = {}
        # a) 剔除 0：即 ② 的二值 κ
        ca = [bin_of(CODE_TO_CLASS[R[x][i]]) for i in com]
        cb = [bin_of(CODE_TO_CLASS[R[y][i]]) for i in com]
        res['剔除0'] = cohen_kappa(ca, cb, 2)
        # b/c) 把 0 分别计入弯/直：在全部唯一项上找双方作答 ∈ {0,1,2,3} 的项
        for tag, tgt in (('0计入弯', 1), ('0计入直', 0)):
            sel = [i for i in uniq
                   if R[x].get(i) in (0, 1, 2, 3)
                   and R[y].get(i) in (0, 1, 2, 3)]
            ca = [bin_of(CODE_TO_CLASS[R[x][i]]) if R[x][i] != 0 else tgt
                  for i in sel]
            cb = [bin_of(CODE_TO_CLASS[R[y][i]]) if R[y][i] != 0 else tgt
                  for i in sel]
            res[tag] = cohen_kappa(ca, cb, 2)
        P(f'  {x}-{y}: ' + '  '.join(
            f'{t} κ={v[0]:+.3f}(n={v[1]})' for t, v in res.items()))

    # ---- ⑧ R1 预注册判读要素 ----
    P('\n' + '=' * 62)
    P('⑧ 预注册判读要素（R1：人–人二值 κ ≥ 0.6 且 人–机二值 κ ≤ 0.30，'
      '人内二值 κ ≥ 0.5 为前提）')
    P('=' * 62)
    P(f'  人–人二值 κ（三人 Fleiss）= {kf2:+.3f}')
    if pair_common:
        pair2 = []
        for (x, y), com in pair_common.items():
            ca = [bin_of(CODE_TO_CLASS[R[x][i]]) for i in com]
            cb = [bin_of(CODE_TO_CLASS[R[y][i]]) for i in com]
            pair2.append(cohen_kappa(ca, cb, 2)[0])
        P(f'  两两二值 κ 范围 = {min(pair2):+.3f} 至 {max(pair2):+.3f}')
    if mach_k2:
        P(f'  人–机二值 κ 三人均值 = {np.mean(list(mach_k2.values())):+.3f}')
    if self_k2:
        P(f'  人内重复二值 κ 三人均值 = {np.mean(list(self_k2.values())):+.3f}')
    ok_hh = kf2 >= 0.6
    ok_hm = bool(mach_k2) and np.mean(list(mach_k2.values())) <= 0.30
    ok_self = bool(self_k2) and np.mean(list(self_k2.values())) >= 0.5
    if ok_hh and ok_hm and ok_self:
        verdict = ('R1', '轴真实存在、操作化规则未捕捉；4.3 结论维持并可加强')
    elif kf2 >= 0.6 and ok_self:
        verdict = ('R2', '人–机二值 κ ≥ 0.30：规则部分复现人的二值判断，'
                         '4.3 需相应弱化（请人工核对后改写）')
    else:
        verdict = ('R4?', '判读条件未齐（请人工核对 ⑧ 各分量后按预注册规则定性）')
    P(f'  判定：{verdict[0]} — {verdict[1]}')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n[写出] {out_path}')


if __name__ == '__main__':
    main()
