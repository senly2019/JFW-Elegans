"""step18_agreement_v2.py — 双标注者批次的一致性、标注噪声与效标效度汇总。

用途
----
配合 make_annotation_batch_v2.py 产出的三份 CSV：
  key.csv       真值侧（item_id / rel_path / is_repeat / 组别 / bbi / pseudo_label）
  items_A.csv   标注者 A 的作答（item_id / label_code / confidence / seconds）
  items_B.csv   标注者 B 的作答

回答四个此前无法回答的问题
  ① 人–人一致性：两名标注者之间三类 κ 与二值（弯 vs 直）κ 各是多少？
  ② 标注噪声：同一人前后两次看同一帧（隐藏重复项，R 编号）一致不一致？
     —— 这是判断"人机不一致究竟是类别不可分、还是标注本身乱"的关键。
  ③ 人–机一致性：与操作化伪标签的三类/二值 κ（可与旧 210 帧实验直接对比）。
  ④ 效标效度：以人的二值判断为金标准，连续弯曲深度轴（BDI）的 ROC AUC
     与最佳切点处的 κ，即"这条轴是否落在人真正在用的那根线上"。

预注册判读规则（实施前已声明，见 4.3 节）
  R1  人–人二值 κ ≥ 0.6 且 人–机二值 κ ≤ 0.30
        -> 轴真实存在，但操作化规则未捕捉到它；4.3 结论维持并可加强
  R2  人–人二值 κ ≥ 0.6 且 人–机二值 κ ≥ 0.50
        -> 仅"三类层级"过度切分，二值可用；4.3 结论限定为三类过度切分
  R3  人–人二值 κ < 0.40
        -> 标注不可靠；4.3 整节结论撤回，主结论改由 4.4.1 结构证据承担
  R4  其余（中间地带或人内重复 κ < 0.5）
        -> 证据不足，4.3 降级为"未能验证"，不得据此推断类别不存在

label_code 口径（与 human_label_gui.py 一致）
  1 = 笔直(straight, 内部码 2)
  2 = Ω 型(omega,     内部码 1)
  3 = 正弦型(sine,    内部码 0)

用法（ele 环境）
--------------
  python step18_agreement_v2.py -d annotation_v2 ^
      -c atlas_out_full/kappa_cache.npz -o annotation_v2/agreement

  # 不用缓存（BDI 效标段跳过，仅报 κ）
  python step18_agreement_v2.py -d annotation_v2 --no-cache

产物
----
  agreement_v2.txt   人可读报告（含预注册判读结论）
  agreement_v2.csv   逐项数字（便于做表）
"""

import os
import csv
import argparse
import numpy as np

GROUP_ORDER = ['control', '5mg', '8mg', '10mg']
CODE_TO_CLASS = {1: 2, 2: 1, 3: 0}          # GUI 码 -> 内部码（2=直,1=Ω,0=正弦）
CLASS_NAMES = {0: 'Sinusoidal', 1: 'Omega', 2: 'Straight'}


# ============================================================
# 读入
# ============================================================

def read_csv(path):
    if not os.path.exists(path):
        raise SystemExit(f'缺少文件：{path}')
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load_key(path):
    rows = read_csv(path)
    key = {}
    for r in rows:
        key[r['item_id']] = dict(
            rel_path=r['rel_path'].replace('\\', '/'),
            is_repeat=int(r['is_repeat']),
            group=r.get('group', ''),
            date=r.get('date', ''),
            bbi=float(r['bbi']) if r.get('bbi') else float('nan'),
            pseudo_label=int(r['pseudo_label']) if r.get('pseudo_label') else -1,
        )
    return key


def count_valid(path):
    """统计某文件里非空 label_code 的条数（用于自动挑选正确的那份作答文件）。"""
    if not os.path.exists(path):
        return 0
    n = 0
    try:
        for r in read_csv(path):
            if (r.get('label_code') or '').strip():
                n += 1
    except Exception:
        return 0
    return n


def resolve_answer_file(dirpath, who, explicit=None):
    """定位标注者 <who> 的作答文件。

    兼容两种工作流，按"有效作答数最多"择优：
      labels_X.csv  human_label_gui_v2.py 的输出（推荐）
      items_X.csv   直接在表格里填 label_code 的旧做法
    返回 (path 或 None, 有效条数)。
    """
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f'指定的作答文件不存在：{explicit}')
        return explicit, count_valid(explicit)
    cands = [os.path.join(dirpath, f'labels_{who}.csv'),
             os.path.join(dirpath, f'items_{who}.csv')]
    scored = [(count_valid(p), p) for p in cands]
    scored = [x for x in scored if x[0] > 0]
    if not scored:
        return None, 0
    scored.sort(reverse=True)
    return scored[0][1], scored[0][0]


def load_answers(path, key):
    """返回 {item_id: (code, confidence, seconds)}。

    code 为原始作答编码：1/2/3 为三类，0 为图像不可用，9 为无法归类。
    不做映射，以便 0/9 能被单独统计（它们本身是研究数据）。
    path 为 None（该标注者尚未作答）时返回空字典，各段自动跳过。
    """
    if not path:
        return {}
    rows = read_csv(path)
    out = {}
    for r in rows:
        iid = r['item_id']
        raw = (r.get('label_code') or '').strip()
        code = None
        if raw:
            try:
                code = int(float(raw))
            except ValueError:
                code = None
        try:
            sec = float(r['seconds']) if (r.get('seconds') or '').strip() else float('nan')
        except ValueError:
            sec = float('nan')
        try:
            conf = float(r['confidence']) if (r.get('confidence') or '').strip() else float('nan')
        except ValueError:
            conf = float('nan')
        out[iid] = (code, conf, sec)
    return out


# ============================================================
# 统计
# ============================================================

def kappa(a, b, n_cls):
    """Cohen's κ（手算，避免 sklearn 版本差异）。返回 (kappa, po, pe, n)。"""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if len(a) == 0:
        return float('nan'), float('nan'), float('nan'), 0
    cm = np.zeros((n_cls, n_cls), dtype=float)
    for x, y in zip(a, b):
        if 0 <= x < n_cls and 0 <= y < n_cls:
            cm[x, y] += 1
    n = cm.sum()
    if n == 0:
        return float('nan'), float('nan'), float('nan'), 0
    po = float(np.trace(cm) / n)
    pe = float((cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n))
    kap = (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else (1.0 if po == 1 else 0.0)
    return float(kap), po, pe, int(n)


def kappa_ci(a, b, n_cls, n_boot=2000, seed=0):
    """κ 的自助法 95% CI。"""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = len(a)
    if n < 5:
        return float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        k, _, _, _ = kappa(a[idx], b[idx], n_cls)
        if k == k:
            vals.append(k)
    if not vals:
        return float('nan'), float('nan')
    vals.sort()
    return float(vals[int(0.025 * len(vals))]), float(vals[int(0.975 * len(vals))])


def bin_of(cls):
    """二值化：0/1 = 弯曲（正弦或 Ω），2 = 笔直 -> 0。"""
    return 0 if cls == 2 else 1


def roc_auc(score, label):
    """Mann-Whitney U 形式的 AUC；label=1 为正类。返回 (auc, n_pos, n_neg)。"""
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float('nan'), n1, n0
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # 并列取平均秩
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    r1 = ranks[y == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0)), n1, n0


def best_cut(score, label):
    """在候选分位点上扫描，返回使 (acc 最大, 其次 κ 最大) 的切点与指标。"""
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    if len(s) < 10:
        return dict(thr=float('nan'), acc=float('nan'), kappa=float('nan'),
                    sens=float('nan'), spec=float('nan'))
    cand = np.unique(np.quantile(s, np.linspace(0.05, 0.95, 91)))
    best = None
    for t in cand:
        pred = (s >= t).astype(int)
        acc = float((pred == y).mean())
        k, _, _, _ = kappa(y, pred, 2)
        key = (acc, k if k == k else -9)
        if best is None or key > best[0]:
            best = (key, t, acc, k)
    _, t, acc, k = best
    pred = (s >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max(1, int((y == 1).sum()))
    spec = tn / max(1, int((y == 0).sum()))
    return dict(thr=float(t), acc=acc, kappa=float(k), sens=sens, spec=spec)


# ============================================================
# BDI（可选）：从缓存重算连续轴，用于效标段
# ============================================================

def bdi_lookup(cache_path, wanted):
    """对缓存做 PCA，取第一主成分作为 BDI，返回 {rel_path: bdi}。

    符号统一：使 BDI 与缓存里的 bbi 特征正相关（弯曲越深 BDI 越大）。
    只对 wanted 集合做字典，控制内存。
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print('[BDI] 环境无 sklearn，跳过效标段')
        return None
    if not os.path.exists(cache_path):
        print(f'[BDI] 缓存不存在（{cache_path}），跳过效标段')
        return None
    z = np.load(cache_path, allow_pickle=True)
    files = [str(x).replace('\\', '/') for x in z['filenames']]
    kappa_mat = z['kappa'].astype(np.float32)
    feats = z['features'] if 'features' in z else None
    p = PCA(n_components=1, random_state=0).fit(kappa_mat)
    pc1 = p.transform(kappa_mat)[:, 0]
    if feats is not None:
        i_bbi = 0                      # cluster_atlas 的 FEATURE_NAMES[0] = 'bbi'
        try:
            corr = float(np.corrcoef(pc1, np.asarray(feats)[:, i_bbi].astype(float))[0, 1])
        except Exception:
            corr = 0.0
        if corr < 0:
            pc1 = -pc1
        print(f'[BDI] PC1 解释方差 {float(p.explained_variance_ratio_[0]):.3f}，'
              f'与 bbi 相关 {abs(corr):+.3f}（已按此定向）')
    else:
        print('[BDI] 缓存无 features，符号未定向（效标 AUC 可能 <0.5，取 1-AUC 解读）')
    return {f: float(v) for f, v in zip(files, pc1) if f in wanted}


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser(description='双标注者一致性 + 标注噪声 + 效标效度')
    ap.add_argument('-d', '--dir', default='annotation_v2', help='批次目录')
    ap.add_argument('-c', '--cache', default='atlas_out_full/kappa_cache.npz')
    ap.add_argument('-o', '--out-prefix', default=None,
                    help='输出前缀（默认 <dir>/agreement）')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--no-cache', action='store_true', help='跳过 BDI 效标段')
    ap.add_argument('--file-a', default=None, help='标注者 A 作答文件（默认自动找 labels_A.csv）')
    ap.add_argument('--file-b', default=None, help='标注者 B 作答文件（默认自动找 labels_B.csv）')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    key = load_key(os.path.join(args.dir, 'key.csv'))
    path_a, n_a = resolve_answer_file(args.dir, 'A', args.file_a)
    path_b, n_b = resolve_answer_file(args.dir, 'B', args.file_b)
    A = load_answers(path_a, key)
    B = load_answers(path_b, key)
    pref = args.out_prefix or os.path.join(args.dir, 'agreement')
    os.makedirs(os.path.dirname(pref) or '.', exist_ok=True)

    lines = []
    def P(s=''):
        print(s)
        lines.append(s)

    def n_valid(ans):
        return sum(1 for v in ans.values() if v[0] is not None)

    P('=' * 62)
    P('双标注者批次一致性汇总')
    P('=' * 62)
    P(f'题面：{len(key)} 条（唯一 '
      f'{sum(1 for k in key.values() if k["is_repeat"] == 0)}，'
      f'隐藏重复 {sum(1 for k in key.values() if k["is_repeat"] == 1)}）')
    P(f"作答文件 A：{path_a or '（未找到）'}")
    P(f"作答文件 B：{path_b or '（未找到）'}")
    if not path_b:
        P('  [注意] 只有 A 一份作答：① 人–人一致性 与 ⑦ 预注册判读中的 R1/R2/R3')
        P('         无法给出，本次仅报 A 的人–机一致性、人内重复与作答行为。')
    P(f'有效作答：A {n_valid(A)} / {len(A)}，B {n_valid(B)} / {len(B)}')
    miss = [i for i, v in A.items() if v[0] is None]
    if miss:
        P(f'  [提示] A 有 {len(miss)} 条未标或无效（如 {miss[:5]}）')
    miss_b = [i for i, v in B.items() if v[0] is None]
    if miss_b:
        P(f'  [提示] B 有 {len(miss_b)} 条未标或无效')

    # 只用唯一项做人际/人机比较（避免重复项翻倍加权）
    uniq = [i for i in key if not key[i]['is_repeat']]
    both = [i for i in uniq if A.get(i, (None,))[0] in (1, 2, 3)
            and B.get(i, (None,))[0] in (1, 2, 3)]

    res = {}

    # ---------- ① 人–人 ----------
    P('\n' + '-' * 62)
    P('① 人–人一致性（唯一项，n = %d）' % len(both))
    P('-' * 62)
    if len(both) >= 10:
        cl_a = [CODE_TO_CLASS[A[i][0]] for i in both]
        cl_b = [CODE_TO_CLASS[B[i][0]] for i in both]
        k3, po3, pe3, n3 = kappa(cl_a, cl_b, 3)
        lo3, hi3 = kappa_ci(cl_a, cl_b, 3, args.n_boot, args.seed)
        ba = [bin_of(c) for c in cl_a]
        bb = [bin_of(c) for c in cl_b]
        k2, po2, pe2, n2 = kappa(ba, bb, 2)
        lo2, hi2 = kappa_ci(ba, bb, 2, args.n_boot, args.seed)
        P(f'  三类 : κ = {k3:+.3f}  95%CI [{lo3:+.3f}, {hi3:+.3f}]  一致率 {po3:.3f}')
        P(f'  二值 : κ = {k2:+.3f}  95%CI [{lo2:+.3f}, {hi2:+.3f}]  一致率 {po2:.3f}')
        res.update(k3=k3, k3_lo=lo3, k3_hi=hi3, po3=po3,
                   k2=k2, k2_lo=lo2, k2_hi=hi2, po2=po2, n_both=len(both))
        # 各自类别分布
        for nm, cl in (('A', cl_a), ('B', cl_b)):
            cnt = np.bincount(cl, minlength=3)
            P(f'  {nm} 分布: ' + '  '.join(
                f'{CLASS_NAMES[c]}={cnt[c]}' for c in range(3)))
    else:
        P('  [跳过] 共同有效作答不足 10 条')

    # ---------- ② 标注噪声（人内重复） ----------
    P('\n' + '-' * 62)
    P('② 标注噪声：同一人前后两次（隐藏重复项）')
    P('-' * 62)
    rep_ids = [i for i in key if key[i]['is_repeat'] == 1]
    # rel_path -> 唯一项 item_id
    path2uniq = {}
    for i in uniq:
        path2uniq.setdefault(key[i]['rel_path'], i)
    for nm, ans in (('A', A), ('B', B)):
        pairs = []
        for r in rep_ids:
            u = path2uniq.get(key[r]['rel_path'])
            if u is None:
                continue
            c1, c2 = ans.get(r, (None,))[0], ans.get(u, (None,))[0]
            if c1 in (1, 2, 3) and c2 in (1, 2, 3):
                pairs.append((CODE_TO_CLASS[c1], CODE_TO_CLASS[c2]))
        if len(pairs) >= 10:
            x = [p[0] for p in pairs]
            y = [p[1] for p in pairs]
            kk, po, pe, nn = kappa(x, y, 3)
            kb, pob, _, _ = kappa([bin_of(c) for c in x], [bin_of(c) for c in y], 2)
            P(f'  {nm}: n={nn}  三类自一致 κ = {kk:+.3f}（一致率 {po:.3f}）  '
              f'二值自一致 κ = {kb:+.3f}')
            res[f'{nm}_self_k3'] = kk
            res[f'{nm}_self_k2'] = kb
        else:
            P(f'  {nm}: [跳过] 可配对重复项不足（{len(pairs)}）')

    # ---------- ③ 人–机 ----------
    P('\n' + '-' * 62)
    P('③ 人–机一致性（对照操作化伪标签）')
    P('-' * 62)
    for nm, ans in (('A', A), ('B', B)):
        sel = [i for i in uniq
               if ans.get(i, (None,))[0] in (1, 2, 3)
               and key[i]['pseudo_label'] >= 0]
        if len(sel) < 10:
            P(f'  {nm}: [跳过] 有效 {len(sel)} 条')
            continue
        hu = [CODE_TO_CLASS[ans[i][0]] for i in sel]
        ps = [key[i]['pseudo_label'] for i in sel]
        kk, po, _, nn = kappa(hu, ps, 3)
        kb, pob, _, _ = kappa([bin_of(c) for c in hu], [bin_of(c) for c in ps], 2)
        P(f'  {nm}: n={nn}  三类 κ = {kk:+.3f}（一致率 {po:.3f}）  '
          f'二值 κ = {kb:+.3f}（一致率 {pob:.3f}）')
        res[f'{nm}_machine_k3'] = kk
        res[f'{nm}_machine_k2'] = kb
    P('  对照：旧单人 210 帧实验 三类 κ = +0.079、 二值 κ = +0.211')

    # ---------- ④ 效标（BDI） ----------
    P('\n' + '-' * 62)
    P('④ 效标效度：连续轴 BDI 是否能复现人的二值判断')
    P('-' * 62)
    if not args.no_cache:
        wanted = {key[i]['rel_path'] for i in uniq}
        lut = bdi_lookup(args.cache, wanted)
        if lut:
            for nm, ans in (('A', A), ('B', B)):
                sel = [i for i in uniq
                       if ans.get(i, (None,))[0] in (1, 2, 3)
                       and key[i]['rel_path'] in lut]
                if len(sel) < 20:
                    P(f'  {nm}: [跳过] 可匹配 {len(sel)} 条')
                    continue
                score = [lut[key[i]['rel_path']] for i in sel]
                lab = [bin_of(CODE_TO_CLASS[ans[i][0]]) for i in sel]
                auc, n1, n0 = roc_auc(score, lab)
                bc = best_cut(score, lab)
                P(f'  {nm}: n={len(sel)}（弯 {n1} / 直 {n0}）  AUC = {auc:.3f}')
                P(f'      最佳切点 BDI = {bc["thr"]:+.4f}  '
                  f'准确率 {bc["acc"]:.3f}  κ = {bc["kappa"]:+.3f}  '
                  f'敏感度 {bc["sens"]:.3f}  特异度 {bc["spec"]:.3f}')
                res[f'{nm}_auc'] = auc
                res[f'{nm}_cut_bdi'] = bc['thr']
                res[f'{nm}_cut_kappa'] = bc['kappa']
                res[f'{nm}_cut_acc'] = bc['acc']
        else:
            P('  [跳过] 未能构建 BDI（缺缓存或无 sklearn）')

    # ---------- ⑤ 用时与信心 ----------
    P('\n' + '-' * 62)
    P('⑤ 作答行为（识别"秒点"与低信心）')
    P('-' * 62)
    for nm, ans in (('A', A), ('B', B)):
        sec = np.array([v[2] for v in ans.values() if np.isfinite(v[2])])
        conf = np.array([v[1] for v in ans.values() if np.isfinite(v[1])])
        if len(sec):
            fast = float((sec < 1.0).mean())
            P(f'  {nm}: 用时中位数 {np.median(sec):.2f}s，'
              f'<1s 占 {fast * 100:.1f}%（共 {len(sec)} 条）')
            res[f'{nm}_median_sec'] = float(np.median(sec))
            res[f'{nm}_fast_frac'] = fast
        if len(conf):
            P(f'      信心均值 {conf.mean():.2f}（1–5）；'
              f'低信心(≤2)占 {(conf <= 2).mean() * 100:.1f}%')
            res[f'{nm}_mean_conf'] = float(conf.mean())

    # ---------- ⑥ 0/9 作答比例 ----------
    P('\n' + '-' * 62)
    P('⑥ 不可用(0) 与 无法归类(9) 的比例')
    P('-' * 62)
    for nm, ans in (('A', A), ('B', B)):
        codes = [v[0] for v in ans.values() if v[0] is not None]
        if not codes:
            continue
        n = len(codes)
        c0 = sum(1 for c in codes if c == 0)
        c9 = sum(1 for c in codes if c == 9)
        P(f'  {nm}: 共 {n} 条  不可用(0) {c0} ({c0 / n * 100:.1f}%)  '
          f'无法归类(9) {c9} ({c9 / n * 100:.1f}%)')
        res[f'{nm}_bad_frac'] = c0 / n
        res[f'{nm}_amb_frac'] = c9 / n
    _amb = [res[k] for k in ('A_amb_frac', 'B_amb_frac') if k in res]
    if _amb:
        m = sum(_amb) / len(_amb)
        P(f'  平均无法归类比例：{m * 100:.1f}%')
        if m > 0.20:
            P('  → 超过 20%：大量帧卡在类别之间。这本身就是"类界模糊"的直接证据，')
            P('    应在论文中作为正面结果报告，而非当作缺陷。')

    # ---------- ⑦ 预注册判读 ----------
    P('\n' + '=' * 62)
    P('⑦ 预注册判读')
    P('=' * 62)
    k2 = res.get('k2', float('nan'))
    km2A = res.get('A_machine_k2', float('nan'))
    km2B = res.get('B_machine_k2', float('nan'))
    km2 = np.nanmean([km2A, km2B]) if (km2A == km2A or km2B == km2B) else float('nan')
    selfk = np.nanmean([res.get('A_self_k2', np.nan), res.get('B_self_k2', np.nan)])
    P(f'  人–人二值 κ = {k2:+.3f}   人–机二值 κ = {km2:+.3f}   '
      f'人内重复二值 κ = {selfk:+.3f}')
    verdict = None
    if not (k2 == k2):
        verdict = ('R?', '人–人 κ 无法计算（有效共同作答不足或缺少第二名标注者）')
    elif not (selfk == selfk) or selfk >= 0.5:
        if k2 >= 0.6 and km2 == km2 and km2 <= 0.30:
            verdict = ('R1', '轴真实存在、操作化规则未捕捉；4.3 结论维持并可加强')
        elif k2 >= 0.6 and km2 == km2 and km2 >= 0.50:
            verdict = ('R2', '仅三类层级过度切分，二值可用；4.3 限定为三类过度切分')
        elif k2 < 0.40:
            verdict = ('R3', '标注不可靠；4.3 整节结论撤回，主结论由 4.4.1 承担')
        else:
            verdict = ('R4', '证据不足；4.3 降级为"未能验证"，不得推断类别不存在')
    else:
        verdict = ('R4', f'人内重复二值 κ={selfk:.3f} < 0.5，标注噪声过大；'
                         '4.3 降级为"未能验证"，建议加训或换标注者')
    P(f'  判定：{verdict[0]} — {verdict[1]}')
    lines.append('')
    lines.append(f'VERDICT={verdict[0]}')
    lines.append(f'DETAIL={verdict[1]}')

    # ---------- 落盘 ----------
    with open(pref + '.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with open(pref + '.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        for k, v in res.items():
            w.writerow([k, f'{v:.6f}' if isinstance(v, float) else v])
        w.writerow(['verdict', verdict[0]])
    print(f'\n已写出：{pref}.txt / {pref}.csv')


if __name__ == '__main__':
    main()
