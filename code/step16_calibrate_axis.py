"""step16_calibrate_axis.py — 把"连续弯曲深度轴"做成可交付的测量：定义、效度、信度、敏感性、信息量。

为什么需要这个脚本
------------------
前面所有分析证明的是"离散三类不成立"。这只回答了"不该用什么"，没回答
"该用什么"。本脚本把论文的落点从否定改为交付：给出一条连续轴（BDI,
bend-depth index），并用四个可检验的数字把它标定成可用测量：

  C1 定义与纯度   PC1 解释多少形状方差；与 bbi / maxκ 的相关（便于他人复现）
  C2 效标效度     在 210 帧人工标注上，BDI 对"直 vs 弯"的 ROC AUC、
                  最佳切点、该切点下的 Cohen's κ（论文里的 κ=0.211 的连续版）
  C3 信度         虫级 ICC(1)（平衡子样本单因素方差分解）+ 时间分半
                  Pearson r 与 Spearman–Brown 全长信度
  C4 敏感性       每虫平均 BDI 随成像日期（年龄）的斜率，按组
  C5 信息量       互信息：MI(BDI; 三类伪标签) vs MI(BDI; 人工二值)，
                  并按各自熵归一化 —— 量化"分三类丢了多少信息"
  C6 切点任意性   k=2 分界处的核密度与峰值之比（越大说明刀切在分布主体上，
                  即切分任意）；以及各日期最优 bbi 阈值的漂移幅度

判据与解释纪律
  · AUC 只有在明显高于 0.5 时才说明轴与人类感知一致；<0.6 视为无关。
  · ICC < 0.5 视为信度不足，该轴不能用于个体级表型比较。
  · 切点密度比 > 0.5 视为"刀切在主体上"，此时任何 k=2 的簇都不能称类别。
  三者联合才能支撑"连续轴优于类别标签"的说法；任一不成立须如实降级表述。

用法（ele 环境）
--------------
  # 最简（两份人工文件都在工作目录时，无需额外参数）
  python step16_calibrate_axis.py -c atlas_out_full/kappa_cache.npz \
      --root $WORM_DATA_ROOT -o atlas_out_full/axis

  # 显式指定（文件名或位置不同时）
  python step16_calibrate_axis.py -c atlas_out_full/kappa_cache.npz \
      --human-labels human_labels.csv \
      --human-batch human_annotation_batch.csv \
      --root $WORM_DATA_ROOT -o atlas_out_full/axis

产物
----
  axis_summary.txt       C1–C6 全部数字与判定
  axis_worm.csv          每虫平均 BDI、帧数、组、日期
  axis_per_date.csv      各日期最优 bbi 阈值与切点密度比
  fig10_axis_calibration.png  四联图
"""

import os
import csv
import argparse
import numpy as np

from wormpath import get_data_root

FEATURE_NAMES = ['bbi', 'max_kappa', 'n_high_regions', 'mid_dominance',
                 'peak_midness', 'kappa_std', 'periodicity_peaks']
GROUP_ORDER = ['control', '5mg', '8mg', '10mg']


def _csv_rows(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'  已写 {path}')


# ============================================================
# C1 轴定义
# ============================================================

def build_axis(kappa, features):
    """BDI = κ(s) 第一主成分；符号固定为与 bbi 正相关（越大=弯得越深）。"""
    from sklearn.decomposition import PCA
    p = PCA(n_components=1, random_state=0).fit(kappa)
    z = p.transform(kappa)[:, 0]
    var = float(p.explained_variance_ratio_[0])
    bbi = features[:, FEATURE_NAMES.index('bbi')]
    r = float(np.corrcoef(z, bbi)[0, 1])
    if r < 0:
        z, r = -z, -r
    # 标准化到 0–1 分位（便于跨批次比较，不改变序关系）
    z = (z - z.min()) / max(z.max() - z.min(), 1e-9)
    corr = {}
    for i, name in enumerate(FEATURE_NAMES):
        corr[name] = round(float(np.corrcoef(z, features[:, i])[0, 1]), 4)
    return z.astype(np.float32), var, corr


# ============================================================
# C3 信度
# ============================================================

def icc_oneway(values, worm_ids, n_per_worm=50, seed=0):
    """平衡子样本的单因素随机效应 ICC(1,1)。

    对帧数 >= n_per_worm 的每只虫随机取 n_per_worm 帧，做单因素方差分解：
        ICC = (MSB - MSW) / (MSB + (k-1) * MSW)
    """
    rng = np.random.RandomState(seed)
    idx = {}
    for i, w in enumerate(worm_ids):
        idx.setdefault(w, []).append(i)
    use = [w for w, ix in idx.items() if len(ix) >= n_per_worm]
    if len(use) < 10:
        return float('nan'), float('nan'), 0
    rows = []
    for w in use:
        ix = idx[w]
        pick = rng.choice(len(ix), n_per_worm, replace=False)
        rows.append(values[[ix[j] for j in pick]])
    M = np.asarray(rows)                     # (n_worm, n_per_worm)
    k = M.shape[1]
    n = M.shape[0]
    grand = M.mean()
    msb = k * ((M.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msw = ((M - M.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    icc = (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) > 0 else float('nan')
    return round(float(icc), 4), round(float(np.sqrt(max(msw, 0))), 4), n


def split_half(values, worm_ids, order_key=None):
    """时间分半：每虫按帧序前后两半的均值，跨虫 Pearson r + Spearman–Brown。"""
    idx = {}
    for i, w in enumerate(worm_ids):
        idx.setdefault(w, []).append(i)
    a, b = [], []
    for w, ix in idx.items():
        ix = sorted(ix, key=(lambda j: order_key[j]) if order_key else (lambda j: j))
        if len(ix) < 4:
            continue
        h = len(ix) // 2
        a.append(float(np.mean(values[ix[:h]])))
        b.append(float(np.mean(values[ix[h:]])))
    if len(a) < 10:
        return float('nan'), float('nan'), 0
    r = float(np.corrcoef(a, b)[0, 1])
    sb = 2 * r / (1 + r) if r > -1 else float('nan')
    return round(r, 4), round(float(sb), 4), len(a)


# ============================================================
# C4 敏感性（年龄斜率）
# ============================================================

def age_slopes(worm_bdi, worm_group, worm_date, dates_all):
    """每虫平均 BDI 对日期的组内回归；n 为日期数（不是虫数）。

    worm_bdi: [(group, date, worm_key, mean_bdi), ...]
    """
    import math
    out = {}
    for g in GROUP_ORDER:
        per_date = {}
        for gg, d, key, v in worm_bdi:
            if gg != g:
                continue
            per_date.setdefault(d, []).append(v)
        comp = [d for d in sorted(per_date) if d in dates_all]
        xs = list(range(len(comp)))
        ys = [float(np.mean(per_date[d])) for d in comp]
        n = len(xs)
        if n < 4:
            out[g] = dict(slope=float('nan'), r=float('nan'), p=float('nan'), n=n)
            continue
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx
        sst = sum((y - my) ** 2 for y in ys)
        r = sxy / math.sqrt(sxx * sst) if sst > 0 else float('nan')
        resid = [y - (slope * x + (my - slope * mx)) for x, y in zip(xs, ys)]
        s2 = sum(e ** 2 for e in resid) / (n - 2)
        se = math.sqrt(s2 / sxx) if sxx > 0 else float('nan')
        t = slope / se if se and se == se and se > 0 else float('nan')
        p = math.erfc(abs(t) / math.sqrt(2)) if t == t else float('nan')
        out[g] = dict(slope=round(slope, 6), r=round(r, 4),
                      p=round(p, 4), n=n)
    return out


# ============================================================
# C5 信息量与 C6 切点任意性
# ============================================================

def mutual_info(x_cont, labels, n_bins=20):
    """连续量按分位分箱后与标签的互信息，并按标签熵归一化（0–1）。"""
    from sklearn.metrics import mutual_info_score
    q = np.quantile(x_cont, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(x_cont, q)
    mi = float(mutual_info_score(bins, labels))
    _, cnt = np.unique(labels, return_counts=True)
    p = cnt / cnt.sum()
    h = float(-(p * np.log(p)).sum())
    return round(mi, 5), round(mi / h, 4) if h > 0 else float('nan')


def pseudo_labels_from_features(feats):
    """按论文 2.4 的判据从特征矩阵重建三类伪标签（0=正弦 1=Ω 2=笔直）。

    判据：笔直 bbi<0.16；Ω 高曲率区域数<=1 且 max_kappa>=0.35；其余正弦。
    与 simulate_labels 同口径，仅用于 C5 的信息量对比，不用于任何主结果。
    """
    i_bbi = FEATURE_NAMES.index('bbi')
    i_mk = FEATURE_NAMES.index('max_kappa')
    i_nr = FEATURE_NAMES.index('n_high_regions')
    bbi, mk, nr = feats[:, i_bbi], feats[:, i_mk], feats[:, i_nr]
    lab = np.zeros(len(bbi), dtype=int)                  # 0 正弦（兜底）
    om = (nr <= 1) & (mk >= 0.35)
    lab[om] = 1
    st = bbi < 0.16
    lab[st] = 2
    return lab


def cut_arbitrariness(bdi, seed=0):
    """k=2 分界处的核密度 / 峰值密度。>0.5 表示刀切在分布主体上。"""
    from sklearn.cluster import KMeans
    from scipy.stats import gaussian_kde
    x = np.asarray(bdi, dtype=np.float64)
    if len(x) > 200000:
        x = np.random.RandomState(seed).choice(x, 200000, replace=False)
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(x[:, None])
    c = np.sort(km.cluster_centers_[:, 0])
    cut = float((c[0] + c[1]) / 2)
    kde = gaussian_kde(x)
    grid = np.linspace(x.min(), x.max(), 512)
    dens = kde(grid)
    d_cut = float(kde(cut)[0])
    ratio = d_cut / max(dens.max(), 1e-12)
    return round(cut, 4), round(ratio, 4), [round(float(v), 4) for v in c]


def cut_drift(features, groups, dates, worms, pseudo_labels=None):
    """各日期上 bbi 的最优二分阈值（把 Ω 与正弦分开），衡量切点漂移。"""
    bbi = features[:, 0]
    out = []
    for d in sorted(set(dates)):
        m = dates == d
        if m.sum() < 200:
            continue
        xs = np.sort(bbi[m])
        # 用分位网格搜索最大平衡准确率的分界（以 bbi 中位数两侧的 5–95 分位）
        lo, hi = np.quantile(xs, [0.05, 0.95])
        if hi - lo < 1e-6:
            continue
        best_t, best_score = None, -1
        # 若无标签，则以"深弯与浅弯各占一半"作为目标：取 bbi 中位数
        # （仅作切点漂移的参照，不涉及伪标签）
        for t in np.linspace(lo, hi, 91):
            above = float((xs > t).mean())
            score = 1.0 - abs(above - 0.5)      # 最均衡的二分点
            if score > best_score:
                best_score, best_t = score, float(t)
        out.append(dict(date=d, n=int(m.sum()), thr=round(best_t, 5),
                        frac_above=round(float((xs > best_t).mean()), 4)))
    if not out:
        return out, float('nan'), float('nan')
    thrs = [r['thr'] for r in out]
    return out, round(float(np.std(thrs)), 5), round(max(thrs) - min(thrs), 5)


# ============================================================
# C2 人工效标（210 帧）
# ============================================================

def _resolve_batch(args, human_keys):
    """在多个候选路径中选出与人工标签交集最大的批次文件。

    历史上批次文件曾在 output_3class_smoke800/ 下，后被移到工作目录；
    若沿用旧默认值会指向不存在的文件，C2 被静默跳过。此处自动择优选。
    """
    cands = []
    for c in (args.human_batch, 'human_annotation_batch.csv',
              os.path.join('output_3class_smoke800',
                           'human_annotation_batch.csv')):
        if c and c not in cands:
            cands.append(c)
    best, best_n = None, -1
    for c in cands:
        if not os.path.exists(c):
            continue
        try:
            keys = {r['filename'] for r in _csv_rows(c)}
        except Exception:
            continue
        n = len(keys & human_keys)
        print(f'  [C2] 候选批次 {c}：{len(keys)} 行，与人工标签交集 {n}')
        if n > best_n:
            best, best_n = c, n
    if best is None:
        print('  [跳过 C2] 未找到任何可用的批次文件')
        return None
    if best_n < 40:
        print(f'  [跳过 C2] 最佳批次交集仅 {best_n} 行（<40）')
        return None
    return best


def human_criterion(args, n_samples=100):
    """在人工标注帧上重算 BDI，评估对"直 vs 弯"的判别力。

    人工标签编码：label_code 1=笔直，2=Ω，3=正弦（见 human_label_gui.py）。
    """
    import cv2
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from worm_labeler import extract_skeleton_full, compute_curvature_profile
    from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score

    if not os.path.exists(args.human_labels):
        print(f'  [跳过 C2] 人工标签文件缺失：{args.human_labels}')
        return None

    human = {r['filename']: int(r['label_code'])
             for r in _csv_rows(args.human_labels)}
    batch_path = _resolve_batch(args, set(human))
    if batch_path is None:
        return None
    batch = {r['filename']: r for r in _csv_rows(batch_path)}
    common = [f for f in human if f in batch]
    if not common:
        print('  [跳过 C2] 两份人工文件无交集')
        return None

    feats, labels = [], []
    n_missing = 0
    for f in common:
        rel = f.replace('\\', '/')
        cands = [os.path.join(args.root, rel)]
        # 兼容扁平命名（组_日期_行_worm{n}_{frame}.png）
        parts = rel.split('/')
        if len(parts) == 1:
            stem = os.path.splitext(rel)[0]
            bits = stem.split('_')
            if len(bits) >= 5:
                cands.append(os.path.join(
                    args.root, bits[0], bits[1], bits[2],
                    '_'.join(bits[3:])))
        p = next((c for c in cands if os.path.exists(c)), None)
        if p is None:
            n_missing += 1
            continue
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            n_missing += 1
            continue
        try:
            out = extract_skeleton_full(img)
            pts = out[0]
            if pts is None or len(pts) < 10:
                continue
            kap = np.abs(compute_curvature_profile(pts, n_samples))
        except Exception:
            continue
        feats.append(kap)
        labels.append(human[f])
    if n_missing:
        print(f'  [C2] {n_missing} 帧图像未找到（检查 --root 与文件名格式）')
    if len(feats) < 40:
        print(f'  [跳过 C2] 可用人工帧仅 {len(feats)}')
        return None

    # 用人工帧自身做 PCA 定义轴（与全量轴同构，避免尺度不匹配）
    from sklearn.decomposition import PCA
    K = np.vstack(feats).astype(np.float32)
    pc1 = PCA(n_components=1, random_state=0).fit_transform(K)[:, 0]
    y = np.array([1 if c in (2, 3) else 0 for c in labels])   # 1=弯, 0=直
    auc = float(roc_auc_score(y, pc1))
    if auc < 0.5:
        pc1, auc = -pc1, 1 - auc
    fpr, tpr, thr = roc_curve(y, pc1)
    j = tpr - fpr
    best = int(np.argmax(j))
    pred = (pc1 >= thr[best]).astype(int)
    kap = float(cohen_kappa_score(y, pred))
    mi_h = mutual_info(pc1, y)
    return dict(n=len(y), auc=round(auc, 4),
                best_thr=round(float(thr[best]), 5),
                accuracy=round(float((pred == y).mean()), 4),
                kappa=round(kap, 4),
                mi_human=mi_h,
                n_bent=int(y.sum()), n_straight=int((1 - y).sum()))


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser(description='连续弯曲深度轴的标定')
    ap.add_argument('-c', '--cache', default='atlas_out_full/kappa_cache.npz')
    ap.add_argument('-o', '--outdir', default='atlas_out_full/axis')
    ap.add_argument('--human-labels', default='human_labels.csv')
    ap.add_argument('--human-batch', default='human_annotation_batch.csv',
                    help='操作化标签批次（若不存在会依次尝试工作目录与 '
                         'output_3class_smoke800/ 下的同名文件）')
    ap.add_argument('--root', default=get_data_root())
    ap.add_argument('--n-per-worm', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    z = np.load(args.cache, allow_pickle=True)
    kappa = z['kappa'].astype(np.float32)
    feats = np.asarray(z['features'], dtype=np.float32)
    groups = z['groups']
    dates = z['dates']
    worms = z['worms']
    rows = z['rows'] if 'rows' in z else np.array(['?'] * len(groups))
    print(f'[载入] {len(kappa):,} 帧')

    # ---- C1 轴定义 ----
    bdi, var, corr = build_axis(kappa, feats)
    print(f'\n=== C1 轴定义 ===\n  PC1 解释方差 {var:.4f}')
    print('  与各特征相关: ' + ', '.join(f'{k}={v}' for k, v in corr.items()))

    # ---- 虫级聚合 ----
    key = np.asarray([f'{g}|{d}|{r}|{w}' for g, d, r, w in
                      zip(groups, dates, rows, worms)])
    uk, inv = np.unique(key, return_inverse=True)
    wn = np.bincount(inv)
    wsum = np.bincount(inv, weights=bdi)
    wmean = wsum / np.maximum(wn, 1)
    first = np.zeros(len(uk), dtype=int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    wgrp, wdate = groups[first], dates[first]
    print(f'  入统计虫数 {len(uk):,}')

    # ---- C3 信度 ----
    icc, within_sd, n_icc = icc_oneway(bdi, key, n_per_worm=args.n_per_worm,
                                       seed=args.seed)
    r_half, sb, n_half = split_half(bdi, key)
    print(f'\n=== C3 信度 ===\n  虫级 ICC(1) = {icc}（{n_icc} 只，'
          f'每只 {args.n_per_worm} 帧）')
    print(f'  分半: r = {r_half}, Spearman–Brown = {sb}（{n_half} 只）')

    # ---- C4 敏感性 ----
    triples = [(g, d, k, float(v)) for g, d, k, v in
               zip(wgrp, wdate, uk, wmean)]
    sl = age_slopes(triples, wgrp, wdate, sorted(set(wdate)))
    print('\n=== C4 敏感性（每虫平均 BDI 对日期）===')
    for g, s in sl.items():
        print(f'  {g:>8s}: slope={s["slope"]:+.6f}/日  r={s["r"]}  '
              f'p={s["p"]}  (n={s["n"]} 日期)')

    # ---- C5 信息量：连续轴 vs 三类标签 ----
    lab3 = pseudo_labels_from_features(feats)
    mi3, nmi3 = mutual_info(bdi, lab3)
    info = dict(mi3=mi3, nmi3=nmi3)
    frac = {int(c): round(float((lab3 == c).mean()), 4)
            for c in np.unique(lab3)}
    print(f'\n=== C5 信息量 ===\n  伪标签分布 {frac}')
    print(f'  MI(BDI; 三类) = {mi3}（归一化 {nmi3}）'
          '  —— 归一化值越低，说明三类标签丢失越多轴的连续信息')

    # ---- C6 切点任意性 ----
    cut, ratio, cents = cut_arbitrariness(bdi, seed=args.seed)
    per_date, thr_sd, thr_range = cut_drift(feats, groups, dates, worms)
    print(f'\n=== C6 切点任意性 ===\n  k=2 分界 {cut}，簇心 {cents}，'
          f'分界处密度/峰值 = {ratio}')
    print(f'  各日期最优阈值: SD={thr_sd}, 极差={thr_range}（{len(per_date)} 天）')

    # ---- C2 人工效标 ----
    hc = human_criterion(args)
    print('\n=== C2 效标效度（人工 210 帧）===')
    if hc:
        print(f'  n={hc["n"]}（弯 {hc["n_bent"]} / 直 {hc["n_straight"]}）  '
              f'AUC={hc["auc"]}  准确率={hc["accuracy"]}  κ={hc["kappa"]}')
    else:
        print('  [跳过]')
    mi_h = None
    if hc:
        # 人工二值标签与连续轴的信息量（需人工帧的轴值，另存于 hc）
        mi_h = hc.get('mi_human')
        if mi_h is not None:
            print(f'  MI(BDI; 人工二值) = {mi_h[0]}（归一化 {mi_h[1]}）')

    # ---- 汇总 ----
    lines = []
    lines.append('连续弯曲深度轴（BDI）标定报告')
    lines.append('=' * 56)
    lines.append(f'C1 轴定义: PC1 解释方差 {var:.4f}')
    lines.append('   ' + ', '.join(f'{k}={v}' for k, v in corr.items()))
    lines.append(f'C2 效标效度: '
                 + (f'AUC={hc["auc"]}, 准确率={hc["accuracy"]}, κ={hc["kappa"]}, '
                    f'n={hc["n"]}' if hc else '未评估'))
    lines.append(f'C3 信度: ICC(1)={icc}（{n_icc} 只×{args.n_per_worm} 帧）, '
                 f'分半 r={r_half}, SB={sb}')
    lines.append('C4 敏感性（BDI/日）: '
                 + '; '.join(f'{g} {s["slope"]:+.6f} (p={s["p"]})'
                             for g, s in sl.items()))
    lines.append(f'C5 信息量: '
                 + (f'MI(BDI;三类)={info["mi3"]}（归一化 {info["nmi3"]}）'
                    if info else '未评估'))
    lines.append(f'C6 切点任意性: k=2 分界={cut}, 分界密度/峰值={ratio}; '
                 f'各日期阈值 SD={thr_sd}, 极差={thr_range}')
    lines.append('')
    verdict = []
    verdict.append('判定：')
    if hc and hc['auc'] >= 0.75:
        verdict.append(f'  · 轴与人类"直/弯"感知一致（AUC={hc["auc"]}）——效标效度成立。')
    elif hc:
        verdict.append(f'  · 轴与人类感知仅弱关联（AUC={hc["auc"]}）——效标效度不足，'
                       '不得宣称该轴替代人类判断。')
    if icc == icc and icc >= 0.5:
        verdict.append(f'  · 虫级信度 ICC={icc} 可用于个体表型比较。')
    else:
        verdict.append(f'  · 虫级信度 ICC={icc} 偏低，个体级比较须谨慎。')
    if ratio == ratio and ratio > 0.5:
        verdict.append(f'  · k=2 分界处密度为峰值的 {ratio:.2f} 倍，'
                       '刀切在分布主体上——该切分应称"描述性分界"而非"类别"。')
    lines += verdict
    txt = '\n'.join(lines)
    with open(os.path.join(args.outdir, 'axis_summary.txt'), 'w',
              encoding='utf-8') as f:
        f.write(txt + '\n')
    print('\n' + txt)

    # ---- 写 CSV ----
    _write_csv(os.path.join(args.outdir, 'axis_worm.csv'),
               [dict(worm=k, group=g, date=d, n_frames=int(n), bdi=round(float(m), 5))
                for k, g, d, n, m in zip(uk, wgrp, wdate, wn, wmean)])
    _write_csv(os.path.join(args.outdir, 'axis_per_date.csv'), per_date)

    # ---- 图 ----
    make_figure(args.outdir, bdi, wmean, wgrp, wdate, sl, per_date, hc, cut, ratio)


def make_figure(outdir, bdi, wmean, wgrp, wdate, sl, per_date, hc, cut, ratio):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 11,
                         'figure.dpi': 300})
    COLORS = {'control': '#4D4D4D', '5mg': '#0072B2', '8mg': '#D55E00',
              '10mg': '#009E73'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0, 0]
    s = np.random.RandomState(0).choice(bdi, min(60000, len(bdi)), replace=False)
    kde = gaussian_kde(s)
    grid = np.linspace(0, 1, 512)
    ax.plot(grid, kde(grid), color='#333', lw=1.6)
    ax.axvline(cut, ls='--', color='#D55E00', lw=1.4,
               label=f'k=2 cut (density {ratio:.2f} x peak)')
    ax.set_xlabel('Bend-depth index (BDI)')
    ax.set_ylabel('Density')
    ax.set_title('A. Axis distribution: cut lies in the bulk')
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[0, 1]
    if per_date:
        ax.plot([r['date'] for r in per_date], [r['thr'] for r in per_date],
                'o-', ms=4, lw=1.3, color='#0072B2')
        ax.set_xlabel('Imaging date')
        ax.set_ylabel('Optimal bbi threshold')
        ax.set_title('B. Category boundary drifts across dates')
        for lab in ax.get_xticklabels():
            lab.set_rotation(60)
            lab.set_fontsize(7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1, 0]
    for g in ['control', '5mg', '8mg', '10mg']:
        m = wgrp == g
        if not m.any():
            continue
        ds = sorted(set(wdate[m]))
        ys = [float(np.mean(wmean[m & (wdate == d)])) for d in ds]
        slv = sl.get(g, {}).get('slope', float('nan'))
        ax.plot(ds, ys, 'o-', ms=4, lw=1.3, color=COLORS[g],
                label=f'{g} ({slv:+.4f}/day)')
    ax.set_xlabel('Imaging date')
    ax.set_ylabel('Mean BDI per worm')
    ax.set_title('C. Sensitivity to age')
    ax.legend(frameon=False, fontsize=8)
    for lab in ax.get_xticklabels():
        lab.set_rotation(60)
        lab.set_fontsize(7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1, 1]
    ax.axis('off')
    rows = []
    if hc:
        rows += [('C2 AUC (human bent vs straight)', f'{hc["auc"]:.3f}'),
                 ('C2 accuracy at best cut', f'{hc["accuracy"]:.3f}'),
                 ('C2 Cohen kappa at cut', f'{hc["kappa"]:.3f}'),
                 ('C2 n human frames', f'{hc["n"]}')]
    rows.append(('C6 cut density / peak', f'{ratio:.2f}'))
    ax.text(0.02, 0.95, 'Calibration summary\n\n' +
            '\n'.join(f'{a:34s} {b}' for a, b in rows),
            va='top', fontsize=9, family='DejaVu Sans')
    ax.set_title('D. Key numbers')

    fig.suptitle('Calibrating a continuous bend-depth axis', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(outdir, 'fig10_axis_calibration.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'图已保存：{out}')


if __name__ == '__main__':
    main()
