"""无监督姿态形状图谱聚类（cluster_atlas.py）。

动机：人工三类（正弦/Ω/笔直）经人工标注验证 κ=0.079 ≈ 随机，Ω↔正弦 人眼不可分。
故放弃先验分类，改为数据驱动的"形状图谱"：把所有帧的姿态形状（κ(s) 曲线）无监督
聚成若干簇（簇数由多准则决定），再统计每虫的姿态直方图，最后做剂量位移检验。

不变性说明（2026-10-24 修正，经 worm_labeler.compute_curvature_profile 源码核实）：
  - 平移/旋转：κ(s) 是内蕴量，天然不变。
  - 大小/尺度：**已天然不变，无需额外归一化**。compute_curvature_profile 的
    κ(s) = |sin(Δθ)|（相邻段的夹角正弦）关于归一化弧长 s∈[0,1] 的分布，
    而相似形缩放不改变夹角 Δθ，故 κ(s) 曲线与体长 L 无关。
    【重要】早期版本曾提供 --scale-normalize（κ × L），该操作反而**引入**
    体长依赖（κ×L²），已于 2026-10-24 移除。该参数现为兼容 no-op。
    如需与旧行为分类管线的物理阈值（BBI 0.16 / max_kappa 0.35）对照，
    可加 --physical-curvature 把 κ(s) 除以 L 转成物理曲率（1/px）。
  - 镜像翻转：默认取 |κ(s)|，左右弯（手性）视为同一姿态；
    同时用曲率质心 s_cm<0.5 统一朝向，消除头尾判错。
    （--no-mirror-invariant 可关闭，但需 compute_curvature_profile 返回有符号 κ。）

流程：
  Stage 1  全帧 κ(s) + 骨架长度 + 弯曲特征矩阵 提取（多进程），
           镜像/头尾规范化（曲率质心 s_cm 统一 < 0.5），缓存到 .npz
  Stage 2  （默认）PCA 降维（SBPD 风格，95% 方差自动定维；子采样拟合+全量 transform）
  Stage 3  多准则定 k：肘部(SSE)、silhouette、Calinski-Harabasz、
           BIC 肘点(GMM)、gap statistic；
           **推荐规则 = silhouette 与 BIC 肘点**，一致则采纳，
           分歧则取较小 k 并报告（分歧本身即"连续谱"证据）
  Stage 4  k-means 聚类（子采样拟合+全量 predict）→ 每虫姿态直方图 →
           每簇弯曲特征统计表（mean/std/median/IQR/p5/p95）→
           每簇代表 κ(s) 曲线 → 剂量位移检验

剂量位移检验（全部按虫聚合，避免帧自相关）：
  · 主姿态 group × cluster 列联表卡方 + Cramér's V（稀疏表自动跳过）
  · 每簇比例 Kruskal-Wallis + ε² 效应量 + BH 校正
  · 事后 Mann-Whitney U（control vs 各给药组）+ BH + Cohen's d + 95%CI
  · Cochran-Armitage 剂量趋势检验（检验剂量-反应单调性）
  · 汇总写入 atlas_dose_tests.csv

全量数据支持（约 160 万帧 / 8000 虫）：
  · 特征存为 float32 矩阵（约 45 MB），不用 object dict 数组
  · --pca-sample / --cluster-sample 控制拟合子采样（全量帧仍全部参与
    transform / predict / 统计），内存与耗时可控
  · 建议：--workers 20（与 CPU 线程数匹配），提取阶段约 4-6 小时

用法（ele 环境）：
  # 冒烟（每组 200 虫，约 800 虫）
  python cluster_atlas.py -i $WORM_DATA_ROOT --max-worms 200 -o atlas_out

  # 全量（推荐，论文最终版）
  python cluster_atlas.py -i $WORM_DATA_ROOT -o atlas_out_full --workers 20

  # 用已有缓存跳过提取，只重跑定 k / 聚类 / 检验
  python cluster_atlas.py -i $WORM_DATA_ROOT -o atlas_out --skip-extract

  # 与旧行为分类管线阈值（BBI 0.16 / max_kappa 0.35）对照时才用：
  python cluster_atlas.py -i $WORM_DATA_ROOT -o atlas_out --physical-curvature

图样式（2026-10-24 起，论文规范）：全英文、DejaVu Sans、300 dpi、
Okabe-Ito 色盲友好配色、统一字号（标题 12 / 轴标签 11 / 刻度 10 / 图例 10）。
"""
import os
import sys
import csv
import argparse
import hashlib
import time
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wormpath import get_data_root

# ---- 论文图样式规范（2026-10-24）：全英文、DejaVu Sans、300 dpi ----
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.6,
    'axes.axisbelow': True,
    'mathtext.default': 'regular',
})
# 色盲友好配色（Okabe-Ito）
PALETTE = {'control': '#4D4D4D', '5mg': '#0072B2', '8mg': '#D55E00',
           '10mg': '#009E73'}
CLUSTER_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#7570B3',
                  '#E69F00', '#56B4E9', '#999999']

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from worm_labeler import parse_five_dims, extract_skeleton_full, \
    compute_curvature_profile, extract_features

N_SAMPLES = 100          # κ(s) 采样点数（与管线一致）
GROUP_ORDER = ['control', '5mg', '8mg', '10mg']


def resolve_group_order(found):
    """按优先序排列实际出现的组；兼容主数据（control/5mg/8mg/10mg）
    与公开数据集 worm_crops（train/test，2026-11-16）。未知组名按字母序排在后面。"""
    found = set(map(str, found))
    order = [g for g in GROUP_ORDER if g in found]
    order += sorted(found - set(order))
    return order

# 弯曲特征的固定顺序（与 _extract_worker / extract_all 的列序一致）。
# 存成 float32 矩阵而非 dict 列表：160 万帧下 object 数组会占 >1 GB。
FEATURE_NAMES = ['bbi', 'max_kappa', 'n_high_regions', 'mid_dominance',
                 'peak_midness', 'kappa_std', 'periodicity_peaks']


# ============================================================
# 0. 镜像规范化
# ============================================================
def canonicalize(kappa, mirror_invariant=True):
    """镜像/头尾规范化：让曲率质心 s_cm 恒落在前半段（s_cm < 0.5）。

    κ(s) 本身已平移+旋转不变（内蕴量）；镜像（或头尾颠倒）等价于
    κ(s) -> κ(L-s)。用曲率加权质心判断朝向，若质心在后半段则反转，
    于是"弯在前半段"成为规范朝向，消除头尾判错 + 镜像歧义。

    尺度不变性（2026-10-24 经 worm_labeler.compute_curvature_profile
    源码核实）：κ(s) = |sin(Δθ)|（相邻段夹角正弦）关于**归一化弧长**
    s∈[0,1] 的分布。相似形缩放不改变夹角 Δθ，故 κ(s) 与体长 L 无关。
    因此本函数**不做**任何长度缩放（早期 --scale-normalize 的 κ×L
    反而引入 L² 依赖，已移除）。

    参数：
      mirror_invariant: True 时取 |κ|，保证左右弯（镜像）等价。
                        （compute_curvature_profile 已返回 |κ|，此为显式保险。）
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    n = len(kappa)
    if n == 0:
        return kappa
    if mirror_invariant:
        kappa = np.abs(kappa)
    w = np.abs(kappa)
    wsum = w.sum()
    if wsum < 1e-9:
        return kappa
    s = np.linspace(0.0, 1.0, n)
    s_cm = float((s * w).sum() / wsum)
    if s_cm > 0.5:
        kappa = kappa[::-1].copy()
    return kappa


# ============================================================
# 1. 全帧 κ(s) 提取（多进程）
# ============================================================
# 多进程子进程的镜像开关（Windows spawn 模式下用 Pool initializer 注入）
_MIRROR_INVARIANT = True


def _init_worker(mirror_invariant):
    global _MIRROR_INVARIANT
    _MIRROR_INVARIANT = bool(mirror_invariant)


def _extract_worker(task):
    """单帧提取：读图 -> 骨架 -> κ(s) + 骨架长度 + 弯曲特征 -> 镜像规范化。

    返回 (rel_path, kappa or None, skeleton_len, feature_vector or None,
          group, date, row, worm)，其中 feature_vector 为 FEATURE_NAMES
    顺序的 float32 向量（外加 skeleton_len 由调用方追加）。
    """
    rel_path, abs_path = task
    try:
        group, date, row, worm, frame = parse_five_dims(rel_path)
    except Exception:
        group, date, row, worm, frame = 'unknown', 'unknown', 'unknown', rel_path, 0
    img = cv2.imread(abs_path, cv2.IMREAD_COLOR)
    if img is None:
        return rel_path, None, None, None, group, date, row, worm
    try:
        skeleton = extract_skeleton_full(img)[0]
        if skeleton is None or len(skeleton) < 10:
            return rel_path, None, None, None, group, date, row, worm
        kappa = compute_curvature_profile(skeleton, N_SAMPLES)
        kappa = canonicalize(kappa, mirror_invariant=_MIRROR_INVARIANT)
        skel_len = float(np.linalg.norm(np.diff(skeleton, axis=0), axis=1).sum())
        fts = extract_features(kappa)
        feat_vec = np.asarray([float(fts.get(fn, 0.0)) for fn in FEATURE_NAMES],
                              dtype=np.float32)
        return (rel_path, kappa.astype(np.float32), skel_len,
                feat_vec, group, date, row, worm)
    except Exception:
        return rel_path, None, None, None, group, date, row, worm


def list_tasks(input_dir, max_worms, max_frames):
    """遍历图片，返回 (rel_path, abs_path) 任务列表，受 max_worms / max_frames 限制。

    max_worms：限制每组（group）唯一虫体数（key = date/row/worm），保证四组均衡；
              例如 --max-worms 200 表示每组最多 200 虫，共约 800 虫。
    max_frames：总帧数硬上限。
    """
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    tasks = []
    per_group = defaultdict(set)   # group -> set of (date, row, worm)
    total_frames = 0
    for root, dirs, files in os.walk(input_dir):
        # 2026-11-13：噪声隔离区在数据根内，os.walk 会把它连帧带组重新扫入，
        # 导致"删了等于没删"。所有下划线开头的目录一律剪枝。
        dirs[:] = [d for d in dirs if not d.startswith('_')]
        dirs.sort()
        files.sort()
        for f in files:
            if not f.lower().endswith(exts):
                continue
            abs_path = os.path.join(root, f)
            rel_path = os.path.relpath(abs_path, input_dir).replace('\\', '/')
            if max_worms:
                try:
                    g, d, rw, wm, _ = parse_five_dims(rel_path)
                    wkey = (d, rw, wm)
                    if len(per_group[g]) >= max_worms:
                        continue
                    per_group[g].add(wkey)
                except Exception:
                    pass
            tasks.append((rel_path, abs_path))
            total_frames += 1
            if max_frames and total_frames >= max_frames:
                return tasks
    return tasks


def _filesig(filenames):
    return hashlib.md5('\n'.join(sorted(filenames)).encode('utf-8')).hexdigest()


def extract_all(tasks, workers, chunksize, cache_path, outdir,
                mirror_invariant=True):
    """多进程提取 κ(s) + 骨架长度 + 弯曲特征矩阵，带内容感知缓存。
    返回 (kappa, skel_len, feat_matrix, filenames, groups, dates, rows, worms)。

    feat_matrix 为 (N, len(FEATURE_NAMES)) 的 float32 矩阵，列序见
    FEATURE_NAMES。全量（约 160 万帧）下约 45 MB，避免 object dict 数组。

    mirror_invariant 会改变缓存内容（是否取 |κ|），因此并入缓存签名；
    切换开关时旧缓存自动失效。
    """
    os.makedirs(outdir, exist_ok=True)
    filenames_now = [t[0] for t in tasks]
    sig = _filesig(filenames_now) + ('_mirror' if mirror_invariant else '_signed')

    # 尝试复用缓存（旧缓存缺少 feat_matrix/mirror 标志时自动重提）
    if cache_path and os.path.exists(cache_path):
        try:
            z = np.load(cache_path, allow_pickle=True)
            if str(z['filesig']) == sig:
                print(f'[缓存] 命中 {cache_path}，跳过提取（{len(z["kappa"])} 帧）')
                return (z['kappa'], z['skel_len'], z['features'],
                        z['filenames'], z['groups'], z['dates'],
                        z['rows'], z['worms'])
            print('[缓存] 文件集或镜像开关变化，重新提取')
        except Exception as e:
            print(f'[缓存] 读取失败（{e}），重新提取')

    n = len(tasks)
    print(f'待提取 κ(s): {n:,} 帧，{workers} 进程（镜像不变={mirror_invariant}）')
    kappa_list, skel_len_list, feats_list = [], [], []
    fname_list, group_list, date_list, row_list, worm_list = \
        [], [], [], [], []
    t0 = time.time()
    done = 0
    ok = 0
    with mp.Pool(processes=workers, initializer=_init_worker,
                 initargs=(mirror_invariant,)) as pool:
        for (rel, kap, skl, fts, g, d, rw, wm) in \
                pool.imap_unordered(_extract_worker, tasks, chunksize=chunksize):
            done += 1
            if kap is not None:
                kappa_list.append(kap)
                skel_len_list.append(skl)
                feats_list.append(fts)
                fname_list.append(rel)
                group_list.append(g)
                date_list.append(d)
                row_list.append(rw)
                worm_list.append(wm)
                ok += 1
            if done % 2000 == 0 or done == n:
                rate = done / max(1e-6, time.time() - t0)
                print(f'  提取 {done:,}/{n:,}  有效 {ok:,}  '
                      f'{rate:.0f} 帧/秒', flush=True)
    print(f'提取完成：{ok:,}/{n:,} 帧有效，耗时 {time.time()-t0:.1f} 秒')

    if ok == 0:
        raise SystemExit('0 有效帧：检查输入目录或骨架提取（对比度/阈值）。')

    kappa = np.vstack(kappa_list).astype(np.float32)
    skel_len = np.asarray(skel_len_list, dtype=np.float32)
    features = np.vstack(feats_list).astype(np.float32)
    groups = np.asarray(group_list)
    dates = np.asarray(date_list)
    rows = np.asarray(row_list)
    worms = np.asarray(worm_list)
    filenames = np.asarray(fname_list)

    if cache_path:
        np.savez(cache_path, kappa=kappa, skel_len=skel_len, features=features,
                 filenames=filenames, groups=groups, dates=dates,
                 rows=rows, worms=worms, filesig=sig)
        print(f'[缓存] 已写入 {cache_path}（{len(kappa):,} 帧，'
              f'约 {kappa.nbytes/1e6:.0f} MB）')
    return kappa, skel_len, features, filenames, groups, dates, rows, worms


# ============================================================
# 2. 多准则定 k
# ============================================================
def _kneedle(ks, vals):
    """Kneedle 法：取点到首尾连线的最大距离处作为拐点。vals 单调即可。"""
    x = np.asarray(ks, dtype=float)
    y = np.asarray(vals, dtype=float)
    if len(x) < 3:
        return int(x[0])
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (y - y[0]) / (y[-1] - y[0])
    d = np.abs(yn - xn)
    return int(x[int(np.argmax(d))])


def _gap_statistic(X, k, seed, n_ref=8):
    """Gap statistic：与均匀参考分布的 log(inertia) 之差。"""
    from sklearn.cluster import KMeans
    rng = np.random.RandomState(seed + k)
    lo = X.min(axis=0)
    hi = X.max(axis=0)

    def log_w(D):
        km = KMeans(n_clusters=k, n_init=6, random_state=seed).fit(D)
        return float(np.log(km.inertia_ + 1e-12))

    log_w_data = log_w(X)
    refs = []
    for _ in range(n_ref):
        R = rng.uniform(lo, hi, size=X.shape)
        refs.append(log_w(R))
    refs = np.asarray(refs)
    return float(refs.mean() - log_w_data), float(refs.std())


def select_k(X, k_min, k_max, seed, k_sample):
    """多准则扫描 k ∈ [k_min, k_max]，返回 (结果表, 各准则最优 k, 推荐 k)。"""
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    # 子采样加速（silhouette 近似 O(n^2)，全量跑不动）
    if k_sample and len(X) > k_sample:
        idx = np.random.RandomState(seed).choice(len(X), k_sample, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    n, d = Xs.shape
    print(f'定 k 扫描：{n:,} 点 × {d} 维，k ∈ [{k_min}, {k_max}]')

    ks = list(range(k_min, k_max + 1))
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        inertia = float(km.inertia_)
        ch = float(calinski_harabasz_score(Xs, km.labels_))
        try:
            sil = float(silhouette_score(Xs, km.labels_,
                                         sample_size=min(5000, n),
                                         random_state=seed))
        except TypeError:
            sub = np.random.RandomState(seed).choice(n, min(5000, n), replace=False)
            sil = float(silhouette_score(Xs[sub], km.labels_[sub]))
        gmm = GaussianMixture(n_components=k, covariance_type='diag',
                              n_init=2, max_iter=150, random_state=seed).fit(Xs)
        bic = float(gmm.bic(Xs))
        rows.append(dict(k=k, inertia=inertia, ch=ch, sil=sil, bic=bic))
        print(f'  k={k:2d}  SSE={inertia:12.1f}  sil={sil:.4f}  '
              f'CH={ch:9.1f}  BIC={bic:10.1f}', flush=True)

    # gap（最慢，单独算）
    gaps, gsd = [], []
    for r in rows:
        g, sd = _gap_statistic(Xs, r['k'], seed, n_ref=8)
        gaps.append(g)
        gsd.append(sd)
    for r, g in zip(rows, gaps):
        r['gap'] = g
        print(f'  k={r["k"]:2d}  gap={g:.4f}', flush=True)

    iner = [r['inertia'] for r in rows]
    sils = [r['sil'] for r in rows]
    chs = [r['ch'] for r in rows]
    bics = [r['bic'] for r in rows]

    # BIC 用「肘点」而非最小值：BIC 对 k 通常单调下降，取最小值恒等于
    # k_max（过切分）。肘点 = 边际解释力骤减处，才是合理的复杂度选择。
    bic_elbow = _kneedle(ks, bics)

    best = {
        'elbow(SSE)': _kneedle(ks, iner),
        'silhouette': int(ks[int(np.argmax(sils))]),
        'CH': int(ks[int(np.argmax(chs))]),
        'BIC_min': int(ks[int(np.argmin(bics))]),
        'BIC_elbow': bic_elbow,
        'gap': int(ks[int(np.argmax(gaps))]),
    }

    # ---- 推荐规则：silhouette × BIC 肘点 ----
    # 两者一致 → 直接采纳（最强证据）；
    # 不一致 → 取较小 k（保守，避免过切分），并显式报告分歧。
    # 「分歧本身」是论文可用的诚实结论：姿态更接近连续谱而非离散类。
    k_sil = best['silhouette']
    k_bic = bic_elbow
    if k_sil == k_bic:
        reco = k_sil
        rule = 'silhouette 与 BIC 肘点一致'
    else:
        reco = min(k_sil, k_bic)
        rule = (f'silhouette(k={k_sil}) 与 BIC 肘点(k={k_bic}) 分歧，'
                f'取较小 k={reco}（保守，避免过切分）')
    print('\n各准则最优 k:', {k: v for k, v in best.items()})
    print(f'推荐 k = {reco}（{rule}）')
    if k_sil != k_bic:
        print('  ⚠ 两准则分歧：姿态可能为连续谱，论文中应如实报告而非强行取一类')
    return rows, best, reco


# ============================================================
# 3. 最终聚类 + 每虫直方图 + 剂量位移检验
# ============================================================
def _bh(pvals):
    """Benjamini-Hochberg FDR 校正。返回校正后 p 值数组。"""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        cur = pvals[i] * n / (rank + 1)
        cur = min(cur, prev)
        prev = cur
        adj[i] = cur
    return adj


def _cohen_d(a, b):
    """Cohen's d（合并标准差）。返回 (d, lo, hi)（近似 95%CI）。"""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float('nan'), float('nan'), float('nan')
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    s1, s2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled < 1e-12:
        return 0.0, 0.0, 0.0
    d = (m2 - m1) / pooled
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2 - 2)))
    return d, d - 1.96 * se, d + 1.96 * se


def _epsilon_squared(h, n, df):
    """Kruskal-Wallis 的效应量 ε² = (H - df + 1) / (n - df)。"""
    if n - df <= 0:
        return float('nan')
    return float((h - df + 1) / (n - df))


def _cochran_armitage(scores, successes, totals):
    """Cochran-Armitage 剂量趋势检验（比例对有序剂量的线性趋势）。

    scores    : 各剂量组的分数（如 0,1,2,3）
    successes : 各剂量组"深弯"虫数
    totals    : 各剂量组总虫数
    返回 (Z, p_two_sided)。用正态近似，无需 scipy 特殊函数。
    """
    scores = np.asarray(scores, dtype=float)
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    N = totals.sum()
    if N < 2 or totals.min() < 1:
        return float('nan'), float('nan')
    pbar = successes.sum() / N
    if pbar <= 0 or pbar >= 1:
        return float('nan'), float('nan')
    sbar = (scores * totals).sum() / N
    num = (successes * (scores - sbar)).sum()
    var = pbar * (1 - pbar) * (totals * (scores - sbar) ** 2).sum()
    if var <= 0:
        return float('nan'), float('nan')
    z = num / np.sqrt(var)
    # 双尾 p，用 erfc 近似（math.erfc 在标准库）
    import math
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return float(z), float(p)


def _cluster_and_dose(X, kappa_cluster, feat_matrix, skel_len, groups, dates,
                      rows, worms, filenames, k, seed, min_worm_frames, outdir,
                      physical_curvature=False, cluster_sample=None):
    """最终 k-means + 每虫直方图 + 每簇弯曲特征统计 + 剂量位移检验。

    X                 : 聚类用的特征矩阵（可能已 PCA 降维）
    kappa_cluster     : 用于聚类的 κ(s) 矩阵（已做镜像规范化）
    feat_matrix       : (N, len(FEATURE_NAMES)) 弯曲特征矩阵，列序见 FEATURE_NAMES
    skel_len          : 每帧骨架长度（px）
    physical_curvature: κ 是否已除以 L 转为物理曲率（仅用于输出标注）
    cluster_sample    : KMeans 拟合的最大样本数（None=全量）。全量帧仍全部
                        参与 predict 与直方图统计，仅簇心估计用子样本。
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples
    from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

    # ---- KMeans：子采样拟合 + 全量 predict ----
    n_total = len(X)
    if cluster_sample and n_total > cluster_sample:
        rng = np.random.RandomState(seed)
        fit_idx = rng.choice(n_total, cluster_sample, replace=False)
        print(f'KMeans 拟合：子采样 {cluster_sample:,}/{n_total:,} 帧'
              f'（全量帧仍参与 predict 与统计）')
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X[fit_idx])
    else:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    labels = km.predict(X)

    # ---- 每帧聚类大小 ----
    counts = np.bincount(labels, minlength=k)
    print('\nClustering result (descending size):')
    order = np.argsort(-counts)
    for r, c in enumerate(order):
        print(f'  cluster {c:2d}: {counts[c]:8,} frames ({100*counts[c]/len(labels):5.1f}%)')

    # ---- 簇内 silhouette（子采样，避免 O(n^2)）----
    sil_means = {}
    try:
        sil_n = min(20000, n_total)
        rng = np.random.RandomState(seed)
        sidx = rng.choice(n_total, sil_n, replace=False)
        sv = silhouette_samples(X[sidx], labels[sidx])
        for c in range(k):
            sil_means[c] = float(sv[labels[sidx] == c].mean())
    except Exception:
        sil_means = {c: float('nan') for c in range(k)}

    # ---- 每虫姿态直方图 ----
    worm_key = np.asarray([f'{g}/{d}/{rw}/{wm}' for g, d, rw, wm in
                           zip(groups, dates, rows, worms)])
    keys, first_idx, inv = np.unique(worm_key, return_index=True, return_inverse=True)
    n_worms = len(keys)

    # 逐虫聚合（np.add.at 一次性完成：onehot[i] 加到 hist[inv[i]]）
    onehot = np.zeros((len(labels), k), dtype=np.int64)
    onehot[np.arange(len(labels)), labels] = 1
    hist = np.zeros((n_worms, k), dtype=np.int64)
    np.add.at(hist, inv, onehot)

    worm_n = hist.sum(axis=1)
    worm_group = groups[first_idx]
    worm_date = dates[first_idx]
    worm_row = rows[first_idx]
    worm_name = worms[first_idx]
    worm_prop = hist / np.maximum(worm_n[:, None], 1)

    # 写每虫直方图 CSV
    csv_path = os.path.join(outdir, 'atlas_worm_hist.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['group', 'date', 'row', 'worm', 'n_frames']
                   + [f'c{c}' for c in range(k)]
                   + [f'p{c}' for c in range(k)])
        for j in range(n_worms):
            w.writerow([worm_group[j], worm_date[j], worm_row[j], worm_name[j],
                        int(worm_n[j])] + list(hist[j]) +
                       [f'{v:.5f}' for v in worm_prop[j]])
    print(f'\nPer-worm posture histogram written: {csv_path} ({n_worms} worms)')

    # ---- 每簇弯曲特征统计（mean/std/median/IQR/p5/p95）----
    cluster_feat_path = os.path.join(outdir, 'atlas_cluster_features.csv')
    with open(cluster_feat_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        header = ['cluster_id', 'n_frames', 'mean_silhouette'] + \
                 [f'{stat}_{fn}' for fn in FEATURE_NAMES
                  for stat in ('mean', 'std', 'median', 'iqr', 'p5', 'p95')] + \
                 ['mean_skeleton_len', 'std_skeleton_len',
                  'median_skeleton_len', 'iqr_skeleton_len']
        w.writerow(header)
        for c in order:
            mask = labels == c
            row = [c, int(mask.sum()), f'{sil_means[c]:.4f}']
            for j in range(len(FEATURE_NAMES)):
                vals = feat_matrix[mask, j]
                q1, q3 = np.percentile(vals, [25, 75])
                row += [f'{vals.mean():.5f}', f'{vals.std():.5f}',
                        f'{np.median(vals):.5f}', f'{q3-q1:.5f}',
                        f'{np.percentile(vals,5):.5f}',
                        f'{np.percentile(vals,95):.5f}']
            sl = skel_len[mask]
            q1, q3 = np.percentile(sl, [25, 75])
            row += [f'{sl.mean():.2f}', f'{sl.std():.2f}',
                    f'{np.median(sl):.2f}', f'{q3-q1:.2f}']
            w.writerow(row)
    print(f'Per-cluster feature statistics written: {cluster_feat_path}')

    # ---- 每簇代表曲线 κ(s)（mean ± 95%CI），论文核心图的数据 ----
    prof_path = os.path.join(outdir, 'atlas_cluster_profiles.csv')
    s_axis = np.linspace(0.0, 1.0, kappa_cluster.shape[1])
    profiles = {}
    with open(prof_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['cluster_id', 's', 'mean', 'ci_lo', 'ci_hi', 'n'])
        for c in order:
            mask = labels == c
            sub = kappa_cluster[mask]
            m = sub.mean(axis=0)
            if mask.sum() > 1:
                se = sub.std(axis=0, ddof=1) / np.sqrt(mask.sum())
                tcrit = 1.96 if mask.sum() > 100 else 2.0
                lo, hi = m - tcrit * se, m + tcrit * se
            else:
                lo = hi = m
            profiles[c] = (m, lo, hi)
            for i in range(len(s_axis)):
                w.writerow([c, f'{s_axis[i]:.4f}', f'{m[i]:.5f}',
                            f'{lo[i]:.5f}', f'{hi[i]:.5f}', int(mask.sum())])
    print(f'Per-cluster kappa(s) profiles written: {prof_path}')

    # 控制台简表
    print('\n=== Per-cluster bending features (mean +/- SD), '
          + ('[physical curvature kappa/L, 1/px]' if physical_curvature
             else '[normalized-arc kappa(s), scale-invariant]') + ' ===')
    for c in order:
        mask = labels == c
        bbi_m = feat_matrix[mask, 0].mean()
        bbi_s = feat_matrix[mask, 0].std()
        max_m = feat_matrix[mask, 1].mean()
        max_s = feat_matrix[mask, 1].std()
        nr_m = feat_matrix[mask, 2].mean()
        md_m = feat_matrix[mask, 3].mean()
        pm_m = feat_matrix[mask, 4].mean()
        print(f'  cluster {c:2d}: n={mask.sum():8,}  sil={sil_means[c]:.3f}  '
              f'bbi={bbi_m:.3f}+/-{bbi_s:.3f}  max_kappa={max_m:.3f}+/-{max_s:.3f}  '
              f'n_regions={nr_m:.2f}  mid_dom={md_m:.2f}  peak_mid={pm_m:.2f}')

    # ---- 剂量位移检验（过滤帧数过少的虫）----
    keep = worm_n >= min_worm_frames
    print(f'\nDose test: {keep.sum()}/{n_worms} worms included '
          f'(>= {min_worm_frames} frames)')
    wg = worm_group[keep]
    wd = worm_date[keep]
    wp = worm_prop[keep]
    gorder = resolve_group_order(wg)   # 动态组序：兼容 train/test 等非给药组

    # 每虫主姿态（argmax 簇）→ 名目列联表卡方
    argmax_cluster = wp.argmax(axis=1)
    agg = np.zeros((len(gorder), k))
    for gi, g in enumerate(gorder):
        m = wg == g
        if m.any():
            agg[gi] = np.bincount(argmax_cluster[m], minlength=k)
    used = agg.sum(axis=0) > 0

    chi2_res = {}
    print('\n=== Dose shift tests ===')
    if used.sum() >= 2:
        sub = agg[:, used]
        # 检查稀疏性：含 0 cell 或最小期望频数 <5 时卡方不可靠/报错
        expected = np.outer(sub.sum(axis=1), sub.sum(axis=0)) / max(sub.sum(), 1)
        if (sub == 0).any() or expected.min() < 5:
            print('Main-posture contingency table too sparse (zero cell or '
                  'min expected freq < 5); chi-square skipped. '
                  'See per-cluster Kruskal-Wallis below.')
            chi2_res = {'skipped': True}
        else:
            chi2, p_chi, dof, _ = chi2_contingency(sub)
            cramer = float(np.sqrt(chi2 / (sub.sum() * (min(sub.shape) - 1))))
            chi2_res = {'chi2': chi2, 'df': dof, 'p': p_chi, 'cramer_v': cramer}
            print(f'Main posture group x cluster chi-square: '
                  f'chi2={chi2:.3f}, df={dof}, p={p_chi:.4g}, '
                  f"Cramer's V={cramer:.3f}")
    else:
        print('Main-posture contingency table degenerate (<2 clusters); '
              'chi-square skipped')
        chi2_res = {'skipped': True}

    # 每簇 Kruskal-Wallis + ε² + 事后 Mann-Whitney（BH 校正）+ Cohen's d + 趋势检验
    print('\nPer-cluster group differences (Kruskal-Wallis on per-worm proportion):')
    kws, eps2s = [], []
    for c in range(k):
        per = [wp[wg == g, c] for g in gorder if (wg == g).any()]
        h, p = kruskal(*per)
        kws.append(p)
        eps2s.append(_epsilon_squared(h, sum(len(x) for x in per), len(per) - 1))
        print(f'  cluster {c:2d}: H={h:.3f}, p={p:.4g}, epsilon^2={eps2s[-1]:.4f}')
    padj = _bh(kws)
    print('BH-adjusted p:', ', '.join(f'c{c}={padj[c]:.3g}' for c in range(k)))
    sig = [c for c in range(k) if padj[c] < 0.05]
    print(f'Significant clusters after BH: {sig if sig else "none"}')

    # 事后两两比较（Mann-Whitney，BH 校正）+ 效应量 d
    print('\nPost-hoc pairwise (Mann-Whitney U, BH-adjusted) and Cohen\'s d:')
    pair_rows = []
    pair_p = []
    pair_meta = []
    base = 'control' if 'control' in gorder else gorder[0]
    for c in range(k):
        for g in gorder[1:]:
            if g == base:
                continue
            a = wp[wg == base, c]
            b = wp[wg == g, c]
            if len(a) < 3 or len(b) < 3:
                continue
            u, p = mannwhitneyu(b, a, alternative='two-sided')
            d, dlo, dhi = _cohen_d(a, b)
            pair_p.append(p)
            pair_meta.append((c, g, u, d, dlo, dhi))
    if pair_p:
        padj_pair = _bh(pair_p)
        for (c, g, u, d, dlo, dhi), p_raw, p_adj in zip(pair_meta, pair_p, padj_pair):
            pair_rows.append(dict(cluster=c, comparison=f'{base} vs {g}',
                                  U=u, p_raw=p_raw, p_bh=p_adj, cohens_d=d,
                                  d_lo=dlo, d_hi=dhi))
            star = ' *' if p_adj < 0.05 else ''
            print(f'  c{c} {base} vs {g:>5s}: U={u:.0f}, p_raw={p_raw:.4g}, '
                  f'p_BH={p_adj:.4g}{star}, d={d:.3f} [{dlo:.3f}, {dhi:.3f}]')

    # Cochran-Armitage 剂量趋势（每虫主姿态 = 深弯/非深弯，用 argmax）
    trend_rows = []
    if 'control' not in gorder or len(gorder) < 3:
        print('\nCochran-Armitage dose trend skipped '
              f'(groups not dose-ordered: {gorder})')
    else:
      print('\nCochran-Armitage dose trend (per-worm main posture):')
      scores = [0, 1, 2, 3]
      for c in range(k):
        succ = np.array([(argmax_cluster[wg == g] == c).sum() for g in gorder],
                        dtype=float)
        tot = np.array([(wg == g).sum() for g in gorder], dtype=float)
        z, p = _cochran_armitage(scores, succ, tot)
        trend_rows.append(dict(cluster=c, successes=succ.tolist(),
                               totals=tot.tolist(), z=z, p=p))
        print(f'  cluster {c:2d}: successes={succ.astype(int).tolist()}, '
              f'totals={tot.astype(int).tolist()}, Z={z:.3f}, p={p:.4g}')

    # 汇总 CSV（论文表格数据源）
    dose_csv = os.path.join(outdir, 'atlas_dose_tests.csv')
    with open(dose_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['test', 'cluster', 'comparison', 'statistic', 'p_value',
                    'p_bh', 'effect_type', 'effect_size', 'ci_lo', 'ci_hi'])
        if 'chi2' in chi2_res:
            w.writerow(['chi2_contingency', 'main_posture', '4groups x kclusters',
                        f"{chi2_res['chi2']:.4f}", f"{chi2_res['p']:.6g}", '',
                        'cramers_v', f"{chi2_res['cramer_v']:.4f}", '', ''])
        for c in range(k):
            w.writerow(['kruskal_wallis', f'c{c}', '4groups',
                        f'{kws[c]:.4f}', f'{padj[c]:.6g}', f'{padj[c]:.6g}',
                        'epsilon_squared', f'{eps2s[c]:.4f}', '', ''])
        for r in pair_rows:
            w.writerow(['mann_whitney', f"c{r['cluster']}", r['comparison'],
                        f"{r['U']:.1f}", f"{r['p_raw']:.6g}", f"{r['p_bh']:.6g}",
                        'cohens_d', f"{r['cohens_d']:.4f}",
                        f"{r['d_lo']:.4f}", f"{r['d_hi']:.4f}"])
        for r in trend_rows:
            w.writerow(['cochran_armitage', f"c{r['cluster']}", 'dose_trend',
                        f"{r['z']:.4f}", f"{r['p']:.6g}", '',
                        'z', f"{r['z']:.4f}", '', ''])
    print(f'\nDose test summary written: {dose_csv}')

    # ---- 图 1：每簇代表曲线 κ(s)（论文核心图）----
    prof_fig = os.path.join(outdir, 'atlas_cluster_profiles.png')
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, c in enumerate(order):
        m, lo, hi = profiles[c]
        col = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.plot(s_axis, m, lw=2.0, color=col,
                label=f'Cluster {c} (n={counts[c]:,}, {100*counts[c]/len(labels):.1f}%)')
        ax.fill_between(s_axis, lo, hi, color=col, alpha=0.20, lw=0)
    ax.axvline(0.5, ls=':', color='#888', lw=1)
    ax.text(0.5, ax.get_ylim()[1] * 0.97, 'body midpoint', ha='center',
            va='top', fontsize=9, color='#666')
    ax.set_xlabel('Normalized body position $s$ (0 = head, 1 = tail)')
    ax.set_ylabel(r'Curvature $\kappa(s)$' +
                  (r' (1/px)' if physical_curvature else ' (scale-invariant)'))
    ax.set_title('Mean curvature profiles per cluster (shaded = 95% CI)')
    ax.legend(frameon=False, loc='upper right')
    ax.set_xlim(0, 1)
    plt.savefig(prof_fig)
    plt.close(fig)
    print(f'Figure saved: {prof_fig}')

    # ---- 图 2：剂量位移（堆叠条 + 每虫 2D 嵌入）----
    fig_path = os.path.join(outdir, 'atlas_dose_shift.png')
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    # 每组平均簇占比（每虫等权）
    prop_by_group = np.zeros((len(gorder), k))
    for gi, g in enumerate(gorder):
        m = worm_group == g
        if m.any():
            prop_by_group[gi] = worm_prop[m].mean(axis=0)
    bottoms = np.zeros(len(gorder))
    for i, c in enumerate(order):
        ax.bar(gorder, prop_by_group[:, c], bottom=bottoms,
               label=f'Cluster {c}', width=0.62,
               color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)])
        bottoms += prop_by_group[:, c]
    ax.set_ylabel('Mean cluster proportion')
    ax.set_ylim(0, 1.0)
    ax.set_title(f'Posture atlas by dose group (k = {k})')
    ax.legend(frameon=False, loc='upper right', ncols=2)

    ax = axes[1]
    from sklearn.decomposition import PCA
    # 每虫平均 κ(s) 的 2D 嵌入（用聚类输入的 κ，保证与簇一致）
    mean_kappa = np.zeros((n_worms, kappa_cluster.shape[1]), dtype=np.float32)
    np.add.at(mean_kappa, inv, kappa_cluster)
    mean_kappa /= np.maximum(worm_n[:, None], 1)
    pca2 = PCA(n_components=2, random_state=seed).fit(mean_kappa)
    emb = pca2.transform(mean_kappa)
    for g in gorder:
        m = worm_group == g
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.65, label=g,
                       color=PALETTE.get(g, '#000'), edgecolors='none')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Per-worm mean curvature shape (2D PCA)')
    ax.legend(frameon=False, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)
    print(f'Figure saved: {fig_path}')

    return labels, counts, order


# ============================================================
# 主程序
# ============================================================
def main():
    ap = argparse.ArgumentParser(description='无监督姿态形状图谱聚类 + 剂量位移检验')
    ap.add_argument('-i', '--input', default=get_data_root(), help='图片根目录')
    ap.add_argument('-o', '--outdir', default='atlas_out', help='输出目录')
    ap.add_argument('--workers', type=int, default=None, help='并行进程数')
    ap.add_argument('--chunksize', type=int, default=200)
    ap.add_argument('--max-worms', type=int, default=None,
                    help='每组（control/5mg/8mg/10mg）最多处理 N 只虫（冒烟，四组均衡）')
    ap.add_argument('--max-frames', type=int, default=None, help='最多处理 N 帧（冒烟）')
    ap.add_argument('--k-min', type=int, default=2)
    ap.add_argument('--k-max', type=int, default=15)
    ap.add_argument('--k', type=int, default=None, help='强制簇数（跳过定 k 扫描）')
    ap.add_argument('--no-pca', action='store_true', help='直接对 κ(s) 聚类，不做 PCA 降维')
    ap.add_argument('--pca-components', type=int, default=None,
                    help='PCA 分量数（默认自动，覆盖 95%% 方差）')
    ap.add_argument('--k-sample', type=int, default=20000, help='定 k 扫描子采样点数')
    ap.add_argument('--cluster-sample', type=int, default=300000,
                    help='KMeans 拟合的最大样本数（全量帧仍全部参与 predict 与统计）')
    ap.add_argument('--pca-sample', type=int, default=300000,
                    help='PCA 拟合的最大样本数（全量帧仍全部 transform）')
    ap.add_argument('--min-worm-frames', type=int, default=10,
                    help='剂量检验时每虫最少帧数')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-extract', action='store_true', help='跳过提取，用缓存')
    ap.add_argument('--no-cache', action='store_true', help='不写/不读缓存')
    ap.add_argument('--physical-curvature', action='store_true',
                    help='把 κ(s) 除以骨架长度转成物理曲率（1/px），'
                         '仅用于与旧行为分类管线阈值（BBI 0.16 / max_kappa 0.35）对照。'
                         '注意：该操作会引入体长依赖，与"大小不变"目标相反，默认关闭')
    ap.add_argument('--mirror-invariant', dest='mirror_invariant',
                    action='store_true', default=True,
                    help='取 |κ(s)|，使左右弯（镜像）等价（默认开启）')
    ap.add_argument('--no-mirror-invariant', dest='mirror_invariant',
                    action='store_false',
                    help='关闭镜像不变，保留曲率符号（需 compute_curvature_profile 返回有符号 κ）')
    args = ap.parse_args()

    input_dir = os.path.abspath(args.input)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    workers = args.workers or max(1, os.cpu_count())

    from sklearn.decomposition import PCA

    # ---- Stage 1: 提取 ----
    cache_path = None if args.no_cache else os.path.join(outdir, 'kappa_cache.npz')
    mirror_tag = '_mirror' if args.mirror_invariant else '_signed'
    if args.skip_extract and cache_path and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        cached_sig = str(z['filesig'])
        # 兼容 2026-10-24 之前的旧缓存：无 mirror 后缀，但旧代码
        # canonicalize 默认 mirror_invariant=True，故视为 mirror 口径。
        if not (cached_sig.endswith(mirror_tag) or
                (mirror_tag == '_mirror' and
                 not cached_sig.endswith('_signed'))):
            raise SystemExit(
                f'[缓存] 镜像开关不一致：缓存为 '
                f'{"mirror" if mirror_tag == "_signed" else "signed"} 口径，'
                f'当前请求 {"mirror" if mirror_tag == "_mirror" else "signed"}。'
                f'请去掉 --skip-extract 重新提取，或切换 --mirror-invariant 开关。')
        if not cached_sig.endswith(('_mirror', '_signed')):
            print('[缓存] 旧格式缓存（无 mirror 标志），按 mirror 口径载入')
        kappa = z['kappa'].astype(np.float32)
        skel_len = z['skel_len'].astype(np.float32)
        groups, dates, rows, worms = z['groups'], z['dates'], z['rows'], z['worms']
        filenames = z['filenames']
        print(f'[缓存] 载入 {len(kappa):,} 帧')
        # 旧缓存 features 为 object dict 数组；新缓存为 float32 矩阵。
        if z['features'].ndim == 1:
            print('[缓存] 旧格式特征（object），从 κ(s) 重建为矩阵……')
            feat_matrix = np.zeros((len(kappa), len(FEATURE_NAMES)), dtype=np.float32)
            for i, kc in enumerate(kappa):
                fts = extract_features(kc)
                for j, fn in enumerate(FEATURE_NAMES):
                    feat_matrix[i, j] = float(fts.get(fn, 0.0))
            print(f'[缓存] 特征矩阵重建完成 {feat_matrix.shape}')
        else:
            feat_matrix = z['features'].astype(np.float32)
    else:
        tasks = list_tasks(input_dir, args.max_worms, args.max_frames)
        print(f'扫描到 {len(tasks):,} 帧待处理')
        if not tasks:
            raise SystemExit('无图片，检查 -i 路径')
        kappa, skel_len, feat_matrix, filenames, groups, dates, rows, worms = \
            extract_all(tasks, workers, args.chunksize, cache_path, outdir,
                        mirror_invariant=args.mirror_invariant)

    print(f'\nκ(s) matrix: {kappa.shape[0]:,} frames x {kappa.shape[1]} dims')
    print(f'Mirror invariant: {args.mirror_invariant} '
          f'({"|kappa|, left/right bends equivalent" if args.mirror_invariant else "signed kappa"})')

    # ---- Stage 1b: 曲率尺度口径 ----
    # κ(s) = |sin(Δθ)| 关于归一化弧长 s 的分布，已天然尺度不变（相似形
    # 缩放不改变夹角）。默认直接使用，无需也不应做长度缩放。
    if args.physical_curvature:
        kappa_cluster = (kappa / np.maximum(skel_len[:, None], 1e-6)).astype(np.float32)
        print(f'Physical curvature: kappa(s) / L (mean L={skel_len.mean():.1f}px) '
              f'-- note this re-introduces body-length dependence')
    else:
        kappa_cluster = kappa
        print(f'Using kappa(s) as-is (normalized-arc, scale-invariant; '
              f'mean L={skel_len.mean():.1f}px for reference only)')

    # ---- Stage 2: PCA（子采样拟合，全量 transform）----
    if not args.no_pca:
        n_all = len(kappa_cluster)
        if args.pca_sample and n_all > args.pca_sample:
            rng = np.random.RandomState(args.seed)
            pfit_idx = rng.choice(n_all, args.pca_sample, replace=False)
            print(f'PCA fit on subsample: {args.pca_sample:,}/{n_all:,} frames')
        else:
            pfit_idx = np.arange(n_all)
        pca = PCA(n_components=args.pca_components, random_state=args.seed)
        pca.fit(kappa_cluster[pfit_idx])
        if args.pca_components is None:
            # 覆盖 95% 方差自动定维
            cum = np.cumsum(pca.explained_variance_ratio_)
            n_comp = int(np.searchsorted(cum, 0.95) + 1)
            n_comp = max(1, min(n_comp, kappa_cluster.shape[1]))
            pca = PCA(n_components=n_comp, random_state=args.seed).fit(kappa_cluster[pfit_idx])
        X = pca.transform(kappa_cluster).astype(np.float32)
        print(f'PCA: {kappa_cluster.shape[1]} -> {X.shape[1]} dims '
              f'(cumulative variance '
              f'{np.cumsum(pca.explained_variance_ratio_)[-1]*100:.1f}%)')
    else:
        X = kappa_cluster

    # ---- Stage 3: 定 k ----
    if args.k:
        k = args.k
        print(f'强制 k = {k}')
    else:
        k_rows, best, k = select_k(X, args.k_min, args.k_max, args.seed, args.k_sample)
        # 画 k 选择图（论文用：全英文、300 dpi、色盲友好配色）
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6))
        ks = [r['k'] for r in k_rows]
        ax = axes[0, 0]
        ax.plot(ks, [r['inertia'] for r in k_rows], 'o-')
        ax.axvline(best['elbow(SSE)'], ls='--', color='#D55E00', label='knee')
        ax.set_title('Elbow (SSE)'); ax.set_xlabel('k'); ax.legend(frameon=False)
        ax = axes[0, 1]
        ax.plot(ks, [r['sil'] for r in k_rows], 'o-')
        ax.axvline(best['silhouette'], ls='--', color='#009E73')
        ax.set_title('Silhouette'); ax.set_xlabel('k')
        ax = axes[0, 2]
        ax.plot(ks, [r['ch'] for r in k_rows], 'o-')
        ax.axvline(best['CH'], ls='--', color='#009E73')
        ax.set_title('Calinski-Harabasz'); ax.set_xlabel('k')
        ax = axes[1, 0]
        ax.plot(ks, [r['bic'] for r in k_rows], 'o-')
        ax.axvline(best['BIC_elbow'], ls='--', color='#D55E00', label='elbow')
        ax.set_title('BIC (GMM, elbow)'); ax.set_xlabel('k'); ax.legend(frameon=False)
        ax = axes[1, 1]
        ax.plot(ks, [r['gap'] for r in k_rows], 'o-')
        ax.axvline(best['gap'], ls='--', color='#009E73')
        ax.set_title('Gap statistic'); ax.set_xlabel('k')
        ax = axes[1, 2]
        ax.axis('off')
        note = f'Recommended k = {k}\nRule: silhouette x BIC-elbow\n\n'
        if best['silhouette'] != best['BIC_elbow']:
            note += (f"silhouette k={best['silhouette']} vs "
                     f"BIC-elbow k={best['BIC_elbow']}\n"
                     f"-> disagreement; smaller k kept\n"
                     f"(postures likely form a continuum)\n\n")
        ax.text(0.03, 0.97, note + '\n'.join(
            f'{kk}: {vv}' for kk, vv in best.items()), fontsize=10,
            va='top', family='monospace')
        plt.tight_layout()
        ksel_path = os.path.join(outdir, 'atlas_k_selection.png')
        plt.savefig(ksel_path)
        plt.close(fig)
        print(f'\nFigure saved: {ksel_path}')

    # ---- Stage 4: 聚类 + 直方图 + 每簇特征统计 + 剂量检验 ----
    _cluster_and_dose(X, kappa_cluster, feat_matrix, skel_len, groups, dates,
                      rows, worms, filenames, k, args.seed, args.min_worm_frames,
                      outdir, physical_curvature=args.physical_curvature,
                      cluster_sample=args.cluster_sample)

    print('\nAll done. Outputs:')
    print(f'  {outdir}/kappa_cache.npz             kappa(s) + skeleton length + features')
    print(f'  {outdir}/atlas_k_selection.png       k-selection (silhouette x BIC elbow)')
    print(f'  {outdir}/atlas_worm_hist.csv         per-worm posture histogram')
    print(f'  {outdir}/atlas_cluster_features.csv  per-cluster feature statistics')
    print(f'  {outdir}/atlas_cluster_profiles.csv  per-cluster mean kappa(s) curves')
    print(f'  {outdir}/atlas_dose_tests.csv        dose-shift test summary')
    print(f'  {outdir}/atlas_cluster_profiles.png  per-cluster mean kappa(s) curves (figure)')
    print(f'  {outdir}/atlas_dose_shift.png        dose shift figure')


if __name__ == '__main__':
    main()
