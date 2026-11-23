"""step14_continuity_crossval.py — 「姿态是离散类别还是连续梯度」的多表示×多算法×多判据交叉验证。

为什么需要这个脚本
------------------
cluster_atlas.py 用 k-means + 五准则在 κ(s) 空间发现"准则分歧"，据此提出"姿态为连续谱"。
该论证有三个可被审稿人一击的弱点：
  ① 只用一种算法族（k-means 隐含球状/凸簇假设，且强行给出 k 个簇）；
  ② 只用一种表示（κ(s) 的 PCA 空间）；
  ③ "准则分歧"本身也可能源于噪声或数据退化，未必等于连续。

本脚本把论证升级为交叉验证：4 种形状表示 × 5 类聚类算法 × 6 类判据，回答同一个问题。

关键设计：一致性本身不是证据
------------------------------
在一条连续的一维分布上，任何合理算法都会在同一位置下刀，跨算法一致性（ARI）会很高——
高一致性只说明"这是同一把刀"，不说明"存在两类"。因此必须同时检验密度谷与内部效度，
二者联合才能区分"离散类别"与"连续体上的任意切分"。若只报一致性而隐去谷与轮廓，
就会把"任意切分"误报成"两类真实"，这正是本脚本要避免的自圆其说。

预注册的否证条件（先声明、后检验）
----------------------------------
若存在某表示 R 与某算法 A，使某 k≥2 的解同时满足：
  F1  轮廓系数 > 0.50（Kaufman–Rousseeuw「强结构」带）
  F2  沿主轴 KDE 存在显著谷：显著峰数 ≥ 2 且谷深 > 0.25
  F3  自助重采样下簇归属 ARI > 0.75
则支持"离散类别"，本文的连续假设被推翻。

支持条件（全部成立才判定为"单轴连续梯度"）
  S1  所有 表示×算法×k 的最大轮廓系数 < 0.35（弱结构带）
  S2  所有表示的最大谷深 < 0.15（或显著峰数 < 2）
  S3  DBSCAN 在宽 eps 扫描范围内，不存在"≥2 簇且覆盖 ≥80% 点且最大簇 ≤90%"的密度可分划分
  S4  Ward 树状图在 k=2..k_max 范围内无合并高度断层（相邻高度比 < 2.0）
  S5  k=2 划分在表示间/算法间 ARI 较高 —— 仅在 S1–S4 同时成立时，才解释为
      "同一把刀切在连续体上"；否则 S5 不可单独引用

解释纪律
--------
- 不得只报 S5（跨方法一致性）而隐去 S1–S4。
- 不得只报单一表示/单一算法的结果；完整矩阵必须落 CSV。
- 每种表示、每种算法、每个 k 的轮廓/CH/DB/BIC、稳定 ARI 都要可查。

四种表示（几何上彼此独立）
  R1 SBPD   : κ(s) 的 PCA 主成分子空间（累计方差 95%）—— 学习型线性子空间
  R2 RAW    : 100 维 κ(s) 原值（z 化）—— 完整轮廓，无降维
  R3 FOURIER: κ(s) 的 11 阶傅里叶幅值 —— 频域几何，与空间域正交的编码
  R4 FEAT   : 8 维可解释弯曲特征 —— 手工描述符空间，与 PCA 无关

五类算法（归纳偏置互不相同）
  k-means   质心/球状；GMM 概率混合（BIC 选 k）；Ward 层次（合并高度断层）；
  DBSCAN    密度连接（不指定 k，直接检验"密度可分"）；Spectral 图划分。

用法（ele 环境）
--------------
  # 冒烟（约 1-2 分钟）
  python step14_continuity_crossval.py -c atlas_out_full/kappa_cache.npz \
      -o atlas_out_full/continuity --quick

  # 正式（约 10-20 分钟）
  python step14_continuity_crossval.py -c atlas_out_full/kappa_cache.npz \
      -o atlas_out_full/continuity

产物
----
  continuity_validity.csv     每 (表示, 算法, k) 的轮廓/CH/DB/BIC
  continuity_stability.csv    每 (表示, 算法, k=2) 的自助重采样 ARI
  continuity_modality.csv     每表示的主轴模态（GMM-BIC 最优分量、峰数、谷深）
  continuity_dbscan.csv       每表示的 eps 扫描（簇数、噪声比、覆盖、最大簇占比）
  continuity_agreement.csv    表示间 / 算法间 k=2 的 ARI 矩阵
  continuity_ward_tree.csv    每表示的 Ward 合并高度与相邻比
  continuity_verdict.txt      预设条件判定（S1–S5 / F1–F3）与最终结论
  fig8_continuity.png         四联图（轮廓谱、主轴密度、eps 扫描、ARI 热图）
"""

import os
import csv
import argparse
import numpy as np

GROUP_ORDER = ['control', '5mg', '8mg', '10mg']


# ============================================================
# 0. 工具
# ============================================================

def write_csv(path, rows, fields=None):
    """写 CSV（无 pandas 依赖）。rows 为 list[dict] 或 list[tuple]。"""
    if not rows:
        return
    if fields is None:
        # 汇总所有行出现过的键（各行字段可不一致），避免 DictWriter 报
        # "dict contains fields not in fieldnames"（2026-11-14 修复）
        fields = []
        for r in rows:
            for k in r.keys():
                if k not in fields:
                    fields.append(k)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'  已写 {path}')


def zscore(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return ((X - mu) / sd).astype(np.float32)


def load_cache(path):
    z = np.load(path, allow_pickle=True)
    feats = z['features'] if 'features' in z else None
    if feats is not None and feats.dtype == object:
        raise SystemExit(
            '缓存 features 为旧版 object 数组。请先用 cluster_atlas.py '
            '重新生成缓存（--skip-extract 即可，特征会重建为 float32 矩阵）。')
    d = {
        'kappa': z['kappa'].astype(np.float32),
        'skel_len': z['skel_len'].astype(np.float32) if 'skel_len' in z else None,
        'features': np.asarray(feats, dtype=np.float32) if feats is not None else None,
        'groups': z['groups'] if 'groups' in z else None,
    }
    print(f'[载入] {path}: {d["kappa"].shape[0]:,} 帧 × {d["kappa"].shape[1]} 维')
    return d


def stratified_idx(groups, n_take, seed):
    """按组分层抽样索引；组缺失时退化为随机抽样。"""
    rng = np.random.RandomState(seed)
    if groups is None:
        return None
    uniq = [g for g in GROUP_ORDER if (groups == g).any()]
    if not uniq:
        return None
    per = max(1, n_take // len(uniq))
    idx = [rng.choice(np.flatnonzero(groups == g),
                      min(per, int((groups == g).sum())), replace=False)
           for g in uniq]
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx[:n_take]


# ============================================================
# 1. 四种表示
# ============================================================

class Representations:
    """在给定数据块上构造四种表示；PCA 在主块上拟合、其余块复用同一变换。"""

    def __init__(self, kappa_main):
        from sklearn.decomposition import PCA
        fit_n = min(50000, len(kappa_main))
        self.pca = PCA(n_components=0.95, svd_solver='full',
                       random_state=0).fit(kappa_main[:fit_n])
        self.n_comp = int(self.pca.n_components_)
        self.var = float(self.pca.explained_variance_ratio_.sum())

    def build(self, kappa, feats=None, skel=None):
        """返回 {name: X(float32)}，均按列 z 化。"""
        reps = {}
        reps['SBPD'] = zscore(
            self.pca.transform(kappa).astype(np.float32))
        reps['RAW'] = zscore(kappa)
        F = np.abs(np.fft.rfft(kappa, axis=1))[:, :11].astype(np.float32)
        reps['FOURIER'] = zscore(np.log1p(F))
        if feats is not None:
            A = feats
            if skel is not None:
                A = np.column_stack([A, skel])
            reps['FEAT'] = zscore(A.astype(np.float32))
        return reps

    def describe(self):
        s = f'SBPD: PCA {self.n_comp} 维（累计方差 {self.var:.3f}）; '
        s += 'RAW: 100 维原值; FOURIER: 11 阶幅值（log1p）; '
        s += 'FEAT: 8 维可解释特征'
        return s


# ============================================================
# 2. 算法
# ============================================================

def fit_kmeans(X, k, seed=0):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def fit_gmm(X, k, seed=0):
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(n_components=k, covariance_type='diag',
                        reg_covar=1e-4, random_state=seed, max_iter=200, n_init=1)
    g.fit(X)
    return g.predict(X), float(g.bic(X))


def fit_ward(X, k):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)


def fit_spectral(X, k, seed=0):
    from sklearn.cluster import SpectralClustering
    return SpectralClustering(n_clusters=k, affinity='nearest_neighbors',
                              n_neighbors=10, assign_labels='kmeans',
                              random_state=seed).fit_predict(X)


def assign_by_centroid(X, labels, k):
    """按簇质心重新指派（Ward/Spectral 无 predict 时用，保证同一比较口径）。"""
    cents = np.zeros((k, X.shape[1]), dtype=np.float64)
    for c in range(k):
        m = labels == c
        cents[c] = X[m].mean(axis=0) if m.any() else X.mean(axis=0)
    d2 = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def dbscan_sweep(X, min_samples=20, pcts=(50, 70, 80, 90, 95, 98)):
    """DBSCAN 密度可分性扫描。密度法不指定 k，是检验"存在密度分离"的关键。"""
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dist, _ = nn.kneighbors(X)
    kd = dist[:, -1]
    out = []
    n = len(X)
    for p in pcts:
        eps = float(np.percentile(kd, p))
        if eps <= 0:
            continue
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        noise = float((lab == -1).sum()) / n
        ids, counts = np.unique(lab[lab >= 0], return_counts=True)
        out.append(dict(eps=round(eps, 5), n_clusters=int(len(ids)),
                        noise_frac=round(noise, 4),
                        coverage=round(1 - noise, 4),
                        largest_frac=round(float(counts.max() / n), 4)
                        if len(ids) else 0.0))
    return out


# ============================================================
# 3. 判据
# ============================================================

def internal_validity(X, labels):
    """轮廓 / CH / DB（传入的 X/labels 已是要评估的子集）。"""
    from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                                 davies_bouldin_score)
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return dict(sil=float('nan'), ch=float('nan'), db=float('nan'))
    out = {}
    try:
        out['sil'] = round(float(silhouette_score(X, labels)), 4)
    except Exception:
        out['sil'] = float('nan')
    try:
        out['ch'] = round(float(calinski_harabasz_score(X, labels)), 2)
        out['db'] = round(float(davies_bouldin_score(X, labels)), 4)
    except Exception:
        out['ch'] = out['db'] = float('nan')
    return out


def bootstrap_ari(X, labels, k, n_boot=50, frac=0.8, seed=0):
    """自助重采样稳定性：拟合样本扰动后，固定评估集上的归属一致性（ARI）。"""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.RandomState(seed)
    n = len(labels)
    ref = assign_by_centroid(X, labels, k)
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(n, int(n * frac), replace=False)
        try:
            lab = fit_kmeans(X[idx], k, seed=seed)
        except Exception:
            continue
        aris.append(float(adjusted_rand_score(ref, assign_by_centroid(X, lab, k))))
    if not aris:
        return float('nan'), float('nan')
    return round(float(np.mean(aris)), 4), round(float(np.std(aris)), 4)


def modality_analysis(x1d, seed=0):
    """主轴模态：1-D GMM-BIC（1..4 分量）+ KDE 峰/谷。"""
    from sklearn.mixture import GaussianMixture
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde
    x = np.asarray(x1d, dtype=np.float64).reshape(-1, 1)
    bics = {}
    for k in (1, 2, 3, 4):
        g = GaussianMixture(n_components=k, covariance_type='full',
                            random_state=seed, max_iter=200, n_init=2)
        g.fit(x)
        bics[k] = float(g.bic(x))
    gmm_best = min(bics, key=bics.get)
    n_peaks, valley = 0, 0.0
    try:
        kde = gaussian_kde(x[:, 0])
        grid = np.linspace(x.min(), x.max(), 512)
        dens = kde(grid)
        peaks, _ = find_peaks(dens, prominence=0.05 * dens.max())
        valleys, _ = find_peaks(-dens, prominence=0.02 * dens.max())
        n_peaks = int(len(peaks))
        if n_peaks >= 2 and len(valleys):
            top = peaks[np.argsort(dens[peaks])[-2:]]
            lo, hi = int(min(top)), int(max(top))
            v = valleys[(valleys > lo) & (valleys < hi)]
            if len(v):
                vv = v[np.argmin(dens[v])]
                valley = float(1.0 - dens[vv] / min(dens[top]))
    except Exception:
        pass
    return dict(gmm_best=gmm_best, n_peaks=n_peaks,
                valley_depth=round(valley, 4),
                **{f'bic{k}': round(v, 1) for k, v in bics.items()})


def ward_tree(X, k_max=10):
    """Ward 合并高度与相邻比（无断层 = 无自然簇数）。"""
    from scipy.cluster.hierarchy import linkage
    Z = linkage(X, method='ward')
    h = Z[:, 2]
    n = len(h)
    rows = []
    for k in range(2, k_max + 1):
        i = n - k + 1
        hk = float(h[i])
        hk1 = float(h[i + 1]) if i + 1 < n else float('nan')
        rows.append(dict(k=k, merge_height=round(hk, 4), next_height=round(hk1, 4),
                         ratio=round(hk / hk1, 4) if hk1 and hk1 > 0 else None))
    return rows


# ============================================================
# 4. 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser(description='姿态连续性：多表示×多算法×多判据交叉验证')
    ap.add_argument('-c', '--cache', default='atlas_out_full/kappa_cache.npz')
    ap.add_argument('-o', '--outdir', default='atlas_out_full/continuity')
    ap.add_argument('--k-min', type=int, default=2)
    ap.add_argument('--k-max', type=int, default=10)
    ap.add_argument('--n-main', type=int, default=200000)
    ap.add_argument('--n-metric', type=int, default=20000)
    ap.add_argument('--n-boot', type=int, default=20000)
    ap.add_argument('--n-db', type=int, default=50000)
    ap.add_argument('--n-tree', type=int, default=5000)
    ap.add_argument('--n-agree', type=int, default=5000)
    ap.add_argument('--n-boot-rep', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--quick', action='store_true', help='十等分规模冒烟')
    args = ap.parse_args()

    if args.quick:
        for a in ('n_main', 'n_metric', 'n_boot', 'n_db', 'n_tree', 'n_agree'):
            setattr(args, a, max(500, getattr(args, a) // 10))
        args.n_boot_rep = 10

    os.makedirs(args.outdir, exist_ok=True)
    d = load_cache(args.cache)
    kappa_all, feats_all, skel_all = d['kappa'], d['features'], d['skel_len']
    N = len(kappa_all)

    def block(n_take, seed):
        if n_take >= N:
            return np.arange(N)
        idx = stratified_idx(d['groups'], n_take, seed)
        if idx is None:
            idx = np.random.RandomState(seed).choice(N, n_take, replace=False)
        return idx

    I_MAIN = block(args.n_main, args.seed)
    I_BOOT = block(args.n_boot, args.seed + 1)
    I_DB = block(args.n_db, args.seed + 2)
    I_TREE = block(args.n_tree, args.seed + 3)
    I_AGR = block(args.n_agree, args.seed + 4)

    rep = Representations(kappa_all[I_MAIN])
    print('[表示] ' + rep.describe())
    print(f'[规模] 主 {len(I_MAIN):,} | 自助 {len(I_BOOT):,} | DBSCAN {len(I_DB):,} '
          f'| Ward/KDE {len(I_AGR):,} | 一致性 {len(I_AGR):,}')

    # 各块的特征切片（FEAT 表示需要）
    def feats_of(idx):
        return None if feats_all is None else feats_all[idx]

    def skel_of(idx):
        return None if skel_all is None else skel_all[idx]

    R_MAIN = rep.build(kappa_all[I_MAIN], feats_of(I_MAIN), skel_of(I_MAIN))
    R_BOOT = rep.build(kappa_all[I_BOOT], feats_of(I_BOOT), skel_of(I_BOOT))
    R_DB = rep.build(kappa_all[I_DB], feats_of(I_DB), skel_of(I_DB))
    R_TREE = rep.build(kappa_all[I_TREE], feats_of(I_TREE), skel_of(I_TREE))
    R_AGR = rep.build(kappa_all[I_AGR], feats_of(I_AGR), skel_of(I_AGR))
    rep_names = [r for r in ['SBPD', 'RAW', 'FOURIER', 'FEAT'] if r in R_MAIN]

    # ---- (1) 内部效度矩阵：表示 × {kmeans, ward, gmm} × k ----
    # 规模分配（避免 O(n^2) 算法炸内存）：
    #   kmeans 在主块（20 万）拟合，在子样本上评估；GMM 在子样本（2 万）拟合并评估；
    #   Ward 在 5 千块上拟合并评估（sklearn Ward 需要 O(n^2) 距离矩阵）。
    print('\n=== (1) 内部效度矩阵 ===')
    val_rows = []
    for name in rep_names:
        Xm = R_MAIN[name]
        n_met = min(args.n_metric, len(I_MAIN))
        met_pos = np.random.RandomState(args.seed).choice(
            len(Xm), n_met, replace=False)
        Xmet = Xm[met_pos]
        Xtree = R_TREE[name]
        for algo in ('kmeans', 'ward'):
            for k in range(args.k_min, args.k_max + 1):
                try:
                    if algo == 'kmeans':
                        lab = fit_kmeans(Xm, k, args.seed)
                        mv = internal_validity(Xmet, lab[met_pos])
                    else:
                        lab = fit_ward(Xtree, k)
                        mv = internal_validity(Xtree, lab)
                except Exception as e:
                    print(f'  [跳过] {name}/{algo}/k={k}: {e}')
                    continue
                val_rows.append(dict(rep=name, algo=algo, k=k, bic='', **mv))
                print(f'  {name:8s} {algo:7s} k={k:2d}  sil={mv["sil"]} '
                      f'CH={mv["ch"]}')
        for k in range(args.k_min, args.k_max + 1):
            try:
                lab, bic = fit_gmm(Xmet, k, args.seed)
                mv = internal_validity(Xmet, lab)
            except Exception as e:
                print(f'  [跳过] {name}/gmm/k={k}: {e}')
                continue
            val_rows.append(dict(rep=name, algo='gmm', k=k, bic=round(bic, 1), **mv))
            print(f'  {name:8s} gmm     k={k:2d}  sil={mv["sil"]} BIC={bic:.1f}')

    # ---- (2) 自助稳定性（k=2；Ward 用 5 千块避免 O(n^2) 内存）----
    print('\n=== (2) 自助重采样稳定性（k=2）===')
    stab_rows = []
    for name in rep_names:
        Xb = R_BOOT[name]
        Xtree = R_TREE[name]
        for algo in ('kmeans', 'ward', 'gmm'):
            try:
                if algo == 'kmeans':
                    lab = fit_kmeans(Xb, 2, args.seed)
                    m, s = bootstrap_ari(Xb, lab, 2, n_boot=args.n_boot_rep,
                                         seed=args.seed)
                elif algo == 'ward':
                    lab = fit_ward(Xtree, 2)
                    m, s = bootstrap_ari(Xtree, lab, 2,
                                         n_boot=args.n_boot_rep, seed=args.seed)
                else:
                    lab = fit_gmm(Xb, 2, args.seed)[0]
                    m, s = bootstrap_ari(Xb, lab, 2, n_boot=args.n_boot_rep,
                                         seed=args.seed)
            except Exception as e:
                print(f'  [跳过] {name}/{algo}: {e}')
                continue
            stab_rows.append(dict(rep=name, algo=algo, k=2,
                                  ari_mean=m, ari_sd=s))
            print(f'  {name:8s} {algo:7s} ARI={m} ± {s}')

    # ---- (3) 主轴模态 ----
    print('\n=== (3) 主轴模态（GMM-BIC + KDE 峰谷）===')
    mod_rows = []
    from sklearn.decomposition import PCA
    for name in rep_names:
        Xt = R_TREE[name]
        pc1 = PCA(n_components=1, random_state=args.seed).fit_transform(Xt)[:, 0]
        md = modality_analysis(pc1, seed=args.seed)
        mod_rows.append(dict(rep=name, **md))
        print(f'  {name:8s} GMM-BIC 最优 {md["gmm_best"]} 分量 | '
              f'峰数 {md["n_peaks"]} | 谷深 {md["valley_depth"]}')

    # ---- (4) DBSCAN 密度可分性扫描 ----
    print('\n=== (4) DBSCAN 密度可分性扫描 ===')
    db_rows = []
    for name in rep_names:
        for r in dbscan_sweep(R_DB[name], min_samples=20):
            db_rows.append(dict(rep=name, **r))
        sep = [r for r in db_rows if r['rep'] == name
               and r['n_clusters'] >= 2 and r['coverage'] >= 0.80
               and r['largest_frac'] <= 0.90]
        best = max((r['n_clusters'] for r in db_rows if r['rep'] == name), default=0)
        print(f'  {name:8s} 最大簇数 {best}；'
              f'{"发现密度可分划分" if sep else "无密度可分划分"}')

    # ---- (5) Ward 合并高度 ----
    print('\n=== (5) Ward 合并高度断层 ===')
    tree_rows = []
    for name in rep_names:
        for r in ward_tree(R_TREE[name], k_max=args.k_max):
            tree_rows.append(dict(rep=name, **r))
        ratios = [r['ratio'] for r in tree_rows
                  if r['rep'] == name and r['ratio'] is not None]
        print(f'  {name:8s} 相邻高度比 k=2..{args.k_max}: '
              f'max={max(ratios) if ratios else float("nan"):.3f}')

    # ---- (6) 跨表示 / 跨算法一致性（k=2，同一样本块）----
    print('\n=== (6) 跨表示/跨算法一致性（k=2）===')
    from sklearn.metrics import adjusted_rand_score
    labs_rep, labs_algo = {}, {}
    for name in rep_names:
        Xa = R_AGR[name]
        labs_rep[name] = fit_kmeans(Xa, 2, args.seed)
        for algo in ('kmeans', 'gmm', 'ward', 'spectral'):
            try:
                if algo == 'kmeans':
                    lab = labs_rep[name]
                elif algo == 'gmm':
                    lab = fit_gmm(Xa, 2, args.seed)[0]
                elif algo == 'ward':
                    lab = fit_ward(Xa, 2)
                else:
                    lab = fit_spectral(Xa, 2, args.seed)
                labs_algo[(name, algo)] = lab
            except Exception as e:
                print(f'  [跳过] {name}/{algo}: {e}')
    agree_rows = []
    rn = rep_names
    for i, a in enumerate(rn):
        for b in rn[i + 1:]:
            if a in labs_rep and b in labs_rep:
                ari = round(float(adjusted_rand_score(labs_rep[a], labs_rep[b])), 4)
                agree_rows.append(dict(type='rep_rep', a=a, b=b, k=2, ari=ari))
                print(f'  表示 {a} × {b}: ARI={ari}')
    algo_keys = sorted(labs_algo.keys())
    for i, ka in enumerate(algo_keys):
        for kb in algo_keys[i + 1:]:
            if ka[0] != kb[0]:
                continue
            ari = round(float(adjusted_rand_score(labs_algo[ka], labs_algo[kb])), 4)
            agree_rows.append(dict(type='algo_algo', a=ka[1], b=kb[1],
                                   rep=ka[0], k=2, ari=ari))
            print(f'  算法 {ka[1]} × {kb[1]}（{ka[0]}）: ARI={ari}')

    # ---- 写 CSV ----
    print('\n=== 写 CSV ===')
    write_csv(os.path.join(args.outdir, 'continuity_validity.csv'), val_rows,
              ['rep', 'algo', 'k', 'bic', 'sil', 'ch', 'db'])
    write_csv(os.path.join(args.outdir, 'continuity_stability.csv'), stab_rows,
              ['rep', 'algo', 'k', 'ari_mean', 'ari_sd'])
    write_csv(os.path.join(args.outdir, 'continuity_modality.csv'), mod_rows)
    write_csv(os.path.join(args.outdir, 'continuity_dbscan.csv'), db_rows,
              ['rep', 'eps', 'n_clusters', 'noise_frac', 'coverage', 'largest_frac'])
    write_csv(os.path.join(args.outdir, 'continuity_agreement.csv'), agree_rows,
              ['type', 'a', 'b', 'rep', 'k', 'ari'])
    write_csv(os.path.join(args.outdir, 'continuity_ward_tree.csv'), tree_rows,
              ['rep', 'k', 'merge_height', 'next_height', 'ratio'])

    # ---- 判定 ----
    verdict = decide(val_rows, mod_rows, db_rows, tree_rows, stab_rows, rep_names)
    vpath = os.path.join(args.outdir, 'continuity_verdict.txt')
    with open(vpath, 'w', encoding='utf-8') as f:
        f.write(verdict)
    print(f'\n{verdict}')
    print(f'判定已写：{vpath}')

    make_figure(args.outdir, val_rows, mod_rows, db_rows, agree_rows, rep_names)
    print('\n全部完成。')


# ============================================================
# 5. 预注册判定与出图
# ============================================================

def _rep_key(row):
    """兼容 'rep'（step14 表示）与 'pipe'（step15 管线）两种键名。"""
    return row.get('rep') if 'rep' in row else row.get('pipe')


def decide(val_rows, mod_rows, db_rows, tree_rows, stab_rows, rep_names):
    """按预注册条件判定，返回文本报告。"""
    sils = [r['sil'] for r in val_rows if r['sil'] == r['sil']]
    max_sil = max(sils) if sils else float('nan')
    valleys = [r['valley_depth'] for r in mod_rows]
    max_valley = max(valleys) if valleys else 0.0
    max_peaks = max((r['n_peaks'] for r in mod_rows), default=0)
    ratios = [r['ratio'] for r in tree_rows if r['ratio'] is not None]
    max_ratio = max(ratios) if ratios else float('nan')
    aris = [r['ari_mean'] for r in stab_rows if r['ari_mean'] == r['ari_mean']]
    min_ari = min(aris) if aris else float('nan')

    sep_rows = [r for r in db_rows if r['n_clusters'] >= 2
                and r['coverage'] >= 0.80 and r['largest_frac'] <= 0.90]
    # 否证条件 F1–F3：任一 (表示/管线, 算法) 同时满足
    fals = []
    for r in val_rows:
        if r['k'] < 2 or r['sil'] != r['sil']:
            continue
        md = next((m for m in mod_rows if _rep_key(m) == _rep_key(r)), None)
        if (r['sil'] > 0.50 and md and md['valley_depth'] > 0.25
                and md['n_peaks'] >= 2):
            fals.append(r)
            break
    S = dict(
        S1=max_sil < 0.35,
        S2=max_valley < 0.15 or max_peaks < 2,
        S3=len(sep_rows) == 0,
        S4=(max_ratio == max_ratio) and max_ratio < 2.0,
    )
    lines = []
    lines.append('姿态连续性交叉验证判定报告')
    lines.append('=' * 56)
    lines.append(f'参与判定：{len(rep_names)} 种表示 × 3 类算法 × '
                 f'k={min((r["k"] for r in val_rows), default=2)}..'
                 f'{max((r["k"] for r in val_rows), default=2)}')
    lines.append('')
    lines.append(f'S1 最大轮廓系数 < 0.35       : {"成立" if S["S1"] else "不成立"}'
                 f'   (max={max_sil:.4f})')
    lines.append(f'S2 最大谷深 < 0.15 或峰数<2   : {"成立" if S["S2"] else "不成立"}'
                 f'   (max_valley={max_valley:.4f}, max_peaks={max_peaks})')
    lines.append(f'S3 无密度可分划分             : {"成立" if S["S3"] else "不成立"}'
                 f'   (满足条件的 eps 扫描行数={len(sep_rows)})')
    lines.append(f'S4 无 Ward 合并断层(比<2.0)   : {"成立" if S["S4"] else "不成立"}'
                 f'   (max_ratio={max_ratio})')
    lines.append(f'S5 参考：自助稳定性 ARI min={min_ari:.4f}'
                 '（仅在 S1–S4 成立时方可解释为"同一把刀切连续体"）')
    lines.append('')
    lines.append(f'F1–F3 否证条件（强轮廓+显著谷+高稳定）: '
                 f'{"触发" if fals else "未触发"}'
                 + (f'（首例：{fals[0]["rep"]}/{fals[0]["algo"]}/k={fals[0]["k"]}）'
                    if fals else ''))
    lines.append('')
    if fals:
        verdict = '结论：存在满足预注册否证条件的解，连续假设被推翻，应报告离散结构。'
    elif all(S.values()):
        verdict = ('结论：支持"单轴连续梯度"——所有表示与算法均落在弱结构带、'
                   '无密度谷、无密度可分划分、无合并断层；唯一复现的是单一主轴'
                   '（弯曲深度），k=2 切分应解释为该轴上的任意切分而非两类。')
    else:
        verdict = ('结论：证据不完整（部分条件未满足），应如实报告为'
                   '"弱结构，无法确定唯一类别数"，不得单向断言连续或离散。')
    lines.append(verdict)
    return '\n'.join(lines)


def make_figure(outdir, val_rows, mod_rows, db_rows, agree_rows, rep_names):
    """四联图：轮廓谱 / 主轴密度 / DBSCAN eps 扫描 / 表示间 ARI 热图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 9, 'figure.dpi': 300,
    })
    COLORS = {'SBPD': '#0072B2', 'RAW': '#D55E00', 'FOURIER': '#009E73',
              'FEAT': '#CC79A7'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # A: 轮廓谱（kmeans，各表示）
    ax = axes[0, 0]
    for name in rep_names:
        rows = sorted([r for r in val_rows
                       if r['rep'] == name and r['algo'] == 'kmeans'],
                      key=lambda r: r['k'])
        if rows:
            ax.plot([r['k'] for r in rows], [r['sil'] for r in rows],
                    'o-', ms=3.5, lw=1.4, color=COLORS[name], label=name)
    ax.axhline(0.35, ls=':', color='#888', lw=1)
    ax.text(ax.get_xlim()[1], 0.35, ' weak-structure ceiling', va='center',
            fontsize=8, color='#666')
    ax.set_xlabel('k')
    ax.set_ylabel('Silhouette width')
    ax.set_title('A. Internal validity ceiling across representations')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # B: 主轴谷深
    ax = axes[0, 1]
    if mod_rows:
        names = [r['rep'] for r in mod_rows]
        vd = [r['valley_depth'] for r in mod_rows]
        npk = [r['n_peaks'] for r in mod_rows]
        xs = np.arange(len(names))
        ax.bar(xs, vd, color=[COLORS.get(n, '#888') for n in names],
               width=0.55, alpha=0.9)
        for i, (v, p) in enumerate(zip(vd, npk)):
            ax.text(i, v, f'{v:.3f}\n({p} peaks)', ha='center', va='bottom',
                    fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(names)
        ax.axhline(0.25, ls=':', color='#888', lw=1)
        ax.set_ylabel('KDE valley depth')
        ax.set_title('B. Valley depth along principal axis')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # C: DBSCAN eps 扫描（每表示的簇数）
    ax = axes[1, 0]
    for name in rep_names:
        rows = [r for r in db_rows if r['rep'] == name]
        if rows:
            ax.plot([r['eps'] for r in rows], [r['n_clusters'] for r in rows],
                    'o-', ms=3.5, lw=1.4, color=COLORS[name], label=name)
    ax.set_xlabel('eps (k-distance percentile)')
    ax.set_ylabel('Number of density clusters')
    ax.set_title('C. Density separability (DBSCAN sweep)')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # D: 表示间 ARI 热图
    ax = axes[1, 1]
    n = len(rep_names)
    M = np.eye(n)
    for r in agree_rows:
        if r.get('type') == 'rep_rep' and r['a'] in rep_names and r['b'] in rep_names:
            i, j = rep_names.index(r['a']), rep_names.index(r['b'])
            M[i, j] = M[j, i] = r['ari']
    im = ax.imshow(M, vmin=0, vmax=1, cmap='Blues')
    ax.set_xticks(range(n))
    ax.set_xticklabels(rep_names, rotation=30, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(rep_names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                    fontsize=9,
                    color='white' if M[i, j] > 0.6 else '#333')
    ax.set_title('D. Cross-representation ARI at k = 2')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Is posture discrete or a single continuous gradient?',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(outdir, 'fig8_continuity.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'图已保存：{out}')


if __name__ == '__main__':
    main()
