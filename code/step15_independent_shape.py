"""step15_independent_shape.py — 用互相独立的测量管线重算姿态形状，
检验"姿态连续（无离散类别）"是否只是 κ(s) 这一条测量管线的产物。

动机与设计
----------
cluster_atlas.py 的"连续谱"结论完全建立在 κ(s) 之上。若 κ(s) 管线本身有畸变
（细化噪声、弧长重采样、毛刺剪枝），结论会被继承。因此本脚本不复用 κ(s)，
而是从原始图像出发，用四条**测量管线互相独立**的描述子重新刻画同一批帧：

  R1 KAPPA     中心线 → |Δθ| 关于归一化弧长的分布（cluster_atlas 口径，基准）
  R2 ANGLE     中心线 → 有符号切向角 θ(s) 的累计序列（保留 S 形/C 形手性模式；
               KAPPA 取绝对值会丢掉该模式，两者是同一输入的不同数学量）
  R3 OUTLINE   体壁掩膜外轮廓 → 傅里叶描述子幅值（完全不经中心线/细化；
               幅值对轮廓起点平移与镜像反射均不变）
  R4 MASKGRID  体壁掩膜像素 → 按主轴对齐并镜像规范化后栅格化（完全不经中心线）

独立性层级（如实说明，不过度声称）：
  · R3/R4 与 R1/R2 完全独立（不经中心线提取与细化）；
  · R1/R2 共享中心线，但计算量不同（绝对值弧长分布 vs 有符号累计角）；
  · 因此：R3/R4 与 R1/R2 结论一致 ⇒ 结论不依赖中心线环节；四者一致 ⇒
    结论对测量管线稳健。

小样本设计的依据
----------------
结构性问题（有无离散类界）不需要大样本：簇的"有无"由分布形状决定，
数千至数万帧已足以稳定估计轮廓与谷深。本脚本默认每 (组, 日期, 行) 抽
5 只虫、每只 20 帧，即 4 组 × 21 日期 × 4 行 ≈ **3.4 万帧**
（提取约 8 分钟，全流程约 15–20 分钟）。需要精确位置参数（主轴标定、
信度）时仍用 cluster_atlas 的全量缓存，两者分工明确。

聚类算法覆盖四类归纳偏置：k-means（质心/球状）、GMM（概率混合，BIC 选 k）、
Ward 层次（合并高度断层）、DBSCAN（密度连接，不指定 k，直接检验密度可分性）。
算法层的进一步扩展（含谱聚类与 κ(s) 的四种编码）见 step14。

判据与预注册条件（与 step14 相同，避免"选着报"）
  F1 轮廓 > 0.50；F2 谷深 > 0.25 且峰数 ≥ 2；F3 自助 ARI > 0.75
     → 同时满足即支持"离散类别"，连续假设被推翻
  S1 最大轮廓 < 0.35；S2 最大谷深 < 0.15 或峰数 < 2；
     S3 无密度可分划分；S4 无 Ward 合并断层（比 < 2.0）
     → 全部成立才判定"单轴连续梯度"
  S5 跨管线 ARI 高 —— 仅在 S1–S4 成立时解释为"同一把刀切连续体"，
     不得单独引用（连续轴上任何算法都会在同一位置下刀）

用法（ele 环境）
--------------
  # 冒烟（每单元 2 只 × 6 帧，约 1 分钟）
  python step15_independent_shape.py --root $WORM_DATA_ROOT \
      -o atlas_out_full/indep --quick

  # 正式（约 3.4 万帧，15-20 分钟）
  python step15_independent_shape.py --root $WORM_DATA_ROOT \
      -o atlas_out_full/indep --workers 20

  # 复用已提取的表示，只重跑判据（秒级）
  python step15_independent_shape.py --root $WORM_DATA_ROOT \
      -o atlas_out_full/indep --skip-extract

产物
----
  indep_cache.npz              四管线的表示矩阵 + 元数据
  indep_validity.csv           每 (管线, 算法, k) 的轮廓/CH/DB/BIC
  indep_stability.csv          每 (管线, 算法) 的自助 ARI
  indep_modality.csv           每管线的主轴模态（GMM-BIC、峰数、谷深）
  indep_dbscan.csv             每管线的 eps 扫描
  indep_ward_tree.csv          每管线的 Ward 合并高度
  indep_agreement.csv          管线间 k=2 的 ARI 矩阵
  indep_verdict.txt            预注册判定报告
  fig9_independent_pipelines.png 四联图
"""

import os
import csv
import glob
import random
import time
import argparse
import multiprocessing as mp
import numpy as np
import cv2

from worm_labeler import extract_skeleton_full, compute_curvature_profile, parse_five_dims

# 判据与聚类工具复用 step14（保证两处口径一致，避免双份实现漂移）
from step14_continuity_crossval import (
    fit_kmeans, fit_gmm, fit_ward, fit_spectral, assign_by_centroid,
    dbscan_sweep, internal_validity, bootstrap_ari, modality_analysis,
    ward_tree, decide, write_csv)

from wormpath import get_data_root

N_SAMPLES = 100      # 弧长采样点数（与 cluster_atlas 一致）
N_HARM = 16          # 轮廓傅里叶谐波数
GRID = 16            # 掩膜栅格边长（→ 256 维）
PIPELINES = ['KAPPA', 'ANGLE', 'OUTLINE', 'MASKGRID']
COLORS = {'KAPPA': '#0072B2', 'ANGLE': '#D55E00',
          'OUTLINE': '#009E73', 'MASKGRID': '#CC79A7'}


# ============================================================
# 1. 抽样：每 (组, 日期, 行) 抽 N 只虫，每只抽 M 帧（按帧号均匀）
# ============================================================

def sample_tasks(root, max_worms, max_frames, seed=42):
    rng = random.Random(seed)
    by_worm = {}
    for path in glob.glob(os.path.join(root, '*', '*', '*', '*.png')):
        if '_noise_quarantine' in path.replace('\\', '/'):
            continue
        rel = os.path.relpath(path, root).replace('\\', '/')
        try:
            g, d, rw, wm, fr = parse_five_dims(rel)
        except Exception:
            continue
        by_worm.setdefault((g, d, rw, wm), []).append((int(fr), rel, path))
    cells = {}
    for k in by_worm:
        cells.setdefault((k[0], k[1], k[2]), []).append(k)
    tasks = []
    for (g, d, rw), worms in sorted(cells.items()):
        worms = sorted(worms)
        rng.shuffle(worms)
        for w in worms[:max_worms]:
            frames = sorted(by_worm[w])
            if len(frames) > max_frames:
                idx = sorted(set(np.linspace(0, len(frames) - 1,
                                             max_frames).astype(int)))
                sel = [frames[i] for i in idx]
            else:
                sel = frames
            tasks += [(rel, p) for _, rel, p in sel]
    print(f'[抽样] {len(tasks):,} 帧'
          f'（每单元 ≤{max_worms} 虫 × ≤{max_frames} 帧）')
    return tasks


# ============================================================
# 2. 四条测量管线的描述子
# ============================================================

def _resample_arclength(pts, n):
    d = np.diff(pts, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-9:
        return None
    s = cum / total
    q = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.interp(q, s, pts[:, 0]),
                            np.interp(q, s, pts[:, 1])])


def _canonical_dir(prof):
    """与 cluster_atlas 同口径：加权质心 >0.5 则反转，统一头尾朝向。"""
    w = np.abs(np.asarray(prof, dtype=np.float64))
    ws = w.sum()
    if ws < 1e-9:
        return prof
    s = np.linspace(0.0, 1.0, len(prof))
    if float((s * w).sum() / ws) > 0.5:
        return prof[::-1].copy()
    return prof


def desc_kappa(pts):
    """κ(s)：|Δθ| 关于归一化弧长的分布；取绝对值后天然镜像不变，
    再按 cluster_atlas 同口径做方向规范化（曲率质心统一在前半段）。"""
    kap = compute_curvature_profile(pts, N_SAMPLES)
    return _canonical_dir(np.abs(np.asarray(kap, dtype=np.float64)))


def desc_angle(pts):
    """有符号切向角累计序列；保留手性模式，按总转角符号做镜像规范化。"""
    P = _resample_arclength(pts, N_SAMPLES)
    if P is None:
        return None
    d = np.diff(P, axis=0)
    th = np.concatenate([[0.0], np.unwrap(np.arctan2(d[:, 1], d[:, 0]))])
    th = th - th.mean()
    if th.sum() < 0:          # 镜像反射会使 θ → −θ，取规范朝向
        th = -th
    return _canonical_dir(th)


def desc_outline(mask):
    """外轮廓傅里叶幅值：不经中心线；幅值对起点平移与镜像不变，尺度已归一。"""
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(c) < 32:
        return None
    z = c[:, 0] + 1j * c[:, 1]
    F = np.fft.fft(z)
    mag = np.abs(F[1:N_HARM + 1])
    return np.log1p(mag / (np.abs(F[1]) + 1e-9))


def desc_maskgrid(mask):
    """掩膜按主轴对齐 + 镜像规范化后栅格化：不经中心线的像素级表示。

    规范化顺序与 κ(s) 口径对齐：先按三阶矩符号消镜像，再把质量质心
    统一到主轴正向半段（对应 κ(s) 的"曲率质心 s_cm < 0.5"约定），
    使头尾颠倒的两帧映射到同一栅格。
    """
    m = (mask > 0)
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        return None
    X = np.column_stack([xs, ys]).astype(np.float64)
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc.T)
    try:
        w, V = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    R = V[:, np.argsort(w)[::-1]]
    P = Xc @ R
    for ax in range(2):                      # 镜像规范化（三阶矩符号）
        if np.mean(P[:, ax] ** 3) < 0:
            P[:, ax] = -P[:, ax]
    if P[:, 0].mean() < 0:                   # 头尾规范化：质心在正向半段
        P[:, 0] = -P[:, 0]
    ext = np.abs(P).max(axis=0) + 1e-6
    Pn = P / ext
    grid = np.zeros((GRID, GRID), dtype=np.float32)
    gi = np.clip(((Pn[:, 0] + 1) / 2 * GRID).astype(int), 0, GRID - 1)
    gj = np.clip(((Pn[:, 1] + 1) / 2 * GRID).astype(int), 0, GRID - 1)
    np.add.at(grid, (gj, gi), 1.0)
    s = grid.sum()
    if s <= 0:
        return None
    return (grid / s).reshape(-1)


def _worker(task):
    rel, abs_path = task
    img = cv2.imread(abs_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    try:
        out = extract_skeleton_full(img)
    except Exception:
        return None
    if not out:
        return None
    pts, _mt, mask_final, _ft = out
    if pts is None or len(pts) < 10 or mask_final is None:
        return None
    try:
        r1 = desc_kappa(pts)
        r2 = desc_angle(pts)
        r3 = desc_outline(mask_final)
        r4 = desc_maskgrid(mask_final)
    except Exception:
        return None
    if any(r is None for r in (r1, r2, r3, r4)):
        return None
    try:
        g, d, rw, wm, fr = parse_five_dims(rel)
    except Exception:
        g, d, rw, wm, fr = 'unknown', 'unknown', 'unknown', rel, 0
    return (rel, r1.astype(np.float32), r2.astype(np.float32),
            r3.astype(np.float32), r4.astype(np.float32), g, d, rw, wm)


# ============================================================
# 3. 提取（多进程 + 缓存）
# ============================================================

def extract(tasks, workers, cache_path, chunksize=50):
    rels, R = [], {p: [] for p in PIPELINES}
    meta = {k: [] for k in ('groups', 'dates', 'rows', 'worms')}
    t0 = time.time()
    done = ok = 0
    with mp.Pool(processes=workers) as pool:
        for res in pool.imap_unordered(_worker, tasks, chunksize=chunksize):
            done += 1
            if res is None:
                continue
            rel, r1, r2, r3, r4, g, d, rw, wm = res
            rels.append(rel)
            R['KAPPA'].append(r1)
            R['ANGLE'].append(r2)
            R['OUTLINE'].append(r3)
            R['MASKGRID'].append(r4)
            for k, v in zip(('groups', 'dates', 'rows', 'worms'),
                            (g, d, rw, wm)):
                meta[k].append(v)
            ok += 1
            if done % 1000 == 0 or done == len(tasks):
                rate = done / max(1e-6, time.time() - t0)
                print(f'  提取 {done:,}/{len(tasks):,}  有效 {ok:,}  '
                      f'{rate:.0f} 帧/秒', flush=True)
    if ok == 0:
        raise SystemExit('0 有效帧：检查输入目录或骨架提取。')
    arrays = {p: np.vstack(R[p]) for p in PIPELINES}
    np.savez(cache_path, filenames=np.asarray(rels),
             groups=np.asarray(meta['groups']),
             dates=np.asarray(meta['dates']),
             rows=np.asarray(meta['rows']),
             worms=np.asarray(meta['worms']),
             **{p: arrays[p] for p in PIPELINES})
    print(f'[缓存] 已写 {cache_path}（{ok:,} 帧）')
    return arrays, meta, np.asarray(rels)


# ============================================================
# 4. 判据矩阵（复用 step14 的口径）
# ============================================================

def analyse(arrays, meta, outdir, args):
    val_rows, stab_rows, mod_rows, db_rows, tree_rows, agree_rows = \
        [], [], [], [], [], []
    for name in PIPELINES:
        X = arrays[name]
        X = ((X - X.mean(0)) / (X.std(0) + 1e-8)).astype(np.float32)
        n = len(X)
        n_met = min(args.n_metric, n)
        pos = np.random.RandomState(args.seed).choice(n, n_met, replace=False)
        Xmet = X[pos]
        n_tree = min(args.n_tree, n)
        pos_t = np.random.RandomState(args.seed + 1).choice(n, n_tree,
                                                            replace=False)
        Xtree = X[pos_t]
        print(f'\n=== 管线 {name}（{X.shape[1]} 维, n={n:,}）===')

        for algo in ('kmeans', 'gmm', 'ward'):
            for k in range(2, args.k_max + 1):
                try:
                    if algo == 'kmeans':
                        lab = fit_kmeans(X, k, args.seed)
                        mv = internal_validity(Xmet, lab[pos])
                    elif algo == 'gmm':
                        lab, bic = fit_gmm(Xmet, k, args.seed)
                        mv = internal_validity(Xmet, lab)
                        val_rows.append(dict(pipe=name, algo=algo, k=k,
                                             bic=round(bic, 1), **mv))
                        print(f'  {algo:7s} k={k}  sil={mv["sil"]}')
                        continue
                    else:
                        lab = fit_ward(Xtree, k)
                        mv = internal_validity(Xtree, lab)
                except Exception as e:
                    print(f'  [跳过] {name}/{algo}/k={k}: {e}')
                    continue
                val_rows.append(dict(pipe=name, algo=algo, k=k, bic='', **mv))
                print(f'  {algo:7s} k={k}  sil={mv["sil"]}')

        for algo in ('kmeans', 'gmm', 'ward'):
            try:
                if algo == 'kmeans':
                    lab = fit_kmeans(Xmet, 2, args.seed)
                    m, s = bootstrap_ari(Xmet, lab, 2,
                                         n_boot=args.n_boot_rep, seed=args.seed)
                elif algo == 'gmm':
                    lab = fit_gmm(Xmet, 2, args.seed)[0]
                    m, s = bootstrap_ari(Xmet, lab, 2,
                                         n_boot=args.n_boot_rep, seed=args.seed)
                else:
                    lab = fit_ward(Xtree, 2)
                    m, s = bootstrap_ari(Xtree, lab, 2,
                                         n_boot=args.n_boot_rep, seed=args.seed)
            except Exception as e:
                print(f'  [跳过稳定性] {name}/{algo}: {e}')
                continue
            stab_rows.append(dict(pipe=name, algo=algo, k=2,
                                  ari_mean=m, ari_sd=s))
            print(f'  稳定性 {algo:7s} ARI={m}')

        # 主轴模态
        from sklearn.decomposition import PCA
        pc1 = PCA(n_components=1, random_state=args.seed).fit_transform(Xtree)[:, 0]
        md = modality_analysis(pc1, seed=args.seed)
        mod_rows.append(dict(pipe=name, **md))
        print(f'  模态: GMM-BIC {md["gmm_best"]} 分量, 峰数 {md["n_peaks"]}, '
              f'谷深 {md["valley_depth"]}')

        # DBSCAN
        n_db = min(args.n_db, n)
        Xdb = X[np.random.RandomState(args.seed + 2).choice(
            n, n_db, replace=False)]
        for r in dbscan_sweep(Xdb, min_samples=20):
            db_rows.append(dict(pipe=name, **r))
        sep = [r for r in db_rows if r['pipe'] == name
               and r['n_clusters'] >= 2 and r['coverage'] >= 0.80
               and r['largest_frac'] <= 0.90]
        print(f'  DBSCAN: {"发现密度可分划分" if sep else "无密度可分划分"}')

        for r in ward_tree(Xtree, k_max=args.k_max):
            tree_rows.append(dict(pipe=name, **r))

    # 跨管线一致性（k=2，同一样本块）
    from sklearn.metrics import adjusted_rand_score
    n_ag = min(args.n_agree, len(arrays['KAPPA']))
    pos_a = np.random.RandomState(args.seed + 3).choice(
        len(arrays['KAPPA']), n_ag, replace=False)
    labs = {}
    for name in PIPELINES:
        X = ((arrays[name] - arrays[name].mean(0)) /
             (arrays[name].std(0) + 1e-8)).astype(np.float32)
        try:
            labs[name] = fit_kmeans(X[pos_a], 2, args.seed)
        except Exception as e:
            print(f'  [跳过一致性] {name}: {e}')
    for i, a in enumerate(PIPELINES):
        for b in PIPELINES[i + 1:]:
            if a in labs and b in labs:
                ari = round(float(adjusted_rand_score(labs[a], labs[b])), 4)
                agree_rows.append(dict(type='pipe_pipe', a=a, b=b, k=2, ari=ari))
                print(f'  跨管线 {a} × {b}: ARI={ari}')

    print('\n=== 写 CSV ===')
    write_csv(os.path.join(outdir, 'indep_validity.csv'), val_rows,
              ['pipe', 'algo', 'k', 'bic', 'sil', 'ch', 'db'])
    write_csv(os.path.join(outdir, 'indep_stability.csv'), stab_rows,
              ['pipe', 'algo', 'k', 'ari_mean', 'ari_sd'])
    write_csv(os.path.join(outdir, 'indep_modality.csv'), mod_rows)
    write_csv(os.path.join(outdir, 'indep_dbscan.csv'), db_rows,
              ['pipe', 'eps', 'n_clusters', 'noise_frac', 'coverage',
               'largest_frac'])
    write_csv(os.path.join(outdir, 'indep_ward_tree.csv'), tree_rows,
              ['pipe', 'k', 'merge_height', 'next_height', 'ratio'])
    write_csv(os.path.join(outdir, 'indep_agreement.csv'), agree_rows)

    verdict = decide(val_rows, mod_rows, db_rows, tree_rows, stab_rows,
                     PIPELINES)
    with open(os.path.join(outdir, 'indep_verdict.txt'), 'w',
              encoding='utf-8') as f:
        f.write(verdict + '\n\n注：KAPPA/ANGLE 共享中心线；'
                     'OUTLINE/MASKGRID 不经中心线，与前者独立。\n'
                     'S5（跨管线 ARI）仅在 S1–S4 成立时方可解释为'
                     '"同一把刀切连续体"。\n')
    print(f'\n{verdict}')
    make_figure(outdir, val_rows, mod_rows, db_rows, agree_rows)


def make_figure(outdir, val_rows, mod_rows, db_rows, agree_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 11,
                         'figure.dpi': 300})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0, 0]
    for name in PIPELINES:
        rows = sorted([r for r in val_rows if r['pipe'] == name
                       and r['algo'] == 'kmeans'], key=lambda r: r['k'])
        if rows:
            ax.plot([r['k'] for r in rows], [r['sil'] for r in rows], 'o-',
                    ms=3.5, lw=1.4, color=COLORS[name], label=name)
    ax.axhline(0.35, ls=':', color='#888', lw=1)
    ax.set_xlabel('k')
    ax.set_ylabel('Silhouette width')
    ax.set_title('A. Internal validity across independent pipelines')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[0, 1]
    if mod_rows:
        xs = np.arange(len(mod_rows))
        vd = [r['valley_depth'] for r in mod_rows]
        ax.bar(xs, vd, color=[COLORS[r['pipe']] for r in mod_rows],
               width=0.55, alpha=0.9)
        for i, r in enumerate(mod_rows):
            ax.text(i, r['valley_depth'],
                    f"{r['valley_depth']:.3f}\n({r['n_peaks']} peaks)",
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([r['pipe'] for r in mod_rows], rotation=15)
        ax.axhline(0.25, ls=':', color='#888', lw=1)
        ax.set_ylabel('KDE valley depth')
        ax.set_title('B. Valley depth along principal axis')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax = axes[1, 0]
    for name in PIPELINES:
        rows = [r for r in db_rows if r['pipe'] == name]
        if rows:
            ax.plot([r['eps'] for r in rows], [r['n_clusters'] for r in rows],
                    'o-', ms=3.5, lw=1.4, color=COLORS[name], label=name)
    ax.set_xlabel('eps (k-distance percentile)')
    ax.set_ylabel('Density clusters')
    ax.set_title('C. Density separability (DBSCAN)')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1, 1]
    n = len(PIPELINES)
    M = np.eye(n)
    for r in agree_rows:
        if r['a'] in PIPELINES and r['b'] in PIPELINES:
            i, j = PIPELINES.index(r['a']), PIPELINES.index(r['b'])
            M[i, j] = M[j, i] = r['ari']
    im = ax.imshow(M, vmin=0, vmax=1, cmap='Blues')
    ax.set_xticks(range(n))
    ax.set_xticklabels(PIPELINES, rotation=20, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(PIPELINES)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                    fontsize=9, color='white' if M[i, j] > 0.6 else '#333')
    ax.set_title('D. Cross-pipeline ARI at k = 2')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Posture structure across independent measurement pipelines',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(outdir, 'fig9_independent_pipelines.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'图已保存：{out}')


def main():
    ap = argparse.ArgumentParser(
        description='独立测量管线交叉验证：姿态是否连续')
    ap.add_argument('--root', default=get_data_root())
    ap.add_argument('-o', '--outdir', default='atlas_out_full/indep')
    ap.add_argument('--max-worms', type=int, default=5,
                    help='每 (组,日期,行) 抽取的虫数')
    ap.add_argument('--max-frames', type=int, default=20,
                    help='每只虫抽取的帧数（按帧号均匀）')
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--k-max', type=int, default=8)
    ap.add_argument('--n-metric', type=int, default=3000)
    ap.add_argument('--n-tree', type=int, default=3000)
    ap.add_argument('--n-db', type=int, default=5000)
    ap.add_argument('--n-agree', type=int, default=3000)
    ap.add_argument('--n-boot-rep', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-extract', action='store_true')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    if args.quick:
        args.max_worms = min(args.max_worms, 2)
        args.max_frames = min(args.max_frames, 6)
        args.n_metric = args.n_tree = args.n_db = args.n_agree = 500
        args.n_boot_rep = 10

    os.makedirs(args.outdir, exist_ok=True)
    cache = os.path.join(args.outdir, 'indep_cache.npz')

    if args.skip_extract and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        arrays = {p: z[p] for p in PIPELINES}
        meta = {k: z[k] for k in ('groups', 'dates', 'rows', 'worms')}
        print(f'[缓存] 载入 {len(z["KAPPA"]):,} 帧')
    else:
        tasks = sample_tasks(args.root, args.max_worms, args.max_frames,
                             args.seed)
        if not tasks:
            raise SystemExit('未抽到帧：检查 --root 与目录结构。')
        arrays, meta, _ = extract(tasks, args.workers, cache)

    analyse(arrays, meta, args.outdir, args)
    print('\n全部完成。')


if __name__ == '__main__':
    main()
