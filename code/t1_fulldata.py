#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""t1_fulldata.py — 全量（405 万帧）Table 1 重算：四种特征表示对比

背景：
  原 Table 1 在 8000 帧分层采样上用 RBF-SVM(GridSearchCV) 训练。
  审稿要求全量口径，且 405 万帧上 RBF 网格不可行，故：
  · 标签：classify_by_rules 的向量化复现（BBI<0.16→笔直；n_regions≤1 且
    max_kappa≥0.35→Ω；否则→正弦），特征取自 kappa_cache 的 features 矩阵。
  · 特征（与 8k 版同名同维）：
      bbi            features[:,0]（1 维）
      sbpd           PCA(κ(s),3)
      sbpd_headtail  PCA(κ(s),4)（第 4 主成分承载头尾方向信息）
      eigenworms     PCA(θ(s),6)（θ 来自 step19_theta_full 全量重提）
  · 分类器：StandardScaler + LinearSVC（primal，C∈{0.1,1,10} 内层选优），
    405 万帧可行；另附 100k 子采样上的 RBF-SVM 对照（协议接近 8k 版）。
  · 两种划分协议：worm-grouped GroupKFold(5)（按虫隔离，防同虫泄漏）
    与 frame-random KFold(5, shuffle)。

用法（服务器，venv）：
  PY t1_fulldata.py --cache atlas_out_all/kappa_cache.npz \
       --theta atlas_out_all/theta_full/theta_full.npy \
       --theta-valid atlas_out_all/theta_full/theta_valid.npy \
       -o atlas_out_all/t1_fulldata
"""
import os
import sys
import time
import argparse
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# 与 worm_labeler 一致的阈值
STRAIGHT_BBI = 0.16
OMEGA_MAX_KAPPA = 0.35

FEATS = [('bbi', 1), ('sbpd', 3), ('sbpd_headtail', 4), ('eigenworms', 6)]


def rule_labels(feat_matrix):
    """classify_by_rules 的向量化复现（0 正弦 / 1 Ω / 2 笔直）。"""
    bbi = feat_matrix[:, 0].astype(np.float64)
    max_kappa = feat_matrix[:, 1].astype(np.float64)
    n_regions = feat_matrix[:, 2].astype(np.float64)
    y = np.zeros(len(bbi), dtype=np.int64)            # 正弦兜底
    omega = (n_regions <= 1) & (max_kappa >= OMEGA_MAX_KAPPA)
    y[omega] = 1
    straight = bbi < STRAIGHT_BBI                     # 笔直优先级最高
    y[straight] = 2
    return y


def pca_fit(X, n_comp, seed=42, chunk_note=''):
    t0 = time.time()
    pca = PCA(n_components=n_comp, svd_solver='randomized',
              random_state=seed, copy=False)
    Z = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_.sum()
    print(f'  PCA{n_comp} {chunk_note}: {time.time()-t0:.0f}s, '
          f'解释方差 {evr:.3f}')
    return Z.astype(np.float32)


def cv_eval(X, y, groups, protocol, seed=42):
    """返回 (mean_acc, sd_acc, folds)。protocol: 'worm' | 'frame'"""
    if protocol == 'worm':
        splits = GroupKFold(n_splits=5).split(X, y, groups)
    else:
        skf = KFold(n_splits=5, shuffle=True, random_state=seed)
        splits = skf.split(X, y)
    accs = []
    for tr, te in splits:
        clf = make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, max_iter=5000, dual=False, random_state=seed))
        clf.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs)), float(np.std(accs)), accs


def cv_eval_linear_grid(X, y, groups, protocol, seed=42):
    """内层在 train 子样本(≤200k)上选 C∈{0.1,1,10}，外层全量训练评估。"""
    rng = np.random.default_rng(seed)
    if protocol == 'worm':
        splits = list(GroupKFold(n_splits=5).split(X, y, groups))
    else:
        splits = list(KFold(n_splits=5, shuffle=True,
                            random_state=seed).split(X, y))
    accs = []
    for tr, te in splits:
        sub = tr if len(tr) <= 200_000 else \
            rng.choice(tr, 200_000, replace=False)
        best_c, best_acc = 1.0, -1.0
        for c in (0.1, 1.0, 10.0):
            clf = make_pipeline(
                StandardScaler(),
                LinearSVC(C=c, max_iter=5000, dual=False,
                          random_state=seed))
            clf.fit(X[sub], y[sub])
            a = accuracy_score(y[sub], clf.predict(X[sub]))
            if a > best_acc:
                best_acc, best_c = a, c
        clf = make_pipeline(
            StandardScaler(),
            LinearSVC(C=best_c, max_iter=5000, dual=False,
                      random_state=seed))
        clf.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs)), float(np.std(accs)), accs


def cv_eval_rbf(X, y, groups, protocol, seed=42, max_n=50_000):
    """100k 子采样上的 RBF-SVM（C=1, gamma=scale），协议接近 8k 版。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), min(max_n, len(y)), replace=False)
    Xs, ys = X[idx], y[idx]
    gs = groups[idx]
    if protocol == 'worm':
        splits = GroupKFold(n_splits=5).split(Xs, ys, gs)
    else:
        splits = KFold(n_splits=5, shuffle=True,
                       random_state=seed).split(Xs, ys)
    accs = []
    for tr, te in splits:
        clf = make_pipeline(StandardScaler(),
                            SVC(C=1.0, kernel='rbf'))
        clf.fit(Xs[tr], ys[tr])
        accs.append(accuracy_score(ys[te], clf.predict(Xs[te])))
    return float(np.mean(accs)), float(np.std(accs)), accs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', default='atlas_out_all/kappa_cache.npz')
    ap.add_argument('--theta', default='atlas_out_all/theta_full/theta_full.npy')
    ap.add_argument('--theta-valid', default='atlas_out_all/theta_full/theta_valid.npy')
    ap.add_argument('-o', '--out-prefix', default='atlas_out_all/t1_fulldata')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    print('[load] kappa_cache ...')
    z = np.load(args.cache, allow_pickle=True)
    kappa = z['kappa']                       # (N,100) float32
    feats = z['features']                    # (N,7)
    groups = np.asarray([str(g) for g in z['groups']])
    worms = np.asarray([str(w) for w in z['worms']])
    rows = np.asarray([str(r) for r in z['rows']])
    dates = np.asarray([str(d) for d in z['dates']])
    n = len(kappa)
    print(f'  {n:,} 帧, {time.time()-t0:.0f}s')

    print('[load] theta ...')
    theta = np.load(args.theta, mmap_mode='r')
    theta_valid = np.load(args.theta_valid)
    assert len(theta) == n, f'theta 帧数 {len(theta)} != 缓存 {n}'
    m = theta_valid
    print(f'  θ 有效帧 {int(m.sum()):,}（{100*m.mean():.2f}%）')

    y_all = rule_labels(feats)
    worm_key = np.asarray([f'{g}|{d}|{r}|{w}'
                           for g, d, r, w in zip(groups, dates, rows, worms)])
    uniq_worms = len(set(worm_key.tolist()))
    dist = {k: int((y_all == v).sum()) for k, v in
            [('sinusoidal', 0), ('omega', 1), ('straight', 2)]}
    print(f'[labels] 分布：{dist}，虫数 {uniq_worms:,}')

    lines = []
    lines.append(f'frames={n:,} theta_valid={int(m.sum()):,}')
    lines.append(f'label_dist={dist}')
    lines.append(f'n_worms={uniq_worms:,}')
    lines.append('feature,n_dim,protocol,classifier,acc_mean,acc_sd,folds')

    # PCA 特征一次性算好（全帧，float32）
    print('[pca] sbpd / sbpd_headtail（κ 全量）...')
    Z3 = pca_fit(kappa, 3, args.seed, 'kappa')
    Z4 = pca_fit(kappa, 4, args.seed, 'kappa')
    print('[pca] eigenworms（θ 全量）...')
    theta_m = np.asarray(theta[m], dtype=np.float32)
    Z6_full = np.full((n, 6), np.nan, dtype=np.float32)
    Z6_full[m] = pca_fit(theta_m, 6, args.seed, 'theta')

    feat_sets = {
        'bbi': feats[:, 0:1].astype(np.float32),
        'sbpd': Z3,
        'sbpd_headtail': Z4,
        'eigenworms': Z6_full,
    }

    # 评估两种口径：全帧（θ 特征仅有效帧子集）与 θ 有效帧交集（四特征同帧）
    eval_masks = {'all_frames': np.ones(n, dtype=bool), 'theta_valid': m}

    for mask_name, mask in eval_masks.items():
        y = y_all[mask]
        g = worm_key[mask]
        print(f'\n[eval] 口径={mask_name}（{int(mask.sum()):,} 帧）')
        for fname, ndim in FEATS:
            X = feat_sets[fname][mask]
            if np.isnan(X).any():
                good = ~np.isnan(X).any(axis=1)
                X, y, g = X[good], y[good], g[good]
            for proto in ('worm', 'frame'):
                mu, sd, folds = cv_eval_linear_grid(X, y, g, proto, args.seed)
                print(f'  {fname:14s} {proto:5s} linear : '
                      f'{mu:.3f}±{sd:.3f} {["%.3f" % f for f in folds]}')
                lines.append(f'{fname},{ndim},{proto},linear_svm,'
                             f'{mu:.4f},{sd:.4f},"{folds}"')
            mu, sd, folds = cv_eval_rbf(X, y, g, 'frame', args.seed)
            print(f'  {fname:14s} frame rbf50k: {mu:.3f}±{sd:.3f}')
            lines.append(f'{fname},{ndim},frame,rbf_svm_100k,'
                         f'{mu:.4f},{sd:.4f},"{folds}"')
            mu, sd, folds = cv_eval_rbf(X, y, g, 'worm', args.seed)
            print(f'  {fname:14s} worm  rbf50k: {mu:.3f}±{sd:.3f}')
            lines.append(f'{fname},{ndim},worm,rbf_svm_100k,'
                         f'{mu:.4f},{sd:.4f},"{folds}"')

    out_txt = args.out_prefix + '.txt'
    out_csv = args.out_prefix + '.csv'
    with open(out_txt, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    with open(out_csv, 'w', encoding='utf-8') as fh:
        fh.write('feature,n_dim,protocol,classifier,acc_mean,acc_sd,folds\n')
        for ln in lines[3:]:
            fh.write(ln + '\n')
    print(f'\n[done] 已写入 {out_txt} / {out_csv}（总耗时 {time.time()-t0:.0f}s）')


if __name__ == '__main__':
    main()
