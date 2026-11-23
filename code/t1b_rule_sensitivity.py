#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""t1b_rule_sensitivity.py — 操作化规则阈值敏感性（全量缓存，500k 子样本评估）

背景：论文的操作化标签由 BBI<0.16（笔直）与 max_kappa≥0.35 且 n_regions≤1（Ω）
定义。审稿人会问：标签与 Table 1 排序对这两个切点多敏感？
本脚本扫描 BBI∈{0.14,0.16,0.18} × Ω∈{0.32,0.35,0.38} 共 9 组：
  · 三类占比（全量特征，零成本）
  · Table 1 三特征（bbi/sbpd/eigenworms，PCA 全量一次性拟合）的
    按虫五折线性 SVM 准确率（500k 固定子样本，seed=42）

用法（服务器）：PY t1b_rule_sensitivity.py -o atlas_out_all/t1b_rule_sens
"""
import argparse
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def rule_labels(feat, bbi_cut, omega_cut):
    bbi = feat[:, 0].astype(np.float64)
    mk = feat[:, 1].astype(np.float64)
    nr = feat[:, 2].astype(np.float64)
    y = np.zeros(len(bbi), dtype=np.int64)
    y[(nr <= 1) & (mk >= omega_cut)] = 1
    y[bbi < bbi_cut] = 2
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='atlas_out_all/kappa_cache.npz')
    ap.add_argument('--theta', default='atlas_out_all/theta_full/theta_full.npy')
    ap.add_argument('--theta-valid', default='atlas_out_all/theta_full/theta_valid.npy')
    ap.add_argument('-o', '--out', default='atlas_out_all/t1b_rule_sens')
    ap.add_argument('--sub', type=int, default=500_000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()

    z = np.load(args.cache, allow_pickle=True)
    kappa = z['kappa']
    feats = z['features']
    worms = np.asarray([str(w) for w in z['worms']])
    rows = np.asarray([str(r) for r in z['rows']])
    dates = np.asarray([str(d) for d in z['dates']])
    groups = np.asarray([str(g) for g in z['groups']])
    n = len(kappa)
    print(f'[load] {n:,} 帧', flush=True)

    print('[pca] 全量 PCA（一次拟合，所有阈值组合共用）...', flush=True)
    Z3 = PCA(n_components=3, svd_solver='randomized',
             random_state=args.seed, copy=False).fit_transform(kappa)
    theta = np.load(args.theta, mmap_mode='r')
    valid = np.load(args.theta_valid)
    Z6 = PCA(n_components=6, svd_solver='randomized',
             random_state=args.seed, copy=False).fit_transform(
        np.asarray(theta[valid], dtype=np.float32))
    print(f'  PCA 完成 {time.time()-t0:.0f}s', flush=True)

    worm_key = np.asarray([f'{g}|{d}|{r}|{w}'
                           for g, d, r, w in zip(groups, dates, rows, worms)])
    rng = np.random.default_rng(args.seed)
    sub = rng.choice(n, args.sub, replace=False)
    sub = np.sort(sub)
    g_sub = worm_key[sub]

    X_by_feat = {
        'bbi': feats[sub, 0:1].astype(np.float32),
        'sbpd': Z3[sub].astype(np.float32),
        'eigenworms': Z6[sub].astype(np.float32),
    }

    bbi_cuts = (0.14, 0.16, 0.18)
    omega_cuts = (0.32, 0.35, 0.38)
    lines = ['bbi_cut,omega_cut,p_sinusoidal,p_omega,p_straight,'
             'acc_bbi,acc_sbpd,acc_eigenworms']
    print(f'[eval] {args.sub:,} 帧子样本，9 组切点 × 3 特征 × 5 折', flush=True)
    for bc in bbi_cuts:
        for oc in omega_cuts:
            y_full = rule_labels(feats, bc, oc)
            prop = [float((y_full == v).mean()) for v in (0, 1, 2)]
            y = y_full[sub]
            accs = {}
            for fname, X in X_by_feat.items():
                fold_acc = []
                for tr, te in GroupKFold(n_splits=5).split(X, y, g_sub):
                    clf = make_pipeline(
                        StandardScaler(),
                        LinearSVC(C=1.0, max_iter=5000, dual=False,
                                  random_state=args.seed))
                    clf.fit(X[tr], y[tr])
                    fold_acc.append(accuracy_score(y[te], clf.predict(X[te])))
                accs[fname] = float(np.mean(fold_acc))
            row = (f'{bc},{oc},{prop[0]:.4f},{prop[1]:.4f},{prop[2]:.4f},'
                   f'{accs["bbi"]:.4f},{accs["sbpd"]:.4f},'
                   f'{accs["eigenworms"]:.4f}')
            lines.append(row)
            print(f'  BBI<{bc} Ω≥{oc}: 占比 {prop[0]:.3f}/{prop[1]:.3f}/'
                  f'{prop[2]:.3f} | acc bbi {accs["bbi"]:.3f} sbpd '
                  f'{accs["sbpd"]:.3f} eig {accs["eigenworms"]:.3f}',
                  flush=True)

    with open(args.out + '.csv', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'[done] {args.out}.csv（总耗时 {time.time()-t0:.0f}s）', flush=True)


if __name__ == '__main__':
    main()
