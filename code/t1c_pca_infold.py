#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""t1c_pca_infold.py — PCA 拟合方式稳健性：全局一次拟合 vs 折内重拟合

背景：Table 1 的 PCA 分量在全帧集上一次性拟合、仅分类器交叉验证（与 8k
协议一致）。审稿人可能质疑特征提取层面的泄漏。本脚本在 50k 子样本上对比：
  · global ：PCA 在全部 50k 帧上拟合一次（= Table 1 协议）
  · infold ：PCA 在每个训练折内重拟合，测试折用训练折 PCA 投影
特征 sbpd(κ→3)/sbpd_ht(κ→4)/eigenworms(θ→6)；bbi 无 PCA（作对照）。
协议：按虫五折 + 帧随机五折；线性 SVM C=1。

用法（服务器）：PY t1c_pca_infold.py -o atlas_out_all/t1c_pca_infold
"""
import argparse
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def pca_apply(mode, Xtr, Xte, n_comp, seed):
    """返回 (Ztr, Zte)。global：用全体数据拟合的投影（预先算好则更快，
    这里为公平直接在全子样本上拟合一次后投影）；infold：折内拟合。"""
    if mode == 'infold':
        pca = PCA(n_components=n_comp, svd_solver='randomized',
                  random_state=seed)
        Ztr = pca.fit_transform(Xtr)
        Zte = pca.transform(Xte)
    else:  # global
        pca = PCA(n_components=n_comp, svd_solver='randomized',
                  random_state=seed)
        Ztr = pca.fit_transform(Xtr)   # 拟合仍只在训练折（global 的意义是
        Zte = pca.transform(Xte)       # 用全子样本拟合，见 prefit 路径）
    return Ztr, Zte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='atlas_out_all/kappa_cache.npz')
    ap.add_argument('--theta', default='atlas_out_all/theta_full/theta_full.npy')
    ap.add_argument('--theta-valid', default='atlas_out_all/theta_full/theta_valid.npy')
    ap.add_argument('-o', '--out', default='atlas_out_all/t1c_pca_infold')
    ap.add_argument('--sub', type=int, default=50_000)
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
    rng = np.random.default_rng(args.seed)
    sub = np.sort(rng.choice(n, args.sub, replace=False))
    print(f'[load] {n:,} 帧，子样本 {args.sub:,}', flush=True)

    worm_key = np.asarray([f'{g}|{d}|{r}|{w}'
                           for g, d, r, w in zip(groups, dates, rows, worms)])
    g_sub = worm_key[sub]
    theta = np.load(args.theta, mmap_mode='r')
    valid = np.load(args.theta_valid)
    sub_v = sub[valid[sub]]

    # 全局 PCA 预拟合（global 模式的正确实现：全子样本拟合一次）
    pre_pca = {
        3: PCA(n_components=3, svd_solver='randomized',
               random_state=args.seed).fit(kappa[sub]),
        4: PCA(n_components=4, svd_solver='randomized',
               random_state=args.seed).fit(kappa[sub]),
        6: PCA(n_components=6, svd_solver='randomized',
               random_state=args.seed).fit(
            np.asarray(theta[sub_v], dtype=np.float32)),
    }
    print(f'[pca] 全局预拟合完成 {time.time()-t0:.0f}s', flush=True)

    # 标签用论文切点
    bbi = feats[:, 0].astype(np.float64)
    mk = feats[:, 1].astype(np.float64)
    nr = feats[:, 2].astype(np.float64)
    y_full = np.zeros(n, dtype=np.int64)
    y_full[(nr <= 1) & (mk >= 0.35)] = 1
    y_full[bbi < 0.16] = 2
    y = y_full[sub]

    feature_sets = {
        'bbi': ('none', feats[sub, 0:1].astype(np.float32)),
        'sbpd': ('kappa', None),
        'sbpd_ht': ('kappa', None),
        'eigenworms': ('theta', None),
    }

    lines = ['feature,mode,protocol,acc_mean,acc_sd,folds']
    for fname, (src, X_direct) in feature_sets.items():
        ndim = {'bbi': 0, 'sbpd': 3, 'sbpd_ht': 4, 'eigenworms': 6}[fname]
        for mode in ('global', 'infold'):
            for proto in ('worm', 'frame'):
                if proto == 'worm':
                    splits = GroupKFold(n_splits=5).split(sub, y, g_sub)
                else:
                    splits = KFold(n_splits=5, shuffle=True,
                                   random_state=args.seed).split(sub, y)
                fold_acc = []
                for tr_pos, te_pos in splits:
                    tr_idx, te_idx = sub[tr_pos], sub[te_pos]
                    ytr, yte = y[tr_pos], y[te_pos]
                    if src == 'none':
                        Xtr = X_direct[tr_pos]
                        Xte = X_direct[te_pos]
                    elif src == 'kappa':
                        if mode == 'global':
                            Xtr = pre_pca[ndim].transform(kappa[tr_idx])
                            Xte = pre_pca[ndim].transform(kappa[te_idx])
                        else:
                            pca = PCA(n_components=ndim,
                                      svd_solver='randomized',
                                      random_state=args.seed)
                            Xtr = pca.fit_transform(kappa[tr_idx])
                            Xte = pca.transform(kappa[te_idx])
                    else:  # theta
                        tv_tr = sub_v[np.isin(sub_v, tr_idx)]
                        tv_te = sub_v[np.isin(sub_v, te_idx)]
                        if mode == 'global':
                            Xtr = np.zeros((len(tr_idx), ndim), np.float32)
                            Xte = np.zeros((len(te_idx), ndim), np.float32)
                            pos_tr = {v: k for k, v in enumerate(tr_idx)}
                            pos_te = {v: k for k, v in enumerate(te_idx)}
                            Xtr[[pos_tr[v] for v in tv_tr]] = \
                                pre_pca[ndim].transform(
                                    np.asarray(theta[tv_tr], np.float32))
                            Xte[[pos_te[v] for v in tv_te]] = \
                                pre_pca[ndim].transform(
                                    np.asarray(theta[tv_te], np.float32))
                        else:
                            pca = PCA(n_components=ndim,
                                      svd_solver='randomized',
                                      random_state=args.seed)
                            Ztr = pca.fit_transform(
                                np.asarray(theta[tv_tr], np.float32))
                            Zte = pca.transform(
                                np.asarray(theta[tv_te], np.float32))
                            Xtr = np.zeros((len(tr_idx), ndim), np.float32)
                            Xte = np.zeros((len(te_idx), ndim), np.float32)
                            pos_tr = {v: k for k, v in enumerate(tr_idx)}
                            pos_te = {v: k for k, v in enumerate(te_idx)}
                            Xtr[[pos_tr[v] for v in tv_tr]] = Ztr
                            Xte[[pos_te[v] for v in tv_te]] = Zte
                    clf = make_pipeline(
                        StandardScaler(),
                        LinearSVC(C=1.0, max_iter=5000, dual=False,
                                  random_state=args.seed))
                    clf.fit(Xtr, ytr)
                    fold_acc.append(accuracy_score(yte, clf.predict(Xte)))
                mu, sd = float(np.mean(fold_acc)), float(np.std(fold_acc))
                row = (f'{fname},{mode},{proto},{mu:.4f},{sd:.4f},'
                       f'"[{' '.join(f'{a:.3f}' for a in fold_acc)}]"')
                lines.append(row)
                print(f'  {fname:11s} {mode:7s} {proto:5s}: {mu:.3f}±{sd:.3f}',
                      flush=True)

    with open(args.out + '.csv', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'[done] {args.out}.csv（总耗时 {time.time()-t0:.0f}s）', flush=True)


if __name__ == '__main__':
    main()
