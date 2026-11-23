"""按虫分组的交叉验证（TBD-06）：GroupShuffleSplit 与 GroupKFold。

问题背景
--------
原 `step8_train_svm.py` 的 `train_test_split(..., stratify=y)` 是**帧级随机划分**：
同一只虫的相邻帧（相差数秒、形状几乎相同）会同时落入训练集与测试集，
等价于"考前偷看答案"，所得性能是**上界**而非泛化能力。

本脚本把划分单位从"帧"改为"虫"：
  · GroupShuffleSplit(groups=worm_id, test_size=0.25)  —— 与旧设置可比
  · GroupKFold(n_splits=5, groups=worm_id)             —— 交叉验证，报均值±标准差
同一只虫的全部帧只进入训练集或只进入测试集，杜绝数据泄漏。

另外报告**虫级投票精度**：把同一虫的测试帧预测结果投票，得到虫级判对率。
该指标比帧级精度更严格，也是审稿人常要求的口径。

数据来源
--------
  · 标签：`output_3class_smoke800/sampled_labels_7999.csv`（filename + label）
  · 特征缓存：`sbpd_raw_cache_7999.pkl`（kappa / theta / bbi / ht / y）
    缓存按 label_data 顺序保存且抽样集全部有效，故与 CSV 行序一一对应。

用法（ele 环境）
----------------
  python step9_group_cv.py -c sbpd_raw_cache_7999.pkl \
      -l output_3class_smoke800/sampled_labels_7999.csv \
      -o output_3class_smoke800/group_cv_results.csv
"""
import os
import re
import csv
import time
import pickle
import argparse

import numpy as np
from sklearn.model_selection import (GroupShuffleSplit, GroupKFold,
                                     train_test_split, GridSearchCV)
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

FEATURES = ['bbi', 'eigenworms', 'sbpd', 'sbpd_headtail']
FEATURE_DIM = {'bbi': 1, 'eigenworms': 6, 'sbpd': 3, 'sbpd_headtail': 4}
CLASS_NAMES = {0: 'Sinusoidal', 1: 'Omega', 2: 'Straight'}
PARAM_GRID = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1]}


def log(msg):
    """带时间戳的进度输出（2026-11-17）：旧版在 6 万样本上 GridSearchCV
    静默运行数小时无任何输出，被误判为卡死而 Ctrl+C。"""
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)


def parse_worm_id(filename):
    """从 `组/日期/行/虫_帧.png` 解析 worm_id（去掉帧号，同虫同 id）。

    2026-11-16：worm_id 用前四段（组/日期/行/虫名）。旧版只取虫名段，
    不同组/皿的同名 worm1 会撞成同一 id，group 划分时跨组泄漏；
    公开数据集 train/test 更是必然重号。"""
    fn = filename.replace('\\', '/')
    parts = fn.split('/')
    if len(parts) >= 4:
        worm = '/'.join(parts[:4])
        worm = re.sub(r'_\d+\.png$', '', worm)
        plate = '_'.join(parts[:3])
    else:
        worm = re.sub(r'_\d+\.png$', '', parts[-1])
        plate = worm
    return worm, plate


def build_X(raw, feat_type, pca=None):
    """按特征类型构建特征矩阵（与 step8_train_svm.build_feature_matrix 同逻辑）。"""
    if feat_type == 'bbi':
        bbi = np.asarray(raw['bbi'])
        # 兼容两种缓存：旧缓存 bbi 为 (N,1)，t1_postdrop 新缓存为 (N,)
        return (bbi.reshape(-1, 1) if bbi.ndim == 1 else bbi), None
    src = raw['theta'] if feat_type == 'eigenworms' else raw['kappa']
    n_comp = FEATURE_DIM[feat_type]
    if pca is None:
        pca = PCA(n_components=n_comp, random_state=42).fit(src)
    return pca.transform(src), pca


def fit_predict(Xtr, ytr, Xte, smote_ratio=0.5, seed=42):
    """在训练折内做 SMOTE + 网格搜索，避免任何跨折信息泄漏。"""
    _t0 = time.time()
    counts = Counter(ytr.tolist())
    drop = [c for c, n in counts.items() if n < 2]
    if drop:
        keep = ~np.isin(ytr, drop)
        Xtr, ytr = Xtr[keep], ytr[keep]
        counts = Counter(ytr.tolist())
    Xtr_b, ytr_b = Xtr, ytr
    if smote_ratio and smote_ratio > 0:
        majority = max(counts.values())
        target = int(majority * smote_ratio)
        strategy = {c: target for c, n in counts.items() if target > n}
        if strategy:
            k = max(1, min(counts.values()) - 1)
            try:
                sm = SMOTE(sampling_strategy=strategy, random_state=seed,
                           k_neighbors=k)
                Xtr_b, ytr_b = sm.fit_resample(Xtr, ytr)
            except Exception:
                Xtr_b, ytr_b = Xtr, ytr
    grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',
                            cache_size=1000),
                        PARAM_GRID, cv=5, scoring='f1_macro', n_jobs=-1,
                        verbose=1)
    grid.fit(Xtr_b, ytr_b)
    global _FIT_CALLS
    _FIT_CALLS += 1
    log(f'fit_predict #{_FIT_CALLS}/77 完成 ({time.time() - _t0:.1f}s) '
        f'train={len(ytr_b)} best={grid.best_params_}')
    return grid.best_estimator_.predict(Xte), grid.best_params_


_FIT_CALLS = 0


def worm_level_vote(y_true, y_pred, groups):
    """虫级投票：同一 worm 的帧预测取多数票，与虫级真值（众数）比较。"""
    uniq = {}
    for g, t, p in zip(groups, y_true, y_pred):
        uniq.setdefault(g, []).append((t, p))
    correct = 0
    total = 0
    for g, tp in uniq.items():
        if len(tp) < 2:
            continue
        t_vote = Counter(t for t, _ in tp).most_common(1)[0][0]
        p_vote = Counter(p for _, p in tp).most_common(1)[0][0]
        correct += int(t_vote == p_vote)
        total += 1
    return correct / max(total, 1), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cache', default='sbpd_raw_cache_7999.pkl')
    ap.add_argument('-l', '--labels',
                    default='output_3class_smoke800/sampled_labels_7999.csv')
    ap.add_argument('-o', '--out',
                    default='output_3class_smoke800/group_cv_results.csv')
    ap.add_argument('--smote-ratio', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    # ---- 载入标签 ----
    filenames, labels = [], []
    with open(args.labels, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            filenames.append(r['filename'])
            labels.append(int(r['label']))
    y_all = np.asarray(labels)
    worm_ids, plate_ids = [], []
    for fn in filenames:
        w, p = parse_worm_id(fn)
        worm_ids.append(w)
        plate_ids.append(p)
    worm_ids = np.asarray(worm_ids)
    plate_ids = np.asarray(plate_ids)
    n_worms = len(np.unique(worm_ids))
    n_plates = len(np.unique(plate_ids))
    print(f'标签样本: {len(y_all):,} 帧，{n_worms} 只虫，{n_plates} 个培养皿')

    # ---- 载入特征缓存 ----
    with open(args.cache, 'rb') as f:
        raw = pickle.load(f)
    y_cache = raw['y']
    if 'filenames' in raw:
        # 新缓存（step8 2026-11-16 起）：按文件名对齐，容忍提取失败帧
        fn2row = {fn: i for i, fn in enumerate(raw['filenames'])}
        keep = [i for i, fn in enumerate(filenames) if fn in fn2row]
        dropped = len(filenames) - len(keep)
        if dropped:
            print(f'缓存缺失 {dropped} 帧（提取失败），按文件名对齐后跳过')
        filenames = [filenames[i] for i in keep]
        labels = [labels[i] for i in keep]
        y_all = y_all[keep]
        worm_ids = worm_ids[keep]
        plate_ids = plate_ids[keep]
        order = [fn2row[fn] for fn in filenames]
        for k2 in ('kappa', 'theta', 'bbi', 'ht', 'y'):
            raw[k2] = raw[k2][order]
        y_cache = raw['y']
        mismatch = int((y_cache != y_all).sum())
        if mismatch:
            print(f'[警告] 缓存标签与标签 CSV 不一致 {mismatch} 行，以 CSV 为准')
    if len(y_cache) != len(y_all):
        raise SystemExit(
            f'缓存行数({len(y_cache)})与标签行数({len(y_all)})不一致，'
            f'请重新运行 step8_train_svm.py 生成配套缓存')
    print(f'缓存特征: kappa{raw["kappa"].shape}, theta{raw["theta"].shape}')

    # ---- 三种划分协议 × 四种特征 ----
    results = []
    protocols = ['frame_random(leaky)', 'group_shuffle', 'group_kfold5']
    for feat in FEATURES:
        X, _ = build_X(raw, feat)
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        print(f'\n===== 特征: {feat}（{FEATURE_DIM[feat]} 维）=====')

        # 协议 1：帧级随机划分（旧，泄漏）—— 作为上界参照
        accs, f1s, waccs = [], [], []
        for rep in range(3):
            idx = np.arange(len(y_all))
            tr, te = train_test_split(idx, test_size=0.25,
                                      random_state=args.seed + rep,
                                      stratify=y_all)
            yp, _ = fit_predict(Xs[tr], y_all[tr], Xs[te],
                                args.smote_ratio, args.seed)
            accs.append(accuracy_score(y_all[te], yp))
            f1s.append(f1_score(y_all[te], yp, average='macro'))
            wa, _ = worm_level_vote(y_all[te], yp, worm_ids[te])
            waccs.append(wa)
        results.append([feat, protocols[0], np.mean(accs), np.std(accs),
                        np.mean(f1s), np.std(f1s), np.mean(waccs), ''])
        print(f'  [帧级随机] acc={np.mean(accs):.4f}±{np.std(accs):.4f}  '
              f'macroF1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  '
              f'worm-vote={np.mean(waccs):.4f}')

        # 协议 2：GroupShuffleSplit（与旧设置同比例，但按虫分组）
        gss = GroupShuffleSplit(n_splits=5, test_size=0.25,
                                random_state=args.seed)
        accs, f1s, waccs = [], [], []
        for tr, te in gss.split(Xs, y_all, groups=worm_ids):
            yp, _ = fit_predict(Xs[tr], y_all[tr], Xs[te],
                                args.smote_ratio, args.seed)
            accs.append(accuracy_score(y_all[te], yp))
            f1s.append(f1_score(y_all[te], yp, average='macro'))
            wa, _ = worm_level_vote(y_all[te], yp, worm_ids[te])
            waccs.append(wa)
        results.append([feat, protocols[1], np.mean(accs), np.std(accs),
                        np.mean(f1s), np.std(f1s), np.mean(waccs), ''])
        print(f'  [GroupShuffle] acc={np.mean(accs):.4f}±{np.std(accs):.4f}  '
              f'macroF1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  '
              f'worm-vote={np.mean(waccs):.4f}')

        # 协议 3：GroupKFold(5)
        gkf = GroupKFold(n_splits=5)
        accs, f1s, waccs = [], [], []
        for tr, te in gkf.split(Xs, y_all, groups=worm_ids):
            yp, _ = fit_predict(Xs[tr], y_all[tr], Xs[te],
                                args.smote_ratio, args.seed)
            accs.append(accuracy_score(y_all[te], yp))
            f1s.append(f1_score(y_all[te], yp, average='macro'))
            wa, _ = worm_level_vote(y_all[te], yp, worm_ids[te])
            waccs.append(wa)
        results.append([feat, protocols[2], np.mean(accs), np.std(accs),
                        np.mean(f1s), np.std(f1s), np.mean(waccs), ''])
        print(f'  [GroupKFold5] acc={np.mean(accs):.4f}±{np.std(accs):.4f}  '
              f'macroF1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  '
              f'worm-vote={np.mean(waccs):.4f}')

    # ---- 最佳协议的混淆矩阵（SBPD + GroupKFold）----
    print('\n===== 混淆矩阵：SBPD + GroupKFold（按帧汇总）=====')
    X, _ = build_X(raw, 'sbpd')
    Xs = StandardScaler().fit_transform(X)
    gkf = GroupKFold(n_splits=5)
    yt_all, yp_all = [], []
    for tr, te in gkf.split(Xs, y_all, groups=worm_ids):
        yp, _ = fit_predict(Xs[tr], y_all[tr], Xs[te],
                            args.smote_ratio, args.seed)
        yt_all.extend(y_all[te].tolist())
        yp_all.extend(yp.tolist())
    cm = confusion_matrix(yt_all, yp_all)
    n_cls = cm.shape[0]   # 某些折/小样本下可能不足 3 类
    print('         ' + ''.join(f'{CLASS_NAMES[c][:10]:>12s}'
                                for c in range(min(n_cls, 3))))
    for c in range(min(n_cls, 3)):
        print(f'  {CLASS_NAMES[c][:8]:<8s}' +
              ''.join(f'{cm[c][k]:>12d}' for k in range(min(cm.shape[1], 3))))

    # ---- 写 CSV ----
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['feature', 'protocol', 'acc_mean', 'acc_std',
                    'macroF1_mean', 'macroF1_std', 'worm_vote_acc', 'note'])
        for row in results:
            w.writerow([row[0], row[1]] + [f'{v:.4f}' for v in row[2:7]] + [row[7]])
        w.writerow([])
        w.writerow(['# confusion matrix (SBPD + GroupKFold, frame-level)'])
        n_cls = cm.shape[0]
        w.writerow(['true\\pred'] + [CLASS_NAMES[c] for c in range(min(n_cls, 3))])
        for c in range(min(n_cls, 3)):
            w.writerow([CLASS_NAMES[c]] + list(cm[c]))
    print(f'\n结果已写入: {args.out}')
    print('\n解读：frame_random 行为泄漏上界；group_* 行为按虫分组的诚实性能。'
          '论文应以 group_* 为准，frame_random 仅作对照说明。')


if __name__ == '__main__':
    main()
