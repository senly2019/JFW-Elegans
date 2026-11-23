"""
第8步：SBPD 特征提取 + SVM 分类器训练
=======================================
实现论文「空间弯曲模式描述子（SBPD）+ SVM」完整方法，并做表2/表3 实验：

  表2 —— 四种特征表示的分类性能对比：
    · BBI（单标量，1 维）          —— 体长归一化平均曲率
    · Eigenworms（角度 PCA，6 维） —— 对切线角函数 θ(s) 做 PCA（Stephens 2008）
    · SBPD（曲率 fPCA，3 维）      —— 对曲率函数 κ(s) 做函数型 PCA（本文）
    · SBPD+头尾（4 维）            —— SBPD 3 维 + 头尾判别方向得分 1 维

  表3 —— SBPD+SVM 混淆矩阵（测试集）

  说明：等弧长采样下，函数型 PCA（fPCA，解 Mercer 积分方程）与离散 PCA
        数学等价——前者的 fPC 基函数 ξ_k(s) 即后者的主成分载荷向量，
        得分 c_ik = ∫(κ_i-κ̄)ξ_k ds 即 PCA 投影得分。故此处用 sklearn PCA
        对 100 点离散曲率做等价实现（论文 2.4 节亦为离散骨架）。

用法：
  python step8_train_svm.py -l labels.csv -i D:/data/image -m svm_model.pkl
  python step8_train_svm.py -l labels.csv -i D:/data/image -m svm_model.pkl \
      --feature sbpd_headtail      # 指定最终模型使用的特征
"""

import cv2
import numpy as np
import os
import sys
import csv
import pickle
import argparse
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from worm_labeler import extract_skeleton, compute_curvature_profile
from head_tail_discriminator import pointwise_curvature

# 动作类别名称映射（三类存活姿态：正弦型/Ω 型/笔直型，2026-10-23 起）。
# 倒退类已取消（单帧形状与正弦型相同、跨帧判别依赖头尾方向，实测污染严重）；
# 原"静止"更名为"笔直型"（纯形状定义，去运动学含义）；
# 原"缠绕型"经三重实证取消（自交检测/几何紧密度/时序运动均无区分度，
# 2026-10-23，详见 worm_labeler 第 4 节注释），其帧并入 Ω 型/正弦型。
# 死虫已由上游 z_tracking_only.py 依据人标注 DeadBoxes 过滤，本管线无死亡类
LABEL_NAMES = {0: '正弦型', 1: 'Ω 型', 2: '笔直型'}
LABEL_SHORT = {0: '正弦', 1: 'Ω', 2: '笔直'}
LABEL_EN = {0: 'Sinusoidal', 1: 'Omega', 2: 'Straight'}

# 四种特征表示
FEATURES = ['bbi', 'eigenworms', 'sbpd', 'sbpd_headtail']
FEATURE_DIM = {'bbi': 1, 'eigenworms': 6, 'sbpd': 3, 'sbpd_headtail': 4}
FEATURE_CN = {'bbi': 'BBI（单标量）', 'eigenworms': 'Eigenworms（6维）',
              'sbpd': 'SBPD（3维）', 'sbpd_headtail': 'SBPD+头尾方向（4维）'}

# SMOTE 过采样比例（少数类目标 = 多数类 × 该比例，2026-10-22 由 'auto' 改为 0.5；
# 见 train_svm docstring）。可由 --smote-ratio 覆盖。
SMOTE_RATIO = 0.5


# ============================================================
# 1. 读取标注 CSV
# ============================================================
def load_labels(csv_path):
    """读取 simulate_labels 输出的 CSV，返回 [(filename, label, head_is_first)]。

    head_is_first 用于 κ(s)/θ(s) 方向归一化（统一 s=0 端为头部，使 fPCA
    特征空间可对齐）；标注 CSV 缺该列时默认 1（不翻转）。
    """
    if os.path.isdir(csv_path):
        csvs = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
        if csvs:
            csv_path = os.path.join(csv_path, csvs[0])
        else:
            raise FileNotFoundError(f'目录 {csv_path} 中没有 CSV 文件')

    data = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            label = int(row['label'])
            if label >= 0:
                hf = int(row.get('head_is_first', 1) or 1)
                data.append((row['filename'], label, hf))
    return data


def stratified_sample(label_data, max_samples, seed=42):
    """分层抽样：保持类别比例，总数不超过 max_samples（加速小样本训练）。"""
    if max_samples <= 0 or len(label_data) <= max_samples:
        return label_data

    by_label = defaultdict(list)
    for item in label_data:
        by_label[item[1]].append(item)

    rng = np.random.RandomState(seed)
    sampled = []
    for label, items in by_label.items():
        n = max(1, round(max_samples * len(items) / len(label_data)))
        n = min(n, len(items))
        idx = rng.choice(len(items), n, replace=False)
        sampled.extend([items[i] for i in idx])

    if len(sampled) > max_samples:
        rng.shuffle(sampled)
        sampled = sampled[:max_samples]
    return sampled


# ============================================================
# 2. 切线角函数 θ(s)（Eigenworms 的输入）
# ============================================================
def compute_tangent_angle_profile(points, n_samples=100):
    """计算沿归一化弧长的切线角累积函数 θ(s)。

    Eigenworms（Stephens 2008）对角度函数 θ(s) 做 PCA；本文 SBPD 对
    曲率函数 κ(s) 做 fPCA。曲率是角度的空间导数（κ=dθ/ds），对局部
    尖锐弯曲更敏感，这是 SBPD 优于 eigenworms 的本质原因。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 10:
        return np.zeros(n_samples)

    # 每个骨架点的切线角（中心差分，端点为单侧差分）——共 n 个点
    tang = np.zeros(n)
    for i in range(1, n - 1):
        dx = pts[i + 1, 0] - pts[i - 1, 0]
        dy = pts[i + 1, 1] - pts[i - 1, 1]
        tang[i] = np.arctan2(dy, dx)
    tang[0] = np.arctan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0])
    tang[-1] = np.arctan2(pts[-1, 1] - pts[-2, 1], pts[-1, 0] - pts[-2, 0])

    # 累积角度（展开 ±π 跳变）——n 个点
    dtheta = np.diff(tang)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    theta = np.concatenate([[tang[0]], tang[0] + np.cumsum(dtheta)])

    # 弧长归一化（与 theta 同为 n 个点）
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0], np.cumsum(seg_len)])
    total = cumulative[-1]
    if total < 1e-6:
        return np.zeros(n_samples)
    s = cumulative / total

    # 样条插值到 n_samples（与 compute_curvature_profile 同款归一化）
    from scipy.interpolate import splrep, splev
    try:
        tck = splrep(s, theta, k=3, s=len(theta) * 0.1)
        return splev(np.linspace(0, 1, n_samples), tck)
    except Exception:
        return np.interp(np.linspace(0, 1, n_samples), s, theta)


def compute_headtail_score(points, W=11):
    """头尾方向得分（SBPD+头尾的第 4 维，连续值）。

    正值 → 骨架起点更像头；负值 → 骨架末点更像头。
    采用论文曲率方差加权：端点邻域(W=11)曲率方差 ×(1+平均曲率)。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < W:
        return 0.0
    curv = pointwise_curvature(pts)

    def score(seg):
        return np.var(seg) * (1.0 + np.mean(seg))

    return score(curv[:W]) - score(curv[-W:])


# ============================================================
# 3. 批量提取原始函数（κ、θ、BBI、头尾得分）—— 多进程
# ============================================================
def _extract_one(task):
    """多进程 worker：提取单张图的 SBPD 原始函数。
    返回 (kappa, theta, bbi, ht) 或 None（读取失败/骨架无效/曲率全零）。
    顶层定义以便 multiprocessing 序列化。

    方向归一化（2026-10-22）：head_is_first==0 时把骨架点序反转，
    保证 κ(s)/θ(s) 的 s=0 端始终是头部——fPCA/PCA 对输入向量按位
    对应，方向混排会把同一姿态学成镜像两团、糊掉特征空间。"""
    filename, image_dir, hf = task
    try:
        img = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_COLOR)
        if img is None:
            return None
        skeleton = extract_skeleton(img)
        if skeleton is None or len(skeleton) < 10:
            return None
        if not hf:                      # head_is_first=0：points[-1] 才是头
            skeleton = skeleton[::-1]   # 反转点序，s=0 统一为头
        kappa = compute_curvature_profile(skeleton, n_samples=100)
        if np.sum(kappa) < 1e-6:
            return None
        theta = compute_tangent_angle_profile(skeleton, n_samples=100)
        bbi = float(np.mean(kappa))
        ht = compute_headtail_score(skeleton)
        return (kappa, theta, bbi, ht)
    except Exception:
        return None


def extract_raw_features(image_dir, label_data, cache_path='sbpd_raw_cache.pkl',
                        workers=None):
    """对每张标注图提取原始函数，供不同特征表示共用（多进程并行）。

    返回 dict：
      kappa : (n,100) 曲率函数 κ(s)
      theta : (n,100) 切线角函数 θ(s)
      bbi   : (n,)    体长归一化平均曲率
      ht    : (n,)    头尾方向得分
      y     : (n,)    标签

    workers：并行进程数（默认 min(16, CPU数)）；<=1 时单进程。
    """
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if len(cached['y']) == len(label_data):
            # 校验标签分布 + 头尾方向分布一致（阈值/定向变化 → 缓存失效）
            cached_counts = Counter(cached['y'].tolist())
            data_counts = Counter(l for _, l, _ in label_data)
            hf_sum = sum(hf for _, _, hf in label_data)
            if cached_counts == data_counts and \
                    cached.get('hf_sum') == hf_sum:
                print(f'发现缓存 {cache_path}（标签与方向分布匹配），直接加载...')
                return cached
            print(f'缓存标签/方向分布与当前不符（阈值或头尾定向可能已变），重新提取...')
        else:
            print(f'缓存数据量({len(cached["y"])})与标注量({len(label_data)})不符，重新提取...')

    if workers is None:
        workers = min(16, max(1, cpu_count() or 1))
    n = len(label_data)
    print(f'开始提取 {n} 张图片的 SBPD 原始函数（workers={workers}）...')

    tasks = [(filename, image_dir, hf) for filename, _, hf in label_data]
    feats = [None] * n       # 与 label_data 顺序对齐，跳过帧保持 None
    if workers > 1 and n > 1:
        with Pool(processes=workers) as pool:
            for idx, feat in enumerate(
                    pool.imap(_extract_one, tasks,
                              chunksize=max(1, n // (workers * 4))), 0):
                feats[idx] = feat
                if (idx + 1) % 2000 == 0 or idx + 1 == n:
                    sys.stdout.write(f'\r  提取 {idx+1}/{n} 张    ')
                    sys.stdout.flush()
    else:
        for idx, task in enumerate(tasks):
            feats[idx] = _extract_one(task)
            if (idx + 1) % 200 == 0:
                sys.stdout.write(f'\r  提取 {idx+1}/{n} 张    ')
                sys.stdout.flush()

    # 按 label_data 顺序收集有效结果（feats[i] 对应 label_data[i]，天然对齐）
    kappa_list, theta_list, bbi_list, ht_list, labels = [], [], [], [], []
    skipped = 0
    for (filename, label, _hf), feat in zip(label_data, feats):
        if feat is None:
            skipped += 1
            continue
        kappa, theta, bbi, ht = feat
        kappa_list.append(kappa)
        theta_list.append(theta)
        bbi_list.append(bbi)
        ht_list.append(ht)
        labels.append(label)

    print()
    print(f'提取完成：有效 {len(labels)}，跳过 {skipped}')

    raw = {
        'kappa': np.array(kappa_list),
        'theta': np.array(theta_list),
        'bbi': np.array(bbi_list).reshape(-1, 1),
        'ht': np.array(ht_list).reshape(-1, 1),
        'y': np.array(labels),
        'hf_sum': int(sum(hf for _, _, hf in label_data)),
        # 2026-11-16：记录有效帧的文件名（与行序对齐）。提取失败帧被跳过时
        # 行数 < 标注行数，下游 step9 需按文件名对齐而不是按行数假设对齐。
        'filenames': [fn for (fn, _l, _hf), feat in zip(label_data, feats)
                      if feat is not None],
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(raw, f)
    print(f'原始函数已缓存到 {cache_path}')
    return raw


# ============================================================
# 4. 按特征类型构建特征矩阵
# ============================================================
def build_feature_matrix(raw, feat_type, pca=None):
    """按特征类型从原始函数构建特征矩阵 X，返回 (X, pca)。

    bbi           : 单标量
    eigenworms    : 对 θ(s) 做 PCA，取 6 维
    sbpd          : 对 κ(s) 做 PCA（fPCA 离散等价），取 3 维
    sbpd_headtail : SBPD 3 维 + 头尾得分 1 维
    """
    if feat_type == 'bbi':
        return raw['bbi'], None

    if feat_type == 'eigenworms':
        n_comp = 6
        source = raw['theta']
        # 去掉直流分量（θ 的均值无判别意义），保留形状信息
        source = source - source.mean(axis=1, keepdims=True)
    elif feat_type in ('sbpd', 'sbpd_headtail'):
        n_comp = 3
        source = raw['kappa']
    else:
        raise ValueError(f'未知特征类型 {feat_type}')

    if pca is None:
        pca = PCA(n_components=n_comp)
        X = pca.fit_transform(source)
        if feat_type == 'eigenworms':
            print(f'  Eigenworms：θ(s) 前{n_comp}主成分解释方差 '
                  f'{sum(pca.explained_variance_ratio_):.1%}')
        else:
            print(f'  SBPD：κ(s) 前{n_comp}主成分解释方差 '
                  f'{sum(pca.explained_variance_ratio_):.1%}')
    else:
        X = pca.transform(source)

    if feat_type == 'sbpd_headtail':
        X = np.hstack([X, raw['ht']])
    return X, pca


# ============================================================
# 5. 训练单个特征类型的 SVM
# ============================================================
def train_svm(X, y, smote_ratio=0.5):
    """训练 SVM-RBF（网格搜索 + SMOTE + 类别权重），返回 (best_svm, y_test, y_pred, classes)。

    类别不平衡处理（2026-10-22 修正）：
      · 原用 SMOTE sampling_strategy='auto'（少数类过采样到多数类规模）。
        在旧四类体系下 Omega 仅占 2.4%、缠绕 1.6%，'auto' 会把它们放大
        几十倍到正弦的规模，决策区随之膨胀——实测测试集 21% 的正弦被
        误判为 Omega、Omega 精确率仅 0.11。现改为按比例过采样
        （默认少数类 = 多数类的 0.5），再由 class_weight='balanced'
        在损失层面补偿剩余不平衡，两者配合比单一 'auto' 更稳。
      · smote_ratio<=0 表示不做 SMOTE，仅靠 class_weight。
      · 多分类下 sampling_strategy 必须为 dict（float 仅二分类合法），
        已按多数类规模×ratio 逐类构造（2026-10-22 修复）。
    """
    # 先剔除样本数<2 的类（stratify 分层划分要求每类至少 2 个样本）
    counts = Counter(y.tolist())
    drop_classes = [c for c, n in counts.items() if n < 2]
    if drop_classes:
        keep = ~np.isin(y, drop_classes)
        X, y = X[keep], y[keep]
        print(f'  警告：类别 {sorted(drop_classes)} 样本<2，已从数据集剔除')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    if smote_ratio and smote_ratio > 0:
        counts = Counter(y_train.tolist())
        majority = max(counts.values())
        target = int(majority * smote_ratio)
        # 多分类下 SMOTE 的 sampling_strategy 必须为 dict（float 仅二分类可用）。
        # 仅对「现有样本数 < 目标」的少数类过采样至 majority*smote_ratio，
        # 多数类及已接近目标的类维持原样。
        strategy = {c: target for c, n in counts.items() if target > n}
        if strategy:
            min_class_size = min(counts.values())
            k_neighbors = max(1, min_class_size - 1)
            smote = SMOTE(sampling_strategy=strategy, random_state=42,
                          k_neighbors=k_neighbors)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
            print(f'  SMOTE: 少数类过采样至多数类的 {smote_ratio:.0%}'
                  f'（{len(y_train)} -> {len(y_train_bal)}）')
        else:
            X_train_bal, y_train_bal = X_train, y_train
            print('  SMOTE: 目标比例下无少数类需过采样（仅用 class_weight）')
    else:
        X_train_bal, y_train_bal = X_train, y_train
        print('  SMOTE: 关闭（仅用 class_weight=balanced）')

    # 网格搜索（论文：C∈{0.1,1,10,100}, γ∈{0.01,0.1,1}，5 折交叉验证）
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1]}
    grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid,
                        cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train_bal, y_train_bal)
    best_svm = grid.best_estimator_

    y_pred = best_svm.predict(X_test)
    classes = sorted(set(np.unique(y_test)) | set(np.unique(y_pred)))
    return best_svm, y_test, y_pred, classes


def per_class_f1(y_test, y_pred, classes):
    """返回 dict: class -> f1（用于表2）。"""
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    return dict(zip(classes, f1))


# ============================================================
# 6. 表2：四种特征对比
# ============================================================
def compare_features(raw, output_csv='table2_feature_comparison.csv'):
    """遍历四种特征表示，各自训练 SVM，输出表2（终端 + CSV）。"""
    from sklearn.metrics import accuracy_score, f1_score

    print('\n' + '=' * 78)
    print('表2 不同特征表示的分类性能对比')
    print('=' * 78)
    all_classes = sorted(set(raw['y']))
    header = (f'{"特征方法":<18s} {"维度":>4s} {"准确率":>8s} {"宏F1":>7s} '
              + ''.join(f'{LABEL_EN[c]:>11s}' for c in all_classes))
    print(header)
    print('-' * 78)

    best = {'f1': -1, 'name': None, 'svm': None, 'pca': None, 'y_test': None,
            'y_pred': None, 'classes': None}
    table_rows = []
    all_results = {}

    for ft in FEATURES:
        X, pca = build_feature_matrix(raw, ft)
        svm, y_test, y_pred, classes = train_svm(X, raw['y'], SMOTE_RATIO)
        all_results[ft] = {'svm': svm, 'pca': pca, 'y_test': y_test,
                           'y_pred': y_pred, 'classes': classes}
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, labels=classes, average='macro',
                            zero_division=0)
        f1_all = f1_score(y_test, y_pred, labels=all_classes, average=None,
                          zero_division=0)

        row_data = {'特征方法': FEATURE_CN[ft], '维度': FEATURE_DIM[ft],
                    '准确率': round(float(acc), 4), '宏F1': round(float(macro_f1), 4)}
        line = f'{FEATURE_CN[ft]:<18s} {FEATURE_DIM[ft]:>4d} {acc:>7.1%} {macro_f1:>7.2f} '
        for c, f1v in zip(all_classes, f1_all):
            row_data[LABEL_EN[c]] = round(float(f1v), 4)
            line += f'{f1v:>11.2f}'
        print(line)
        table_rows.append(row_data)

        if macro_f1 > best['f1']:
            best = {'f1': macro_f1, 'name': ft, 'svm': svm, 'pca': pca,
                    'y_test': y_test, 'y_pred': y_pred, 'classes': classes}

    # 表2 结果写 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['特征方法', '维度', '准确率', '宏F1'] + [LABEL_EN[c] for c in all_classes]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_data in table_rows:
            writer.writerow(row_data)

    print('-' * 78)
    print(f'最优特征表示：{FEATURE_CN[best["name"]]}（宏F1={best["f1"]:.2f}）')
    print(f'表2 已保存到 {output_csv}')
    return best, all_results


# ============================================================
# 7. 表3：混淆矩阵
# ============================================================
def print_confusion_matrix(y_test, y_pred, classes, output_csv='table3_confusion_matrix.csv'):
    print('\n' + '=' * 64)
    print('表3 SBPD+SVM 分类器混淆矩阵')
    print('=' * 64)
    names = [LABEL_EN[c] for c in classes]
    print(classification_report(y_test, y_pred, labels=classes, target_names=names))

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print('混淆矩阵（行=真实，列=预测）：')
    print(f'{"":>14s}' + ''.join(f'{n:>11s}' for n in names))
    for i, n in enumerate(names):
        print(f'{n:>12s}', end='')
        for j in range(len(classes)):
            print(f'{cm[i][j]:>11d}', end='')
        print()

    # 混淆矩阵写 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['真实\\预测'] + names)
        for i, n in enumerate(names):
            w.writerow([n] + [int(cm[i][j]) for j in range(len(classes))])
    print(f'表3 混淆矩阵已保存到 {output_csv}')
    return cm


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='第8步：SBPD+SVM 训练（表2/表3）')
    parser.add_argument('--labels', '-l', required=True, help='标注 CSV')
    parser.add_argument('--images', '-i', required=True, help='图片目录')
    parser.add_argument('--model', '-m', default='svm_model.pkl', help='模型输出路径')
    parser.add_argument('--feature', default='sbpd', choices=FEATURES,
                        help='最终保存模型的特征类型（默认 sbpd，与论文一致）')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='分层抽样小样本训练（0=全量；>0 时按类别比例抽 N 张，加速）')
    parser.add_argument('--out-dir', default='output',
                        help='表2/表3/抽样样本 CSV 的输出目录（默认 output/）')
    parser.add_argument('--workers', type=int, default=min(16, max(1, cpu_count() or 1)),
                        help='特征提取并行进程数（默认 min(16, CPU数)）')
    parser.add_argument('--smote-ratio', type=float, default=0.5,
                        help='SMOTE 过采样比例：少数类目标 = 多数类×该值'
                             '（默认 0.5；0=关闭 SMOTE 仅用 class_weight=balanced）')
    args = parser.parse_args()

    print('=' * 60)
    print('第8步：SBPD 特征提取 + SVM 分类器训练')
    print('=' * 60)

    os.makedirs(args.out_dir, exist_ok=True)

    global SMOTE_RATIO
    SMOTE_RATIO = args.smote_ratio

    label_data = load_labels(args.labels)
    print(f'读取到 {len(label_data)} 条有效标注（三类存活姿态，无死亡类）')


    # 分层抽样小样本（加速训练）
    if args.max_samples > 0 and len(label_data) > args.max_samples:
        label_data = stratified_sample(label_data, args.max_samples)
        print(f'已分层抽样到 {len(label_data)} 张（--max-samples {args.max_samples}）')
        # 保存抽样样本，供 step9 --labels 只对这批样本推理
        sample_csv = os.path.join(args.out_dir, f'sampled_labels_{len(label_data)}.csv')
        with open(sample_csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['filename', 'label', 'head_is_first'])
            w.writerows(label_data)
        print(f'抽样样本已保存到 {sample_csv}（供 step9 --labels 复用）')

    label_counts = Counter(l for _, l, _ in label_data)
    print('标注分布：')
    for lbl in sorted(label_counts):
        print(f'  [{lbl}] {LABEL_NAMES[lbl]}: {label_counts[lbl]} 张')

    # 提取原始函数（抽样时用独立缓存，避免与全量缓存冲突）
    cache_name = ('sbpd_raw_cache.pkl' if args.max_samples <= 0
                  else f'sbpd_raw_cache_{len(label_data)}.pkl')
    raw = extract_raw_features(args.images, label_data, cache_path=cache_name,
                               workers=args.workers)
    print(f'\n有效样本 {len(raw["y"])} 张，κ(s)/θ(s) 均 100 点等弧长采样')

    # 表2：四种特征对比（结果同时用于表3，避免重复训练）
    best, all_results = compare_features(
        raw, output_csv=os.path.join(args.out_dir, 'table2_feature_comparison.csv'))

    # 表3：复用 compare_features 已训练的结果，不再重复网格搜索
    final_ft = args.feature
    res = all_results[final_ft]
    svm, pca = res['svm'], res['pca']
    cm = print_confusion_matrix(
        res['y_test'], res['y_pred'], res['classes'],
        output_csv=os.path.join(args.out_dir, 'table3_confusion_matrix.csv'))

    # 保存模型 + 特征缓存（供 step9 复用 PCA）
    with open(args.model, 'wb') as f:
        pickle.dump({'svm': svm, 'pca': pca, 'feat_type': final_ft}, f)
    print(f'\n模型已保存到 {args.model}（特征类型：{FEATURE_CN[final_ft]}）')

    # 供 step9 使用的 features_cache.pkl（kappa 的 PCA）
    if final_ft in ('sbpd', 'sbpd_headtail'):
        with open('features_cache.pkl', 'wb') as f:
            pickle.dump({'pca': pca, 'feat_type': final_ft}, f)
        print('特征缓存已保存到 features_cache.pkl（供 step9 复用）')

    print('完成！下一步运行 step9_fingerprint.py 生成行为指纹')


if __name__ == '__main__':
    main()
