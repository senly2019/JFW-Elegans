"""
线虫头尾判别（论文 2.2 节，表1）
=================================
实现三种头尾判别方法，用于为 SBPD 提供统一的空间方向基准
（保证每条线虫的 s=0 端为头部，使 fPCA 的特征空间可对齐）：

  1. 质心启发式 (Centroid heuristic)          —— 头 = 距身体质心更远的端点
  2. 宽度轮廓分析 (Width profile)              —— 【已弃用】头 = 较窄端；
      宽度测量依赖二值化掩膜质量，本数据二值化噪声大，判据无从谈可靠性
  3. 曲率方差加权
生产管线的头尾定向已迁移到 simulate_labels.worm_head_orientation：
  · 主证据：时序质心位移投票（头先行、尾跟随，约15帧窗内质心位移
    方向即头的朝向）——运动学证据，与单帧形状无关；
  · 一致性硬约束：端点近邻链式传播——同一只虫头尾位置不会突变，
    已定向帧逐帧把方向传给相邻帧；
  · 兜底：质心启发式（仅零运动虫）。

本脚本现仅用于：对保留的单帧方法做表1 口径的一致性评估（参考性质，
伪 GT 为"同虫时序多数一致性"，非人工标注准确率）。

注意：三类姿态判据（0正弦型/1Ω型/2笔直型）已全部去头尾化，
头尾方向只影响 step8/step9 的 κ(s) 方向归一化（fPCA 特征空间对齐），
判错不再污染行为标签。

评估：若提供 ground-truth CSV（列 filename,head_is_first，head_is_first=1
     表示骨架起点 points[0] 为头），则输出三种方法的准确率/精确率/召回率
     （即论文表1）；否则仅输出每帧的头尾判定结果，供 SBPD 方向归一化使用。

用法：
  # 只做头尾判定（无 ground-truth，输出 head_tail_labels.csv）
  python head_tail_discriminator.py -i D:/data/image/10mg/0116/B  -o head_tail_labels.csv --method curvature_variance

  # 带 ground-truth 评估三种方法（输出表1）
  python head_tail_discriminator.py -i D:/data/image/10mg/0116/B \
      --ground-truth head_tail_gt.csv -o head_tail_labels.csv --eval
"""

import os
import sys
import csv
import argparse
import numpy as np
import cv2
from collections import defaultdict

from worm_labeler import extract_skeleton_full, parse_five_dims


# ============================================================
# 0. 逐点曲率（头尾判别用，非重采样版）
# ============================================================
def pointwise_curvature(points, half_window=3):
    """对每个骨架点计算局部曲率（7 点窗口，叉积/模长近似）。

    返回与 points 等长的数组，端点附近无法取满窗口的点曲率置 0。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    curv = np.zeros(n)
    for i in range(half_window, n - half_window):
        v1 = pts[i] - pts[i - half_window]
        v2 = pts[i + half_window] - pts[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            curv[i] = abs(cross) / (n1 * n2)
    return curv


# ============================================================
# 1. 三种头尾判别方法
# ============================================================
def centroid_heuristic(points):
    """质心启发式：头部 = 距身体质心更远的那个端点。"""
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    d_head = np.linalg.norm(pts[0] - centroid)
    d_tail = np.linalg.norm(pts[-1] - centroid)
    # head_is_first = 1 表示 points[0] 是头
    return 1 if d_head >= d_tail else 0


def width_profile(points, mask):
    """宽度轮廓分析：比较两端邻域（前后各 20% 弧长）的平均身体宽度，
    较窄的一端判为头（线虫头部通常略窄）。"""
    pts = np.asarray(points, dtype=np.float64)
    mask_bin = (mask > 0).astype(np.uint8)
    # 距离变换：每个像素到最近背景(0)的距离 = 半宽
    dist = cv2.distanceTransform((mask_bin == 0).astype(np.uint8),
                                 cv2.DIST_L2, 3)
    H, W = dist.shape

    widths = []
    for p in pts:
        x = int(round(p[0]))
        y = int(round(p[1]))
        if 0 <= y < H and 0 <= x < W:
            widths.append(dist[y, x] * 2.0)   # 直径 = 半宽×2
        else:
            widths.append(0.0)
    widths = np.array(widths)

    n = len(pts)
    head_seg = int(n * 0.2)
    head_w = widths[:head_seg].mean() if head_seg > 0 else 0.0
    tail_w = widths[-head_seg:].mean() if head_seg > 0 else 0.0

    # 较窄一端为头；宽度为 0（提取失败）时退回质心启发式
    if head_w <= 0 or tail_w <= 0:
        return centroid_heuristic(points)
    return 1 if head_w <= tail_w else 0


def curvature_variance_weighted(points, W=11):
    """曲率方差加权（论文方法）：对每个骨架点用 W 窗口算局部曲率方差，
    比较两个端点邻域的方差加权得分，得分高者为头。

    得分 = 端点邻域曲率方差 × (1 + 端点邻域平均曲率)
    （方差主导，反映"曲率变异性"；均值项作轻微加权）
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < W:
        return centroid_heuristic(points)

    curv = pointwise_curvature(pts)

    def endpoint_score(seg):
        # seg 是端点邻域的曲率序列（长度 W）
        var = np.var(seg)
        mean = np.mean(seg)
        return var * (1.0 + mean)

    head_seg = curv[:W]      # points[0] 端邻域
    tail_seg = curv[-W:]     # points[-1] 端邻域

    s_head = endpoint_score(head_seg)
    s_tail = endpoint_score(tail_seg)

    return 1 if s_head >= s_tail else 0


METHODS = {
    'centroid': (centroid_heuristic, '质心启发式'),
    'width_profile': (width_profile, '宽度轮廓分析'),
    'curvature_variance': (curvature_variance_weighted, '曲率方差加权'),
}


# ============================================================
# 2. 单张判定
# ============================================================
def discriminate_single(image, method='curvature_variance'):
    """对单张图判定头尾。返回 (head_is_first, 骨架点数, 掩膜, 字段类型)
    骨架提取失败返回 (None, None, None, None)。"""
    points, mask_thresh, mask_final, field_type = extract_skeleton_full(image)
    if points is None or len(points) < 10:
        return None, None, None, None

    func, _ = METHODS[method]
    if method == 'width_profile':
        head_is_first = func(points, mask_final)
    else:
        head_is_first = func(points)

    return head_is_first, len(points), mask_final, field_type


# ============================================================
# 3. 评估（表1）
# ============================================================
def evaluate(ground_truth_path, image_dir, max_images=None):
    """对三种方法评估准确率/精确率/召回率。

    ground-truth CSV 格式：filename, head_is_first
      head_is_first=1 → points[0] 是头；0 → points[-1] 是头
    """
    gt = {}
    with open(ground_truth_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row['filename']] = int(row['head_is_first'])

    if max_images:
        gt = dict(list(gt.items())[:max_images])

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    results = {name: {'tp': 0, 'fp': 0, 'fn': 0, 'n': 0}
               for name in METHODS}

    for filename, truth in gt.items():
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        for name in METHODS:
            pred, npts, _, _ = discriminate_single(img, method=name)
            if pred is None:
                continue
            results[name]['n'] += 1
            # "positive" = points[0] 是头（head_is_first=1）
            if pred == truth:
                results[name]['tp'] += 1
            elif pred == 1 and truth == 0:
                results[name]['fp'] += 1
            elif pred == 0 and truth == 1:
                results[name]['fn'] += 1

    return results


def print_table(results, output_csv='table1_head_tail_comparison.csv'):
    """打印论文表1 格式的性能对比，并输出 CSV。"""
    print('\n' + '=' * 64)
    print('表1 头尾判别方法性能对比')
    print('=' * 64)
    print(f'{"方法":<20s} {"样本数":>6s} {"准确率":>8s} {"精确率":>8s} {"召回率":>8s}')
    print('-' * 64)

    rows = []
    for name in METHODS:
        _, cn = METHODS[name]
        r = results[name]
        n = r['n'] or 1
        acc = (r['tp']) / n
        prec = r['tp'] / (r['tp'] + r['fp']) if (r['tp'] + r['fp']) else 0.0
        rec = r['tp'] / (r['tp'] + r['fn']) if (r['tp'] + r['fn']) else 0.0
        print(f'{cn:<20s} {r["n"]:>6d} {acc:>7.1%} {prec:>7.1%} {rec:>7.1%}')
        rows.append([cn, r['n'], round(acc, 4), round(prec, 4), round(rec, 4)])
    print('-' * 64)

    # 写 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['方法', '样本数', '准确率', '精确率', '召回率'])
        w.writerows(rows)
    print(f'表1 已保存到 {output_csv}')


def generate_pseudo_gt(image_dir, max_worms=None, output='head_tail_gt.csv'):
    """用「同虫时序多数一致性」生成伪头尾 ground-truth。

    依据：同一只虫在视频中头尾方向基本稳定（不会频繁翻转）。故对每只虫
    逐帧用三种方法投票，取该虫所有帧所有票的多数方向作为统一头尾方向。
    这个时序先验比单帧判定更可靠，可作为表1 的伪 ground-truth。

    注意：这是「伪标注」，反映三种方法相对时序一致性的稳定度，不是论文
    里相对人工标注的 94.7%（后者必须人工标注才能复现）。
    """
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # 若 -i 指向 data 子目录，从绝对路径补全 group/date 前缀
    GROUPS = {'control', '5mg', '8mg', '10mg'}
    abs_segs = [p for p in os.path.abspath(image_dir).replace('\\', '/').split('/')
                if p and not p.endswith(':')]
    start = -1
    for i, s in enumerate(abs_segs):
        if s.lower() in GROUPS:
            start = i
            break
    prefix = abs_segs[start:] if start >= 0 else []

    # 按虫分组
    worm_frames = defaultdict(list)
    for root, dirs, fs in os.walk(image_dir):
        dirs.sort()
        for f in sorted(fs):
            if not f.lower().endswith(exts):
                continue
            rel = os.path.relpath(os.path.join(root, f), image_dir).replace('\\', '/')
            full_rel = '/'.join(prefix + [p for p in rel.split('/') if p])
            g, d, r, w, fr = parse_five_dims(full_rel)
            worm_frames[f'{g}/{d}/{r}/{w}'.lower()].append(rel)

    worm_keys = sorted(worm_frames.keys())
    if max_worms:
        worm_keys = worm_keys[:max_worms]
    print(f'共 {len(worm_keys)} 只虫，生成伪头尾 ground-truth...')

    rows = []
    for wi, worm_key in enumerate(worm_keys, 1):
        votes_first = 0
        votes_total = 0
        for rel in worm_frames[worm_key]:
            img = cv2.imread(os.path.join(image_dir, rel), cv2.IMREAD_COLOR)
            if img is None:
                continue
            points, _, mask_final, _ = extract_skeleton_full(img)
            if points is None or len(points) < 10:
                continue
            votes = [
                centroid_heuristic(points),
                width_profile(points, mask_final),
                curvature_variance_weighted(points),
            ]
            votes_first += sum(votes)
            votes_total += 3
        if votes_total == 0:
            continue
        unified = 1 if votes_first >= votes_total / 2 else 0
        for rel in worm_frames[worm_key]:
            rows.append((rel, unified))
        if wi % 50 == 0:
            print(f'  [{wi}/{len(worm_keys)}]')

    with open(output, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'head_is_first'])
        w.writerows(rows)
    print(f'伪 ground-truth 已保存到 {output}（{len(rows)} 帧）')
    return rows


# ============================================================
# 4. 主程序
# ============================================================
def main():
    ap = argparse.ArgumentParser(description='线虫头尾判别（表1）')
    ap.add_argument('--input', '-i', required=True, help='图片目录')
    ap.add_argument('--output', '-o', default='head_tail_labels.csv',
                    help='输出头尾判定 CSV')
    ap.add_argument('--method', default='curvature_variance',
                    choices=list(METHODS), help='判定方法（默认曲率方差加权）')
    ap.add_argument('--ground-truth', default=None,
                    help='可选 ground-truth CSV（filename,head_is_first）')
    ap.add_argument('--eval', action='store_true',
                    help='对三种方法做性能评估（需 --ground-truth）')
    ap.add_argument('--pseudo-gt', action='store_true',
                    help='用同虫时序一致性生成伪 ground-truth 并评估表1（无需人工标注）')
    ap.add_argument('--max-worms', type=int, default=None,
                    help='伪 ground-truth 最多处理的虫数（控制规模）')
    ap.add_argument('--max-images', type=int, default=None,
                    help='仅处理前 N 张（测试用）')
    ap.add_argument('--out-dir', default='output',
                    help='表1/伪 ground-truth CSV 的输出目录（默认 output/）')
    args = ap.parse_args()

    image_dir = os.path.abspath(args.input)
    os.makedirs(args.out_dir, exist_ok=True)

    # 伪 ground-truth 评估模式（无需人工标注）
    if args.pseudo_gt:
        gt_path = os.path.join(args.out_dir, 'head_tail_gt.csv')
        generate_pseudo_gt(image_dir, args.max_worms, gt_path)
        results = evaluate(gt_path, image_dir, args.max_images)
        print_table(results, output_csv=os.path.join(args.out_dir,
                                                     'table1_head_tail_comparison.csv'))
        return

    # 评估模式（人工 ground-truth）
    if args.eval:
        if not args.ground_truth:
            print('错误：--eval 需要 --ground-truth 或 --pseudo-gt')
            return
        results = evaluate(args.ground_truth, image_dir, args.max_images)
        print_table(results, output_csv=os.path.join(args.out_dir,
                                                     'table1_head_tail_comparison.csv'))
        return

    # 批量判定模式
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = []
    for root, dirs, fs in os.walk(image_dir):
        for f in fs:
            if f.lower().endswith(exts):
                files.append(os.path.relpath(os.path.join(root, f), image_dir))
    files.sort()
    if args.max_images:
        files = files[:args.max_images]

    print(f'共 {len(files)} 张图片，方法：{METHODS[args.method][1]}')
    with open(args.output, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'head_is_first', 'n_points'])
        ok = 0
        for idx, rel in enumerate(files, 1):
            img = cv2.imread(os.path.join(image_dir, rel), cv2.IMREAD_COLOR)
            if img is None:
                continue
            head_is_first, npts, _, _ = discriminate_single(img, args.method)
            if head_is_first is None:
                continue
            writer.writerow([rel.replace('\\', '/'), head_is_first, npts])
            ok += 1
            if idx % 200 == 0:
                print(f'  [{idx}/{len(files)}] 已判定 {ok} 张')

    print(f'完成：{ok} 张有效判定 -> {args.output}')
    print('head_is_first=1 表示骨架起点为头，0 表示骨架末点为头')


if __name__ == '__main__':
    main()
