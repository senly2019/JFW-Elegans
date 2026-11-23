# -*- coding: utf-8 -*-
r"""extract_bdi_v2_feats.py — 为 426 项共识提虫级新特征（BDI v2 重造用）。

背景（2026-11-13）：共识分 vs 帧级 mean_k 的 Spearman 只有 0.355（历史
"0.33"的口径），但 vs 虫级校准 BDI（bdi_pre，10 帧均值过单调校准）是
0.591——一半增益来自消噪。要再往上拉必须换构念：mean|κ|（累积曲率）
不是人眼"最深弯曲"的单调函数（"小弯钩"：整体直+局部深弯 → mean_k 被
摊薄，人却看局部深弯打分）。

新特征族（全部虫级 10 帧均值，与 bdi_pre 同口径才公平）：
  mean_k   平均|κ|（= step8 bbi，BDI 原值；校准是它的单调变换，不携信息）
  sinBBI   mean|sin(Δθ)|——体轴折角口径（工作记忆 DEAD_BBI=0.16 同族）；
           小角度与 κ 近似，大角度 sin 饱和，对"折"比对"弯"敏感
  e_front / e_mid / e_back  |κ(s)| 前/中/后三份能量（中段占比高=中段弯）
  mid_dom  e_mid / (e_front + e_back)——中段主导度（历史方案单独用会
           误捕，作特征之一）
  max_area argmax|κ| 邻域 ±5 点的平均|κ|——"最深弯曲"的局部化度量，
           正对"小弯钩"：整体摊薄但局部深
  p99_k    更极端分位
  sklen    骨架点数（体长代理；短团=盘成一团）
  lap_var  拉普拉斯方差（画质对照，预期与共识无关）

输出：annotation_v3/retest/bdi_v2_feats.csv（item_id + 各特征虫级均值）
     annotation_v3/retest/bdi_v2_feats_raw.csv（虫级 + n_frames，审计用）

用法：python extract_bdi_v2_feats.py [--workers N] [--n-frames 10]
多进程；断点缓存（虫键已有记录则跳过，重跑只补缺）。
"""
import argparse
import csv
import glob
import os
import re
import sys
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A3 = os.path.join(HERE, 'annotation_v3')
OUT_CSV = os.path.join(A3, 'retest', 'bdi_v2_feats.csv')
OUT_RAW = os.path.join(A3, 'retest', 'bdi_v2_feats_raw.csv')

_REL = re.compile(r'^([^/]+)/(\d{4})/([A-E])/(worm\d+)_(\d+)\.png$')
N_SAMP = 100          # 与 compute_curvature_profile 默认一致
AREA_HALF = 5         # max_area 邻域半宽（点数）


def worm_key(rel):
    m = _REL.match((rel or '').replace('\\', '/'))
    return m.groups()[:4] if m else None


def _images_root():
    for p in ('E:/data_new/images', os.path.join(A3, 'images'),
              'D:/data_new/images'):
        if os.path.isdir(p) and any(
                os.path.isdir(os.path.join(p, g)) for g in
                ('control', '5mg', '8mg', '10mg')):
            return p
    raise RuntimeError('找不到图片根（E:/data_new/images 等）')


def _sin_bbi(skel):
    """体轴折角 sinBBI = mean|sin(Δθ)|（工作记忆 DEAD_BBI=0.16 同族）。

    与 mean|κ| 的构念差：κ=dθ/ds 是每单位弧长的折角（局部尖锐敏感），
    sinBBI 是每步折角的正弦平均（大角度饱和）。对"折"（Ω 折叠急转）
    比对"弯"（大弧）更敏感——正对"小弯钩"假说：整体直+局部深折。
    自包含实现（step8 的 compute_tangent_angle_profile 返回平滑累积角，
    diff 会抹平局部尖锐；这里直接用原始切线角序列）。
    """
    pts = np.asarray(skel, dtype=np.float64)
    n = len(pts)
    if n < 10:
        return None
    tang = np.zeros(n)
    for i in range(1, n - 1):
        tang[i] = np.arctan2(pts[i + 1, 1] - pts[i - 1, 1],
                             pts[i + 1, 0] - pts[i - 1, 0])
    tang[0] = np.arctan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0])
    tang[-1] = np.arctan2(pts[-1, 1] - pts[-2, 1], pts[-1, 0] - pts[-2, 0])
    d = np.diff(tang)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float(np.mean(np.abs(np.sin(d))))


def _frame_feats(path):
    """单帧全部特征；失败返回 None。链路与 bdi_calib._frame_bbi 一致。"""
    try:
        from worm_labeler import extract_skeleton, compute_curvature_profile
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        skel = extract_skeleton(img)
        if skel is None or len(skel) < 10:
            return None
        kappa = np.abs(compute_curvature_profile(skel, n_samples=N_SAMP))
        if kappa.sum() <= 1e-6:
            return None
        sin_bbi = _sin_bbi(skel)
        if sin_bbi is None:
            return None
        t = N_SAMP // 3
        e_f = float(kappa[:t].mean())
        e_m = float(kappa[t:2 * t].mean())
        e_b = float(kappa[2 * t:].mean())
        imax = int(np.argmax(kappa))
        lo, hi = max(0, imax - AREA_HALF), min(N_SAMP, imax + AREA_HALF + 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return dict(
            mean_k=float(kappa.mean()),
            sin_bbi=sin_bbi,
            e_front=e_f, e_mid=e_m, e_back=e_b,
            mid_dom=(e_m / (e_f + e_b)) if (e_f + e_b) > 1e-9 else np.nan,
            max_area=float(kappa[lo:hi].mean()),
            p99_k=float(np.percentile(kappa, 99)),
            max_k=float(kappa.max()),
            sklen=float(len(skel)),
            lap_var=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        )
    except Exception:
        return None


def _worm_feats(task):
    """一只虫的 10 帧等距采样 → 逐特征均值（帧间失败跳过）。"""
    g, d, r, w, root = task
    wdir = os.path.join(root, g, d, r)
    frames = sorted(glob.glob(os.path.join(wdir, f'{w}_*.png')))
    if len(frames) < 5:
        return (g, d, r, w), None, 0
    idx = np.linspace(0, len(frames) - 1, 10).round().astype(int)
    vals = [v for v in (_frame_feats(frames[i]) for i in idx)
            if v is not None]
    if len(vals) < 5:
        return (g, d, r, w), None, len(vals)
    out = {k: float(np.nanmean([v[k] for v in vals]))
           for k in vals[0]}
    return (g, d, r, w), out, len(vals)


def main():
    ap = argparse.ArgumentParser(description='提取 BDI v2 虫级新特征')
    ap.add_argument('--workers', type=int, default=0,
                    help='进程数（0=min(12, CPU)）')
    ap.add_argument('--n-frames', type=int, default=10)
    args = ap.parse_args()

    root = _images_root()
    print(f'[图片根] {root}')

    hm = list(csv.DictReader(open(
        os.path.join(A3, 'pre', 'hook_metrics.csv'), encoding='utf-8-sig')))
    cons = {r['item_id']: r for r in csv.DictReader(open(
        os.path.join(A3, 'retest', 'consensus_426.csv'),
        encoding='utf-8-sig'))}
    worms = []
    for r in hm:
        iid = r['item_id']
        if iid not in cons:
            continue
        k = worm_key(r['rel_path'])
        if k:
            worms.append((iid, k))
    print(f'[任务] {len(worms)} 项 → {len({k for _, k in worms})} 只虫')

    # 断点缓存
    done = {}
    if os.path.exists(OUT_RAW):
        for r in csv.DictReader(open(OUT_RAW, encoding='utf-8-sig')):
            if r.get('item_id') and r.get('mean_k') not in ('', None):
                done[r['item_id']] = r
        print(f'[缓存] 已有 {len(done)} 项，只补缺')

    todo = [(iid, k) for iid, k in worms if iid not in done]
    print(f'[待提取] {len(todo)} 项')

    n_workers = args.workers or min(12, max(1, os.cpu_count() or 1))
    if todo:
        tasks = [(k[0], k[1], k[2], k[3], root) for _, k in todo]
        with Pool(n_workers) as pool:
            for (iid, _), (key, feats, nf) in zip(
                    todo, pool.imap(_worm_feats, tasks, chunksize=4)):
                if feats is None:
                    print(f'  [跳过] {iid}：有效帧 {nf}/10（不足 5）',
                          file=sys.stderr)
                    continue
                done[iid] = dict(item_id=iid,
                                 rel_path=f'{key[0]}/{key[1]}/{key[2]}/'
                                          f'{key[3]}',
                                 n_frames=nf, **feats)
        print(f'[提取完成] {len(done)} 项有特征')

    # 虫级表（审计）
    fields = ['item_id', 'rel_path', 'n_frames', 'mean_k', 'sin_bbi',
              'e_front', 'e_mid', 'e_back', 'mid_dom', 'max_area', 'p99_k',
              'max_k', 'sklen', 'lap_var']
    with open(OUT_RAW, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for iid, _ in worms:
            if iid in done:
                w.writerow(done[iid])
    print(f'[写出] {OUT_RAW}')

    # 合并共识 → 分析表
    hm_by = {r['item_id']: r for r in hm}
    out_rows = []
    for iid, k in worms:
        if iid not in done or iid not in cons:
            continue
        row = dict(done[iid])
        row['consensus'] = cons[iid]['consensus']
        row['bdi_pre'] = hm_by[iid].get('bdi_pre', '')
        row['seg'] = hm_by[iid].get('seg', '') if 'seg' in hm_by[iid] else ''
        out_rows.append(row)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields + ['consensus', 'bdi_pre'],
                           extrasaction='ignore')
        w.writeheader()
        w.writerows(out_rows)
    print(f'[写出] {OUT_CSV}（{len(out_rows)} 行）')


if __name__ == '__main__':
    main()
