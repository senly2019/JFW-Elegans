#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""step19_theta_full.py — 全量 θ(s) 切线角函数提取（Eigenworms 全量化专用）

背景：
  kappa_cache.npz 只存了无符号曲率 κ(s)=|sinΔθ|（compute_curvature_profile
  天然返回 |κ|，canonicalize 再取 abs），带方向的 θ(s) 无法从缓存积分重构，
  故 Eigenworms 特征需从原图重新提取 θ。本脚本按 kappa_cache 的 filenames
  （同一有效帧集）重新读图、提取骨架、计算 θ(s)，并沿用
  cluster_atlas.canonicalize 的曲率质心规则（s_cm>0.5 则反转）决定骨架
  点序是否反转，使 θ 与缓存 κ 的"弯在前半段"规范朝向逐帧一致。
  θ 保持带符号（不取 abs），保留波动方向信息——与 step8 的 8k 协议一致
  （step8 仅做头尾归一化，不做镜像取 abs）。

设计要点（吸取 33h 白跑教训）：
  · 分片落盘：每 SHARD 帧写 shard_%04d.npy（float32）+ .done 标记，
    中断后重跑自动跳过已完成分片（断点续跑）。
  · 失败帧写 NaN 行；meta.json 记 valid mask 与校验结果。

用法（服务器，venv）：
  PY step19_theta_full.py --cache atlas_out_all/kappa_cache.npz \
       --root "$WORM_DATA_ROOT" --out atlas_out_all/theta_full --workers 16
  PY step19_theta_full.py --out atlas_out_all/theta_full --assemble
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from multiprocessing import Pool

import cv2

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from worm_labeler import parse_five_dims, extract_skeleton_full, \
    compute_curvature_profile

SHARD = 200_000          # 每分片帧数（约 80 MB float32）
N_SAMPLES = 100          # θ(s) 采样点数（与 kappa_cache 一致）

_cache = {}


def compute_tangent_angle_profile(points, n_samples=N_SAMPLES):
    """切线角累积函数 θ(s)——与 step8_train_svm.compute_tangent_angle_profile
    逐行一致（协议对齐），仅去掉无关注释。"""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 10:
        return np.zeros(n_samples)
    tang = np.zeros(n)
    for i in range(1, n - 1):
        dx = pts[i + 1, 0] - pts[i - 1, 0]
        dy = pts[i + 1, 1] - pts[i - 1, 1]
        tang[i] = np.arctan2(dy, dx)
    tang[0] = np.arctan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0])
    tang[-1] = np.arctan2(pts[-1, 1] - pts[-2, 1], pts[-1, 0] - pts[-2, 0])
    dtheta = np.diff(tang)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    theta = np.concatenate([[tang[0]], tang[0] + np.cumsum(dtheta)])
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0], np.cumsum(seg_len)])
    total = cumulative[-1]
    if total < 1e-6:
        return np.zeros(n_samples)
    s = cumulative / total
    from scipy.interpolate import splrep, splev
    try:
        tck = splrep(s, theta, k=3, s=len(theta) * 0.1)
        return splev(np.linspace(0, 1, n_samples), tck)
    except Exception:
        return np.interp(np.linspace(0, 1, n_samples), s, theta)


def _init_worker():
    pass


def _theta_worker(task):
    """单帧 θ 提取 worker：读图 -> 骨架 -> κ 质心规范朝向 -> θ(s)。
    返回 (i, theta or None)。顶层定义以便多进程序列化。"""
    i, abs_path = task
    try:
        img = cv2.imread(abs_path, cv2.IMREAD_COLOR)
        if img is None:
            return i, None
        skeleton = extract_skeleton_full(img)[0]
        if skeleton is None or len(skeleton) < 10:
            return i, None
        kappa = compute_curvature_profile(skeleton, N_SAMPLES)
        # 与 cluster_atlas.canonicalize 相同的朝向规则：|κ| 质心 > 0.5 则反转
        w = np.abs(np.asarray(kappa, dtype=np.float64))
        wsum = w.sum()
        if wsum >= 1e-9:
            s = np.linspace(0.0, 1.0, len(w))
            if float((s * w).sum() / wsum) > 0.5:
                skeleton = skeleton[::-1]
        theta = compute_tangent_angle_profile(skeleton, N_SAMPLES)
        if not np.all(np.isfinite(theta)):
            return i, None
        return i, theta.astype(np.float32)
    except Exception:
        return i, None


def run_extract(args):
    from wormpath import get_data_root
    root = args.root or get_data_root()
    print(f'[root] 数据根：{root}')

    z = np.load(args.cache, allow_pickle=True)
    filenames = [str(f) for f in z['filenames']]
    n = len(filenames)
    print(f'[cache] {args.cache}：{n:,} 帧')

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    # 起始自检：随机抽 N 帧重算 κ，与缓存逐帧比对（验证链路一致）
    # 注意：npz 的 z['kappa'] 每次下标访问都会完整解压整个数组，
    # 必须一次性物化后再逐帧索引（否则 N 帧 = N 次 1.6GB IO）。
    rng = np.random.default_rng(42)
    check_idx = rng.choice(n, size=min(args.check, n), replace=False)
    kappa_ref = z['kappa'] if len(check_idx) else None   # 物化（约 1.6 GB）
    ok = 0
    for i in check_idx:
        img = cv2.imread(os.path.join(root, filenames[i]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        try:
            sk = extract_skeleton_full(img)[0]
            if sk is None or len(sk) < 10:
                continue
            kap = np.asarray(compute_curvature_profile(sk, N_SAMPLES),
                             dtype=np.float64)
            kap = np.abs(kap)
            w = kap
            wsum = w.sum()
            if wsum >= 1e-9:
                s = np.linspace(0.0, 1.0, len(w))
                if float((s * w).sum() / wsum) > 0.5:
                    kap = kap[::-1]
            ref = np.asarray(kappa_ref[i], dtype=np.float64)
            if np.max(np.abs(kap - ref)) < 5e-3:
                ok += 1
        except Exception:
            continue
    print(f'[自检] κ 复算一致率：{ok}/{len(check_idx)}'
          f'（<80% 请中止排查链路差异）')
    del kappa_ref, z

    tasks_all = [(i, os.path.join(root, f)) for i, f in enumerate(filenames)]
    shards = [(k, tasks_all[k:k + SHARD])
              for k in range(0, n, SHARD)]

    t0 = time.time()
    n_done_frames = 0
    with Pool(args.workers) as pool:
        for si, tasks in shards:
            marker = os.path.join(outdir, f'shard_{si:05d}.done')
            path = os.path.join(outdir, f'shard_{si:05d}.npy')
            if os.path.exists(marker) and os.path.exists(path):
                print(f'[shard {si:05d}] 已存在，跳过')
                continue
            buf = np.full((len(tasks), N_SAMPLES), np.nan, dtype=np.float32)
            valid = np.zeros(len(tasks), dtype=bool)
            for i, th in pool.imap(_theta_worker, tasks, chunksize=64):
                if th is not None:
                    buf[i - tasks[0][0]] = th
                    valid[i - tasks[0][0]] = True
            np.save(path, buf)
            np.save(os.path.join(outdir, f'shard_{si:05d}_valid.npy'), valid)
            with open(marker, 'w') as fh:
                fh.write(f'{int(valid.sum())}/{len(tasks)}\n')
            n_done_frames += len(tasks)
            rate = n_done_frames / max(time.time() - t0, 1e-6)
            eta = (n - (si + len(tasks))) / max(rate, 1e-6) / 3600
            print(f'[shard {si:05d}] 有效 {int(valid.sum())}/{len(tasks)}'
                  f' | 累计 {si + len(tasks):,}/{n:,} 帧'
                  f' | {rate:.0f} 帧/s | 剩余约 {eta:.1f} h')

    # meta
    meta = {'n_frames': n, 'shard': SHARD, 'n_samples': N_SAMPLES,
            'cache': os.path.abspath(args.cache),
            'root': os.path.abspath(root),
            'protocol': 'theta via compute_tangent_angle_profile (step8-identical), '
                        'skeleton reversed by kappa-centroid rule (canonicalize-equivalent)'}
    with open(os.path.join(outdir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print('[done] 提取完成，可运行 --assemble 合并')


def run_assemble(args):
    outdir = args.out
    z = np.load(args.cache, allow_pickle=True)
    n = len(z['filenames'])
    theta = np.empty((n, N_SAMPLES), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for si in range(0, n, SHARD):
        p = os.path.join(outdir, f'shard_{si:05d}.npy')
        v = os.path.join(outdir, f'shard_{si:05d}_valid.npy')
        theta[si:si + SHARD] = np.load(p)
        valid[si:si + SHARD] = np.load(v)
    np.save(os.path.join(outdir, 'theta_full.npy'), theta)
    np.save(os.path.join(outdir, 'theta_valid.npy'), valid)
    print(f'[assemble] theta_full.npy {theta.shape}，'
          f'有效 {int(valid.sum()):,}/{n:,} 帧'
          f'（{100 * valid.mean():.2f}%）')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', default='atlas_out_all/kappa_cache.npz')
    ap.add_argument('--root', default='', help='数据根（默认 wormpath.get_data_root）')
    ap.add_argument('--out', default='atlas_out_all/theta_full')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--check', type=int, default=200, help='κ 复算自检帧数')
    ap.add_argument('--assemble', action='store_true', help='仅合并分片')
    args = ap.parse_args()
    if args.assemble:
        run_assemble(args)
    else:
        run_extract(args)


if __name__ == '__main__':
    main()
