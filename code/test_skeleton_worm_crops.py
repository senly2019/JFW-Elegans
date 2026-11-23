#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公开数据集 worm_crops 骨架提取效果测试。

对 /media/yczxserver/SSD-windows/wjf/worm_crops 的图片抽样，
用 worm_5class_pipeline 的 extract_skeleton_full（含 2%~98% 百分位对比度
拉伸修复版）逐张提取骨架，报告：
  · 成功率（总体 + 按子目录分层）
  · 失败原因分解（读图失败 / 阈值后无前景 / 掩膜被拒 / 骨架过短）
  · 骨架质量（点数、弧长像素、BBI = mean|κ(s)| 分布）
  · 图片尺寸与灰度动态范围（验证对比度拉伸是否必要/生效）
  · 失败与成功样本的诊断拼图（原图 | 阈值掩膜 | 最终掩膜）

用法（服务器）：
  cd /media/yczxserver/SSD-windows/wjf/worm_Sclass_pipeline_f
  /home/yczxserver/.virtualenvs/worm_5class_pipeline/bin/python \
      test_skeleton_worm_crops.py --n 300

输出：worm_crops_skeleton_test/ 下的 summary.txt、results.csv、诊断图。
"""
import argparse
import csv
import os
import random
import sys
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from worm_labeler import extract_skeleton_full, compute_curvature_profile  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PANELS = 12  # 失败诊断拼图份数


def scan_images(root):
    """递归扫描图片，返回 [(rel_path, top_dir), ...]。"""
    items = []
    root = os.path.abspath(root)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                top = rel.split(os.sep)[0] if os.sep in rel else "(root)"
                items.append((rel, top))
    return items


def stratified_sample(items, n):
    """按 top 目录分层抽样：每层轮转取，直至取满 n 张。"""
    random.seed(42)
    buckets = {}
    for rel, top in items:
        buckets.setdefault(top, []).append(rel)
    for b in buckets.values():
        random.shuffle(b)
    picked, names = [], list(buckets)
    i = 0
    while len(picked) < n and any(buckets[b] for b in names):
        b = names[i % len(names)]
        if buckets[b]:
            picked.append((buckets[b].pop(), b))
        i += 1
    return picked


def skeleton_arc_len(points):
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(d.sum())


def classify_reason(detail):
    """把 dbg['reason'] 归为简短原因类。"""
    if not detail:
        return ""
    if "多虫交叉" in detail:
        return "cross_2worm_width" if "宽度比" in detail else "cross_2worm_arm"
    if "三遍形态学" in detail:
        return "no_valid_candidate"
    if "中心线<5点" in detail:
        return "centerline<5"
    return "other"

def process_one(task):
    """worker：返回单张图片的测试记录 dict。"""
    data_root, rel = task
    abs_path = os.path.join(data_root, rel)
    rec = {"file": rel, "status": "", "reason_class": "", "reason_detail": "",
           "field": "", "w": 0, "h": 0, "p2": 0.0, "p98": 0.0,
           "dyn_range": 0.0, "n_points": 0, "skel_len": 0.0, "bbi": 0.0,
           "max_kappa": 0.0, "mask_fg_px": 0, "sel_thr": 0, "area_frac": 0.0,
           "n_junc": 0, "n_end": 0}

    img = cv2.imread(abs_path, cv2.IMREAD_COLOR)
    if img is None:
        rec["status"], rec["reason_class"] = "fail", "read_fail"
        return rec

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    p2, p98 = np.percentile(gray, (2, 98))
    rec.update(h=h, w=w, p2=float(p2), p98=float(p98),
               dyn_range=float(p98 - p2))

    points, mask_thresh, mask_final, field, dbg = extract_skeleton_full(
        img, return_debug=True)
    rec["field"] = str(field)
    rec["mask_fg_px"] = int((mask_thresh > 0).sum())
    rec["sel_thr"] = int(dbg.get("thresh", 0) or 0)
    rec["area_frac"] = round(float(dbg.get("area_frac", 0) or 0), 3)
    rec["n_junc"] = int(dbg.get("n_junctions", 0) or 0)
    rec["n_end"] = int(dbg.get("n_endpoints", 0) or 0)

    if points is None or len(points) < 10:
        detail = str(dbg.get("reason", ""))
        rec["reason_detail"] = detail
        rec["reason_class"] = classify_reason(detail)
        rec["status"] = "fail"
        rec["_debug"] = (rel, img, mask_thresh, mask_final)
        return rec

    kappa = compute_curvature_profile(points)
    rec.update(status="ok", reason_class="ok", reason_detail="",
               n_points=int(len(points)),
               skel_len=round(skeleton_arc_len(points), 1),
               bbi=round(float(np.mean(kappa)), 4),
               max_kappa=round(float(np.max(kappa)), 4))
    rec["_debug"] = (rel, img, mask_thresh, mask_final)
    return rec


def save_panels(records, out_dir, status, tag, limit):
    """把诊断拼图保存为 png：每张 = 原图 | 阈值掩膜 | 最终掩膜。"""
    sel = [r for r in records if r["status"] == status][:limit]
    if not sel:
        return
    from matplotlib import pyplot as plt
    cols = 3
    rows = len(sel)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2),
                             squeeze=False)
    for i, r in enumerate(sel):
        rel, img, mt, mf = r["_debug"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for j, (panel, t) in enumerate([(gray, "orig"), (mt, "mask_thresh"),
                                        (mf, "mask_final")]):
            ax = axes[i][j]
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(os.path.basename(rel)[:22], fontsize=5)
            if i == 0:
                ax.set_title(t, fontsize=7)
    fig.tight_layout(pad=0.4)
    out = os.path.join(out_dir, f"panels_{tag}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out} ({rows} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default="/media/yczxserver/SSD-windows/wjf/worm_crops")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "worm_crops_skeleton_test"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"data root: {args.data}")
    if not os.path.isdir(args.data):
        print("ERROR: 目录不存在")
        sys.exit(1)

    items = scan_images(args.data)
    print(f"total images found: {len(items)}")
    tops = {}
    for _, top in items:
        tops[top] = tops.get(top, 0) + 1
    print("top-level distribution (前 15):")
    for top, cnt in sorted(tops.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {top}: {cnt}")
    if len(tops) > 15:
        print(f"  ... 共 {len(tops)} 个顶层条目")

    sample = stratified_sample(items, args.n)
    print(f"sampled: {len(sample)} (stratified, seed=42)")

    tasks = [(args.data, rel) for rel, _ in sample]
    with Pool(args.workers) as pool:
        records = pool.map(process_one, tasks, chunksize=4)

    # ---- 汇总 ----
    ok = [r for r in records if r["status"] == "ok"]
    fail = [r for r in records if r["status"] == "fail"]
    lines = []
    lines.append("=== worm_crops 骨架提取测试 ===")
    lines.append(f"样本: {len(records)}  成功: {len(ok)}  失败: {len(fail)}"
                 f"  成功率: {len(ok) / max(len(records), 1):.1%}")

    # 图片尺寸与动态范围
    sizes = sorted({(r["w"], r["h"]) for r in records})
    dr = [r["dyn_range"] for r in records if r["w"]]
    lines.append(f"图片尺寸种类: {len(sizes)} 种，常见: "
                 + ", ".join(f"{w}x{h}" for w, h in sizes[:8]))
    lines.append(f"灰度动态范围 p98-p2: 中位 {np.median(dr):.0f}  "
                 f"p10 {np.percentile(dr, 10):.0f}  p90 {np.percentile(dr, 90):.0f}"
                 f"  (<35 表示必须依赖百分位拉伸)")

    # 分层成功率
    by_top = {}
    for r, (_, top) in zip(records, sample):
        by_top.setdefault(top, []).append(r)
    lines.append("--- 按子目录成功率 ---")
    for top in sorted(by_top, key=lambda t: -len(by_top[t]))[:15]:
        rs = by_top[top]
        o = sum(1 for r in rs if r["status"] == "ok")
        lines.append(f"  {top}: {o}/{len(rs)} = {o / len(rs):.1%}")

    # 失败原因
    lines.append("--- 失败原因分解 ---")
    reasons = {}
    for r in fail:
        reasons[r["reason_class"]] = reasons.get(r["reason_class"], 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k}: {v}")

    # 成功样本质量
    if ok:
        np_arr = [r["n_points"] for r in ok]
        sl = [r["skel_len"] for r in ok]
        bb = [r["bbi"] for r in ok]
        mk = [r["max_kappa"] for r in ok]
        lines.append("--- 成功样本骨架质量 (n=%d) ---" % len(ok))
        for name, arr in (("骨架点数", np_arr), ("骨架弧长 px", sl),
                          ("BBI=mean|κ|", bb), ("max_kappa", mk)):
            lines.append(
                f"  {name}: 中位 {np.median(arr):.1f}  "
                f"p10 {np.percentile(arr, 10):.1f}  p90 {np.percentile(arr, 90):.1f}")
        # 与本数据已知分布对照：BBI 噪声底 ~0.12，直形阈值 0.16
        lines.append("  参照: 本项目 BBI 噪声底≈0.12, 笔直阈值 0.16, "
                     "Ω 型 max_kappa≥0.35")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # CSV
    cols = ["file", "status", "reason_class", "reason_detail", "field",
            "w", "h", "p2", "p98", "dyn_range", "n_points", "skel_len",
            "bbi", "max_kappa", "mask_fg_px", "sel_thr", "area_frac",
            "n_junc", "n_end"]
    with open(os.path.join(args.out, "results.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(records)

    # 诊断拼图（只在本进程有 _debug 时，即非 Pool 场景；Pool 下由主进程重读）
    if ok or fail:
        # Pool 返回的 _debug 含 ndarray，可直接用
        save_panels(records, args.out, "ok", "ok", 6)
        if fail:
            save_panels(records, args.out, "fail", "fail", PANELS)

    print(f"\n输出目录: {args.out}")


if __name__ == "__main__":
    main()
