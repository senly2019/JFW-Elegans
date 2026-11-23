#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 worm_crops（train/test 扁平命名）重排成流水线要求的五维目录树。

源：  {root}/train/24_2_1_1_26_worm018.png
目标：{out}/train/0000/{row}/worm018_001.png
      - group = train|test（源一级目录）
      - date  = 0000（公开数据无时序）
      - row   = s + 源文件名 md5 前 8 位（每张源图唯一，保证 (group,date,row,worm)
                四元组唯一；直接用 wormNNN 会因 train/test 重号合并不同虫）
      - frame = 001（每虫单帧）

默认建符号链接（不复制数据）；Windows/无权限环境自动回退为复制。
幂等：已存在的链接跳过。输出 manifest CSV（五维路径 ↔ 源路径）。

用法（服务器）：
  python prep_wormcrops_5dim.py \
      --src /media/yczxserver/SSD-windows/wjf/worm_crops \
      --out /media/yczxserver/SSD-windows/wjf/worm_Sclass_pipeline_f/wormcrops_5dim
"""
import argparse
import csv
import hashlib
import os
import re
import sys

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLIT_DIRS = ("train", "test")


def iter_images(src):
    for split in SPLIT_DIRS:
        sdir = os.path.join(src, split)
        if not os.path.isdir(sdir):
            print(f"[警告] 缺少 {sdir}，跳过该 split")
            continue
        for dirpath, _, files in os.walk(sdir):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in IMG_EXT:
                    yield split, os.path.relpath(os.path.join(dirpath, fn), src)


def worm_index(stem):
    m = re.search(r"(?i)worm(\d+)", stem)
    return m.group(1) if m else re.sub(r"\D", "", stem)[-3:] or "000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force-copy", action="store_true",
                    help="不用符号链接，直接复制文件")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"ERROR: 源目录不存在 {args.src}")
        sys.exit(1)

    manifest = []
    made, skipped, failed = 0, 0, 0
    for split, rel in iter_images(args.src):
        stem = os.path.splitext(os.path.basename(rel))[0]
        row = "s" + hashlib.md5(rel.encode("utf-8")).hexdigest()[:8]
        worm = "worm" + worm_index(stem)
        five = os.path.join(args.out, split, "0000", row,
                            f"{worm}_001{os.path.splitext(rel)[1].lower()}")
        os.makedirs(os.path.dirname(five), exist_ok=True)
        src_abs = os.path.join(args.src, rel)
        if os.path.lexists(five):
            skipped += 1
        else:
            try:
                if args.force_copy:
                    raise OSError("forced copy")
                os.symlink(os.path.abspath(src_abs), five)
                made += 1
            except OSError:
                try:
                    import shutil
                    shutil.copyfile(src_abs, five)
                    made += 1
                except Exception as e:
                    print(f"[失败] {rel}: {e}")
                    failed += 1
                    continue
        manifest.append({"five_dim": os.path.relpath(five, args.out),
                         "src": rel, "split": split})

    man_csv = os.path.join(args.out, "manifest.csv")
    os.makedirs(args.out, exist_ok=True)
    with open(man_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["five_dim", "src", "split"])
        w.writeheader()
        w.writerows(manifest)

    print(f"完成：新建 {made}，已存在跳过 {skipped}，失败 {failed}")
    print(f"五维树: {args.out}")
    print(f"映射清单: {man_csv}（{len(manifest)} 条）")
    # 抽样自检：列 3 条
    for m in manifest[:3]:
        print(f"  例: {m['src']}  ->  {m['five_dim']}")


if __name__ == "__main__":
    main()
