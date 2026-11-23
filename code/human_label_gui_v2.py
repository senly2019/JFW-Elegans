"""human_label_gui_v2.py — 双标注者批次标注器（支持隐藏重复项与新编码）。

与旧版 human_label_gui.py 的差别
--------------------------------
1. 以 **item_id** 为键，而不是文件名 —— 批次里有 60 张隐藏重复项（R001…），
   它们与某张唯一项（F001…）指向同一张图，用文件名作键会互相覆盖。
2. 支持五档编码：1=笔直 2=Ω型 3=正弦型 0=图像不可用 9=能看清但无法归类。
   0 与 9 是研究数据（"看不清"与"卡在两类之间"性质不同），不要混用。
3. 记录每张的作答用时（seconds），用于识别"秒点"。
4. 断点续标：随时退出，重新打开从上次位置继续。

界面：一屏 5×5=25 张，黄框=当前光标，绿框=已标。
界面文字全英文（cv2 不支持中文渲染），规则说明见 `标注规则_给标注者.md`。

按键
----
  1 / 2 / 3   笔直 / Ω 型 / 正弦型
  0           图像不可用（模糊、裁断、杂质、空图）
  9           能看清但无法归类
  WASD 移动光标            + / -   放大/缩小
  n / 空格    下一页        b       上一页
  u           撤销当前页最后一次标注
  q / ESC     退出（自动保存）

用法（ele 环境）
--------------
  # 标注者 A
  python human_label_gui_v2.py -b annotation_v2/items_A.csv ^
      -i $WORM_DATA_ROOT -o annotation_v2/labels_A.csv

  # 标注者 B
  python human_label_gui_v2.py -b annotation_v2/items_B.csv ^
      -i $WORM_DATA_ROOT -o annotation_v2/labels_B.csv
"""
import os
import csv
import time
import argparse

import cv2
import numpy as np

CLASS_CODES = {1: 'STR', 2: 'OMG', 3: 'SIN', 0: 'BAD', 9: 'AMB'}
BATCH_W, BATCH_H = 5, 5
MAX_WIN_W, MAX_WIN_H = 1480, 940
HEADER_H, FOOTER_H = 84, 54


def load_items(path):
    """返回 [(item_id, rel_path)]；兼容旧批次（只有 filename 列）。"""
    items = []
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            iid = (r.get('item_id') or '').strip()
            rel = (r.get('rel_path') or r.get('filename') or '').strip()
            if not rel:
                continue
            items.append((iid or rel, rel.replace('\\', '/')))
    return items


def build_window(page, labels, secs, cursor, zoom, imgs,
                 page_idx, n_pages, n_done, n_all):
    avail_w = MAX_WIN_W // BATCH_W
    avail_h = (MAX_WIN_H - HEADER_H - FOOTER_H) // BATCH_H
    base_h, base_w = 90, 80
    z = max(1, min(zoom, avail_w // base_w, avail_h // base_h))
    iw, ih = base_w * z, base_h * z
    cw, ch = iw + 8, ih + 34
    W, H = cw * BATCH_W, HEADER_H + ch * BATCH_H + FOOTER_H
    canvas = np.full((H, W, 3), 250, dtype=np.uint8)

    cv2.putText(canvas, 'HUMAN POSTURE LABELING v2  -  BLIND (labels hidden)',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (40, 40, 40), 1,
                cv2.LINE_AA)
    cv2.putText(canvas, f'Page {page_idx + 1}/{n_pages}    '
                        f'labeled {n_done}/{n_all}',
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1,
                cv2.LINE_AA)
    cv2.putText(canvas,
                '1=Straight  2=Omega(one deep bend)  3=Sinusoidal(2+ waves)  '
                '0=Unusable  9=Ambiguous',
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 160), 1,
                cv2.LINE_AA)

    for i, (iid, _rel) in enumerate(page):
        r, c = divmod(i, BATCH_W)
        x0, y0 = c * cw, HEADER_H + r * ch
        img = imgs[i]
        if img is not None and getattr(img, 'size', 0):
            canvas[y0:y0 + ih, x0:x0 + iw] = cv2.resize(
                img, (iw, ih), interpolation=cv2.INTER_NEAREST)
        code = labels.get(iid)
        if code is None:
            txt, col = '-', (150, 150, 150)
        else:
            txt = CLASS_CODES.get(code, '?')
            col = (60, 160, 60) if code in (1, 2, 3) else (40, 40, 190)
        cv2.putText(canvas, f'{i + 1}', (x0 + 3, y0 + ih + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (165, 165, 165), 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, txt, (x0 + iw - 32, y0 + ih + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        if i == cursor:
            cv2.rectangle(canvas, (x0, y0), (x0 + cw - 2, y0 + ch - 2),
                          (0, 180, 255), 2)
        elif code is not None:
            cv2.rectangle(canvas, (x0, y0), (x0 + cw - 2, y0 + ch - 2),
                          (120, 200, 120), 1)
        else:
            cv2.rectangle(canvas, (x0, y0), (x0 + cw - 2, y0 + ch - 2),
                          (205, 205, 205), 1)

    yf = H - FOOTER_H + 30
    cv2.putText(canvas,
                '[WASD] move   [+/-] zoom   [n] next   [b] back   '
                '[u] undo   [q] quit (auto-save)',
                (10, yf), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (70, 70, 70), 1,
                cv2.LINE_AA)
    return canvas, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--batch', required=True, help='items_A.csv / items_B.csv')
    ap.add_argument('-i', '--input', required=True, help='图片根目录')
    ap.add_argument('-o', '--out', required=True, help='输出 labels_*.csv')
    ap.add_argument('--zoom', type=int, default=2)
    args = ap.parse_args()

    items = load_items(args.batch)
    total = len(items)
    n_unique = len({iid for iid, _ in items if iid.startswith('F')})
    print(f'待标注 {total} 条（其中隐藏重复约 {total - n_unique if n_unique else 0} 条）')
    print('规则：1=笔直  2=Ω型  3=正弦型  0=图像不可用  9=无法归类')

    labels, secs = {}, {}
    if os.path.exists(args.out):
        with open(args.out, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                iid = (r.get('item_id') or r.get('filename') or '').strip()
                raw = (r.get('label_code') or '').strip()
                if not iid or not raw:
                    continue
                try:
                    labels[iid] = int(float(raw))
                except ValueError:
                    continue
                try:
                    secs[iid] = float(r.get('seconds') or 0)
                except ValueError:
                    secs[iid] = 0.0
        print(f'已有 {len(labels)} 条标注，继续')

    def flush():
        tmp = args.out + '.tmp'
        with open(tmp, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['item_id', 'rel_path', 'label_code', 'label_name',
                        'seconds'])
            rel_of = dict(items)
            for iid, code in labels.items():
                w.writerow([iid, rel_of.get(iid, ''), code,
                            CLASS_CODES.get(code, ''), f'{secs.get(iid, 0):.1f}'])
        os.replace(tmp, args.out)

    # ---- 图片缓存（按需读取，只缓存当前页）----
    page = 0
    cursor = 0
    zoom = args.zoom
    per_page = BATCH_W * BATCH_H
    n_pages = max(1, (total + per_page - 1) // per_page)
    cache = {}
    t_enter = time.time()

    def page_items(p):
        s = p * per_page
        return items[s:s + per_page]

    while True:
        cur_page = page_items(page)
        imgs = []
        for iid, rel in cur_page:
            p = os.path.join(args.input, rel)
            if p not in cache:
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    img = np.zeros((90, 80, 3), dtype=np.uint8)
                cache[p] = img
            imgs.append(cache[p])
        canvas, zoom = build_window(cur_page, labels, secs, cursor, zoom, imgs,
                                   page, n_pages, len(labels), total)
        cv2.imshow('Human Posture Labeling v2', canvas)
        k = cv2.waitKey(20) & 0xFF

        if k in (27, ord('q')):
            break

        # ---- 标注 ----
        if k in (ord('1'), ord('2'), ord('3'), ord('0'), ord('9')):
            if cursor < len(cur_page):
                iid = cur_page[cursor][0]
                code = int(chr(k))
                labels[iid] = code
                secs[iid] = max(0.0, time.time() - t_enter)
                flush()
                nxt = None
                for j in range(cursor + 1, len(cur_page)):
                    if cur_page[j][0] not in labels:
                        nxt = j
                        break
                if nxt is None:
                    for j in range(0, cursor):
                        if cur_page[j][0] not in labels:
                            nxt = j
                            break
                if nxt is not None:
                    cursor = nxt
                t_enter = time.time()
            continue

        # ---- 翻页 ----
        if k in (ord('n'), 32):
            if page + 1 < n_pages:
                page += 1
                cursor = 0
                cache.clear()
                t_enter = time.time()
            continue
        if k == ord('b'):
            if page > 0:
                page -= 1
                cursor = 0
                cache.clear()
                t_enter = time.time()
            continue

        # ---- 缩放 / 撤销 ----
        if k in (ord('+'), ord('=')):
            zoom = min(zoom + 1, 6)
            continue
        if k in (ord('-'), ord('_')):
            zoom = max(zoom - 1, 1)
            continue
        if k == ord('u'):
            for j in range(len(cur_page) - 1, -1, -1):
                iid = cur_page[j][0]
                if iid in labels:
                    labels.pop(iid, None)
                    secs.pop(iid, None)
                    flush()
                    cursor = j
                    break
            continue

        # ---- 光标移动（WASD；方向键码值跨平台不一致，不启用）----
        moved = False
        if k == ord('w'):
            cursor = max(0, cursor - BATCH_W)
            moved = True
        elif k == ord('s'):
            cursor = min(len(cur_page) - 1, cursor + BATCH_W)
            moved = True
        elif k == ord('a'):
            cursor = max(0, cursor - 1)
            moved = True
        elif k == ord('d'):
            cursor = min(len(cur_page) - 1, cursor + 1)
            moved = True
        if moved:
            t_enter = time.time()

    flush()
    cv2.destroyAllWindows()
    done = sum(1 for v in labels.values() if v in (1, 2, 3))
    bad = sum(1 for v in labels.values() if v == 0)
    amb = sum(1 for v in labels.values() if v == 9)
    print(f'\n完成 {len(labels)}/{total} 条：三类 {done}，不可用 {bad}，无法归类 {amb}')
    print(f'已保存：{args.out}')
    if len(labels) < total:
        print('提示：还有未标注的条目，重新运行同一命令可继续。')


if __name__ == '__main__':
    main()
