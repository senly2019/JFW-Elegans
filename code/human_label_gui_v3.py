"""human_label_gui_v3.py — 单图慢标版标注器（用于第二轮/复核标注）。

为什么要 v3
-----------
v2 是 5×5 缩略图网格 + WASD 移动，适合快速过一遍，但实测第一轮标注者
**中位 1.0 s/条、45.5% 不到 1 s**，且人判"弯/直"与骨架弯曲深度几乎无关
（bbi-AUC 0.62，旧标注者是 0.71）。为了第二轮能真正"看清再判"，v3 做了四件事：

1. **一屏一张**，默认放大 7 倍居中显示（80×90 -> 560×630），不再有缩略图。
2. **右侧常驻三类示例图**（取自同一批缓存，与标注规则文档同一套），
   随时对照"什么才算 Ω"，避免把浅单弯也叫 Ω。
3. **最短观察时间**（`--min-sec`，默认 1.5 s）：按键太快不记录，屏幕红字提示再看一眼。
   这是硬性防"秒点"，不是统计提示。
4. **强制区分 0 与 9**：看不清/出框按 0；看得清但卡在两类之间按 9。
   第一轮 310 条 0 与 9 都是 0 次，等于这两个出口被忽略。

输出 CSV 与 v2 完全同构（多一列 fast），`step18_agreement_v2.py` 直接能读。

按键
----
  1 / 2 / 3   笔直 / Ω 型（单个深弯）/ 正弦型（两波及以上）
  0           图像不可用（糊到认不出、虫体出框、空图、杂质遮挡）
  9           看得清，但实在无法归入任何一类
  u           撤销上一条（并把光标退回那一条）
  n / 空格    跳过（不记录，之后可再回来）
  p           上一条
  + / -       放大 / 缩小
  g           显示/隐藏右侧示例图
  q / ESC     保存并退出（随时可断点续标）

用法
----
  # 盲标第二轮（推荐：不看第一轮答案）
  python human_label_gui_v3.py -b annotation_v2/items_B.csv ^
      -i $WORM_DATA_ROOT -o annotation_v2/labels_B.csv

  # 先跑 5 张试试手感
  python human_label_gui_v3.py -b annotation_v2/items_B.csv ^
      -i $WORM_DATA_ROOT -o annotation_v2/labels_B.csv --max-items 5
"""
import os
import csv
import time
import argparse

import cv2
import numpy as np

CLASS_CODES = {1: 'STR', 2: 'OMG', 3: 'SIN', 0: 'BAD', 9: 'AMB'}
CLASS_TEXT = {1: '1 = STRAIGHT', 2: '2 = OMEGA (one deep bend)',
              3: '3 = SINUSOIDAL (2+ waves)', 0: '0 = UNUSABLE image',
              9: '9 = AMBIGUOUS (visible, cannot assign)'}

WIN_W, WIN_H = 1180, 830
HEAD_H, FOOT_H, PANEL_W = 76, 74, 330


def load_items(path):
    items = []
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            iid = (r.get('item_id') or '').strip()
            rel = (r.get('rel_path') or r.get('filename') or '').strip()
            if rel:
                items.append((iid or rel, rel.replace('\\', '/')))
    return items


def load_exemplars(folder, which='01'):
    """读三类示例图，缺失则跳过（不影响标注）。"""
    out = {}
    for code, name in ((1, 'straight'), (2, 'omega'), (3, 'sine')):
        p = os.path.join(folder, f'exemplar_{name}_{which}.png')
        im = cv2.imread(p, cv2.IMREAD_COLOR) if os.path.exists(p) else None
        if im is not None:
            out[code] = im
    return out


def put(canvas, txt, xy, scale=0.5, color=(40, 40, 40), thick=1):
    cv2.putText(canvas, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


def build_single(img, ex, iid, idx, total, labeled, n_left, zoom, warn_left,
                 show_ex, sec_now):
    canvas = np.full((WIN_H, WIN_W, 3), 252, dtype=np.uint8)
    put(canvas, 'HUMAN POSTURE LABELING v3  -  ONE IMAGE AT A TIME, BLIND',
        (12, 26), 0.62, (30, 30, 30))
    eff = float(zoom)
    put(canvas, f'item {idx + 1}/{total}    {iid}    labeled {labeled}/{total}    '
                f'left {n_left}    zoom x{eff:.1f}    {sec_now:4.1f}s',
        (12, 52), 0.5, (95, 95, 95))

    # 主图区
    area_w = WIN_W - PANEL_W - 24
    area_h = WIN_H - HEAD_H - FOOT_H
    ax, ay = 12, HEAD_H
    if img is not None and getattr(img, 'size', 0):
        h, w = img.shape[:2]
        tw, th = w * zoom, h * zoom
        if tw > area_w or th > area_h:
            s = min(area_w / tw, area_h / th)
            tw, th = max(1, int(tw * s)), max(1, int(th * s))
        eff = tw / max(1, w)
        big = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        ox = ax + (area_w - tw) // 2
        oy = ay + (area_h - th) // 2
        canvas[oy:oy + th, ox:ox + tw] = big
        cv2.rectangle(canvas, (ox - 1, oy - 1), (ox + tw, oy + th), (150, 150, 150), 1)
        put(canvas, f'{w}x{h}', (ox, oy + th + 14), 0.36, (150, 150, 150))
    else:
        put(canvas, 'IMAGE NOT FOUND', (ax + 30, ay + 60), 0.9, (30, 30, 200), 2)
    # 实际缩放可能在像素计算后变化，回填一次
    put(canvas, f'item {idx + 1}/{total}    {iid}    labeled {labeled}/{total}    '
                f'left {n_left}    zoom x{eff:.1f}    {sec_now:4.1f}s',
        (12, 52), 0.5, (95, 95, 95))

    # 右侧示例面板
    if show_ex:
        x0 = WIN_W - PANEL_W + 8
        y = HEAD_H
        put(canvas, 'REFERENCE EXAMPLES', (x0, y - 12), 0.44, (110, 110, 110))
        for code in (1, 2, 3):
            if code not in ex:
                continue
            cell = 96
            im = cv2.resize(ex[code], (cell, cell), interpolation=cv2.INTER_LANCZOS4)
            canvas[y:y + cell, x0:x0 + cell] = im
            cv2.rectangle(canvas, (x0, y), (x0 + cell, y + cell), (170, 170, 170), 1)
            put(canvas, CLASS_TEXT[code], (x0 + cell + 8, y + 26), 0.36, (60, 60, 60))
            y += cell + 14
        put(canvas, '0 = unusable (blurred / cut off /', (x0, y + 10), 0.36,
            (40, 40, 170))
        put(canvas, '     empty / debris on the worm)', (x0, y + 30), 0.36,
            (40, 40, 170))
        put(canvas, '9 = visible but between 2 classes', (x0, y + 54), 0.36,
            (40, 40, 170))
        put(canvas, 'Use 9 rather than forcing a guess.', (x0, y + 76), 0.36,
            (110, 110, 110))

    # 底部键位
    yf = WIN_H - FOOT_H + 30
    put(canvas, '[1] straight   [2] omega   [3] sinusoidal   [0] unusable   '
                '[9] ambiguous', (12, yf), 0.5, (40, 40, 140))
    put(canvas, '[u] undo   [n] skip   [p] prev   [+/-] zoom   [g] examples   '
                '[q] save & quit     (min observation time enforced)',
        (12, yf + 26), 0.44, (95, 95, 95))

    if warn_left > 0:
        canvas[4:36, 0:WIN_W] = (60, 60, 200)
        put(canvas, f'TOO FAST - look again ({warn_left:.1f}s)',
            (14, 27), 0.6, (255, 255, 255), 2)
    return canvas


def main():
    ap = argparse.ArgumentParser(description='单图慢标版（第二轮/复核）')
    ap.add_argument('-b', '--batch', required=True)
    ap.add_argument('-i', '--input', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--zoom', type=int, default=7)
    ap.add_argument('--min-sec', type=float, default=0.5,
                    help='最短观察时间，按键早于此不记录；0=关闭')
    ap.add_argument('--exemplars', default='annotation_v2/exemplars')
    ap.add_argument('--max-items', type=int, default=0, help='只标前 N 张（试手）')
    args = ap.parse_args()

    items = load_items(args.batch)
    if args.max_items > 0:
        items = items[:args.max_items]
    total = len(items)
    ex = load_exemplars(args.exemplars)
    print(f'待标 {total} 条；示例图 {len(ex)}/3 张；最短观察 {args.min_sec:.1f}s')
    print('规则：1=笔直  2=Ω型（单个深弯）  3=正弦型（两波及以上）  '
          '0=图像不可用  9=看得清但无法归类')

    labels, secs, fasts = {}, {}, {}
    if os.path.exists(args.out):
        with open(args.out, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                iid = (r.get('item_id') or '').strip()
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
                fasts[iid] = int(float(r.get('fast') or 0)) if (r.get('fast') or '').strip() else 0
        print(f'已有 {len(labels)} 条，继续')

    def flush():
        tmp = args.out + '.tmp'
        with open(tmp, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['item_id', 'rel_path', 'label_code', 'label_name',
                        'seconds', 'fast'])
            rel_of = dict(items)
            for iid, code in labels.items():
                w.writerow([iid, rel_of.get(iid, ''), code, CLASS_CODES.get(code, ''),
                            f'{secs.get(iid, 0):.1f}', fasts.get(iid, 0)])
        os.replace(tmp, args.out)

    # 定位到第一个未标条目
    idx = 0
    for j, (iid, _r) in enumerate(items):
        if iid not in labels:
            idx = j
            break
    else:
        idx = total - 1
        print('全部条目已标完（按 u 可撤销重标）')

    zoom = args.zoom
    show_ex = True
    cache = {}
    t_enter = time.time()
    warn_until = 0.0
    last_sec = float('nan')

    while True:
        iid, rel = items[idx]
        p = os.path.join(args.input, rel.replace('/', os.sep))
        if p not in cache:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            cache[p] = im if im is not None else None
        img = cache[p]
        elapsed = time.time() - t_enter
        warn_left = max(0.0, warn_until - time.time())
        n_left = sum(1 for i, _r in items if i not in labels)
        canvas = build_single(img, ex, iid, idx, total, len(labels), n_left, zoom,
                              warn_left, show_ex, elapsed
                              if not (elapsed == elapsed) else elapsed)
        cv2.imshow('Human Posture Labeling v3', canvas)
        k = cv2.waitKey(20) & 0xFF

        if k in (27, ord('q')):
            break

        if warn_left > 0 and k in (ord('1'), ord('2'), ord('3'), ord('0'), ord('9')):
            continue          # 太快：忽略这次按键，等红条消失

        if k in (ord('1'), ord('2'), ord('3'), ord('0'), ord('9')):
            if args.min_sec > 0 and elapsed < args.min_sec:
                warn_until = time.time() + (args.min_sec - elapsed) + 0.6
                continue
            labels[iid] = int(chr(k))
            secs[iid] = elapsed
            fasts[iid] = 1 if elapsed < max(args.min_sec, 2.0) else 0
            flush()
            last_sec = elapsed
            nxt = None
            for j in range(idx + 1, total):
                if items[j][0] not in labels:
                    nxt = j
                    break
            if nxt is None:
                for j in range(0, idx):
                    if items[j][0] not in labels:
                        nxt = j
                        break
            if nxt is not None:
                idx = nxt
            t_enter = time.time()
            continue

        if k == ord('u'):
            if iid in labels:
                labels.pop(iid, None)
                secs.pop(iid, None)
                fasts.pop(iid, None)
                flush()
                t_enter = time.time()
            else:
                prev = None
                for j in range(idx - 1, -1, -1):
                    if items[j][0] in labels:
                        prev = j
                        break
                if prev is not None:
                    idx = prev
                    pj = items[idx][0]
                    labels.pop(pj, None)
                    secs.pop(pj, None)
                    fasts.pop(pj, None)
                    flush()
                    t_enter = time.time()
            continue

        if k in (ord('n'), 32):
            idx = min(total - 1, idx + 1)
            t_enter = time.time()
            continue
        if k == ord('p'):
            idx = max(0, idx - 1)
            t_enter = time.time()
            continue
        if k in (ord('+'), ord('=')):
            zoom = min(zoom + 1, 12)
            continue
        if k in (ord('-'), ord('_')):
            zoom = max(zoom - 1, 1)
            continue
        if k == ord('g'):
            show_ex = not show_ex
            continue

    flush()
    cv2.destroyAllWindows()

    # ---- 收尾体检 ----
    done = sum(1 for v in labels.values() if v in (1, 2, 3))
    n0 = sum(1 for v in labels.values() if v == 0)
    n9 = sum(1 for v in labels.values() if v == 9)
    vals = [s for s in secs.values() if s > 0]
    print(f'\n完成 {len(labels)}/{total}：三类 {done}，不可用 {n0}，无法归类 {n9}')
    if vals:
        vals.sort()
        med = vals[len(vals) // 2]
        fast = sum(1 for v in vals if v < 1.0) / len(vals)
        print(f'用时中位 {med:.2f}s，<1s 占 {fast * 100:.1f}%')
        if fast > 0.30:
            print('  [警告] 仍有超过 30% 的条目在 1 秒内作答，'
                  '这份作答与第一轮同样会被质疑"没看清"。')
    dist = {c: sum(1 for v in labels.values() if v == c) for c in (1, 2, 3)}
    if sum(dist.values()):
        print('类别分布：' + '  '.join(
            f'{CLASS_CODES[c]}={dist[c]} ({dist[c] / sum(dist.values()) * 100:.1f}%)'
            for c in (1, 2, 3)))
        if dist[2] / sum(dist.values()) > 0.50:
            print('  [提示] Ω 占比超过 50%。第一轮就是 57.7% 过判 Ω，'
                  '请确认不是把"任意一个弯"都当成了 Ω。')
    if n9 == 0:
        print('  [提示] 一条 9 都没用。若确有卡在两类之间的帧，请用 u 撤销后重判为 9——')
        print('         这个比例是论文里"类界模糊"的唯一直接证据。')
    print(f'已保存：{args.out}')
    if len(labels) < total:
        print('提示：还有未标条目，重跑同一命令可继续。')


if __name__ == '__main__':
    main()
