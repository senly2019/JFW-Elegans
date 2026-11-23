"""step18c_unusable_audit.py — 客观体检每一张送标图，核对"0（图像不可用）"是否被漏用。

为什么需要
----------
标注规则里 0 = 图像不可用（空白/全黑/虫体出框/糊到认不出）。若某标注者 310 条
一条 0 都没用，有两种可能：
  (a) 这批图确实张张可用 —— 那 0 的比例=0 是合理结果，可写进论文；
  (b) 他没在判断这条规则（或没真正看图）—— 那"0% 无法归类/不可用"不能当证据。
本脚本给出客观依据，逐图算可用性指标，并与实际作答交叉列表。

指标（全部只看像素，不看作答）
  read     能否解码（GUI 解码失败时会塞一张纯黑 90x80，标记者可能照标）
  std      灰度标准差，<5 基本是空白/纯色
  rng      p2~p98 动态范围（新数据原图仅约 35 级，已确认偏灰）
  fg       Otsu 前景占比，<0.03 认为无虫体、>0.92 认为几乎满视野
  border   前景像素落在最外 1px 边框的比例，高 = 虫体出框被截断
  lap      Laplacian 方差，<20 认为严重失焦/模糊

用法
----
  python step18c_unusable_audit.py -l annotation_v2/labels_A.csv
  python step18c_unusable_audit.py -l annotation_v2/labels_A.csv annotation_v2/labels_B.csv
"""

import os
import csv
import argparse
import numpy as np
import cv2

from wormpath import get_data_root

ROOT_DEFAULT = get_data_root()


def gray_stats(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    std = float(g.std())
    p2, p98 = np.percentile(g, [2, 98])
    rng = float(p98 - p2)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    # Otsu 前景（虫体通常比背景暗）
    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fgmask = (g < thr).astype(np.uint8)
    fg = float(fgmask.mean())
    border = 0.0
    if fgmask.sum() > 0:
        b = np.concatenate([fgmask[0, :], fgmask[-1, :], fgmask[:, 0], fgmask[:, -1]])
        border = float(b.mean())
    return dict(w=w, h=h, mean=float(g.mean()), std=std, rng=rng, fg=fg,
                border=border, lap=lap, thr=float(thr))


def flags(s):
    """返回该图的可用性问题清单（空 = 可用）。阈值与论文 Methods 一致或更宽。"""
    if s is None:
        return ['read']
    f = []
    if s['std'] < 5.0:
        f.append('blank')
    if s['fg'] < 0.03:
        f.append('no_worm')
    if s['fg'] > 0.92:
        f.append('full_frame')
    if s['lap'] < 20.0:
        f.append('blur')
    if s['border'] > 0.25:
        f.append('cropped')
    return f


def main():
    ap = argparse.ArgumentParser(description='送标图可用性客观体检')
    ap.add_argument('-l', '--labels', nargs='+', required=True)
    ap.add_argument('--root', default=ROOT_DEFAULT)
    ap.add_argument('-o', '--out', default='annotation_v2/unusable_audit.csv')
    ap.add_argument('--montage', default='annotation_v2/diag_unusable.png')
    ap.add_argument('--montage-n', type=int, default=12)
    args = ap.parse_args()

    lines = []

    def P(s=''):
        print(s)
        lines.append(s)

    # 先做全量图像体检（按路径去重）
    stats = {}
    for lp in args.labels:
        with open(lp, encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                rel = (r.get('rel_path') or '').replace('\\', '/')
                if rel and rel not in stats:
                    p = os.path.join(args.root, rel.replace('/', os.sep))
                    stats[rel] = gray_stats(p)

    P('=' * 72)
    P(f'送标图可用性体检：{len(stats)} 张唯一图像')
    P('=' * 72)
    allf = {rel: flags(s) for rel, s in stats.items()}
    n_read_fail = sum(1 for f in allf.values() if 'read' in f)
    P(f'  解码失败（GUI 会显示纯黑占位图）：{n_read_fail} 张')
    for name, lab in (('blank(空白/纯色)', 'blank'), ('no_worm(无虫体)', 'no_worm'),
                      ('full_frame(几乎满视野)', 'full_frame'), ('blur(严重模糊)', 'blur'),
                      ('cropped(虫体出框)', 'cropped')):
        n = sum(1 for f in allf.values() if lab in f)
        P(f'  {name:24s} {n:3d} 张')
    clean = sum(1 for f in allf.values() if not f)
    P(f'  无任何问题的图：{clean} / {len(allf)}（{clean/len(allf)*100:.1f}%）')

    # 与作答交叉
    for lp in args.labels:
        with open(lp, encoding='utf-8-sig', newline='') as f:
            rows = [r for r in csv.DictReader(f)
                    if (r.get('label_code') or '').strip() in ('0', '1', '2', '3', '9')]
        P('')
        P('-' * 72)
        P(f'{os.path.basename(lp)}：n={len(rows)}')
        P('-' * 72)
        ct = {}
        for r in rows:
            rel = (r.get('rel_path') or '').replace('\\', '/')
            fs = allf.get(rel, ['read'])
            key = '+'.join(fs) if fs else 'ok'
            code = r['label_code'].strip()
            ct.setdefault(key, {}).setdefault(code, 0)
            ct[key][code] += 1
        P('  图像状态 \\ 作答码      ' + ''.join(f'{c:>6s}' for c in '01239'))
        for key in sorted(ct, key=lambda k: -sum(ct[k].values())):
            tot = sum(ct[key].values())
            P(f'  {key:22s} ' + ''.join(f'{ct[key].get(c, 0):6d}' for c in '01239')
              + f'   共{tot}')
        # 明确回答用户的问题
        bad_but_labeled = [r for r in rows
                           if allf.get((r.get('rel_path') or '').replace('\\', '/'), [])
                           and r['label_code'].strip() in ('1', '2', '3')]
        P(f'\n  -> 有客观缺陷、却被判为 1/2/3（未用 0）的条目：{len(bad_but_labeled)} 条'
          f'（占 {len(bad_but_labeled)/max(1,len(rows))*100:.1f}%）')
        if bad_but_labeled:
            P('     样例：' + ', '.join(
                f"{r['item_id']}={'+'.join(allf[(r['rel_path']).replace(chr(92),'/')])}"
                for r in bad_but_labeled[:8]))
        n0 = sum(1 for r in rows if r['label_code'].strip() == '0')
        n9 = sum(1 for r in rows if r['label_code'].strip() == '9')
        P(f'  -> 实际用了 0 的：{n0} 条；用了 9 的：{n9} 条')

    # 落盘逐图明细
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rel_path', 'std', 'rng', 'fg', 'border', 'lap', 'flags'])
        for rel, s in sorted(stats.items()):
            if s is None:
                w.writerow([rel, '', '', '', '', '', 'read'])
            else:
                w.writerow([rel, f"{s['std']:.2f}", f"{s['rng']:.1f}", f"{s['fg']:.4f}",
                            f"{s['border']:.3f}", f"{s['lap']:.1f}",
                            '+'.join(allf[rel]) or 'ok'])
    P(f'\n逐图明细已写出：{args.out}')

    # 拼图：挑最可疑的若干张
    with open(args.labels[0], encoding='utf-8-sig') as f:
        lab0 = {r['item_id']: r for r in csv.DictReader(f)}

    def badness(rel):
        s = stats.get(rel)
        if s is None:
            return -1
        return abs(np.log(max(s['fg'], 1e-4) / 0.15)) + max(0, 5 - s['std'])

    # 2026-11-17：候选仅限真实可读的图。缺失/损坏图 badness=-1 会排到最前，
    # 曾导致拼图 12 格全选了服务器上不存在的引用路径，输出整版空白。
    cand = [(badness(rel), rel) for rel, fs in allf.items()
            if fs and stats.get(rel) is not None]
    cand.sort()
    pick = cand[:args.montage_n]
    if not pick:
        P('（无可疑图，跳过拼图）')
    else:
        CELL, PAD, TOP, COLS = 170, 6, 32, 6
        rows_n = (len(pick) + COLS - 1) // COLS
        W = COLS * (CELL + PAD) + PAD
        H = rows_n * (CELL + PAD + TOP) + PAD
        canvas = 255 * np.ones((H, W, 3), dtype=np.uint8)
        for i, (_, rel) in enumerate(pick):
            x = PAD + (i % COLS) * (CELL + PAD)
            y = PAD + (i // COLS) * (CELL + PAD + TOP)
            p = os.path.join(args.root, rel.replace('/', os.sep))
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                # 2026-11-17：不可读图片已由数据维护步骤从数据集中删除；
                # 此处直接跳过，不再生成占位块（占位块曾混入诊断拼图）。
                print(f'[warn] 无法读取图片，跳过: {p}', flush=True)
                continue
            if im.shape[0] != CELL or im.shape[1] != CELL:
                im = cv2.resize(im, (CELL, CELL),
                                interpolation=cv2.INTER_LANCZOS4)
            canvas[y:y + CELL, x:x + CELL] = im
            cv2.rectangle(canvas, (x, y), (x + CELL - 1, y + CELL - 1), (120, 120, 120), 1)
            # 找到该路径对应的作答
            iid, code = '', '?'
            for r in lab0.values():
                if (r.get('rel_path') or '').replace('\\', '/') == rel:
                    iid, code = r['item_id'], r['label_code']
                    break
            s = stats[rel] or {}
            cv2.putText(canvas, f"{iid} label={code}", (x, y - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, '+'.join(allf[rel])[:24], (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 40, 40), 1, cv2.LINE_AA)
        cv2.imwrite(args.montage, canvas)
        P(f'可疑样本拼图：{args.montage}（{len(pick)} 张，红字=客观缺陷）')

    with open(os.path.splitext(args.out)[0] + '.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
