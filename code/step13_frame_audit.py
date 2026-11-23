"""帧级保留率审计：检验各组骨架提取失败率是否均衡（step13_frame_audit.py）。

动机（2026-10-24）
----------------
atlas_worm_hist.csv 以"每虫有效帧数 >= 阈值"为纳入条件，该条件混合了三层损失：
  (1) 上游层：死虫过滤、BoTSORT 跟踪丢失、YOLO 漏检、交叠多虫框剔除
      —— 已体现在 $WORM_DATA_ROOT 目录内的文件本身；
  (2) 骨架层：模糊、边框带、退化骨架、两虫交叉过滤导致的提取失败
      —— 发生在 cluster_atlas 的 Stage 1；
  (3) 真实死亡。
若第 (2) 层在组间不均衡（例如某组虫体缠绕交叉更多而被大量过滤），则
"样本量随日期下降"会被误读为生存差异，组间比较亦受污染。本脚本对三层
分别计量，并对各组骨架成功率与虫级保留率做两比例 z 检验（BH 校正）：

  · 目录层：各 (组,日期) 的文件数与唯一 worm 数（只读文件名，不解码图像）；
  · 缓存层：各 (组,日期) 的有效帧数（读 kappa_cache.npz 的 filenames）；
  · 统计层：各 (组,日期) 进入 atlas_worm_hist.csv 的虫数。

判定口径（帧成功率组间极差）：< 2 pp 基本均衡；2–5 pp 轻度不均衡需在论文
报告；> 5 pp 显著不均衡，保留率不得解释为生存。

注意：本脚本只能计量"提取失败的总率"，无法把失败归因到交叉（缓存不记录
失败原因）。若需归因，须带原因码重跑提取。另外，两虫交叉帧无法拆分为两条
骨架，故"交叉帧计数为 2"在当前管线不可行；本脚本的回答是"测量其组间差异"。

用法：
  python step13_frame_audit.py --root $WORM_DATA_ROOT \
      --cache atlas_out_full/kappa_cache.npz \
      --hist atlas_out_full/atlas_worm_hist.csv \
      -o atlas_out_full/frame_audit.csv
  # 快速冒烟（只扫前 N 个文件，验证路径解析）：
  python step13_frame_audit.py --root $WORM_DATA_ROOT --max-files 20000 \
      -o _audit_smoke.csv
"""
import os
import re
import csv
import math
import argparse
from collections import Counter, defaultdict

from wormpath import get_data_root

GROUP_ORDER = ['control', '5mg', '8mg', '10mg']
EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
FLAT_RE = re.compile(
    r'^(?P<group>[^/_]+)_(?P<date>\d{4})_(?P<row>[A-E])_'
    r'(?P<worm>worm\d+)_(?P<frame>\d+)$')


def parse_rel(rel):
    """从相对路径解析 (group, date, row, worm, frame)；失败返回 None。

    兼容三种布局：
      五维目录  `组/日期/行/worm{n}_{frame}.png`（本项目实际布局，4 个组件）
      五层目录  `组/日期/行/worm{n}/{frame}.png`（5 个组件）
      扁平命名  `组_日期_行_worm{n}_{frame}.png`
    """
    rel = rel.replace('\\', '/')
    parts = [p for p in rel.split('/') if p]
    if not parts:
        return None
    if len(parts) >= 4:
        group, date, row = parts[0], parts[1], parts[2]
        # 布局 A: .../worm{n}_{frame}.png
        m = re.match(r'^(worm\d+)_(\d+)$', os.path.splitext(parts[-1])[0])
        if m:
            return group, date, row, m.group(1), m.group(2)
        # 布局 B: .../worm{n}/{frame}.png
        if len(parts) >= 5:
            worm_stem = os.path.splitext(parts[-2])[0]
            fm = re.match(r'^(\d+)$', os.path.splitext(parts[-1])[0])
            if re.match(r'^worm\d+$', worm_stem) and fm:
                return group, date, row, worm_stem, fm.group(1)
        # 退化：文件名非 worm{n}_{frame}，仍按目录前三层归属（与 list_tasks
        # 的计数口径一致：所有图像文件都计入扫描分母）
        return group, date, row, os.path.splitext(parts[-1])[0], '0'
    m = FLAT_RE.match(os.path.splitext(parts[-1])[0])
    if m:
        return (m.group('group'), m.group('date'), m.group('row'),
                m.group('worm'), m.group('frame'))
    return None


def scan_directory(root, max_files=None):
    """遍历root，返回 (files, worms)：各(组,日期)文件数与唯一worm集合。"""
    files = Counter()
    worms = defaultdict(set)
    n_seen = 0
    n_bad = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # 2026-11-13：跳过 _noise_quarantine 等隔离目录，避免删掉的帧被重新计入
        dirnames[:] = [d for d in dirnames if not d.startswith('_')]
        dirnames.sort()
        for fn in sorted(filenames):
            if not fn.lower().endswith(EXTS):
                continue
            n_seen += 1
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            info = parse_rel(rel)
            if info is None:
                n_bad += 1
                continue
            g, d, _row, worm, _frame = info
            files[(g, d)] += 1
            worms[(g, d)].add(worm)
            if max_files and n_seen >= max_files:
                print(f'[冒烟] 达到 --max-files {max_files}，提前结束扫描')
                return files, worms, n_seen, n_bad
            if n_seen % 200000 == 0:
                print(f'  已扫描 {n_seen:,} 个文件...', flush=True)
    return files, worms, n_seen, n_bad


def scan_cache(cache_path):
    """读 kappa_cache.npz 的 filenames，返回 (valid, valid_worms)。

    无 numpy 或文件缺失时返回 (None, None) 并提示。
    """
    if not cache_path or not os.path.exists(cache_path):
        print('[提示] 未提供缓存或缓存不存在，跳过缓存层统计')
        return None, None
    try:
        import numpy as np
    except ImportError:
        print('[提示] 环境无 numpy，跳过缓存层统计（目录层与统计层仍可用）')
        return None, None
    try:
        z = np.load(cache_path, allow_pickle=True)
        fns = z['filenames']
    except Exception as e:
        print(f'[提示] 缓存读取失败（{e}），跳过缓存层统计')
        return None, None
    valid = Counter()
    valid_worms = defaultdict(set)
    for rel in fns:
        info = parse_rel(str(rel))
        if info is None:
            continue
        g, d, _row, worm, _frame = info
        valid[(g, d)] += 1
        valid_worms[(g, d)].add(worm)
    return valid, valid_worms


def scan_hist(hist_path, min_frames):
    """读 atlas_worm_hist.csv，返回 {(组,日期): [虫数, 总帧数]}（按min_frames过滤）。"""
    rows = defaultdict(lambda: [0, 0])
    if not hist_path or not os.path.exists(hist_path):
        print('[提示] 未提供 atlas_worm_hist.csv，跳过统计层')
        return rows
    with open(hist_path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                nf = int(r['n_frames'])
            except (KeyError, ValueError):
                continue
            if nf < min_frames:
                continue
            key = (r['group'], r['date'])
            rows[key][0] += 1
            rows[key][1] += nf
    return rows


def two_prop_z(x1, n1, x2, n2):
    """两比例 z 检验，返回 (z, p)。输入不一致（n<=0 或 x>n）时返回 (nan, nan)。"""
    if n1 <= 0 or n2 <= 0 or x1 > n1 or x2 > n2 or x1 < 0 or x2 < 0:
        return float('nan'), float('nan')
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se <= 0:
        return float('nan'), float('nan')
    z = (p1 - p2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def bh(pvals):
    """Benjamini-Hochberg 校正，返回校正后 p（与全项目一致）。"""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [float('nan')] * n
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        cur = min(pvals[i] * n / (rank + 1), prev)
        prev = cur
        adj[i] = cur
    return adj


def main():
    ap = argparse.ArgumentParser(description='帧级保留率审计（三层损失分解）')
    ap.add_argument('--root', default=get_data_root(), help='图片根目录')
    ap.add_argument('--cache', default='atlas_out_full/kappa_cache.npz',
                    help='kappa_cache.npz（可选；注意运行中的全量缓存可能未更新）')
    ap.add_argument('--hist', default='atlas_out_full/atlas_worm_hist.csv')
    ap.add_argument('--min-frames', type=int, default=10,
                    help='统计层每虫最少帧数（与 cluster_atlas 一致）')
    ap.add_argument('--max-files', type=int, default=None, help='冒烟：只扫前N个文件')
    ap.add_argument('-o', '--out', default='atlas_out_full/frame_audit.csv')
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f'未找到目录 {args.root}')

    print(f'=== 层1 目录扫描: {args.root} ===')
    files, worms, n_seen, n_bad = scan_directory(args.root, args.max_files)
    print(f'扫描 {n_seen:,} 个图像文件，路径解析失败 {n_bad:,} 个，'
          f'覆盖 {len(files)} 个 (组,日期) 组合')

    print('\n=== 层2 缓存统计（有效帧）===')
    valid, valid_worms = scan_cache(args.cache)
    if valid is not None:
        cache_dates = sorted({d for (_g, d) in valid})
        print(f'缓存覆盖 {len(valid)} 个 (组,日期)，日期: {cache_dates}')
        dir_dates = sorted({d for (_g, d) in files})
        missing = [d for d in dir_dates if d not in cache_dates]
        if missing:
            print(f'[警告] 目录中有 {len(missing)} 个日期不在缓存内: {missing}')
            print('        （缓存可能是旧运行产物；全量跑完后请重跑本审计）')

    print('\n=== 层3 统计层（atlas_worm_hist.csv）===')
    hist = scan_hist(args.hist, args.min_frames)

    # ---- 汇总表（目录与 hist 的键取并集，容忍新旧运行混用）----
    hist_only = sorted(set(hist.keys()) - set(files.keys()))
    if hist_only:
        print(f'[警告] hist 含有 {len(hist_only)} 个目录中不存在的 (组,日期)，'
              f'如 {hist_only[:3]}——hist 可能是旧运行产物')
    keys = sorted(set(files.keys()) | set(hist.keys()))
    rows_out = []
    print('\n=== 各 (组,日期) 三层对照 ===')
    print(f'{"group":>8s} {"date":>6s} {"files":>8s} {"worms_dir":>9s} '
          f'{"valid":>8s} {"succ%":>7s} {"worms_hist":>10s} {"keep%":>7s}')
    for (g, d) in keys:
        nf = files.get((g, d), 0)
        nw = len(worms.get((g, d), ()))
        nv = valid.get((g, d)) if valid is not None else None
        nh, nfh = hist.get((g, d), [0, 0])
        if nh > nw:
            print(f'[警告] ({g},{d}) hist虫数 {nh} > 目录虫数 {nw}，两者不是同一次运行')
        succ = (100.0 * nv / nf) if nv is not None and nf else float('nan')
        keep = (100.0 * nh / nw) if nw else float('nan')
        rows_out.append(dict(group=g, date=d, files=nf, worms_dir=nw,
                             valid_frames=nv if nv is not None else '',
                             success_pct=f'{succ:.2f}' if succ == succ else '',
                             worms_hist=nh, worm_keep_pct=f'{keep:.2f}',
                             mean_frames_hist=(f'{nfh / nh:.1f}' if nh else '')))
        if succ == succ:
            print(f'{g:>8s} {d:>6s} {nf:>8d} {nw:>9d} {nv:>8d} '
                  f'{succ:>6.2f}% {nh:>10d} {keep:>6.2f}%')
        else:
            print(f'{g:>8s} {d:>6s} {nf:>8d} {nw:>9d} {"--":>8s} '
                  f'{"--":>7s} {nh:>10d} {keep:>6.2f}%')

    # ---- 分组汇总 ----
    print('\n=== 分组汇总 ===')
    agg = {}
    for g in GROUP_ORDER:
        nf = sum(files[(g, d)] for (gg, d) in files if gg == g)
        nw = sum(len(worms[(g, d)]) for (gg, d) in worms if gg == g)
        nv = sum(valid[(g, d)] for (gg, d) in valid if gg == g) if valid else None
        nh = sum(v[0] for (gg, d), v in hist.items() if gg == g)
        agg[g] = dict(files=nf, worms=nw, valid=nv, worms_hist=nh)
        succ_s = f'{100.0 * nv / nf:.2f}%' if nv is not None and nf else '--'
        keep_s = f'{100.0 * nh / nw:.2f}%' if nw else '--'
        print(f'  {g:>8s}: files={nf:>9,}  worms_dir={nw:>5d}  '
              f'valid={nv if nv is not None else "--":>9}  succ={succ_s:>7s}  '
              f'worms_hist={nh:>5d}  keep={keep_s:>7s}')

    # ---- 组间均衡性检验 ----
    print('\n=== 组间均衡性检验（control 为参照，两比例 z 检验 + BH 校正）===')
    frame_tests, worm_tests = [], []
    for g in GROUP_ORDER[1:]:
        if g not in agg:
            continue
        # 帧成功率
        if (agg['control']['valid'] is not None
                and agg[g]['valid'] is not None
                and agg['control']['files'] and agg[g]['files']):
            z, p = two_prop_z(agg['control']['valid'], agg['control']['files'],
                              agg[g]['valid'], agg[g]['files'])
            dpp = (100.0 * agg['control']['valid'] / agg['control']['files']
                   - 100.0 * agg[g]['valid'] / agg[g]['files'])
            frame_tests.append((g, dpp, z, p))
        # 虫级保留率
        z2, p2 = two_prop_z(agg['control']['worms_hist'], agg['control']['worms'],
                            agg[g]['worms_hist'], agg[g]['worms'])
        worm_tests.append((g, z2, p2))
    if frame_tests:
        padj = bh([t[3] for t in frame_tests])
        print('帧成功率（有效帧/文件数）:')
        for (g, dpp, z, p), pa in zip(frame_tests, padj):
            print(f'  control vs {g:>5s}: 差={dpp:+.2f} pp  z={z:+.2f}  '
                  f'p={p:.3g}  p_BH={pa:.3g}')
    if worm_tests:
        padj = bh([t[2] for t in worm_tests])
        print('虫级保留率（hist虫数/目录虫数）:')
        for (g, z, p), pa in zip(worm_tests, padj):
            print(f'  control vs {g:>5s}: z={z:+.2f}  p={p:.3g}  p_BH={pa:.3g}')

    # ---- 判定 ----
    print('\n=== 判定 ===')
    if frame_tests:
        spread = max(abs(t[1]) for t in frame_tests)
        sig = any(t[3] < 0.05 for t in frame_tests)
        if spread < 2:
            print(f'帧成功率组间最大差 {spread:.2f} pp（<2 pp）：骨架提取失败基本均衡，'
                  f'保留率差异主要来自上游层（死亡/跟踪/检测）。')
        elif spread < 5:
            print(f'帧成功率组间最大差 {spread:.2f} pp（2–5 pp）：轻度不均衡，'
                  f'建议在论文中报告各组提取成功率。')
        else:
            print(f'帧成功率组间最大差 {spread:.2f} pp（>5 pp）：显著不均衡！'
                  f'保留率与组间比较均受污染，不得将保留率解释为生存差异。')
        if sig and spread >= 2:
            print('注意：样本量大时微小差异也会"统计显著"，请以百分点差值为准。')
    print('局限：本审计无法把提取失败归因到两虫交叉（缓存不记录失败原因）；'
          '交叉帧也无法拆分为两条骨架，故不能"计数为 2"。')

    # ---- 写 CSV ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['group', 'date', 'files', 'worms_dir',
                                          'valid_frames', 'success_pct',
                                          'worms_hist', 'worm_keep_pct',
                                          'mean_frames_hist'])
        w.writeheader()
        w.writerows(rows_out)
    print(f'\n审计表已写入: {args.out}')


if __name__ == '__main__':
    main()
