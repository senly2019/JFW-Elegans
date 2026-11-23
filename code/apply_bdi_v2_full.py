# -*- coding: utf-8 -*-
r"""apply_bdi_v2_full.py — 全量双口径 BDI（服务器运行，只新增文件）。

背景（2026-11-13 用户裁决：双口径并行）
--------------------------------------
BDI v2 的权重是用六人共识分拟合的，因此它是「人评对齐弯曲度」的计算代理，
不是独立于人的客观量。为让论文结论不依赖是否引入人评先验，全量同时输出两列：

  * bdi_v2  : 人评对齐分数（0–100 量纲，用 bdi_v2_formula.json 的权重）
  * bdi_geo : 纯几何口径 = 虫级 10 帧均值的 mean|κ|，零人评参与

所有组间 / 剂量 / 年龄结论两列各算一遍（敏感性分析）。若两列结论一致，
即可在论文里写「结论不依赖是否引入人评先验」，直接回应主观性质疑。

口径一致性（关键）
------------------
训练特征由 extract_bdi_v2_feats.py 以「每虫等距抽 10 帧 → 逐特征均值」得到。
本脚本直接复用它的 _worm_feats，**不**用 kappa_cache 的全帧均值代替
（全帧 vs 10 帧是不同口径，套同一权重会有系统偏差）。

输出（默认 atlas_out_all/，不覆盖 v17 任何已有结果）
---------------------------------------------------
  bdi_v2_full.csv          虫级表：group/date/row/worm/n_frames + 11 特征 + 两列
  bdi_v2_worm_cache.csv    特征缓存（断点续跑，中断后重跑只补缺）
  bdi_v2_full_report.md    效度 + 双口径敏感性报告（论文素材）
  bdi_v2_full_summary.json 机器可读汇总

用法
----
# 服务器全量（PyCharm SSH 终端）
python apply_bdi_v2_full.py --root $WORM_DATA_ROOT --out atlas_out_all --workers 20

# 本地小样本自检（E 盘副本，只跑 20 只虫）
python apply_bdi_v2_full.py --root E:/data_new/images --out /tmp/bdi_smoke \
    --limit 20 --workers 4
"""
import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wormpath import get_data_root                      # noqa: E402
from extract_bdi_v2_feats import _worm_feats  # noqa: E402

FEAT_NAMES = ['mean_k', 'sin_bbi', 'e_front', 'e_mid', 'e_back', 'mid_dom',
              'max_area', 'p99_k', 'max_k', 'sklen', 'lap_var']
GROUPS = ('control', '5mg', '8mg', '10mg')
DATE_RE = re.compile(r'^\d{4}$')
ROW_RE = re.compile(r'^[A-Za-z]$')
FRAME_RE = re.compile(r'^(worm\d+)_(\d+)\.png$', re.I)
DEF_FORMULA = os.path.join(HERE, 'annotation_v3', 'retest',
                           'bdi_v2_formula.json')
DEF_CONSENSUS = os.path.join(HERE, 'tools_picker', 'consensus_by_worm.csv')
DEF_DROPS = os.path.join(HERE, 'tools_picker', 'worms_to_delete.json')


def resolve_root(root):
    """兼容「组目录在数据根的下一层」的情况（本地 E 盘是 data_new/images）。"""
    if any(os.path.isdir(os.path.join(root, g)) for g in GROUPS):
        return root
    for sub in ('images', os.path.join('data_new', 'images'), 'data_new'):
        p = os.path.join(root, sub)
        if any(os.path.isdir(os.path.join(p, g)) for g in GROUPS):
            print(f'[数据根] 自动定位到下一层：{p}')
            return p
    return root


def enumerate_worms(root):
    """单次遍历列出全部虫（跳过 _noise_quarantine 等非组目录）。"""
    out = []
    for g in sorted(os.listdir(root)):
        gp = os.path.join(root, g)
        if g not in GROUPS or not os.path.isdir(gp):
            continue
        for d in sorted(os.listdir(gp)):
            dp = os.path.join(gp, d)
            if not DATE_RE.match(d) or not os.path.isdir(dp):
                continue
            for r in sorted(os.listdir(dp)):
                rp = os.path.join(dp, r)
                if not ROW_RE.match(r) or not os.path.isdir(rp):
                    continue
                worms = {}
                for fn in os.listdir(rp):
                    m = FRAME_RE.match(fn)
                    if m:
                        worms.setdefault(m.group(1).lower(), 0)
                        worms[m.group(1).lower()] += 1
                for w, n in sorted(worms.items()):
                    out.append((g, d, r.upper(), w, n))
    return out


def key_of(g, d, r, w):
    return f'{g}/{d}/{r}/{w}'


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float('nan')
    ra = np.argsort(np.argsort(a[ok]))
    rb = np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def load_cache(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r.get('worm_key') and r.get('mean_k') not in ('', None):
                out[r['worm_key']] = r
    return out


def write_cache(path, cache):
    fields = ['worm_key', 'n_valid_frames'] + FEAT_NAMES
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for k in sorted(cache):
            w.writerow(cache[k])


def compute_scores(formula, feats):
    """按训练好的公式算 bdi_v2；bdi_geo 就是同一批帧的 mean_k，零人评。"""
    W = formula['weights']
    n_fill = 0
    z = 0.0
    for c, spec in W.items():
        v = feats.get(c)
        if v is None or not np.isfinite(v):
            v = spec['mean']          # 非有限值按训练均值处理 ⇒ 该项 z=0
            n_fill += 1
        z += spec['w'] * (v - spec['mean']) / (spec['sd'] or 1e-12)
    bdi_v2 = formula['cons_mean'] + formula['cons_sd'] * z
    return float(bdi_v2), float(feats['mean_k']), n_fill


def main():
    ap = argparse.ArgumentParser(description='全量双口径 BDI v2')
    ap.add_argument('--root', default=get_data_root(), help='图片数据根')
    ap.add_argument('--out', default='atlas_out_all', help='输出目录')
    ap.add_argument('--formula', default=DEF_FORMULA)
    ap.add_argument('--consensus', default=DEF_CONSENSUS)
    ap.add_argument('--drops', default=DEF_DROPS)
    ap.add_argument('--workers', type=int, default=0, help='进程数（0=min(20,CPU)）')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 只虫（自检用）')
    ap.add_argument('--flush-every', type=int, default=200, help='缓存写盘间隔')
    ap.add_argument('--no-probe', action='store_true',
                    help='跳过启动自检（断点续跑且剩余均为上次已知失败虫时使用）')
    args = ap.parse_args()

    root = resolve_root(args.root)
    if not any(os.path.isdir(os.path.join(root, g)) for g in GROUPS):
        raise SystemExit(f'数据根下找不到组目录：{root}')
    os.makedirs(args.out, exist_ok=True)
    with open(args.formula, encoding='utf-8') as f:
        formula = json.load(f)
    W = formula['weights']
    print(f'[数据根] {root}')
    print(f'[公式] n_train={formula["n_train"]}  λ={formula["lam"]}  '
          f'LOO={formula["spe_loo_train"]}  盲标={formula["spe_blind_check"]}')

    worms = enumerate_worms(root)
    if args.limit:
        worms = worms[:args.limit]
    print(f'[枚举] {len(worms):,} 只虫（隔离区已自动排除）')

    cache_path = os.path.join(args.out, 'bdi_v2_worm_cache.csv')
    cache = load_cache(cache_path)
    todo = [t for t in worms if key_of(*t[:4]) not in cache]
    print(f'[缓存] 已有 {len(cache):,} 只；待提取 {len(todo):,} 只')

    n_workers = args.workers or min(20, max(1, os.cpu_count() or 1))
    n_ok = 0
    failed_keys = []
    # ---- 启动自检：先同步跑 5 只，防止骨架链路整批失效导致数小时白跑 ----
    # 断点续跑时 todo 可能全是上次已知失败的虫（成功者都已在缓存里），
    # 此时自检必然 0/5 通过——用 --no-probe 跳过（2026-11-14）。
    if todo and not args.no_probe:
        probe = todo[:5]
        probe_ok = 0
        for t in probe:
            k = t[:4]
            _, feats, nf = _worm_feats(k + (root,))
            if feats is not None:
                cache[key_of(*k)] = dict(worm_key=key_of(*k),
                                         n_valid_frames=nf, **feats)
                probe_ok += 1
        if probe_ok < 3:
            raise SystemExit(
                f'[自检中止] 前 {len(probe)} 只虫仅 {probe_ok} 只提取成功——'
                '检查图片对比度 / 骨架链路后再跑全量，避免白跑数小时')
        print(f'[自检] 前 {len(probe)} 只虫有效 {probe_ok} 只，继续全量')
        todo = todo[len(probe):]
        write_cache(cache_path, cache)
    if todo:
        keys = [t[:4] for t in todo]
        tasks = [k + (root,) for k in keys]
        with Pool(n_workers) as pool:
            for i, (k, (_, feats, nf)) in enumerate(
                    zip(keys, pool.imap(_worm_feats, tasks, chunksize=4)), 1):
                if feats is None:
                    failed_keys.append(key_of(*k))
                    continue
                cache[key_of(*k)] = dict(worm_key=key_of(*k),
                                         n_valid_frames=nf, **feats)
                n_ok += 1
                if i % args.flush_every == 0:
                    write_cache(cache_path, cache)
                    print(f'  [进度] {i:,}/{len(todo):,}  有效 {n_ok:,}',
                          flush=True)
        write_cache(cache_path, cache)
        print(f'[提取完成] 本次新增 {n_ok:,} 只，缓存共 {len(cache):,} 只')
        if failed_keys:
            fp = os.path.join(args.out, 'bdi_v2_failed.txt')
            with open(fp, 'w', encoding='utf-8') as f:
                f.write('\n'.join(failed_keys) + '\n')
            print(f'[失败清单] {len(failed_keys):,} 只 → {fp}')

    # ---- 计算双口径分数 ----
    rows, n_fill_total = [], 0
    for g, d, r, w, n_frames in worms:
        rec = cache.get(key_of(g, d, r, w))
        if not rec:
            continue
        feats = {}
        for c in FEAT_NAMES:
            try:
                feats[c] = float(rec[c])
            except (KeyError, TypeError, ValueError):
                feats[c] = float('nan')
        v2, geo, n_fill = compute_scores(formula, feats)
        n_fill_total += n_fill
        rows.append(dict(group=g, date=d, row=r, worm=w,
                         worm_key=key_of(g, d, r, w),
                         n_frames=n_frames,
                         n_valid_frames=rec.get('n_valid_frames', ''),
                         bdi_v2=round(v2, 4), bdi_geo=round(geo, 6),
                         **{c: (round(feats[c], 6)
                               if np.isfinite(feats[c]) else '')
                            for c in FEAT_NAMES}))
    out_csv = os.path.join(args.out, 'bdi_v2_full.csv')
    fields = (['group', 'date', 'row', 'worm', 'worm_key', 'n_frames',
               'n_valid_frames', 'bdi_v2', 'bdi_geo'] + FEAT_NAMES)
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)
    n_failed = len(worms) - len(rows)
    print(f'[写出] {out_csv}（{len(rows):,} 行，补值 {n_fill_total} 项次）')
    print(f'[提取] 有效 {len(rows):,} / {len(worms):,} 只；'
          f'失败 {n_failed:,} 只（{n_failed / max(1, len(worms)):.1%}）')
    if len(worms) > 100 and n_failed / len(worms) > 0.2:
        print('[警告] 提取失败率 >20%，结果可能不可靠，请先检查骨架链路')

    # ---- 效度：与六人共识映射核对 ----
    cons = {}
    if os.path.exists(args.consensus):
        with open(args.consensus, encoding='utf-8-sig') as f:
            cons = {r['worm_key']: float(r['cons_mean'])
                    for r in csv.DictReader(f)}
    by_key = {r['worm_key']: r for r in rows}
    matched = [k for k in cons if k in by_key]
    dropped = [k for k in cons if k not in by_key]
    drop_keys = set()
    if os.path.exists(args.drops):
        with open(args.drops, encoding='utf-8-sig') as f:
            drop_keys = {f"{x['group']}/{x['date']}/{x['row']}/{x['worm']}"
                         for x in json.load(f)}
    n_deleted = sum(1 for k in dropped if k in drop_keys)
    v2_c = [by_key[k]['bdi_v2'] for k in matched]
    geo_c = [by_key[k]['bdi_geo'] for k in matched]
    y_c = [cons[k] for k in matched]
    sp_v2 = spearman(v2_c, y_c)
    sp_geo = spearman(geo_c, y_c)
    print(f'[效度] 共识虫 {len(cons)}，匹配 {len(matched)}，'
          f'未匹配 {len(dropped)}（其中 {n_deleted} 只为本次噪声剔除，'
          f'其余 {len(dropped) - n_deleted} 只未在本次输出中'
          f'（--limit 小样本或组标签差异）')
    print(f'       bdi_v2 vs 共识 Spearman {sp_v2:+.3f}；'
          f'bdi_geo vs 共识 {sp_geo:+.3f}')

    # ---- 双口径敏感性：整体 / 组间 / 年龄 ----
    sp_two = spearman([r['bdi_v2'] for r in rows],
                      [r['bdi_geo'] for r in rows])
    group_stat = {}
    for g in GROUPS:
        sel = [r for r in rows if r['group'] == g]
        if not sel:
            continue
        group_stat[g] = dict(
            n=len(sel),
            v2_mean=round(float(np.mean([r['bdi_v2'] for r in sel])), 3),
            geo_mean=round(float(np.mean([r['bdi_geo'] for r in sel])), 6),
            v2_median=round(float(np.median([r['bdi_v2'] for r in sel])), 3),
            geo_median=round(float(np.median([r['bdi_geo'] for r in sel])), 6))
    gs = sorted(group_stat, key=lambda k: group_stat[k]['v2_mean'])
    gs_geo = sorted(group_stat, key=lambda k: group_stat[k]['geo_mean'])
    order_same = gs == gs_geo
    age_stat = {}
    for d in sorted({r['date'] for r in rows}):
        sel = [r for r in rows if r['date'] == d]
        age_stat[d] = dict(n=len(sel),
                           v2_mean=round(float(np.mean(
                               [r['bdi_v2'] for r in sel])), 3),
                           geo_mean=round(float(np.mean(
                               [r['bdi_geo'] for r in sel])), 6))
    dates = sorted(age_stat)
    age_v2 = spearman(range(len(dates)),
                      [age_stat[d]['v2_mean'] for d in dates])
    age_geo = spearman(range(len(dates)),
                       [age_stat[d]['geo_mean'] for d in dates])
    print(f'[敏感性] 两列整体 Spearman {sp_two:+.3f}；'
          f'组均值排序一致：{order_same}')
    print(f'          年龄趋势 v2 {age_v2:+.3f} / geo {age_geo:+.3f}')

    verdict = ('一致：结论不依赖是否引入人评先验'
               if order_same and np.isfinite(sp_two) and sp_two > 0.7
               else '不一致或弱一致：需在论文中分别报告并讨论')
    summary = dict(
        built=datetime.now().isoformat(timespec='seconds'),
        data_root=root, n_worms=len(rows),
        n_worms_total=len(worms), n_failed=n_failed,
        formula=dict(n_train=formula['n_train'], lam=formula['lam'],
                     spe_loo=formula['spe_loo_train'],
                     spe_blind=formula['spe_blind_check']),
        validity=dict(n_consensus=len(cons), n_matched=len(matched),
                      n_unmatched=len(dropped),
                      spe_v2_vs_cons=round(sp_v2, 4),
                      spe_geo_vs_cons=round(sp_geo, 4),
                      unmatched_examples=dropped[:10]),
        sensitivity=dict(spe_two_columns=round(sp_two, 4),
                         group_order_v2=gs, group_order_geo=gs_geo,
                         group_order_same=order_same,
                         age_trend_v2=round(age_v2, 4),
                         age_trend_geo=round(age_geo, 4),
                         group_stat=group_stat, age_stat=age_stat,
                         verdict=verdict),
        caveat=('bdi_v2 权重由六人共识拟合，属人评对齐代理；'
                '与同一批共识的相关是复现度而非独立效度。'
                '独立效度见盲标验证（删后子集）与 bdi_geo 双口径一致性。'))
    with open(os.path.join(args.out, 'bdi_v2_full_summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    lines = [
        '# 全量双口径 BDI 报告',
        '',
        f'- 生成：{summary["built"]}　数据根：`{root}`',
        f'- 虫数：**{len(rows):,}**（噪声隔离后；枚举 {len(worms):,}，'
        f'骨架提取失败 {n_failed:,}）',
        f'- 公式：n_train={formula["n_train"]}，λ={formula["lam"]}，'
        f'LOO={formula["spe_loo_train"]}，盲标={formula["spe_blind_check"]}',
        '',
        '## 一、两列的定义（回应主观性质疑）',
        '',
        '| 列 | 定义 | 是否含人评 |',
        '|---|---|---|',
        '| `bdi_v2` | 岭回归拟合六人共识的几何公式，换算回 0–100 分 | 是（权重） |',
        '| `bdi_geo` | 虫级 10 帧等距采样的 mean\\|κ\\| 均值 | **否（零人评）** |',
        '',
        '`cons_mean` / `cons_sd` 只做「加常数、乘正数」的换单位，'
        '不改变排序，Spearman 完全不变；主观性只来自拟合权重。',
        '',
        '## 二、与六人共识的效度（复现度，非独立效度）',
        '',
        f'- 共识虫 {len(cons)}，匹配 **{len(matched)}**，'
        f'未匹配 {len(dropped)}（其中 {n_deleted} 只为本次噪声剔除，'
        f'其余 {len(dropped) - n_deleted} 只未在本次输出中）',
        f'- `bdi_v2` vs 共识：**{sp_v2:+.3f}**；'
        f'`bdi_geo` vs 共识：**{sp_geo:+.3f}**',
        '- 注意：训练目标就是这批共识，上述数字是**拟合复现度**；'
        f'独立效度请用盲标验证（本公式 LOO={formula["spe_loo_train"]}、'
        f'盲标 Spearman={formula["spe_blind_check"]}，均为删后子集）'
        '与下文双口径一致性。',
        '',
        '## 三、双口径敏感性（论文关键表）',
        '',
        f'- 两列整体 Spearman：**{sp_two:+.3f}**',
        f'- 组均值排序一致：**{order_same}**　'
        f'（v2：{" < ".join(gs)}；geo：{" < ".join(gs_geo)}）',
        f'- 年龄趋势：v2 {age_v2:+.3f} / geo {age_geo:+.3f}',
        f'- **判定：{verdict}**',
        '',
        '| 组 | n | v2 均值 | geo 均值 |',
        '|---|---|---|---|',
    ]
    for g in GROUPS:
        if g in group_stat:
            s = group_stat[g]
            lines.append(f'| {g} | {s["n"]:,} | {s["v2_mean"]} | '
                         f'{s["geo_mean"]} |')
    lines += [
        '',
        '> 组标签映射以用户 2026-10-24 裁决为准'
        '（皿号 1→control，2→10mg/mL，3→5mg/mL，4→8mg/mL）；'
        '若服务器目录标签校正状态有变，剂量顺序需重新核对。',
        '',
        '## 四、局限',
        '',
        f'1. 训练用删后 {formula["n_train"]} 项（已剔除被删虫对应评分项），'
        f'与全量应用口径一致；{len(dropped)} 只共识虫未进入全量输出'
        f'（其中 {n_deleted} 只为噪声剔除、其余为骨架提取失败）；',
        '2. 拟合目标（主实验共识）被预填 BDI 建议分锚定'
        '（删后合并 AI=0.578，60% 欠调整）；',
        '3. 盲标 39 项是删后子集且同一批标注者，属条件独立；',
        '4. 每虫仅抽 10 帧（与训练口径一致），不是全帧均值。',
        '',
        '## 五、文件',
        '',
        '- `bdi_v2_full.csv`：虫级双口径表（可直接并入组间分析）',
        '- `bdi_v2_worm_cache.csv`：11 特征缓存（断点续跑）',
        '- `bdi_v2_full_summary.json`：机器可读汇总',
    ]
    with open(os.path.join(args.out, 'bdi_v2_full_report.md'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[写出] {os.path.join(args.out, "bdi_v2_full_report.md")}')
    print(f'[判定] {verdict}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
