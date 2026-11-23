r"""bdi_calib.py — 本地现算 BBI → 管线口径 BDI 的单调校准。

为什么需要
----------
标注器徽章和预标注都要显示"当前图的 BDI"，但本地没有服务器的曲率缓存
（kappa_cache.npz / atlas_out_all/profile），只能本地现算
（extract_skeleton + compute_curvature_profile + mean）。实测
（2026-11-09，60 个锚点候选）：本地现算值系统性低于管线口径，且差距
随曲率增大——

    tick   0（canonical 0.146）  帧级差 ~0.000
    tick  20（0.284）            ~-0.044
    tick  40（0.348）            ~-0.147
    tick  60（0.406）            ~-0.135
    tick  80（0.474）            ~-0.185
    tick 100（0.693）            ~-0.348

根因是细化路径不同：本地 cv2 5.0.0 无 ximgproc（opencv-contrib 才有），
走 skimage / zhangsuen；服务器走 ximgproc。曲率对骨架细节敏感，高弯处
被压缩（Ω 折叠的尖角在 6px 基线窗口下被"切圆"）。

若不校准就把本地值当管线值：五段分区归段全面偏斜——实测 431 项预标注
60% 落 seg1，而批次按 worm 级 bdi_mean 五等分采样，本应各约 20%；
预标注分整体偏低，五人检查时大部分时间在纠正一个已知系统偏差。

做法
----
用锚点候选表 anchor_candidates.csv（每候选带管线口径 bdi，覆盖刻度
0/20/40/60/80/100）拟合一张分段线性单调映射（过各刻度中位数）：

    本地 BBI（worm 级均值）→ 管线口径 BDI
    0.116→0.146  0.163→0.284  0.185→0.348
    0.207→0.406  0.224→0.474  0.318→0.693

为什么只用 worm 级定标（2026-11-09 实测）：帧级单帧本地值噪声大
（±0.05），按刻度取中位数都非单调（tick20 的帧中位数 0.247 竟高于
tick40 的 0.209），帧图不可用；worm 级均值（10 帧等距采样）按刻度
干净单调、段内很紧。映射缓存于 annotation_v3/pre/bdi_calib.json；
指纹（细化路径）变化时自动重拟。

两个应用场景（都过同一张图，噪声特性不同）：
  徽章（帧级）：把这一帧的本地 BBI 单独过图——帧间噪声 ±0.05 本地值，
    高段放大到约 ±0.1 管线值，界面标注"帧级·校准"，是估值不是真值；
  预标注（worm 级）：先把该虫 10 帧本地 BBI 平均再过图——噪声
    ±0.03-0.06，归段稳，用作预填分。留一验证段间误差 ≤0.023，
    端部为 clamp（预期行为）。
映射缓存于 annotation_v3/pre/bdi_calib.json；指纹（细化路径 + 采样数）
变化时自动重拟。

局限（必须向标注者/论文声明）
----------------------------
1. 校准是单调近似，不是管线真值。有服务器缓存（--bdi-cache）时徽章
   优先用缓存真值（来源标'帧级'），不走校准（来源标'帧级·校准'）。
2. 帧级应用噪声 ±0.05 本地值（高段约 ±0.1 管线值），worm 级 ±0.03-0.06，
   边界项可能归错段——这正是要人工检查的，不是 bug。
3. 校准只在本地工具链内自洽：徽章、五段区间、预标注分三者同口径，
   与论文中 kappa_cache.npz 的真值可能有校准残差（发表分析不用本模块）。
"""
import csv
import glob
import json
import os
import re

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A3 = os.path.join(HERE, 'annotation_v3')
DEFAULT_JSON = os.path.join(A3, 'pre', 'bdi_calib.json')
DEFAULT_CAND = os.path.join(A3, 'anchor_candidates.csv')

_REL_RE = re.compile(r'^([^/]+)/(\d{4})/([A-E])/(worm\d+)_(\d+)\.png$')


def thin_fingerprint():
    """当前环境的细化路径标识（复刻 worm_labeler._thin_mask 的选择序）。

    ximgproc（opencv-contrib）→ skimage → zhangsuen(numpy)。校准依赖
    细化路径，路径变则本地 BBI 变，缓存自动失效重拟。
    """
    if hasattr(cv2, 'ximgproc'):
        return 'ximgproc'
    try:
        import skimage.morphology  # noqa: F401
        return 'skimage'
    except ImportError:
        return 'zhangsuen'


def _frame_bbi(path):
    """单帧本地 BBI（raw，未校准）：与 step8 _extract_one 同链路。"""
    try:
        from worm_labeler import extract_skeleton, compute_curvature_profile
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        skel = extract_skeleton(img)
        if skel is None or len(skel) < 10:
            return None
        kappa = compute_curvature_profile(skel, n_samples=100)
        if np.sum(kappa) <= 1e-6:
            return None
        return float(np.mean(kappa))
    except Exception:
        return None


def _worm_mean_bbi(images_root, group, date, row, worm, n_frames=10):
    """该虫 n_frames 帧等距采样的本地 BBI 均值（raw）。

    帧清单按帧号排序后等距取点；有效帧少于 5 张时返回 None
    （宁可无预填，不用 3 帧噪声硬归段）。
    """
    wdir = os.path.join(images_root, group, date, row)
    frames = sorted(glob.glob(os.path.join(wdir, f'{worm}_*.png')))
    if len(frames) < 5:
        return None, 0
    idx = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
    vals = [v for v in (_frame_bbi(frames[i]) for i in idx) if v is not None]
    if len(vals) < 5:
        return None, len(vals)
    return float(np.mean(vals)), len(vals)


def _piecewise(x, y):
    """分段线性查找表：x 必须单调不减。clamp 到端值。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]

    def f(v):
        if v is None or not np.isfinite(v):
            return None
        return float(np.interp(v, x, y))
    return f


def fit(cand_csv=DEFAULT_CAND, images_root='', out_json=DEFAULT_JSON,
        n_worm_frames=10, verbose=True):
    """用锚点候选拟合单调映射并写 JSON；返回 fit dict。

    失败一律抛 RuntimeError（不是 SystemExit）——调用方（GUI 徽章、
    dry-run 自检）都用 `except Exception` 降级，SystemExit 是
    BaseException 会直接掀翻进程（2026-11-13 dry-run 实测崩过一次）。
    """
    if not os.path.exists(cand_csv):
        raise RuntimeError(f'找不到锚点候选表：{cand_csv}')
    if not images_root or not os.path.isdir(images_root):
        raise RuntimeError(f'图片根不可用：{images_root!r}——校准需要读锚点候选图')
    cands = list(csv.DictReader(open(cand_csv, encoding='utf-8-sig')))
    fp = thin_fingerprint()

    per_tick = {}   # tick -> dict(canon=[], worm=[])
    for c in cands:
        try:
            tick = int(c['tick'])
            bdi = float(c['bdi'])
        except (KeyError, ValueError):
            continue
        m = _REL_RE.match((c.get('rel_path') or '').replace('\\', '/'))
        if not m:
            continue
        g, d, r, w, _ = m.groups()
        wm, nf = _worm_mean_bbi(images_root, g, d, r, w, n_worm_frames)
        if wm is None:
            continue
        per_tick.setdefault(tick, dict(canon=[], worm=[]))
        per_tick[tick]['canon'].append(bdi)
        per_tick[tick]['worm'].append(wm)

    ticks = sorted(per_tick)
    if len(ticks) < 3:
        raise RuntimeError(f'可用刻度不足（{ticks}）——检查图片根与候选表')
    med = {t: {k: float(np.median(v)) for k, v in per_tick[t].items()}
           for t in ticks}

    xs = [med[t]['worm'] for t in ticks]
    ys = [med[t]['canon'] for t in ticks]
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise RuntimeError(f'worm 级中位数非单调（{xs}）——数据异常，拒绝拟合')

    # 留一刻度验证：用其余刻度拟合，预测被留刻度（段间插值误差；
    # 端部刻度被留时为 clamp，误差偏大属预期，看中间刻度即可）
    val = {}
    for t in ticks:
        rest = [u for u in ticks if u != t]
        if len(rest) < 2:
            continue
        pred = _piecewise([med[u]['worm'] for u in rest],
                          [med[u]['canon'] for u in rest])(med[t]['worm'])
        val[t] = round(pred - med[t]['canon'], 4)

    fit_dict = dict(
        fingerprint=fp,
        n_worm_frames=n_worm_frames,
        n_ticks=len(ticks),
        ticks=ticks,
        knots=[[round(a, 5), round(b, 5)] for a, b in zip(xs, ys)],
        validation={str(k): v for k, v in val.items()},
    )
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(fit_dict, f, ensure_ascii=False, indent=1)

    if verbose:
        print(f'[校准] 细化路径 {fp}，{len(ticks)} 个刻度 × '
              f'{sum(len(per_tick[t]["canon"]) for t in ticks)} 候选')
        print('[校准] 映射（本地 BBI → 管线口径）：' +
              '  '.join(f'{a:.3f}→{b:.3f}' for a, b in fit_dict['knots']))
        errs = list(val.values())
        mid = {t: e for t, e in val.items()
               if t not in (ticks[0], ticks[-1])}
        print('[校准] 留一验证：' +
              '  '.join(f't{t}:{e:+.3f}' for t, e in val.items()) +
              (f'  中间刻度最大|误差| {max(abs(e) for e in mid.values()):.3f}'
               if mid else ''))
        print(f'[校准] 已写 {out_json}')
    return fit_dict


_CACHE = {'fit': None}


def load(json_path=DEFAULT_JSON, cand_csv=DEFAULT_CAND, images_root='',
         verbose=True):
    """读校准（指纹不符/缺失则自动重拟）。返回 fit dict。"""
    fp = thin_fingerprint()
    if os.path.exists(json_path):
        try:
            with open(json_path, encoding='utf-8') as f:
                d = json.load(f)
            if d.get('fingerprint') == fp and len(d.get('knots', [])) >= 3:
                _CACHE.update(loaded=True, fit=d)
                return d
        except Exception:
            pass
    if verbose:
        print(f'[校准] 缓存不可用（缺失或细化路径变为 {fp}）——重新拟合')
    d = fit(cand_csv=cand_csv, images_root=images_root, out_json=json_path,
            verbose=verbose)
    _CACHE.update(loaded=True, fit=d)
    return d


def worm_key(rel):
    """rel_path → (group, date, row, worm) 虫键；不匹配返回 None。"""
    m = _REL_RE.match((rel or '').replace('\\', '/'))
    return m.groups()[:4] if m else None


_WORM_EST = {}   # 会话缓存：虫键 -> 校准 BDI 估值 | None


def worm_estimate(images_root, rel, n_frames=10):
    """rel_path → 该虫校准 BDI 估值（10 帧本地均值过校准；带会话缓存）。

    为什么徽章/预填都用虫级而不是帧级（2026-11-09 实测 + 三张图人工
    判读）：锚点/批次帧本身比其虫均值偏弯（本地值 +0.04~0.08， pick_frame
    选的是最接近虫均值的帧，而帧分布左偏），过校准斜率 2-4 放大后帧级
    估值系统性偏高约 +0.12 管线值——F068（S 波）帧级 0.61（说成折叠），
    而虫级 0.39 与目视一致。帧级只作虫级采样失败时的兜底。
    """
    key = worm_key(rel)
    if key is None:
        return None
    if key in _WORM_EST:
        return _WORM_EST[key]
    raw, _ = _worm_mean_bbi(images_root, *key, n_frames=n_frames)
    val = calibrate(raw) if raw is not None else None
    _WORM_EST[key] = val
    return val


def seed_worm_estimates(mapping):
    """把外部算好的虫级估值灌进会话缓存（如预填表的 bdi 列，免预热）。"""
    for k, v in (mapping or {}).items():
        if k not in _WORM_EST:
            _WORM_EST[k] = v


def distinct_worms(items):
    """批次 [(item_id, rel_path), ...] 里去重的虫键列表（预热用）。"""
    keys, seen = [], set()
    for _, rel in items:
        k = worm_key(rel)
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def start_worm_warm(items, images_root, n_frames=10):
    """后台守护线程：为批次里每只虫算校准 BDI 估值（徽章显示用）。

    每虫 10 帧等距采样（约 1.5s/虫），400 只约 10 分钟跑完；没跑完的项
    在翻页到时按需现算（同一条代码路径）。任何异常只在控制台报告。
    """
    import threading

    def work():
        try:
            keys = distinct_worms(items)
            for i, k in enumerate(keys):
                if k in _WORM_EST:
                    continue
                raw, _ = _worm_mean_bbi(images_root, *k, n_frames=n_frames)
                _WORM_EST[k] = calibrate(raw) if raw is not None else None
                if (i + 1) % 25 == 0:
                    n_ok = sum(1 for v in _WORM_EST.values() if v is not None)
                    print(f'[BDI] 虫级预热 {i + 1}/{len(keys)}'
                          f'（成功 {n_ok}）', flush=True)
            n_ok = sum(1 for v in _WORM_EST.values() if v is not None)
            print(f'[BDI] 虫级预热完成 {len(keys)} 只（成功 {n_ok}，'
                  f'口径=管线校准）', flush=True)
        except Exception as e:
            print(f'[BDI] 虫级预热中断（不影响标注）：{e}', flush=True)

    t = threading.Thread(target=work, name='bdi-worm-warm', daemon=True)
    t.start()
    return t


def calibrate(v, json_path=DEFAULT_JSON, cand_csv=DEFAULT_CAND,
              images_root=''):
    """本地 BBI → 管线口径 BDI 估值（分段线性 + 端值 clamp）。"""
    if v is None:
        return None
    d = _CACHE['fit'] or load(json_path, cand_csv, images_root, verbose=False)
    knots = d.get('knots') or d.get('worm_knots')
    xs = [k[0] for k in knots]
    ys = [k[1] for k in knots]
    return float(np.interp(v, xs, ys))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='拟合本地→管线 BDI 校准映射')
    ap.add_argument('--cand-csv', default=DEFAULT_CAND)
    ap.add_argument('--images', default='', help='图片根目录')
    ap.add_argument('--out', default=DEFAULT_JSON)
    ap.add_argument('--n-worm-frames', type=int, default=10)
    args = ap.parse_args()
    if not args.images:
        for p in ('E:/data_new/images', os.path.join(A3, 'images'),
                  'D:/data_new/images'):
            if os.path.isdir(p) and any(
                    os.path.isdir(os.path.join(p, g)) for g in
                    ('control', '5mg', '8mg', '10mg')):
                args.images = p
                break
    fit(args.cand_csv, args.images, args.out, args.n_worm_frames)
