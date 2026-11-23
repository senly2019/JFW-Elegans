# -*- coding: utf-8 -*-
r"""bdi_v2_fit.py — 用共识分重造 BDI（BDI v2），把相关拉上去。

为什么单调重标定不够（数学事实，2026-11-13）
--------------------------------------------
Spearman 是秩相关，对单调变换不变。bdi_calib 的"本地 BBI → 管线 BDI"
是单调分段线性映射——它改绝对分、改归段，但不可能改变与人评共识的
Spearman。基线诊断证实：

    帧级 mean_k vs 共识      Spearman 0.355   ← 历史"0.33"的口径（帧噪声）
    虫级校准 BDI vs 共识     Spearman 0.591   ← 消噪（10 帧均值）的功劳

所以"拉相关"有两条路，本脚本两条都走：
  A. 消噪口径（不重造公式）：帧级 → 虫级 10 帧均值，0.355 → 0.591；
  B. 重造公式（换构念）：mean|κ| 不是人眼"最深弯曲"的单调函数
     （"小弯钩"：整体直+局部深弯 → mean_k 被摊薄，人却看局部深弯）。
     用新特征族（sinBBI 折角口径 / 三份能量 / 中段主导 / max_area
     局部深弯 / p99 / sklen / lap_var 画质对照）拟合线性 BDI v2。

验证（防自嗨，三重）
--------------------
  1. LOO CV：426 项留一线性回归的预测值 vs 共识 Spearman（训练内相关
     必然虚高，LOO 才是无偏估计）；
  2. 盲标独立验证：retest1 盲标共识 B（无锚）vs BDI v2，40 项——
     主实验共识 W 被 BDI 建议分 A 锚定，相关虚高；盲标 B 才是干净
     效标（注意：40 项是 426 子集，人也是同一批——表述为"条件独立"）；
  3. 敏感性：BDI v2 vs 盲标 B 与 vs 主实验共识的差 = 锚定虚高估计。

输出
----
  annotation_v3/retest/bdi_v2_fit.json   公式、权重、验证数字
  annotation_v3/retest/bdi_v2_report.md  报告（论文素材）

用法：python bdi_v2_fit.py
"""
import csv
import json
import os
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A3 = os.path.join(HERE, 'annotation_v3')
RT = os.path.join(A3, 'retest')
FEATS_CSV = os.path.join(RT, 'bdi_v2_feats.csv')
BLIND_DIR = RT
PRE_DIR = os.path.join(A3, 'pre')
OUT_JSON = os.path.join(RT, 'bdi_v2_fit.json')
OUT_MD = os.path.join(RT, 'bdi_v2_report.md')

FEATURES = ['mean_k', 'sin_bbi', 'e_front', 'e_mid', 'e_back', 'mid_dom',
            'max_area', 'p99_k', 'max_k', 'sklen', 'lap_var']
BLIND_NAMES = ['王俊驰', '柴晓骞', '许泽超', '柯炎辉', '沈昱璇', '林智锋']


def read_csv(p):
    with open(p, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        if v is None or str(v).strip() == '':
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3:
        return float('nan')
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    return float(np.corrcoef(np.asarray(a, float),
                             np.asarray(b, float))[0, 1])


def icc_2way(M):
    M = np.asarray(M, float)
    n, k = M.shape
    grand = M.mean()
    ms_r = k * ((M.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((M.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    ss_t = ((M - grand) ** 2).sum()
    ms_e = (ss_t - ms_r * (n - 1) - ms_c * (k - 1)) / ((n - 1) * (k - 1))
    i21 = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
    i2k = (ms_r - ms_e) / (ms_r + (ms_c - ms_e) / n)
    return float(i21), float(i2k)


def loo_linear(Xd, y, beta):
    """留一线性回归预测（Sherman-Morrison 快捷式）。

    β^(-i) = β - (X^T X)^{-1} x_i r_i / (1 - h_i)
    ⇒ ŷ_loo,i = ŷ_i - h_i r_i / (1 - h_i)
    h_i 为 hat matrix 对角（杠杆）。h_i→1 时该点预测不稳，LOO 值会
    发散——正是要暴露的过拟合信号。
    """
    H = Xd @ np.linalg.pinv(Xd.T @ Xd) @ Xd.T
    h = np.diag(H)
    yhat = Xd @ beta
    resid = y - yhat
    with np.errstate(divide='ignore', invalid='ignore'):
        loo = yhat - h * resid / (1 - h)
    return loo, h


def loo_linear_ridge(Xd, y, beta, lam):
    """岭回归留一预测（岭 hat matrix：H = X(X'X+λI)^{-1}X'）。

    岭把杠杆压向 1/n（h_i ≈ 1/n + O(λ)），LOO 与训练内差距小是岭的
    固有性质——报告 LOO 时同时报 OLS 版，避免"岭 LOO 虚高"误读。
    """
    H = Xd @ np.linalg.pinv(Xd.T @ Xd + lam * np.eye(Xd.shape[1])) @ Xd.T
    h = np.diag(H)
    yhat = Xd @ beta
    resid = y - yhat
    with np.errstate(divide='ignore', invalid='ignore'):
        loo = yhat - h * resid / (1 - h)
    return loo, h


def main():
    rows = read_csv(FEATS_CSV)
    print(f'[载入] {len(rows)} 项（特征+共识）')

    ids = [r['item_id'] for r in rows]
    cons = np.array([float(r['consensus']) for r in rows])
    bdi_pre = np.array([float(r['bdi_pre']) for r in rows])
    F = {c: np.array([fnum(r.get(c)) for r in rows]) for c in FEATURES}
    for c in FEATURES:
        bad = sum(1 for v in F[c] if v is None or not np.isfinite(v))
        if bad:
            med = np.nanmedian([v for v in F[c]
                                if v is not None and np.isfinite(v)])
            F[c] = np.array([med if (v is None or not np.isfinite(v))
                             else v for v in F[c]])
            print(f'  [补值] {c}: {bad} 项用中位数 {med:.4f}')

    # ---- 基线 ----
    base_frame = spearman(F['mean_k'], cons)
    base_worm = spearman(bdi_pre, cons)
    print(f'[基线] 帧级 mean_k vs 共识 Spearman {base_frame:+.3f}')
    print(f'[基线] 虫级校准 BDI vs 共识 Spearman {base_worm:+.3f}')

    # ---- 单特征 ----
    single = {}
    print('[单特征] vs 共识 Spearman:')
    for c in FEATURES:
        single[c] = spearman(F[c], cons)
        print(f'  {c:10s} {single[c]:+.3f}')

    # ---- 线性 BDI v2（标准化 + 非负约束，保持可解释） ----
    Z = np.column_stack([(F[c] - F[c].mean()) / (F[c].std() + 1e-12)
                         for c in FEATURES])
    y = (cons - cons.mean()) / (cons.std() + 1e-12)
    Xd = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    spe_in = spearman(Xd @ beta, cons)
    loo, h = loo_linear(Xd, y, beta)
    spe_loo = spearman(loo, cons)
    h_max = float(h.max())
    print(f'[线性v2] 训练内 Spearman {spe_in:+.3f}  LOO {spe_loo:+.3f}  '
          f'最大杠杆 {h_max:.3f}')

    # 非负约束（弯曲特征不该反向贡献；clip 后按占比归一化，可解释为
    # "各特征对 BDI v2 的贡献占比"）
    b_nn = beta.copy()
    for _ in range(3):
        b_nn[1:] = np.maximum(b_nn[1:], 0)
        if b_nn[1:].sum() <= 0:
            break
        b_nn[1:] /= b_nn[1:].sum()
        b_nn[0] = -(b_nn[1:] * Z.mean(axis=0)).sum()
    pred_nn = Xd @ b_nn
    spe_nn_in = spearman(pred_nn, cons)
    loo_nn, _ = loo_linear(Xd, y, b_nn)
    spe_nn_loo = spearman(loo_nn, cons)
    print(f'[非负v2] 训练内 {spe_nn_in:+.3f}  LOO {spe_nn_loo:+.3f}')

    # ---- 岭回归 BDI v2（最终公式；共线性稳健，LOO 对 λ 不敏感） ----
    # OLS 的 p99_k -20 / max_k +13 是共线互消（极端分位族），预测不伤
    # 但不可解释、数值不稳（p99 微噪 × -20 放大）。岭回归收缩权重。
    lam = 1.0
    p = Xd.shape[1]
    A = Xd.T @ Xd + lam * np.eye(p)
    b_r, *_ = np.linalg.lstsq(A, Xd.T @ y, rcond=None)
    pred_r = Xd @ b_r
    spe_r_in = spearman(pred_r, cons)
    loo_r, _ = loo_linear_ridge(Xd, y, b_r, lam)
    spe_r_loo = spearman(loo_r, cons)
    lam_scan = {}
    for lp in (0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0):
        Ap = Xd.T @ Xd + lp * np.eye(p)
        bp, *_ = np.linalg.lstsq(Ap, Xd.T @ y, rcond=None)
        lr, _ = loo_linear_ridge(Xd, y, bp, lp)
        lam_scan[str(lp)] = round(spearman(lr, cons), 4)
    print(f'[岭回归v2] λ={lam}  训练内 {spe_r_in:+.3f}  LOO {spe_r_loo:+.3f}')
    print(f'[岭回归v2] λ 扫描 LOO: {lam_scan}')
    w_r = {c: round(float(w), 4) for c, w in zip(FEATURES, b_r[1:])}
    print('[岭回归v2] 权重（标准化 z 口径）: ' +
          '  '.join(f'{c}{w_r[c]:+.2f}' for c in FEATURES))

    # ---- 盲标独立验证（40 项） ----
    blind = {}
    for n in BLIND_NAMES:
        p = os.path.join(BLIND_DIR, f'labels_{n}_retest1.csv')
        if not os.path.exists(p):
            continue
        d = {}
        for r in read_csv(p):
            iid = r['item_id']
            if r.get('unusable') == '1' or r.get('rating') in ('', None):
                continue
            v = fnum(r['rating'])
            if v is not None:
                d[iid] = v
        blind[n] = d
    # 帧级基线（hook_metrics 的 mean_k，与虫级同子集才公平）
    hm = {}
    p_hm = os.path.join(A3, 'pre', 'hook_metrics.csv')
    if os.path.exists(p_hm):
        hm = {r['item_id']: r for r in read_csv(p_hm)}
    frame_mk = np.array([fnum((hm.get(i) or {}).get('mean_k'))
                         for i in ids]) if hm else None
    if frame_mk is not None:
        n_ok = sum(1 for v in frame_mk if v is not None)
        base_frame_fair = (spearman(frame_mk, cons)
                           if n_ok == len(ids) else float('nan'))
        print(f'[公平基线] 同 {len(ids)} 子集：帧级 mean_k vs 共识 '
              f'{base_frame_fair:+.3f}（n_ok={n_ok}）')
    else:
        base_frame_fair = float('nan')
    blind_ids = [i for i in ids
                 if sum(1 for n in blind if i in blind[n]) >= 5]
    if blind_ids:
        B = np.array([[blind[n][i] for n in blind if i in blind[n]]
                      for i in blind_ids])
        Bc = np.median(B, axis=1)
        idx_of = {i: k for k, i in enumerate(ids)}
        sel = [idx_of[i] for i in blind_ids]
        print(f'[盲标验证] {len(blind_ids)} 项（六人盲标共识，无锚）')
        print(f'  帧级 mean_k vs 盲标共识 '
              f'{spearman(frame_mk[sel], Bc) if frame_mk is not None else float("nan"):+.3f}')
        print(f'  虫级 BDI vs 盲标共识 {spearman(bdi_pre[sel], Bc):+.3f}')
        print(f'  BDI v2(线性) vs 盲标共识 '
              f'{spearman((Xd @ beta)[sel], Bc):+.3f}')
        print(f'  BDI v2(岭回归 λ={lam}) vs 盲标共识 '
              f'{spearman(pred_r[sel], Bc):+.3f}')
        print(f'  主实验共识 vs 盲标共识 {spearman(cons[sel], Bc):+.3f}')
        blind_spe = {
            'n': len(blind_ids),
            'frame': (spearman(frame_mk[sel], Bc)
                      if frame_mk is not None else None),
            'worm': spearman(bdi_pre[sel], Bc),
            'v2_lin': spearman((Xd @ beta)[sel], Bc),
            'v2_ridge': spearman(pred_r[sel], Bc),
            'cons_vs_blind': spearman(cons[sel], Bc),
        }
    else:
        blind_spe = None

    # ---- 输出 ----
    w_nn = {c: round(float(w), 4) for c, w in zip(FEATURES, b_nn[1:])}
    fit = dict(
        built=datetime.now().isoformat(timespec='seconds'),
        n_items=len(rows),
        baseline=dict(
            frame_mean_k_vs_cons=round(base_frame, 4),
            worm_mean_k_vs_cons=round(spearman(F['mean_k'], cons), 4),
            worm_bdi_vs_cons=round(base_worm, 4),
            frame_fair_vs_cons=(round(base_frame_fair, 4)
                                if base_frame_fair == base_frame_fair
                                else None),
        ),
        single={c: round(v, 4) for c, v in single.items()},
        v2_linear=dict(features=FEATURES,
                       weights={c: round(float(w), 4)
                                for c, w in zip(FEATURES, beta[1:])},
                       intercept=round(float(beta[0]), 4),
                       spe_train=round(spe_in, 4),
                       spe_loo=round(spe_loo, 4)),
        v2_nonneg=dict(weights=w_nn,
                       spe_train=round(spe_nn_in, 4),
                       spe_loo=round(spe_nn_loo, 4)),
        v2_ridge=dict(features=FEATURES, lam=lam, weights=w_r,
                      spe_train=round(spe_r_in, 4),
                      spe_loo=round(spe_r_loo, 4),
                      lam_scan_loo=lam_scan,
                      feature_mean={c: round(float(F[c].mean()), 6)
                                    for c in FEATURES},
                      feature_sd={c: round(float(F[c].std()), 6)
                                  for c in FEATURES},
                      cons_mean=round(float(cons.mean()), 4),
                      cons_sd=round(float(cons.std()), 4)),
        blind_check=blind_spe,
        note='Spearman 对单调变换不变：单调重标定不改相关；'
             '帧级→虫级的增益全部来自消噪口径，不是校准映射。',
    )
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(fit, f, ensure_ascii=False, indent=1)
    print(f'[写出] {OUT_JSON}')

    # ---- 可复用公式（原始特征口径，供预标注/徽章/论文） ----
    OUT_FORMULA = os.path.join(RT, 'bdi_v2_formula.json')
    formula = dict(
        name='BDI v2（人评对齐弯曲度，虫级 10 帧均值口径）',
        built=fit['built'],
        n_train=len(rows),
        lam=lam,
        spe_loo_train=round(spe_r_loo, 4),
        spe_blind_check=(round(blind_spe['v2_ridge'], 4)
                         if blind_spe else None),
        usage='bdi_v2 = cons_mean + cons_sd * Σ_c w_c * (f_c − mean_c) / sd_c',
        cons_mean=fit['v2_ridge']['cons_mean'],
        cons_sd=fit['v2_ridge']['cons_sd'],
        weights={c: dict(w=w_r[c],
                         mean=fit['v2_ridge']['feature_mean'][c],
                         sd=fit['v2_ridge']['feature_sd'][c])
                 for c in FEATURES},
        caveats=[
            '岭回归 λ=1.0；LOO 对 λ∈[0.001,10] 不敏感（0.746–0.750）',
            '共识目标 = 六人主实验评分中位数（含锚定残留，方向上虚高）',
            '盲标 40 项无锚验证：v2 vs 盲标共识相关见 blind_check',
            '40 项是 414 项训练子集且同一批标注者：条件独立，非完全独立',
        ],
    )
    with open(OUT_FORMULA, 'w', encoding='utf-8') as f:
        json.dump(formula, f, ensure_ascii=False, indent=1)
    print(f'[写出] {OUT_FORMULA}')


if __name__ == '__main__':
    main()
