# -*- coding: utf-8 -*-
"""b6_feature_ceiling.py —— 判定性实验（论文附录 B6 / 4.2 节）：

曲率族特征对"人眼判的弯与直"的判别力止于 0.71 附近（弯曲指数在 310 帧
含重复口径下三人 AUC 0.623/0.712/0.671）。加入体宽、锥度等非曲率线索后，
AUC 能否越过这条线？两个方向的答案都能判定。

设计要点
--------
1. 数据：annotation_v2/ 的 key.csv（310 项，含 60 个隐藏重复）+ labels_A/
   B/C.csv（三人作答）。二值化与 step18_agreement_v2.py 一致：
   label_code 1（笔直）-> 0，2（Ω）/3（正弦）-> 1；0（不可用）/9（无法归类）
   剔除。0.71 口径 = 310 帧含重复，故本脚本主口径同为全 310 项。
2. 方向对称化：key.csv 无 head_is_first 列，曲率轮廓与宽度轮廓一律取
   双边对称 kap_sym(s) = (κ(s)+κ(1-s))/2、w_sym(s) = (w(s)+w(1-s))/2，
   头尾翻转与镜像同时消除；锥度取 |log(w_head/w_tail)| 的绝对形式。
   bbi = mean(κ) 本身方向不变，直接复用 key.csv 的值并作校验锚点。
3. 特征集：
   S0 曲率族（复现基线）：bbi + fPCA(kap_sym, 3)
   S1 宽度族（新线索）：中位宽/最大宽/粗端占比/锥度/宽度变异系数 + fPCA(w_sym, 3)
   S2 联合：S0 + S1
4. 判别框架（无训练泄漏）：
   - bbi 单特征直接 Mann-Whitney AUC（与 step18 同口径，锚定 0.71）；
   - 多特征用岭回归闭型 LOO（PRESS 恒等式：一次求逆得全部 LOO 预测）
     与 SVM LOO 双学习器对照，AUC 均取 Mann-Whitney U 口径；
   - 95% 自助 CI：分层重采样 1000 次；
   - 配对零分布：标签置换 4000 次，考察 (AUC_S2 - AUC_S0) 的差。
5. 路径解析：数据根用 wormpath.get_data_root()；组目录名若因 relabel_groups
   轮换而错位，遍历 6 种组名置换并用 key.csv 的 bbi 列校验（重提 bbi 与
   记录值最大偏差 < 0.01 即确认命中）。

用法（服务器）：
  python b6_feature_ceiling.py --labels annotation_v2 --out b6_report
输出：b6_report/b6_feature_ceiling.txt（主报告）
      b6_report/b6_by_rater.csv（每人 × 特征集 AUC 表）
      b6_report/b6_features.npz（特征缓存，复算不重提）
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wormpath import get_data_root          # noqa: E402
from worm_labeler import extract_skeleton_full   # noqa: E402

CODE_STRAIGHT, CODE_OMEGA, CODE_SIN = 1, 2, 3     # label_code 语义
GROUPS = ["control", "5mg", "8mg", "10mg"]
N_PTS = 100                                        # 与 step8 同款归一化点数


def P(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 工具
def roc_auc(score, label):
    """Mann-Whitney U 形式 AUC（并列取平均秩），与 step18 / bdi_ci_check 一致。"""
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    r1 = ranks[y == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(score, label, n_boot=1000, seed=42):
    """分层自助 95% CI。"""
    rng = np.random.RandomState(seed)
    s = np.asarray(score, float); y = np.asarray(label, int)
    i1 = np.where(y == 1)[0]; i0 = np.where(y == 0)[0]
    if not len(i1) or not len(i0):
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(i1, len(i1), replace=True),
                              rng.choice(i0, len(i0), replace=True)])
        v = roc_auc(s[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def ridge_loo_score(X, y, lam=1.0):
    """岭回归严格 LOO：每折在训练子集上重新标准化并求逆。

    不用 PRESS 闭式——闭式与循环式的差别在于标准化是否随折重算，
    两者都是合理的 LOO 变体但数值有差；为与 SVM LOO 口径一致
    （SVM 折内重新 fit scaler），这里用严格循环，n≈310 时仍是毫秒级。
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, d = X.shape
    out = np.zeros(n)
    eye = np.eye(d)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        mu = X[m].mean(0)
        sd = X[m].std(0); sd[sd < 1e-12] = 1.0
        Xtr = (X[m] - mu) / sd
        Xte = (X[i] - mu) / sd
        beta = np.linalg.pinv(Xtr.T @ Xtr + lam * eye) @ (Xtr.T @ y[m])
        out[i] = float(Xte @ beta)
    return out


def svm_loo_score(X, y, seed=42):
    """SVM LOO（标准化 + RBF，class_weight 平衡）。返回 OOF decision_function。"""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, float); y = np.asarray(y, int)
    n = len(y)
    out = np.zeros(n)
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        sc = StandardScaler().fit(X[tr])
        clf = SVC(kernel="rbf", C=1.0, class_weight="balanced",
                  random_state=seed).fit(sc.transform(X[tr]), y[tr])
        out[i] = clf.decision_function(sc.transform(X[i:i + 1]))[0]
    return out


def fpca_sym(profiles, n_comp=3):
    """双边对称化轮廓的函数 PCA：返回 (对称轮廓矩阵, 主分量得分)。

    对称化同时消除头尾翻转与镜像：p_sym(s) = (p(s)+p(1-s))/2。
    """
    Pm = np.asarray(profiles, float)
    sym = 0.5 * (Pm + Pm[:, ::-1])
    mu = sym.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(sym - mu, full_matrices=False)
    k = min(n_comp, Vt.shape[0])
    score = (sym - mu) @ Vt[:k].T
    evr = (S[:k] ** 2 / (S ** 2).sum())
    return sym, score, evr


# ---------------------------------------------------------------- 数据
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_annotation(label_dir):
    """读 key.csv + labels_X.csv，返回 (items, answers)。

    items: [(item_id, rel_path, is_repeat, bbi_key)]
    answers: {rater: {item_id: code}}
    """
    key = read_csv(os.path.join(label_dir, "key.csv"))
    items = [(r["item_id"], r["rel_path"], int(r.get("is_repeat", 0) or 0),
              float(r["bbi"]) if r.get("bbi") else float("nan"))
             for r in key]
    answers = {}
    for who in ("A", "B", "C"):
        p = os.path.join(label_dir, f"labels_{who}.csv")
        if not os.path.exists(p):
            p = os.path.join(label_dir, f"items_{who}.csv")
        if not os.path.exists(p):
            continue
        d = {}
        for r in read_csv(p):
            raw = (r.get("label_code") or "").strip()
            if raw:
                try:
                    d[r["item_id"]] = int(raw)
                except ValueError:
                    pass
        answers[who] = d
    return items, answers


def binarize(codes):
    """label_code -> 二值标签；剔除 0/9 与非三类码。"""
    y = {}
    for iid, c in codes.items():
        if c == CODE_STRAIGHT:
            y[iid] = 0
        elif c in (CODE_OMEGA, CODE_SIN):
            y[iid] = 1
    return y


# ---------------------------------------------------------------- 特征提取
def _width_stats(mask, pts):
    """沿中心线的宽度统计（距离变换，与 worm_labeler._width_profile 同源）。"""
    import cv2
    if mask is None or pts is None or len(pts) < 4:
        return None
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    p = np.round(np.asarray(pts, float)).astype(int)
    p[:, 0] = np.clip(p[:, 0], 0, dt.shape[1] - 1)
    p[:, 1] = np.clip(p[:, 1], 0, dt.shape[0] - 1)
    w = 2.0 * dt[p[:, 1], p[:, 0]]
    k = max(1, len(w) // 10)
    core = np.sort(w[k:-k]) if len(w) > 2 * k + 2 else np.sort(w)
    if not len(core):
        return None
    med = float(np.median(core))
    mx = float(core[-1])
    fat = float((core > 1.45 * med).mean()) if med > 0 else 0.0
    return w, med, mx, fat


def extract_one(args):
    """worker：单帧提取曲率族 + 宽度族原始量。返回 dict 或 None。

    口径对齐：曲率用 worm_labeler.compute_curvature_profile（7 点窗口，
    与 step8/kappa_cache/key.csv 的 bbi 同源，保证 bbi 校验锚点有效）；
    宽度剖面按中心线点的弧长坐标插值（点级坐标，避免段级错位）。
    """
    import cv2
    from scipy.interpolate import splrep, splev
    from worm_labeler import compute_curvature_profile
    iid, img_path = args
    try:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        points, _mth, mask_final, _ft = extract_skeleton_full(img)
        if points is None or len(points) < 10:
            return None
        pts = np.asarray(points, float)
        # ---- 曲率轮廓（worm_labeler 口径：7 点窗口曲率，与 step8 同源）----
        kappa = np.asarray(compute_curvature_profile(points, N_PTS), float)
        kappa = np.clip(kappa, 0, None)
        if kappa.sum() < 1e-6:
            return None
        # ---- 宽度剖面（点级弧长坐标对齐）----
        ws = _width_stats(mask_final, points)
        if ws is None:
            return None
        w_raw, w_med, w_max, w_fat = ws
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        total = cum[-1]
        if total < 1e-6:
            return None
        s_pt = cum / total                       # 点级弧长坐标（与 w_raw 等长）
        try:
            tck = splrep(s_pt, w_raw, k=3, s=len(w_raw) * 0.1)
            wprof = np.asarray(splev(np.linspace(0, 1, N_PTS), tck), float)
        except Exception:
            wprof = np.interp(np.linspace(0, 1, N_PTS), s_pt, w_raw)
        wprof = np.clip(wprof, 0, None)
        # 锥度（头尾 10% 段中位宽之比，取绝对对数——方向无关）
        k = max(1, N_PTS // 10)
        wh = float(np.median(wprof[:k]))
        wt = float(np.median(wprof[-k:]))
        taper = abs(np.log((wh + 1e-6) / (wt + 1e-6)))
        w_cv = float(np.std(wprof) / max(np.mean(wprof), 1e-6))
        return dict(iid=iid, kappa=kappa.astype(np.float32),
                    wprof=wprof.astype(np.float32),
                    bbi=float(kappa.mean()), w_med=w_med, w_max=w_max,
                    w_fat=w_fat, taper=taper, w_cv=w_cv,
                    skel_len=float(total))
    except Exception:
        return None


def resolve_paths(items, root, workers):
    """组目录错位时遍历 6 种置换，用 bbi 校验命中。返回 [(iid, path)]。"""
    from itertools import permutations
    # 直接命中比例
    def probe(perm):
        hit = 0
        for iid, rel, _r, _b in items:
            g = rel.split("/")[0]
            g2 = perm.get(g, g)
            p = os.path.join(root, g2, *rel.split("/")[1:])
            if os.path.exists(p):
                hit += 1
        return hit
    direct = {g: g for g in GROUPS}
    if probe(direct) == len(items):
        P(f"[路径] 数据根直接命中全部 {len(items)} 帧")
        return [(iid, os.path.join(root, *rel.split("/")))
                for iid, rel, _r, _b in items], {g: g for g in GROUPS}
    best = None
    movable = [g for g in GROUPS if g != "control"]
    for perm_t in permutations(movable):
        perm = dict(zip(movable, perm_t)); perm["control"] = "control"
        h = probe(perm)
        if best is None or h > best[1]:
            best = (perm, h)
        if h == len(items):
            break
    perm = best[0]
    P(f"[路径] 组目录置换命中 {best[1]}/{len(items)}：{perm}")
    return [(iid, os.path.join(root, perm.get(rel.split('/')[0],
                                             rel.split('/')[0]),
                               *rel.split("/")[1:]))
            for iid, rel, _r, _b in items], perm


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(HERE, "annotation_v2"))
    ap.add_argument("--out", default=os.path.join(HERE, "b6_report"))
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=4000)
    ap.add_argument("--max-fit", type=int, default=50000)
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟模式：用合成虫图跑通全流程（不读真实数据）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    workers = args.workers or os.cpu_count() or 1

    root = get_data_root()
    P("=" * 72)
    P("B6 判定性实验：非曲率线索能否越过 0.71")
    P("=" * 72)
    P(f"数据根：{root}")

    items, answers = load_annotation(args.labels)
    P(f"[数据] {len(items)} 项（隐藏重复 "
      f"{sum(1 for _i, _r, rep, _b in items if rep)} 个）；"
      f"标注者 {sorted(answers)}")

    # ---- 提取（或读缓存） ----
    cache_npz = os.path.join(args.out, "b6_features.npz")
    if args.smoke:
        # 冒烟模式：伪造与合成帧对齐的 items/answers，标签由弯曲幅度生成
        feats = smoke_features(items, args)
        items = [(i, f"smoke/{i}.png", 0, float(feats[i]["bbi"]))
                 for i in feats]
        rng = np.random.RandomState(1)
        answers = {who: {i: (1 if rng.rand() < feats[i]["bbi"] * 2.5 else 2)
                         for i in feats}
                   for who in ("A", "B", "C")}
        P("[冒烟] 已用合成数据替换 items/answers（标签由 bbi 随机生成，"
          "仅验证流程可跑通，数值无意义）")
    elif os.path.exists(cache_npz):
        z = np.load(cache_npz, allow_pickle=True)
        feats = {i: dict(kappa=k, wprof=w, bbi=float(b), **rest)
                 for i, k, w, b, rest in
                 zip(z["iids"], z["kappa"], z["wprof"], z["bbi"], z["rest"])}
        P(f"[缓存] 已加载 {cache_npz}（{len(feats)} 帧）")
    else:
        jobs, _perm = resolve_paths(items, root, workers)
        P(f"[提取] {len(jobs)} 帧，{workers} 进程……")
        if workers > 1:
            with Pool(workers) as pool:
                res = pool.map(extract_one, jobs)
        else:
            res = [extract_one(j) for j in jobs]
        feats = {r["iid"]: r for r in res if r is not None}
        P(f"[提取] 有效 {len(feats)}/{len(jobs)} 帧")
        if not args.smoke:
            np.savez(cache_npz,
                     iids=np.array(list(feats)),
                     kappa=np.array([feats[i]["kappa"] for i in feats]),
                     wprof=np.array([feats[i]["wprof"] for i in feats]),
                     bbi=np.array([feats[i]["bbi"] for i in feats]),
                     rest=np.array([{k: v for k, v in feats[i].items()
                                     if k not in ("iid", "kappa", "wprof", "bbi")}
                                    for i in feats], dtype=object))
            P(f"[缓存] 已写入 {cache_npz}")

    # ---- 校验：重提 bbi vs key.csv 记录值 ----
    diffs = [abs(feats[i]["bbi"] - b) for i, _r, _rep, b in items
             if i in feats and np.isfinite(b)]
    if diffs:
        P(f"[校验] 重提 bbi 与 key.csv 记录值最大偏差 {max(diffs):.4f}"
          f"（<0.01 即路径命中正确）")

    # ---- 特征矩阵 ----
    ids = [i for i, _r, _rep, _b in items if i in feats]
    kap = np.array([feats[i]["kappa"] for i in ids])
    wpr = np.array([feats[i]["wprof"] for i in ids])
    kap_sym, kap_sc, kap_evr = fpca_sym(kap, 3)
    w_sym, w_sc, w_evr = fpca_sym(wpr, 3)
    bbi = np.array([feats[i]["bbi"] for i in ids])
    w_med = np.array([feats[i]["w_med"] for i in ids])
    w_max = np.array([feats[i]["w_max"] for i in ids])
    w_fat = np.array([feats[i]["w_fat"] for i in ids])
    taper = np.array([feats[i]["taper"] for i in ids])
    w_cv = np.array([feats[i]["w_cv"] for i in ids])
    P(f"[特征] 对称曲率轮廓 fPCA 解释率 {kap_evr.round(3)}；"
      f"对称宽度轮廓 fPCA 解释率 {w_evr.round(3)}")

    S0 = np.column_stack([bbi, kap_sc])                       # 曲率族
    S1 = np.column_stack([w_med, w_max, w_fat, taper, w_cv, w_sc])  # 宽度族
    S2 = np.column_stack([S0, S1])                            # 联合
    SETS = {"S0_curvature": S0, "S1_width": S1, "S2_joint": S2}

    # ---- 每人判定 ----
    is_rep = np.array([rep for i, _r, rep, _b in items if i in feats])
    rows, verdict_lines = [], []
    auc_map = {}
    for who in sorted(answers):
        y_map = binarize(answers[who])
        idx = [k for k, i in enumerate(ids) if i in y_map]
        if not idx:
            continue
        y = np.array([y_map[ids[k]] for k in idx])
        P()
        P(f"---- 标注者 {who}：有效作答 {len(idx)} 项"
          f"（正类 {int(y.sum())} / 负类 {int((y == 0).sum())}）----")
        P(f"  {'特征集':<14s}{'维度':>4s} {'AUC(LOO岭)':>11s} {'95% CI':>18s}"
          f" {'AUC(LOO-SVM)':>12s}")
        sub = {"S0_curvature": S0[idx], "S1_width": S1[idx],
               "S2_joint": S2[idx]}
        sc_map = {}
        for name, X in sub.items():
            sc_r = ridge_loo_score(X, y)
            auc_r = roc_auc(sc_r, y)
            lo, hi = boot_ci(sc_r, y, args.n_boot)
            sc_s = svm_loo_score(X, y)
            auc_s = roc_auc(sc_s, y)
            sc_map[name] = sc_r
            auc_map[(who, name)] = auc_r
            rows.append([who, name, X.shape[1], f"{auc_r:.3f}",
                         f"[{lo:.3f}, {hi:.3f}]", f"{auc_s:.3f}",
                         int(y.sum()), len(idx)])
            P(f"  {name:<14s}{X.shape[1]:>4d} {auc_r:>11.3f} "
              f"{'[' + format(lo, '.3f') + ', ' + format(hi, '.3f') + ']':>18s}"
              f" {auc_s:>12.3f}")
        # 配对零分布：S2 - S0（标签置换，分数固定重算 AUC）
        base = sc_map["S0_curvature"]; full = sc_map["S2_joint"]
        obs = roc_auc(full, y) - roc_auc(base, y)
        rng = np.random.RandomState(7)
        cnt = 0
        for _ in range(args.n_perm):
            yp = rng.permutation(y)
            d = roc_auc(full, yp) - roc_auc(base, yp)
            if abs(d) >= abs(obs) - 1e-12:
                cnt += 1
        p_perm = cnt / args.n_perm
        P(f"  配对差 AUC(S2)-AUC(S0) = {obs:+.3f}，"
          f"置换零分布 p = {p_perm:.4f}（{args.n_perm} 次）")
        verdict_lines.append((who, obs, p_perm))

    # ---- bbi 单特征锚点（310 帧含重复口径，对齐 0.71 的出处）----
    P()
    P("---- bbi 单特征直接 AUC（step18 同口径锚点，应复现 0.623/0.712/0.671）----")
    for who in sorted(answers):
        y_map = binarize(answers[who])
        idx = [k for k, i in enumerate(ids) if i in y_map]
        if not idx:
            continue
        y = np.array([y_map[ids[k]] for k in idx])
        a = roc_auc(bbi[idx], y)
        lo, hi = boot_ci(bbi[idx], y, args.n_boot)
        P(f"  标注者 {who}: AUC = {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
          f"（n = {len(idx)}，含重复 {int(is_rep[idx].sum())} 项）")

    # ---- 判定 ----
    # 判据是"联合集是否显著超过曲率族"（配对检验），不是全局最大 AUC。
    # （旧版 bug：把三个特征集的全局最大 AUC 误标为 S2——标注者 B 的 0.777
    #  实际来自 S0 曲率族——并据此宣称体宽/锥度线索有效。配对差全部不显著
    #  时，正确结论是"非曲率线索无显著增益"；若曲率族自身越过 0.71，越过
    #  它的是曲率轮廓而非宽度特征。）
    P()
    P("=" * 72)
    for name in ("S0_curvature", "S1_width", "S2_joint"):
        P(f"  {name:<14s} 三人 AUC: " +
          "  ".join(f"{who}={auc_map[(who, name)]:.3f}"
                    for who in sorted(answers) if (who, name) in auc_map))
    sig = [p < 0.05 for _who, _obs, p in verdict_lines]
    P()
    if any(sig):
        P("结论：**非曲率线索有显著增益**（联合集显著超过曲率族，"
          "至少一名标注者的配对检验 p < 0.05）。")
    else:
        P("结论：**非曲率线索无显著增益**——配对差（S2−S0）全部不显著。"
          "若曲率族自身越过 0.71，越过它的是曲率轮廓（多维表示），"
          "不是宽度/锥度特征；病因应到曲率族内部（一维标量改多维轮廓）"
          "找，而非归因于非曲率线索。")
    P("=" * 72)

    # ---- 落盘 ----
    rep = os.path.join(args.out, "b6_feature_ceiling.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("B6 判定性实验：非曲率线索能否越过 0.71\n")
        f.write(f"数据根 {root}；帧数 {len(ids)}；"
                f"自助 {args.n_boot} 次；置换 {args.n_perm} 次\n\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
        f.write("\n")
        for who, obs, p in verdict_lines:
            f.write(f"配对差(S2-S0) 标注者 {who}: {obs:+.3f}, p = {p:.4f}\n")
        f.write(f"\n判定：非曲率线索（S2 vs S0 配对差）"
                f"{'有' if any(p < 0.05 for _w, _o, p in verdict_lines) else '无'}"
                f"显著增益；三人各特征集 AUC 见上表\n")
    P(f"报告：{rep}")
    csv_path = os.path.join(args.out, "b6_by_rater.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rater", "feature_set", "dim", "auc_ridge_loo",
                    "ci95", "auc_svm_loo", "n_pos", "n_total"])
        w.writerows(rows)
    P(f"CSV：{csv_path}")


def smoke_features(items, args):
    """冒烟：合成弯曲虫形图，跑通提取与判定全流程。"""
    import cv2
    import tempfile
    rng = np.random.RandomState(0)
    tmp = tempfile.mkdtemp(prefix="b6_smoke_")
    n = 40
    out = {}
    for k in range(n):
        img = np.full((90, 80, 3), 40, np.uint8)     # 暗背景
        amp = rng.uniform(0, 14)                     # 弯曲幅度
        phase = rng.uniform(0, 2 * np.pi)
        width = rng.uniform(5, 9)
        pts = []
        for t in np.linspace(0, 1, 60):
            x = 8 + t * 64
            y = 45 + amp * np.sin(2 * np.pi * t + phase)
            pts.append((x, y))
        pts = np.array(pts, np.int32)
        cv2.polylines(img, [pts], False, (200, 200, 200),
                      thickness=int(width), lineType=cv2.LINE_AA)
        p = os.path.join(tmp, f"smoke_{k:03d}.png")
        cv2.imwrite(p, img)
        r = extract_one((f"S{k:03d}", p))
        if r is not None:
            out[f"S{k:03d}"] = r
    P(f"[冒烟] 合成 {n} 帧，有效提取 {len(out)} 帧 -> {tmp}")
    if len(out) < 10:
        raise SystemExit("冒烟失败：有效帧过少，检查合成图与提取链")
    return out


if __name__ == "__main__":
    main()
