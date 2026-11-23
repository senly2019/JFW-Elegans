# -*- coding: utf-8 -*-
"""B6 去重复算：把 310 项（含 60 个同图隐藏重复）按图片分组重做 LOO。

动机：b6_feature_ceiling.py 的条目级 LOO 中，同图的重复项会同时落入训练折
与测试折（60/310 = 19.4%），绝对 AUC 被系统性抬高。本脚本给出三个口径：
  item  ：逐项 LOO（fPCA 全量拟合，复现服务器报告数字，作 sanity check）
  group ：按图片（rel_path）分组 LOO——同图全部副本一起留出，fPCA 基与
          标准化统计量均在训练折内估计（诚实口径）
  uniq  ：仅唯一图片（每图取首个条目）的逐项 LOO
另做配对差（S0-bbi、S2-S0、S2-bbi）的图片级自助检验。

用法：
  python b6_grouped_recheck.py --npz <b6_features.npz> --labels annotation_v2 \
      --out _recheck_b6
"""
import argparse
import csv
import os

import numpy as np

SET_NAMES = ("bbi", "S0", "S1", "S2")


def P(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 基础工具
def roc_auc(score, label):
    s = np.asarray(score, float)
    y = np.asarray(label, int)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ridge_loo_item(X, y, lam=1.0):
    """逐项 LOO 岭回归分数（与 b6_feature_ceiling.ridge_loo_score 同口径）。"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, d = X.shape
    out = np.zeros(n)
    eye = np.eye(d)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        mu = X[m].mean(0)
        sd = X[m].std(0)
        sd[sd < 1e-12] = 1.0
        Xtr = (X[m] - mu) / sd
        Xte = (X[i] - mu) / sd
        beta = np.linalg.pinv(Xtr.T @ Xtr + lam * eye) @ (Xtr.T @ y[m])
        out[i] = float(Xte @ beta)
    return out


def sym_prof(kap, wpr):
    return 0.5 * (kap + kap[:, ::-1]), 0.5 * (wpr + wpr[:, ::-1])


def fpca_project(sym, n_comp=3, basis=None):
    """basis=None 时自身拟合并投影；否则用给定 (mu, Vt) 投影。"""
    if basis is None:
        mu = sym.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(sym - mu, full_matrices=False)
        basis = (mu, Vt[:n_comp])
    mu, Vt = basis
    return (sym - mu) @ Vt.T, basis


def build_raw(kap, wpr, bbi, rest):
    """返回 dict：原始列（未拟合依赖）：bbi 列、宽度 5 标量。"""
    w_scal = np.array([[r["w_med"], r["w_max"], r["w_fat"], r["taper"], r["w_cv"]]
                       for r in rest], float)
    return bbi.reshape(-1, 1), w_scal


def sets_from_parts(bbi_col, kap_sc, w_scal, w_sc):
    S0 = np.hstack([bbi_col, kap_sc])
    S1 = np.hstack([w_scal, w_sc])
    S2 = np.hstack([S0, S1])
    return {"bbi": bbi_col, "S0": S0, "S1": S1, "S2": S2}


def boot_paired_auc(sc_a, sc_b, y, groups, n_boot=4000, seed=11):
    """配对自助：按图片组重采样，给出 AUC(a)-AUC(b) 的 95% CI 与双侧 p。

    注意：必须重算 AUC 差（不是分数均值差）——LOO 分数的秩变才会改变 AUC，
    均值差为 0 与 AUC 差为 +0.06 完全可以并存。
    """
    rng = np.random.RandomState(seed)
    uniq = sorted(set(groups))
    gidx = [np.array([i for i, g in enumerate(groups) if g == gname])
            for gname in uniq]
    obs = roc_auc(sc_a, y) - roc_auc(sc_b, y)
    diffs = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([gidx[p] for p in pick])
        if len(set(y[idx].tolist())) < 2:
            continue
        diffs.append(roc_auc(sc_a[idx], y[idx]) - roc_auc(sc_b[idx], y[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = float((np.abs(diffs - obs) >= abs(obs) - 1e-15).mean())
    return float(obs), float(lo), float(hi), p


# ---------------------------------------------------------------- 数据
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="_recheck_b6")
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.npz, allow_pickle=True)
    iids = list(z["iids"])
    kap_all = np.asarray(z["kappa"], float)
    wpr_all = np.asarray(z["wprof"], float)
    bbi_all = np.asarray(z["bbi"], float)
    rest_all = list(z["rest"])

    key = {r["item_id"]: r
           for r in read_csv(os.path.join(args.labels, "key.csv"))}
    rows_out = []
    for who in ("A", "B", "C"):
        lab = {r["item_id"]: int(r["label_code"])
               for r in read_csv(os.path.join(args.labels, f"labels_{who}.csv"))
               if (r.get("label_code") or "").strip()}
        sel = np.array([k for k, iid in enumerate(iids)
                        if iid in lab and lab[iid] != 0])
        y = np.array([1 if lab[iids[k]] in (2, 3) else 0 for k in sel])
        groups = [key[iids[k]]["rel_path"] for k in sel]
        kap, wpr, bbi, rest = (kap_all[sel], wpr_all[sel], bbi_all[sel],
                               [rest_all[k] for k in sel])
        bbi_col, w_scal = build_raw(kap, wpr, bbi, rest)
        kap_sym, w_sym = sym_prof(kap, wpr)

        P("=" * 76)
        P(f"标注者 {who}：{len(sel)} 项 / {len(set(groups))} 张唯一图"
          f"（正 {int(y.sum())} 负 {int((y == 0).sum())}）")
        P("=" * 76)

        # --- 口径 1：逐项 LOO（fPCA 全量拟合，复现服务器）---
        kap_sc_all, _ = fpca_project(kap_sym)
        w_sc_all, _ = fpca_project(w_sym)
        sets_all = sets_from_parts(bbi_col, kap_sc_all, w_scal, w_sc_all)
        item_auc = {k: roc_auc(ridge_loo_item(X, y), y)
                    for k, X in sets_all.items()}
        P("  [item ] 逐项 LOO（fPCA 全量拟合，复现服务器口径）: " +
          "  ".join(f"{k}={item_auc[k]:.3f}" for k in SET_NAMES))

        # --- 口径 2：按图片分组 LOO（训练折内估 fPCA 基与标准化量）---
        uniq = sorted(set(groups))
        sc_g = {k: np.zeros(len(y)) for k in SET_NAMES}
        eye = {d: np.eye(d) for d in (1, 4, 8, 12)}
        for g in uniq:
            te = np.array([gg == g for gg in groups])
            tr = ~te
            # 训练折 fPCA 基
            mu_k = kap_sym[tr].mean(0, keepdims=True)
            _, _, Vt_k = np.linalg.svd(kap_sym[tr] - mu_k, full_matrices=False)
            mu_w = w_sym[tr].mean(0, keepdims=True)
            _, _, Vt_w = np.linalg.svd(w_sym[tr] - mu_w, full_matrices=False)
            kap_sc = {None: None}
            ktr = (kap_sym[tr] - mu_k) @ Vt_k[:3].T
            kte = (kap_sym[te] - mu_k) @ Vt_k[:3].T
            wtr = (w_sym[tr] - mu_w) @ Vt_w[:3].T
            wte = (w_sym[te] - mu_w) @ Vt_w[:3].T
            parts = {
                "bbi": (bbi_col[tr], bbi_col[te]),
                "S0": (np.hstack([bbi_col[tr], ktr]), np.hstack([bbi_col[te], kte])),
                "S1": (np.hstack([w_scal[tr], wtr]), np.hstack([w_scal[te], wte])),
                "S2": (np.hstack([bbi_col[tr], ktr, w_scal[tr], wtr]),
                       np.hstack([bbi_col[te], kte, w_scal[te], wte])),
            }
            for name, (Xtr, Xte) in parts.items():
                mu = Xtr.mean(0)
                sd = Xtr.std(0)
                sd[sd < 1e-12] = 1.0
                Xtr_s = (Xtr - mu) / sd
                Xte_s = (Xte - mu) / sd
                beta = np.linalg.pinv(Xtr_s.T @ Xtr_s + eye[Xtr.shape[1]]) \
                    @ (Xtr_s.T @ y[tr])
                sc_g[name][te] = Xte_s @ beta
        group_auc = {k: roc_auc(sc_g[k], y) for k in SET_NAMES}
        P("  [group] 按图分组 LOO（同图副本一起留出，训练折内估计）: " +
          "  ".join(f"{k}={group_auc[k]:.3f}" for k in SET_NAMES))

        # --- 口径 3：仅唯一图（每图取首个条目）---
        seen, uidx = set(), []
        for i, g in enumerate(groups):
            if g not in seen:
                seen.add(g)
                uidx.append(i)
        uidx = np.array(uidx)
        kap_su, w_su = sym_prof(kap[uidx], wpr[uidx])
        kap_scu, _ = fpca_project(kap_su)
        w_scu, _ = fpca_project(w_su)
        sets_u = sets_from_parts(bbi_col[uidx], kap_scu, w_scal[uidx], w_scu)
        yu = y[uidx]
        uniq_auc = {k: roc_auc(ridge_loo_item(X, yu), yu)
                    for k, X in sets_u.items()}
        P(f"  [uniq ] {len(uidx)} 张唯一图逐项 LOO: " +
          "  ".join(f"{k}={uniq_auc[k]:.3f}" for k in SET_NAMES))

        # --- 配对差（分组口径，图片级聚簇自助，重算 AUC 差）---
        for pair, (a, b) in (("S0-bbi", ("S0", "bbi")),
                             ("S2-S0", ("S2", "S0")),
                             ("S2-bbi", ("S2", "bbi"))):
            d, lo, hi, p = boot_paired_auc(sc_g[a], sc_g[b], y, groups,
                                           n_boot=args.n_boot)
            P(f"  配对差 {pair:>7s}: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  p = {p:.4f}（图片级自助 {args.n_boot} 次）")
            rows_out.append([who, pair, f"{d:+.4f}",
                             f"[{lo:+.4f}, {hi:+.4f}]", f"{p:.4f}"])
        P()
        for name in SET_NAMES:
            rows_out.append([who, f"auc_{name}", f"{item_auc[name]:.4f}",
                             f"{group_auc[name]:.4f}", f"{uniq_auc[name]:.4f}"])

    csv_path = os.path.join(args.out, "b6_recheck.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rater", "quantity", "value", "extra1", "extra2"])
        for r in rows_out:
            w.writerow(r)
    P(f"明细：{csv_path}")


if __name__ == "__main__":
    main()
