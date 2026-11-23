# -*- coding: utf-8 -*-
"""bdi_ci_check.py —— 服务器补算：帧级 BDI 的效标 AUC、95% 自助 CI 与第三人 C。

背景：本机 kappa_cache.npz 已释放磁盘，帧级 BDI 无法在本机复算；本脚本在服务器
用全量缓存重算，口径与 step18_agreement_v2.py 的 bdi_lookup 完全一致：
  PCA(n_components=1, random_state=0).fit(kappa 全量矩阵)，按与缓存 features
  第 0 列（bbi）的正相关定向。补三件事：
  ① 三名标注者（A/B/C）各自对二值人工判断的 BDI AUC 与最佳切点 κ；
  ② 每人 AUC 的 95% 自助 CI（--n-boot 次，与 bbi 的 CI 同源可比）；
  ③ BDI 与 bbi（key.csv 帧级值）在同帧集上的配对差及 r=0.976 零分布模拟。

用法（服务器）：
  python bdi_ci_check.py --cache atlas_out_all/kappa_cache.npz
  # 内存不足时加 --max-frames 1000000（随机抽样 fit PC1，估计几乎不变）
输出：终端 + annotation_v2/bdi_ci_report.txt（含可直接回填论文表 5 C2 的文字）
"""
import argparse
import csv
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_TO_CLASS = {1: 2, 2: 1, 3: 0}  # label_code -> class（2=笔直, 1=Ω, 0=正弦）


def bin_of(cls):
    """二值化：笔直 -> 0，Ω/正弦 -> 1（与 step18 一致）。"""
    return 0 if cls == 2 else 1


def roc_auc(score, label):
    """Mann-Whitney U 形式 AUC（并列取平均秩），与 step18 一致。"""
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan"), n1, n0
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    r1 = ranks[y == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0)), n1, n0


def cohen_kappa(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    po = float((y == pred).mean())
    cats = sorted(set(y.tolist()) | set(pred.tolist()))
    pe = sum((y == c).mean() * (pred == c).mean() for c in cats)
    if abs(1 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1 - pe)


def best_cut(score, label):
    """5%-95% 分位扫描，准确率优先、κ 次之（与 step18 一致）。"""
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=int)
    cand = np.unique(np.quantile(s, np.linspace(0.05, 0.95, 91)))
    best = None
    for t in cand:
        pred = (s >= t).astype(int)
        acc = float((pred == y).mean())
        k = cohen_kappa(y, pred)
        key = (acc, k if k == k else -9)
        if best is None or key > best[0]:
            best = (key, float(t), acc, k)
    return dict(thr=best[1], acc=best[2], kappa=best[3])


def load_key(dirpath):
    key = {}
    with open(os.path.join(dirpath, "key.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key[r["item_id"]] = r
    return key


def load_labels(dirpath, rater):
    path = os.path.join(dirpath, f"labels_{rater}.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"[错误] 缺少 {path}——该文件尚未同步到服务器。\n"
            f"       请先 scp annotation_v2/labels_C.csv（及 key.csv）到服务器，"
            f"或在本机跑一次 sync_to_server.sh。"
        )
    ans = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ans[r["item_id"]] = int(r["label_code"])
    return ans


def pca_pc1(kappa_mat, feats, max_frames=0, seed=0):
    """PC1 投影，口径与 step18_agreement_v2.bdi_lookup 一致。"""
    if max_frames and len(kappa_mat) > max_frames:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(kappa_mat), max_frames, replace=False)
        kappa_mat = kappa_mat[idx]
        if feats is not None:
            feats = feats[idx]
    try:
        from sklearn.decomposition import PCA
        p = PCA(n_components=1, random_state=0).fit(kappa_mat)
        pc1 = p.transform(kappa_mat)[:, 0]
        evr = float(p.explained_variance_ratio_[0])
    except ImportError:
        Xc = kappa_mat - kappa_mat.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        pc1 = Xc @ vt[0]
        evr = float("nan")
    if feats is not None:
        try:
            corr = float(np.corrcoef(pc1, np.asarray(feats)[:, 0].astype(float))[0, 1])
        except Exception:
            corr = 0.0
        if corr < 0:
            pc1 = -pc1
        orient = f"与 bbi 相关 {abs(corr):+.3f}（已按此定向）"
    else:
        orient = "缓存无 features，符号未定向"
    return pc1, evr, orient


def main():
    ap = argparse.ArgumentParser(description="帧级 BDI 效标 AUC / 95% CI / 第三人 C")
    ap.add_argument("-c", "--cache", default="atlas_out_all/kappa_cache.npz",
                    help="kappa 缓存 npz（全量跑默认 atlas_out_all/）")
    ap.add_argument("-d", "--dir", default="annotation_v2")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="内存不足时抽样 fit PC1 的帧数上限（0=全量）")
    ap.add_argument("--r-target", type=float, default=0.9761,
                    help="BDI-bbi 相关的零分布模拟目标（表 5 C1）")
    args = ap.parse_args()

    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    if not os.path.exists(args.cache):
        P(f"[错误] 缓存不存在：{args.cache}")
        for root in ("atlas_out_all", "atlas_out_full", "."):
            if os.path.isdir(root):
                nps = [f for f in os.listdir(root) if f.endswith(".npz")]
                if nps:
                    P(f"  {root}/ 下的 npz：{nps}")
        raise SystemExit(1)

    key = load_key(args.dir)
    P("=" * 68)
    P("帧级 BDI 补算：效标 AUC / 95% 自助 CI / 第三人 C")
    P(f"缓存：{args.cache}（{'全量' if not args.max_frames else f'抽样 {args.max_frames} 帧'}）")
    P("=" * 68)

    wanted = {k["rel_path"] for k in key.values() if k["is_repeat"] == "0"}
    P(f"[1/3] 载入缓存并拟合 PC1（标注唯一项 {len(wanted)} 帧）……")
    z = np.load(args.cache, allow_pickle=True)
    files = [str(x).replace("\\", "/") for x in z["filenames"]]
    kappa_mat = z["kappa"].astype(np.float32)
    feats = z["features"] if "features" in z else None
    P(f"      缓存帧数 {len(files)}，kappa 矩阵 {kappa_mat.shape}")
    pc1, evr, orient = pca_pc1(kappa_mat, feats, args.max_frames)
    del kappa_mat
    P(f"      PC1 解释方差 {evr:.4f}，{orient}")
    lut = {f: float(v) for f, v in zip(files, pc1) if f in wanted}
    P(f"      匹配到帧级 BDI {len(lut)} / {len(wanted)}")
    if len(lut) < len(wanted):
        miss = wanted - set(lut)
        P(f"      [警告] 未匹配 {len(miss)} 帧，例：{sorted(miss)[:3]}")

    P()
    P("[2/3] 三名标注者的效标 AUC 与自助 CI")
    P("-" * 68)
    rng = np.random.default_rng(20261105)
    results = {}
    for rater in ("A", "B", "C"):
        ans = load_labels(args.dir, rater)
        rows = [(i, k) for i, k in key.items()
                if k["is_repeat"] == "0" and ans.get(i) in (1, 2, 3)
                and k["rel_path"] in lut and k["bbi"] not in ("", None)]
        if len(rows) < 20:
            P(f"  {rater}: [跳过] 可匹配 {len(rows)} 条")
            continue
        s = np.array([lut[k["rel_path"]] for _, k in rows])
        b = np.array([float(k["bbi"]) for _, k in rows])
        y = np.array([bin_of(CODE_TO_CLASS[ans[i]]) for i, _ in rows])
        auc, n1, n0 = roc_auc(s, y)
        auc_b, _, _ = roc_auc(b, y)
        boots = np.empty(args.n_boot)
        for t in range(args.n_boot):
            idx = rng.integers(0, len(s), len(s))
            boots[t] = roc_auc(s[idx], y[idx])[0]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        bc = best_cut(s, y)
        results[rater] = (auc, lo, hi, auc_b, len(s))
        P(f"  {rater}: n={len(s)}（弯 {n1} / 直 {n0}）")
        P(f"      BDI  AUC = {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        P(f"      bbi  AUC = {auc_b:.3f}（同帧集）  BDI−bbi = {auc - auc_b:+.3f}")
        P(f"      最佳切点 BDI = {bc['thr']:+.4f}  准确率 {bc['acc']:.3f}  "
          f"κ = {bc['kappa']:+.3f}")

    P()
    P("[3/3] 配对零分布模拟（r = %.4f）" % args.r_target)
    P("-" * 68)
    for rater, (auc, lo, hi, auc_b, n) in results.items():
        ans = load_labels(args.dir, rater)
        rows = [(i, k) for i, k in key.items()
                if k["is_repeat"] == "0" and ans.get(i) in (1, 2, 3)
                and k["rel_path"] in lut and k["bbi"] not in ("", None)]
        s = np.array([lut[k["rel_path"]] for _, k in rows])
        b = np.array([float(k["bbi"]) for _, k in rows])
        y = np.array([bin_of(CODE_TO_CLASS[ans[i]]) for i, _ in rows])
        obs = auc - auc_b
        diffs = np.empty(args.n_boot)
        for t in range(args.n_boot):
            idx = rng.integers(0, len(s), len(s))
            xb, yb = b[idx], y[idx]
            sd = xb.std()
            if sd < 1e-12:
                diffs[t] = 0.0
                continue
            xz = (xb - xb.mean()) / sd
            eps = rng.standard_normal(len(xb))
            xp = args.r_target * xz + math.sqrt(1 - args.r_target ** 2) * eps
            diffs[t] = roc_auc(xp, yb)[0] - roc_auc(xb, yb)[0]
        lo2, hi2 = np.percentile(diffs, [2.5, 97.5])
        p2 = 2 * min((diffs <= obs).mean(), (diffs >= obs).mean())
        P(f"  {rater}: 观测差 {obs:+.3f}；零分布 95% 区间 [{lo2:+.3f}, {hi2:+.3f}]，"
          f"双侧 p ≈ {p2:.3f}")

    P()
    P("=" * 68)
    P("回填建议（表 5 C2 行，第三人 C 补入）")
    P("=" * 68)
    if "A" in results and "B" in results:
        a, alo, ahi, ab, an = results["A"]
        b_, blo, bhi, bb, bn = results["B"]
        P(f"  {a:.3f}（A，n = {an}，95% CI [{alo:.3f}, {ahi:.3f}]）、"
          f"{b_:.3f}（B，n = {bn}，[{blo:.3f}, {bhi:.3f}]）")
        if "C" in results:
            c, clo, chi, cb, cn = results["C"]
            P(f"  、{c:.3f}（C，n = {cn}，[{clo:.3f}, {chi:.3f}]）")
        P(f"  ；同帧集弯曲指数 AUC 为 {ab:.3f}（A）与 {bb:.3f}（B）"
          f"{f'、{cb:.3f}（C）' if 'C' in results else ''}；"
          f"BDI−bbi 配对差与零分布判读见 [3/3] 段（显著为负则如实写"
          f"\"BDI 相对弯曲指数呈一致、小幅但显著的劣势\"，不得写\"无可测差异\"）")

    out = os.path.join(args.dir, "bdi_ci_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    P()
    P(f"报告已写入 {out}，把 [2/3] 与回填建议两段发回即可更新论文。")


if __name__ == "__main__":
    main()
