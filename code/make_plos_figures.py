# -*- coding: utf-8 -*-
"""make_plos_figures.py —— 论文图 1-3 的 PLoS Computational Biology 投稿版。

与 make_v19_figures.py（中文内部评审版）的关系：同一套图、两个版本；
本脚本按 PLOS 系列投稿规格重排：
  1. 矢量输出：PDF（PLOS 接受 PDF/EPS/TIFF，矢量首选 PDF）；
  2. 版式：双栏全宽 7.0 in（17.8 cm），高度按内容定；
  3. 字体：Arial（PLOS 要求 Arial/Helvetica），印刷字号 7-9 pt；
  4. 图例不放在图内——PLOS 要求 legend 随图题（图下方文本），图内改用
     直接标注（direct labeling）；
  5. 图 2D 印入六对 ARI 的四位小数值（两位小数全是 0.00，审稿人无法辨认），
     色标改发散映射（负 ARI 也能显示）；
  6. 图内文字英文化（PLOS 为英文期刊；中文版仍由 make_v19_figures.py 维护）。

数据来源：
  图 2 四面板 <- <indep>/ 的 indep_validity/modality/dbscan/agreement.csv
      （自动选 atlas_out_all/indep，删后重跑结果；缺文件时回退 atlas_out_full/indep）
  图 1 为布局图，无数字，不受数据重跑影响。
  图 3 六项刻画的**数字写在本脚本里会过期**，现统一从 axis_cards.json 读取；
      删数据后须先按新表 5 / axis_summary.txt 更新该 json，再重出图 3。

用法：
  python make_plos_figures.py [--outdir figures_plos] [--indep atlas_out_all/indep]
                              [--cards axis_cards.json]
输出：fig1_pipeline.pdf / fig2_pipelines.pdf / fig3_axis.pdf
      + 600 dpi PNG（Word 版内嵌用，矢量 PDF 供投稿上传）
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
CARDS_JSON = os.path.join(HERE, "axis_cards.json")
INDEP = None          # main() 里按 --indep / 自动探测赋值

# axis_cards.json 缺失时的兜底（删数据前的 v25 表 5 数字）
DEFAULT_CARDS_EN = [
    ("0.4942", "PC1 explained variance;\nr = 0.976 with BBI — nearly\na purified bending depth"),
    ("0.555 / 0.638", "AUC against human binary\njudgment — not passed: humans\nuse one axis, not bending depth"),
    ("0.964", "time-split Spearman–Brown;\nworm-level ICC 0.556 — usable\nfor individual phenotypes"),
    ("4 groups p < 0.002", "per-worm means decline across\nimaging dates; small slope\n(~0.001/day); demonstration only"),
    ("0.5448", "normalized mutual information\nwith three-class labels —\nmoderate sharing, not a replacement"),
    ("0.266", "density at k = 2 boundary is\n0.91× the peak; per-date SD\n0.011 — stable ≠ correct"),
]


def find_indep(arg):
    """优先删后全量目录；只有它缺 CSV 时才回退旧目录（并明确打印）。"""
    cands = [arg] if arg else []
    cands += [os.path.join(HERE, "atlas_out_all", "indep"),
              os.path.join(HERE, "atlas_out_full", "indep")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "indep_validity.csv")):
            return c
    raise SystemExit(
        "[错误] 找不到四管线 CSV（indep_validity.csv）。已尝试：\n  "
        + "\n  ".join(cands)
        + "\n请先完成 run_full_v17.sh 的 05_indep 步，或用 --indep 指定目录。")


def load_cards(path):
    """返回 (六项 (big, sub_en), stale, data_version)。"""
    if not os.path.isfile(path):
        print(f"[警告] 找不到 {path}，图 3 将使用内置删前旧数字！")
        return DEFAULT_CARDS_EN, True, "builtin-predrop"
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    cards = [(c["big"], c["sub_en"]) for c in d["cards"]]
    if len(cards) != 6:
        raise SystemExit(f"[错误] {path} 的 cards 应为 6 项，实为 {len(cards)}")
    return cards, bool(d.get("stale", False)), d.get("data_version", "?")

# ---------------------------------------------------------------- 字体与全局
# PLOS 要求 Arial / Helvetica；本机注册真 Arial，服务器回退 Liberation Sans
for f in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
    if os.path.exists(f):
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            pass
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,          # TrueType 嵌入，PLOS 要求文字可编辑
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "font.size": 7.5,            # 印刷基准字号
})

INK = "#1f2937"
GRAY_F, GRAY_E = "#F1EFE8", "#888780"
BLUE_F, BLUE_E = "#E6F1FB", "#185FA5"
ORAN_F, ORAN_E = "#FAEEDA", "#BA7517"
RED = "#A32D2D"
TEAL_F, TEAL_E = "#E1F5EE", "#0F6E56"
PINK_F, PINK_E = "#FCEBEB", "#A32D2D"
PIPES = ["KAPPA", "ANGLE", "OUTLINE", "MASKGRID"]
PCOL = {"KAPPA": "#185FA5", "ANGLE": "#BA7517",
        "OUTLINE": "#0F6E56", "MASKGRID": "#A32D2D"}

DOUBLE_W = 7.0   # PLOS 双栏全宽（英寸）


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def box(ax, x, y, w, h, fc, ec, title, sub=None, ts=8.0, ss=6.8, lw=0.9,
        dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                       fc=fc, ec=ec, lw=lw, linestyle="--" if dashed else "-",
                       mutation_aspect=0.6)
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=ts, color=INK, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=ss, color="#444441", linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=ts, color=INK, fontweight="bold", linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color="#5F5E5A", lw=1.0, dashed=False, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle="--" if dashed else "-",
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=2, shrinkB=2))


# ================================================================ 图 1
def fig1(outdir):
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 4.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

    # 列 1 输入
    box(ax, 1.2, 26, 15, 16, "#ffffff", GRAY_E, "Input",
        "Single-worm image\nsequences\n80 × 90 px\ngroup/date/row/\nworm/frame",
        ts=8.0, ss=6.2)
    # 列 2 上游（已发表系统）
    box(ax, 17.8, 38, 19.2, 9, GRAY_F, GRAY_E, "YOLO + SSIM/IoU",
        "live/dead filtering\n(human-boxed\ngold standard)", ts=7.0, ss=6.2)
    box(ax, 17.8, 26.5, 19.2, 9, GRAY_F, GRAY_E, "Skeleton extraction",
        "contrast stretch + multi-\nthreshold; dynamic range\n~35 grey levels",
        ts=7.0, ss=6.2)
    box(ax, 17.8, 15, 19.2, 9, GRAY_F, GRAY_E, "Centerline + head–tail",
        "centroid chain; orientation\nkept as output marker only", ts=7.0, ss=6.2)
    arrow(ax, 16.2, 34, 17.8, 34)
    arrow(ax, 27.4, 38, 27.4, 35.5); arrow(ax, 27.4, 26.5, 27.4, 24)
    ax.text(27.4, 12.6, "Upstream (published system)", ha="center",
            fontsize=6.6, color="#5F5E5A")

    # 列 3 形状表示
    box(ax, 40, 26, 15, 16, ORAN_F, ORAN_E, "κ(s)",
        "100-point normalized\ncurvature, |sin Δθ|\nmirror/scale invariant,\nclass-free",
        ts=8.5, ss=6.2)
    arrow(ax, 37, 34, 40, 34)
    ax.text(43.5, 23.2, "Shape representation", ha="center", fontsize=6.6,
            color="#5F5E5A")

    # 列 4 分析层（本文）
    box(ax, 58, 39.5, 22, 8.5, BLUE_F, BLUE_E, "Ruler: SBPD (3-dim PCA)",
        "3.1  can rules be read out?", ts=7.0, ss=6.2)
    box(ax, 58, 27.5, 22, 8.5, BLUE_F, BLUE_E, "Structure: 4 pipelines",
        "3.3  do frame-level class\nboundaries exist?", ts=7.0, ss=6.2)
    box(ax, 58, 15.5, 22, 8.5, BLUE_F, BLUE_E, "Continuous axis: BDI",
        "3.4  reliability / criterion /\ncut point", ts=7.0, ss=6.2)
    arrow(ax, 55, 34, 58, 43.7, rad=-0.15)
    arrow(ax, 55, 34, 58, 31.7, rad=0.0)
    ax.text(69, 50.8, "Analysis layer (this paper)", ha="center",
            fontsize=7, color=BLUE_E, fontweight="bold")

    # 红色数据驱动路径（注释放置在 κ(s) 与 BDI 之间的空白弧线区，避开其他文字）
    arrow(ax, 47.5, 26, 58, 19.7, color=RED, lw=1.4, dashed=True, rad=0.25)
    ax.text(51, 6.2, "Data-driven path: κ(s) enters the\ncontinuous axis and segment\n"
                     "fingerprints without any class label",
            ha="center", fontsize=6.6, color=RED, linespacing=1.5)

    # 列 5 输出
    box(ax, 82.5, 26, 16.5, 16, TEAL_F, TEAL_E, "Per-worm\naggregation",
        "old readout: class\nproportions\nnew readout: segment\nfingerprints",
        ts=8.0, ss=6.2)
    arrow(ax, 80, 43.7, 82.5, 38); arrow(ax, 80, 31.7, 82.5, 34)
    arrow(ax, 80, 19.7, 82.5, 30)

    # 图例横排（保留在图内底部，符合直接标注原则）
    lg = [(GRAY_F, GRAY_E, "Upstream (published)"),
          (BLUE_F, BLUE_E, "This paper"),
          (None, RED, "Data-driven path")]
    x0 = 8
    for fc, ec, name in lg:
        if fc:
            ax.add_patch(FancyBboxPatch((x0, 2.6), 3.0, 2.0,
                                        boxstyle="round,pad=0.3,rounding_size=0.8",
                                        fc=fc, ec=ec, lw=0.8, mutation_aspect=0.6))
        else:
            ax.plot([x0, x0 + 3.0], [3.6, 3.6], color=RED, lw=1.4,
                    linestyle="--")
        ax.text(x0 + 4.0, 3.6, name, fontsize=6.8, color=INK, va="center")
        x0 += 4.0 + len(name) * 1.28 + 4

    for ext in ("pdf",):
        fig.savefig(os.path.join(outdir, f"fig1_pipeline.{ext}"),
                    bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(outdir, "fig1_pipeline.png"), dpi=600,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig1 ok")


# ================================================================ 图 2
def fig2(outdir):
    val = read_csv(os.path.join(INDEP, "indep_validity.csv"))
    mod = read_csv(os.path.join(INDEP, "indep_modality.csv"))
    db = read_csv(os.path.join(INDEP, "indep_dbscan.csv"))
    agr = read_csv(os.path.join(INDEP, "indep_agreement.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_W, 5.4))

    # A. 轮廓系数谱（直接标注，图例入图题）
    ax = axes[0, 0]
    for name in PIPES:
        rows = sorted([r for r in val if r["pipe"] == name
                       and r["algo"] == "kmeans"], key=lambda r: int(r["k"]))
        if rows:
            ax.plot([int(r["k"]) for r in rows], [float(r["sil"]) for r in rows],
                    "o-", ms=2.5, lw=0.9, color=PCOL[name])
            ax.annotate(name, (int(rows[-1]["k"]), float(rows[-1]["sil"])),
                        xytext=(4, -2), textcoords="offset points",
                        fontsize=6.6, color=PCOL[name], va="center")
    ax.axhline(0.35, ls=":", color="#888", lw=0.7)
    ax.text(8.0, 0.375, "0.35", fontsize=6, color="#888", ha="right")
    ax.set_xlabel("Number of clusters k", fontsize=7.5)
    ax.set_ylabel("Silhouette width", fontsize=7.5)
    ax.set_title("A. Internal validity across pipelines", fontsize=8,
                 fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    # B. 主轴谷深（全部管线标注数值与峰数；0.25 阈值线固定进入画幅）
    ax = axes[0, 1]
    if mod:
        xs = range(len(mod))
        vd = [float(r["valley_depth"]) for r in mod]
        ax.bar(xs, vd, color=[PCOL[r["pipe"]] for r in mod],
               width=0.55, alpha=0.9)
        for i, r in enumerate(mod):
            v = float(r["valley_depth"])
            n = int(r["n_peaks"])
            label = f"{v:.3f}\n({n} peak" + ("s)" if n > 1 else ")")
            ax.text(i, v + 0.008, label, ha="center", va="bottom",
                    fontsize=6.2, linespacing=1.3)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([r["pipe"] for r in mod], rotation=15,
                           fontsize=6.6, ha="right")
        ax.set_ylim(0, 0.30)
        ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax.axhline(0.25, ls=":", color="#888", lw=0.7)
    ax.text(len(mod) - 0.5, 0.256, "0.25", fontsize=6, color="#888", ha="right")
    ax.set_ylabel("KDE valley depth", fontsize=7.5)
    ax.set_title("B. Valley depth along principal axis", fontsize=8,
                 fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    # C. DBSCAN eps 扫描（全部管线均为单一密度集群；标注分置首尾避开重叠）
    ax = axes[1, 0]
    for name in PIPES:
        rows = [r for r in db if r["pipe"] == name]
        if rows:
            ax.plot([float(r["eps"]) for r in rows],
                    [int(r["n_clusters"]) for r in rows],
                    "o-", ms=2.5, lw=0.9, color=PCOL[name])
            if name == "MASKGRID":   # 唯一位于右侧的线，末端标注
                ax.annotate(name, (float(rows[-1]["eps"]),
                                   int(rows[-1]["n_clusters"])),
                            xytext=(4, -2), textcoords="offset points",
                            fontsize=6.6, color=PCOL[name], va="center")
            else:                    # 重叠在一起的线，起点上方错开标注
                dy = {"OUTLINE": 5, "ANGLE": -7, "KAPPA": -14}[name]
                ax.annotate(name, (float(rows[0]["eps"]),
                                   int(rows[0]["n_clusters"])),
                            xytext=(6, dy), textcoords="offset points",
                            fontsize=6.6, color=PCOL[name], va="center")
    ax.set_xlabel("eps (k-distance percentile)", fontsize=7.5)
    ax.set_ylabel("Density clusters", fontsize=7.5)
    ax.set_title("C. Density separability (DBSCAN)", fontsize=8,
                 fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    # D. 跨管线 ARI：四位小数 + 发散色标（负值可见）
    ax = axes[1, 1]
    n = len(PIPES)
    M = [[1.0 if i == j else None for j in range(n)] for i in range(n)]
    for r in agr:
        a, b = r["a"], r["b"]
        if a in PIPES and b in PIPES:
            i, j = PIPES.index(a), PIPES.index(b)
            v = float(r["ari"])
            M[i][j] = M[j][i] = v
    M = [[0.0 if v is None else v for v in row] for row in M]
    im = ax.imshow(M, vmin=-0.05, vmax=0.05, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(PIPES, rotation=20, ha="right", fontsize=6.6)
    ax.set_yticklabels(PIPES, fontsize=6.6)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i][j]:+.4f}" if i != j else "1.0000",
                    ha="center", va="center", fontsize=6.2,
                    color="white" if abs(M[i][j]) > 0.03 else "#1f2937")
    ax.set_title("D. Cross-pipeline ARI at k = 2", fontsize=8,
                 fontweight="bold", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6.2, width=0.5)
    cb.outline.set_linewidth(0.5)
    ax.tick_params(length=0)

    fig.tight_layout(h_pad=1.6, w_pad=2.0)
    fig.savefig(os.path.join(outdir, "fig2_pipelines.pdf"),
                bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(outdir, "fig2_pipelines.png"), dpi=600,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig2 ok")


# ================================================================ 图 3
def fig3(outdir):
    nums, stale, version = load_cards(CARDS_JSON)
    if stale:
        print("=" * 70)
        print(f"[警告] 图 3 六项数字仍是删数据前的旧值（data_version={version}），")
        print("       与重跑后的表 5 / axis_summary.txt 不一致！")
        print("       请先更新 axis_cards.json，再重跑本脚本。")
        print("=" * 70)

    fig, ax = plt.subplots(figsize=(DOUBLE_W, 3.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

    titles = ["C1  Definition strength", "C2  Criterion validity",
              "C3  Reliability", "C4  Sensitivity",
              "C5  Information content", "C6  Cut-point arbitrariness"]
    palette = [(BLUE_F, BLUE_E, INK), (PINK_F, PINK_E, RED),
               (BLUE_F, BLUE_E, INK), (BLUE_F, BLUE_E, INK),
               (BLUE_F, BLUE_E, INK), (ORAN_F, ORAN_E, "#854F0B")]
    cards = [(titles[i], big, sub) + palette[i]
             for i, (big, sub) in enumerate(nums)]
    W, H = 30.5, 20.5
    xs = [3.5, 35.2, 66.9]
    ys = [27.5, 3.5]
    for i, (t, big, sub, fc, ec, tc) in enumerate(cards):
        x, y = xs[i % 3], ys[i // 3]
        ax.add_patch(FancyBboxPatch((x, y), W, H,
                                    boxstyle="round,pad=0.8,rounding_size=1.6",
                                    fc=fc, ec=ec, lw=1.0, mutation_aspect=0.5))
        ax.text(x + W / 2, y + H - 3.2, t, ha="center", fontsize=7.6,
                color=tc, fontweight="bold")
        ax.text(x + W / 2, y + H - 7.6, big, ha="center", fontsize=13,
                color=tc, fontweight="bold")
        ax.text(x + W / 2, y + 4.8, sub, ha="center", fontsize=6.6,
                color="#444441", linespacing=1.55)
    fig.savefig(os.path.join(outdir, "fig3_axis.pdf"),
                bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(outdir, "fig3_axis.png"), dpi=600,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig3 ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures_plos"))
    ap.add_argument("--indep", default=None,
                    help="四管线 CSV 目录（默认自动选 atlas_out_all/indep）")
    ap.add_argument("--cards", default=CARDS_JSON,
                    help="图 3 六项数字 JSON（默认同目录 axis_cards.json）")
    args = ap.parse_args()

    INDEP = find_indep(args.indep)
    CARDS_JSON = args.cards
    print(f"[数据] 图 2 源目录：{INDEP}")
    tag = os.path.basename(os.path.dirname(os.path.abspath(INDEP)))
    if tag != "atlas_out_all":
        print(f"[警告] 图 2 数据源是 {tag}/（非 atlas_out_all），"
              "可能是删数据前的旧结果，请确认第 1 步全量重跑已完成。")

    os.makedirs(args.outdir, exist_ok=True)
    fig1(args.outdir)
    fig2(args.outdir)
    fig3(args.outdir)
    print("ALL DONE ->", args.outdir)
