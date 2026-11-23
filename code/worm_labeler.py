"""
线虫弯曲行为预标注工具（五维数据版）
=============================
功能：递归读取 浓度/日期/行号/虫号/帧号 目录结构中的单只线虫截图，
      自动提取骨架 → 计算曲率特征 → 规则判断 4 种动作(0-3)
输出：CSV 文件（含 5 维分组列 + 标签 + 置信度 + 特征值）+ 可选可视化图片

数据结构：
  D:/data/image/
  ├── control/
  │   ├── 0116/
  │   │   ├── B/
  │   │   │   ├── worm1_001.png
  │   │   │   ├── worm1_002.png
  │   │   │   └── ...
  │   │   └── ...
  ├── 5mg/
  └── ...

动作标签（2026-10-23 起 3 类；判据全部不依赖头尾方向——头尾单帧判不可靠，
误差会沿管线传播污染下游，故所有类判据只用方向无关量）：
  0 = 正弦型 (Sinusoidal)  —— 多波浪/中等弯曲（兜底类）
  1 = Ω 型 (Omega)          —— 单个深弯：n_high_regions<=1 且 max_kappa>=0.35
                               （峰位置不限，中段或头颈/尾部）
  2 = 笔直型 (Straight)     —— 全身曲率极低，接近直线（BBI<0.16）
                               （原"静止"：静止是运动学概念需跨帧判断，
                               本类纯形状定义，更名笔直以避免歧义。
                               倒退类已取消——单帧无法判别，跨帧运动
                               信号又依赖头尾方向，实测污染严重。
                               缠绕类已取消——自交检测/几何紧密度/时序
                               运动三重实证均无区分度，2026-10-23）

可视化：必须指定数量参数才会生成，--viz-dir 只决定输出目录，两者要配合使用
  python worm_labeler.py -i $WORM_DATA_ROOT -o labels.csv --max-images 2000 --viz 12 --viz-dir viz/                    # 随机 12 张
  python worm_labeler.py -i $WORM_DATA_ROOT
  python worm_labeler.py -i $WORM_DATA_ROOT -o labels.csv --max-images 5000 --viz-per-class 50 --viz-failed 5 --viz-dir viz
  生成的每张图含 5 个面板（用于骨架提取调参）：
      原图 | 顶帽/黑帽阈值分割 | 最大连通域掩膜 | 骨架叠加(红线/绿头/蓝尾) | 曲率曲线 κ(s)
  其中顶帽用于暗场(暗底亮虫)，黑帽用于明场(浅底深虫)，自动判定

进度：扫描与处理阶段都会原地刷新百分比进度条，每 10% 换行留档
  默认全程约刷新 10 次，可用 --progress-interval N 自定义刷新粒度
"""

import cv2
import numpy as np
import math
import os
import sys
import csv
import json
import argparse
import re
import time
import random
import multiprocessing as mp
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.interpolate import splrep, splev
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 尝试使用中文字体，避免可视化图片中中文显示为方块
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun',
                                    'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


# ============================================================
# 0. 五维数据结构解析
# ============================================================
WORM_NAME_RE = re.compile(r'(worm\d+)_(\d+)\.([^.]+)$', re.IGNORECASE)


def parse_five_dims(rel_path):
    """把相对路径解析成 (group, date, row, worm, frame)。

    示例：
        'control/0116/B/worm1_001.png' -> ('control','0116','B','worm1',1)
        '10mg/0128/E/worm12_050.png'   -> ('10mg','0128','E','worm12',50)
    """
    rel_path = rel_path.replace('\\', '/')
    parts = rel_path.split('/')

    # 如果是扁平命名（如 control_0116_B_worm1_001.png）也兼容
    if len(parts) == 1:
        parts = rel_path.replace('_', '/').split('/')
        if len(parts) < 5:
            # 无法解析，整张图作为一个独立样本
            name = os.path.splitext(rel_path)[0]
            return 'unknown', 'unknown', 'unknown', name, 0

    group = parts[0]
    date = parts[1]
    row = parts[2]
    basename = parts[-1]
    name, _ = os.path.splitext(basename)

    m = WORM_NAME_RE.match(basename)
    if m:
        worm = m.group(1).lower()
        frame = int(m.group(2))
    else:
        # 文件名不符合 wormN_MMM 格式，整张图作为一个独立样本
        worm = name
        frame = 0

    return group, date, row, worm, frame


# ============================================================
# 0b. 虫体清单过滤（跳过 / 只跑指定虫体）
# ============================================================
# 典型用法：人工在点选器里挑出“不像虫子”的虫体，导出 worms_to_delete.txt，
# 然后用 --exclude-list 直接跳过它们，无需真的删除磁盘上的图片文件。
#
#   python worm_labeler.py -i D:/data/image -o labels.csv -w 16 \
#       --exclude-list "worms_to_delete.txt"
#
# 清单每行一条，支持 4 种粒度（大小写不敏感，\\ 与 / 都认）：
#   control                 -> 整个组
#   control/0211            -> 整个日期
#   control/0211/D          -> 整行
#   control/0211/D/worm1    -> 单只虫（点选器导出就是这个格式）
# 行首可以带盘符或 data 根目录（自动取最后 4 段），行首 # 为注释。
# 也支持点选器导出的 .json 文件。

def load_worm_list(path):
    """读取虫体清单文件（TXT / JSON）。

    返回 (dir_rules, worm_rules, n_worms)
        dir_rules  : set[tuple]，长度 1~3，表示整目录命中
        worm_rules : dict[(group, date, row)] -> set[worm]
        n_worms    : 单只虫级别的条目数
    """
    items = []
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ('worms', 'selected', 'items', 'list'):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                data = []
        for it in data:
            if isinstance(it, str):
                items.append(it)
            elif isinstance(it, dict):
                segs = [str(it.get(k, '') or '') for k in ('group', 'date', 'row', 'worm')]
                items.append('/'.join(s for s in segs if s))
    else:
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip().strip('"').strip("'")
                if line and not line.startswith('#'):
                    items.append(line)

    dir_rules = set()
    worm_rules = defaultdict(set)
    n_worms = 0

    for raw in items:
        segs = [s.strip().lower() for s in raw.replace('\\', '/').split('/') if s.strip()]
        if not segs:
            continue
        if len(segs) > 4:            # 带盘符/根目录时只取最后 4 段
            segs = segs[-4:]
        if len(segs) == 4:
            worm_rules[tuple(segs[:3])].add(segs[3])
            n_worms += 1
        else:
            dir_rules.add(tuple(segs))

    return dir_rules, dict(worm_rules), n_worms


# ============================================================
# 0c. 阈值覆盖清单（指定帧/虫强制用某个阈值）
# ============================================================
# 典型用法：个别帧"干净候选太短/脏候选太糙"的二选一困境（统计双胞胎，
# 全局规则无解），由人工点名指定阈值。例如 worm11_022 这类细虫+原图
# 软边帧，全局规则保了淡尾长度（thr20/81点）但掩膜坡脚切割有毛边，
# 人工看原图确认应该切坡中段（thr61/62点、掩膜干脆）。
#
#   python worm_labeler.py -i $WORM_DATA_ROOT -o labels.csv \
#       --threshold-override thr_override.txt
#
# 清单每行一条 `<路径> <阈值>`（逗号/空格/制表符分隔均可），行首 # 注释，
# 支持 3 种粒度（大小写不敏感，\ 与 / 都认，.png 可省）：
#   10mg/0116/C/worm11_022 61   -> 单帧（wormN_MMM）
#   10mg/0116/C/worm11      61  -> 整只虫所有帧（wormN）
#   10mg/0116/C             61  -> 整行所有虫所有帧
# 也支持 JSON：{"10mg/0116/C/worm11_022": 61} 或
#   [{"path": "10mg/0116/C/worm11", "thr": 61}]
#
# 语义：只改"选哪个阈值候选"，**不绕过两虫交叉过滤**——指定阈值若产不出
# 合格单虫骨架（含触发交叉判据），该帧回退常规选择并在调试日志里注明。
# 阈值会被补进该图的阈值梯队，所以梯队外的值（如 61 不在 {12,20,25,35,
# 0.6/0.8/1.0/1.3×otsu} 里时）也能生效。

def load_threshold_override(path):
    """读取阈值覆盖清单（TXT / JSON）。

    返回 (frame_rules, worm_rules, dir_rules, n_rules)
        frame_rules : dict[relpath_no_ext] -> thr   帧级
        worm_rules  : dict[(group,date,row,worm)] -> thr
        dir_rules   : dict[(group,...)] -> thr      1~3 段目录级
    """
    items = []                       # [(path_str, thr_str), ...]
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ('items', 'list', 'overrides', 'rules'):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                data = [{'path': k, 'thr': v} for k, v in data.items()
                        if not isinstance(v, (list, dict))]
        for it in data:
            if isinstance(it, str):
                items.append(it)
            elif isinstance(it, dict):
                p = it.get('path') or it.get('file') or it.get('frame') or ''
                t = it.get('thr') or it.get('threshold') or it.get('value')
                if p and t is not None:
                    items.append('%s %s' % (p, t))
    else:
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip().strip('"').strip("'")
                if line and not line.startswith('#'):
                    items.append(line)

    frame_rules, worm_rules, dir_rules = {}, {}, {}
    n_rules = 0
    for raw in items:
        toks = raw.replace(',', ' ').replace('\t', ' ').split()
        if len(toks) < 2:
            raise ValueError('行格式应为 "<路径> <阈值>": %r' % raw)
        path_str, thr_str = toks[0], toks[-1]
        try:
            thr = int(float(thr_str))
        except ValueError:
            raise ValueError('阈值不是数字: %r' % raw)
        segs = [s.strip().lower() for s in
                path_str.replace('\\', '/').split('/') if s.strip()]
        if not segs:
            continue
        if len(segs) > 5:                       # 带盘符/根目录时只取最后 5 段
            segs = segs[-5:]
        if segs[-1].endswith('.png'):
            segs[-1] = segs[-1][:-4]
        if len(segs) >= 5:                      # group/date/row/worm/frame
            g, d, r, w, fr = segs[-5:]
            frame_rules['%s/%s/%s/%s_%s' % (g, d, r, w, fr)] = thr
        elif len(segs) == 4:
            g, d, r, last = segs
            if re.match(r'^worm\d+_\d+$', last):        # 帧级（4 段 relpath）
                frame_rules['%s/%s/%s/%s' % (g, d, r, last)] = thr
            elif re.match(r'^worm\d+$', last):          # 整虫
                worm_rules[(g, d, r, last)] = thr
            else:                                        # 整行
                dir_rules[(g, d, r)] = thr
        else:
            dir_rules[tuple(segs)] = thr
        n_rules += 1
    return frame_rules, worm_rules, dir_rules, n_rules


def build_thr_resolver(frame_rules, worm_rules, dir_rules):
    """生成 rel_path -> thr|None 的解析器（帧级 > 整虫 > 目录级）。"""
    frame_rules = frame_rules or {}
    worm_rules = worm_rules or {}
    dir_rules = dir_rules or {}

    def resolve(rel_path):
        p = rel_path.replace('\\', '/').lower()
        if p.endswith('.png'):
            p = p[:-4]
        if p in frame_rules:
            return frame_rules[p]
        try:
            group, date, row, worm, _frame = parse_five_dims(rel_path)
        except Exception:
            return None
        group, date, row = group.lower(), date.lower(), row.lower()
        wk = (group, date, row, worm)
        if wk in worm_rules:
            return worm_rules[wk]
        for k in ((group, date, row), (group, date), (group,)):
            if k in dir_rules:
                return dir_rules[k]
        return None

    return resolve


def build_worm_filter(dir_rules, worm_rules, mode='exclude', base_parts=()):
    """生成目录级过滤器。

    mode='exclude'：清单里的虫体跳过，其余照常处理
    mode='only'   ：只处理清单里的虫体，其余跳过
    base_parts    ：输入根目录本身已包含的层级，例如 -i 直接指向
                    D:/data/image/control/0211/D 时传入 ('control','0211','d')，
                    这样 leaf 只有 '.' 也能正确匹配规则

    返回 filter_files(leaf, files) -> (kept_files, dropped_count)
        leaf  为相对输入根目录的当前目录，如 'control/0211/D'
        files 为该目录下的文件名列表
    """
    base = [p.lower() for p in base_parts if p]
    dir_rules = set(dir_rules or ())
    worm_rules = {k: set(v) for k, v in (worm_rules or {}).items()}
    active = bool(dir_rules or worm_rules)
    exclude = (mode == 'exclude')

    def dir_hit(parts):
        for i in (1, 2, 3):
            if len(parts) >= i and tuple(parts[:i]) in dir_rules:
                return True
        return False

    def worm_hit(parts, worm):
        ws = worm_rules.get(tuple(parts[:3])) if len(parts) >= 3 else None
        return bool(ws) and worm in ws

    def filter_files(leaf, files):
        if not active:
            return files, 0

        parts = [p.lower() for p in leaf.replace('\\', '/').split('/') if p and p != '.']
        parts = base + parts

        # 快路径：标准三层目录 control/0211/D，整目录只判定一次
        if len(parts) >= 3:
            if dir_hit(parts):
                return ([], len(files)) if exclude else (files, 0)
            ws = worm_rules.get(tuple(parts[:3]), set())
            if not ws:
                return ([], len(files)) if not exclude else (files, 0)
            keep, drop = [], 0
            for f in files:
                m = WORM_NAME_RE.match(f)
                w = m.group(1).lower() if m else None
                hit = bool(w) and (w in ws)
                if hit != exclude:          # exclude: 未命中保留 / only: 命中保留
                    keep.append(f)
                else:
                    drop += 1
            return keep, drop

        # 慢路径：目录层级不足（例如 -i 直接指向 row 目录），逐文件解析
        keep, drop = [], 0
        for f in files:
            segs = [p.lower() for p in (leaf + '/' + f).replace('\\', '/').split('/')
                    if p and p != '.']
            segs = base + segs
            dparts = segs[:-1]
            m = WORM_NAME_RE.match(f)
            w = m.group(1).lower() if m else None
            hit = dir_hit(dparts) or (bool(w) and worm_hit(dparts, w))
            if hit != exclude:
                keep.append(f)
            else:
                drop += 1
        return keep, drop

    return filter_files


# ============================================================
# 1. 骨架提取
# ============================================================
def extract_skeleton(image):
    """从图片中提取线虫骨架。
    返回：骨架点坐标数组 (N, 2)，按从头到尾排序。

    step9_fingerprint.py 依赖此函数，签名保持不变。
    """
    points, _, _, _ = extract_skeleton_full(image)
    return points


def estimate_field_type(gray):
    """自动判断显微镜场型。

    明场（brightfield）: 浅底深色虫 → 边缘背景比中心（虫体）亮。
    暗场（darkfield）  : 暗底亮色虫 → 边缘背景比中心暗。

    返回 'brightfield' 或 'darkfield'。
    """
    h, w = gray.shape
    # 边缘背景均值
    border = np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ])
    bg_mean = np.mean(border)

    # 中心区域均值（大概率覆盖虫体）
    cy, cx = h // 2, w // 2
    ry, rx = max(1, h // 5), max(1, w // 5)
    center = gray[max(0, cy - ry):cy + ry, max(0, cx - rx):cx + rx]
    fg_mean = np.mean(center)

    return 'brightfield' if bg_mean > fg_mean else 'darkfield'


def _thin_mask(binary):
    """细化：ximgproc -> skimage -> 内置 Zhang-Suen(simple_thin)。
    返回 (thin_uint8, method_name)。抽成函数供多阈值梯队复用。"""
    try:
        thin = cv2.ximgproc.thinning(binary,
                                     thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return thin, 'ximgproc'
    except AttributeError:
        pass
    try:
        from skimage.morphology import skeletonize
        return (skeletonize(binary > 0) * 255).astype(np.uint8), 'skimage'
    except ImportError:
        return simple_thin(binary), 'zhangsuen(numpy)'


def _is_border_band(bx, by, bw, bh, h, w):
    """判断连通域是否为"背景带状伪影"而非虫体。

    特征：沿某一条图像边横跨近整图（>=85%）且垂直方向很扁（<=40%），
    且贴着所跨的那条边。典型如培养基底材渐变、刮痕、边缘光带——
    它们会在黑帽增强图里拉出与虫体相当的连通域，既抢占"最大连通域"，
    又会在多虫判定里冒充"第二只虫"（实测 worm15_181 顶部横带害得
    8 个阈值全被判多虫、worm12_083 右侧竖带被当成虫骨架化）。
    横贯整图的斜向/水平虫体不会被误判：要么垂直方向不扁，要么居中不贴边。
    """
    if bw >= 0.85 * w and bh <= 0.40 * h and (by <= 2 or by + bh >= h - 3):
        return True
    if bh >= 0.85 * h and bw <= 0.40 * w and (bx <= 2 or bx + bw >= w - 3):
        return True
    return False


def _skel_topology(thin):
    """骨架拓扑统计：返回 (像素数, 端点数, 结点数)。

    端点数 = 8 邻域中仅有 1 个前景邻居的像素；结点数 = >=3 个邻居的像素。
    一条单虫骨架（剪除毛刺后）应是简单曲线：2 个端点、0 结点；
    虫+背景带粘连会给出高 面积/骨架长 比，环化给出 0 端点，糊块给出多结点。
    这些量是"分割对了但内容不对"的廉价探针（实测 worm12_083 粘连糊块
    骨架仅 27px 而掩膜 1200px，面积比 45；单虫仅 5~14）。
    """
    m = (thin > 0).astype(np.uint8)
    mp = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    mp[1:-1, 1:-1] = m
    ker = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    nb = cv2.filter2D(mp, -1, ker)[1:-1, 1:-1]
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return 0, 0, 0
    vals = nb[ys, xs]
    return int(len(xs)), int((vals == 1).sum()), int((vals >= 3).sum())


def _merge_near_components(mask, max_gap):
    """把 mask 中相距很近的连通域并成组，修复"虫体被阈值打断成碎片"。

    黑帽响应在虫体中段（较宽/较淡处）常低于阈值，二值图把一条虫打成
    头/中/尾几段；旧版只取最大连通域，骨架凭空少一截（用户报"识别短了"：
    worm7_187 掩膜 370->314px、worm30_104 360->270px）。

    规则：
      · 只合并 面积 >= 10%最大域面积 且非背景带状伪影 的连通域
        （小噪点不参与，避免把远处污渍粘进来）；
      · 两两 bbox 的欧氏间隙 <= max_gap 即并成一组（并查集）；
      · 背景带状伪影自成一两组，不参与合并也不当选虫体。

    返回按组面积降序的组列表，每组:
        {'area': 总像素, 'is_band': 是否含带状伪影,
         'bbox': (x1,y1,x2,y2), 'mask': 该组像素的 uint8 掩膜}
    """
    nc, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if nc <= 1:
        return []
    h, w = mask.shape
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    max_a = float(areas.max()) if len(areas) else 0.0
    is_band = np.zeros(nc, bool)
    for i in range(1, nc):
        is_band[i] = _is_border_band(
            int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
            h, w)
    # 并查集
    parent = list(range(nc))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    big = [i for i in range(1, nc)
           if (not is_band[i]) and areas[i - 1] >= 0.10 * max_a]
    for x in range(len(big)):
        for y in range(x + 1, len(big)):
            a, b = big[x], big[y]
            ax1 = int(stats[a, cv2.CC_STAT_LEFT])
            ay1 = int(stats[a, cv2.CC_STAT_TOP])
            ax2 = ax1 + int(stats[a, cv2.CC_STAT_WIDTH]) - 1
            ay2 = ay1 + int(stats[a, cv2.CC_STAT_HEIGHT]) - 1
            bx1 = int(stats[b, cv2.CC_STAT_LEFT])
            by1 = int(stats[b, cv2.CC_STAT_TOP])
            bx2 = bx1 + int(stats[b, cv2.CC_STAT_WIDTH]) - 1
            by2 = by1 + int(stats[b, cv2.CC_STAT_HEIGHT]) - 1
            dx = max(bx1 - ax2, ax1 - bx2, 0)
            dy = max(by1 - ay2, ay1 - by2, 0)
            if math.hypot(dx, dy) <= max_gap:
                parent[find(a)] = find(b)
    # 汇总分组
    members = {}
    for i in range(1, nc):
        members.setdefault(find(i), []).append(i)
    groups = []
    for ids in members.values():
        m = np.zeros(mask.shape, np.uint8)
        total = 0.0
        band = False
        x1, y1, x2, y2 = w, h, -1, -1
        for i in ids:
            m[labels == i] = 255
            total += areas[i - 1]
            if is_band[i]:
                band = True
            cx1 = int(stats[i, cv2.CC_STAT_LEFT])
            cy1 = int(stats[i, cv2.CC_STAT_TOP])
            cx2 = cx1 + int(stats[i, cv2.CC_STAT_WIDTH]) - 1
            cy2 = cy1 + int(stats[i, cv2.CC_STAT_HEIGHT]) - 1
            x1, y1 = min(x1, cx1), min(y1, cy1)
            x2, y2 = max(x2, cx2), max(y2, cy2)
        groups.append({'area': total, 'is_band': band,
                       'bbox': (x1, y1, x2, y2), 'mask': m})
    groups.sort(key=lambda g: -g['area'])
    return groups


def _fill_small_holes(mask, max_hole_area):
    """填充二值掩膜内部的小孔（面积 <= max_hole_area）。

    虫体内部的肠线/明暗纹理会在黑帽二值图上留下小洞，细化后骨架绕孔成环、
    结点数爆炸——旧版"结点<=4"闸门因此整批误杀真虫（5000 张 94.8% 失败的
    主因之一）。贴边的背景是一个大连通域、面积远超上限，不会被误填；
    盘绕虫体围住的大孔同样不填，避免 C 形虫被填实把骨架走直。
    """
    inv = (mask == 0).astype(np.uint8)
    nc, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if nc <= 1:
        return mask
    out = mask.copy()
    for i in range(1, nc):
        if stats[i, cv2.CC_STAT_AREA] <= max_hole_area:
            out[labels == i] = 255
    return out


def _skel_graph(thin):
    """骨架像素图 -> 8 邻接图。

    返回 (adj, ys, xs, deg, n)：adj[i] 是第 i 个骨架像素的邻域下标列表；
    deg[i] 为度数（1=端点，2=中间，>=3=结点）。空骨架返回 (None,...,0)。
    """
    m = (thin > 0)
    ys, xs = np.where(m)
    n = len(xs)
    if n == 0:
        return None, None, None, None, 0
    H, W = m.shape
    idx = -np.ones((H, W), np.int32)
    idx[ys, xs] = np.arange(n)
    adj = [[] for _ in range(n)]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = ys + dy, xs + dx
            ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            jj = idx[np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)]
            for i in np.where(ok & (jj >= 0))[0]:
                adj[i].append(int(jj[i]))
    deg = np.array([len(a) for a in adj], np.int32)
    return adj, ys, xs, deg, n


def _skeleton_centerline(thin):
    """从细化骨架提取中心线：骨架像素图上做双端 BFS 取最长路径。

    细化+剪枝后的骨架可能仍带小环/毛刺/阶梯，旧版用"贪心最近邻 + 跳步>6px
    截断"排序，截断会把虫尾砍掉（"识别短了"的直接原因），毛刺末端还会被
    误当成头尾。这里在建图后：
      · 端点(度=1)齐全时，从所有端点分别 BFS 找最远者，再从其 BFS 一次
        取直径路径——对树形骨架即严格最长路径；
      · 纯环(端点=0，如咬尾)时从任意像素双重 BFS 取近似直径；
    返回 (points(N,2) 有序坐标, coverage=路径像素/骨架总像素, 骨架总像素)。
    points 为 None 表示输入退化（空骨架/无法成链）。
    """
    adj, ys, xs, deg, n = _skel_graph(thin)
    if adj is None:
        return None, 0.0, 0

    def bfs(src):
        dist = -np.ones(n, np.int32)
        par = -np.ones(n, np.int32)
        dist[src] = 0
        q = [src]
        for u in q:
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    q.append(v)
        return dist, par

    endpoints = np.where(deg == 1)[0]
    starts = [int(e) for e in endpoints]
    if not starts:                       # 纯环：从度数最大的像素出发
        starts = [int(np.argmax(deg))]
    # 第一轮：所有起点中最远者
    best_s, best_d = starts[0], -1
    for s in starts:
        d, _ = bfs(s)
        j = int(np.argmax(d))
        if d[j] > best_d:
            best_d, best_s = int(d[j]), s
    # 第二轮：从最远起点再找最远，得到直径两端
    d1, p1 = bfs(best_s)
    u = int(np.argmax(d1))
    d2, p2 = bfs(u)
    v = int(np.argmax(d2))
    if d2[v] <= 0:
        return None, 0.0, n
    path = [v]
    while path[-1] != u:
        nxt = int(p2[path[-1]])
        if nxt < 0:
            break
        path.append(nxt)
        if len(path) > n:
            break
    if len(path) < 2:
        return None, 0.0, n
    pts = np.column_stack([xs[path], ys[path]]).astype(np.float64)
    return pts, len(path) / float(n), n


def _crossing_arms(thin, pts_c):
    """检测"两只虫交叉"：中心线未覆盖的长分支（用户 2026-10-19 要求过滤）。

    两只虫交叉/搭茬时，掩膜是一个大连通域——现有"第二连通域面积>0.4x"的
    多虫判据完全看不出来，于是被当成一只奇形怪状的虫做分类（实测同一只
    worm15 的 088/126 两帧因此分进正弦和 Omega 两个类）。但交叉点在骨架
    上必然形成结点(度>=3)，而最长路径(中心线)只能盖住其中两条臂，剩下
    的臂整段不被中心线覆盖。

    单只虫即使盘绕自搭、Omega 头搭身，未覆盖的也只是小环/短刺，臂长相对
    中心线很小；两虫交叉的未覆盖臂是一整条虫段，臂长与中心线可比。
    【2026-10-20 晚】额外返回臂端距 arm_end_d：未覆盖臂自由端（非结点端）
    到中心线的最近距离。单虫 Omega/C 自搭时折返段的末端贴在虫身上
    （实测 worm12_778 端距=0.0，臂端就落在中心线上），而真两虫的臂是另一条
    虫的身子、末端伸向远方（实测 15~33px）——用于把"自搭环"从交叉证据里
    排除，避免单虫 Omega 被两虫判据误杀（跨批次普查发现的假阳性）。
    返回 (最长未覆盖臂长px, 臂长/中心线长, 结点数, 骨架臂数, 臂端距px)。
    """
    adj, ys, xs, deg, n = _skel_graph(thin)
    if adj is None:
        return 0.0, 0.0, 0, 0, 0.0
    special = [i for i in range(n) if deg[i] != 2]
    n_junc = sum(1 for i in special if deg[i] >= 3)
    if n_junc == 0:                     # 无结点：简单路径或纯环，不可能是交叉
        return 0.0, 0.0, 0, len(special), 0.0
    # 把骨架切成"特殊结点之间的段"：从每个结点/端点沿 deg==2 走到下一个特殊点
    seen, edges = set(), []
    for s in special:
        for nb in adj[s]:
            prev, cur, path, guard = s, nb, [s, nb], 0
            while deg[cur] == 2 and guard <= n:
                nxts = [v for v in adj[cur] if v != prev]
                if not nxts:
                    break
                prev, cur = cur, nxts[0]
                path.append(cur)
                guard += 1
            key = frozenset(path)       # 两端相向走同一条边只记一次
            if key in seen:
                continue
            seen.add(key)
            edges.append(path)
    if not edges:
        return 0.0, 0.0, n_junc, 0, 0.0
    center = set(map(tuple, np.round(pts_c).astype(int)))
    cpts = np.asarray(pts_c, float)
    longest, arm_end_d = 0, 0.0
    for path in edges:
        mid = path[len(path) // 2]      # 边中点：未覆盖边上必然不在中心线上
        if (int(xs[mid]), int(ys[mid])) in center:
            continue
        longest = max(longest, len(path) - 1)
        # 自由端（非结点端）到中心线最近点的距离：自搭环≈0，真两虫伸向远方
        for e in (path[0], path[-1]):
            if deg[e] >= 3:
                continue                # 结点端=附着点，跳过
            ep = np.array([xs[e], ys[e]], float)
            d = float(np.hypot(cpts[:, 0] - ep[0], cpts[:, 1] - ep[1]).min())
            arm_end_d = max(arm_end_d, d)
    L = max(len(pts_c) - 1, 1)
    return float(longest), longest / float(L), n_junc, len(edges), arm_end_d


def _width_profile(mask_final, pts):
    """沿中心线测量掩膜宽度剖面（距离变换采样）。

    单只虫的宽度沿体长基本均匀；两只虫交叉/搭茬处，掩膜是两条虫体的并集，
    局部宽度接近单虫的 2 倍——这是"两虫"最直接的几何证据（用户提示
    "虫身面积是大的"）。返回 (中位宽, 最大宽, 最大/中位, 粗端占比)。
    头尾各去掉 10% 渐细段再统计。
    """
    if mask_final is None or pts is None or len(pts) < 4:
        return 0.0, 0.0, 0.0, 0.0
    dt = cv2.distanceTransform((mask_final > 0).astype(np.uint8),
                               cv2.DIST_L2, 5)
    p = np.round(np.asarray(pts, float)).astype(int)
    p[:, 0] = np.clip(p[:, 0], 0, dt.shape[1] - 1)
    p[:, 1] = np.clip(p[:, 1], 0, dt.shape[0] - 1)
    w = 2.0 * dt[p[:, 1], p[:, 0]]
    k = max(1, len(w) // 10)
    core = np.sort(w[k:-k]) if len(w) > 2 * k + 2 else np.sort(w)
    if not len(core):
        return 0.0, 0.0, 0.0, 0.0
    med = float(np.median(core))
    mx = float(core[-1])
    fat_frac = float((core > 1.45 * med).mean()) if med > 0 else 0.0
    return med, mx, (mx / med if med > 0 else 0.0), fat_frac


def _trim_noise_tails(pts, resp, body_resp):
    """剪除中心线两端延伸进背景噪声的"噪声桥"段。

    背景的响应远低于虫身，但交界处响应剖面形态不同
    （2026-10-20 实测标定）：
      · 真虫尾尖：低响应段（<0.5*虫身）只有 6~17 点（渐细的尾巴），
        且响应不会跌到近 0（掩膜像素响应 >= 阈值，3x3 均值不会太小）；
      · 噪声桥：低响应段长达 18~58 点（占路径 28%~50%），或出现近零
        响应（resp<8，即 1px 细线穿过背景，邻域几乎全为背景），
        桥与虫身交界处响应在 2~5 点内跳回虫身水平。
    规则：从端点向内走，剪掉满足桥证据的连续段，到第一个虫身水平
    响应点为止。返回 (修剪后点集, 左剪点数, 右剪点数)。
    """
    n = len(pts)
    if n < 12 or body_resp < 40.0:
        return pts, 0, 0
    t_low = 0.50 * body_resp

    def _one(seq):
        k, run, near0 = 0, 0, False
        for i, v in enumerate(seq):
            if v < 8.0:            # 近零响应：一定是背景（1px 线穿背景）
                near0 = True
                run += 1
                k = i + 1
                continue
            if v < t_low:          # 低响应段
                run += 1
                if run >= 18:      # 连续 >=18 点低响应：真尾尖最长 17 点
                    k = i + 1
                continue
            break                  # 到达虫身水平响应，停止
        return k if (run >= 18 or near0) else 0

    kl = _one(resp)
    kr = _one(resp[::-1])
    kl = min(kl, int(0.7 * n))
    kr = min(kr, int(0.7 * n))
    if kl + kr == 0:
        return pts, 0, 0
    if kl + kr >= n - 8:           # 剪完剩不到 8 点：判定不可靠，放弃修剪
        return pts, 0, 0
    pts2 = pts[kl:n - kr] if kr > 0 else pts[kl:]
    return pts2, kl, kr


def _resp_along_path(resp_img, pts):
    """采样响应图沿中心线的剖面，返回 (resp数组, 虫身中位响应)。"""
    p = np.round(np.asarray(pts, float)).astype(int)
    p[:, 0] = np.clip(p[:, 0], 0, resp_img.shape[1] - 1)
    p[:, 1] = np.clip(p[:, 1], 0, resp_img.shape[0] - 1)
    rp = resp_img[p[:, 1], p[:, 0]]
    k = max(1, len(rp) // 5)
    core = np.sort(rp[k:-k]) if len(rp) > 2 * k + 2 else np.sort(rp)
    body = float(np.median(core)) if len(core) else float(np.median(rp))
    return rp, body


def extract_skeleton_full(image, return_debug=False, thr_override=None):
    """提取骨架，并返回中间处理结果（用于可视化对比/调参诊断）。

    返回：(points, mask_thresh, mask_final, field_type)
        points      : (N,2) 骨架点，失败时为 None
        mask_thresh : 顶帽/黑帽+阈值后的原始二值图
        mask_final  : 经过闭运算+取最大连通域后的掩膜（真正送去细化的图）
        field_type  : 'brightfield'(明场,浅底深虫) 或 'darkfield'(暗场,暗底亮虫)

    thr_override : 阈值覆盖（--threshold-override 解析出的单帧阈值）。
        非 None 时：该阈值若不在梯队会被补进梯队；候选选择强制取该阈值
        的合格候选（仍受两虫交叉过滤约束，产不出合格候选则回退常规选择）。

    return_debug=True 时额外返回第 5 个元素 dbg(dict)：含失败原因 reason 与各阶段
    统计（p2/p98/dyn、otsu、thr_ladder、选中的 thresh/area_frac/fill_ratio、
    n_skel_px/n_endpoints/n_junctions/area_per_skelpx、thin_method、morph_pass、
    passes 各遍拒绝记录、thr_override/thr_override_hit 等），
    供 diag_skeleton.py 精确定位；正常调用仍返回 4 元组。

    分割策略（2026-10-19 晚再修正：修复"腐蚀太猛把虫弄短"与"有虫却报失败"）：
      1. 黑帽/顶帽增强只算一次；Otsu 自适应阈值 + 固定锚点组成阈值梯队；
      2. 三遍形态学配置依次尝试：仅闭运算 -> 3x3 开运算(兜底) -> 原始阈值；
         （5x5 开运算会啃掉渐变细的虫尾、勒断细 S 虫，已废）
      3. 每遍内每个阈值：
         a. 连通域分析 + 背景带状伪影排除（_is_border_band）；
         b. 【新】碎片合并（_merge_near_components）：面积 >=10%最大域 的非带状
            连通域，两两 bbox 间距 <= merge_gap(≈0.07*对角线，5~9px) 并成一组。
            黑帽响应在虫体中段常低于阈值，把虫体打成头/中/尾几段，旧版只取
            最大连通域，骨架凭空少一截（实测 worm7_187 掩膜 370->314px、
            worm30_104 360->270px、worm7_078 372->339px，用户报"识别短了"）；
         c. 【新】小组件孔洞填充（_fill_small_holes）：<=8%组面积 的内部孔
            （肠线/虫体纹理在二值图上的空洞）填实。旧版这些孔让骨架绕孔成环、
            结点数爆炸，被"结点<=4"闸门整批误杀（5000 张实测 94.8% 失败的
            主因；worm11_091/worm1_109 等掩膜很干净却报"未检测到"即此因）。
            盘绕虫体围住的大孔不填，防止 C 形虫被填实把骨架走直；
         d. 多虫判定：其他组（非带状）面积 >0.4x最大组 且虫形
            (fill∈[0.15,0.85]) 才拒，防刮痕冒充第二只虫；
         e. 面积/填充率闸门（fill∈[0.05,0.85]，下限照顾横贯整帧的细 S 虫）
            + 虫形评分，最多取 6 个候选做细化（旧版只试前 3 个，常把能过的
            高档阈值漏掉，例 worm7_187 的 thr137 实为最干净掩膜）；
      4. 【新】中心线提取（_skeleton_centerline）替代"贪心最近邻+跳断截断"：
         细化剪枝后，在骨架像素图上做双端 BFS 取最长路径作为中心线，头尾即
         路径两端。毛刺/小环/阶梯状双边缘不再靠结点数闸门硬拒；也不再出现
         旧版 max_jump=6 截断把虫尾砍掉（"识别短了"的直接原因）；
      5. 健全性检查：中心线长 8~max_skel、覆盖率(路径像素/骨架像素)>=0.5、
         掩膜面积/中心线长 ∈[2,25]；端点/结点仅作诊断记录，不再当闸门。
      6. 【新】多虫交叉过滤（用户 2026-10-19 要求）：两只虫交叉时掩膜连成
         一个大连通域，"第二连通域"判据看不出来，整帧过滤（label=-1，
         reason 注明"疑似多虫交叉"），不做形态分类。两条判据满足其一即过滤：
         a. 臂判据（X 形交叉）：骨架在交叉点形成结点、中心线只盖得住两条臂，
            剩下的臂整段不被覆盖（_crossing_arms）；未覆盖臂 >=
            max(14px, 0.32*中心线长)。单虫盘绕自搭/Omega 头搭身的未覆盖
            分支只是小环短刺，不会触发；
         b. 宽度判据（λ 形/近平行搭茬）：两虫头尾搭茬时中心线把两条虫身
            粘成一条长路径，未覆盖臂只剩 8~13px 小桩，臂判据抓不住；但掩膜
            局部宽度接近单虫 2 倍（_width_profile 的 max/median）且总面积
            偏大。5000 张实测标定：wr>=1.50 的单虫 af<=0.105、两虫
            af>=0.129，故 af>=0.115 且 wr>=1.50 判交叉。盘绕单虫 af 高达
            0.166 但 wr<=1.52，worm30 类局部肥厚单虫 wr 1.55 但 af<=0.105，
            均不会误触发（单维判据都会误杀，必须二维联合）。
    """
    dbg = {}

    def _ret(points, mth, mfin, reason):
        dbg['reason'] = reason
        if return_debug:
            return points, mth, mfin, field_type, dbg
        return points, mth, mfin, field_type

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 低对比度归一化（百分位拉伸）：z_tracking_only 裁剪的活虫图对比度极低
    # （淡虫淡底），黑帽响应会低于固定阈值 25 导致二值图全空、整批无有效骨架。
    # 把 2%~98% 灰度百分位线性映射到 0~255：低对比图对比度被放大，本就正常的
    # 高对比图因 p98-p2 已接近全量程而近似无操作（向后兼容旧数据）。
    _p2, _p98 = np.percentile(gray, (2, 98))
    dbg['p2'], dbg['p98'], dbg['dyn'] = float(_p2), float(_p98), float(_p98 - _p2)
    if _p98 > _p2 + 1:
        gray = np.clip((gray.astype(np.float32) - _p2) /
                       (_p98 - _p2) * 255.0, 0, 255).astype(np.uint8)

    empty = np.zeros_like(gray)

    # 中值滤波去椒盐噪声，保边
    blurred = cv2.medianBlur(gray, 3)

    # 自动判断明暗场，选择对应形态学滤波
    field_type = estimate_field_type(gray)
    dbg['field_type'] = field_type
    h, w = gray.shape
    dbg['size'] = (int(h), int(w))
    # 背景估计核：只需要比虫体最宽处大即可（虫宽约 8~14px）。
    # 旧版 min(h,w)//2 ≈ 41px 接近整图，导致黑帽响应带光晕、掩膜比虫体胖一圈，
    # 且容易把背景污渍粘连进来（实测 8096px 的糊块）。限制在 15~25px。
    kernel_bg_size = min(h, w) // 3
    kernel_bg_size = max(15, min(25, kernel_bg_size))
    if kernel_bg_size % 2 == 0:
        kernel_bg_size += 1
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (kernel_bg_size, kernel_bg_size))

    # 边界反射填充后再做形态学，消除图像边界的伪响应。
    # 黑帽/顶帽在图像边角会产生贴边的假前景，让虫尾"延伸"出图像、
    # 或把边角噪点放大，填充后这些伪响应都会消失。
    pad = kernel_bg_size
    blurred = cv2.copyMakeBorder(blurred, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    if field_type == 'darkfield':
        # 白帽（top-hat）：原图 - 开运算(估计背景)，保留比背景亮的细小物体
        background = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel_bg)
        enhanced = cv2.subtract(blurred, background)
        # 暗场顶帽响应里背景接近 0，低阈值即可
        thresh = 10
    else:
        # 黑帽（black-hat）：闭运算(估计背景) - 原图，保留比背景暗的细小物体
        background = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel_bg)
        enhanced = cv2.subtract(background, blurred)
        # 明场黑帽响应里虫体边缘有一圈半暗过渡带（光晕），
        # 阈值 10 会把光晕也抓进来导致掩膜发胖；实测 25 既能贴合虫体
        # 又能滤掉污渍粘连（35 开始细虫断裂，不可再高）
        thresh = 25

    enhanced = enhanced[pad:-pad, pad:-pad]
    dbg['enh_max'] = int(enhanced.max())
    dbg['enh_mean'] = round(float(enhanced.mean()), 1)
    dbg['enhanced'] = enhanced   # 黑帽/顶帽响应图，供端点修剪与诊断使用
    img_area = float(h * w)
    dbg['img_area'] = int(img_area)

    # Otsu 自适应阈值：按本图对比度自动分离虫体/背景（低对比/高对比都稳）
    otsu_t, _ = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dbg['otsu'] = int(otsu_t)

    # 多套阈值方案（梯队）：以 Otsu 为核心向高低各扩一档，再加固定锚点。
    # 不同图最优阈值差异大（诊断实测 thr8~Otsu 都出现过），故不死守单一阈值。
    cand = {max(5, int(otsu_t * f)) for f in (0.6, 0.8, 1.0, 1.3)}
    cand |= {12, 20, 25, 35}
    if field_type == 'darkfield':
        cand.add(10)
    thr_ladder = sorted(t for t in cand if 5 <= t <= 250)
    # 【2026-10-20 晚 低端泛洪剔除】百分位拉伸把低对比图的背景纹理一起放大
    # （dyn=42 的图噪声被放大 6 倍），低端阈值下纹理连成一片、最大连通域
    # 直接泛洪整帧——这种阈值在本图没有分割意义，产出的候选只会让骨架沿
    # 纹理游走（实测 control/0120/C/worm22_014：t12~t54 掩膜 bbox 盖住
    # 84%~92% 画面，"选最长"选中泛洪候选、骨架 110 点里约 40 点在纹理里；
    # 同图 t72 掩膜 556px 干净）。判别量：原始二值最大域的 bbox 覆盖率
    # >0.80（单只虫上限 ~0.73，实测 15 帧健康样本；泛洪帧 0.84~1.00）。
    # 只剔 <0.75*otsu 的低端：高阈值才是两虫持久证据的来源，不受影响
    # （真两虫帧 t60+ 的 bbox 仅 0.45~0.60，从不到泛洪线）。
    flood_dropped = []
    for t in thr_ladder:
        if t >= 0.75 * otsu_t:
            break
        try:
            _, bfl = cv2.threshold(enhanced, t, 255, cv2.THRESH_BINARY)
            nf, labf, stf, _ = cv2.connectedComponentsWithStats(
                bfl, connectivity=8)
            if nf <= 1:
                continue
            aif = int(np.argmax(stf[1:, cv2.CC_STAT_AREA])) + 1
            if (stf[aif, cv2.CC_STAT_WIDTH] * stf[aif, cv2.CC_STAT_HEIGHT]
                    > 0.80 * img_area):
                flood_dropped.append(t)
        except Exception:
            continue
    if flood_dropped:
        thr_ladder = [t for t in thr_ladder if t not in flood_dropped]
        dbg['flood_dropped'] = flood_dropped
    dbg['thr_ladder'] = thr_ladder
    # 【2026-10-20 阈值覆盖】--threshold-override 指定的阈值补进梯队，
    # 保证它一定能产出候选（梯队外的值如 61 不在 {12,20,25,35,
    # 0.6/0.8/1.0/1.3×otsu} 里，不补则覆盖静默失效）
    dbg['thr_override'] = thr_override
    dbg['thr_override_hit'] = False
    if thr_override is not None:
        if not (5 <= thr_override <= 250):
            dbg['thr_override_invalid'] = True
            thr_override = None
        elif thr_override not in thr_ladder:
            thr_ladder = sorted(thr_ladder + [thr_override])
            dbg['thr_ladder'] = thr_ladder

    # 单条虫的尺寸/形状合理区间（按图大小归一）。
    # min_area 下限从 0.02*img_area 降到 0.012：本数据集全是 64~96px 小图，
    # 单虫约占画面 0.02~0.05，旧下限 0.02 已接近虫体本身大小，会把偏淡/
    # 偏细的真虫一并卡掉（实测多例干净掩膜 250~300px 被 min_area 拒）。
    min_area = max(20.0, 0.012 * img_area)
    max_area = 0.55 * img_area          # 单虫不超过半个画面；超过多为糊块或两只虫
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    diag = float(np.hypot(h, w))
    max_skel = max(120.0, 1.25 * diag)  # 单虫骨架长度上限；过长=两只虫粘连
    # 碎片合并间隙：黑帽响应不均会把虫体打成头/中/尾几段，bbox 间距不超过
    # 此值的连通域并回一组。取 0.07*对角线(约 8~9px)：足以跨过典型 1~5px
    # 断口，又小于两只虫的间距（虫宽约 8~12px），不会粘并相邻两只虫。
    merge_gap = max(5.0, min(9.0, 0.07 * diag))
    # 多虫交叉判据（用户 2026-10-19 要求：两虫交叉不做形态分类，整帧过滤）。
    # 交叉时掩膜连成一个连通域，"第二连通域"判据看不出来，但骨架在交叉点
    # 形成结点，中心线(最长路径)只盖得住两条臂，剩下的臂整段不被覆盖。
    # 单虫盘绕自搭 / Omega 头搭身的未覆盖分支只是小环短刺，臂长占比远小于此；
    # 面积占比佐证只作记录，主判据是臂长（用户提示"折角小、面积大、体长长"）。
    cross_arm_min = 14.0                # 未覆盖臂长绝对值下限(px)，防微环误判
    cross_arm_ratio = 0.32              # 未覆盖臂长 / 中心线长
    # 宽度×面积联合判据（λ 形/近平行搭茬，臂判据抓不住的情形）：
    # 两虫头尾搭茬近平行时，中心线把两条虫身粘成一条长路径，未覆盖臂只剩
    # 8~13px 小桩（实测 worm15_088/126），臂判据失效。但掩膜是两条虫体的
    # 并集：局部宽度接近单虫 2 倍(宽度剖面 wr=max/median) 且总面积偏大。
    # 5000 张实测标定（2026-10-19）：wr>=1.50 的帧里，单虫（worm30 局部肥厚）
    # af<=0.105，两虫（worm15/28/32 序列）af>=0.129，间隙干净；盘绕单虫
    # af 高达 0.166 但 wr<=1.52，不受影响。阈值取间隙中点。
    cross_af_min = 0.115                # 面积占比下限（单虫上限 0.105 / 两虫下限 0.129）
    cross_wr_min = 1.50                 # 最大/中位宽度比下限（单虫上限 1.52 是最密缠绕类）
    # 高阈值持久判据（2026-10-20）：真两虫物理叠压、重叠区更暗，阈值调高后
    # 双宽结构去不掉——wr 仍 >=1.55（实测 4 例 1.67~1.78）；而低阈值光晕
    # 糊块的宽度膨胀是假的，阈值一高 wr 塌回 1.0~1.33（实测 4 例单虫）。
    # 该判据不要求 af（高阈值下光晕面积已褪去，af 自然回落到 0.05~0.11）。
    cross_wr_persist = 1.55
    # 臂端距闸门（2026-10-20 晚，跨批次普查后加）：未覆盖臂判据要求臂的
    # 自由端离中心线 >= 该值才算"另一条虫的身子"。单虫 Omega/C 自搭时
    # 折返段末端贴在虫身上（实测 worm12_778 端距 0.0——臂端就落在中心
    # 线上；worm7_126 的 C 形同理），真两虫的臂末端伸向远方（实测
    # 15~33px）。取 8px：>虫身宽(4~7px)的 1.2 倍，<真两虫实测下限 15px。
    arm_end_self = 8.0
    # 干净掩膜优先（2026-10-20 下午，用户报 5 帧"二值化噪声严重/骨架扎进
    # 噪声"后标定）：低锚点阈值(12/20/25/35)的糊掩膜把背景暗噪声并进来，
    # 中心线沿噪声多走 14~22 点，"选最长"反而让这种脏候选胜出——但剖面
    # 层面（响应/宽度/转向角）无法区分"噪声细丝"与"渐细真尾尖"（实测
    # worm7_187 真尾尖与 worm1_004 噪声丝的响应游程同为 14~16 点），
    # 只能靠掩膜整体干净度在候选间裁决：
    #   · 最长候选 resp_frac < mask_clean_min（掩膜大部分不是虫身）；
    #   · 且存在干净候选(>= mask_clean_min)长度 >= clean_len_tol*最长；
    #   → 改选最干净的那个（resp_frac 高者优先，长度/覆盖率决胜）。
    # 干净候选落后超过 tol 时仍取最长：control 明场组淡尾必须靠低阈值糊膜
    # 才够得到端点（ctl worm11_001 65pt vs 干净 52pt，tol=0.82 恰好保住），
    # 实测 12 帧标定：worm27_056/worm1_004/worm23_068 修正（98→81/80→66/
    # 82→78 点且掩膜变干净），全部 control 哨兵与 worm7_187 长尾不变。
    mask_clean_min = 0.85              # 掩膜"干净"的响应占比下限
    clean_len_tol = 0.82               # 干净候选可接受的最短长度比例

    def _collect(morph):
        """对一个形态学配置跑整条阈值梯队。

        morph: 'close'=仅close / 'open3'=close+3x3开运算(兜底) / 'raw'=不处理。
        返回 (候选列表, 多虫拒数, 其他拒数, 末张二值图)。每个阈值：
          阈值 -> 形态学 -> 碎片合并(merge_gap) -> 小组件孔洞填充 ->
          取最大非带状组为虫体 -> 多虫判定(其他组须虫形) -> 面积/填充率闸门。
        候选选取规则：
          · 背景带状伪影既不能当选虫体，也不参与多虫判定（见 _is_border_band）；
          · 碎片合并：同一只虫被阈值打断成几段时并回整条，避免只取最大段
            导致骨架短一截（实测 worm7_187 370->314px、worm30_104 360->270px）；
          · 多虫拒：其他组面积 > 0.4x最大组 且 fill∈[0.15,0.85]；
          · 面积/填充率闸门 + 虫形评分。评分目标面积占比 0.035：本数据集为
            64~96px 小图，单虫约占画面 0.02~0.05，旧目标 0.12 偏大 3 倍，
            会系统性偏爱带光晕/粘连的低阈值糊掩膜（那正是结点爆炸的来源）。
        """
        cands, n_multi, n_other = [], 0, 0
        last_binary = None
        for thr in thr_ladder:
            try:
                _, binary = cv2.threshold(enhanced, thr, 255, cv2.THRESH_BINARY)
                last_binary = binary
                cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                           kernel3, iterations=1)
                if morph == 'open3':
                    # 3x3 开运算仅作兜底：去掉 1~2px 的粘连细线，
                    # 比 5x5 温和，不会啃断虫体/虫尾
                    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel3)
                # 碎片合并：把被阈值打断的虫体各段并回一组
                groups = _merge_near_components(cleaned, merge_gap)
                if not groups:
                    n_other += 1
                    continue
                best_g = None
                for g in groups:          # 最大域跳过带状伪影：背景污带永远不是虫
                    if not g['is_band']:
                        best_g = g
                        break
                if best_g is None:
                    n_other += 1
                    continue
                best_a = best_g['area']
                # 多虫判定：其他非带状组面积 >0.4x最大组且虫形才拒
                multi = False
                for g in groups:
                    if g is best_g or g['is_band']:
                        continue
                    if g['area'] > 0.40 * best_a and g['area'] > min_area:
                        gx1, gy1, gx2, gy2 = g['bbox']
                        fillg = g['area'] / ((gx2 - gx1 + 1) *
                                             (gy2 - gy1 + 1) + 1e-6)
                        if 0.15 <= fillg <= 0.85:
                            multi = True
                            break
                if multi:
                    n_multi += 1
                    continue
                if best_a < min_area or best_a > max_area:
                    n_other += 1
                    continue
                # 小组件孔洞填充（肠线/纹理空洞），再按填充后的掩膜算面积/填充率
                mask_final = _fill_small_holes(best_g['mask'],
                                               max(16.0, 0.08 * best_a))
                bx1, by1, bx2, by2 = best_g['bbox']
                bw = bx2 - bx1 + 1
                bh = by2 - by1 + 1
                m_area = float((mask_final > 0).sum())
                fill = m_area / (bw * bh + 1e-6)
                # fill 下限 0.05：横贯整帧的细 S 虫 bbox 大而面积小
                if fill > 0.85 or fill < 0.05:      # 实心糊块 / 过于稀疏
                    n_other += 1
                    continue
                area_frac = m_area / img_area
                score = -(abs(area_frac - 0.035) + 1.0 * abs(fill - 0.32))
                cands.append({'score': score, 'thr': thr, 'mask_thresh': binary,
                              'mask_final': mask_final,
                              'area_frac': round(area_frac, 3),
                              'fill': round(fill, 3),
                              'n_groups': len(groups)})
            except Exception:
                n_other += 1
                continue
        return cands, n_multi, n_other, last_binary

    def _try_thin(cands):
        """对得分最高的前 6 个候选做细化 + 中心线提取 + 健全性检查。

        合格条件（单条虫中心线的拓扑/比例特征）：
          · 中心线长 8 ~ max_skel；
          · 覆盖率 coverage = 中心线像素/骨架像素 >= 0.5
            （阶梯状双边缘/烂网的覆盖率低，借此拒掉；简单虫形应接近 1）；
          · 掩膜面积/中心线长 ∈ [2, 25]（单虫约 5~14；粘连糊块 25+，拒）；
          · 【新】不是多虫交叉：中心线未覆盖的长臂
            (>= max(cross_arm_min, cross_arm_ratio*中心线长)) 说明骨架在
            交叉点分成三条以上长臂，是两只虫交叉（用户要求整帧过滤，
            不做形态分类）。单虫自搭环/Omega 头搭身的未覆盖分支很小，
            不会触发。
        端点/结点数不再当闸门——毛刺/小环由"剪枝 + 最长路径"天然消化，
        旧版"结点<=4"把带内部孔洞的真虫整批误杀（5000 张 94.8% 失败主因）。
        返回 (最佳候选 or None, 各候选检查记录,
              交叉拒绝统计(拒绝数, 最长臂px, 最大臂占比))。
        """
        best, why = None, []
        oks = []                        # 全部合格候选，末尾统一裁决
        n_cross, cross_arm_px, cross_arm_seen = 0, 0.0, 0.0
        n_fat, fat_wr, fat_af = 0, 0.0, 0.0
        # 【2026-10-20 control 组修正】不再"第一个通过闸门就选中"：肥掩膜
        # （明场光晕）细化出的梯状渔网上，虫体两端常是 3~6px 短刺，旧
        # max_spur=6 会把尖端剪掉、最长路径随之够不到两端（骨架偏短的
        # 直接原因）。现在：剪枝只去 1~2px 毛刺，且全部候选都细化
        # 评估，选中心线最长（其次覆盖率最高）的——直接以"骨架贯穿
        # 全虫"为目标选候选。
        # 【帧级过滤的证据分级】低阈值候选的交叉旗常是"光晕糊块"伪证据
        # （单虫+光晕 wr 可达 2.5+、af 0.3+，阈值一高就塌回 1.0~1.3）；
        # 真两虫物理叠压、重叠区更暗，高阈值下 wr 仍维持 >=1.55
        # （实测 4 例真两虫 1.67~1.78，4 例误杀单虫高阈 1.0~1.33）。
        # 因此：交叉旗一律取消该候选资格；但只有 thr>=persist_thr 的
        # 候选仍触发交叉，才作为整帧过滤的"持久证据"。
        persist_thr = max(25, int(0.75 * otsu_t))
        n_persist = 0
        # 响应图（enhanced 的 3x3 均值）：用于噪声桥修剪
        _k3 = np.ones((3, 3), np.float32) / 9.0
        resp_img = cv2.filter2D(enhanced.astype(np.float32), -1, _k3,
                                borderType=cv2.BORDER_REPLICATE)
        # 【2026-10-20 噪声桥修正】掩膜"真实虫体占比" resp_frac：
        # 掩膜像素中黑帽响应 >= persist_thr 的比例。干净掩膜（真虫体为主）
        # 接近 1.0；低阈值(thr=12/20)糊掩膜把背景暗噪声桥并进来，桥上
        # 响应只有 20~40（虫身 130~180），resp_frac 掉到 0.5~0.6。
        # "选最长中心线"会偏爱这种糊掩膜（骨架沿噪声桥多延伸 30~70px，
        # 实测 worm3_084=155点/worm23_148=154点/worm13_014=97点，
        # 其中 44/56/49 点在背景里）。resp_frac 低 = 掩膜大部分不是虫，
        # 取消候选资格（高阈值候选响应天然全 >= persist_thr，不受影响）。
        for c in cands:
            try:
                thin, method = _thin_mask(c['mask_final'])
                ncc, labc, stc, _ = cv2.connectedComponentsWithStats(
                    (thin > 0).astype(np.uint8), connectivity=8)
                if ncc > 2:              # 骨架碎成多段时只保留最长段
                    keep = stc[1:, cv2.CC_STAT_AREA].argmax() + 1
                    thin = ((labc == keep) * 255).astype(np.uint8)
                thin = prune_skeleton_spurs(thin, max_spur=2)
                pts_c, coverage, n_skel_px = _skeleton_centerline(thin)
                _, n_end, n_junc = _skel_topology(thin)
                if pts_c is None:
                    why.append('thr%d:中心线提取失败' % c['thr'])
                    continue
                # 【2026-10-20 噪声桥修剪】先按响应剖面剪掉两端背景噪声桥，
                # 长度/覆盖率/面积比全部按修剪后的真实虫体中心线评估——
                # "选最长"比的必须是真虫长度，否则低阈值糊掩膜靠噪声桥
                # 偏长的骨架总会胜出（实测 worm3_084 155点中 91 点在背景）。
                # 交叉/宽度判据仍用原始路径（与既有标定一致，且剪短的
                # 路径会把噪声桥错算成"未覆盖臂"触发假交叉）。
                rp, body_r = _resp_along_path(resp_img, pts_c)
                pts_t, n_tr_l, n_tr_r = _trim_noise_tails(pts_c, rp, body_r)
                if len(pts_t) < 8:
                    why.append('thr%d:修剪后中心线过短(%d点)'
                               % (c['thr'], len(pts_t)))
                    continue
                L = len(pts_t)
                coverage = L / max(n_skel_px, 1)
                ratio = (c['area_frac'] * img_area) / max(L, 1)
                arm_px, arm_ratio, _nj2, n_arms, arm_end_d = _crossing_arms(
                    thin, pts_c)
                w_med, w_max, w_ratio, w_fat = _width_profile(c['mask_final'],
                                                              pts_c)
                # 掩膜真实虫体占比（黑帽响应 >= persist_thr 的像素比例），
                # 用于拒绝"低阈值糊掩膜把背景噪声桥并进来"的候选
                _m = c['mask_final'] > 0
                resp_frac = float((_m & (enhanced >= persist_thr)).sum()) \
                    / max(1.0, float(_m.sum()))
                # 两条多虫判据（满足其一即交叉，见 cross_arm_min/cross_af_min 注释）：
                #   臂判据：X 形交叉，中心线漏掉一整条臂；
                #   宽度判据：λ 形/近平行搭茬，中心线粘成一条但掩膜
                #     局部宽度翻倍 + 总面积偏大（二维联合，防单虫误杀）
                # 【2026-10-20 晚】臂判据加"臂端距"闸门：未覆盖臂的自由端
                # 贴在虫身上(端距 < arm_end_self，Omega/C 自搭折返)说明这是
                # 单虫自己折回来的，不是另一条虫——只有臂端伸向远方才算交叉
                # 证据（跨批次普查：单虫 Omega 曾被臂判据误杀，实测端距=0.0）
                arm_cross = (arm_px >= max(cross_arm_min, cross_arm_ratio * L)
                             and arm_end_d >= arm_end_self)
                fat_cross = (c['area_frac'] >= cross_af_min
                             and w_ratio >= cross_wr_min)
                # 高阈值下宽度仍翻倍：两只真虫叠压的持久证据（无论 af），
                # 该候选本身也 disqualified——它的掩膜里是两条虫
                hi_double = (c['thr'] >= persist_thr
                             and w_ratio >= cross_wr_persist)
                crossing = arm_cross or fat_cross or hi_double
                ok = (8 <= L <= max_skel and coverage >= 0.5
                      and 2.0 <= ratio <= 25.0 and not crossing)
                why.append('thr%d:%s' % (
                    c['thr'], ('ok(长%d/实%.2f)' % (L, resp_frac)) if ok else
                    ('交叉(臂%.0fpx/端%.0f/宽%.2f/面积%.3f/实%.2f)'
                     % (arm_px, arm_end_d, w_ratio, c['area_frac'], resp_frac))
                    if crossing else
                    '盖%.2f/端%d/结%d/长%d/比%.0f/实%.2f' % (
                        coverage, n_end, n_junc, L, ratio, resp_frac)))
                if not ok:
                    if crossing:
                        n_cross += 1
                        cross_arm_px = max(cross_arm_px, arm_px)
                        cross_arm_seen = max(cross_arm_seen, arm_ratio)
                        if fat_cross:
                            n_fat += 1
                            fat_wr = max(fat_wr, w_ratio)
                            fat_af = max(fat_af, c['area_frac'])
                        if c['thr'] >= persist_thr:
                            n_persist += 1
                    continue
                c = dict(c)
                c.update({'thin': thin, 'points': pts_t, 'n_pts': L,
                          'method': method, 'n_end': n_end,
                          'n_junc': n_junc, 'ratio': round(ratio, 1),
                          'coverage': round(coverage, 2),
                          'arm_px': round(arm_px, 1),
                          'arm_ratio': round(arm_ratio, 2),
                          'n_arms': n_arms,
                          'w_med': round(w_med, 1), 'w_max': round(w_max, 1),
                          'w_ratio': round(w_ratio, 2),
                          'w_fat': round(w_fat, 2),
                          'resp_frac': round(resp_frac, 2),
                          'n_trim': int(n_tr_l + n_tr_r)})
                # 全候选择优（2026-10-20 下午修订）：先收集，末尾裁决。
                # 默认"中心线最长优先（贯穿全虫），覆盖率决胜"；但最长候选
                # 掩膜不干净(resp_frac<mask_clean_min)时，若干净候选长度
                # 在 clean_len_tol 内则改选干净候选——不让"靠并背景噪声
                # 变长"的脏掩膜胜出（规则与标定见 mask_clean_min 注释）。
                oks.append(c)
            except Exception:
                continue
        if not oks:
            return None, why, (n_cross, cross_arm_px, cross_arm_seen,
                               n_fat, fat_wr, fat_af, n_persist)
        best = max(oks, key=lambda c: (c['n_pts'], c['coverage']))
        if thr_override is not None:
            # 【2026-10-20 阈值覆盖】人工点名阈值最高优先：只在该阈值的
            # 合格候选里挑（覆盖率决胜）。该阈值产不出合格候选（含触发
            # 两虫交叉被拒）时不强杀，回退下面的常规选择。
            forced = [c for c in oks if c['thr'] == thr_override]
            if forced:
                usual = best
                best = max(forced, key=lambda c: (c['coverage'], c['n_pts']))
                dbg['thr_override_hit'] = True
                why.append('==> 阈值覆盖: 强制 thr%d(%d点/实%.2f)，'
                           '常规最长为 thr%d(%d点/实%.2f)'
                           % (best['thr'], best['n_pts'], best['resp_frac'],
                              usual['thr'], usual['n_pts'],
                              usual['resp_frac']))
            else:
                why.append('==> 阈值覆盖 thr%d 无合格候选(可能触发两虫'
                           '交叉)，回退常规选择' % thr_override)
        elif best['resp_frac'] < mask_clean_min:
            l_max = best['n_pts']
            cleans = [c for c in oks
                      if c['resp_frac'] >= mask_clean_min
                      and c['n_pts'] >= clean_len_tol * l_max]
            if cleans:
                switched = max(cleans, key=lambda c: (c['resp_frac'],
                                                      c['n_pts'],
                                                      c['coverage']))
                why.append('==> 干净优先: thr%d(%d点/实%.2f) 替代 '
                           'thr%d(%d点/实%.2f)'
                           % (switched['thr'], switched['n_pts'],
                              switched['resp_frac'], best['thr'],
                              best['n_pts'], best['resp_frac']))
                best = switched
        return best, why, (n_cross, cross_arm_px, cross_arm_seen,
                           n_fat, fat_wr, fat_af, n_persist)

    # 三遍形态学配置：仅闭运算 -> 3x3 开运算兜底 -> 原始阈值。
    # 教训：曾用 5x5 开运算"去细线刮痕"，实测会把逐渐变细的虫尾啃掉
    # （worm7_187 掩膜 370->314px、骨架仅 51pt，Omega 判定基于残骨架），
    # 甚至把细 S 虫从中间勒断成碎片导致整帧失败（worm1_109 等，5000 张
    # 实测 94.8% 失败的主因之一）。已废除 5x5；去细线改由
    # "多虫判定的第二连通域须虫形(fill2∈[0.15,0.85])" + 骨架拓扑检查承担，
    # 粘连细线由 3x3 兜底遍处理。
    passes = [('close', '仅闭运算'), ('open3', '3x3开运算兜底'), ('raw', '原始阈值')]
    best = None
    last_binary = None
    pass_log = []
    n_multi_total = 0
    cross_cnt, cross_arm_px, cross_arm_seen = 0, 0.0, 0.0
    fat_cnt, fat_wr, fat_af = 0, 0.0, 0.0
    persist_cnt = 0
    for morph, mname in passes:
        cands, n_multi, n_other, last_binary = _collect(morph)
        n_multi_total += n_multi
        pass_log.append('%s(多虫拒%d/其他拒%d/候选%d)'
                        % (mname, n_multi, n_other, len(cands)))
        if not cands:
            continue
        best, why, cstat = _try_thin(cands)
        cross_cnt += cstat[0]
        cross_arm_px = max(cross_arm_px, cstat[1])
        cross_arm_seen = max(cross_arm_seen, cstat[2])
        fat_cnt += cstat[3]
        fat_wr = max(fat_wr, cstat[4])
        fat_af = max(fat_af, cstat[5])
        persist_cnt += cstat[6]
        pass_log.extend(why)
        if best is not None:
            best['pass'] = mname
            break

    dbg['n_multi_reject'] = n_multi_total
    dbg['n_cross_reject'] = cross_cnt
    dbg['n_fat_cross'] = fat_cnt
    dbg['n_persist_cross'] = persist_cnt
    dbg['passes'] = pass_log
    if last_binary is None:
        last_binary = np.zeros_like(enhanced, dtype=np.uint8)

    if best is not None and persist_cnt > 0:
        # 帧级过滤（用户 2026-10-19 要求"两只虫不分类"）：只认"高阈值持久
        # 证据"——thr>=0.75*otsu 的候选（光晕/噪声已被阈值滤掉）仍触发交叉
        # 判据，说明双宽/多臂结构是两只真虫物理叠压所致。低阈值候选的
        # 孤立交叉旗只是光晕糊块伪证据，只取消该候选资格，不处死整帧
        # （否则 worm30_104/6_532/18_367/15_150 等单虫被误杀，
        # 2026-10-20 实测）。
        dbg['cross_frame_override'] = True
        best = None

    if best is None:
        if cross_cnt > 0:
            # 两虫交叉：掩膜连成一个大连通域，"第二连通域"判据看不出来。
            # 臂判据抓 X 形交叉；宽度判据抓 λ 形/近平行搭茬。按用户要求
            # 整帧过滤，不做形态分类（label=-1，reason 注明多虫交叉）。
            if fat_cnt > 0:
                return _ret(None, last_binary, empty,
                            '疑似多虫交叉(未覆盖臂%.0fpx=中心线%.0f%%；'
                            '宽度比%.2f+面积占比%.3f 双超单虫上限)，'
                            '按两只虫过滤，不做形态分类'
                            % (cross_arm_px, cross_arm_seen * 100,
                               fat_wr, fat_af))
            return _ret(None, last_binary, empty,
                        '疑似多虫交叉(未覆盖臂%.0fpx=中心线%.0f%%)，按两只虫过滤，'
                        '不做形态分类' % (cross_arm_px, cross_arm_seen * 100))
        return _ret(None, last_binary, empty,
                    '三遍形态学均无合格单虫骨架: ' + ' | '.join(pass_log))

    points = smooth_skeleton(best['points'])
    if len(points) < 5:
        return _ret(None, best['mask_thresh'], best['mask_final'], '中心线<5点')
    dbg.update({'thresh': best['thr'], 'max_area': int(best['area_frac'] * img_area),
                'fill_ratio': best['fill'], 'n_skel_px': best['n_pts'],
                'thin_method': best['method'], 'area_frac': best['area_frac'],
                'coverage': best.get('coverage'), 'n_endpoints': best['n_end'],
                'n_junctions': best['n_junc'], 'area_per_skelpx': best['ratio'],
                'arm_px': best.get('arm_px'), 'arm_ratio': best.get('arm_ratio'),
                'n_arms': best.get('n_arms'),
                'w_med': best.get('w_med'), 'w_max': best.get('w_max'),
                'w_ratio': best.get('w_ratio'), 'w_fat': best.get('w_fat'),
                'resp_frac': best.get('resp_frac'),
                'n_trim': best.get('n_trim', 0),
                'morph_pass': best['pass']})
    return _ret(points, best['mask_thresh'], best['mask_final'],
                'ok(thr=%d,%d点,%s)' % (best['thr'], len(points), best['pass']))


def simple_thin(binary):
    """Zhang-Suen 骨架细化（纯 numpy，无外部依赖）。

    当 opencv-contrib 的 ximgproc 与 scikit-image 均不可用时的 fallback。
    旧版 fallback 用"邻居 2~6 即删"的简化规则，会把实心虫体腐蚀到只剩 1~2 像素
    （曾致 z_tracking_only 新数据整批骨架提取失败、细化为 n=1/2）。
    现改为标准 Zhang-Suen 算法，已用"实心矩形->单像素线"在沙箱验证正确。
    """
    img = (binary > 0)
    h, w = img.shape
    if h < 3 or w < 3:
        return binary.copy()
    changed = True
    while changed:
        changed = False
        for step in (1, 2):
            p2 = np.zeros_like(img); p2[1:, :]   = img[:-1, :]   # N  = img[y-1,x]
            p3 = np.zeros_like(img); p3[1:, :-1] = img[:-1, 1:]  # NE = img[y-1,x+1]
            p4 = np.zeros_like(img); p4[:, :-1]  = img[:, 1:]    # E  = img[y,x+1]
            p5 = np.zeros_like(img); p5[:-1, :-1]= img[1:, 1:]   # SE = img[y+1,x+1]
            p6 = np.zeros_like(img); p6[:-1, :]  = img[1:, :]    # S  = img[y+1,x]
            p7 = np.zeros_like(img); p7[:-1, 1:] = img[1:, :-1]  # SW = img[y+1,x-1]
            p8 = np.zeros_like(img); p8[:, 1:]   = img[:, :-1]   # W  = img[y,x-1]
            p9 = np.zeros_like(img); p9[1:, 1:]  = img[:-1, :-1] # NW = img[y-1,x-1]
            seq = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(a.astype(np.uint8) for a in seq)
            A = np.zeros(img.shape, dtype=np.int16)
            for i in range(8):
                A += (~seq[i] & seq[(i + 1) % 8])
            cond = img & (B >= 2) & (B <= 6) & (A == 1)
            if step == 1:
                cond &= ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                cond &= ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            # 不删最外圈（与 ximgproc 一致，避免边界误删）
            cond[0, :] = False; cond[-1, :] = False
            cond[:, 0] = False; cond[:, -1] = False
            if cond.any():
                img = img & ~cond
                changed = True
    return (img.astype(np.uint8) * 255)


def prune_skeleton_spurs(skel, max_spur=6, rounds=3):
    """删除骨架上的短毛刺分支。

    细化后的骨架常带 1~6px 的小毛刺，而头尾判定取"最远两点"，
    毛刺末端会被误当成头或尾，导致头尾点跑到虫体外。
    算法：从每个端点（仅 1 个邻居）沿链走到分叉点，链长 <= max_spur
    则整条删除；迭代直到稳定。
    """
    skel = (skel > 0).astype(np.uint8)
    H, W = skel.shape

    def nbrs(y, x):
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                    out.append((ny, nx))
        return out

    for _ in range(rounds):
        changed = False
        ys, xs = np.where(skel)
        for y, x in zip(ys, xs):
            if not skel[y, x]:
                continue
            nb = nbrs(y, x)
            if len(nb) != 1:
                continue
            path = [(y, x)]
            prev, cur = (y, x), nb[0]
            fork = False
            while True:
                nbs2 = [p for p in nbrs(*cur) if p != prev]
                if len(nbs2) == 0:
                    break                  # 走到另一端点：整条骨架，不删
                if len(nbs2) >= 2:
                    fork = True            # 到达分叉点
                    break
                path.append(cur)
                prev, cur = cur, nbs2[0]
                if len(path) > max_spur:
                    break
            if fork and len(path) <= max_spur:
                for py, px in path:
                    skel[py, px] = 0
                changed = True
        if not changed:
            break
    return (skel * 255).astype(np.uint8)


def _points_to_mask(points):
    """把 (N,2) 点阵放回像素网格，供 _skeleton_centerline 建图用。"""
    p = np.round(np.asarray(points, dtype=np.float64)).astype(np.int32)
    H = max(int(p[:, 1].max()) + 3, 3)
    W = max(int(p[:, 0].max()) + 3, 3)
    m = np.zeros((H, W), np.uint8)
    m[p[:, 1], p[:, 0]] = 255
    return m


def order_skeleton_points(points):
    """把无序的骨架点按从头到尾排列。

    策略（2026-10-19 改）：先在像素网格上重建骨架图，用 _skeleton_centerline
    做双端 BFS 取最长路径——头尾即路径两端，沿真实骨架行走：
      · 不会被"跳步截断"砍掉虫尾（旧版 max_jump=6 在断裂处把之后的虫体
        整段丢掉，即用户看到的"识别短了"）；
      · 毛刺/小环末端不会被误当成头或尾。
    网格重建失败时回退旧版贪心最近邻。
    """
    if len(points) < 2:
        return points
    try:
        pts, _cov, _n = _skeleton_centerline(_points_to_mask(points))
        if pts is not None and len(pts) >= 2:
            return smooth_skeleton(pts)
    except Exception:
        pass

    # ---- 回退：旧版贪心最近邻 ----
    max_jump = 6.0
    n = len(points)
    dist = cdist(points, points)

    # 端点：距离最远的两个点
    i, j = np.unravel_index(np.argmax(dist), dist.shape)

    visited = {i}
    ordered = [points[i]]
    current = i

    # 贪心最近邻
    while len(visited) < n:
        dists = dist[current].copy()
        dists[list(visited)] = np.inf
        nearest = np.argmin(dists)
        if dists[nearest] > max_jump:
            break
        ordered.append(points[nearest])
        visited.add(nearest)
        current = nearest

    if len(ordered) < 2:
        return points

    ordered = np.array(ordered)
    ordered = smooth_skeleton(ordered)
    return ordered


def smooth_skeleton(points, max_window=3):
    """对骨架点做移动平均平滑，消除细化锯齿。"""
    n = len(points)
    if n < 30:
        return points

    window = min(max_window, 3)
    if window < 3 or n < window:
        return points

    half = window // 2
    smoothed = np.zeros_like(points, dtype=np.float64)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        smoothed[i] = np.mean(points[start:end], axis=0)
    return smoothed


# ============================================================
# 2. 曲率计算
# ============================================================
def compute_curvature_profile(points, n_samples=100):
    """计算沿归一化弧长的曲率分布 κ(s)。
    返回：(n_samples,) 的曲率数组，s 从 0(头) 到 1(尾)。"""
    if len(points) < 10:
        return np.zeros(n_samples)

    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cumulative[-1]
    if total_len < 1e-6:
        return np.zeros(n_samples)
    s = cumulative / total_len

    # 7点窗口曲率计算
    half_window = 3
    curvatures = []
    s_pts = []

    for i in range(half_window, len(points) - half_window):
        p_back = points[i - half_window]
        p_mid = points[i]
        p_fwd = points[i + half_window]

        v1 = p_mid - p_back
        v2 = p_fwd - p_mid
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 < 1e-6 or n2 < 1e-6:
            curvatures.append(0)
        else:
            curvatures.append(abs(cross) / (n1 * n2))
        s_pts.append(s[i])

    curvatures = np.array(curvatures)
    s_pts = np.array(s_pts)

    if len(s_pts) < 4:
        return np.zeros(n_samples)

    # 【2026-10-22 端点外插修复】s_pts 只覆盖 [3, n-4] 号点（窗口取不满的
    # 端点区域无曲率值）。旧版在 [0,1] 全域求样条值，端点外插 overshoot
    # 造出系统性伪峰（实测：68% 帧 argmax 落在 s=0/s=1 端点，端点 κ 均值
    # 0.27 vs 内部 0.20），既污染 BBI，又让"峰值位置"类判据全部失真。
    # 修复：只在 s_pts 覆盖范围内取值，范围外（两端各约 3%）置 0。
    try:
        tck = splrep(s_pts, curvatures, k=3, s=len(curvatures) * 0.1)
        s_fine = np.linspace(s_pts[0], s_pts[-1], n_samples)
        kappa = np.asarray(splev(s_fine, tck), dtype=np.float64)
        kappa = np.abs(kappa)
    except Exception:
        kappa = np.interp(np.linspace(s_pts[0], s_pts[-1], n_samples),
                          s_pts, curvatures)
        kappa = np.abs(kappa)

    kappa = gaussian_filter1d(kappa, sigma=1.0)
    return kappa


# ============================================================
# 3. 特征提取
# ============================================================
def extract_features(kappa):
    """从曲率分布 κ(s) 中提取判别特征。

    方向无关设计（2026-10-21）：核心特征全部不依赖 s 的正方向——
    bbi/max_kappa/n_high_regions/periodicity 本身对称；新增的
    mid_dominance/peak_midness 只用"与体中点的距离"，把头尾判错的
    影响挡在分类判据之外。anterior/posterior 旧字段保留仅为兼容，
    规则判据已不再使用。
    """
    n = len(kappa)

    bbi = np.mean(kappa)
    max_kappa = np.max(kappa)
    anterior_kappa = np.mean(kappa[:int(n * 0.3)])
    posterior_kappa = np.mean(kappa[int(n * 0.6):])
    kappa_std = np.std(kappa)
    anterior_max = np.max(kappa[:int(n * 0.3)])
    anterior_ratio = anterior_max / (max_kappa + 1e-8)

    # 【新】中段特征（方向无关）：中段 = s∈[0.3, 0.7]
    _i0, _i1 = int(n * 0.3), int(n * 0.7)
    mid_mean = float(np.mean(kappa[_i0:_i1]))
    mid_dominance = mid_mean / (bbi + 1e-8)          # 中段曲率相对全身的集中度
    peak_pos = int(np.argmax(kappa)) / max(n - 1, 1)  # 峰值位置 ∈ [0,1]
    peak_midness = 1.0 - 2.0 * abs(peak_pos - 0.5)    # 1=正中，0=两端（与方向无关）

    # 高曲率连续区域数（带最小宽度过滤）
    threshold = max(0.35, np.mean(kappa) + 0.8 * np.std(kappa))
    high_curvature_mask = kappa > threshold
    min_width = 5
    n_regions = 0
    in_region = False
    region_count = 0
    for h in high_curvature_mask:
        if h:
            if not in_region:
                in_region = True
                region_count = 1
            else:
                region_count += 1
        else:
            if in_region and region_count >= min_width:
                n_regions += 1
            in_region = False
            region_count = 0
    if in_region and region_count >= min_width:
        n_regions += 1

    # 周期性（自相关峰值数）
    if kappa_std > 1e-6:
        autocorr = np.correlate(kappa - np.mean(kappa), kappa - np.mean(kappa),
                                mode='full')
        autocorr = autocorr[n - 1:]
        autocorr = autocorr / (autocorr[0] + 1e-8)
        peaks = 0
        for i in range(1, len(autocorr) - 1):
            if (autocorr[i] > autocorr[i - 1] and
                    autocorr[i] > autocorr[i + 1] and
                    autocorr[i] > 0.3):
                peaks += 1
    else:
        peaks = 0

    return {
        'bbi': bbi,
        'max_kappa': max_kappa,
        'anterior_kappa': anterior_kappa,
        'posterior_kappa': posterior_kappa,
        'mid_mean': mid_mean,
        'mid_dominance': mid_dominance,
        'peak_midness': peak_midness,
        'n_high_regions': n_regions,
        'kappa_std': kappa_std,
        'anterior_ratio': anterior_ratio,
        'periodicity_peaks': peaks,
    }


def detect_self_crossing(points, mask=None, body_radius=None,
                         min_gap=8, min_count=6):
    """判断单虫骨架是否自身交叉（缠绕/自交型）。

    判据：非相邻骨架点（沿虫体索引间隔 >= min_gap）之间出现大量
    距离小于体半径 body_radius 的点对。真正交叉/盘绕时，交叉区
    会有多个非邻近距离 < 体宽的点对；普通弯曲的最近邻点对只来自
    骨架局部连续段，不会被计入。

    参数：
      points      : (N,2) 骨架点序列，ordered along the worm。
      mask        : 二值掩膜（前景=虫体）。提供时自动用距离变换
                    估算体半径（骨架点处到背景的中位距离）。
      body_radius : 体半径（半宽），None 时自动估算；手动给定时
                    优先于 mask 估算。
      min_gap     : 沿骨架索引最小间隔，避免把相邻点误判为交叉。
      min_count   : 近距离点对数 >= min_count 才判定为自交。

    返回：
      (is_crossing, n_crossings, body_radius_used)
    """
    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)
    if n < min_gap * 2 + 2:
        return False, 0, float(body_radius) if body_radius else 0.0

    # ---- 估算体半径 ----
    if body_radius is None:
        if mask is not None and np.count_nonzero(mask) > 0:
            try:
                m = (mask > 0).astype(np.uint8)
                dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
                idx = pts.astype(np.int32)
                h, w = dt.shape
                idx[:, 0] = np.clip(idx[:, 0], 0, w - 1)
                idx[:, 1] = np.clip(idx[:, 1], 0, h - 1)
                radii = dt[idx[:, 1], idx[:, 0]]
                if np.any(radii > 0):
                    r_est = float(np.median(radii[radii > 0]))
                    if r_est >= 1.5:
                        body_radius = r_est
            except Exception:
                pass
        if body_radius is None:
            # 按细长比估算：典型线虫 L/w ≈ 10，半宽 ≈ L/20
            arc_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            body_radius = max(2.5, arc_len * 0.055)
    body_radius = float(body_radius)

    # ---- 非相邻点对的欧氏距离 ----
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    r2 = body_radius * body_radius

    i_idx = np.arange(n)[:, None]
    j_idx = np.arange(n)[None, :]
    valid = (np.abs(i_idx - j_idx) >= min_gap) & (d2 > 1e-6)
    close = (d2 < r2) & valid

    # 对称矩阵，每对无序点对被计算两次
    n_crossings = int(np.count_nonzero(close)) // 2
    is_crossing = n_crossings >= min_count
    return is_crossing, n_crossings, body_radius


# ============================================================
# 4. 规则判断
# ============================================================
# 三类姿态定义（2026-10-23 定稿：四类实证后回归三类）：
#   0 正弦型(Sinusoidal) : 多波浪/中等弯曲，默认兜底类（含 n_regions>=2 的尖锐多峰）；
#   1 Ω 型(Omega)        : 单个深弯——n_high_regions<=1 且 max_kappa>=OMEGA_MAX_KAPPA。
#                          峰位置不限（中段或头颈/尾部）：线虫 Ω 转头峰本就在头颈部，
#                          按峰位置再分一类属"去头尾化"要屏蔽的噪声维度；
#   2 笔直型(Straight)   : 低曲率直形，BBI < STRAIGHT_BBI。
# 判据优先级：笔直 > Ω > 正弦（兜底），互斥。
#
# 【四类→三类的实证过程（论文方法学素材）】
#   原第四类"缠绕型"经三条独立判据验证均不成立（output_routeB_smoke 7151 帧）：
#   ① 骨架自交检测（detect_self_crossing，下方保留实现）：命中仅 4/7151 帧(0.06%)，
#      聚类"缠绕簇"实为"尖锐但峰偏端的单弯"，与 Ω 转头同类；
#   ② 几何紧密度（fill=面积/弧长、e2a=头尾距/弧长）：与正弦/Ω 分布重合；
#   ③ 时序运动（net_disp/path_len）：整批虫 77-93% 帧近乎静止，无区分度。
#   故缠绕类整体取消，其帧按曲率特征并入 Ω 型/正弦型。
STRAIGHT_BBI = 0.16
OMEGA_MAX_KAPPA = 0.35             # 正弦类 p75(0.377)与深弯类 p25(0.369)的自然交界
# 以下常量仅服务于已否决的自交检测（保留供论文方法学引用，分类不再使用）
CROSSING_MIN_GAP = 8
CROSSING_MIN_COUNT = 6


def classify_by_rules(features):
    """用规则判断姿态类别(0-2)。返回：(label, confidence, reason)

    三类判据（2026-10-23 定稿）：
      笔直型(2)  BBI < STRAIGHT_BBI(0.16)——与其他类曲率量级差最大，最清晰；
      Ω 型(1)    单个深弯：n_high_regions <= 1 且
                max_kappa >= OMEGA_MAX_KAPPA(0.35)。
                峰位置不限；用绝对 max_kappa 而非相对突出度（BBI 分布密集，
                相对阈值在连续谱上误捕严重，2026-10-22 实测分离比仅 0.67）；
      正弦型(0)  多波浪/中等弯曲，兜底类。

    阈值标定（output_routeB_smoke 7151 帧 + diagnose_cluster_sep）：
      max_kappa=0.35 位于正弦类 p75(0.377) 与深弯类 p25(0.369) 交界。
    Ω 型与正弦型本质位于"曲率连续谱"两端，任何单帧阈值都是操作性定义；
    论文需如实说明该局限，并建议结合视频/时序核验。
    """
    bbi = features['bbi']
    max_kappa = features['max_kappa']
    n_regions = features['n_high_regions']
    periodicity = features['periodicity_peaks']

    # 1. 笔直型（直形）
    if bbi < STRAIGHT_BBI:
        return 2, 0.85, f'低曲率(BBI={bbi:.3f})，判定为笔直型'

    # 2. Ω 型：单个深弯（峰位置不限）
    if n_regions <= 1 and max_kappa >= OMEGA_MAX_KAPPA:
        conf = min(0.85, 0.70 + 0.15 * min((max_kappa - OMEGA_MAX_KAPPA) / 0.30, 1.0))
        return 1, conf, (f'单峰深弯(max_kappa={max_kappa:.2f}>='
                         f'{OMEGA_MAX_KAPPA}, n_regions={n_regions})，判定为Ω型')

    # 3. 正弦型：多波浪，兜底
    conf = 0.65 + 0.05 * min(periodicity, 3)
    return 0, min(conf, 0.80), (f'多波浪/中等弯曲(bbi={bbi:.3f}, '
                                f'max_kappa={max_kappa:.2f})，判定为正弦型')


# ============================================================
# 5. 可视化
# ============================================================
def visualize(image, skeleton, kappa, label, confidence, reason, save_path,
              mask_raw=None, mask_final=None, info='', field_type='unknown'):
    """生成可视化对比图：提取流程的中间结果 + 骨架叠加 + 曲率曲线。

    面板（当提供 mask_raw / mask_final 时为 5 面板）：
        [1] 原图              —— 原始单只线虫截图
        [2] 顶帽/黑帽阈值分割 —— 自动判定明暗场，明场用黑帽，暗场用顶帽
        [3] 最大连通域        —— 看是否被背景blob抢走，或虫体是否被背景淹没
        [4] 骨架叠加          —— 红线=骨架，绿点=头，蓝点=尾
        [5] 曲率曲线 κ(s)     —— 用于核对弯曲行为的判定依据

    若未提供掩膜则退化为 3 面板（原图 / 骨架 / 曲率）。
    """
    label_names = {0: 'Sinusoidal', 1: 'Omega', 2: 'Straight', -1: 'Failed'}
    label_colors = {0: '#2196F3', 1: '#4CAF50', 2: '#F44336', -1: '#607D8B'}
    ft_text = {'brightfield': 'Black-hat', 'darkfield': 'Top-hat'}.get(
        field_type, 'Morph')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    full = mask_raw is not None and mask_final is not None
    n_panels = 5 if full else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels + 1.2, 4.4))

    # [1] 原图
    axes[0].imshow(gray, cmap='gray', interpolation='nearest')
    axes[0].set_title('1. Original', fontsize=10)
    axes[0].axis('off')

    if full:
        # [2] 顶帽/黑帽阈值分割
        axes[1].imshow(mask_raw, cmap='gray', interpolation='nearest')
        axes[1].set_title(f'2. {ft_text} thresh\n({int((mask_raw > 0).sum())} px)',
                          fontsize=9)
        axes[1].axis('off')

        # [3] 最大连通域
        axes[2].imshow(mask_final, cmap='gray', interpolation='nearest')
        axes[2].set_title(f'3. Largest blob\n({int((mask_final > 0).sum())} px)',
                          fontsize=10)
        axes[2].axis('off')

        # [4] 骨架叠加
        ax = axes[3]
        ax.imshow(gray, cmap='gray', interpolation='nearest')
        if skeleton is not None and len(skeleton) > 1:
            ax.plot(skeleton[:, 0], skeleton[:, 1], 'r-', linewidth=1.5)
            ax.plot(skeleton[0, 0], skeleton[0, 1], 'go', markersize=6, label='Head')
            ax.plot(skeleton[-1, 0], skeleton[-1, 1], 'bo', markersize=6, label='Tail')
            ax.legend(fontsize=7, loc='upper right')
        n_pts = 0 if skeleton is None else len(skeleton)
        ax.set_title(f'4. Skeleton\n({n_pts} pts)', fontsize=10)
        ax.axis('off')

        # [5] 曲率曲线
        ax = axes[4]
    else:
        # 退化模式：骨架叠加放中间
        ax = axes[1]
        ax.imshow(gray, cmap='gray', interpolation='nearest')
        if skeleton is not None and len(skeleton) > 1:
            ax.plot(skeleton[:, 0], skeleton[:, 1], 'r-', linewidth=1.5)
            ax.plot(skeleton[0, 0], skeleton[0, 1], 'go', markersize=6, label='Head')
            ax.plot(skeleton[-1, 0], skeleton[-1, 1], 'bo', markersize=6, label='Tail')
            ax.legend(fontsize=7, loc='upper right')
        ax.set_title('Skeleton', fontsize=10)
        ax.axis('off')
        ax = axes[2]

    s = np.linspace(0, 1, len(kappa))
    ax.plot(s, kappa, color=label_colors[label], linewidth=2)
    ax.fill_between(s, 0, kappa, alpha=0.2, color=label_colors[label])
    ax.set_xlabel('Normalized Arc Length s', fontsize=9)
    ax.set_ylabel('Curvature k(s)', fontsize=9)
    ax.set_title('5. Curvature Profile' if full else 'Curvature', fontsize=10)
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='thresh')
    ax.tick_params(labelsize=8)

    head = f'[{label}] {label_names.get(label, label)}  (confidence: {confidence:.0%})'
    ft_info = f'field={field_type} ({ft_text})'
    if info:
        head = f'{info}   |   ' + head + '   |   ' + ft_info
    else:
        head += '   |   ' + ft_info
    fig.suptitle(f'{head}\n{reason}',
                 fontsize=11, fontweight='bold',
                 color=label_colors.get(label, '#333333'))

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()


def visualize_failure(image, mask_raw, reason, save_path, info='',
                      field_type='unknown'):
    """提取失败样本的诊断图：原图 + 顶帽/黑帽阈值结果 + 失败原因。
    用来判断是该调『核大小』还是调『阈值』还是调『最小面积』。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    n_px = 0 if mask_raw is None else int((mask_raw > 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2))
    axes[0].imshow(gray, cmap='gray', interpolation='nearest')
    axes[0].set_title('1. Original', fontsize=10)
    axes[0].axis('off')

    ft_text = {'brightfield': 'Black-hat', 'darkfield': 'Top-hat'}.get(
        field_type, 'Morph')
    axes[1].imshow(mask_raw if mask_raw is not None else gray, cmap='gray',
                   interpolation='nearest')
    axes[1].set_title(f'2. {ft_text} thresh ({n_px} px)', fontsize=10)
    axes[1].axis('off')

    head = info if info else 'Extraction Failed'
    tail = f'field={field_type} ({ft_text})   |   reason: {reason}'
    fig.suptitle(f'{head}\n{tail}',
                 fontsize=11, fontweight='bold', color='#F44336')
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()


# ============================================================
# 6. 并行处理 Worker
# ============================================================
def process_image(task):
    """单张图片处理函数（在子进程中运行）。

    task = (rel_path, abs_path) 或 (rel_path, abs_path, thr_override)
    返回字典，包含 5 维列 + 标签 + 特征。
    """
    rel_path, abs_path = task[0], task[1]
    thr_override = task[2] if len(task) > 2 else None

    try:
        group, date, row, worm, frame = parse_five_dims(rel_path)
    except Exception:
        group, date, row, worm, frame = 'unknown', 'unknown', 'unknown', rel_path, 0

    base_result = {
        'group': group,
        'date': date,
        'row': row,
        'worm': worm,
        'frame': frame,
        'filename': rel_path,
        'label': -1,
        'confidence': 0.0,
        'reason': '',
        'bbi': 0.0,
        'max_kappa': 0.0,
        'anterior_kappa': 0.0,
        'posterior_kappa': 0.0,
        'n_high_regions': 0,
        'kappa_std': 0.0,
        'anterior_ratio': 0.0,
        'periodicity_peaks': 0,
        'skeleton_n_points': 0,
    }

    try:
        image = cv2.imread(abs_path, cv2.IMREAD_COLOR)
        if image is None:
            base_result['reason'] = '读取失败'
            return base_result

        skeleton, _mth, _mfin, _ft, dbg = extract_skeleton_full(
            image, return_debug=True, thr_override=thr_override)
        if skeleton is None or len(skeleton) < 10:
            # 带上精确失败原因（dbg），labels.csv 与 _failed 可视化可直接定位
            base_result['reason'] = '未检测到线虫: %s' % dbg.get('reason', '?')
            return base_result

        kappa = compute_curvature_profile(skeleton)
        features = extract_features(kappa)
        label, confidence, reason = classify_by_rules(features)

        base_result.update({
            'label': label,
            'confidence': confidence,
            'reason': reason,
            'bbi': round(features['bbi'], 6),
            'max_kappa': round(features['max_kappa'], 6),
            'anterior_kappa': round(features['anterior_kappa'], 6),
            'posterior_kappa': round(features['posterior_kappa'], 6),
            'n_high_regions': features['n_high_regions'],
            'kappa_std': round(features['kappa_std'], 6),
            'anterior_ratio': round(features['anterior_ratio'], 6),
            'periodicity_peaks': features['periodicity_peaks'],
            'skeleton_n_points': len(skeleton),
        })
        return base_result

    except Exception as e:
        base_result['reason'] = f'处理异常: {str(e)}'
        return base_result


# ============================================================
# 7. 主程序
# ============================================================
CSV_COLUMNS = [
    'group', 'date', 'row', 'worm', 'frame', 'filename',
    'label', 'confidence', 'reason',
    'bbi', 'max_kappa', 'anterior_kappa', 'posterior_kappa',
    'n_high_regions', 'kappa_std', 'anterior_ratio',
    'periodicity_peaks', 'skeleton_n_points',
]


def collect_stats(rows):
    """统计标签分布。"""
    label_names = {0: '正弦型', 1: 'Ω 型', 2: '笔直型'}
    counts = defaultdict(int)
    errors = 0
    for r in rows:
        lbl = r['label']
        if lbl == -1:
            errors += 1
        else:
            counts[lbl] += 1
    return counts, errors


BAR_WIDTH = 28


def make_bar(cur, total, width=BAR_WIDTH):
    """生成文本进度条，并用 \r 回到行首实现原地刷新。"""
    pct = cur / total if total > 0 else 1.0
    filled = int(pct * width)
    filled = max(0, min(width, filled))
    return '#' * filled + '-' * (width - filled), pct


_LAST_LINE_LEN = [0]
_LIVE_MODE = None  # 惰性判定：是否用 \r 原地刷新


def use_live(force_off=False):
    """是否用 \\r 原地刷新进度。

    - 真实终端(CMD/PowerShell/Bash)：是 -> 单行不断刷新
    - 输出重定向到管道/文件：否 -> 每行独立输出（避免 \\r 变成连成一片的乱码）
    """
    global _LIVE_MODE
    # force_off 一旦为真就写回全局，否则扫描阶段先调用本函数把模式锁死后，
    # 后面再传 force_off 就失效了（--no-live 会形同虚设）
    if force_off:
        _LIVE_MODE = False
        return False
    if _LIVE_MODE is None:
        try:
            _LIVE_MODE = sys.stdout.isatty()
        except Exception:
            _LIVE_MODE = False
    return _LIVE_MODE


def emit_progress(prefix, cur, total, rate, eta_sec, extra='', bar=True,
                  width=BAR_WIDTH, unit='张/秒'):
    """输出一行进度。

    参数：
        bar: True=带百分比进度条（知道总数时用）
             False=只显示累计计数（扫描阶段没有总数分母，画百分比会误导）
    返回是否到达 100%。
    """
    if bar:
        bar_str, pct = make_bar(cur, total, width)
        head = f'{prefix} [{bar_str}] {pct * 100:5.1f}% | {cur:,}/{total:,}'
    else:
        head = f'{prefix} {cur:,} 张'
    line = f'{head} | {rate:,.0f} {unit}'
    if bar:
        eta_str = time.strftime('%H:%M:%S', time.gmtime(max(0, eta_sec)))
        line += f' | ETA {eta_str}'
    if extra:
        line += f' | {extra}'

    if use_live():
        pad = max(0, _LAST_LINE_LEN[0] - len(line))
        sys.stdout.write('\r' + line + ' ' * pad)
        _LAST_LINE_LEN[0] = len(line)
    else:
        sys.stdout.write(line + '\n')
    sys.stdout.flush()
    return bar and cur >= total


def break_line():
    """结束当前刷新行，另起一行并把已用长度清零。"""
    if use_live() and _LAST_LINE_LEN[0] > 0:
        sys.stdout.write('\n')
        sys.stdout.flush()
    _LAST_LINE_LEN[0] = 0


def reservoir_add(sample, item, seen, capacity):
    """蓄水池采样：第 seen 个候选以 capacity/seen 的概率进入容量为 capacity 的样本池。"""
    if capacity <= 0:
        return
    if len(sample) < capacity:
        sample.append(item)
    else:
        j = random.randint(1, seen)
        if j <= capacity:
            sample[j - 1] = item


def dim_label(g, d, rw, wm, fm):
    """把五维字段拼成可视化标题用的短标签。"""
    return f'{g}/{d}/{rw}/{wm}_{fm:03d}'


def main():
    parser = argparse.ArgumentParser(description='线虫弯曲行为预标注工具（五维数据版）')
    parser.add_argument('--input', '-i', required=True,
                        help='输入图片根目录（如 D:/data/image）')
    parser.add_argument('--output', '-o', default='labels.csv',
                        help='输出 CSV 文件路径')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='并行进程数（默认：CPU 核心数）')
    parser.add_argument('--chunksize', '-c', type=int, default=200,
                        help='每个子任务批次大小（默认 200）')
    parser.add_argument('--resume', action='store_true',
                        help='断点续跑：跳过已输出 CSV 中存在的文件')
    # ---- 可视化相关：必须指定数量才会触发，viz-dir 只是输出目录 ----
    parser.add_argument('--viz', '-v', type=int, default=0,
                        help='随机可视化 N 张成功分类的图片（默认 0=不生成）')
    parser.add_argument('--viz-per-class', type=int, default=0,
                        help='每一类各可视化 N 张（0=不启用，稀有类别推荐用这个）')
    parser.add_argument('--viz-failed', type=int, default=0,
                        help='可视化 N 张提取失败的样本，用于诊断调参（默认 0）')
    parser.add_argument('--viz-dir', default='worms_viz',
                        help='可视化图片输出目录（默认 worms_viz/）')
    parser.add_argument('--max-images', type=int, default=None,
                        help='仅处理前 N 张图片（用于快速测试）')
    parser.add_argument('--progress-interval', type=int, default=0,
                        help='每处理 N 张刷新一次进度（0=自动，约 10 次刷新）')
    parser.add_argument('--no-live', action='store_true',
                        help='进度逐行输出而非原地刷新（重定向日志时用）')
    parser.add_argument('--exclude-list', '-x', action='append', default=None,
                        metavar='FILE',
                        help='虫体清单文件：清单里的虫直接跳过不读（可多次指定）。'
                             '支持 TXT/JSON，每行 control/0211/D/worm1')
    parser.add_argument('--only-list', action='append', default=None,
                        metavar='FILE',
                        help='与 --exclude-list 相反：只处理清单里的虫体（可多次指定）')
    parser.add_argument('--threshold-override', action='append', default=None,
                        metavar='FILE',
                        help='阈值覆盖清单（可多次指定）。每行 "<路径> <阈值>"，'
                             '支持帧(group/date/row/wormN_MMM)/整虫(wormN)/'
                             '整行(group/date/row)三种粒度；点名帧强制用该阈值的'
                             '候选，用于"干净候选太短/脏候选太糙"的人工裁决。'
                             '不绕过两虫交叉过滤。例：'
                             '--threshold-override thr_override.txt')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_csv = args.output
    n_viz = args.viz
    n_viz_class = args.viz_per_class
    n_viz_fail = args.viz_failed
    need_viz = n_viz > 0 or n_viz_class > 0 or n_viz_fail > 0
    viz_dir = args.viz_dir if need_viz else None

    if not os.path.isdir(input_dir):
        print(f'错误：输入目录不存在: {input_dir}')
        return

    # ---- 虫体清单过滤 ----
    exclude_files = args.exclude_list or []
    only_files = args.only_list or []
    if exclude_files and only_files:
        print('错误：--exclude-list 与 --only-list 不能同时使用')
        return

    filter_mode = 'exclude' if exclude_files else ('only' if only_files else None)
    filter_files = None
    n_rule_worms = 0
    n_rule_dirs = 0
    if filter_mode:
        list_files = exclude_files if exclude_files else only_files
        dir_rules, worm_rules, n_rule_worms = set(), defaultdict(set), 0
        for lf in list_files:
            try:
                d, w, n = load_worm_list(lf)
            except Exception as e:
                print(f'错误：读取清单失败 {lf} -> {e}')
                return
            dir_rules |= d
            for k, v in w.items():
                worm_rules[k] |= v
            n_rule_worms += n
        n_rule_dirs = len(dir_rules)

        # 若 -i 指向的不是 data 根目录（例如直接指向 control/0211/D），
        # 从路径里识别出已包含的层级，保证相对路径仍能匹配规则
        base_parts = []
        groups = {k[0] for k in worm_rules} | {d[0] for d in dir_rules if d}
        segs = [p.lower() for p in os.path.abspath(input_dir).replace('\\', '/').split('/')
                if p and not p.endswith(':')]
        for i, s in enumerate(segs):
            if s in groups:
                base_parts = segs[i:]
                break

        filter_files = build_worm_filter(dir_rules, worm_rules, mode=filter_mode,
                                         base_parts=base_parts)
        if base_parts:
            print(f'  输入根目录已含层级: {" / ".join(base_parts)}')
        verb = '排除' if filter_mode == 'exclude' else '仅保留'
        print(f'虫体清单({verb}): {len(list_files)} 个文件，'
              f'{n_rule_worms} 只虫，{len(worm_rules)} 个行目录，'
              f'{n_rule_dirs} 条目录级规则')
        for lf in list_files:
            print(f'  - {lf}')

    # 只在扫描阶段统计一次被过滤掉的数量（处理阶段还会再过滤，但不重复计数）
    filter_stat = {'counting': True, 'dropped': 0}

    # ---- 阈值覆盖清单 ----
    thr_resolve = None
    n_thr_rules = 0
    if args.threshold_override:
        fr_all, wr_all, dr_all = {}, {}, {}
        try:
            for lf in args.threshold_override:
                fr, wr, dr, n = load_threshold_override(lf)
                fr_all.update(fr)
                wr_all.update(wr)
                dr_all.update(dr)
                n_thr_rules += n
        except Exception as e:
            print(f'错误：读取阈值覆盖清单失败 -> {e}')
            return
        if n_thr_rules:
            thr_resolve = build_thr_resolver(fr_all, wr_all, dr_all)
            print(f'阈值覆盖: {len(args.threshold_override)} 个文件，'
                  f'{n_thr_rules} 条规则（帧级 {len(fr_all)} / '
                  f'整虫 {len(wr_all)} / 目录级 {len(dr_all)}）')
            for lf in args.threshold_override:
                print(f'  - {lf}')
        else:
            print('阈值覆盖: 清单为空，忽略')

    # 在任何进度输出之前先锁定刷新模式，否则扫描阶段的 emit_progress 会抢先初始化
    use_live(force_off=args.no_live)

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # 断点续跑：读取已处理文件
    done_paths = set()
    resume_mode = args.resume and os.path.exists(output_csv)
    if resume_mode:
        print(f'正在读取已有输出以断点续跑: {output_csv}')
        with open(output_csv, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                filename_idx = header.index('filename')
            except (StopIteration, ValueError):
                filename_idx = 5  # 默认列位置
            for row in reader:
                if len(row) > filename_idx:
                    done_paths.add(row[filename_idx])
        print(f'已处理 {len(done_paths)} 张，将跳过')

    # 递归遍历生成器
    def walk_images():
        for root, dirs, files in os.walk(input_dir):
            dirs.sort()
            files.sort()
            leaf = os.path.relpath(root, input_dir).replace('\\', '/')
            files = [f for f in files if f.lower().endswith(exts)]
            if not files:
                continue
            if filter_files is not None:
                files, dropped = filter_files(leaf, files)
                if filter_stat['counting']:
                    filter_stat['dropped'] += dropped
                if not files:
                    continue
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, input_dir).replace('\\', '/')
                yield rel_path, abs_path, leaf

    def task_generator():
        for rel_path, abs_path, _ in walk_images():
            if rel_path in done_paths:
                continue
            yield (rel_path, abs_path,
                   thr_resolve(rel_path) if thr_resolve else None)

    # ---- 扫描 ----
    # 全量模式：只计数不存路径（795 万个路径放内存要几百 MB）
    # 测试模式(--max-images)：收集够数立即停止遍历，避免白等几分钟全盘扫描
    sys.stdout.write('正在扫描图片...\n')
    sys.stdout.flush()
    total_images = 0
    pending_images = 0
    limited_tasks = None
    n_thr_hit = 0
    scan_t0 = time.time()
    scan_step = 20000
    next_report = scan_step
    last_report = scan_t0

    limit = args.max_images
    if limit:
        limited_tasks = []

    for rel_path, abs_path, leaf in walk_images():
        total_images += 1
        if rel_path not in done_paths:
            pending_images += 1
            if thr_resolve is not None and thr_resolve(rel_path) is not None:
                n_thr_hit += 1
            if limited_tasks is not None:
                limited_tasks.append(
                    (rel_path, abs_path,
                     thr_resolve(rel_path) if thr_resolve else None))

        now = time.time()
        if total_images >= next_report and now - last_report >= 0.3:
            elapsed = now - scan_t0
            emit_progress('扫描', total_images, 0, total_images / elapsed, 0,
                          f'当前 {leaf}', bar=False)
            last_report = now
            next_report = total_images + scan_step

        # 测试模式：待处理数量够了就停止遍历
        if limited_tasks is not None and len(limited_tasks) >= limit:
            break

    if limited_tasks is not None:
        limited_tasks = limited_tasks[:limit]
        pending_images = len(limited_tasks)

    elapsed_scan = time.time() - scan_t0
    emit_progress('扫描', total_images, 0, total_images / max(1e-6, elapsed_scan),
                  0, '完成', bar=False)
    break_line()
    if thr_resolve is not None:
        print(f'阈值覆盖命中: {n_thr_hit} 张待处理图片'
              f'（{"全部" if n_thr_hit == pending_images else "部分"}）')

    filter_stat['counting'] = False

    print(f'输入目录: {input_dir}')
    if filter_files is not None:
        verb = '排除' if filter_mode == 'exclude' else '未列入清单而跳过'
        print(f'清单过滤: 跳过 {filter_stat["dropped"]:,} 张'
              f'（{verb}，规则 {n_rule_worms} 只虫 / '
              f'{n_rule_dirs} 条目录级规则）')
    print(f'已扫描图片: {total_images:,}'
          + ('' if not limit else f'（达到上限提前停止）'))
    print(f'待处理: {pending_images:,}')
    print(f'扫描耗时: {elapsed_scan:.1f} 秒')
    print(f'输出文件: {output_csv}')
    if need_viz:
        plan = []
        if n_viz:
            plan.append(f'随机 {n_viz} 张')
        if n_viz_class:
            plan.append(f'每类 {n_viz_class} 张')
        if n_viz_fail:
            plan.append(f'失败诊断 {n_viz_fail} 张')
        print(f'可视化: {" + ".join(plan)} -> {viz_dir}')
    print('-' * 60)

    if pending_images == 0:
        print('没有需要处理的图片。')
        return

    # 准备输出文件
    file_exists = os.path.exists(output_csv)
    if file_exists and not resume_mode:
        # 非续跑模式覆盖
        open(output_csv, 'w', encoding='utf-8-sig', newline='').close()

    # 进程数
    workers = args.workers if args.workers else max(1, os.cpu_count())
    workers = min(workers, pending_images)
    print(f'使用 {workers} 个并行进程')

    # 创建可视化目录
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)

    # 处理与实时写入
    processed = 0
    written = 0
    errors = 0
    label_counts = defaultdict(int)
    viz_sample = []                          # 全局随机样本
    viz_by_class = defaultdict(list)         # 每类样本
    viz_fail_sample = []                     # 失败样本
    class_seen = defaultdict(int)
    success_for_viz = 0
    fail_seen = 0

    # 进度刷新步长：默认把全程切成约 10 段，最多每 0.3 秒刷一次
    if args.progress_interval > 0:
        step = max(1, args.progress_interval)
    else:
        step = max(100, pending_images // 10)
    last_print = 0.0
    last_decile = -1

    start_time = time.time()
    use_live(force_off=args.no_live)
    print('开始处理' + ('（进度每 10% 换行留档，其余原地刷新同一行）'
                     if use_live() else '（逐行输出进度）'))
    sys.stdout.flush()

    FLUSH_EVERY = 1000  # CSV 写盘节流（795 万张时逐行 flush 太慢）

    with open(output_csv, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists or not resume_mode:
            writer.writeheader()
            f.flush()

        pool = mp.Pool(processes=workers)
        try:
            # 测试模式用已收集的任务列表，全量模式用生成器（避免占内存）
            if limited_tasks is not None:
                task_iter = iter(limited_tasks)
            else:
                task_iter = task_generator()

            for result in pool.imap_unordered(process_image, task_iter,
                                              chunksize=args.chunksize):
                writer.writerow(result)
                if processed % FLUSH_EVERY == 0:
                    f.flush()
                processed += 1
                written += 1

                if result['label'] == -1:
                    errors += 1
                    fail_seen += 1
                    if n_viz_fail > 0:
                        reservoir_add(viz_fail_sample, result, fail_seen,
                                      n_viz_fail)
                else:
                    label_counts[result['label']] += 1
                    # 可视化蓄水池采样
                    if n_viz > 0:
                        success_for_viz += 1
                        reservoir_add(viz_sample, result, success_for_viz, n_viz)
                    if n_viz_class > 0:
                        cls = result['label']
                        class_seen[cls] += 1
                        reservoir_add(viz_by_class[cls], result,
                                      class_seen[cls], n_viz_class)

                # 进度报告：就地刷新同一行，每 10% 换行留档
                now = time.time()
                due = (processed % step == 0 or processed == pending_images)
                if due and (now - last_print >= 0.3 or processed == pending_images):
                    elapsed = now - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = pending_images - processed
                    eta_sec = remaining / rate if rate > 0 else 0
                    err_pct = errors / processed * 100 if processed else 0
                    extra = (f'失败 {errors:,} ({err_pct:.1f}%) | '
                             f'{result["filename"]}')
                    done = emit_progress('处理', processed, pending_images,
                                         rate, eta_sec, extra)
                    last_print = now
                    decile = int(processed / pending_images * 10) \
                        if pending_images else 10
                    if done or decile > last_decile:
                        break_line()
                        last_decile = decile

            f.flush()
        finally:
            pool.close()
            pool.join()

    elapsed_total = time.time() - start_time
    print('-' * 60)
    print(f'完成！本次处理 {processed} 张，写入 {written} 行')
    print(f'总耗时: {elapsed_total / 60:.2f} 分钟，平均 {processed / elapsed_total:.1f} 张/秒'
          if elapsed_total > 0 else '总耗时: 0 秒')

    # 标签统计
    print('\n标签分布统计:')
    label_names = {0: '正弦型', 1: 'Ω 型', 2: '笔直型'}
    for lbl in sorted(label_counts.keys()):
        print(f'  [{lbl}] {label_names[lbl]}: {label_counts[lbl]} 张')
    if errors:
        print(f'  [-1] 未检测到/读取失败: {errors} 张')

    # 生成可视化
    def strip_ext(name):
        return os.path.splitext(name)[0]

    def safe_name(rel_path):
        return rel_path.replace('/', '_').replace('\\', '_')

    # 汇总所有需要可视化的样本
    cls_short = {0: '0_sinusoid', 1: '1_omega', 2: '2_coil', 3: '3_reversal',
                 4: '4_resting'}
    viz_targets = []
    if viz_sample:
        # 随机样本也按预测类别归档，避免全部堆在根目录
        rand_by_class = defaultdict(list)
        for r in viz_sample:
            rand_by_class[r['label']].append(r)
        for cls in sorted(rand_by_class.keys()):
            short = cls_short.get(cls, str(cls))
            viz_targets.append((f'随机样本_{short}',
                                rand_by_class[cls],
                                os.path.join(viz_dir, '_random', short)))
    for cls in sorted(viz_by_class.keys()):
        short = cls_short.get(cls, str(cls))
        viz_targets.append((f'类别{short}',
                            viz_by_class[cls],
                            os.path.join(viz_dir, short)))
    if viz_fail_sample:
        viz_targets.append(('失败诊断', viz_fail_sample,
                            os.path.join(viz_dir, '_failed')))

    for tag, samples, out_dir in viz_targets:
        print(f'\n生成可视化 [{tag}] {len(samples)} 张 -> {out_dir}')
        for i, r in enumerate(samples, 1):
            abs_path = os.path.join(input_dir, r['filename'])
            image = cv2.imread(abs_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            skeleton, mask_raw, mask_final, field_type = extract_skeleton_full(
                image,
                thr_override=thr_resolve(r['filename']) if thr_resolve else None)
            info = dim_label(r['group'], r['date'], r['row'], r['worm'],
                             r['frame'])
            save_path = os.path.join(out_dir, f'{safe_name(strip_ext(r["filename"]))}.png')

            if tag == '失败诊断':
                visualize_failure(image, mask_raw, r['reason'], save_path,
                                  info=info, field_type=field_type)
                continue

            if skeleton is None or len(skeleton) < 10:
                visualize_failure(image, mask_raw, r['reason'] or '骨架过短',
                                  save_path, info=info, field_type=field_type)
                continue
            kappa = compute_curvature_profile(skeleton)
            visualize(image, skeleton, kappa, r['label'], r['confidence'],
                      r['reason'], save_path,
                      mask_raw=mask_raw, mask_final=mask_final, info=info,
                      field_type=field_type)
        print(f'  [{tag}] 完成')

    if viz_targets:
        print(f'\n可视化根目录: {viz_dir}')
        print(f'  目录结构:')
        print(f'    {viz_dir}/<类别子目录>/<组_日期_行_虫_帧>.png')
        print(f'    {viz_dir}/_random/<类别子目录>/<组_日期_行_虫_帧>.png')
        print(f'    {viz_dir}/_failed/<组_日期_行_虫_帧>.png')
        print('  每个图含 5 个面板: 原图 | 顶帽/黑帽阈值 | 最大连通域 | 骨架叠加 | 曲率曲线')
        print('  明场(浅底深虫)自动使用黑帽变换，暗场(暗底亮虫)自动使用顶帽变换')


if __name__ == '__main__':
    main()
