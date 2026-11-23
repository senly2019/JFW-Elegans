"""
模拟标注（多方法集成，替代人工标注）—— 三类姿态版
=====================================================
由于无法逐帧人工标注，本脚本用「三个独立判别信号 + 集成共识」生成伪标签，
尽量逼近人工标注的准确率。核心思想：**单一信号判不准的类别，用另一类
信号互补修正**。

三个姿态类别（2026-10-23 定稿；原四类中的"缠绕型"经三重实证取消，见下）：
  0 正弦型 Sinusoidal  —— 多波浪/中等弯曲（兜底类）
  1 Ω 型    Omega       —— 单个深弯：n_high_regions<=1 且 max_kappa>=0.35
                           （峰位置不限：中段或头颈/尾部）
  2 笔直型  Straight    —— 直形（低曲率 BBI<0.16）。死虫已由 z_tracking_only.py
                           依据人标注 DeadBoxes 在上游过滤，故直形即存活，
                           原"静止"更名为"笔直"以强调这是纯形状定义，
                           不含任何运动学判断。

已取消的类别：
  · 倒退：单帧形状与正弦完全相同，唯一判别途径是跨帧运动方向，而运动方向
    依赖"头尾哪端是头"——头尾单帧判别不可靠（表1 实测对齐率仅 51%~73%），
    方向判反的虫运动信号全部反转，伪标签被严重污染。故取消，判据全部去头尾化。
  · 缠绕（2026-10-23 取消）：原判据为骨架自交检测（路线 B），output_routeB_smoke
    7151 帧三重实证均不成立——①自交检测命中仅 4 帧(0.06%)，聚类"缠绕簇"实为
    "尖锐但峰偏端的单弯"，与 Ω 转头同类；②几何紧密度（fill/e2a/af）与正弦/Ω
    分布重合；③时序运动无区分度（整批虫 77-93% 帧近乎静止）。其帧按曲率特征
    并入 Ω 型/正弦型。详见 worm_labeler.py 第 4 节注释。

三个独立信号：
  1. 形状信号  shape   —— 单帧曲率规则 classify_by_rules（静态形状，
                          全部方向无关：BBI/高曲率区域数/中段峰/周期性）
  2. 结构信号  cluster —— 无监督聚类：κ(s) 经 PCA 后 GMM 聚 3 类，
                           按簇的曲率统计量自动映射到行为类（不依赖规则）
  3. 头尾信号  temporal —— 时序运动学：线虫头先动、尾跟随，约 15 帧
                           间隔的质心位移方向即头的朝向（移动帧直接
                           判定为锚点）；未移动帧由端点近邻链式传播
                           （同虫头尾位置不突变）承接锚点结果。

笔直判据说明：死虫已在视频截图阶段（z_tracking_only.py，人标注 DeadBoxes
IoU 过滤）剔除，因此本管线内"直形(低 BBI)"可直接判为笔直（原静止）。
STRAIGHT_BBI 为直形阈值（原 REST_BBI，经全量数据校准，BBI 噪声底约 0.12，
见 README）。

头尾定向（2026-10-22 重建，见 worm_head_orientation）：
  时序质心位移锚点 + 端点近邻链式传播（同虫一致性硬约束）+ 质心启发式
  兜底（零运动虫）。宽度轮廓法（依赖不可靠的二值化宽度）与曲率方差加权
  （依据咽泵活动、拍摄分辨率拍不到）已弃用。
  头尾方向用于输出 head_is_first 列（逐帧）并供 step8/step9 做 κ(s)
  方向归一化（fPCA 特征空间对齐）；三类判据本身不依赖它——判错只影响
  特征对齐质量，不再污染标签。

用法：
  # 全量数据（不限制虫数）：--max-worms 设一个极大值即可
  python simulate_labels.py -i $WORM_DATA_ROOT -o output/simulated_labels.csv \
      --workers 16

  # 小批量测试
  python simulate_labels.py -i $WORM_DATA_ROOT -o simulated_labels.csv --max-worms 100 --workers 8
"""

import os
import sys
import csv
import re
import argparse
import time
import numpy as np
import cv2
from collections import defaultdict, deque
from multiprocessing import Pool, cpu_count

from worm_labeler import (extract_skeleton_full, compute_curvature_profile,
                          extract_features, classify_by_rules, STRAIGHT_BBI)
from head_tail_discriminator import centroid_heuristic
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# 三类姿态（2026-10-23 定稿：缠绕类经三重实证取消，见 worm_labeler 注释）
LABEL_CN = {0: '正弦型', 1: 'Ω 型', 2: '笔直型'}

# 笔直（直形）阈值已由 worm_labeler 导出（STRAIGHT_BBI=0.16），
# 与 classify_by_rules 保持一致。BBI = mean(|sinΔθ|) 为体长归一化平均曲率；
# 直形虫 BBI 低。死虫已在上游过滤，故 BBI < STRAIGHT_BBI 即判为笔直。
# 校准依据：全量数据 BBI 5th≈0.155，噪声底约 0.12（2026-10 实测）。


# ============================================================
# 1. 头尾定向（2026-10-22 重建：时序为主、单帧形状为辅）
# ============================================================
# 弃用说明（用户 2026-10-22 评审）：
#   · 宽度轮廓法——宽度测量依赖二值化掩膜质量，本数据二值化噪声大，
#     头尾两端的宽度差无从谈可靠性，弃用；
#   · 曲率方差加权（原论文方法）——依据是"头部咽泵活动导致曲率变异大"，
#     但拍摄分辨率根本拍不到咽泵，判据无从考证，弃用；
#   · 质心启发式（头离身体质心更远）——方向性大体正确但误差大，
#     仅保留作零运动虫的兜底，不作决定性依据。
#
# 有效证据：
#   · 时序运动学（主）：头先行、尾跟随，质心位移方向即头的朝向；
#   · 同虫一致性硬约束：同一只虫的头尾位置不会突变——端点近邻
#     链式传播把已定向帧的结果逐帧传给相邻帧。


def temporal_orientation_votes(frame_data, gap=15, min_shift=0.08):
    """时序质心位移投票（每帧一票，可为 None）。

    原理：线虫爬行时头先行、尾跟随，短时间窗内质心位移方向近似
    "头的朝向"。对当前帧 t：
      · 在约 gap 帧之前找一个内容不同的参考帧 r；
      · 位移 d = centroid_t - centroid_r，若 |d| >= min_shift*体长
        （虫确实移动了），则 t 帧离"运动前方"更近的那个端点判为头。
    投票与骨架点排序无关：只比较 t 帧自己的两个端点谁在位移方向上，
    不需要跨帧端点匹配（骨架 BFS 起点逐帧可能换端，跨帧匹配不可靠）。

    参数：
      gap       参考帧间隔（约 15 帧累积位移，帧率高时单帧位移太小）
      min_shift 触发投票的最小位移（×体长；不动/微动的帧不投票）
    返回 [1/0/None] * len(frame_data)
    """
    n = len(frame_data)
    votes = [None] * n
    recent = deque(maxlen=64)          # (帧序号, 质心, 图像哈希)
    for i, d in enumerate(frame_data):
        c = (d['head'] + d['tail']) / 2.0
        ref = None
        for j, cr, hj in recent:       # deque 按时间升序，取第一个足够早的
            if i - j >= gap and np.mean(np.abs(d['h'] - hj)) >= 2.0:
                ref = (cr, hj)
                break
        if ref is not None:
            disp = c - ref[0]
            shift = float(np.linalg.norm(disp))
            body_len = max(float(np.linalg.norm(d['head'] - d['tail'])), 1e-6)
            if shift >= min_shift * body_len:
                dh = disp / shift
                # 离运动前方更近的端点为头
                s_head = float(np.dot(d['head'] - c, dh))
                s_tail = float(np.dot(d['tail'] - c, dh))
                votes[i] = 1 if s_head >= s_tail else 0
        recent.append((i, c, d['h']))
    return votes


def worm_head_orientation(frame_data, gap=15, min_shift=0.08, jump=0.45):
    """按帧确定 head_is_first（1 = points[0] 为头）。

    流程：
      1. 时序投票（temporal_orientation_votes）在移动帧上直接判定，
         作为锚点；
      2. 从第一个锚点向前后链式传播：当前帧离上一帧"头部位置"更近的
         端点判为头（同虫相邻帧头尾位置不会突变）；两端都跳变超过
         jump×体长（画面切换/大位移）时链断，由下一个锚点重新定向；
      3. 完全没有时序锚点的虫（从未移动足够位移）：逐帧质心启发式，
         标记为弱证据。
      4. 链传播结果与时序票冲突时以时序票为准（运动学证据强于
         位置近邻），冲突计数记入诊断。

    返回 (hf_list, diag)：hf_list 为逐帧 head_is_first；
    diag = {'temporal': n, 'chain': n, 'fallback': n, 'conflict': n}
    """
    n = len(frame_data)
    if n == 0:
        return [], {}
    tvotes = temporal_orientation_votes(frame_data, gap, min_shift)
    anchors = [i for i, v in enumerate(tvotes) if v is not None]

    def _fallback(i):
        """质心启发式兜底（弱证据）：头 = 离两端中点更远的端点。"""
        d = frame_data[i]
        c = (d['head'] + d['tail']) / 2.0
        d0 = float(np.linalg.norm(d['head'] - c))
        d1 = float(np.linalg.norm(d['tail'] - c))
        return 1 if d0 >= d1 else 0

    def _head_pos(i, hf_list):
        d = frame_data[i]
        return d['head'] if hf_list[i] == 1 else d['tail']

    def _link(i, j, hf_list):
        """以 i 帧（已定向）传播到 j 帧。成功返回 head_is_first，失败 None。"""
        dj = frame_data[j]
        hp = _head_pos(i, hf_list)
        body = max(float(np.linalg.norm(dj['head'] - dj['tail'])), 1e-6)
        d0 = float(np.linalg.norm(dj['head'] - hp))
        d1 = float(np.linalg.norm(dj['tail'] - hp))
        if min(d0, d1) > jump * body:
            return None                 # 链断
        return 1 if d0 <= d1 else 0

    hf = [None] * n
    src = [None] * n

    if anchors:
        a0 = anchors[0]                 # 第一个时序锚点作为全局起点
        hf[a0] = tvotes[a0]
        src[a0] = 'temporal'
        for i in range(a0 - 1, -1, -1):        # 反向链：锚点 → 首帧
            v = _link(i + 1, i, hf)
            if v is None:
                v = _fallback(i)
                src[i] = 'fallback'
            else:
                src[i] = 'chain'
            hf[i] = v
    else:
        hf[0] = _fallback(0)            # 零运动虫：逐帧质心启发式
        src[0] = 'fallback'

    for i in range(1, n):               # 前向链
        if hf[i] is not None:
            continue
        if hf[i - 1] is not None:
            v = _link(i - 1, i, hf)
            if v is None:
                v = _fallback(i)
                src[i] = 'fallback'
            else:
                src[i] = 'chain'
        else:
            v = _fallback(i)
            src[i] = 'fallback'
        hf[i] = v

    # 时序票覆盖链结果，但过一致性检验（同虫头尾不突变硬约束）：
    # 时序票与链方向冲突时，若该帧两侧已定向邻居都站在链一边，判定为
    # 噪声翻转（如原地转头导致的质心位移方向突变），保持链方向（吸收）；
    # 否则以运动学证据为准（覆盖）。
    n_conflict = 0
    n_absorbed = 0
    for i in anchors:
        if hf[i] == tvotes[i]:
            src[i] = 'temporal'
            continue
        n_conflict += 1
        neigh = [hf[j] for j in (i - 1, i + 1)
                 if 0 <= j < n and hf[j] is not None]
        if len(neigh) >= 2 and all(v != tvotes[i] for v in neigh):
            n_absorbed += 1          # 孤立翻转：吸收，保持链方向
        else:
            hf[i] = tvotes[i]
            src[i] = 'temporal'

    diag = {'temporal': len(anchors) - n_absorbed,
            'chain': src.count('chain'),
            'fallback': src.count('fallback'),
            'conflict': n_conflict,
            'absorbed': n_absorbed}
    return hf, diag


# ============================================================
# 2. 图像哈希（时序参考帧去重用）
# ============================================================
def image_hash(img, size=12):
    """缩放到 size×size 的灰度均值序列作为去重指纹。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return g.reshape(-1)


# ============================================================
# 4. 多进程 worker：处理单只虫的全部帧
# ============================================================
def _process_worm(task):
    """处理单只虫：逐帧提取骨架 + 曲率 + 特征。
    返回 (worm_key, [per_frame_dict, ...])。
    per_frame_dict 包含主程序后续做头尾定向与融合所需的全部字段。
    头尾判定不在 worker 里做——它需要跨帧时序信息，由主程序串行完成。
    （死虫已由上游 z_tracking_only.py 过滤，无需死亡判据。）
    """
    worm_key, frames, input_dir = task
    frame_data = []
    for fr, rel in frames:
        img = cv2.imread(os.path.join(input_dir, rel), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ret = extract_skeleton_full(img)
        points = ret[0] if isinstance(ret, (tuple, list)) else None
        if points is None or len(points) < 10:
            continue
        kappa = compute_curvature_profile(points, n_samples=100)
        feats = extract_features(kappa)
        rule_label, rule_conf, _ = classify_by_rules(feats)
        sklen = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        h = image_hash(img)
        frame_data.append({
            'rel': rel, 'fr': fr,
            'head': np.asarray(points[0], dtype=np.float32),
            'tail': np.asarray(points[-1], dtype=np.float32),
            'h': h,
            'kappa': kappa.astype(np.float32),
            'feats': feats,
            'rule_label': rule_label, 'sklen': sklen,
        })
    return worm_key, frame_data


# ============================================================
# 5. 主程序
# ============================================================
def main():
    ap = argparse.ArgumentParser(description='多方法集成模拟标注（三类存活姿态）')
    ap.add_argument('--input', '-i', required=True, help='图片根目录')
    ap.add_argument('--output', '-o', default='simulated_labels.csv',
                    help='输出模拟标注 CSV')
    ap.add_argument('--max-worms', type=int, default=100,
                    help='最多处理多少只虫（默认 100；设为极大值即全量，如 99999999）')
    ap.add_argument('--workers', type=int, default=min(16, max(1, cpu_count() or 16)),
                    help='并行进程数（默认 min(16, CPU数)）')
    ap.add_argument('--exclude-list', '-x', action='append', default=None,
                    help='虫体排除清单（同 worm_labeler 语义）')
    args = ap.parse_args()

    t0 = time.time()
    input_dir = os.path.abspath(args.input)

    # 输出目录不存在则创建（直接指定子目录路径时非常容易忘，曾致 2 次
    # 全量处理后写盘才 FileNotFoundError，15 分钟白跑）。
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    exclude_worms = set()
    if args.exclude_list:
        for lf in args.exclude_list:
            with open(lf, encoding='utf-8-sig') as f:
                for line in f:
                    line = line.strip().lower().replace('\\', '/')
                    if line:
                        exclude_worms.add(line)

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    from worm_labeler import parse_five_dims

    GROUPS = {'control', '5mg', '8mg', '10mg', 'train', 'test'}
    # train/test 为公开数据集 worm_crops 的两个 split（2026-11-16 加入）；
    # 主数据 data_new 下不存在这两个目录名，加进来对主流程无影响。
    abs_segs = [p for p in input_dir.replace('\\', '/').split('/')
                if p and not p.endswith(':')]
    start = -1
    for i, s in enumerate(abs_segs):
        if s.lower() in GROUPS:
            start = i
            break
    prefix = abs_segs[start:] if start >= 0 else []

    worm_frames = defaultdict(list)
    worm_re = re.compile(r'(worm\d+)_(\d+)', re.IGNORECASE)

    # 标准五维结构（input_dir 是 data 根目录，group/date/row 是其子目录）时，
    # 用分层遍历 + 每组配额抽样；--max-worms 设极大值即处理全部虫。
    top_dirs = sorted(d for d in os.listdir(input_dir)
                      if os.path.isdir(os.path.join(input_dir, d)))
    group_dirs = [d for d in top_dirs if d.lower() in GROUPS]

    if len(group_dirs) >= 2:
        n_groups = len(group_dirs)
        group_quota = max(1, args.max_worms // n_groups)
        group_count = defaultdict(int)

        for group in group_dirs:
            gdir = os.path.join(input_dir, group)
            for date in sorted(os.listdir(gdir)):
                ddir = os.path.join(gdir, date)
                if not os.path.isdir(ddir):
                    continue
                for row in sorted(os.listdir(ddir)):
                    rdir = os.path.join(ddir, row)
                    if not os.path.isdir(rdir):
                        continue
                    for f in sorted(os.listdir(rdir)):
                        if not f.lower().endswith(exts):
                            continue
                        m = worm_re.match(f)
                        if not m:
                            continue
                        worm = m.group(1).lower()
                        frame = int(m.group(2))
                        worm_key = f'{group}/{date}/{row}/{worm}'.lower()
                        if worm_key in exclude_worms:
                            continue
                        if worm_key not in worm_frames:
                            if group_count[group] >= group_quota:
                                break
                            group_count[group] += 1
                        worm_frames[worm_key].append(
                            (frame, f'{group}/{date}/{row}/{f}'))
                    if group_count[group] >= group_quota:
                        break
                if group_count[group] >= group_quota:
                    break
    else:
        # 子目录或非标准结构：os.walk + 收集满即停
        for root, dirs, fs in os.walk(input_dir):
            dirs.sort()
            for f in sorted(fs):
                if not f.lower().endswith(exts):
                    continue
                rel = os.path.relpath(os.path.join(root, f), input_dir).replace('\\', '/')
                full_rel = '/'.join(prefix + [p for p in rel.split('/') if p])
                g, d, r, w, fr = parse_five_dims(full_rel)
                worm_key = f'{g}/{d}/{r}/{w}'.lower()
                if worm_key in exclude_worms:
                    continue
                worm_frames[worm_key].append((fr, rel))
            if len(worm_frames) >= args.max_worms:
                break

    print(f'共收集 {len(worm_frames)} 只虫')
    worm_keys = sorted(worm_frames.keys())
    print(f'处理 {len(worm_keys)} 只虫（workers={args.workers}）')

    # ---- 启动自检：先串行处理前 3 只虫，0 有效帧立即中止，避免白跑几十小时 ----
    tasks = [(wk, worm_frames[wk], input_dir) for wk in worm_keys]
    probe_n = min(3, len(tasks))
    probe_valid = 0
    for t in tasks[:probe_n]:
        _, fd = _process_worm(t)
        probe_valid += len(fd)
    print(f'启动自检：前 {probe_n} 只虫共 {probe_valid} 帧有效骨架')
    if probe_valid == 0:
        print('\n[错误] 前 3 只虫提取出 0 帧有效骨架，骨架提取对当前数据失效！')
        print('       可能：图像对比度过低 / 明暗场不符 / 裁剪格式与骨架算法不匹配。')
        print('       请先用 --max-worms 10 小批量调试，可视化确认骨架正确后再全量。')
        sys.exit(1)

    # ---- 并行逐虫提取（imap 保序 + 滚动进度条）----
    rows = []
    kappa_all = []
    kappa_stat = []
    n_tasks = len(tasks)
    results = []
    if args.workers > 1 and n_tasks > 1:
        chunksize = max(1, n_tasks // (args.workers * 4))
        with Pool(processes=args.workers) as pool:
            for i, res in enumerate(
                    pool.imap(_process_worm, tasks, chunksize=chunksize), 1):
                results.append(res)
                if i % 50 == 0 or i == n_tasks:
                    sys.stdout.write(
                        f'\r  进度 {i}/{n_tasks} 只虫 ({i/n_tasks*100:.1f}%)')
                    sys.stdout.flush()
        sys.stdout.write('\n')
    else:
        for i, t in enumerate(tasks, 1):
            results.append(_process_worm(t))
            if i % 10 == 0 or i == n_tasks:
                sys.stdout.write(
                    f'\r  进度 {i}/{n_tasks} 只虫 ({i/n_tasks*100:.1f}%)')
                sys.stdout.flush()
        sys.stdout.write('\n')

    # ---- 串行：头尾定向（时序锚点 + 链传播 + 质心兜底）+ 收集 ----
    ht_diag = defaultdict(int)
    for worm_key, frame_data in results:
        if not frame_data:
            continue
        frame_data.sort(key=lambda d: d['fr'])

        # 【新头尾体系】逐帧 head_is_first（时序为主，链传播保证同虫一致）
        hf_list, diag = worm_head_orientation(frame_data)
        for k, v in diag.items():
            ht_diag[k] += v

        for d, tv_hf in zip(frame_data, hf_list):
            rows.append({
                'rel': d['rel'], 'worm': worm_key, 'frame': d['fr'],
                'head_first': tv_hf,
                'rule_label': d['rule_label'],
                'bbi': d['feats']['bbi'], 'sklen': d['sklen'],
                # 形状诊断列（2026-10-23 新增）：供 Ω 类内部分型分析
                # （中段峰 vs 端部峰），不参与标签生成，下游 step8/9 不读
                'mid_dominance': d['feats']['mid_dominance'],
                'peak_midness': d['feats']['peak_midness'],
                'max_kappa': d['feats']['max_kappa'],
            })
            kappa_all.append(d['kappa'])
            kappa_stat.append(d['feats'])

    if not rows:
        print('没有有效帧，退出')
        return
    n_all = len(rows)
    print(f'共 {n_all} 帧有效  (耗时 {time.time()-t0:.1f}s)')
    print(f'头尾定向证据分布：时序运动 {ht_diag["temporal"]} 帧 '
          f'({ht_diag["temporal"]/max(n_all,1)*100:.0f}%) / '
          f'链式传播 {ht_diag["chain"]} 帧 / 质心兜底 {ht_diag["fallback"]} 帧；'
          f'链与时序冲突 {ht_diag["conflict"]} 帧'
          f'（孤立噪声翻转被一致性约束吸收 {ht_diag["absorbed"]} 帧，'
          f'其余以时序为准）')

    # ---- 无监督聚类（3 类行为簇：正弦/Ω/笔直）----
    K = np.array(kappa_all)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(K)
    n_clusters = min(3, len(rows))
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full',
                          random_state=42, n_init=5)
    cluster_ids = gmm.fit_predict(X_pca)

    cluster_feat = defaultdict(list)
    for cid, feats in zip(cluster_ids, kappa_stat):
        cluster_feat[cid].append(feats)
    cluster_stats = {}
    for cid, fl in cluster_feat.items():
        cluster_stats[cid] = {
            'bbi': np.mean([f['bbi'] for f in fl]),
            'n_regions': np.mean([f['n_high_regions'] for f in fl]),
            'mid_dominance': np.mean([f['mid_dominance'] for f in fl]),
            'max_kappa': np.mean([f['max_kappa'] for f in fl]),
            'periodicity': np.mean([f['periodicity_peaks'] for f in fl]),
        }
    cids = sorted(cluster_stats.keys())

    # 簇映射（三类）：最低 BBI -> 笔直型；最高 max_kappa -> Ω 型（单峰深弯）；
    # 剩余 -> 正弦型（多波浪）。max_kappa 与 classify_by_rules 的 Ω 判据一致。
    mapping = {}
    remaining = set(cids)
    if remaining:
        straight = min(remaining, key=lambda c: cluster_stats[c]['bbi'])
        mapping[straight] = 2
        remaining.discard(straight)
    if remaining:
        omega = max(remaining, key=lambda c: cluster_stats[c]['max_kappa'])
        mapping[omega] = 1
        remaining.discard(omega)
    for c in remaining:
        mapping[c] = 0

    print('簇映射（聚类ID -> 行为类）：',
          {int(c): LABEL_CN[l] for c, l in sorted(mapping.items())})

    # ---- 集成融合（双信号加权：三类体系）----
    # 设计（2026-10-23）：三类判据去头尾化后，规则与聚类可能给出不同结论。
    #   · 笔直型 = 低曲率(bbi<STRAIGHT_BBI)，纯形状判据、死虫已上游过滤，最清晰；
    #   · Ω 型 = 单峰深弯(max_kappa>=0.35 且 n_regions<=1)，「规则」或「聚类」
    #     任一信号命中即采纳——规则侧用绝对曲率阈值，聚类侧对 κ(s) 整体
    #     形状更敏感，二者互补；
    #   · 两信号都判正弦 → 正弦型（兜底类）；
    #   · rule_label / cluster_label 均保留入 CSV 供一致性分析。
    label_counts = defaultdict(int)
    for idx, row in enumerate(rows):
        cluster_label = mapping[cluster_ids[idx]]
        rule_label = row['rule_label']
        bbi = row['bbi']

        if bbi < STRAIGHT_BBI:
            final, conf, src = 2, 0.85, '笔直(直形)'
        else:
            r, c = rule_label, cluster_label
            # 三类体系下 Ω 型(1) 是唯一的"结构化"姿态，正弦(0) 为默认
            if r == 1 and c == 1:
                final, conf, src = 1, 0.90, 'rule+cluster一致'
            elif r == 1 or c == 1:
                final, conf, src = 1, 0.65, ('rule(Ω)' if r == 1 else 'cluster(Ω)')
            else:
                final, conf, src = 0, 0.80, '正弦(双信号一致)'

        row['cluster_label'] = cluster_label
        row['final_label'] = final
        row['confidence'] = conf
        row['source'] = src
        label_counts[final] += 1

    # ---- 输出 ----
    with open(args.output, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'label', 'confidence', 'source',
                    'rule_label', 'cluster_label', 'head_is_first',
                    'bbi', 'skeleton_len', 'mid_dominance', 'peak_midness',
                    'max_kappa'])
        for row in rows:
            w.writerow([row['rel'], row['final_label'], f"{row['confidence']:.2f}",
                        row['source'], row['rule_label'], row['cluster_label'],
                        row['head_first'], f"{row['bbi']:.4f}",
                        f"{row['sklen']:.1f}",
                        f"{row['mid_dominance']:.4f}",
                        f"{row['peak_midness']:.4f}",
                        f"{row['max_kappa']:.4f}"])

    print('\n标注分布：')
    for lbl in sorted(label_counts):
        print(f'  [{lbl}] {LABEL_CN[lbl]}: {label_counts[lbl]} 张')
    print(f'\n模拟标注已保存到 {args.output}')
    print('可直接作为 step8_train_svm.py 的 -l 输入（含 filename 与 label 列）')
    print(f'总耗时 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
