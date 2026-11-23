"""
z_tracking_only.py
只做：YOLOv8 + BoTSORT 跟踪 -> 逐只线虫逐帧裁剪小图 -> 保存 PNG

输出结构（根目录不变，其余层命名严格对齐原 z.py）：
    {OUT_ROOT}/{浓度}/{MMDD}/{皿字母}/worm{虫号}_{顺序号}.png
    例：.../control/0116/B/worm3_001.png

文件名约定：260116_B1_0000_20260116_165138.avi
    B1          -> 皿号（B=行字母, 1=浓度列）
    1/2/3/4     -> 浓度：1=control, 2=8mg, 3=10mg, 4=5mg
    20260116    -> 拍摄日期，取后四位 -> 0116
    5/6         -> 无浓度含义，跳过这些视频（共 168 个）

死虫过滤：依据标注表的 DeadBoxes，截图时绕开“大部分重叠”(IoU>DEAD_OVERLAP_IOU) 的死虫，只截活虫。

性能要点：
  1) 模型按进程缓存（extract.py 的 _get_model），整个进程只加载一次模型、只做一次 CUDA 初始化，
     不再每个视频 reload（原实现 504 个视频 reload 504 次）。
  2) 进程池跨视频并行：进程 A 跟踪(GPU) 视频 N+1 时，进程 B 可并行裁剪(CPU/IO) 视频 N，
     解决“跟踪(GPU) 与 裁剪(CPU/IO) 串行无法重叠”的问题。
  3) 主进程监听线程 + 队列，滚动刷新一行进度百分比。
  4) 每个视频成功完成后写 .done_<fname> 标记，中断重跑可整段跳过（连跟踪都省）。
"""

import os
import sys
import csv
import time
import datetime
import threading
import contextlib
import io
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
from main import main

# ====================== 可配置项 ======================
SRC_DIR = r"D:/260116"          # 视频根目录（递归查找 .avi）
OUT_ROOT = r"D:/data_new/images"     # 输出根目录
PAD = 20                        # 裁剪时虫子四周留的边距（像素）
CSV_PATH = r"Detail-260116-20260116-20260211.csv"  # 死/活虫标注
DEAD_OVERLAP_IOU = 0.5          # “方框大部分重叠” 判定阈值（IoU），超过即视为死虫跳过

# 并行度：每个工作进程会独立加载一份 YOLO 模型（占显存）。笔记本 T600(约4GB) 建议 2；
# 显存更充裕可上调；设为 1 则单进程（仍有模型缓存+进度，最稳）。
MAX_WORKERS = 2
FRAME_STEP = 20                 # 进度上报抽稀：每 FRAME_STEP 帧上报一次，减少队列压力

# 浓度映射（与 z.py 的 dic 完全一致）：皿号里的数字 -> 组别名。5/6 不在映射内 -> 跳过。
CONC_MAP = {"1": "control", "2": "8mg", "3": "10mg", "4": "5mg"}
# =====================================================


def parse_video_name(fname):
    """严格按 z.py 的切片方式解析：浓度=fname[8], 日期=fname[19:23](MMDD), 皿字母=fname[7]。
    数字 5、6 没有浓度含义 -> 返回 None 以跳过该视频。"""
    if len(fname) < 23:
        return None
    conc = CONC_MAP.get(fname[8])
    if conc is None:
        return None
    return conc, fname[19:23], fname[7]


def crop_worm(frame, box, pad=PAD):
    """按中心格式 [cx, cy, w, h] 裁剪虫子，做边界 clamp 防止越界回绕。"""
    cx, cy, w, h = map(int, box)
    H, W = frame.shape[:2]
    x1 = max(0, cx - w // 2 - pad)
    y1 = max(0, cy - h // 2 - pad)
    x2 = min(W, cx + w // 2 + pad)
    y2 = min(H, cy + h // 2 + pad)
    return frame[y1:y2, x1:x2]


def parse_boxes(s):
    """把 'x y w h|x y w h' 解析为整数四元组列表；空串返回 []。"""
    if not s or not s.strip():
        return []
    out = []
    for part in s.strip().split("|"):
        nums = part.split()
        if len(nums) == 4:
            out.append(tuple(int(float(n)) for n in nums))
    return out


def load_dead_boxes(csv_path):
    """读取 GBK 编码的标注表，建立 (皿号, YYYYMMDD) -> [(x,y,w,h), ...] 死虫框字典。"""
    dead_map, alive_map = {}, {}
    with open(csv_path, "r", encoding="gbk", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        pn_i = idx.get("PN", 2)
        date_i = idx.get("Date", 3)
        dead_i = idx.get("DeadBoxes", 11)
        alive_i = idx.get("AliveBoxes", 12)
        for row in reader:
            if len(row) <= max(pn_i, date_i, dead_i):
                continue
            pn = row[pn_i].strip()
            ds = row[date_i].split()[0]          # "2026/1/16 0:00" -> "2026/1/16"
            try:
                dt = datetime.datetime.strptime(ds, "%Y/%m/%d")
            except ValueError:
                dt = datetime.datetime.strptime(ds, "%Y-%m-%d")
            ymd = dt.strftime("%Y%m%d")
            dead_map[(pn, ymd)] = parse_boxes(row[dead_i])
            alive_map[(pn, ymd)] = parse_boxes(row[alive_i])
    return dead_map, alive_map


def boxes_overlap_mostly(track_box, dead_boxes, iou_thr):
    """判断跟踪框 track_box([cx,cy,w,h] 中心格式) 是否与任一死虫框“大部分重叠”。
    死虫框格式未知，对 左上/中心 两种解释都算一遍取 IoU 最大者，保证格式无关。"""
    cx, cy, w, h = map(float, track_box)
    tx1, ty1, tx2, ty2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    ta = (tx2 - tx1) * (ty2 - ty1)
    if ta <= 0:
        return False
    for (dx, dy, dw, dh) in dead_boxes:
        cands = ((dx, dy, dx + dw, dy + dh),
                 (dx - dw / 2, dy - dh / 2, dx + dw / 2, dy + dh / 2))
        for (x1, y1, x2, y2) in cands:
            ix1, iy1 = max(tx1, x1), max(ty1, y1)
            ix2, iy2 = min(tx2, x2), min(ty2, y2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                continue
            da = (x2 - x1) * (y2 - y1)
            union = ta + da - inter
            if union > 0 and inter / union > iou_thr:
                return True
    return False


def _get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    return n


def _process_video(task):
    """工作进程：对单个视频做 跟踪 + 死虫过滤 + 裁剪。返回 (fname, saved, dead_this, err)。"""
    (video_path, fname, conc, today, letter, out_root, dead_boxes, prog_q) = task
    out_dir = os.path.join(out_root, conc, today, letter)
    os.makedirs(out_dir, exist_ok=True)

    # 跟踪：屏蔽子进程里 main/各阶段的 print（避免多进程刷屏），进度走队列
    cnt = {"n": 0}
    def cb(_):
        cnt["n"] += 1
        if prog_q is not None and cnt["n"] % FRAME_STEP == 0:
            prog_q.put(("frame", FRAME_STEP))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            tracks_by_id = main(video_path, progress_cb=cb)
    except Exception as e:
        return (fname, 0, 0, f"main()失败: {e}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (fname, 0, 0, "无法打开视频")

    frame_map = defaultdict(list)                 # frame -> [(tid, det), ...]，同一帧只解码一次
    for tid, dets in tracks_by_id.items():
        for det in dets:
            frame_map[det["frame"]].append((tid, det))
    seq_cnt = defaultdict(int)                    # 每只虫的顺序帧号（按帧升序累加）

    saved = 0
    dead_this = 0
    for fidx in sorted(frame_map):
        # 先按死虫过滤（det['box'] 不读帧即可判），本帧全是死虫则连解码都省掉
        alive_dets = []
        for tid, det in frame_map[fidx]:
            if dead_boxes and boxes_overlap_mostly(det["box"], dead_boxes, DEAD_OVERLAP_IOU):
                dead_this += 1
                continue
            alive_dets.append((tid, det))
        if not alive_dets:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        for tid, det in alive_dets:
            box = det["box"]
            seq_cnt[tid] += 1
            out_path = os.path.join(out_dir, f"worm{tid}_{seq_cnt[tid]:03d}.png")
            if os.path.exists(out_path):           # 断点续跑（裁剪级）
                continue
            cv2.imwrite(out_path, crop_worm(frame, box))
            saved += 1
    cap.release()

    # 整段成功 -> 写完成标记，重跑可整段跳过（连跟踪都省）
    try:
        with open(os.path.join(out_dir, f".done_{fname}"), "w"):
            pass
    except OSError:
        pass
    return (fname, saved, dead_this, None)


def _progress_listener(q, total_videos, total_frames, start):
    """主进程监听线程：读队列事件，滚动刷新一行进度百分比。"""
    done_videos = 0
    proc_frames = 0
    saved = 0
    dead = 0
    while True:
        ev = q.get()
        if ev[0] == "STOP":
            break
        if ev[0] == "frame":
            proc_frames += ev[1]
        elif ev[0] == "done":
            res = ev[1]
            done_videos += 1
            saved += res[1]
            dead += res[2]
        pct = min(100.0, proc_frames / total_frames * 100) if total_frames else 0.0
        el = time.time() - start
        sys.stdout.write(
            f"\r帧进度 {pct:5.1f}% ({proc_frames}/{total_frames}) | "
            f"视频 {done_videos}/{total_videos} | 存 {saved} 跳死 {dead} | {el:6.1f}s   ")
        sys.stdout.flush()
    el = time.time() - start
    sys.stdout.write(
        f"\r帧进度 100.0% ({total_frames}/{total_frames}) | "
        f"视频 {done_videos}/{total_videos} | 存 {saved} 跳死 {dead} | {el:6.1f}s   \n")
    sys.stdout.flush()


def tracking_crop(src_dir=SRC_DIR, out_root=OUT_ROOT, max_workers=MAX_WORKERS):
    # 1) 扫描视频 + 预读帧数（作为进度分母）；已完成(.done)的整段跳过
    raw = []
    skipped = 0
    resumed = 0
    for root, _, files in os.walk(src_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(".avi"):
                continue
            parsed = parse_video_name(fname)
            if parsed is None:
                skipped += 1
                continue
            conc, today, letter = parsed
            marker = os.path.join(out_root, conc, today, letter, f".done_{fname}")
            if os.path.exists(marker):
                resumed += 1
                continue
            raw.append((os.path.join(root, fname), fname, conc, today, letter))

    # 2) 加载死虫标注（主进程一次）
    try:
        dead_map, _ = load_dead_boxes(CSV_PATH)
    except Exception as e:
        print(f"[警告] 加载死虫标注失败({e})，将不做死虫过滤")
        dead_map = {}

    total_frames = 0
    tasks = []
    for (vp, fname, conc, today, letter) in raw:
        total_frames += _get_frame_count(vp)
        parts = fname[:-4].split("_")
        csv_key = (parts[1] if len(parts) > 1 else "", parts[3] if len(parts) > 3 else "")
        tasks.append((vp, fname, conc, today, letter, out_root, dead_map.get(csv_key, [])))

    total_videos = len(tasks)
    print(f"待处理 {total_videos} 个视频（总帧数 {total_frames}）；"
          f"跳过无浓度 {skipped} 个、已完成续跑 {resumed} 个；并行进程数={max_workers}")
    if total_videos == 0:
        print("没有需要处理的视频。")
        return

    mp.freeze_support()
    mgr = mp.Manager()
    q = mgr.Queue()
    tasks = [t + (q,) for t in tasks]

    start = time.time()
    listener = threading.Thread(target=_progress_listener,
                                args=(q, total_videos, total_frames, start), daemon=True)
    listener.start()

    results = []
    try:
        if max_workers and max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_process_video, t) for t in tasks]
                for fut in as_completed(futs):
                    res = fut.result()
                    results.append(res)
                    q.put(("done", res))
        else:
            for t in tasks:
                res = _process_video(t)
                results.append(res)
                q.put(("done", res))
    except Exception as e:
        print(f"\n[警告] 进程池执行异常({e})，回退为单进程顺序执行（已存文件/标记会自动跳过）")
        for t in tasks:
            res = _process_video(t)
            results.append(res)
            q.put(("done", res))
    q.put(("STOP",))
    listener.join()
    mgr.shutdown()

    total_saved = sum(r[1] for r in results)
    total_dead = sum(r[2] for r in results)
    errors = [(r[0], r[3]) for r in results if r[3]]
    el = time.time() - start
    print(f"\n全部完成：共保存 {total_saved} 张活虫小图（跳过重叠死虫 {total_dead} 张），"
          f"跳过无浓度 {skipped} 个、续跑 {resumed} 个，用时 {el:.1f}s")
    if errors:
        print(f"有 {len(errors)} 个视频出错：")
        for fn, err in errors[:20]:
            print(f"  - {fn}: {err}")
    print(f"结构: {out_root}/浓度/日期/皿字母/文件名")


if __name__ == "__main__":
    mp.freeze_support()
    # 用法：python z_tracking_only.py [视频根目录] [并行进程数]
    sd = sys.argv[1] if len(sys.argv) > 1 else SRC_DIR
    mw = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_WORKERS
    tracking_crop(sd, OUT_ROOT, mw)
