#!/usr/bin/env bash
# V17 全量复现流水线 —— 在原项目 worm_5class_pipeline 下运行
# 用法:
#   bash run_full_v17.sh            正常跑；已完成的步骤自动跳过（断点续跑）
#   bash run_full_v17.sh --redo-all 清空完成标记，从第 1 步重新开始
# 输出: atlas_out_all/（新目录，不动 atlas_out_full）；日志 logs_v17/
#
# 断点续跑说明:
#   每步成功后写 logs_v17/<步骤名>.done 标记，重跑本脚本时有标记的步骤直接跳过。
#   想单独重跑某一步: 删除对应标记再跑，例如 rm logs_v17/05_indep.done
#   注意: 第 1 步（01_cluster）与第 5 步（05_indep）的提取缓存是整步结束才写盘的，
#   这两步跑到一半被打断，仍需整步重来（脚本层无法改变这一点）。
set -u
cd "$(dirname "$0")" || exit 1

OUT=atlas_out_all
# 数据根目录（2026-11-13 用户实测修正）：组目录在 .../data_new/images 这一层，
# 隔离区也在 data_new/images/_noise_quarantine。旧默认值少了 images 一层，
# 会让 parse_five_dims 把 'images' 当成组名、step15 的 4 层 glob 直接 0 命中。
# 保留对旧布局（组目录直接在 data_new 下）的自动回退，避免不同批次数据踩坑。
ROOT_BASE=/media/yczxserver/SSD-windows/wjf/worm_Sclass_pipeline_f/data_new
if [ -d "$ROOT_BASE/images/control" ]; then
  DATA_DEFAULT="$ROOT_BASE/images"
else
  DATA_DEFAULT="$ROOT_BASE"
fi
DATA=${WORM_DATA_ROOT:-$DATA_DEFAULT}
if [ ! -d "$DATA/control" ]; then
  echo "[警告] 数据根 $DATA 下没有 control/ 组目录，请用 WORM_DATA_ROOT 显式指定" >&2
fi
echo "[数据根] $DATA"
LOG=logs_v17
mkdir -p "$LOG" "$OUT"

if [ "${1:-}" = "--redo-all" ]; then
  rm -f "$LOG"/*.done
  echo "已清空全部完成标记，将从第 1 步重新开始。" | tee -a "$LOG/driver.log"
fi

# 服务器解释器（PyCharm SSH 解释器，2026-10-29 确认）；可用 WORM_PYTHON 覆盖
PY() { "${WORM_PYTHON:-/home/yczxserver/.virtualenvs/worm_5class_pipeline/bin/python}" "$@"; }

step() {
  name=$1; shift
  if [ -f "$LOG/$name.done" ]; then
    echo "===== $name SKIP（已完成，跳过）$(date '+%F %T') =====" | tee -a "$LOG/driver.log"
    return 0
  fi
  echo "===== $name START $(date '+%F %T') =====" | tee -a "$LOG/driver.log"
  if "$@" > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.done"
    echo "===== $name OK $(date '+%F %T') =====" | tee -a "$LOG/driver.log"
  else
    rc=$?
    echo "===== $name FAIL rc=$rc $(date '+%F %T') =====" | tee -a "$LOG/driver.log"
    exit 1
  fi
}

# 1. 骨架/曲率提取 + 图谱（全量，不设 max-worms / max-frames）
step 01_cluster PY cluster_atlas.py -i "$DATA" -o "$OUT" --workers 20

# 2. 层级 / 年龄 / 帧审计
step 02_hierarchy PY step11_hierarchy.py -i "$OUT/atlas_worm_hist.csv" -o "$OUT/hierarchy_results.csv"
step 03_age PY step12_age_trajectory.py -i "$OUT/atlas_worm_hist.csv" -o "$OUT/age_trajectory.csv"
step 04_frame_audit PY step13_frame_audit.py --root "$DATA" --cache "$OUT/kappa_cache.npz" \
  --hist "$OUT/atlas_worm_hist.csv" -o "$OUT/frame_audit.csv"

# 3. 四条独立测量管线 + 轮廓伪结构诊断
step 05_indep PY step15_independent_shape.py --root "$DATA" -o "$OUT/indep" --workers 20
step 06_outline PY step20_outline_check.py -c "$OUT/indep/indep_cache.npz" \
  -o "$OUT/indep/outline_check.txt"

# 4. 连续轴标定 + 段内指纹
step 07_axis PY step16_calibrate_axis.py -c "$OUT/kappa_cache.npz" --root "$DATA" \
  --human-labels human_labels.csv --human-batch human_annotation_batch.csv -o "$OUT/axis"
step 08_profile PY step17_posture_profile.py -c "$OUT/kappa_cache.npz" -o "$OUT/profile" --min-frames 50

# 5. 人工验证（三人 A/B/C + 预注册判定）
step 09_agreement PY step18_agreement_v2.py -d annotation_v2 -c "$OUT/kappa_cache.npz" \
  -o annotation_v2/agreement_full
step 10_rater PY step18b_rater_profile.py \
  -l annotation_v2/labels_A.csv annotation_v2/labels_B.csv annotation_v2/labels_C.csv \
  --ref human_labels.csv -c "$OUT/kappa_cache.npz" -o annotation_v2/rater_profile_full.txt
step 11_unusable PY step18c_unusable_audit.py \
  -l annotation_v2/labels_A.csv annotation_v2/labels_B.csv annotation_v2/labels_C.csv \
  --root "$DATA" -o annotation_v2/unusable_audit_full.csv
step 12_verdict PY step18d_verdict_audit.py -d annotation_v2 -c "$OUT/kappa_cache.npz" \
  -o annotation_v2/verdict_audit_full.txt

echo "ALL DONE $(date '+%F %T')" | tee -a "$LOG/driver.log"
