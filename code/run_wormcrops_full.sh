#!/usr/bin/env bash
# worm_crops 公开数据集 —— 论文核心链路复现流水线
# 复现范围（无时序/无给药组/无人工标注，故论文 12 步中取 6 步通用核心链）：
#   00_prep      worm_crops → 五维符号链接树（wormcrops_5dim/）
#   01_cluster   cluster_atlas：骨架/曲率提取 + κ(s) 聚类图谱（kappa_cache + atlas_worm_hist）
#   02_indep     step15：四条独立测量管线（indep_validity/modality/dbscan/agreement）
#   03_outline   step20：轮廓伪结构诊断
#   04_simulate  simulate_labels：规则模拟标注（GROUPS 已含 train/test）
#   05_step8     step8_train_svm：SBPD/fPCA + 四特征 SVM 对比（表2/表3 同款）
#   06_step9     step9_group_cv：跨组（train↔test）交叉验证
# 不适用（依赖时序/人工标注，已跳过）：step11/12/13（年龄/层级/帧审计需 21 日期结构）、
#   step16/18（人工标注校准与一致性）、step17（每虫 ≥50 帧指纹，公开数据每虫仅 1 帧）。
#
# 用法（服务器）：
#   bash run_wormcrops_full.sh            # 断点续跑
#   bash run_wormcrops_full.sh --redo-all # 清标记重跑
# 变量：WORM_PYTHON 覆盖解释器；CROPS_SRC 覆盖 worm_crops 源路径；WORKERS 覆盖并行数
set -u
cd "$(dirname "$0")" || exit 1
ROOT=$(pwd)

CROPS_SRC=${CROPS_SRC:-/media/yczxserver/SSD-windows/wjf/worm_crops}
DATA=$ROOT/wormcrops_5dim
OUT=atlas_out_wormcrops
LOG=logs_wormcrops
WORKERS=${WORKERS:-20}
mkdir -p "$LOG" "$OUT"

PY() { "${WORM_PYTHON:-/home/yczxserver/.virtualenvs/worm_5class_pipeline/bin/python}" "$@"; }

step() {
  name=$1; shift
  if [ -f "$LOG/$name.done" ]; then
    echo "===== $name SKIP $(date '+%F %T') =====" | tee -a "$LOG/driver.log"
    return 0
  fi
  echo "===== $name START $(date '+%F %T') =====" | tee -a "$LOG/driver.log"
  if "$@" > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.done"
    echo "===== $name OK $(date '+%F %T') =====" | tee -a "$LOG/driver.log"
  else
    echo "===== $name FAIL rc=$? $(date '+%F %T') =====（详见 $LOG/$name.log）" \
      | tee -a "$LOG/driver.log"
    exit 1
  fi
}

# 0. 五维树（幂等，已有链接自动跳过）
step 00_prep PY prep_wormcrops_5dim.py --src "$CROPS_SRC" --out "$DATA"

# 1. 骨架/曲率提取 + 图谱。--min-worm-frames 1：公开数据每虫仅 1 帧（默认 10 会全部过滤）
step 01_cluster PY cluster_atlas.py -i "$DATA" -o "$OUT" \
  --workers "$WORKERS" --min-worm-frames 1

# 2. 四条独立测量管线
step 02_indep PY step15_independent_shape.py --root "$DATA" \
  -o "$OUT/indep" --workers "$WORKERS"

# 3. 轮廓伪结构诊断
step 03_outline PY step20_outline_check.py -c "$OUT/indep/indep_cache.npz" \
  -o "$OUT/indep/outline_check.txt"

# 4. 规则模拟标注（全量；--max-worms 拉满即不抽样）
step 04_simulate PY simulate_labels.py -i "$DATA" \
  -o "$OUT/simulated_labels.csv" --max-worms 999999 --workers "$WORKERS"

# 5. SBPD/fPCA + 四特征 SVM（表2/表3 同款）。
#    step8 的缓存文件落在 CWD，切到 $OUT/step8 执行以免覆盖主数据的缓存。
mkdir -p "$OUT/step8"
step 05_step8 bash -c "cd '$OUT/step8' && PY() { \"\${WORM_PYTHON:-/home/yczxserver/.virtualenvs/worm_5class_pipeline/bin/python}\" \"\$@\"; } && PY '$ROOT/step8_train_svm.py' -l '$ROOT/$OUT/simulated_labels.csv' -i '$DATA' --out-dir . --model svm_model_wormcrops.pkl --workers '$WORKERS'"

# 6. 跨组交叉验证（train ↔ test）
mkdir -p "$OUT/step9"
step 06_step9 PY step9_group_cv.py -c "$OUT/step8/sbpd_raw_cache.pkl" \
  -l "$OUT/simulated_labels.csv" -o "$OUT/step9/group_cv_results.csv" \
  --seed 42

echo "ALL DONE $(date '+%F %T')" | tee -a "$LOG/driver.log"
echo "输出: $OUT/（summary：$OUT/indep/、$OUT/step8/table2*.csv、$OUT/step9/group_cv_results.csv）"
