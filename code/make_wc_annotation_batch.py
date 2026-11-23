# -*- coding: utf-8 -*-
"""worm_crops 人工标注批次生成：按规则类别分层抽样（盲标协议）。

- 每个规则类（0 正弦 / 1 Ω / 2 笔直）各抽 n_per 张，类内均匀随机（seed 固定）
- item_id 前缀 W，与主数据 F/C 系列区分
- 输出：batch.csv（item_id,rel_path，喂 human_label_gui_v3）、key.csv（规则标签，
  仅分析用，不得给标注者看）、sample_list.txt（供 SFTP 下载图片）
"""
import csv
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'annotation_wc', 'simulated_labels.csv')
OUTDIR = os.path.join(HERE, 'annotation_wc')
N_PER = 100
SEED = 42

rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))
by_cls = defaultdict(list)
for r in rows:
    if r['rule_label'] in ('0', '1', '2'):
        by_cls[r['rule_label']].append(r)

rng = random.Random(SEED)
picked = []
for cls in ('0', '1', '2'):
    pool = by_cls[cls]
    take = min(N_PER, len(pool))
    smp = rng.sample(pool, take)
    picked.extend(smp)
    print(f'class {cls}: pool={len(pool)}, picked={take}')

rng.shuffle(picked)  # 打乱顺序，避免界面上一段连续同类

with open(os.path.join(OUTDIR, 'batch.csv'), 'w', newline='',
          encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['item_id', 'rel_path'])
    for i, r in enumerate(picked, 1):
        w.writerow([f'W{i:03d}', r['filename']])

with open(os.path.join(OUTDIR, 'key.csv'), 'w', newline='',
          encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['item_id', 'rel_path', 'rule_label', 'label', 'confidence',
                'source', 'bbi', 'max_kappa'])
    for i, r in enumerate(picked, 1):
        w.writerow([f'W{i:03d}', r['filename'], r['rule_label'], r['label'],
                    r['confidence'], r['source'], r['bbi'], r['max_kappa']])

with open(os.path.join(OUTDIR, 'sample_list.txt'), 'w', encoding='utf-8') as f:
    for r in picked:
        f.write(r['filename'].replace('\\', '/') + '\n')

print(f'共 {len(picked)} 张 -> annotation_wc/batch.csv, key.csv, sample_list.txt')
