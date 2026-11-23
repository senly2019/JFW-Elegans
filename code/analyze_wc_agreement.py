# -*- coding: utf-8 -*-
"""worm_crops 人工标注 vs 规则判定 一致性分析。

用法：
  python analyze_wc_agreement.py annotation_wc/labels_R1.csv \
      [--r2 annotation_wc/labels_R2.csv]

输出：
  - 人–规则 3 类 κ（人类别映射回规则空间：1=STR->2, 2=OMG->1, 3=SIN->0）
  - 每规则类的人–规则一致率（规则召回视角）+ 混淆表
  - 人–人 κ（若提供 --r2）
  - BAD/AMB 使用计数
  - 报告写入 annotation_wc/agreement_report.txt
"""
import csv
import os
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.join(HERE, 'annotation_wc', 'key.csv')
OUT = os.path.join(HERE, 'annotation_wc', 'agreement_report.txt')

H2R = {'1': 2, '2': 1, '3': 0}   # 人工码 -> 规则码（0=SIN,1=OMG,2=STR）
RNAME = {0: 'SIN', 1: 'OMG', 2: 'STR'}
CODENAME = {'0': 'BAD', '1': 'STR', '2': 'OMG', '3': 'SIN', '9': 'AMB'}


def load_labels(path):
    d = {}
    for r in csv.DictReader(open(path, encoding='utf-8-sig')):
        iid = (r.get('item_id') or '').strip()
        code = (r.get('label_code') or '').strip()
        if iid and code != '':
            d[iid] = code
    return d


def kappa(a, b):
    return cohen_kappa_score(a, b)


def main():
    r1_path = sys.argv[1]
    r2_path = None
    if '--r2' in sys.argv:
        r2_path = sys.argv[sys.argv.index('--r2') + 1]

    key = {r['item_id']: r for r in csv.DictReader(open(KEY, encoding='utf-8-sig'))}
    r1 = load_labels(r1_path)
    lines = []
    P = lines.append
    P('worm_crops 人工标注 vs 规则判定 一致性报告')
    P('=' * 60)
    P(f'批次: 300 张（每规则类 100），标注文件: {os.path.basename(r1_path)}'
      + (f' + {os.path.basename(r2_path)}' if r2_path else ''))
    P(f'已标注: {len(r1)}/{len(key)}')
    P('')

    raters = [('R1', r1_path, r1)]
    if r2_path:
        raters.append(('R2', r2_path, load_labels(r2_path)))
    for tag, _path, lab in raters:
        common = [iid for iid in key if iid in lab]
        codes = Counter(lab[iid] for iid in common)
        P(f'[{tag}] n={len(common)}  码分布: ' +
          ', '.join(f'{CODENAME.get(k, k)}={v}' for k, v in sorted(codes.items())))
        # 人–规则 3 类 κ（排除 BAD/AMB）
        ya, yb = [], []
        for iid in common:
            h = lab[iid]
            if h in H2R:
                ya.append(H2R[h])
                yb.append(int(key[iid]['rule_label']))
        if ya:
            P(f'  人–规则 3类κ = {kappa(yb, ya):+.3f}  (n={len(ya)}, '
              f'一致率 {np.mean(np.asarray(ya)==np.asarray(yb))*100:.1f}%)')
            cm = confusion_matrix(yb, ya, labels=[0, 1, 2])
            P('  规则\\人工 混淆（行=规则类）: ' +
              '  '.join(f'{RNAME[c]}={int(cm[c].sum())}' for c in range(3)))
            for c in range(3):
                P(f'    {RNAME[c]}: 规则判对 {int(cm[c][c])}/{int(cm[c].sum())}'
                  f' = {cm[c][c]/max(1,cm[c].sum())*100:.1f}%  '
                  f'人工精确率 {cm[c][c]/max(1,int(cm[:,c].sum()))*100:.1f}%')
        bad = sum(1 for iid in common if lab[iid] == '0')
        amb = sum(1 for iid in common if lab[iid] == '9')
        P(f'  BAD(图像不可用)={bad}  AMB(拿不准)={amb}')
        P('')

    if r2_path:
        r2 = raters[1][2]
        common = [iid for iid in key if iid in r1 and iid in r2
                  and r1[iid] in H2R and r2[iid] in H2R]
        if common:
            a = [H2R[r1[i]] for i in common]
            b = [H2R[r2[i]] for i in common]
            P(f'人–人 3类κ = {kappa(a, b):+.3f} (n={len(common)}, '
              f'一致率 {np.mean(np.asarray(a)==np.asarray(b))*100:.1f}%)')

    txt = '\n'.join(lines) + '\n'
    print(txt)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(f'报告已写入: {OUT}')


if __name__ == '__main__':
    main()
