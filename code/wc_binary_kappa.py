# -*- coding: utf-8 -*-
"""wc_binary_kappa.py — 跨数据集二值 κ 补充（Table 7 增补）

口径与主数据一致：二值 0=笔直(straight)、1=弯(omega/sinusoidal)。
只取给出三类有效作答的项（label_code ∈ {1,2,3}）。
输出：
  人–规则 二值 κ（R1、R2）
  人–人   二值 κ
"""
import csv
import os

from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
WC = os.path.join(HERE, 'annotation_wc')
CODE_TO_CLASS = {'1': 2, '2': 1, '3': 0}      # 与 analyze_wc_agreement 一致


def bin_of_class(c):
    return 0 if c == 2 else 1                  # straight→0, bent→1


def load_labels(path):
    out = {}
    with open(path, encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            out[row['item_id']] = row['label_code']
    return out


def load_key():
    out = {}
    with open(os.path.join(WC, 'key.csv'), encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            out[row['item_id']] = int(row['rule_label'])
    return out


def main():
    key = load_key()
    r1 = load_labels(os.path.join(WC, 'labels_R1.csv'))
    r2 = load_labels(os.path.join(WC, 'labels_R2.csv'))

    def human3(codes):
        return {i: CODE_TO_CLASS[c] for i, c in codes.items()
                if c in CODE_TO_CLASS}

    h1, h2 = human3(r1), human3(r2)
    common = sorted(set(h1) & set(h2))
    lines = []

    for name, h in (('R1', h1), ('R2', h2)):
        ids = sorted(set(h) & set(key))
        yh = [bin_of_class(h[i]) for i in ids]
        yr = [bin_of_class(key[i]) for i in ids]
        k = cohen_kappa_score(yh, yr)
        agree = sum(a == b for a, b in zip(yh, yr)) / len(ids)
        cm = confusion_matrix(yr, yh, labels=[0, 1])
        line = (f'人–规则 二值 κ ({name}) = {k:+.3f}  (n={len(ids)}, '
                f'一致率 {100*agree:.1f}%)  混淆矩阵[规则0直,1弯; 列=人]\n'
                f'  {cm.tolist()}')
        print(line)
        lines.append(line)

    yh1 = [bin_of_class(h1[i]) for i in common]
    yh2 = [bin_of_class(h2[i]) for i in common]
    k = cohen_kappa_score(yh1, yh2)
    agree = sum(a == b for a, b in zip(yh1, yh2)) / len(common)
    cm = confusion_matrix(yh2, yh1, labels=[0, 1])
    line = (f'人–人 二值 κ = {k:+.3f}  (n={len(common)}, '
            f'一致率 {100*agree:.1f}%)  混淆矩阵[R2行, R1列]\n  {cm.tolist()}')
    print(line)
    lines.append(line)

    out = os.path.join(WC, 'binary_kappa_report.txt')
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('saved:', out)


if __name__ == '__main__':
    main()
