# Code archive — Frame-level posture categories lack evidential support in *Caenorhabditis elegans*

Analysis code for the manuscript. Python 3.10+; dependencies:
`numpy scipy scikit-learn opencv-contrib-python matplotlib pandas imbalanced-learn statsmodels scikit-image`.

All scripts are self-contained within this archive: every local module
imported by any script (`worm_labeler`, `wormpath`, `step14_continuity_crossval`,
etc.) is included, and every script referenced by `run_full_v17.sh`,
`run_wormcrops_full.sh` or the Supporting Information is present. The two
annotation interfaces actually used in the study are archived
(`human_label_gui_v2.py` for the main three-rater experiment,
`human_label_gui_v3.py` for the cross-dataset blind annotation); they require
a local display (OpenCV windows) and cannot run headless.

## Pipeline (in-house dataset, 4,055,924 frames / 6,028 worms / 21 days)

```
bash run_full_v17.sh          # 12-step pipeline, checkpointed (logs_v17/*.done)
```

Key steps:
| Step | Script | Output |
|---|---|---|
| Skeleton/curvature + atlas | `cluster_atlas.py` | `kappa_cache.npz`, `atlas_worm_hist.csv` |
| Four independent pipelines | `step15_independent_shape.py` | `indep/indep_*.csv` |
| Outline pseudo-structure check | `step20_outline_check.py` | `indep/outline_check.txt` |
| Rule-based pseudo labels | `simulate_labels.py` | `simulated_labels.csv` |
| SBPD/fPCA + 4-feature SVM (Tables 2–3) | `step8_train_svm.py` | `table2*.csv`, `table3*.csv` |
| Grouped cross-validation | `step9_group_cv.py` | `group_cv_results.csv` |
| Human-annotation agreement | `step18_agreement_v2.py` | agreement stats |
| Full-data Table 1 (4,055,924 frames) | `step19_theta_full.py` → `t1_fulldata.py` | `t1_fulldata.csv` |
| Rule-threshold sensitivity | `t1b_rule_sensitivity.py` | `t1b_rule_sens.csv` |
| PCA-in-fold robustness | `t1c_pca_infold.py` | `t1c_pca_infold.csv` |

## Cross-dataset validation (public worm image crops)

```
python prep_wormcrops_5dim.py --src worm_crops --out wormcrops_5dim
bash run_wormcrops_full.sh    # 6-step core chain, checkpointed (logs_wormcrops/*.done)
python test_skeleton_worm_crops.py --n 300   # extraction benchmark (89.3% success)
python wc_binary_kappa.py    # human-rule vs human-human binary κ (Table 7 / text)
```

## Notes

- Data root is resolved by `wormpath.get_data_root()` (override with `WORM_DATA_ROOT`).
- `worm_labeler.extract_skeleton_full` includes the 2–98 percentile contrast
  stretch, Otsu-based threshold ladder, low-end flood rejection, and the
  two-worm crossing filter (width-ratio and uncovered-arm criteria).
- All clustering decision criteria were preregistered before implementation
  (see manuscript Methods).
