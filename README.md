# Frame-level posture categories lack evidential support in *Caenorhabditis elegans*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24393307.svg)](https://doi.org/10.5281/zenodo.24393307)

Analysis code, annotations and result tables for the manuscript:

> **No evidence for discrete frame-level posture categories in *Caenorhabditis elegans*: a continuous bending-depth readout.**
> Jinfang Weng, Qizhao Lin (corresponding: linqizhao@fjpit.edu.cn)

The premise that single video frames can be sorted into named posture
categories (sinusoidal / omega / straight) is treated as a falsifiable
hypothesis with preregistered decision criteria. Three independent lines of
evidence — blind human annotation, four independent measurement pipelines,
and a cross-dataset check — converge on a negative answer, and a continuous
bending-depth axis (BDI) is provided as the replacement readout.

## Repository layout

```
code/                     all analysis scripts (self-contained; see code/README_SUBMISSION.md)
annotations/main/         raw blind annotations of the main three-rater experiment
                          (labels_A/B/C.csv + retest labels_A2.csv, 310 frames)
annotations/wormcrops/    raw blind annotations of the cross-dataset experiment
                          (labels_R1/R2.csv, 300 frames + key.csv with rule labels)
results/                  derived result tables and logs referenced in the manuscript
```

## Data availability

The in-house imaging corpus (4,055,924 frames from 6,028 worms, 21 imaging
days) exceeds practical Git hosting and is deposited on Zenodo
([DOI: 10.5281/zenodo.24393307](https://doi.org/10.5281/zenodo.24393307),
84 tar archives + SHA256SUMS.txt, 41.9 GB). The public cross-dataset corpus
is WormSwin ([Deserno & Bozek, Sci Rep 13, 11021 (2023)](https://doi.org/10.1038/s41598-023-38213-7)).

All small derived outputs referenced in the text are in `results/`:
full-data Table 1 (`t1_fulldata.csv`), rule-threshold sensitivity
(`t1b_rule_sens.csv`), PCA-in-fold robustness (`t1c_pca_infold.csv`),
atlas histograms, dose tests, frame audit, independent-pipeline outputs.

## Requirements

Python 3.10+; see `requirements.txt`.

```bash
pip install -r requirements.txt
```

The annotation GUIs require a local display (OpenCV windows) and cannot run
headless.

## Reproducing the analyses

All data paths are resolved by `code/wormpath.get_data_root()` — override
with the `WORM_DATA_ROOT` environment variable instead of editing scripts.

| Manuscript item | Entry point |
|---|---|
| Skeleton extraction + κ(s) atlas (full corpus) | `code/cluster_atlas.py` (or `bash code/run_full_v17.sh`, 12-step checkpointed pipeline) |
| Table 1, full-data (4,055,924 frames) | `code/t1_fulldata.py` (+ `code/step19_theta_full.py` for the θ(s) extraction it consumes) |
| Rule-threshold sensitivity | `code/t1b_rule_sensitivity.py` |
| PCA-in-fold robustness | `code/t1c_pca_infold.py` |
| Table 1, 8,000-frame resample (comparison) | `code/step8_train_svm.py`, `code/step9_group_cv.py` |
| Table 2/3, human annotation agreement + bootstrap CIs | `code/make_annotation_batch_v2.py` → GUI `code/human_label_gui_v2.py` → `code/t2_postdrop_three_rater.py`, `code/t2_bootstrap_ci.py` |
| Table 4, four independent pipelines | `code/step15_independent_shape.py`, `code/step20_outline_check.py` |
| Table 5/6, BDI characterization & longitudinal validity | `code/step16_calibrate_axis.py`, `code/step17_posture_profile.py`, `code/step12_age_trajectory.py` |
| Table 7 + binary κ, cross-dataset check | `code/prep_wormcrops_5dim.py` → `bash code/run_wormcrops_full.sh` → `code/make_wc_annotation_batch.py` → GUI `code/human_label_gui_v3.py` → `code/analyze_wc_agreement.py`, `code/wc_binary_kappa.py` |
| Figures | `code/make_plos_figures.py` |

Preregistered clustering criteria (silhouette / density-valley / stability
cuts) are declared in the manuscript Methods and implemented verbatim in the
clustering scripts; run parameters are in Supporting Information S7
(`code/run_full_v17.sh` archives the exact command lines).

## Key results (full corpus)

| Representation | Worm-grouped CV | Frame-random | RBF-50k |
|---|---|---|---|
| BBI (1-D) | 0.761 | 0.762 | 0.765 |
| Eigenworms (6-D) | 0.382 | 0.384 | 0.587 |
| SBPD (3-D) | **0.865** | 0.865 | **0.915** |

Human–human binary κ = 0.648 (96.5% agreement) on the cross-dataset corpus,
while human–rule binary κ = 0.079–0.163: the topological *definitions*
transfer across imaging setups, the calibrated *thresholds* do not.

## License

Code: MIT (see `LICENSE`). The Zenodo data deposit is released under CC-BY-4.0.
