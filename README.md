# Terrain Classification for Small Legged Robots from Tactile Data

**UE24CS352A Machine Learning – Mini-Project**
Re-implementation of Choi & Thomasson, *"Terrain Classification for Small Legged Robots Using Deep Learning on Tactile Data"* (Stanford CS229).

A small legged robot (SAIL-R) has a capacitive tactile array on its C-shaped legs: 5 normal-force taxels + 1 shear taxel, sampled at 217 Hz. From the signal of **one step** we predict:

- the **terrain** (8 classes): asphalt, lab floor, laminate wood, cork, grass, gravel, pebble, sand
- the higher-level **terrain class** (4 classes): High-Friction, Low-Friction, Deformable, Granular

We compare a **PUK-kernel SVM** on 33 hand-designed features (prior work on SAIL-R) with a **1D CNN** on the raw 6 × 69 time series, with and without hand-designed features fused in.

## Results

Test accuracy, mean ± std over 3 random splits (seeds 42, 7, 2026). Paper numbers for reference.

| Model | Class (ours) | Terrain (ours) | Class (paper) | Terrain (paper) |
|---|---|---|---|---|
| SVM Baseline (PUK, 33 feats) | 91.9% ± 0.4 | 76.3% ± 1.7 | 84% | 70% |
| Raw Data | 86.4% ± 0.5 | 67.1% ± 0.9 | 87% | 69% |
| Raw Data + All Features | 92.6% ± 1.2 | 79.6% ± 0.6 | 97% | 80% |
| Raw Data + Control Input Features | 90.5% ± 1.5 | 73.6% ± 1.5 | 93% | 76% |

Per-class test accuracy (class target, first seed):

| Model | HF | LF | D | G |
|---|---|---|---|---|
| SVM Baseline (PUK, 33 feats) | 72% | 90% | 98% | 96% |
| Raw Data | 59% | 78% | 97% | 96% |
| Raw Data + All Features | 77% | 92% | 98% | 97% |
| Raw Data + Control Input Features | 75% | 89% | 98% | 98% |

**Takeaways**

- The CNN variants land close to the paper (Raw 86/67 vs 87/69, Raw+All 93/80 vs 97/80, Raw+Ctrl 91/74 vs 93/76).
- Fusing hand-designed features helps most; just the 3 gait-parameter inputs (not inferable from the tactile signal) recover most of that gain.
- High-friction vs low-friction is the hardest split, as in the paper (HF has half the samples of LF and μ differs only 1.0 vs 0.5).
- Our SVM is stronger than the paper's (92/76 vs 84/70) because C and σ were tuned on the validation set; with tuning it beats the raw-only CNN, and Raw + All Features is the best model for both tasks.
- Caveat: the split is random over steps, so steps from the same run can sit in train and test. A run-wise split would be stricter but run IDs are not in the dataset.


Figures in `results/`: confusion matrices (`cm_<model>_<target>.png`), training curves, a sample step from each terrain (`step_signals.png`).

## Repository layout

```
data/                       dataset CSV + column description
src/terrain_cls/
  config.py                 constants: terrains, class mapping, 33 feature names, feature sets
  data.py                   loading, stratified 70/15/15 split, train-only normalisation
  models.py                 TactileCNN (1D CNN + feature fusion), PUK kernel SVM
  train.py                  training loops, early stopping on val accuracy, metrics
  plots.py                  confusion matrices, training curves, signal plots
scripts/
  run_all.py                runs every experiment, writes results/ and checkpoints/
  demo.py                   live demo on held-out steps using saved checkpoints
results/                    summary.md, metrics.json, figures
checkpoints/                trained CNN weights (one per variant × target)
```

## Setup

Python 3.10+. Single terminal:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Run

Everything (SVM + 3 CNN variants × 2 targets), paper hyper-parameters:

```bash
python scripts/run_all.py
```

Options: `--seeds 42 7 2026` (mean ± std over splits), `--epochs 200`, `--lr 1e-3`, `--skip-svm`, `--quick` (5-epoch smoke test, ~30 s).
On a laptop CPU one seed takes ~15–20 min.

Demo (needs `checkpoints/` – included, or regenerate with `run_all.py`):

```bash
python scripts/demo.py                     # 10 random held-out steps
python scripts/demo.py --stream            # robot "walks" across all 8 terrains, 6 steps each
python scripts/demo.py --index 42 --plot   # one step + probability plot -> results/demo_step.png
python scripts/demo.py --variant raw       # raw-only CNN instead of Raw + All Features
```

## Method

**Data.** 5,990 steps, balanced across the 8 terrains (~750 each). Each step = 6 channels × 69 samples plus 33 hand-designed features (peak / area / average of the summed normal force, per-taxel peaks and ratios, shear min/max/avg, motor RPM and current at peak, and the gait parameters motor RPM, Ts/Tc ratio and ω_slow). Because HF and G contain one terrain each while D contains four, the 4-class problem is imbalanced (D ≈ 50 %).
Split: stratified 70 / 15 / 15 (train / val / test). Normalisation statistics come from the training set only.

**SVM baseline.** SVC with a Pearson VII Universal Kernel (PUK, as in Wu et al.), on the 33 standardised features. C and σ chosen on the validation set.

**1D CNN.** Input 6 × 69. Four blocks of Conv1d (kernel 5, stride 1, zero padding) → ReLU → MaxPool(2); channels 6 → 24 → 24 → 24 → 24. Dropout 0.5 on the learned features. Optional hand-designed features are concatenated before two fully-connected layers (ReLU, hidden 128). Adam, batch 64, cross-entropy, up to 200 epochs with early stopping (patience 50) on validation accuracy; the best-validation checkpoint is evaluated on test.

*Deviation from the paper:* lr = 1e-3 instead of 1e-4. With 1e-4 validation accuracy was still rising at 300 epochs (the paper notes the same, Sec. 6.3); `--lr 1e-4 --epochs 300` reproduces the paper setting.

| Variant | Extra inputs to FC layers |
|---|---|
| Raw Data | none |
| Raw Data + Control Input Features | 3 gait parameters (motor RPM, Ts/Tc, ω_slow) |
| Raw Data + All Features | all 33 hand-designed features |

## Reference

H. Choi, R. Thomasson. *Terrain Classification for Small Legged Robots Using Deep Learning on Tactile Data.* Stanford CS229.
X. A. Wu, T. M. Huh, A. Sabin, S. A. Suresh, M. R. Cutkosky. *Tactile Sensing and Terrain-Based Gait Control for Small Legged Robots.* IEEE T-RO, 2019.
