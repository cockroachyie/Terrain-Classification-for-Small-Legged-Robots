Seeds: [42, 7, 2026] | epochs ≤ 200 | lr 0.001 | test split 15% (stratified)

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
