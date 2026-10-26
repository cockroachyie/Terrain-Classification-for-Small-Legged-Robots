# Data

`dataSet_withFeatures.csv` — 5,990 rows, one per robot step (SAIL-R tactile dataset, Wu, Huh et al., Stanford BDML).

| Columns | Content |
|---|---|
| 0–413 | Raw tactile time series, channel-major: 6 channels × 69 samples (217 Hz). Ch1–5 = normal taxels, Ch6 = shear taxel |
| 414–446 | 33 hand-designed features (paper Table 1; names in `src/terrain_cls/config.py`) |
| 447 | Terrain label |

`dataSet_withFeatures_labels.csv` — the terrain labels alone (same order).

Terrains (8) and higher-level classes (4):

| Class | Terrains |
|---|---|
| HF – high friction | asphalt |
| LF – low friction | lab, laminate_wood |
| D – deformable | cork, grass, gravel, pebble |
| G – granular | sand |
