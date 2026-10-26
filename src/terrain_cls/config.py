"""Central constants for the SAIL-R tactile terrain dataset."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"
DATA_CSV = DATA_DIR / "dataSet_withFeatures.csv"

N_CHANNELS = 6          # 5 normal taxels + 1 shear taxel (channel 6)
N_TIMESTEPS = 69        # samples per step (AD7147 CDC @ 217 Hz)
N_RAW = N_CHANNELS * N_TIMESTEPS  # 414
N_FEATS = 33            # hand-designed features (paper Table 1)

TERRAINS = ["asphalt", "lab", "laminate_wood", "cork", "grass", "gravel", "pebble", "sand"]

# Higher-level classes (paper Sec. 3.1.2): HF has one terrain, LF two, D four, G one.
CLASSES = ["HF", "LF", "D", "G"]
TERRAIN_TO_CLASS = {
    "asphalt": "HF",
    "lab": "LF", "laminate_wood": "LF",
    "cork": "D", "grass": "D", "gravel": "D", "pebble": "D",
    "sand": "G",
}

# Names of the 33 hand-designed features, in column order (paper Table 1).
FEATURE_NAMES = (
    ["peak_sum", "auc_sum", "gait_motor_rpm", "gait_ts_tc_ratio", "avg_sum", "gait_omega_slow"]
    + [f"taxel{i}_peak" for i in range(1, 6)]
    + [f"taxel{i}_peak_ratio" for i in range(1, 6)]
    + ["shear_min", "shear_max", "shear_avg", "rpm_at_peak_sum", "current_at_peak_sum",
       "current_avg", "current_range"]
    + [f"taxel{i}_avg" for i in range(1, 6)]
    + [f"rpm_at_taxel{i}_peak" for i in range(1, 6)]
)
assert len(FEATURE_NAMES) == N_FEATS

# Control inputs (gait parameters) — not inferable from raw tactile data.
CONTROL_IDX = [2, 3, 5]

# Feature sets used by the CNN variants.
FEATURE_SETS = {
    "raw": [],
    "raw_ctrl": CONTROL_IDX,
    "raw_all": list(range(N_FEATS)),
}
MODEL_LABELS = {
    "svm": "SVM Baseline (PUK, 33 feats)",
    "raw": "Raw Data",
    "raw_all": "Raw Data + All Features",
    "raw_ctrl": "Raw Data + Control Input Features",
}
