"""Loading, splitting and normalising the tactile dataset."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


@dataclass
class Dataset:
    raw: np.ndarray        # (N, 6, 69) float32 tactile time series
    feats: np.ndarray      # (N, 33)   float32 hand-designed features
    terrain: np.ndarray    # (N,) int  index into C.TERRAINS
    cls: np.ndarray        # (N,) int  index into C.CLASSES

    def __len__(self):
        return len(self.terrain)

    def subset(self, idx):
        return Dataset(self.raw[idx], self.feats[idx], self.terrain[idx], self.cls[idx])

    def labels(self, target):
        return self.terrain if target == "terrain" else self.cls


def load(path=C.DATA_CSV) -> Dataset:
    df = pd.read_csv(path, header=None)
    values = df.iloc[:, : C.N_RAW + C.N_FEATS].to_numpy(dtype=np.float32)
    names = df.iloc[:, C.N_RAW + C.N_FEATS].astype(str).str.strip().to_numpy()
    raw = values[:, : C.N_RAW].reshape(-1, C.N_CHANNELS, C.N_TIMESTEPS)  # channel-major layout
    feats = values[:, C.N_RAW:]
    terrain = np.array([C.TERRAINS.index(n) for n in names])
    cls = np.array([C.CLASSES.index(C.TERRAIN_TO_CLASS[n]) for n in names])
    return Dataset(raw, feats, terrain, cls)


def split(ds: Dataset, seed=42, val=0.15, test=0.15):
    """Stratified (by terrain) train/val/test split. Returns three Datasets."""
    idx = np.arange(len(ds))
    tr, tmp = train_test_split(idx, test_size=val + test, stratify=ds.terrain, random_state=seed)
    va, te = train_test_split(tmp, test_size=test / (val + test), stratify=ds.terrain[tmp], random_state=seed)
    return ds.subset(tr), ds.subset(va), ds.subset(te)


class Normalizer:
    """Per-channel z-score for raw signals, per-column z-score for features. Fit on train only."""

    def fit(self, ds: Dataset):
        self.raw_mu = ds.raw.mean(axis=(0, 2), keepdims=True)
        self.raw_sd = ds.raw.std(axis=(0, 2), keepdims=True) + 1e-6
        self.f_mu = ds.feats.mean(0, keepdims=True)
        self.f_sd = ds.feats.std(0, keepdims=True) + 1e-6
        return self

    def raw(self, x):
        return ((x - self.raw_mu) / self.raw_sd).astype(np.float32)

    def feats(self, f):
        return ((f - self.f_mu) / self.f_sd).astype(np.float32)

    def state_dict(self):
        return {k: getattr(self, k) for k in ("raw_mu", "raw_sd", "f_mu", "f_sd")}

    @classmethod
    def from_state(cls, state):
        n = cls()
        for k, v in state.items():
            setattr(n, k, np.asarray(v, dtype=np.float32))
        return n
