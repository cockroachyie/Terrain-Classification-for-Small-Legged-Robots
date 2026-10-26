"""1D-CNN terrain classifier (Choi & Thomasson) and PUK-kernel SVM baseline."""
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.svm import SVC

from . import config as C


class TactileCNN(nn.Module):
    """4 x [Conv1d(k=5, stride 1, zero pad) -> ReLU -> MaxPool(2)], 6->24->24->24->24 channels,
    dropout 0.5, optional hand-designed features concatenated before two FC layers."""

    def __init__(self, n_out, n_extra=0, width=24, hidden=128, dropout=0.5):
        super().__init__()
        layers, c_in = [], C.N_CHANNELS
        for _ in range(4):
            layers += [nn.Conv1d(c_in, width, kernel_size=5, stride=1, padding=2),
                       nn.ReLU(), nn.MaxPool1d(2, ceil_mode=True)]
            c_in = width
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            self.n_learned = self.features(torch.zeros(1, C.N_CHANNELS, C.N_TIMESTEPS)).numel()
        self.n_extra = n_extra
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(self.n_learned + n_extra, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, raw, extra=None):
        z = self.dropout(self.features(raw).flatten(1))
        if self.n_extra:
            z = torch.cat([z, extra], dim=1)
        return self.head(z)


def puk_kernel(A, B, sigma=1.0, omega=1.0):
    """Pearson VII Universal Kernel (Üstün et al. 2006), as used by Weka / Wu et al. for SAIL-R."""
    d = cdist(A, B, "euclidean")
    return 1.0 / (1.0 + (2.0 * d * np.sqrt(2.0 ** (1.0 / omega) - 1.0) / sigma) ** 2) ** omega


class PukSVM:
    def __init__(self, C=10.0, sigma=1.0, omega=1.0):
        self.C, self.sigma, self.omega = C, sigma, omega

    def fit(self, X, y):
        self.X_ = X
        self.svc = SVC(C=self.C, kernel="precomputed")
        self.svc.fit(puk_kernel(X, X, self.sigma, self.omega), y)
        return self

    def predict(self, X):
        return self.svc.predict(puk_kernel(X, self.X_, self.sigma, self.omega))

    def get_params(self):
        return {"C": self.C, "sigma": self.sigma, "omega": self.omega}
