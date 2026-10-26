"""Training / evaluation loops for the CNN variants and the SVM baseline."""
import itertools
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from . import config as C
from .data import Normalizer
from .models import PukSVM, TactileCNN


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _names(target):
    return C.TERRAINS if target == "terrain" else C.CLASSES


def metrics(y_true, y_pred, target):
    n = len(_names(target))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    per_class = (cm.diagonal() / cm.sum(1).clip(min=1)).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_acc": dict(zip(_names(target), per_class)),
        "confusion": cm.tolist(),
    }


# ----------------------------------------------------------------------------- CNN
def _tensors(ds, norm, feat_idx, target):
    raw = torch.from_numpy(norm.raw(ds.raw))
    extra = torch.from_numpy(norm.feats(ds.feats)[:, feat_idx]) if feat_idx else torch.zeros(len(ds), 0)
    y = torch.from_numpy(ds.labels(target)).long()
    return raw, extra, y


@torch.no_grad()
def predict_cnn(model, raw, extra, batch=1024):
    model.eval()
    out = [model(raw[i:i + batch], extra[i:i + batch]).softmax(1) for i in range(0, len(raw), batch)]
    return torch.cat(out).numpy()


def train_cnn(train, val, test, variant, target, epochs=300, lr=1e-4, batch=64, patience=60,
              seed=42, verbose=True):
    """Train one CNN variant. Model selection = best validation accuracy (early stopping)."""
    set_seed(seed)
    feat_idx = C.FEATURE_SETS[variant]
    norm = Normalizer().fit(train)
    tr, va, te = (_tensors(d, norm, feat_idx, target) for d in (train, val, test))
    n_out = len(_names(target))
    model = TactileCNN(n_out, n_extra=len(feat_idx))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(*tr), batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    hist = {"train_acc": [], "val_acc": [], "train_loss": []}
    best, best_state, best_ep, t0 = -1, None, 0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xr, xf, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(xr, xf), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        tr_acc = (predict_cnn(model, tr[0], tr[1]).argmax(1) == tr[2].numpy()).mean()
        va_acc = (predict_cnn(model, va[0], va[1]).argmax(1) == va[2].numpy()).mean()
        hist["train_acc"].append(float(tr_acc))
        hist["val_acc"].append(float(va_acc))
        hist["train_loss"].append(tot / len(tr[2]))
        if va_acc > best:
            best, best_ep = va_acc, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 25 == 0 or ep == 1):
            print(f"  [{variant}/{target}] ep {ep:3d} loss {tot / len(tr[2]):.3f} "
                  f"train {tr_acc:.3f} val {va_acc:.3f} (best {best:.3f}@{best_ep}) {time.time() - t0:.0f}s")
        if ep - best_ep >= patience:
            break

    model.load_state_dict(best_state)
    y_pred = predict_cnn(model, te[0], te[1]).argmax(1)
    res = metrics(te[2].numpy(), y_pred, target)
    res.update(best_epoch=best_ep, val_accuracy=float(best), history=hist)
    ckpt = {"state_dict": model.state_dict(), "variant": variant, "target": target,
            "feat_idx": feat_idx, "n_out": n_out, "normalizer": norm.state_dict()}
    return model, res, ckpt


def load_cnn(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TactileCNN(ckpt["n_out"], n_extra=len(ckpt["feat_idx"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt, Normalizer.from_state(ckpt["normalizer"])


# ----------------------------------------------------------------------------- SVM
SVM_GRID = {"C": [1, 10, 100, 1000], "sigma": [1.0, 3.0, 10.0, 30.0], "omega": [1.0]}


def train_svm(train, val, test, target, grid=SVM_GRID, verbose=True):
    """PUK-kernel SVM on the 33 hand-designed features; hyper-params picked on validation."""
    norm = Normalizer().fit(train)
    Xtr, Xva, Xte = (norm.feats(d.feats) for d in (train, val, test))
    ytr, yva, yte = (d.labels(target) for d in (train, val, test))
    best, best_p = -1, None
    for Cv, s, w in itertools.product(grid["C"], grid["sigma"], grid["omega"]):
        acc = accuracy_score(yva, PukSVM(Cv, s, w).fit(Xtr, ytr).predict(Xva))
        if verbose:
            print(f"  [svm/{target}] C={Cv} sigma={s} omega={w} val {acc:.3f}")
        if acc > best:
            best, best_p = acc, (Cv, s, w)
    svm = PukSVM(*best_p).fit(Xtr, ytr)
    res = metrics(yte, svm.predict(Xte), target)
    res.update(val_accuracy=float(best), params=svm.get_params())
    return svm, res


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
