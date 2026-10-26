"""Reproduce the full experiment: SVM baseline + 3 CNN variants, for class (4) and terrain (8) targets.

    python scripts/run_all.py                    # paper settings, 1 seed
    python scripts/run_all.py --seeds 0 1 2      # mean ± std over 3 splits
    python scripts/run_all.py --quick            # smoke test (few epochs)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from terrain_cls import config as C  # noqa: E402
from terrain_cls import data, plots  # noqa: E402
from terrain_cls.train import save_json, train_cnn, train_svm  # noqa: E402

TARGETS = ("class", "terrain")
VARIANTS = ("raw", "raw_all", "raw_ctrl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam LR (paper: 1e-4; 1e-3 converges within 200 epochs)")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--skip-svm", action="store_true")
    ap.add_argument("--quick", action="store_true", help="5 epochs, tiny SVM grid")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.patience = 5, 5
    torch.set_num_threads(max(1, torch.get_num_threads()))

    C.RESULTS_DIR.mkdir(exist_ok=True)
    C.CKPT_DIR.mkdir(exist_ok=True)
    ds = data.load()
    print(f"Loaded {len(ds)} steps | raw {ds.raw.shape[1:]} | {ds.feats.shape[1]} hand features")
    plots.step_signals(ds, C.RESULTS_DIR / "step_signals.png")

    runs = {}   # runs[model][target] -> list of per-seed results
    histories = {}
    t0 = time.time()
    for si, seed in enumerate(a.seeds):
        tr, va, te = data.split(ds, seed=seed)
        print(f"\n=== seed {seed}: train {len(tr)} / val {len(va)} / test {len(te)} ===")
        for target in TARGETS:
            if not a.skip_svm:
                grid = {"C": [10], "sigma": [10.0], "omega": [1.0]} if a.quick else None
                svm, res = train_svm(tr, va, te, target, **({"grid": grid} if grid else {}), verbose=False)
                print(f"SVM/{target}: test acc {res['accuracy']:.3f}  params {res['params']}")
                runs.setdefault("svm", {}).setdefault(target, []).append(res)
            for v in VARIANTS:
                _, res, ckpt = train_cnn(tr, va, te, v, target, epochs=a.epochs, lr=a.lr,
                                         patience=a.patience, seed=seed, verbose=True)
                print(f"CNN {v}/{target}: test acc {res['accuracy']:.3f} (best epoch {res['best_epoch']})")
                runs.setdefault(v, {}).setdefault(target, []).append(res)
                if si == 0:
                    torch.save(ckpt, C.CKPT_DIR / f"cnn_{v}_{target}.pt")
                    histories[f"{v}/{target}"] = res["history"]

    # ---- figures (first seed)
    for model, by_t in runs.items():
        for target, lst in by_t.items():
            names = C.TERRAINS if target == "terrain" else C.CLASSES
            plots.confusion(lst[0]["confusion"], names, f"{C.MODEL_LABELS[model]} ({target})",
                            C.RESULTS_DIR / f"cm_{model}_{target}.png")
    if histories:
        plots.training_curves(histories, C.RESULTS_DIR / "training_curves.png")

    # ---- summary table
    def fmt(lst):
        acc = np.array([r["accuracy"] for r in lst]) * 100
        return f"{acc.mean():.1f}%" + (f" ± {acc.std():.1f}" if len(acc) > 1 else "")

    paper = {"svm": ("84%", "70%"), "raw": ("87%", "69%"), "raw_all": ("97%", "80%"), "raw_ctrl": ("93%", "76%")}
    lines = [f"Seeds: {a.seeds} | epochs ≤ {a.epochs} | lr {a.lr} | test split 15% (stratified)\n",
             "| Model | Class (ours) | Terrain (ours) | Class (paper) | Terrain (paper) |",
             "|---|---|---|---|---|"]
    for m in ("svm", "raw", "raw_all", "raw_ctrl"):
        if m in runs:
            lines.append(f"| {C.MODEL_LABELS[m]} | {fmt(runs[m]['class'])} | {fmt(runs[m]['terrain'])} "
                         f"| {paper[m][0]} | {paper[m][1]} |")
    # per-class accuracy for the class models
    lines += ["", "Per-class test accuracy (class target, first seed):", "",
              "| Model | " + " | ".join(C.CLASSES) + " |", "|---" * (len(C.CLASSES) + 1) + "|"]
    for m in runs:
        pc = runs[m]["class"][0]["per_class_acc"]
        lines.append(f"| {C.MODEL_LABELS[m]} | " + " | ".join(f"{pc[c] * 100:.0f}%" for c in C.CLASSES) + " |")
    table = "\n".join(lines)
    (C.RESULTS_DIR / "summary.md").write_text(table + "\n")
    for by_t in runs.values():
        for lst in by_t.values():
            for r in lst:
                r.pop("history", None)
    save_json({"args": vars(a), "runs": runs}, C.RESULTS_DIR / "metrics.json")
    print("\n" + table)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results -> {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
