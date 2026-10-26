"""Live demo: classify held-out robot steps with the trained CNNs.

    python scripts/demo.py                       # 10 random test steps, Raw+All Features model
    python scripts/demo.py --variant raw --n 5   # raw-only model
    python scripts/demo.py --index 123 --plot    # one specific step + figure (results/demo_step.png)
    python scripts/demo.py --stream              # simulate the robot walking: step-by-step predictions
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from terrain_cls import config as C  # noqa: E402
from terrain_cls import data  # noqa: E402
from terrain_cls.train import load_cnn  # noqa: E402


def predict(model, ckpt, norm, raw, feats):
    xr = torch.from_numpy(norm.raw(raw))
    fi = ckpt["feat_idx"]
    xf = torch.from_numpy(norm.feats(feats)[:, fi]) if fi else torch.zeros(len(raw), 0)
    with torch.no_grad():
        return model(xr, xf).softmax(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="raw_all", choices=list(C.FEATURE_SETS))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--index", type=int, help="index into the test split")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    models = {}
    for target in ("class", "terrain"):
        p = C.CKPT_DIR / f"cnn_{a.variant}_{target}.pt"
        if not p.exists():
            sys.exit(f"Missing {p}. Run `python scripts/run_all.py` first.")
        models[target] = load_cnn(p)

    _, _, test = data.split(data.load(), seed=a.seed)  # same held-out split as training
    rng = np.random.default_rng()
    if a.index is not None:
        idx = np.array([a.index])
    elif a.stream:
        # walk across terrains: 6 consecutive steps per terrain in a random terrain order
        idx = np.concatenate([rng.choice(np.where(test.terrain == t)[0], 6, replace=False)
                              for t in rng.permutation(len(C.TERRAINS))])
    else:
        idx = rng.choice(len(test), a.n, replace=False)
    sub = test.subset(idx)

    p_cls = predict(*models["class"], sub.raw, sub.feats)
    p_ter = predict(*models["terrain"], sub.raw, sub.feats)
    print(f"Model: {C.MODEL_LABELS[a.variant]}\n")
    print(f"{'#':>5} | {'true terrain':<14}{'pred terrain':<16}{'conf':>5} | {'true':<5}{'pred':<5}{'conf':>6}")
    print("-" * 68)
    ok_t = ok_c = 0
    for k, i in enumerate(idx):
        tt, pt = C.TERRAINS[sub.terrain[k]], C.TERRAINS[p_ter[k].argmax()]
        tc, pc = C.CLASSES[sub.cls[k]], C.CLASSES[p_cls[k].argmax()]
        ok_t += tt == pt
        ok_c += tc == pc
        mark = lambda b: "✓" if b else "✗"  # noqa: E731
        print(f"{i:>5} | {tt:<14}{pt:<14}{mark(tt == pt)} {p_ter[k].max():5.2f} | "
              f"{tc:<5}{pc:<4}{mark(tc == pc)} {p_cls[k].max():5.2f}")
        if a.stream:
            time.sleep(0.25)
    print(f"\nterrain {ok_t}/{len(idx)} correct | class {ok_c}/{len(idx)} correct")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = 0
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
        t = np.arange(C.N_TIMESTEPS) / 217.0 * 1000
        for ch in range(C.N_CHANNELS):
            ax1.plot(t, sub.raw[k, ch], ls="-" if ch < 5 else "--", label=f"Ch{ch + 1}" + (" shear" if ch == 5 else ""))
        ax1.set_title(f"Test step #{idx[k]} — true: {C.TERRAINS[sub.terrain[k]]}")
        ax1.set_xlabel("time in step (ms)")
        ax1.legend(fontsize=7)
        ax2.barh(C.TERRAINS, p_ter[k], color=["tab:green" if j == sub.terrain[k] else "tab:gray"
                                               for j in range(len(C.TERRAINS))])
        ax2.set_xlim(0, 1)
        ax2.set_title("predicted terrain probability")
        fig.tight_layout()
        out = C.RESULTS_DIR / "demo_step.png"
        fig.savefig(out, dpi=150)
        print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
