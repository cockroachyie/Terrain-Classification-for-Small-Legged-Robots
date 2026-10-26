"""Build the 2-page project write-up PDF. Edit TEAM / REPO and re-run."""
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

TEAM = "[Name] (PES1UG24CS126) &nbsp;·&nbsp; [Teammate name] ([Teammate SRN])"
REPO = "github.com/Fickleimagination/terrain-tactile-cnn (private)"
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parent / "ML_MiniProject_Writeup.pdf")

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=9.3, leading=12.2, spaceAfter=4)
h = ParagraphStyle("h", parent=body, fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=5, spaceAfter=2,
                   textColor=colors.HexColor("#1F3A5F"))
title = ParagraphStyle("t", parent=body, fontName="Helvetica-Bold", fontSize=14, leading=17, alignment=TA_CENTER, spaceAfter=2)
sub = ParagraphStyle("s", parent=body, fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=1)
cap = ParagraphStyle("c", parent=body, fontSize=8, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
cell = ParagraphStyle("cell", parent=body, fontSize=8.6, leading=10.5, spaceAfter=0)
bullet = ParagraphStyle("bl", parent=body, leftIndent=10, bulletIndent=1, spaceAfter=2)

P = lambda t, s=body: Paragraph(t, s)  # noqa: E731
B = lambda t: Paragraph(t, bullet, bulletText="•")  # noqa: E731

story = [
    P("Terrain Classification for Small Legged Robots from Tactile Data", title),
    P("UE24CS352A Machine Learning — Mini-Project Write-up", sub),
    P(TEAM, sub),
    P(f"Code: {REPO}", sub),
    Spacer(1, 4),

    P("1. Problem statement", h),
    P("Small legged robots move very differently on hard floors, slippery tiles, grass or sand. If the robot knows "
      "what it is walking on, it can change its gait and save energy or avoid slipping. SAIL-R, a small hexapod from "
      "Stanford, has no camera for this, but its C-shaped front legs carry a capacitive tactile array. Following Choi "
      "&amp; Thomasson, we predict the terrain from the tactile signal of a <b>single step</b>, at two levels: the "
      "<b>8 individual terrains</b> and <b>4 terrain classes</b> (high-friction HF, low-friction LF, deformable D, "
      "granular G). The question is whether a 1D CNN on the raw time series beats the SVM on hand-made features used "
      "in earlier SAIL-R work, and which hand-made features are still worth keeping."),

    P("2. Dataset", h),
    P("The SAIL-R tactile dataset (Wu, Huh et al., Stanford BDML) has <b>5,990 steps</b>. Each step contains:"),
    B("<b>Raw signal</b>: 6 channels × 69 samples at 217 Hz (5 normal-force taxels + 1 shear taxel), 414 values."),
    B("<b>33 hand-designed features</b>: peak, area and mean of the summed normal force; per-taxel peaks, ratios "
      "and means; shear min/max/mean; motor RPM and current at peak force; and 3 gait <i>control inputs</i> "
      "(motor RPM, Ts/Tc ratio, ω<sub>slow</sub>) that cannot be recovered from the tactile signal."),
    B("<b>Labels</b>: asphalt (HF); lab floor, laminate wood (LF); cork, grass, gravel, pebble (D); sand (G). "
      "Terrains are balanced (~750 steps each), but the classes are not: D holds half the data and HF only 13%."),
    P("We use a stratified 70/15/15 train/validation/test split. All normalisation statistics come from the "
      "training set only. Every experiment is repeated on 3 random splits (seeds 42, 7, 2026)."),

    P("3. Approach", h),
    B("<b>Baseline</b> — SVM with a Pearson VII (PUK) kernel on the 33 standardised features, the same setup as "
      "the earlier SAIL-R papers. C and σ are tuned on the validation set."),
    B("<b>1D CNN</b> — input 6 × 69. Four blocks of Conv1d (kernel 5, stride 1, zero padding) → ReLU → MaxPool(2), "
      "with 24 channels each, then dropout 0.5. Two fully-connected layers (hidden 128, ReLU) give the prediction."),
    B("<b>Feature fusion</b> — hand-made features are concatenated with the learned CNN features before the FC "
      "layers. Three variants: <i>Raw Data</i> (no extra features), <i>Raw + Control Inputs</i> (3 gait "
      "parameters) and <i>Raw + All Features</i> (all 33)."),
    P("Each model is trained twice, once for the 4 classes and once for the 8 terrains. Training uses cross-entropy "
      "loss, Adam (lr 10<super>-3</super>), batch 64, up to 200 epochs, and keeps the epoch with the best validation "
      "accuracy (early stopping, patience 50). The paper used lr 10<super>-4</super>; with that setting our "
      "validation accuracy was still climbing at 300 epochs, so we raised it."),
]

story += [
    P("4. Implementation", h),
    P("Python, PyTorch and scikit-learn. The code is a small package (<font face='Courier'>src/terrain_cls</font>): "
      "<font face='Courier'>data.py</font> loads the CSV, reshapes the 414 values into 6 × 69, splits and normalises; "
      "<font face='Courier'>models.py</font> has the CNN and a precomputed-kernel PUK SVM; "
      "<font face='Courier'>train.py</font> has the training loop and metrics. "
      "<font face='Courier'>scripts/run_all.py</font> runs all 8 experiments and saves the results table, confusion "
      "matrices, training curves and model checkpoints (~8 min per seed on a laptop CPU). "
      "<font face='Courier'>scripts/demo.py</font> loads the saved models and classifies held-out steps; its "
      "<font face='Courier'>--stream</font> mode walks the robot across all 8 terrains step by step."),

    P("5. Results", h),
]

rows = [
    ["Model", "Class (4)", "Terrain (8)", "Paper class", "Paper terrain"],
    ["SVM baseline (PUK, 33 features)", "91.9 ± 0.4", "76.3 ± 1.7", "84", "70"],
    ["CNN — Raw Data", "86.4 ± 0.5", "67.1 ± 0.9", "87", "69"],
    ["CNN — Raw + Control Inputs", "90.5 ± 1.5", "73.6 ± 1.5", "93", "76"],
    ["CNN — Raw + All Features", "92.6 ± 1.2", "79.6 ± 0.6", "97", "80"],
]
t = Table(rows, colWidths=[6.0 * cm, 2.6 * cm, 2.6 * cm, 2.4 * cm, 2.6 * cm])
t.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, -1), "Helvetica", 8.6),
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.6),
    ("FONT", (0, 4), (2, 4), "Helvetica-Bold", 8.6),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E3EAF3")),
    ("TEXTCOLOR", (3, 1), (4, -1), colors.HexColor("#666666")),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#1F3A5F")),
    ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#1F3A5F")),
    ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
]))
story += [t, P("Table 1. Test accuracy (%), mean ± std over 3 splits. Paper numbers shown for reference.", cap),
           Spacer(1, 4)]

from PIL import Image as PILImage


def fig(path, width):
    w, hh = PILImage.open(path).size
    return Image(str(path), width=width, height=width * hh / w)


img1 = fig(RES / "cm_raw_all_class.png", 6.6 * cm)
img2 = fig(RES / "cm_raw_all_terrain.png", 8.4 * cm)
figs = Table([[img1, img2]], colWidths=[8.2 * cm, 9.0 * cm])
figs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
story += [figs, P("Figure 1. Normalised confusion matrices of the best model (Raw + All Features), seed 42: "
                  "4 classes (left) and 8 terrains (right).", cap), Spacer(1, 3)]

story += [
    B("Our CNN numbers come out close to the paper's: Raw 86/67 vs 87/69, Raw + All 93/80 vs 97/80."),
    B("<b>Fusing hand-made features helps most.</b> Adding all 33 lifts terrain accuracy from 67% to 80%. "
      "Just the 3 gait control inputs give +6 points on terrain and +4 on class. The same terrain looks different "
      "at a different gait, and the CNN cannot learn the gait from the tactile signal alone."),
    B("<b>HF vs LF is the hardest split</b>, as in the paper. Asphalt (μ ≈ 1.0) gets confused with lab floor and "
      "laminate (μ ≈ 0.5), and HF has half as many samples as LF. Sand and grass are easy (98%, 87%)."),
    B("Our tuned SVM is stronger than the paper's (92/76 vs 84/70) and beats the raw-only CNN. Raw + All Features "
      "is still the best model on both tasks, with the lowest variance on terrain."),

    P("6. Conclusions", h),
    P("A 1D CNN on one step of tactile data classifies terrain class with ~93% accuracy and the exact terrain with "
      "~80%. The learned features alone are not enough. The CNN only beats a well-tuned SVM once it is also given "
      "the hand-made features, and the gait control inputs matter most because they cannot be inferred from touch. "
      "<b>Limitations:</b> the split is random over steps, so steps from the same run can land in both train and "
      "test (the dataset has no run IDs). The model also needs one complete step before it can predict. "
      "<b>Next steps:</b> a run-wise split, higher-rate shear sensing to separate HF from LF, a single model that "
      "predicts class and terrain together, and sequence models (GRU/Transformer) for mid-step prediction."),
    Spacer(1, 3),
    P("<b>Reference.</b> H. Choi, R. Thomasson, <i>Terrain Classification for Small Legged Robots Using Deep "
      "Learning on Tactile Data</i>, Stanford CS229. X. A. Wu et al., <i>Tactile Sensing and Terrain-Based Gait "
      "Control for Small Legged Robots</i>, IEEE T-RO, 2019.", ParagraphStyle("r", parent=body, fontSize=8, leading=10)),
]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.4 * cm,
                        bottomMargin=1.3 * cm, title="Terrain Classification from Tactile Data — Write-up",
                        author="UE24CS352A Mini-Project")
doc.build(story)
print("wrote", OUT)
