#!/usr/bin/env python3
"""Render the expanded progress-detection / subtask-prediction head diagram.

Outputs (next to this script):
    figure_subtask_progress.pdf   -- vector, use this in the paper
    figure_subtask_progress.png   -- 400 dpi raster preview

Usage:
    python make_figure_subtask_progress.py

Only matplotlib is required; no LaTeX installation is needed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ----------------------------------------------------------------- typography
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,   # embed TrueType so the PDF stays editable
    "ps.fonttype": 42,
})

# --------------------------------------------------------------------- colors
BLUE = (33 / 255, 102 / 255, 172 / 255)
ORANGE = (217 / 255, 95 / 255, 14 / 255)
PURPLE = (140 / 255, 81 / 255, 160 / 255)
INK = "0.25"
LINE = "0.42"


def tint(color, frac):
    """Blend `color` toward white; frac=1 keeps the color, frac=0 gives white."""
    return tuple(1.0 - frac * (1.0 - c) for c in color)


# ------------------------------------------------------------------- geometry
XLIM = (0.55, 15.10)
YLIM = (-14.35, 0.55)

fig_w = 7.28
fig_h = fig_w * (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_aspect("equal")
ax.axis("off")


# -------------------------------------------------------------------- helpers
def box(x, y, w, h, text="", fc="white", ec=LINE, lw=0.8, fs=8.0,
        weight="normal", family=None, z=3, pad=0.07):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={pad}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z,
    ))
    if text:
        kw = {"family": family} if family else {}
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color=INK, weight=weight, zorder=z + 1, linespacing=1.45, **kw)


def token(x, y, fc, ec, s=0.42):
    ax.add_patch(FancyBboxPatch(
        (x - s / 2, y - s / 2), s, s,
        boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=0.6, edgecolor=ec, facecolor=fc, zorder=3,
    ))


def arrow(p, q, color=LINE, lw=0.8):
    ax.annotate("", xy=q, xytext=p, zorder=2, arrowprops=dict(
        arrowstyle="-|>", color=color, linewidth=lw,
        shrinkA=0, shrinkB=0, mutation_scale=8, joinstyle="miter",
    ))


def line(p, q, color=LINE, lw=0.8):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw,
            solid_capstyle="round", zorder=2)


def panel(x0, y0, x1, y1, color):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.16",
        linewidth=0.9, edgecolor=tint(color, 0.45),
        facecolor=tint(color, 0.045), zorder=0,
    ))


# ---------------------------------------------------------------- backbone bar
box(7.875, 0.0, 13.85, 0.80,
    "Final block of the layer-aligned unified transformer   ($\\ell = 29$)",
    fc="0.955", ec="0.55", fs=8.5)

# ----------------------------------------------------- action / VLM token rows
KEEP_FC, KEEP_EC = tint(ORANGE, 0.35), tint(ORANGE, 0.85)
DROP_FC, DROP_EC = "0.90", "0.62"
VLM_FC, VLM_EC = tint(BLUE, 0.30), tint(BLUE, 0.85)

# one state token at position 0 and four register tokens at the end are dropped
action_row = [(2.30, "drop"), (2.78, "keep"), (3.26, "keep"), (3.74, "keep"),
              (4.14, "dots"), (4.54, "keep"),
              (5.02, "drop"), (5.50, "drop"), (5.98, "drop"), (6.46, "drop")]
for x, kind in action_row:
    if kind == "dots":
        ax.text(x, -1.35, "$\\cdots$", ha="center", va="center", fontsize=8,
                color=INK, zorder=4)
    elif kind == "keep":
        token(x, -1.35, KEEP_FC, KEEP_EC)
    else:
        token(x, -1.35, DROP_FC, DROP_EC)
ax.text(1.98, -1.35, "$h_t^A$", ha="right", va="center", fontsize=8.5,
        color=INK)

vlm_row = [(9.60, "tok"), (10.12, "tok"), (10.64, "tok"),
           (11.10, "dots"), (11.56, "tok"), (12.08, "tok")]
for x, kind in vlm_row:
    if kind == "dots":
        ax.text(x, -1.35, "$\\cdots$", ha="center", va="center", fontsize=8,
                color=INK, zorder=4)
    else:
        token(x, -1.35, VLM_FC, VLM_EC)
ax.text(9.28, -1.35, "$h_t^L$", ha="right", va="center", fontsize=8.5,
        color=INK)

arrow((4.38, -0.40), (4.38, -1.13))
arrow((10.84, -0.40), (10.84, -1.13))

# action tokens fan out to both heads; VLM tokens feed only the subtask memory
line((4.38, -1.57), (4.38, -2.30))
line((3.50, -2.30), (8.85, -2.30))
arrow((3.50, -2.30), (3.50, -4.00))
arrow((8.85, -2.30), (8.85, -4.00))

line((10.84, -1.57), (10.84, -2.60))
line((10.84, -2.60), (11.35, -2.60))
arrow((11.35, -2.60), (11.35, -4.00))

# ========================================================= (a) PROGRESS HEAD
panel(0.86, -11.95, 6.16, -3.40, PURPLE)
ax.text(1.10, -3.64, "(a)", ha="center", va="center", fontsize=9,
        weight="bold", color=tint(PURPLE, 1.0), zorder=4)

box(3.50, -4.40, 2.90, 0.90,
    "MeanPool over $\\tilde{h}_t^A$\n$\\bar{h}_t^A \\in \\mathbb{R}^{1024}$",
    fs=8.0)
box(3.50, -5.60, 3.00, 0.75, "Linear  $1024 \\rightarrow 512$")
box(3.50, -6.60, 3.00, 0.75, "SiLU")
box(3.50, -7.60, 3.00, 0.75, "Linear  $512 \\rightarrow 1$")

box(3.50, -8.90, 3.00, 1.20, "", fc=tint(PURPLE, 0.08),
    ec=tint(PURPLE, 0.55))
ax.text(3.50, -8.56, "predicted progress $\\hat{p}_t$",
        ha="center", va="center", fontsize=7.5, color=INK, zorder=5)
ax.add_patch(Rectangle((2.35, -9.32), 2.30, 0.32, linewidth=0.6,
                       edgecolor="0.55", facecolor="white", zorder=5))
ax.add_patch(Rectangle((2.35, -9.32), 1.38, 0.32, linewidth=0,
                       facecolor=tint(PURPLE, 0.55), zorder=6))

box(3.50, -10.30, 3.00, 0.70, "$\\mathcal{L}_{\\mathrm{prog}} = "
    "\\|\\hat{p}_t - p_t\\|_2^2$",
    fc=tint(PURPLE, 0.14), ec=tint(PURPLE, 0.60))
ax.text(3.50, -11.20,
        "linear output, no output squashing;\n"
        "phase target $p_t = k_t / \\max(N-1,\\,1) \\in [0,1]$,\n"
        "constant within a phase",
        ha="center", va="center", fontsize=7.0, color="0.32", linespacing=1.5)

for p, q in [(-4.85, -5.22), (-5.98, -6.22), (-6.98, -7.22),
             (-7.98, -8.30), (-9.50, -9.95)]:
    arrow((3.50, p), (3.50, q))

# ========================================================== (b) SUBTASK HEAD
panel(7.02, -12.55, 14.90, -3.40, BLUE)
ax.text(7.30, -3.64, "(b)", ha="center", va="center", fontsize=9,
        weight="bold", color=tint(BLUE, 1.0), zorder=4)

box(8.85, -4.40, 2.40, 0.75, "$W_A$:  $1024 \\rightarrow 2048$", fs=7.6)
box(11.35, -4.40, 2.40, 0.75, "$W_L$:  $2048 \\rightarrow 2048$", fs=7.6)
box(10.10, -5.60, 5.00, 0.75,
    "$\\mathrm{Mem}_t = [\\, W_A \\tilde{h}_t^A ; \\; W_L h_t^L \\,]$")

box(10.10, -6.70, 4.00, 0.75, "Causal self-attention")
box(10.10, -7.75, 4.00, 0.75,
    "Cross-attention  ($K, V \\leftarrow \\mathrm{Mem}_t$)", fs=7.6)
box(10.10, -8.80, 4.00, 0.75, "FFN  ($8192$)")
ax.text(7.45, -7.75, "$\\times\\,4$ layers,   $d = 2048$,   $16$ heads",
        ha="center", va="center", fontsize=7.2, rotation=90,
        weight="bold", color=tint(BLUE, 1.0), zorder=4)

box(10.10, -9.90, 4.00, 0.75, "lm_head reused from the VLM", family="monospace",
    fs=7.2)
box(10.10, -10.95, 4.00, 0.80,
    "$\\hat{y}_t$:  \u201cpick the blue block\u201d",
    fc=tint(BLUE, 0.08), ec=tint(BLUE, 0.55))
box(10.10, -12.05, 5.00, 0.70,
    "$\\mathcal{L}_{\\mathrm{sub}} = -\\sum_j m_{t,j} \\log "
    "p_\\theta(y_{t,j} \\mid y_{t,<j}, \\mathrm{Mem}_t)$",
    fc=tint(BLUE, 0.14), ec=tint(BLUE, 0.60), fs=7.4)

box(13.65, -8.00, 2.20, 3.40,
    "embed_tokens\nreused from the\nVLM\n\n"
    "train: teacher\nforcing on\n$y_{t,<j}$\n\n"
    "test: greedy\ndecoding, up\nto $64$ tokens",
    fc="0.975", ec="0.60", fs=6.5)

arrow((8.85, -4.78), (8.85, -5.20))
arrow((11.35, -4.78), (11.35, -5.20))
for p, q in [(-5.98, -6.30), (-7.08, -7.35), (-8.13, -8.40),
             (-9.18, -9.51), (-10.28, -10.53), (-11.36, -11.68)]:
    arrow((10.10, p), (10.10, q))
arrow((12.55, -6.70), (12.12, -6.70))

# ---------------------------------------------------------- legend and caption
legend = [
    (1.00, tint(ORANGE, 0.35), tint(ORANGE, 0.85),
     "retained action tokens $\\tilde{h}_t^A$"),
    (5.30, DROP_FC, DROP_EC, "discarded state and register positions"),
    (10.55, tint(BLUE, 0.30), tint(BLUE, 0.85), "VLM tokens $h_t^L$"),
]
for x, fc, ec, label in legend:
    ax.add_patch(Rectangle((x, -12.98), 0.24, 0.24, linewidth=0.6,
                           edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + 0.36, -12.86, label, ha="left", va="center", fontsize=7.2,
            color="0.30")

ax.text(7.88, -13.45,
        "Neither head detaches its input, so both auxiliary losses "
        "backpropagate into the unified transformer.\n"
        "Post-training optimizes $\\mathcal{L}_{\\mathrm{post}} = "
        "\\lambda_v \\mathcal{L}_{\\mathrm{video}} + "
        "\\lambda_a \\mathcal{L}_{\\mathrm{action}} + "
        "\\lambda_s \\mathcal{L}_{\\mathrm{sub}} + "
        "\\lambda_p \\mathcal{L}_{\\mathrm{prog}}$, "
        "with $\\lambda_v = \\lambda_a = 1$ and "
        "$\\lambda_s = \\lambda_p = 0.5$.",
        ha="center", va="top", fontsize=7.4, color="0.30", linespacing=1.6)

# ----------------------------------------------------------------- write files
out = Path(__file__).resolve().parent
fig.savefig(out / "figure_subtask_progress.pdf", bbox_inches="tight",
            pad_inches=0.02)
fig.savefig(out / "figure_subtask_progress.png", dpi=400,
            bbox_inches="tight", pad_inches=0.02)
print("wrote", out / "figure_subtask_progress.pdf")
print("wrote", out / "figure_subtask_progress.png")
