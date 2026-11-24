"""Plot val20 accuracy vs. training-set size (report figure)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent / "results"
rows = [r for r in json.load(open(RES / "learning_curve.json")) if r["protocol"] == "honest"]
base = json.load(open(RES / "baseline.json"))["baseline_train80_val20"]

curve = sorted([(r["n_train"], r["acc"]) for r in rows if r["name"].startswith("balanced") and "C=" not in r["name"]])
allimb = next(r for r in rows if r["name"] == "all (imbalanced)")
prov = next(r for r in rows if r["name"] == "provided train80 only")

INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
ax.plot([n for n, _ in curve], [a * 100 for _, a in curve], color=BLUE, lw=2, marker="o", ms=6,
        label="TF-IDF + LogReg, scraped data (balanced)", zorder=3)
ax.plot([allimb["n_train"]], [allimb["acc"] * 100], color=BLUE, marker="*", ms=14, ls="none",
        markeredgecolor="white", zorder=4, label="TF-IDF + LogReg, all 636k links")
ax.plot([prov["n_train"]], [prov["acc"] * 100], color=ORANGE, marker="s", ms=8, ls="none",
        markeredgecolor="white", zorder=4, label="TF-IDF + LogReg, provided links only")
ax.axhline(base * 100, color=MUTED, lw=1.5, ls="--", zorder=2)
ax.text(1.6e3, base * 100 + 0.4, f"official baseline (provided links) {base*100:.1f}%", color=MUTED, fontsize=8)

ax.annotate(f"{allimb['acc']*100:.1f}%", (allimb["n_train"], allimb["acc"] * 100), textcoords="offset points",
            xytext=(-6, 9), ha="right", fontsize=9, color=INK)
ax.annotate(f"{prov['acc']*100:.1f}%", (prov["n_train"], prov["acc"] * 100), textcoords="offset points",
            xytext=(0, -16), ha="center", fontsize=9, color=INK)

ax.set_xscale("log")
ax.set_xlabel("training links (log scale)", color=INK)
ax.set_ylabel("accuracy on held-out val20 (%)", color=INK)
ax.set_title("More scraped data keeps helping", loc="left", color=INK, fontsize=12, fontweight="bold")
ax.grid(True, color=GRID, lw=0.8, zorder=0)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.97), labelcolor=INK)
fig.tight_layout()
fig.savefig(RES / "learning_curve.png")
print("saved", RES / "learning_curve.png")
