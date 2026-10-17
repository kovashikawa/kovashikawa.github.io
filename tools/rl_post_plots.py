# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Plots for the RL-for-language-models post. Same numbers as the 'Reproduce the numbers' block.

Run: uv run tools/rl_post_plots.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).parent.parent / "assets" / "images" / "rl-for-language-models"
INK, MUTED, LIGHT, RULE, ACCENT = "#1E1E1E", "#555555", "#888888", "#e6e6e3", "#2a78d6"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": LIGHT, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "figure.facecolor": "white", "axes.facecolor": "white",
})

sig = lambda x: 1 / (1 + np.exp(-x))
logit = lambda x: np.log(x / (1 - x))


def sub(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")


# ---------------------------------------------------------------- figure 1
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
fig.subplots_adjust(wspace=0.32, top=0.76, bottom=0.17, left=0.05, right=0.985)

# (a) RLHF: Bradley-Terry
ax = axs[0]
g = np.linspace(0, 6, 200)
ax.plot(g, sig(g), color=INK, lw=2)
pts = np.array([0, 1, 2, 4])
ax.scatter(pts, sig(pts), color=INK, s=22, zorder=3)
for x in pts:
    ax.annotate(f"{sig(x):.3f}", (x, sig(x)), textcoords="offset points",
                xytext=(6, -13), fontsize=8.5, color=MUTED)
ax.axhline(0.5, color=RULE, lw=1, zorder=0)
ax.set_ylim(0.45, 1.02)
ax.set_xlabel(r"reward gap  $r_w - r_l$")
ax.set_ylabel("P(preferred)")
ax.set_title("RLHF", loc="left", pad=34)
sub(ax, "Reward model fit to comparisons:\na gap becomes a preference probability")

# (b) RLVR / GRPO: degenerate groups
ax = axs[1]
p = np.linspace(0, 1, 400)
shades = {4: LIGHT, 8: MUTED, 16: INK}
for G, c in shades.items():
    ax.plot(p, p**G + (1 - p)**G, color=c, lw=2, label=f"G = {G}")
ax.legend(frameon=False, loc="upper center", fontsize=9, labelcolor="linecolor")
ax.set_ylim(0, 1.05)
ax.set_xlabel("true success probability  p")
ax.set_ylabel("P(all G rewards equal)")
ax.set_title("RLVR with GRPO", loc="left", pad=34)
sub(ax, "Binary reward: a group with equal rewards\nhas zero advantage and no gradient")

# (c) RLCR: expected reward vs stated confidence
ax = axs[2]
p0 = 0.7
q = np.linspace(0, 1, 400)
rlcr = p0 - (p0 * (1 - q) ** 2 + (1 - p0) * q**2)
ax.plot(q, np.full_like(q, p0), color=LIGHT, lw=2, ls=(0, (5, 3)))
ax.plot(q, rlcr, color=ACCENT, lw=2.2)
ax.scatter([p0], [p0**2], color=ACCENT, s=30, zorder=3)
ax.text(0.97, 0.22, "peak at q = p,\nvalue p² = 0.49", fontsize=8.5, color=ACCENT, ha="right")
ax.text(0.02, 0.715, "RLVR: flat in q", color=LIGHT, fontsize=9)
ax.text(0.40, 0.12, "RLCR", color=ACCENT, fontsize=9)
ax.set_ylim(-0.02, 0.82)
ax.set_xlabel("stated confidence  q   (true p = 0.7)")
ax.set_ylabel("expected reward")
ax.set_title("RLCR", loc="left", pad=34)
sub(ax, "Correctness minus Brier:\nhonest confidence is the optimum")

fig.savefig(OUT / "rl-reward-signals.png", dpi=220)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
rng = np.random.default_rng(0)
n = 200_000
pt = rng.beta(2, 1.2, size=n)
y = (rng.random(n) < pt).astype(float)


def conf(k):
    return np.clip(sig(k * logit(np.clip(pt, 1e-6, 1 - 1e-6))), 1e-6, 1 - 1e-6)


ks = np.exp(np.linspace(np.log(0.4), np.log(6), 40))
acc = y.mean()
rl = [(y - (conf(k) - y) ** 2).mean() for k in ks]

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.1), gridspec_kw={"width_ratios": [1.15, 1]})
fig.subplots_adjust(wspace=0.28, top=0.76, bottom=0.16, left=0.07, right=0.985)

ax = axs[0]
ax.plot(ks, np.full_like(ks, acc), color=LIGHT, lw=2, ls=(0, (5, 3)))
ax.plot(ks, rl, color=ACCENT, lw=2.2)
ax.scatter([1], [(y - (conf(1) - y) ** 2).mean()], color=ACCENT, s=30, zorder=3)
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 5]); ax.set_xticklabels(["0.5", "1", "2", "5"])
ax.minorticks_off()
ax.text(0.42, acc + 0.008, f"RLVR reward = accuracy = {acc:.3f}", color=LIGHT, fontsize=9)
ax.text(1.05, (y - (conf(1) - y) ** 2).mean() + 0.012, "k = 1, calibrated by construction",
        color=ACCENT, fontsize=8.5)
ax.text(2.6, rl[-12] + 0.012, "RLCR reward", color=ACCENT, fontsize=9)
ax.set_ylim(0.38, 0.68)
ax.set_xlabel("confidence sharpness  k   (k > 1 overconfident, k < 1 underconfident)")
ax.set_ylabel("mean reward")
ax.set_title("Same accuracy, different rewards", loc="left", pad=34)
sub(ax, "Same answers in every model;\nonly the stated confidence changes")

ax = axs[1]
M = 15
ax.plot([0, 1], [0, 1], color=RULE, lw=1.2, zorder=0)
for k, c, lab in [(1, ACCENT, "k = 1"), (3, INK, "k = 3")]:
    qk = conf(k)
    b = np.minimum((qk * M).astype(int), M - 1)
    xs = [qk[b == m].mean() for m in range(M) if (b == m).any()]
    ys = [y[b == m].mean() for m in range(M) if (b == m).any()]
    ax.plot(xs, ys, color=c, lw=1.8, marker="o", ms=3.5)
ax.text(0.62, 0.70, "k = 1", color=ACCENT, fontsize=9, rotation=40)
ax.text(0.80, 0.50, "k = 3", color=INK, fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("mean stated confidence (15 bins)")
ax.set_ylabel("observed accuracy")
ax.set_title("Reliability", loc="left", pad=34)
sub(ax, "On the diagonal is calibrated;\nk = 3 is overconfident at high q")

fig.savefig(OUT / "rl-toy-calibration.png", dpi=220)
plt.close(fig)

print("accuracy", round(acc, 3), "rlcr@k=1", round((y - (conf(1) - y) ** 2).mean(), 3))
