from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


fixed = {
    "Fixed Gemini": (0.00012380, 0.7798),
    "Fixed Qwen": (0.00015400, 0.9300),
    "Fixed GPT": (0.00058100, 0.8992),
    "Fixed Claude": (0.00238463, 0.9547),
}

category = {
    r"Category $\alpha \geq 0.3$": (0.00012900, 0.9383),
    r"Category $\alpha = 0.1$": (0.00039200, 0.9403),
}

utility = {
    r"U $\alpha = 0.1$": (0.00064059, 0.9403),
    r"U $\alpha = 0.3$": (0.00023132, 0.9321),
    r"U $\alpha = 0.5$": (0.00017915, 0.9259),
    r"U $\alpha = 0.7$": (0.00015645, 0.9300),
    r"U $\alpha = 1.0$": (0.00014618, 0.9321),
}

risk = {
    r"R $\alpha = 0.1$": (0.00080440, 0.9424),
    r"R $\alpha = 0.3$": (0.00021200, 0.9300),
    r"R $\alpha = 0.5$": (0.00018591, 0.9321),
    r"R $\alpha = 0.7$": (0.00016518, 0.9300),
    r"R $\alpha = 1.0$": (0.00014675, 0.9321),
}


COLOR_FIXED = "tab:orange"
COLOR_CATEGORY = "tab:green"
COLOR_UTILITY = "tab:blue"
COLOR_RISK = "tab:red"
COLOR_FRONTIER = "grey"


frontier = [
    ("Fixed Gemini", fixed["Fixed Gemini"]),
    (r"Category $\alpha \geq 0.3$", category[r"Category $\alpha \geq 0.3$"]),
    (r"Category $\alpha = 0.1$", category[r"Category $\alpha = 0.1$"]),
    (r"R $\alpha = 0.1$", risk[r"R $\alpha = 0.1$"]),
    ("Fixed Claude", fixed["Fixed Claude"]),
]

frontier_x = [point[1][0] for point in frontier]
frontier_y = [point[1][1] for point in frontier]


fig, axes = plt.subplots(
    1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.15, 1]}
)

ax1, ax2 = axes


# Full test-set view
for i, (label, (cost, acc)) in enumerate(fixed.items()):
    ax1.scatter(
        cost,
        acc,
        marker="D",
        s=65,
        color=COLOR_FIXED,
        label="Fixed model" if i == 0 else None,
        zorder=3,
    )


for i, (label, (cost, acc)) in enumerate(category.items()):
    ax1.scatter(
        cost,
        acc,
        marker="P",
        s=75,
        color=COLOR_CATEGORY,
        label="Category-only" if i == 0 else None,
        zorder=4,
    )


for i, (label, (cost, acc)) in enumerate(utility.items()):
    ax1.scatter(
        cost,
        acc,
        marker="o",
        s=55,
        color=COLOR_UTILITY,
        label="Utility-only" if i == 0 else None,
        zorder=3,
    )


for i, (label, (cost, acc)) in enumerate(risk.items()):
    ax1.scatter(
        cost,
        acc,
        marker="s",
        s=55,
        color=COLOR_RISK,
        label="Risk-aware" if i == 0 else None,
        zorder=3,
    )


ax1.plot(
    frontier_x,
    frontier_y,
    linestyle="--",
    linewidth=1.5,
    color=COLOR_FRONTIER,
    label="Pareto frontier",
    zorder=2,
)

ax1.set_xscale("log")
ax1.set_xlabel("Average cost per query ($)")
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0.77, 0.96)
ax1.set_title("(a) Full test-set view")


full_offsets = {
    "Fixed Gemini": (6, 4),
    r"Category $\alpha \geq 0.3$": (8, -20),
    r"Category $\alpha = 0.1$": (8, 12),
    r"R $\alpha = 0.1$": (10, -24),
    "Fixed Claude": (-88, -18),
}


for label, (cost, acc) in frontier:
    dx, dy = full_offsets[label]

    ax1.annotate(
        label,
        (cost, acc),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8.3,
    )


ax1.legend(frameon=False, fontsize=8.5, loc="lower right")


# Routing-policy region
for i, (label, (cost, acc)) in enumerate(category.items()):
    ax2.scatter(
        cost,
        acc,
        marker="P",
        s=80,
        color=COLOR_CATEGORY,
        label="Category-only" if i == 0 else None,
        zorder=4,
    )


for i, (label, (cost, acc)) in enumerate(utility.items()):
    ax2.scatter(
        cost,
        acc,
        marker="o",
        s=60,
        color=COLOR_UTILITY,
        label="Utility-only" if i == 0 else None,
        zorder=3,
    )


for i, (label, (cost, acc)) in enumerate(risk.items()):
    ax2.scatter(
        cost,
        acc,
        marker="s",
        s=60,
        color=COLOR_RISK,
        label="Risk-aware" if i == 0 else None,
        zorder=3,
    )


routing_frontier = [
    category[r"Category $\alpha \geq 0.3$"],
    category[r"Category $\alpha = 0.1$"],
    risk[r"R $\alpha = 0.1$"],
]


ax2.plot(
    [point[0] for point in routing_frontier],
    [point[1] for point in routing_frontier],
    linestyle="--",
    linewidth=1.5,
    color=COLOR_FRONTIER,
    label="Pareto frontier",
    zorder=2,
)


key_labels = {
    r"Category $\alpha \geq 0.3$": category[r"Category $\alpha \geq 0.3$"],
    r"Category $\alpha = 0.1$": category[r"Category $\alpha = 0.1$"],
    r"U $\alpha = 0.1$": utility[r"U $\alpha = 0.1$"],
    r"R $\alpha = 0.1$": risk[r"R $\alpha = 0.1$"],
}


key_offsets = {
    r"Category $\alpha \geq 0.3$": (6, 6),
    r"Category $\alpha = 0.1$": (6, 8),
    r"U $\alpha = 0.1$": (8, -20),
    r"R $\alpha = 0.1$": (8, 6),
}


for label, (cost, acc) in key_labels.items():
    dx, dy = key_offsets[label]

    ax2.annotate(
        label,
        (cost, acc),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8,
    )


ax2.set_xscale("log")
ax2.set_xlim(0.00012, 0.0009)
ax2.set_ylim(0.923, 0.945)
ax2.set_xlabel("Average cost per query ($)")
ax2.set_ylabel("Accuracy")
ax2.set_title("(b) Routing-policy region")

ax2.legend(frameon=False, fontsize=8.5, loc="lower right")


plt.tight_layout()

pdf_path = FIGURE_DIR / "pareto_frontier_final.pdf"
png_path = FIGURE_DIR / "pareto_frontier_final.png"

plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")

plt.close()

print(f"Saved Pareto frontier to {FIGURE_DIR}")