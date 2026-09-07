"""Publication figures for the CHN manuscript.

Every figure is written as both a vector PDF (for LaTeX) and a 300 DPI PNG
(for the repository README and for previewing).  Run :mod:`benchmark` and
:mod:`experiments` first; this module only reads their saved output.

Colour assignment follows a fixed categorical order -- one hue per method,
never recycled -- and every series additionally carries its own marker and dash
pattern, so the figures stay readable in greyscale print and under colour
vision deficiency.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"

# Fixed categorical assignment: one hue per method, in slot order.
COLOR = {
    "chn_t": "#2a78d6",   # slot 1, blue    -- the recommended method
    "yuksel": "#eb6834",  # slot 2, orange  -- the baseline being compared to
    "chn": "#1baf7a",     # slot 3, aqua    -- full-RNM hybrid
    "newton": "#4a3aa7",  # slot 4, violet  -- ablation
}
MARKER = {"chn_t": "o", "yuksel": "s", "chn": "^", "newton": "D"}
DASH = {"chn_t": "-", "yuksel": "--", "chn": "-.", "newton": (0, (3, 1, 1, 1))}
LABEL = {
    "chn_t": "CHN-T (truncated RNM)",
    "chn": "CHN (full RNM)",
    "newton": "Newton + deflation",
    "yuksel": "Yuksel (2022)",
}

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dcdcd8"


def configure() -> None:
    """House style: recessive axes, serif text to match the manuscript."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK_SOFT,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
    })


def save(fig, name: str) -> None:
    """Write a figure as vector PDF and 300 DPI PNG."""
    FIGS.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf and .png")


def series(ax, x, y, key, **kw):
    """Draw one method's series with its fixed hue, marker and dash."""
    return ax.plot(x, y, color=COLOR[key], marker=MARKER[key],
                   linestyle=DASH[key], label=LABEL[key],
                   markeredgecolor="white", markeredgewidth=0.6, **kw)


# --------------------------------------------------------------------------- #
def figure_performance(bench: dict) -> None:
    """Wall-clock cost and the resulting speedup over the baseline."""
    d = bench["real_rooted"]
    degrees = np.array(d["meta"]["degrees"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0))

    for key in ("yuksel", "newton", "chn", "chn_t"):
        series(ax1, degrees, d[key]["time_med"], key)
    ax1.set_yscale("log")
    ax1.set_xlabel("polynomial degree $n$")
    ax1.set_ylabel("median time per polynomial (ms)")
    ax1.set_title("(a)  Cost of computing all roots", loc="left")
    ax1.grid(True, which="major", axis="both")
    ax1.set_xticks(degrees[::2])
    ax1.legend(loc="upper left")

    sp_t = np.array(d["speedup_median"]["chn_t"])
    sp = np.array(d["speedup_median"]["chn"])
    ax2.axhline(1.0, color=INK_SOFT, lw=0.8, ls=":")
    ax2.fill_between(degrees, 1.0, sp_t, where=(sp_t >= 1),
                     color=COLOR["chn_t"], alpha=0.10, linewidth=0)
    series(ax2, degrees, sp_t, "chn_t")
    series(ax2, degrees, sp, "chn")
    ax2.set_xlabel("polynomial degree $n$")
    ax2.set_ylabel("speedup over Yuksel  ($T_{\\mathrm{Y}}/T$)")
    ax2.set_title("(b)  Speedup; above 1 the hybrid is faster", loc="left")
    ax2.grid(True, axis="y")
    ax2.set_xticks(degrees[::2])
    ax2.legend(loc="upper left")

    peak = int(np.argmax(sp_t))
    ax2.annotate(f"{sp_t[peak]:.1f}$\\times$ at $n={degrees[peak]}$",
                 xy=(degrees[peak], sp_t[peak]),
                 xytext=(-10, -30), textcoords="offset points",
                 ha="right", fontsize=8, color=INK_SOFT)

    fig.tight_layout()
    save(fig, "fig_performance")


def figure_accuracy(bench: dict) -> None:
    """Forward error and root recovery -- the claims the draft got wrong."""
    d = bench["real_rooted"]
    degrees = np.array(d["meta"]["degrees"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0))

    for key in ("yuksel", "newton", "chn", "chn_t"):
        series(ax1, degrees, d[key]["err_med"], key)
    ax1.set_yscale("log")
    ax1.set_xlabel("polynomial degree $n$")
    ax1.set_ylabel(r"median $\max_i \min_j |\zeta_i - \hat{z}_j|$")
    ax1.set_title("(a)  Forward error against the planted roots", loc="left")
    ax1.grid(True, axis="both")
    ax1.set_xticks(degrees[::2])
    ax1.legend(loc="upper left")

    for key in ("yuksel", "newton", "chn", "chn_t"):
        series(ax2, degrees, d[key]["recovery"], key)
    ax2.set_xlabel("polynomial degree $n$")
    ax2.set_ylabel("roots recovered to $10^{-6}$ (%)")
    ax2.set_title("(b)  Recovery rate; all four methods agree", loc="left")
    ax2.set_ylim(94, 100.6)
    ax2.grid(True, axis="y")
    ax2.set_xticks(degrees[::2])
    ax2.legend(loc="lower left")

    fig.tight_layout()
    save(fig, "fig_accuracy")


def figure_coverage(bench: dict) -> None:
    """What fraction of the roots does each method actually return?"""
    m = bench["mixed"]
    degrees = np.array(m["degrees"])
    width = 0.38
    idx = np.arange(len(degrees))

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.bar(idx - width / 2, m["chn_all"], width, color=COLOR["chn_t"],
           label="CHN-T", zorder=3)
    ax.bar(idx + width / 2, m["yuksel_all"], width, color=COLOR["yuksel"],
           label="Yuksel (2022)", zorder=3)
    for i, v in enumerate(m["yuksel_all"]):
        ax.text(idx[i] + width / 2, v + 1.5, f"{v:.0f}", ha="center",
                fontsize=7.5, color=INK_SOFT)
    ax.set_xticks(idx)
    ax.set_xticklabels(degrees)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("polynomial degree $n$")
    ax.set_ylabel("roots returned (% of $n$)")
    ax.set_title("Polynomials with real and complex roots", loc="left", pad=16)
    ax.grid(True, axis="y", zorder=0)
    # Placed above the axes: at 100% the bars reach the top of the plot area,
    # so any in-axes position would sit on top of the data.
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.14), ncols=2)
    fig.tight_layout()
    save(fig, "fig_coverage")


def figure_truncation(robust: dict) -> None:
    """Why the truncated fallback replaces the full one."""
    stall = robust["stall_study"]
    hist = {int(k): v for k, v in stall["truncation_histogram"].items()}
    degrees = np.array(stall["degrees"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.1),
                                   gridspec_kw={"wspace": 0.32})

    orders = sorted(hist)[:6]
    ax1.bar([str(m) for m in orders], [hist[m] for m in orders],
            color=COLOR["chn_t"], width=0.6, zorder=3)
    for i, m in enumerate(orders):
        ax1.text(i, hist[m] + 2.0, f"{hist[m]:.1f}%", ha="center",
                 fontsize=8, color=INK_SOFT)
    ax1.set_xlabel("truncation order $m$ accepted")
    ax1.set_ylabel("share of fallback steps (%)")
    ax1.set_title("(a)  Truncation order accepted", loc="left")
    ax1.set_ylim(0, 110)
    ax1.grid(True, axis="y", zorder=0)

    width = 0.38
    idx = np.arange(len(degrees))
    ax2.bar(idx - width / 2, stall["chn_capped_pct"], width,
            color=COLOR["chn"], label=LABEL["chn"], zorder=3)
    ax2.bar(idx + width / 2, stall["chn_t_capped_pct"], width,
            color=COLOR["chn_t"], label=LABEL["chn_t"], zorder=3)
    ax2.set_xticks(idx)
    ax2.set_xticklabels(degrees)
    ax2.set_xlabel("polynomial degree $n$")
    ax2.set_ylabel("instances hitting the cap (%)")
    ax2.set_title("(b)  Stalling of the guaranteed step", loc="left")
    ax2.grid(True, axis="y", zorder=0)
    ax2.set_ylim(0, max(stall["chn_capped_pct"]) * 1.45)
    ax2.legend(loc="upper center", ncols=2, fontsize=7.5)

    save(fig, "fig_truncation")


def figure_robustness(robust: dict) -> None:
    """Outright failure -- cycles and divergence -- on adversarial inputs."""
    sweep = robust["seed_sweep"]
    names = list(sweep)
    pretty = {
        "smale_cycle": "$z^3-2z+2$\n(attracting cycle)",
        "z3_minus_1": "$z^3-1$",
        "triple_root": "$(z-1)^3$",
        "clustered": "clustered\nroots",
        "chebyshev_20": "Chebyshev\n$T_{20}$",
        "wilkinson_12": "Wilkinson\n$n=12$",
    }
    idx = np.arange(len(names))
    width = 0.27

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    for offset, key in ((-width, "newton"), (0.0, "chn"), (width, "chn_t")):
        vals = [sweep[n][f"{key}_diverged_pct"] for n in names]
        ax.bar(idx + offset, vals, width, color=COLOR[key], label=LABEL[key],
               zorder=3)
    ax.set_xticks(idx)
    ax.set_xticklabels([pretty[n] for n in names], fontsize=7.5)
    ax.set_ylabel("starting points that fail outright (%)")
    ax.set_title("Failure from 4000 starting points in $|z| \\leq 2.5$; "
                 "lower is better", loc="left")
    ax.grid(True, axis="y", zorder=0)
    ax.legend(loc="upper right")
    ax.annotate("Newton is captured\nby its 2-cycle",
                xy=(0 - width, sweep["smale_cycle"]["newton_diverged_pct"]),
                xytext=(14, 16), textcoords="offset points", fontsize=7.5,
                color=INK_SOFT,
                arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=0.7))
    fig.tight_layout()
    save(fig, "fig_robustness")


# --------------------------------------------------------------------------- #
def _layered_image(basin: np.ndarray, iters: np.ndarray, n_roots: int,
                   bands: int = 12) -> np.ndarray:
    """Layered modular colouring: basin sets the hue, iteration count the shade.

    Each converged pixel is coloured by the root it reached and then banded by
    ``iteration mod bands``, which is the layering used in the polynomiographs
    of Kalantari's LASER article.  Non-converged pixels are left black, so a
    method's failure set is immediately visible.
    """
    hues = [COLOR["chn_t"], COLOR["yuksel"], COLOR["chn"], COLOR["newton"],
            "#e87ba4", "#eda100"]
    height, width = basin.shape
    out = np.zeros((height, width, 3))
    for r in range(n_roots):
        base = np.array(to_rgb(hues[r % len(hues)]))
        mask = basin == r
        if not mask.any():
            continue
        band = (iters[mask] % bands) / (bands - 1.0)
        # Ramp from a light tint to the full hue: monotone in lightness.
        shade = 0.35 + 0.65 * band
        out[mask] = base[None, :] * shade[:, None] + (1.0 - shade)[:, None] * 1.0
    return out


def figure_basins(basins: dict, name: str, title: str, filename: str) -> None:
    """Side-by-side polynomiographs for Newton and the two hybrids."""
    roots = basins[f"{name}_roots"]
    limit = float(basins[f"{name}_limit"])
    extent = [-limit, limit, -limit, limit]

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.85))
    panels = [("newton", "Newton"), ("chn", "CHN (full RNM)"),
              ("chn_t", "CHN-T (truncated RNM)")]
    for ax, (key, label) in zip(axes, panels):
        basin = basins[f"{name}_{key}_basin"]
        iters = basins[f"{name}_{key}_iters"]
        image = _layered_image(basin, iters, len(roots))
        ax.imshow(image, extent=extent, origin="lower", interpolation="nearest")
        failures = int((basin < 0).sum())
        ax.set_title(f"{label}\n{failures} of {basin.size} points fail "
                     f"({100.0 * failures / basin.size:.2f}%)", fontsize=8.5)
        for r in roots:
            ax.scatter(r.real, r.imag, s=42, marker="*", c="white",
                       edgecolors="black", linewidths=0.6, zorder=5)
        ax.set_xlabel(r"$\mathrm{Re}\,z$")
        ax.set_xticks([-limit, 0, limit])
        ax.set_yticks([-limit, 0, limit])
    axes[0].set_ylabel(r"$\mathrm{Im}\,z$")
    fig.suptitle(title, y=1.04, fontsize=9.5, ha="left", x=0.02)
    fig.tight_layout()
    save(fig, filename)


def main() -> None:
    configure()
    bench = json.loads((DATA / "benchmark.json").read_text())
    robust = json.loads((DATA / "robustness.json").read_text())
    basins = dict(np.load(DATA / "basins.npz"))

    print("building figures:")
    figure_performance(bench)
    figure_accuracy(bench)
    figure_coverage(bench)
    figure_truncation(robust)
    figure_robustness(robust)
    figure_basins(
        basins, "z3_minus_1",
        r"Basins of attraction for $z^3-1$, coloured by root and banded by "
        r"iteration count",
        "fig_basins_z3",
    )
    figure_basins(
        basins, "smale_cycle",
        r"Basins for $z^3-2z+2$: black is the set of starting points Newton "
        r"never escapes",
        "fig_basins_smale",
    )


if __name__ == "__main__":
    main()
