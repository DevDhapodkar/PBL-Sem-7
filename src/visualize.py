"""
Figures for the prototype: registration before/after, cross-modal matches,
confidence maps and precision-recall comparison.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt
import numpy as np


def plot_registration(scene, proposed, traditional, path):
    """Two-panel: point sets before registration vs after (traditional & proposed)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    sar = scene.sar_xy
    opt = scene.optical_xy
    truth = scene.true_debris

    # -- before --
    ax = axes[0]
    ax.scatter(opt[:, 0], opt[:, 1], c="#1f77b4", s=28, marker="o",
               label="optical detections", alpha=0.8)
    ax.scatter(sar[:, 0], sar[:, 1], c="#d62728", s=28, marker="^",
               label="SAR detections (own frame)", alpha=0.8)
    ax.set_title("Before registration\n(SAR & optical point sets misaligned)")
    ax.legend(loc="upper right", fontsize=8)

    # -- traditional (ICP) --
    ax = axes[1]
    si = traditional.sar_in_opt
    ax.scatter(truth[:, 0], truth[:, 1], c="0.6", s=70, marker="*",
               label="ground-truth debris", zorder=1)
    ax.scatter(opt[:, 0], opt[:, 1], c="#1f77b4", s=24, marker="o", alpha=0.8)
    if si is not None:
        ax.scatter(si[:, 0], si[:, 1], c="#d62728", s=24, marker="^", alpha=0.8)
    ax.set_title(f"Traditional: ICP registration\n(RMSE={traditional.icp_rmse:.1f})")
    ax.legend(loc="upper right", fontsize=8)

    # -- proposed (RANSAC + CPD) --
    ax = axes[2]
    si = proposed.sar_in_opt
    ax.scatter(truth[:, 0], truth[:, 1], c="0.6", s=70, marker="*",
               label="ground-truth debris", zorder=1)
    ax.scatter(opt[:, 0], opt[:, 1], c="#1f77b4", s=24, marker="o", alpha=0.8)
    ax.scatter(si[:, 0], si[:, 1], c="#2ca02c", s=24, marker="^", alpha=0.8,
               label="SAR -> optical (RANSAC+CPD)")
    # draw cross-modal matches
    for c in proposed.fusion.candidates:
        if c.matched:
            ax.scatter(*c.xy, s=90, facecolors="none", edgecolors="k", linewidths=0.8)
    ax.set_title(f"Proposed: RANSAC+CPD\n(D_reg={proposed.fusion.d_registration:.1f}, "
                 f"matches={proposed.fusion.n_matches})")
    ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confidence(scene, proposed, threshold, path):
    """Confidence map: candidates coloured by C, ground truth overlaid."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    truth = scene.true_debris
    ax.scatter(truth[:, 0], truth[:, 1], c="0.55", s=120, marker="*",
               label="ground-truth debris", zorder=1)

    xs = np.array([c.xy for c in proposed.fusion.candidates])
    cs = np.array([c.confidence for c in proposed.fusion.candidates])
    matched = np.array([c.matched for c in proposed.fusion.candidates])
    if len(xs):
        sc = ax.scatter(xs[matched, 0], xs[matched, 1], c=cs[matched],
                        cmap="viridis", vmin=0, vmax=1, s=70, marker="o",
                        edgecolors="k", linewidths=0.4, label="corroborated", zorder=3)
        ax.scatter(xs[~matched, 0], xs[~matched, 1], c=cs[~matched],
                   cmap="viridis", vmin=0, vmax=1, s=40, marker="x",
                   label="single-modality", zorder=2)
        plt.colorbar(sc, ax=ax, label="debris confidence  C")
    # ring reported-as-debris (above threshold)
    for c in proposed.fusion.candidates:
        if c.confidence >= threshold:
            ax.scatter(*c.xy, s=150, facecolors="none", edgecolors="#d62728",
                       linewidths=1.3, zorder=4)
    ax.set_title(f"Proposed debris confidence  (red ring: C >= {threshold})")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pr_curves(curves, path):
    """Overlaid precision-recall curves. ``curves`` maps method -> (rec, prec, ap)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    style = {"proposed": ("#2ca02c", "-", "RANSAC+CPD (proposed)"),
             "traditional_icp": ("#ff7f0e", "--", "ICP (traditional)"),
             "optical_only": ("#1f77b4", ":", "optical-only")}
    for name, (rec, prec, apv) in curves.items():
        col, ls, lab = style.get(name, ("k", "-", name))
        order = np.argsort(rec)
        ax.plot(rec[order], prec[order], ls, color=col, lw=2.2,
                label=f"{lab} — AP={apv:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Debris detection: precision–recall\n"
                 "(proposed & ICP share the same fusion stage)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_robustness(sweep, path):
    """Precision & F1 vs increasing false-positive load, per method (2 panels)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = sweep["x"]
    styles = [("proposed", "#2ca02c", "RANSAC+CPD (proposed)"),
              ("traditional_icp", "#ff7f0e", "ICP (traditional)"),
              ("optical_only", "#1f77b4", "optical-only")]
    for ax, metric, title in [(axes[0], "precision", "Precision vs clutter"),
                              (axes[1], "f1", "F1 vs clutter")]:
        for name, col, lab in styles:
            key = f"{name}_{metric}"
            if key in sweep:
                ax.plot(x, sweep[key], "-o", color=col, label=lab, lw=2, ms=5)
        ax.set_xlabel("clutter load  (SAR false positives; optical = 1.2x)")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Robustness to modality-specific clutter", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
