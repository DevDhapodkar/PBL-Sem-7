"""
Figures for the real-data (Nagpur lakes) mode: true-colour + detections,
water/debris index panels, and cross-modal confidence over the imagery.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .detect_real import fdi, ndvi, ndwi


def _stretch(x, p=2):
    lo, hi = np.percentile(x, [p, 100 - p])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def true_color(bs):
    return np.dstack([_stretch(bs["B04"]), _stretch(bs["B03"]), _stretch(bs["B02"])])


def plot_scene_overview(bs, scene, diag, path=None):
    """True colour + detections, NDWI, and NDVI/FDI floating-matter panels."""
    rgb = true_color(bs)
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    m = bs.meta

    ax[0].imshow(rgb)
    opt = np.array([d.xy for d in scene.optical]) if scene.optical else np.empty((0, 2))
    sar = np.array([d.xy for d in scene.sar]) if scene.sar else np.empty((0, 2))
    if len(sar):
        ax[0].scatter(sar[:, 0], sar[:, 1], s=22, marker="^", c="#ff3b30",
                      alpha=.85, label="SAR (S1) detections")
    if len(opt):
        ax[0].scatter(opt[:, 0], opt[:, 1], s=22, marker="o",
                      edgecolors="#00e5ff", facecolors="none", linewidths=1.1,
                      label="optical (S2) detections")
    ax[0].set_title(f"Sentinel-2 true colour — {m.get('lake','Nagpur lake')}\n"
                    f"{m.get('date','')}  ·  detections")
    ax[0].legend(loc="lower right", fontsize=7)

    im1 = ax[1].imshow(diag["ndwi"], cmap="RdBu", vmin=-0.6, vmax=0.6)
    ax[1].contour(diag["water"], levels=[0.5], colors="k", linewidths=0.6)
    ax[1].set_title("NDWI  (water mask outlined)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    floating = np.where(diag["water"], np.maximum(diag["ndvi"], 0), np.nan)
    im2 = ax[2].imshow(rgb)
    im2 = ax[2].imshow(floating, cmap="YlOrRd", vmin=0, vmax=0.5, alpha=0.75)
    ax[2].set_title("Floating-matter cue on water\n(NDVI/FDI anomaly)")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130)
    return fig


def plot_confidence_over_image(bs, proposed, threshold, path=None):
    """Cross-modal debris confidence painted over the true-colour lake image."""
    rgb = true_color(bs)
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    ax.imshow(rgb)
    cands = proposed.fusion.candidates
    if cands:
        xs = np.array([c.xy for c in cands])
        cs = np.array([c.confidence for c in cands])
        matched = np.array([c.matched for c in cands])
        sc = ax.scatter(xs[matched, 0], xs[matched, 1], c=cs[matched], cmap="viridis",
                        vmin=0, vmax=1, s=70, edgecolors="w", linewidths=0.5,
                        label="corroborated (S1∩S2)")
        ax.scatter(xs[~matched, 0], xs[~matched, 1], c=cs[~matched], cmap="viridis",
                   vmin=0, vmax=1, s=34, marker="x", label="single-sensor")
        plt.colorbar(sc, ax=ax, fraction=0.046, label="debris confidence  C")
        for c in cands:
            if c.confidence >= threshold:
                ax.scatter(*c.xy, s=150, facecolors="none", edgecolors="#ff3b30",
                           linewidths=1.4)
    ax.set_title(f"Cross-modal debris confidence over {bs.meta.get('lake','lake')}\n"
                 f"red ring: reported debris (C ≥ {threshold})")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130)
    return fig
