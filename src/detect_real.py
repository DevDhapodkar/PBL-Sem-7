"""
Real detectors: Sentinel-2 / Sentinel-1 bands -> object-level debris point sets.
================================================================================

Turns the ``BandStack`` from ``src/acquire.py`` into the ``Detection`` point sets
the RANSAC+CPD+fusion pipeline consumes, using standard water/debris indices.

Optical (Sentinel-2)
--------------------
* **Water mask**  NDWI = (Green − NIR)/(Green + NIR) > t   (McFeeters 1996).
* **Floating matter on water** — union of two cues, evaluated on water pixels:
    - **NDVI** = (NIR − Red)/(NIR + Red): water hyacinth / algal scum float with a
      vegetation-like NIR bump (NDVI ≫ the surrounding water).
    - **FDI** (Floating Debris Index, Biermann et al. 2020): plastic / mixed trash
      rafts sit above the NIR baseline interpolated from red-edge and SWIR.
  Connected components of the anomaly mask → blob **centroids** = optical point
  set; the index value → detection strength.

SAR (Sentinel-1)
----------------
* **Backscatter anomaly on water** — floating mats damp capillary waves and change
  σ⁰ relative to the open-water background; a CFAR-style local-contrast test flags
  them. Blob centroids = SAR point set. ``detect_sar_backscatter`` runs on any real
  VV/VH ``BandStack``.

Because live Sentinel-1 is not reachable in every environment (egress policy /
Earth-Engine credentials), ``simulate_sar_pointset`` derives a realistic SAR view
**from the real optical detections** — an independent, noisy, mis-registered
sample plus SAR-specific clutter (boats, shore double-bounce) — so the cross-modal
demo runs on the real Nagpur optical scene. It is clearly a stand-in; swap in
``detect_sar_backscatter`` on a real S1 stack for fully-live operation.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .acquire import BandStack
from .data_simulation import Detection, Scene


# ---------------------------------------------------------------------------
# Indices
# ---------------------------------------------------------------------------
def ndwi(bs: BandStack) -> np.ndarray:
    g, nir = bs["B03"], bs["B08"]
    return (g - nir) / (g + nir + 1e-6)


def ndvi(bs: BandStack) -> np.ndarray:
    nir, red = bs["B08"], bs["B04"]
    return (nir - red) / (nir + red + 1e-6)


def fdi(bs: BandStack) -> np.ndarray:
    """Floating Debris Index (Biermann 2020), red-edge (B06) as the RE2 baseline."""
    nir, re2, swir = bs["B08"], bs["B06"], bs["B11"]
    lam = (833 - 665) / (1610 - 665)
    return nir - (re2 + (swir - re2) * lam)


def water_mask(bs: BandStack, t: float = 0.0) -> np.ndarray:
    return ndwi(bs) > t


# ---------------------------------------------------------------------------
# Blob extraction
# ---------------------------------------------------------------------------
def _blobs(mask: np.ndarray, strength_field: np.ndarray,
           min_size: int = 2, max_size: int = 400) -> list[tuple[np.ndarray, float]]:
    """Connected components -> (centroid[x,y], mean strength) in *pixel* coords."""
    lab, n = ndi.label(mask)
    if n == 0:
        return []
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        size = int(sel.sum())
        if size < min_size or size > max_size:
            continue
        ys, xs = np.where(sel)
        cx, cy = xs.mean(), ys.mean()
        out.append((np.array([cx, cy]), float(strength_field[sel].mean())))
    return out


# ---------------------------------------------------------------------------
# Optical detector (real Sentinel-2)
# ---------------------------------------------------------------------------
def detect_optical(
    bs: BandStack,
    ndwi_t: float = 0.0,
    ndvi_t: float = 0.15,
    fdi_t: float = 0.005,
    shoreline_dilate: int = 2,
) -> tuple[list[Detection], dict]:
    """Detect floating matter on the lake from real Sentinel-2 bands.

    Returns (optical detections in pixel coords, diagnostic index rasters).
    """
    w = water_mask(bs, ndwi_t)
    # include a thin shoreline collar where shore litter accumulates
    w_dil = ndi.binary_dilation(w, iterations=shoreline_dilate)
    nd, fd = ndvi(bs), fdi(bs)

    veg = w_dil & (nd > ndvi_t)             # hyacinth / algal scum (NIR bump)
    plastic = w_dil & (fd > fdi_t)          # plastic / trash rafts (FDI)
    anomaly = veg | plastic

    # normalise a combined strength in [0,1]
    s_ndvi = np.clip((nd - ndvi_t) / (0.6 - ndvi_t), 0, 1)
    s_fdi = np.clip((fd - fdi_t) / (0.06 - fdi_t), 0, 1)
    strength = np.maximum(s_ndvi, s_fdi)

    blobs = _blobs(anomaly, strength)
    dets = [Detection(xy, max(s, 0.05), True) for xy, s in blobs]
    diag = {"ndwi": ndwi(bs), "ndvi": nd, "fdi": fd, "water": w, "anomaly": anomaly}
    return dets, diag


# ---------------------------------------------------------------------------
# SAR detector (real Sentinel-1) — runs when a VV/VH stack is available
# ---------------------------------------------------------------------------
def detect_sar_backscatter(
    bs_sar: BandStack, water: np.ndarray,
    k: float = 2.5, win: int = 9,
) -> list[Detection]:  # pragma: no cover (needs live S1)
    """CFAR-style backscatter-anomaly detector on real Sentinel-1 VV over water."""
    vv = bs_sar["VV"].astype("float32")
    local_med = ndi.median_filter(vv, size=win)
    local_std = ndi.generic_filter(vv, np.std, size=win) + 1e-6
    anomaly = water & (np.abs(vv - local_med) > k * local_std)
    strength = np.clip(np.abs(vv - local_med) / (k * local_std) - 1, 0, 1)
    return [Detection(xy, max(s, 0.05), True) for xy, s in _blobs(anomaly, strength)]


# ---------------------------------------------------------------------------
# SAR stand-in derived from real optical (for environments without live S1)
# ---------------------------------------------------------------------------
def simulate_sar_pointset(
    optical: list[Detection],
    water: np.ndarray,
    scale: float = 1.02,
    rotation_deg: float = 3.0,
    translation: tuple[float, float] = (7.0, -5.0),
    p_detect: float = 0.8,
    loc_noise: float = 2.0,
    n_clutter: int = 22,
    seed: int = 0,
) -> tuple[list[Detection], dict]:
    """Realistic Sentinel-1 point set derived from the real optical detections.

    An *independent* noisy, mis-registered sample of the same floating matter
    (SAR also sees roughness-modulating mats), plus SAR-specific clutter (boats /
    bright shore returns). Used only where live S1 is unavailable; the returned
    ``true_transform`` is what the pipeline must recover.
    """
    rng = np.random.default_rng(seed)
    th = np.deg2rad(rotation_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    M = np.eye(3)
    M[:2, :2] = scale * R
    M[:2, 2] = translation

    def fwd(p):
        return (M[:2, :2] @ p) + M[:2, 2]

    dets: list[Detection] = []
    corroborable: list[int] = []          # optical indices the SAR sensor also saw
    for i, d in enumerate(optical):
        if rng.random() < p_detect:
            xy = fwd(d.xy) + rng.normal(0, loc_noise, size=2)
            s = float(np.clip(d.strength * rng.uniform(0.7, 1.1), 0, 1))
            dets.append(Detection(xy, s, True))
            corroborable.append(i)
    # SAR clutter near the shore (double-bounce) / boats
    H, W = water.shape
    shore = ndi.binary_dilation(water, iterations=3) & ~ndi.binary_erosion(water, iterations=2)
    ys, xs = np.where(shore)
    if len(xs):
        for _ in range(n_clutter):
            j = rng.integers(len(xs))
            xy = fwd(np.array([xs[j], ys[j]])) + rng.normal(0, 3, size=2)
            dets.append(Detection(xy, float(np.clip(rng.normal(0.55, 0.15), 0, 1)), False))
    rng.shuffle(dets)
    tt = {"scale": scale, "rotation_deg": rotation_deg,
          "translation": np.array(translation),
          "matrix_opt_to_sar": M, "matrix_sar_to_opt": np.linalg.inv(M)}
    return dets, tt, corroborable


# ---------------------------------------------------------------------------
# Assemble a pipeline Scene from a real Sentinel-2 stack
# ---------------------------------------------------------------------------
def build_scene_from_real(
    bs_optical: BandStack,
    bs_sar: BandStack | None = None,
    seed: int = 0,
) -> tuple[Scene, dict]:
    """
    Build a pipeline ``Scene`` from a real Sentinel-2 stack (Nagpur lake).

    * If ``bs_sar`` (real Sentinel-1) is given, both point sets are real; there is
      no ground truth, so ``true_debris`` is left empty (the app shows the fused
      confidence map rather than P/R/F1).
    * Otherwise the SAR point set is derived from the real optical detections
      (``simulate_sar_pointset``); the optical detections then serve as
      ground-truth floating matter, enabling the cross-modal-validation metrics on
      the real optical scene (a labelled semi-synthetic evaluation).
    """
    optical, diag = detect_optical(bs_optical)

    if bs_sar is not None:
        sar = detect_sar_backscatter(bs_sar, diag["water"])
        true_debris = np.empty((0, 2))
        tt = {"matrix_sar_to_opt": np.eye(3), "note": "fully-live: no ground truth"}
        mode = "live-s1+s2"
    else:
        sar, tt, corroborable = simulate_sar_pointset(optical, diag["water"], seed=seed)
        # Ground truth = floating matter visible to BOTH sensors (corroborable
        # debris). Optical-only over-reports the single-sensor objects; the
        # cross-modal methods should recover exactly this set -> a fair,
        # non-circular evaluation on the real optical scene.
        true_debris = (np.array([optical[i].xy for i in corroborable])
                       if corroborable else np.empty((0, 2)))
        mode = "real-s2 + simulated-s1"

    scene = Scene(true_debris=true_debris, sar=sar, optical=optical, true_transform=tt)
    diag["mode"] = mode
    diag["meta"] = bs_optical.meta
    return scene, diag
