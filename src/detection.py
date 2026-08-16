"""
Detector interface -- where REAL Sentinel-1 / Sentinel-2 data plugs in.
======================================================================

The registration + fusion pipeline consumes *object-level point sets* with a
per-object strength score. This module documents and stubs the two independent
detectors that would produce those point sets from real imagery, so the
prototype's synthetic front-end (``data_simulation.py``) can be swapped for real
Sentinel scenes without touching the pipeline.

Sentinel-1 (SAR) debris cue
---------------------------
Floating debris / rafts modulate the sea-surface roughness and appear as
localised **backscatter anomalies** (often *darker* damping slicks, sometimes
brighter) in the calibrated VV/VH GRD product. A practical detector:
    1. Radiometric calibration -> sigma-0, speckle filtering (e.g. refined Lee).
    2. Local contrast / CFAR anomaly detection against the surrounding sea.
    3. Blob extraction -> centroids = SAR point set; anomaly magnitude -> strength.

Sentinel-2 (optical) debris cue
-------------------------------
Floating plastic-rich debris has a characteristic reflectance in the NIR/SWIR.
Standard spectral indices:
    * FDI  (Floating Debris Index)  -- Biermann et al., 2020
    * NDVI / plastic index          -- vegetation-like NIR bump on water
A detector thresholds FDI (with cloud/sun-glint masking) -> blobs -> centroids =
optical point set; index value -> strength.

Both detectors output the same ``Detection`` records used everywhere else, so the
RANSAC+CPD+fusion pipeline is agnostic to whether the points came from the
simulator or from real ``.SAFE`` products read with rasterio / snappy.
"""

from __future__ import annotations

import numpy as np

from .data_simulation import Detection


def detections_from_arrays(xy: np.ndarray, strength: np.ndarray) -> list[Detection]:
    """Wrap raw (N,2) centroids + (N,) scores as Detection records.

    Use this to feed a *real* detector's output into the pipeline:

        sar_dets = detections_from_arrays(sar_centroids, sar_scores)
        opt_dets = detections_from_arrays(opt_centroids, opt_scores)
        scene = Scene(true_debris=<optional/eval only>, sar=sar_dets,
                      optical=opt_dets, true_transform={})
        result = run_proposed(scene)
    """
    xy = np.asarray(xy, float).reshape(-1, 2)
    strength = np.asarray(strength, float).reshape(-1)
    if len(xy) != len(strength):
        raise ValueError("xy and strength must have equal length")
    return [Detection(p, float(s), False) for p, s in zip(xy, strength)]


# ---- Real-detector stubs (raise until implemented) ------------------------
def detect_sar(sigma0_vv: np.ndarray, **kw) -> list[Detection]:  # pragma: no cover
    """CFAR/anomaly SAR debris detector. Implement against real S1 GRD data."""
    raise NotImplementedError(
        "Plug a real Sentinel-1 CFAR/anomaly detector here; return Detection list."
    )


def detect_optical(bands: dict, **kw) -> list[Detection]:  # pragma: no cover
    """FDI-based optical debris detector. Implement against real S2 L2A bands."""
    raise NotImplementedError(
        "Plug a real Sentinel-2 FDI detector here; return Detection list."
    )
