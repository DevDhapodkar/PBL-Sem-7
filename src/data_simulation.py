"""
Synthetic Sentinel-1 (SAR) / Sentinel-2 (optical) debris-detection simulator.
=============================================================================

Real Sentinel-1 and Sentinel-2 products are *nominally* geo-referenced, but in
practice a floating-debris pipeline that runs an *independent* detector on each
modality ends up with two **object-level point sets** that do NOT line up:

  * The two sensors image at different times, geometries and resolutions, so the
    same debris slick appears slightly shifted / rotated / scaled between the two
    detection point sets (a residual similarity mis-registration).
  * Each detector produces **false positives** that are modality specific:
        - SAR   -> ships, wind streaks, internal waves  (bright in SAR, absent in optical)
        - Optical -> sun-glint, thin cloud, turbid water (bright in optical, absent in SAR)
  * Each detector **misses** some real debris (detection probability < 1).
  * Detected positions are corrupted by localisation noise.

This module generates a controlled ground-truth scene together with the two
imperfect detection point sets, so the registration + cross-modal validation
pipeline can be evaluated against a known answer.

Everything downstream consumes plain (N, 2) numpy arrays plus per-detection
"strength" scores, so the *same* code path works on real detections once a real
detector is plugged in (see ``src/detection.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """A single object-level detection in one modality."""

    xy: np.ndarray          # (2,) position in that modality's pixel/geo frame
    strength: float         # detector confidence in [0, 1]
    is_true_debris: bool    # bookkeeping ONLY -- never seen by the pipeline
    descriptor: np.ndarray = field(default=None)  # local shape-context descriptor


@dataclass
class Scene:
    """A full simulated scene with ground truth and both detection point sets."""

    true_debris: np.ndarray            # (K, 2) ground-truth debris in the OPTICAL frame
    sar: list[Detection]               # SAR detections in the SAR frame
    optical: list[Detection]           # optical detections in the OPTICAL frame
    true_transform: dict               # SAR->optical similarity that the pipeline must recover

    # ---- convenience views -------------------------------------------------
    @property
    def sar_xy(self) -> np.ndarray:
        return np.array([d.xy for d in self.sar]) if self.sar else np.empty((0, 2))

    @property
    def optical_xy(self) -> np.ndarray:
        return np.array([d.xy for d in self.optical]) if self.optical else np.empty((0, 2))

    @property
    def sar_strength(self) -> np.ndarray:
        return np.array([d.strength for d in self.sar])

    @property
    def optical_strength(self) -> np.ndarray:
        return np.array([d.strength for d in self.optical])


def _similarity_matrix(scale: float, theta_deg: float, tx: float, ty: float) -> np.ndarray:
    """Return the 3x3 homogeneous similarity transform (scale * R | t)."""
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    M = np.eye(3)
    M[:2, :2] = scale * R
    M[:2, 2] = [tx, ty]
    return M


def _apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous transform to (N, 2) points."""
    if len(pts) == 0:
        return pts
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (h @ M.T)[:, :2]


def simulate_scene(
    n_debris: int = 40,
    field_size: float = 1000.0,
    # --- SAR<-optical geometric mis-registration the pipeline must recover ---
    # Realistic RESIDUAL co-registration error between two geocoded products:
    # small scale/rotation, a tens-of-metres translation offset.
    scale: float = 1.03,
    rotation_deg: float = 4.0,
    translation: tuple[float, float] = (30.0, -18.0),
    # --- detector behaviour --------------------------------------------------
    p_detect_sar: float = 0.78,
    p_detect_optical: float = 0.82,
    loc_noise_sar: float = 6.0,
    loc_noise_optical: float = 4.0,
    n_false_sar: int = 25,        # ships / wind streaks
    n_false_optical: int = 30,    # sun-glint / thin cloud
    seed: int | None = 42,
) -> Scene:
    """
    Build a synthetic scene.

    The optical frame is treated as the common reference frame. Ground-truth
    debris live there. The SAR frame is the optical frame pushed through a
    similarity transform (``true_transform``); the SAR detector then samples a
    noisy, incomplete subset of the debris in *that* frame and adds its own
    false positives. The pipeline never sees ``true_transform`` -- recovering it
    (robustly, despite the false positives) is exactly the registration task.
    """
    rng = np.random.default_rng(seed)

    # 1. Ground-truth debris field (optical / common frame) -------------------
    true_debris = rng.uniform(0, field_size, size=(n_debris, 2))

    # 2. SAR<-optical similarity transform ------------------------------------
    M_opt_to_sar = _similarity_matrix(scale, rotation_deg, *translation)
    true_transform = {
        "scale": scale,
        "rotation_deg": rotation_deg,
        "translation": np.array(translation),
        "matrix_opt_to_sar": M_opt_to_sar,
        "matrix_sar_to_opt": np.linalg.inv(M_opt_to_sar),
    }

    # 3. Optical detections (in optical frame) --------------------------------
    optical: list[Detection] = []
    for p in true_debris:
        if rng.random() < p_detect_optical:
            xy = p + rng.normal(0, loc_noise_optical, size=2)
            strength = float(np.clip(rng.normal(0.80, 0.10), 0, 1))  # strong FDI/NDVI-style evidence
            optical.append(Detection(xy, strength, True))
    for _ in range(n_false_optical):  # sun-glint / cloud false positives
        xy = rng.uniform(0, field_size, size=2)
        strength = float(np.clip(rng.normal(0.45, 0.15), 0, 1))
        optical.append(Detection(xy, strength, False))

    # 4. SAR detections (in SAR frame) ----------------------------------------
    debris_in_sar = _apply(M_opt_to_sar, true_debris)
    sar: list[Detection] = []
    for p in debris_in_sar:
        if rng.random() < p_detect_sar:
            xy = p + rng.normal(0, loc_noise_sar, size=2)
            strength = float(np.clip(rng.normal(0.75, 0.12), 0, 1))  # backscatter anomaly
            sar.append(Detection(xy, strength, True))
    # SAR false positives live in the SAR frame too (ships, wind streaks)
    sar_frame_lo = _apply(M_opt_to_sar, np.array([[0, 0]]))[0]
    sar_frame_hi = _apply(M_opt_to_sar, np.array([[field_size, field_size]]))[0]
    for _ in range(n_false_sar):
        xy = rng.uniform(np.minimum(sar_frame_lo, sar_frame_hi),
                         np.maximum(sar_frame_lo, sar_frame_hi), size=2)
        strength = float(np.clip(rng.normal(0.55, 0.15), 0, 1))
        sar.append(Detection(xy, strength, False))

    rng.shuffle(sar)
    rng.shuffle(optical)
    return Scene(true_debris, sar, optical, true_transform)


if __name__ == "__main__":
    sc = simulate_scene()
    print(f"true debris        : {len(sc.true_debris)}")
    print(f"SAR detections     : {len(sc.sar)}  "
          f"({sum(d.is_true_debris for d in sc.sar)} real / "
          f"{sum(not d.is_true_debris for d in sc.sar)} false)")
    print(f"optical detections : {len(sc.optical)}  "
          f"({sum(d.is_true_debris for d in sc.optical)} real / "
          f"{sum(not d.is_true_debris for d in sc.optical)} false)")
    print(f"true SAR->opt shift: {sc.true_transform['translation']}, "
          f"rot {sc.true_transform['rotation_deg']} deg, scale {sc.true_transform['scale']}")
