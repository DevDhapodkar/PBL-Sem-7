"""
Baseline / traditional methods to compare against the proposed pipeline.
========================================================================

Two reference points, chosen so the comparison isolates *what the proposed
pipeline adds*:

1. ``run_optical_only`` -- single-modality detection: threshold the optical
   detector alone. This is the "no fusion" world. It shows why cross-modal
   corroboration is needed at all: optical sun-glint / cloud false positives are
   reported as debris because nothing contradicts them.

2. ``run_traditional`` -- classic cross-modal fusion using **ICP** for
   registration instead of RANSAC+CPD, then the *same* spatial matching and
   decision as the proposed method. Because ICP has no outlier model and starts
   from the identity, the modality-specific false positives and the SAR<-optical
   mis-registration pull the alignment off, so genuinely corresponding
   detections fail to overlap. Holding Stage 3 fixed means any performance gap is
   attributable to the registration stage (RANSAC+CPD vs ICP).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_simulation import Scene
from .fusion import FusionResult, cross_modal_validation
from .registration import apply_transform, icp_rigid


@dataclass
class BaselineResult:
    method: str
    fusion: FusionResult | None
    sar_in_opt: np.ndarray | None
    icp_rmse: float | None = None


def run_optical_only(scene: Scene) -> BaselineResult:
    """Single-modality baseline: every optical detection is a candidate, scored
    by its own strength only (no cross-modal check)."""
    from .fusion import DebrisCandidate

    cands = [
        DebrisCandidate(d.xy, float(d.strength), False, 0.0, float(d.strength), float("nan"))
        for d in scene.optical
    ]
    fusion = FusionResult(cands, d_registration=float("nan"),
                          n_matches=0, q_reg=1.0)
    return BaselineResult("optical_only", fusion, None)


def run_traditional(
    scene: Scene,
    *,
    match_radius: float = 20.0,
) -> BaselineResult:
    """Traditional cross-modal fusion using ICP registration (no robust init)."""
    sar_xy = scene.sar_xy
    opt_xy = scene.optical_xy

    icp = icp_rigid(sar_xy, opt_xy, init_T=None)   # cold-start, no outlier model
    sar_in_opt = apply_transform(icp.T, sar_xy)

    fusion = cross_modal_validation(
        sar_in_opt, scene.sar_strength,
        opt_xy, scene.optical_strength,
        match_radius=match_radius,
    )
    return BaselineResult("traditional_icp", fusion, sar_in_opt, icp.rmse)
