"""
Proposed pipeline: RANSAC init -> CPD refinement -> cross-modal validation.
===========================================================================

    Sentinel-1 SAR  --detector-->  SAR point set  --.
                                                      \
    Sentinel-2 optical --detector--> optical point set >-- Stage 1: RANSAC init
                                                      /     Stage 2: CPD refine
                                                     '      Stage 3: cross-modal
                                                             validation -> C_i

Each stage has a *distinct* job (the design point the task description stresses):

    Stage 1 (RANSAC) : from putative shape-context correspondences, estimate a
                       robust initial similarity transform and reject gross
                       outlier correspondences.
    Stage 2 (CPD)    : starting from the RANSAC transform, probabilistically
                       refine the alignment of the two imperfect distributions,
                       with a uniform component absorbing the remaining outliers.
    Stage 3 (fusion) : match registered detections, gate by registration quality,
                       and emit per-candidate debris confidence C_i.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_simulation import Scene
from .fusion import FusionResult, cross_modal_validation
from .registration import (
    apply_transform,
    cpd_rigid,
    putative_matches,
    ransac_similarity,
    shape_context_descriptors,
    spatial_putative_matches,
)


@dataclass
class PipelineResult:
    T_ransac: np.ndarray
    T_final: np.ndarray            # RANSAC composed with CPD refinement
    n_putative: int
    n_ransac_inliers: int
    ransac_success: bool
    cpd_iterations: int
    fusion: FusionResult
    sar_in_opt: np.ndarray         # SAR detections registered into optical frame


def run_proposed(
    scene: Scene,
    *,
    putative_radius: float = 55.0,
    desc_k: int = 6,
    use_descriptors: bool = False,
    ransac_threshold: float = 15.0,
    cpd_w: float = 0.4,
    match_radius: float = 20.0,
    confidence_threshold: float = 0.30,
    verbose: bool = False,
) -> PipelineResult:
    """Run the full proposed pipeline on a scene. Registers SAR -> optical frame.

    ``use_descriptors=False`` (default) forms putative RANSAC correspondences by
    spatial proximity -- correct for geocoded S1/S2 with a residual offset.
    Set ``use_descriptors=True`` for large, unknown transforms (raw scenes),
    which forms them from rotation/scale-invariant shape-context descriptors.
    """
    sar_xy = scene.sar_xy
    opt_xy = scene.optical_xy

    # ---- Stage 0: putative correspondences ----------------------------------
    if use_descriptors:
        desc_sar = shape_context_descriptors(sar_xy, k=desc_k)
        desc_opt = shape_context_descriptors(opt_xy, k=desc_k)
        matches = putative_matches(desc_sar, desc_opt)
    else:
        matches = spatial_putative_matches(sar_xy, opt_xy, radius=putative_radius)

    # ---- Stage 1: RANSAC robust initial similarity (SAR -> optical) ---------
    ransac = ransac_similarity(sar_xy, opt_xy, matches, threshold=ransac_threshold)
    T_init = ransac.T if ransac.success else np.eye(3)

    # ---- Stage 2: CPD refinement, warm-started by RANSAC --------------------
    # RANSAC already fixed the scale, so CPD refines the RIGID pose only
    # (rotation + translation). Rigid CPD has no scale-shrink degeneracy and its
    # uniform-noise component absorbs the residual clutter RANSAC did not use.
    cpd = cpd_rigid(sar_xy, opt_xy, init_T=T_init, w=cpd_w, allow_scale=False)
    T_final = cpd.T
    sar_in_opt = apply_transform(T_final, sar_xy)

    # ---- Stage 3: cross-modal validation & confidence -----------------------
    fusion = cross_modal_validation(
        sar_in_opt, scene.sar_strength,
        opt_xy, scene.optical_strength,
        match_radius=match_radius,
    )

    if verbose:
        print(f"[proposed] putative matches   : {len(matches)}")
        print(f"[proposed] RANSAC inliers      : {ransac.n_inliers} "
              f"(success={ransac.success})")
        print(f"[proposed] CPD iterations      : {cpd.iterations}")
        print(f"[proposed] cross-modal matches : {fusion.n_matches}")
        print(f"[proposed] D_reg (RMSE)        : {fusion.d_registration:.2f}")
        print(f"[proposed] Q_reg gate          : {fusion.q_reg:.3f}")

    return PipelineResult(
        T_ransac=ransac.T,
        T_final=T_final,
        n_putative=len(matches),
        n_ransac_inliers=ransac.n_inliers,
        ransac_success=ransac.success,
        cpd_iterations=cpd.iterations,
        fusion=fusion,
        sar_in_opt=sar_in_opt,
    )
