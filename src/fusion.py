"""
Cross-modal validation & debris-confidence scoring.
===================================================

After the SAR point set has been registered into the optical frame, this module
decides *which* detections are corroborated across both sensors and assigns each
a debris-confidence score

    C_i = Q_reg * [ alpha * g_i + beta * S_SAR_i + gamma * S_opt_i ]

where

    Q_reg  = exp(-D_reg^2 / (2 tau^2)) * min(1, N_matches / N_support)
             global registration quality: shrinks *all* confidences if the two
             modalities could not be aligned (D_reg large) or if too few
             detections corroborate (fewer than N_support consistent matches).
             The support term *saturates* -- it asks "are there enough coherent
             cross-modal matches to trust the alignment?", not "what fraction of
             detections are debris" -- so a scene that is mostly clutter is not
             penalised for it. This is the term that makes the score honest: a
             failed registration cannot manufacture confident debris.
    g_i    = exp(-d_i^2 / (2 s^2))     geometric agreement of the specific pair
                                        (d_i = residual between the matched
                                        SAR & optical points after registration)
    S_SAR_i, S_opt_i                    per-detection evidence strengths.

A candidate seen in only ONE modality (no cross-modal partner) is heavily
discounted (factor ``single_penalty``) -- it is likely a modality-specific false
positive (ship, sun-glint). ``C_i > T`` => reported as debris.

This object-level, registration-gated cross-modal agreement is the part of the
pipeline put forward as novel for Sentinel-1/Sentinel-2 debris (see docs/NOVELTY.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .registration import apply_transform


@dataclass
class DebrisCandidate:
    xy: np.ndarray           # location in the optical/common frame
    confidence: float        # C_i in [0, 1]
    matched: bool            # corroborated across both modalities?
    s_sar: float             # SAR evidence (0 if unmatched & optical-only)
    s_opt: float             # optical evidence
    residual: float          # cross-modal registration residual (nan if unmatched)


@dataclass
class FusionResult:
    candidates: list[DebrisCandidate]
    d_registration: float    # global RMSE of matched pairs (D_reg)
    n_matches: int           # N_matches
    q_reg: float             # global registration-quality gate Q_reg

    def confident(self, threshold: float) -> list[DebrisCandidate]:
        return [c for c in self.candidates if c.confidence >= threshold]

    def locations(self, threshold: float) -> np.ndarray:
        pts = [c.xy for c in self.candidates if c.confidence >= threshold]
        return np.array(pts) if pts else np.empty((0, 2))


def cross_modal_validation(
    sar_in_opt: np.ndarray,       # SAR detections registered into optical frame (M, 2)
    sar_strength: np.ndarray,     # (M,)
    optical_xy: np.ndarray,       # optical detections (N, 2)
    optical_strength: np.ndarray, # (N,)
    match_radius: float = 20.0,   # spatial tolerance for "agreement" (opt units)
    alpha: float = 0.45,          # weight: geometric agreement
    beta: float = 0.275,          # weight: SAR evidence
    gamma: float = 0.275,         # weight: optical evidence
    tau: float = 25.0,            # registration-quality length scale
    n_support: float = 6.0,       # matches needed to saturate registration trust
    single_penalty: float = 0.35, # discount for single-modality candidates
) -> FusionResult:
    """
    Match registered SAR detections to optical detections (mutual nearest
    neighbour within ``match_radius``) and score every candidate.
    """
    M, N = len(sar_in_opt), len(optical_xy)

    # --- mutual-nearest-neighbour cross-modal matching -----------------------
    pairs: list[tuple[int, int, float]] = []
    matched_sar = np.zeros(M, bool)
    matched_opt = np.zeros(N, bool)
    if M and N:
        tree_opt = cKDTree(optical_xy)
        tree_sar = cKDTree(sar_in_opt)
        d_so, j_so = tree_opt.query(sar_in_opt, k=1)   # nearest opt for each sar
        _, i_os = tree_sar.query(optical_xy, k=1)      # nearest sar for each opt
        for i in range(M):
            j = int(j_so[i])
            if i_os[j] == i and d_so[i] <= match_radius:   # mutual + within radius
                pairs.append((i, j, float(d_so[i])))
                matched_sar[i] = matched_opt[j] = True

    # --- global registration-quality gate Q_reg ------------------------------
    residuals = np.array([p[2] for p in pairs]) if pairs else np.array([])
    d_reg = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) else float("inf")
    n_matches = len(pairs)
    if n_matches == 0:
        q_reg = 0.0
    else:
        support = min(1.0, n_matches / n_support)
        q_reg = np.exp(-d_reg ** 2 / (2 * tau ** 2)) * support
    q_reg = float(np.clip(q_reg, 0.0, 1.0))

    s_geom = max(match_radius / 2.0, 1e-6)
    candidates: list[DebrisCandidate] = []

    # --- corroborated (matched) candidates -----------------------------------
    for i, j, d in pairs:
        g = float(np.exp(-d ** 2 / (2 * s_geom ** 2)))
        s_sar = float(sar_strength[i])
        s_opt = float(optical_strength[j])
        C = q_reg * (alpha * g + beta * s_sar + gamma * s_opt)
        xy = 0.5 * (sar_in_opt[i] + optical_xy[j])       # fused location
        candidates.append(DebrisCandidate(xy, float(np.clip(C, 0, 1)),
                                          True, s_sar, s_opt, d))

    # --- single-modality candidates (discounted) -----------------------------
    for i in range(M):
        if matched_sar[i]:
            continue
        s_sar = float(sar_strength[i])
        C = q_reg * single_penalty * (beta / (beta + gamma)) * s_sar
        candidates.append(DebrisCandidate(sar_in_opt[i], float(np.clip(C, 0, 1)),
                                          False, s_sar, 0.0, float("nan")))
    for j in range(N):
        if matched_opt[j]:
            continue
        s_opt = float(optical_strength[j])
        C = q_reg * single_penalty * (gamma / (beta + gamma)) * s_opt
        candidates.append(DebrisCandidate(optical_xy[j], float(np.clip(C, 0, 1)),
                                          False, 0.0, s_opt, float("nan")))

    return FusionResult(candidates, d_reg, n_matches, q_reg)
