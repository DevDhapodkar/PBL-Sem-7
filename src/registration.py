"""
Point-set registration for cross-modal SAR/optical debris matching.
===================================================================

This module implements, from scratch (numpy only), the three registration
building blocks the prototype compares:

    * ICP           -- the *traditional* iterative-closest-point rigid registration.
                       Classic, but has no outlier model: modality-specific false
                       positives (ships, sun-glint) drag the estimate off.

    * RANSAC        -- robust estimation of an initial similarity transform from
                       putative descriptor correspondences, explicitly rejecting
                       gross outliers.  This is *Stage 1* of the proposed pipeline.

    * CPD (rigid)   -- Coherent Point Drift, a probabilistic registration that
                       treats one point set as a Gaussian Mixture and softly
                       assigns the other, with a uniform-noise component that
                       absorbs outliers.  Initialised by the RANSAC transform,
                       this is *Stage 2* of the proposed pipeline.

Reference for CPD: Myronenko & Song, "Point Set Registration: Coherent Point
Drift", IEEE TPAMI 2010.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Similarity-transform helpers (scale * R + t, i.e. 4 DOF in 2-D)
# ---------------------------------------------------------------------------
def apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous transform to (N, 2) points."""
    if len(pts) == 0:
        return pts
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (h @ T.T)[:, :2]


def estimate_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Least-squares similarity transform (uniform scale + rotation + translation)
    mapping ``src`` onto ``dst`` (Umeyama, 1991). Needs >= 2 correspondences.
    Returns a 3x3 homogeneous matrix.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = (d_c.T @ s_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = (s_c ** 2).sum() / len(src)
    scale = np.trace(np.diag(D) @ S) / var_s if var_s > 1e-12 else 1.0
    t = mu_d - scale * R @ mu_s
    T = np.eye(3)
    T[:2, :2] = scale * R
    T[:2, 2] = t
    return T


# ---------------------------------------------------------------------------
# Shape-context descriptors -> putative cross-modal correspondences
# ---------------------------------------------------------------------------
def shape_context_descriptors(pts: np.ndarray, k: int = 6) -> np.ndarray:
    """
    A rotation/translation/scale-invariant local descriptor for each point:
    the *normalised, sorted* distances to its k nearest neighbours.

    Cross-modal detectors cannot rely on appearance, so we describe each debris
    candidate by the geometry of its neighbourhood. Sorting -> rotation
    invariance; dividing by the mean -> scale invariance. This lets us propose
    correspondences across two point sets related by an unknown similarity
    transform, which RANSAC then filters for geometric consistency.
    """
    n = len(pts)
    if n <= 1:
        return np.zeros((n, k))
    kk = min(k, n - 1)
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=kk + 1)   # first neighbour is the point itself
    d = d[:, 1:]
    scale = d.mean(1, keepdims=True)
    scale[scale < 1e-9] = 1.0
    desc = np.sort(d / scale, axis=1)
    if desc.shape[1] < k:              # pad if fewer neighbours than k
        desc = np.hstack([desc, np.zeros((n, k - desc.shape[1]))])
    return desc


def spatial_putative_matches(pts_a: np.ndarray, pts_b: np.ndarray,
                             radius: float = 55.0) -> np.ndarray:
    """
    Putative correspondences from spatial mutual-nearest-neighbours.

    Sentinel-1 and Sentinel-2 products are already geocoded, so the point sets
    differ only by a *residual* co-registration error (a small translation /
    rotation / scale). In that regime a genuine cross-modal pair is spatially
    close, so mutual-NN within ``radius`` yields putative matches that are mostly
    correct -- ideal seeds for RANSAC, which then removes the false-positive-to-
    false-positive spurious pairs. (For large, unknown transforms -- e.g. raw,
    un-geocoded scenes -- use ``putative_matches`` on shape-context descriptors
    instead.) Returns (M, 2) index pairs (i in A, j in B).
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return np.empty((0, 2), int)
    tree_b = cKDTree(pts_b)
    tree_a = cKDTree(pts_a)
    d_ab, j_ab = tree_b.query(pts_a, k=1)
    _, i_ba = tree_a.query(pts_b, k=1)
    matches = [(i, int(j_ab[i])) for i in range(len(pts_a))
               if i_ba[j_ab[i]] == i and d_ab[i] <= radius]
    return np.array(matches, int) if matches else np.empty((0, 2), int)


def putative_matches(desc_a: np.ndarray, desc_b: np.ndarray,
                     max_ratio: float = 0.95) -> np.ndarray:
    """
    Mutual-nearest-neighbour matches in descriptor space, with Lowe's ratio
    test to keep only distinctive matches. Returns an (M, 2) array of index
    pairs (i in A, j in B). These are *candidates* only -- many are wrong,
    which is precisely why RANSAC follows.
    """
    if len(desc_a) == 0 or len(desc_b) == 0:
        return np.empty((0, 2), int)
    tree_b = cKDTree(desc_b)
    tree_a = cKDTree(desc_a)
    # nearest + second nearest B for each A (ratio test)
    kk = min(2, len(desc_b))
    dist_ab, idx_ab = tree_b.query(desc_a, k=kk)
    if kk == 1:
        dist_ab = dist_ab[:, None]
        idx_ab = idx_ab[:, None]
    _, idx_ba = tree_a.query(desc_b, k=1)

    matches = []
    for i in range(len(desc_a)):
        j = idx_ab[i, 0]
        if idx_ba[j] != i:                       # enforce mutual NN
            continue
        if kk == 2 and dist_ab[i, 0] > max_ratio * (dist_ab[i, 1] + 1e-12):
            continue                              # not distinctive enough
        matches.append((i, j))
    return np.array(matches, int) if matches else np.empty((0, 2), int)


# ---------------------------------------------------------------------------
# Stage 1 : RANSAC robust similarity estimation
# ---------------------------------------------------------------------------
@dataclass
class RansacResult:
    T: np.ndarray                 # 3x3 similarity  src -> dst
    inliers: np.ndarray           # boolean mask over the putative matches
    n_inliers: int
    success: bool


def ransac_similarity(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    matches: np.ndarray,
    threshold: float = 15.0,
    max_iter: int = 2000,
    min_inliers: int = 4,
    seed: int | None = 0,
) -> RansacResult:
    """
    RANSAC over putative correspondences to estimate a robust similarity
    transform mapping ``src_pts`` onto ``dst_pts`` and to reject outlier
    correspondences.

    * ``matches``   : (M, 2) putative index pairs (src_idx, dst_idx).
    * ``threshold`` : inlier residual (in dst units) after transform.
    """
    rng = np.random.default_rng(seed)
    if len(matches) < 2:
        return RansacResult(np.eye(3), np.zeros(len(matches), bool), 0, False)

    src = src_pts[matches[:, 0]]
    dst = dst_pts[matches[:, 1]]
    best_inliers = np.zeros(len(matches), bool)
    best_count = 0
    n = len(matches)

    for _ in range(max_iter):
        idx = rng.choice(n, size=2, replace=False)   # 2 pairs define a similarity
        try:
            T = estimate_similarity(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        resid = np.linalg.norm(apply_transform(T, src) - dst, axis=1)
        inliers = resid < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    success = best_count >= min_inliers
    if success:                                       # refit on all inliers
        T = estimate_similarity(src[best_inliers], dst[best_inliers])
    else:
        T = np.eye(3)
    return RansacResult(T, best_inliers, best_count, success)


# ---------------------------------------------------------------------------
# Stage 2 : Coherent Point Drift (rigid + isotropic scale)
# ---------------------------------------------------------------------------
@dataclass
class CpdResult:
    T: np.ndarray                 # 3x3 similarity src -> dst (composed w/ init)
    transformed_src: np.ndarray   # source points after full transform
    P: np.ndarray                 # (M, N) soft correspondence probabilities
    sigma2: float                 # final GMM variance
    iterations: int


def cpd_rigid(
    source: np.ndarray,
    target: np.ndarray,
    init_T: np.ndarray | None = None,
    w: float = 0.4,
    max_iter: int = 100,
    tol: float = 1e-6,
    allow_scale: bool = True,
) -> CpdResult:
    """
    Rigid (optionally + isotropic scale) Coherent Point Drift.

    ``source`` (M x 2) is modelled as GMM centroids that coherently move to fit
    ``target`` (N x 2). A uniform component of weight ``w`` in [0, 1) absorbs
    outliers -- this is what makes CPD robust to the modality-specific false
    positives that break ICP.

    ``init_T`` (from RANSAC) pre-aligns the source so CPD starts inside the basin
    of attraction of the correct solution. The returned ``T`` is the *composition*
    of ``init_T`` with the refinement CPD estimates.
    """
    X = np.asarray(target, float)     # N x 2
    init_T = np.eye(3) if init_T is None else init_T
    Y0 = np.asarray(source, float)    # M x 2 (original)
    Y = apply_transform(init_T, Y0)   # warm-started source
    N, D = X.shape
    M = Y.shape[0]
    if M == 0 or N == 0:
        return CpdResult(init_T, Y, np.zeros((M, N)), 0.0, 0)

    # cumulative refinement (on top of init) : R, s, t
    R, s, t = np.eye(D), 1.0, np.zeros(D)
    TY = Y.copy()
    sigma2 = np.sum((X[None, :, :] - Y[:, None, :]) ** 2) / (D * N * M)
    c_out = (2 * np.pi * sigma2) ** (D / 2) * w / (1 - w) * M / N

    it = 0
    for it in range(1, max_iter + 1):
        # ---- E-step : responsibilities P (M x N) ----------------------------
        sq = np.sum((X[None, :, :] - TY[:, None, :]) ** 2, axis=2)      # M x N
        P = np.exp(-sq / (2 * sigma2))
        denom = P.sum(0, keepdims=True) + c_out + 1e-12
        P = P / denom
        Np = P.sum()
        if Np < 1e-9:
            break

        # ---- M-step : closed-form rigid(+scale) update ----------------------
        mu_x = (P @ X).sum(0) / Np           # actually (P^T 1)^T X / Np
        mu_x = (X * P.sum(0)[:, None]).sum(0) / Np
        mu_y = (Y * P.sum(1)[:, None]).sum(0) / Np
        Xc = X - mu_x
        Yc = Y - mu_y
        A = Xc.T @ (P.T @ Yc)                 # D x D
        U, Dsv, Vt = np.linalg.svd(A)
        C = np.eye(D)
        C[-1, -1] = np.sign(np.linalg.det(U @ Vt))
        R = U @ C @ Vt
        if allow_scale:
            YPY = np.sum(P.sum(1) * np.sum(Yc ** 2, axis=1))
            s = np.trace(np.diag(Dsv) @ C) / (YPY + 1e-12)
        t = mu_x - s * R @ mu_y
        TY = s * (Y @ R.T) + t

        # ---- variance update + convergence test -----------------------------
        trAR = np.trace(np.diag(Dsv) @ C)
        XPX = np.sum(P.sum(0) * np.sum(Xc ** 2, axis=1))
        sigma2_new = (XPX - s * trAR) / (Np * D)
        sigma2_new = max(sigma2_new, 1e-8)
        c_out = (2 * np.pi * sigma2_new) ** (D / 2) * w / (1 - w) * M / N
        if abs(sigma2_new - sigma2) < tol:
            sigma2 = sigma2_new
            break
        sigma2 = sigma2_new

    # compose refinement (on warm-started Y) with the RANSAC init
    T_refine = np.eye(3)
    T_refine[:2, :2] = s * R
    T_refine[:2, 2] = t
    T_full = T_refine @ init_T
    return CpdResult(T_full, TY, P, sigma2, it)


# ---------------------------------------------------------------------------
# Baseline : ICP (traditional rigid registration, no outlier model)
# ---------------------------------------------------------------------------
@dataclass
class IcpResult:
    T: np.ndarray
    transformed_src: np.ndarray
    rmse: float
    iterations: int


def icp_rigid(
    source: np.ndarray,
    target: np.ndarray,
    init_T: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    allow_scale: bool = True,
) -> IcpResult:
    """
    Classic iterative-closest-point rigid(+scale) registration -- the
    *traditional* method. Each iteration hard-assigns every source point to its
    nearest target point and refits a similarity. With no outlier/uniform
    component, ships and sun-glint act as phantom correspondences and bias the
    transform; ICP is also sensitive to the (here, identity) initialisation.
    """
    src = np.asarray(source, float)
    tgt = np.asarray(target, float)
    T = np.eye(3) if init_T is None else init_T.copy()
    cur = apply_transform(T, src)
    tree = cKDTree(tgt)
    prev = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        dist, idx = tree.query(cur, k=1)
        matched = tgt[idx]
        step = estimate_similarity(cur, matched) if allow_scale else _rigid_only(cur, matched)
        T = step @ T
        cur = apply_transform(T, src)
        rmse = float(np.sqrt(np.mean(dist ** 2)))
        if abs(prev - rmse) < tol:
            break
        prev = rmse
    dist, _ = tree.query(cur, k=1)
    return IcpResult(T, cur, float(np.sqrt(np.mean(dist ** 2))), it)


def _rigid_only(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotation + translation only (no scale) similarity fit."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    S = np.eye(2)
    if np.linalg.det(Vt.T @ U.T) < 0:
        S[-1, -1] = -1
    R = Vt.T @ S @ U.T
    t = mu_d - R @ mu_s
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
    return T
