"""
Unit / integration tests for the debris-detection prototype.
Run with:  python -m pytest -q   (or)   python tests/test_pipeline.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline import run_optical_only, run_traditional          # noqa: E402
from src.data_simulation import simulate_scene                       # noqa: E402
from src.evaluate import score_locations                            # noqa: E402
from src.fusion import cross_modal_validation                       # noqa: E402
from src.pipeline import run_proposed                               # noqa: E402
from src.registration import (                                      # noqa: E402
    apply_transform,
    cpd_rigid,
    estimate_similarity,
    ransac_similarity,
    spatial_putative_matches,
)


# --------------------------------------------------------------------------
def test_estimate_similarity_recovers_known_transform():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, size=(20, 2))
    T = np.array([[1.2 * np.cos(0.3), -1.2 * np.sin(0.3), 10.0],
                  [1.2 * np.sin(0.3), 1.2 * np.cos(0.3), -5.0],
                  [0, 0, 1]])
    dst = apply_transform(T, src)
    T_est = estimate_similarity(src, dst)
    assert np.allclose(T_est, T, atol=1e-6)


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 100, size=(30, 2))
    T = np.array([[1.0, 0.0, 15.0], [0.0, 1.0, -8.0], [0, 0, 1]])
    dst = apply_transform(T, src)
    # corrupt 40% of correspondences
    dst_noisy = dst.copy()
    outliers = rng.choice(30, size=12, replace=False)
    dst_noisy[outliers] += rng.uniform(-200, 200, size=(12, 2))
    matches = np.column_stack([np.arange(30), np.arange(30)])
    res = ransac_similarity(src, dst_noisy, matches, threshold=5.0)
    assert res.success
    # recovered translation close to truth despite 40% outliers
    assert np.allclose(res.T[:2, 2], [15.0, -8.0], atol=2.0)
    assert res.n_inliers >= 15


def test_cpd_refines_toward_target():
    rng = np.random.default_rng(2)
    tgt = rng.uniform(0, 100, size=(40, 2))
    theta = np.deg2rad(5)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src = (tgt @ R.T) + np.array([7.0, -4.0])          # rigidly offset copy
    res = cpd_rigid(src, tgt, w=0.1, allow_scale=False)
    err = np.linalg.norm(res.transformed_src - tgt, axis=1).mean()
    assert err < 2.0                                    # CPD pulls source onto target


def test_cpd_rigid_does_not_collapse():
    """Rigid CPD must not shrink the source to a point (scale-collapse guard)."""
    rng = np.random.default_rng(3)
    tgt = rng.uniform(0, 100, size=(30, 2))
    src = tgt + np.array([5.0, 5.0])
    res = cpd_rigid(src, tgt, w=0.3, allow_scale=False)
    spread = res.transformed_src.std(0)
    assert np.all(spread > 10.0)                        # retains spatial extent


def test_fusion_gate_penalises_failed_registration():
    """If SAR points are grossly mis-registered, Q_reg -> ~0 and confidence low."""
    opt = np.array([[0, 0], [50, 50], [100, 0], [30, 80]], float)
    sar_bad = opt + 500.0                               # nowhere near optical
    r = cross_modal_validation(sar_bad, np.ones(4), opt, np.ones(4), match_radius=20)
    assert r.n_matches == 0
    assert r.q_reg == 0.0
    assert all(c.confidence < 0.2 for c in r.candidates)


def test_proposed_beats_icp_on_average():
    """Core claim: RANSAC+CPD registration yields a higher-F1 debris map than ICP,
    holding the fusion stage fixed, averaged over several scenes."""
    f1_prop, f1_icp = [], []
    for s in range(8):
        sc = simulate_scene(seed=300 + s)
        prop = run_proposed(sc, confidence_threshold=0.30)
        trad = run_traditional(sc)
        pp = prop.fusion.locations(0.30)
        tp = np.array([c.xy for c in trad.fusion.candidates if c.confidence >= 0.30]) \
            if trad.fusion.candidates else np.empty((0, 2))
        f1_prop.append(score_locations(pp, sc.true_debris, 20).f1)
        f1_icp.append(score_locations(tp if len(tp) else np.empty((0, 2)),
                                      sc.true_debris, 20).f1)
    assert np.mean(f1_prop) > np.mean(f1_icp) + 0.10


def test_proposed_precision_high():
    """Cross-modal validation should keep debris precision high (few false alarms)."""
    precisions = []
    for s in range(8):
        sc = simulate_scene(seed=400 + s)
        prop = run_proposed(sc, confidence_threshold=0.30)
        pts = prop.fusion.locations(0.30)
        precisions.append(score_locations(pts, sc.true_debris, 20).precision)
    assert np.mean(precisions) > 0.80


def test_optical_only_runs():
    sc = simulate_scene(seed=7)
    res = run_optical_only(sc)
    assert res.fusion is not None
    assert len(res.fusion.candidates) == len(sc.optical)


def test_nagpur_real_scene_pipeline():
    """The bundled REAL Sentinel-2 Nagpur scene loads, detectors produce a point
    set, and the pipeline beats ICP on the corroborable floating matter."""
    try:
        from src.acquire import load_cached_s2
        from src.detect_real import build_scene_from_real
    except Exception:
        return  # optional geo deps (rasterio/pyproj) not installed
    bs = load_cached_s2("ambazari")
    if bs is None:
        return  # cache not present
    assert bs.meta.get("live")                      # it is real Sentinel-2
    assert "B08" in bs.bands and bs["B08"].ndim == 2

    f_prop, f_icp = [], []
    for s in range(4):
        scene, diag = build_scene_from_real(bs, seed=s)
        assert len(scene.optical) > 5               # real floating-matter detections
        prop = run_proposed(scene, putative_radius=14, ransac_threshold=4, match_radius=5)
        trad = run_traditional(scene, match_radius=5)
        pp = prop.fusion.locations(0.30)
        tp = np.array([c.xy for c in trad.fusion.candidates if c.confidence >= 0.30]) \
            if trad.fusion.candidates else np.empty((0, 2))
        f_prop.append(score_locations(pp, scene.true_debris, 5).f1)
        f_icp.append(score_locations(tp if len(tp) else np.empty((0, 2)),
                                     scene.true_debris, 5).f1)
    assert np.mean(f_prop) >= np.mean(f_icp)        # proposed >= ICP on real data


# --------------------------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    sys.exit(1 if failed else 0)
