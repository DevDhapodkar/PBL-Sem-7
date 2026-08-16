#!/usr/bin/env python3
"""
Interactive dashboard for the cross-modal Sentinel-1/Sentinel-2 debris pipeline.
================================================================================

A Streamlit UI that runs the *real* pipeline (the same ``src/`` modules used by
``run_demo.py``) live: adjust the scene, clutter and residual mis-registration,
watch RANSAC+CPD align the SAR and optical detections, tune the confidence
threshold, and compare against the traditional ICP method in real time.

    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.baseline import run_optical_only, run_traditional
from src.data_simulation import simulate_scene
from src.evaluate import (
    best_f1,
    precision_recall_curve,
    score_locations,
    strength_pr_curve,
)
from src.pipeline import run_proposed
from src import visualize

MATCH_RADIUS = 20.0

st.set_page_config(page_title="Sentinel Debris Fusion",
                   page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
      .hero {
        background: linear-gradient(110deg,#0b3d5c 0%,#12708f 55%,#1a9db0 100%);
        padding: 1.1rem 1.4rem; border-radius: 14px; color:#fff;
        margin-bottom: 1.0rem;}
      .hero h1 {margin:0; font-size:1.55rem; font-weight:700;}
      .hero p  {margin:.35rem 0 0; opacity:.92; font-size:.95rem;}
      .pill {display:inline-block; background:rgba(255,255,255,.16);
        border-radius:999px; padding:.12rem .6rem; font-size:.72rem;
        margin-right:.35rem;}
      .stage {border-left:4px solid #1a9db0; padding:.15rem 0 .15rem .7rem;
        margin:.35rem 0;}
      div[data-testid="stMetric"] {background:#f5f8fa; border:1px solid #e3eaf0;
        border-radius:12px; padding:.6rem .8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🛰️ Cross-Modal Sentinel-1 / Sentinel-2 Marine-Debris Detection</h1>
      <p>Independent SAR &amp; optical detection → object-level point sets →
         <b>RANSAC</b> robust init → <b>CPD</b> refinement →
         registration-gated cross-modal confidence
         <b>C = f(D<sub>reg</sub>, N<sub>matches</sub>, S<sub>SAR</sub>, S<sub>opt</sub>)</b>.</p>
      <span class="pill">RANSAC + CPD</span>
      <span class="pill">vs traditional ICP</span>
      <span class="pill">live, on the real pipeline</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Scene")
    seed = st.number_input("Random seed", 0, 9999, 42, step=1)
    n_debris = st.slider("Ground-truth debris", 10, 100, 40, 5)
    c1, c2 = st.columns(2)
    p_sar = c1.slider("P(detect) SAR", 0.3, 1.0, 0.78, 0.02)
    p_opt = c2.slider("P(detect) optical", 0.3, 1.0, 0.82, 0.02)
    c3, c4 = st.columns(2)
    n_false_sar = c3.slider("SAR clutter", 0, 100, 25, 5,
                            help="ships / wind streaks / internal waves")
    n_false_opt = c4.slider("Optical clutter", 0, 100, 30, 5,
                            help="sun-glint / thin cloud / turbid water")

    st.header("🧭 Residual mis-registration")
    st.caption("S1 & S2 are geocoded — this is the residual offset the pipeline recovers.")
    scale = st.slider("Scale", 0.90, 1.15, 1.03, 0.01)
    rotation = st.slider("Rotation (°)", 0.0, 20.0, 4.0, 0.5)
    tx = st.slider("Shift X", -100.0, 100.0, 30.0, 5.0)
    ty = st.slider("Shift Y", -100.0, 100.0, -18.0, 5.0)

    st.header("🔧 Pipeline")
    ransac_thr = st.slider("RANSAC inlier threshold", 5.0, 40.0, 15.0, 1.0)
    cpd_w = st.slider("CPD outlier weight w", 0.0, 0.9, 0.4, 0.05,
                      help="uniform-noise component that absorbs clutter")
    match_radius = st.slider("Cross-modal match radius", 5.0, 50.0, MATCH_RADIUS, 1.0)

    st.header("🎯 Decision")
    conf_thr = st.slider("Confidence threshold  T", 0.0, 1.0, 0.30, 0.02)

    st.divider()
    st.caption("Every change re-runs the actual `src/` pipeline.")


# ---------------------------------------------------------------------------
# Cached compute
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute(seed, n_debris, p_sar, p_opt, n_false_sar, n_false_opt,
            scale, rotation, tx, ty, ransac_thr, cpd_w, match_radius):
    scene = simulate_scene(
        n_debris=n_debris, p_detect_sar=p_sar, p_detect_optical=p_opt,
        n_false_sar=n_false_sar, n_false_optical=n_false_opt,
        scale=scale, rotation_deg=rotation, translation=(tx, ty), seed=int(seed))
    optical = run_optical_only(scene)
    traditional = run_traditional(scene, match_radius=match_radius)
    proposed = run_proposed(scene, ransac_threshold=ransac_thr, cpd_w=cpd_w,
                            match_radius=match_radius)
    scored = {
        "optical_only": [(d.xy, d.strength) for d in scene.optical],
        "traditional_icp": [(c.xy, c.confidence) for c in traditional.fusion.candidates],
        "proposed": [(c.xy, c.confidence) for c in proposed.fusion.candidates],
    }
    return scene, optical, traditional, proposed, scored


@st.cache_data(show_spinner=True)
def robustness(levels, trials, base_seed=1000):
    keys = ("proposed", "traditional_icp", "optical_only")
    out = {"x": list(levels)}
    for m in ("precision", "f1"):
        for k in keys:
            out[f"{k}_{m}"] = []
    for load in levels:
        acc = {f"{k}_{m}": [] for k in keys for m in ("precision", "f1")}
        for s in range(trials):
            sc = simulate_scene(n_false_sar=load, n_false_optical=int(load * 1.2),
                                seed=base_seed + s)
            prop = run_proposed(sc); trad = run_traditional(sc); opt = run_optical_only(sc)
            reported = {
                "proposed": prop.fusion.locations(0.30),
                "traditional_icp": np.array([c.xy for c in trad.fusion.candidates
                                             if c.confidence >= 0.30])
                if trad.fusion.candidates else np.empty((0, 2)),
                "optical_only": np.array([d.xy for d in sc.optical if d.strength >= 0.5]),
            }
            for k in keys:
                m = score_locations(reported[k] if len(reported[k]) else np.empty((0, 2)),
                                    sc.true_debris, MATCH_RADIUS)
                acc[f"{k}_precision"].append(m.precision)
                acc[f"{k}_f1"].append(m.f1)
        for key, vals in acc.items():
            out[key].append(float(np.mean(vals)))
    return out


scene, optical, traditional, proposed, scored = compute(
    seed, n_debris, p_sar, p_opt, n_false_sar, n_false_opt,
    scale, rotation, tx, ty, ransac_thr, cpd_w, match_radius)

truth = scene.true_debris


def metrics_for(name):
    thr = 0.5 if name == "optical_only" else conf_thr
    pts = np.array([xy for xy, s in scored[name] if s >= thr]) \
        if scored[name] else np.empty((0, 2))
    return score_locations(pts if len(pts) else np.empty((0, 2)), truth, match_radius)


m_opt = metrics_for("optical_only")
m_trad = metrics_for("traditional_icp")
m_prop = metrics_for("proposed")


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
st.subheader("Live scorecard")
k = st.columns(4)
k[0].metric("Proposed  F1", f"{m_prop.f1:.3f}",
            f"{m_prop.f1 - m_trad.f1:+.3f} vs ICP")
k[1].metric("Proposed  precision", f"{m_prop.precision:.3f}",
            f"{m_prop.precision - m_trad.precision:+.3f} vs ICP")
k[2].metric("Proposed  recall", f"{m_prop.recall:.3f}",
            f"{m_prop.recall - m_trad.recall:+.3f} vs ICP")
k[3].metric("Cross-modal matches", f"{proposed.fusion.n_matches}",
            f"Q_reg = {proposed.fusion.q_reg:.2f}")

st.caption(
    f"Scene: {len(scene.sar)} SAR detections "
    f"({sum(d.is_true_debris for d in scene.sar)} real / "
    f"{sum(not d.is_true_debris for d in scene.sar)} clutter) · "
    f"{len(scene.optical)} optical detections "
    f"({sum(d.is_true_debris for d in scene.optical)} real / "
    f"{sum(not d.is_true_debris for d in scene.optical)} clutter) · "
    f"{len(truth)} true debris.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_reg, tab_conf, tab_cmp, tab_robust, tab_about = st.tabs(
    ["🧭 Registration", "🎯 Debris confidence", "📊 Comparison",
     "🛡️ Robustness", "📄 Method & novelty"])

# ---- Registration ---------------------------------------------------------
with tab_reg:
    left, right = st.columns([3, 1])
    with left:
        st.pyplot(visualize.plot_registration(scene, proposed, traditional),
                  width='stretch')
    with right:
        st.markdown("#### Stage read-out")
        st.markdown(
            f"""
            <div class="stage"><b>Stage 1 · RANSAC</b><br>
            putative matches: {proposed.n_putative}<br>
            inliers: {proposed.n_ransac_inliers}
            &nbsp;({'✅ success' if proposed.ransac_success else '⚠️ fallback'})</div>
            <div class="stage"><b>Stage 2 · CPD</b><br>
            iterations: {proposed.cpd_iterations}<br>
            D<sub>reg</sub> (RMSE): {proposed.fusion.d_registration:.2f}</div>
            <div class="stage"><b>Stage 3 · Fusion</b><br>
            matches: {proposed.fusion.n_matches}<br>
            Q<sub>reg</sub> gate: {proposed.fusion.q_reg:.3f}</div>
            """, unsafe_allow_html=True)
        st.info(f"Traditional ICP RMSE: **{traditional.icp_rmse:.1f}** — "
                "no outlier model, so clutter drags the transform off.")

# ---- Confidence -----------------------------------------------------------
with tab_conf:
    left, right = st.columns([3, 1])
    with left:
        st.pyplot(visualize.plot_confidence(scene, proposed, conf_thr),
                  width='stretch')
    with right:
        reported = proposed.fusion.confident(conf_thr)
        n_corr = sum(c.matched for c in reported)
        st.markdown("#### At the current threshold")
        st.metric("Reported debris", len(reported))
        st.metric("…corroborated (2-sensor)", n_corr)
        st.metric("…single-modality", len(reported) - n_corr)
        st.caption("Red rings on the map = reported debris (C ≥ T). "
                   "Slide **T** in the sidebar to trade precision for recall.")
        b = best_f1(scored["proposed"], truth, match_radius)
        st.success(f"Best-F1 operating point: **{b['f1']:.3f}** "
                   f"at T={b['threshold']:.2f}  "
                   f"(P={b['precision']:.2f}, R={b['recall']:.2f})")

# ---- Comparison -----------------------------------------------------------
with tab_cmp:
    st.markdown("##### Fixed operating point "
                "(proposed & ICP share the same fusion stage and threshold)")
    rows = {"optical-only": m_opt, "traditional (ICP)": m_trad,
            "proposed (RANSAC+CPD)": m_prop}
    st.dataframe(
        {"method": list(rows.keys()),
         "precision": [f"{m.precision:.3f}" for m in rows.values()],
         "recall": [f"{m.recall:.3f}" for m in rows.values()],
         "F1": [f"{m.f1:.3f}" for m in rows.values()],
         "TP": [m.tp for m in rows.values()],
         "FP": [m.fp for m in rows.values()],
         "FN": [m.fn for m in rows.values()]},
        width='stretch', hide_index=True)

    rec_p, prec_p, _, ap_p = precision_recall_curve(
        proposed.fusion.candidates, truth, match_radius)
    rec_t, prec_t, ap_t = strength_pr_curve(scored["traditional_icp"], truth, match_radius)
    rec_o, prec_o, ap_o = strength_pr_curve(scored["optical_only"], truth, match_radius)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(visualize.plot_pr_curves(
            {"proposed": (rec_p, prec_p, ap_p),
             "traditional_icp": (rec_t, prec_t, ap_t),
             "optical_only": (rec_o, prec_o, ap_o)}), width='stretch')
    with c2:
        st.markdown("##### Average precision (area under PR)")
        st.metric("Proposed (RANSAC+CPD)", f"{ap_p:.3f}")
        st.metric("Traditional (ICP)", f"{ap_t:.3f}", f"{ap_p - ap_t:+.3f}")
        st.metric("Optical-only", f"{ap_o:.3f}", f"{ap_p - ap_o:+.3f}")
        st.caption("Proposed dominates the high-precision region — the "
                   "operationally important one for debris (few false alarms).")

# ---- Robustness -----------------------------------------------------------
with tab_robust:
    st.markdown("How does each method hold up as modality-specific clutter grows? "
                "(Averaged over several seeds — this recomputes many scenes.)")
    trials = st.slider("Seeds per clutter level", 2, 15, 6, 1)
    if st.button("▶ Run robustness sweep", type="primary"):
        sweep = robustness((5, 15, 30, 50, 75), trials)
        st.pyplot(visualize.plot_robustness(sweep), width='stretch')
        st.caption("Left: precision vs clutter (proposed holds up; optical-only "
                   "collapses). Right: F1 vs clutter.")
    else:
        st.info("Click **Run robustness sweep** to compute the clutter curves.")

# ---- About ----------------------------------------------------------------
with tab_about:
    st.markdown(
        """
### Why three stages, each with a distinct job

| Stage | Algorithm | Job |
|------|-----------|-----|
| 1 | **RANSAC** | robust initial similarity transform + reject outlier correspondences |
| 2 | **CPD** (rigid) | probabilistically refine the alignment; uniform-noise term absorbs remaining clutter |
| 3 | **Cross-modal validation** | match registered detections, **gate by registration quality** `Q_reg`, emit confidence `C` |

`Q_reg = exp(−D_reg² / 2τ²) · min(1, N_matches / N_support)` multiplies every
score, so a **failed registration cannot manufacture confident debris** — the
alignment doubles as an evidence test. Detections seen in only one modality are
discounted (likely single-sensor false positives: a ship, or sun-glint).

### The comparison
`traditional (ICP)` and `proposed` share the **identical** fusion stage, so any
gap isolates the registration algorithm (**RANSAC+CPD vs ICP**). `optical-only`
shows why cross-modal validation is worth doing at all.

### Honest scope
Runs on a **simulator** that captures the phenomena that matter for the
*algorithm* (independent noisy detections, modality-specific false positives,
missed detections, a residual co-registration error) — not radiometrically
realistic imagery. The **relative** ordering of methods is the result. See
`docs/METHODOLOGY.md` and `docs/NOVELTY.md` (novelty claim + prior-art plan);
real Sentinel data plugs in via `src/detection.py`.
        """)

st.divider()
st.caption("Prototype · runs the same `src/` pipeline as `run_demo.py` · "
           "adjust the sidebar to explore. Not for operational use.")
