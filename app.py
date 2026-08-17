#!/usr/bin/env python3
"""
Interactive dashboard — cross-modal Sentinel-1/Sentinel-2 debris detection
focused on the water bodies of Nagpur.
==========================================================================

Two modes (top of the sidebar):

* 🛰️ **Nagpur — real Sentinel-2**: loads a real Sentinel-2 L2A scene over a
  chosen Nagpur lake (bundled from the public Copernicus/AWS bucket, tile 44QKJ),
  detects floating matter (NDWI water mask + NDVI/FDI), pairs it with a Sentinel-1
  view, and runs RANSAC+CPD cross-modal validation live over the lake image.
* 🧪 **Simulation**: fully controllable synthetic scenes to stress-test the
  algorithm (clutter, mis-registration, detection rates).

    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.baseline import run_optical_only, run_traditional
from src.data_simulation import simulate_scene
from src.evaluate import best_f1, precision_recall_curve, score_locations, strength_pr_curve
from src.pipeline import run_proposed
from src import visualize

st.set_page_config(page_title="Nagpur Debris — S1×S2", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """<style>
      .block-container {padding-top: 1.3rem;}
      .hero {background: linear-gradient(110deg,#0b3d5c,#12708f 55%,#1a9db0);
        padding:1.05rem 1.35rem;border-radius:14px;color:#fff;margin-bottom:.9rem;}
      .hero h1{margin:0;font-size:1.5rem;font-weight:700;}
      .hero p{margin:.3rem 0 0;opacity:.92;font-size:.92rem;}
      .pill{display:inline-block;background:rgba(255,255,255,.16);border-radius:999px;
        padding:.1rem .6rem;font-size:.7rem;margin-right:.3rem;}
      .stage{border-left:4px solid #1a9db0;padding:.1rem 0 .1rem .7rem;margin:.3rem 0;}
      div[data-testid="stMetric"]{background:#f5f8fa;border:1px solid #e3eaf0;
        border-radius:12px;padding:.55rem .8rem;}
    </style>""", unsafe_allow_html=True)

st.markdown(
    """<div class="hero">
      <h1>🛰️ Cross-Modal Debris Detection · Water Bodies of Nagpur</h1>
      <p>Sentinel-1 SAR × Sentinel-2 optical → object point sets →
         <b>RANSAC</b> init → <b>CPD</b> refine →
         registration-gated confidence
         <b>C = f(D<sub>reg</sub>, N<sub>match</sub>, S<sub>SAR</sub>, S<sub>opt</sub>)</b>.</p>
      <span class="pill">RANSAC + CPD</span><span class="pill">vs traditional ICP</span>
      <span class="pill">real Sentinel-2 over Nagpur</span></div>""",
    unsafe_allow_html=True)

MATCH_SIM = 20.0

# ---------------------------------------------------------------------------
mode = st.sidebar.radio("🗺️ Data source",
                        ["🛰️ Nagpur — real Sentinel-2", "🧪 Simulation"])
st.sidebar.divider()


# ===========================================================================
# Cached compute
# ===========================================================================
@st.cache_data(show_spinner="Loading Sentinel data & running pipeline…")
def compute_nagpur(lake_key, seed, match_radius, live_latest=False):
    from src.nagpur import get_lake
    from src.acquire import get_nagpur_s2, fetch_latest_clear_s2
    from src.detect_real import build_scene_from_real
    lake = get_lake(lake_key)
    if live_latest:
        try:
            bs = fetch_latest_clear_s2(lake)
        except Exception:
            bs = get_nagpur_s2(lake)
    else:
        bs = get_nagpur_s2(lake)
    scene, diag = build_scene_from_real(bs, seed=int(seed))
    proposed = run_proposed(scene, putative_radius=14, ransac_threshold=4.0,
                            match_radius=match_radius)
    traditional = run_traditional(scene, match_radius=match_radius)
    return bs, scene, diag, proposed, traditional


@st.cache_data(show_spinner=False)
def compute_sim(seed, n_debris, p_sar, p_opt, n_fs, n_fo, scale, rot, tx, ty,
                rthr, w, mr):
    scene = simulate_scene(n_debris=n_debris, p_detect_sar=p_sar, p_detect_optical=p_opt,
                           n_false_sar=n_fs, n_false_optical=n_fo, scale=scale,
                           rotation_deg=rot, translation=(tx, ty), seed=int(seed))
    proposed = run_proposed(scene, ransac_threshold=rthr, cpd_w=w, match_radius=mr)
    traditional = run_traditional(scene, match_radius=mr)
    return scene, proposed, traditional


def render_metrics(scene, proposed, traditional, match_radius, conf_thr,
                   optical_thr, report_all_optical=False):
    truth = scene.true_debris
    scored = {
        "optical_only": [(d.xy, d.strength) for d in scene.optical],
        "traditional_icp": [(c.xy, c.confidence) for c in traditional.fusion.candidates],
        "proposed": [(c.xy, c.confidence) for c in proposed.fusion.candidates],
    }
    thr = {"optical_only": 0.0 if report_all_optical else optical_thr,
           "traditional_icp": conf_thr, "proposed": conf_thr}
    rows = {}
    for name, pts in scored.items():
        sel = np.array([xy for xy, s in pts if s >= thr[name]]) if pts else np.empty((0, 2))
        rows[name] = score_locations(sel if len(sel) else np.empty((0, 2)), truth, match_radius)
    return rows, scored


# ===========================================================================
# NAGPUR MODE
# ===========================================================================
if mode.startswith("🛰️"):
    from src.nagpur import LAKES
    from src import visualize_nagpur as vn

    with st.sidebar:
        st.header("🌊 Lake")
        lake_key = st.selectbox("Nagpur water body", list(LAKES),
                                format_func=lambda k: LAKES[k].name, index=0)
        st.caption(LAKES[lake_key].note)
        live_latest = st.checkbox("Fetch most-recent clear scene (live)", value=False,
                                  help="Live-fetch the newest low-cloud Sentinel-2 "
                                       "scene from the public bucket (needs network). "
                                       "Off = bundled real scene (offline).")
        st.header("🛰️ Sentinel-1 pairing")
        st.caption("Live S1 needs Earth-Engine credentials; here the SAR view is "
                   "derived from the real optical scene (see *Method & data*).")
        seed = st.number_input("SAR realisation seed", 0, 999, 1)
        st.header("🎯 Decision")
        match_radius = st.slider("Match radius (px, 10 m)", 3.0, 12.0, 5.0, 0.5)
        conf_thr = st.slider("Confidence threshold T", 0.0, 1.0, 0.30, 0.02)
        st.divider()
        st.caption("Real Sentinel-2 L2A · tile 44QKJ · Jan-2024 · AWS `sentinel-cogs`.")

    bs, scene, diag, proposed, traditional = compute_nagpur(lake_key, seed, match_radius,
                                                            live_latest)
    rows, scored = render_metrics(scene, proposed, traditional, match_radius,
                                  conf_thr, 0.0, report_all_optical=True)
    m_prop, m_trad, m_opt = rows["proposed"], rows["traditional_icp"], rows["optical_only"]

    st.subheader(f"{bs.meta.get('lake','Lake')} — live scorecard")
    k = st.columns(4)
    k[0].metric("Proposed F1", f"{m_prop.f1:.2f}", f"{m_prop.f1-m_trad.f1:+.2f} vs ICP")
    k[1].metric("Proposed precision", f"{m_prop.precision:.2f}",
                f"{m_prop.precision-m_opt.precision:+.2f} vs optical-only")
    k[2].metric("Corroborated debris", f"{proposed.fusion.n_matches}",
                f"D_reg {proposed.fusion.d_registration:.1f}px")
    k[3].metric("Scene", bs.meta.get("date", "—"), "real Sentinel-2")
    st.caption(f"Real Sentinel-2 floating-matter detections: **{len(scene.optical)}** · "
               f"paired Sentinel-1 detections: **{len(scene.sar)}** · "
               f"corroborable (both sensors) ground truth: **{len(scene.true_debris)}**.")

    t_scene, t_overlay, t_reg, t_conf, t_cmp, t_about = st.tabs(
        ["🌊 Lake & detections", "🛰️ S1×S2 overlay", "🧭 Registration",
         "🎯 Debris confidence", "📊 Comparison", "📄 Method & data"])

    with t_scene:
        st.pyplot(vn.plot_scene_overview(bs, scene, diag), width='stretch')
        st.caption("Left: Sentinel-2 true colour with S1 (▲) and S2 (○) detections. "
                   "Middle: NDWI with the water mask outlined. Right: NDVI/FDI "
                   "floating-matter cue on water (hyacinth / scum / trash).")

    with t_overlay:
        st.pyplot(vn.plot_overlay(bs, scene, proposed,
                                  s1_date=diag.get("meta", {}).get("date", "SAR pass"),
                                  s2_date=bs.meta.get("date", "?")), width='stretch')
        drift_px = float(np.hypot(*proposed.T_final[:2, 2]))
        c = st.columns(3)
        c[0].metric("Estimated drift / offset", f"~{drift_px*10:.0f} m",
                    "S1↔S2 debris cloud")
        c[1].metric("Registration error D_reg", f"{proposed.fusion.d_registration:.1f} px")
        c[2].metric("Corroborated by both", proposed.fusion.n_matches)
        st.caption("Sentinel-1 and Sentinel-2 image at different times, so floating "
                   "debris **drifts** between passes. **Left** overlays the raw "
                   "detections (offset by drift + co-registration); **right** shows "
                   "them after RANSAC+CPD registration — yellow rings mark debris "
                   "seen by **both** sensors. Run `fetch_and_overlay.py` for the "
                   "latest-S1 × latest-clear-S2 version from the command line.")

    with t_reg:
        c1, c2 = st.columns([3, 1])
        c1.pyplot(visualize.plot_registration(scene, proposed, traditional), width='stretch')
        with c2:
            st.markdown("#### Stage read-out")
            st.markdown(
                f"""<div class="stage"><b>1 · RANSAC</b><br>putative {proposed.n_putative}
                · inliers {proposed.n_ransac_inliers}
                {'✅' if proposed.ransac_success else '⚠️'}</div>
                <div class="stage"><b>2 · CPD</b><br>iters {proposed.cpd_iterations}
                · D<sub>reg</sub> {proposed.fusion.d_registration:.2f}px</div>
                <div class="stage"><b>3 · Fusion</b><br>matches {proposed.fusion.n_matches}
                · Q<sub>reg</sub> {proposed.fusion.q_reg:.2f}</div>""",
                unsafe_allow_html=True)
            st.info(f"ICP RMSE: {traditional.icp_rmse:.1f}px")

    with t_conf:
        c1, c2 = st.columns([3, 1])
        c1.pyplot(vn.plot_confidence_over_image(bs, proposed, conf_thr), width='stretch')
        with c2:
            rep = proposed.fusion.confident(conf_thr)
            n_corr = sum(c.matched for c in rep)
            st.metric("Reported debris (C≥T)", len(rep))
            st.metric("…corroborated S1∩S2", n_corr)
            st.metric("…single-sensor", len(rep) - n_corr)
            st.caption("Red rings = reported debris. Single-sensor floating-matter "
                       "(no SAR corroboration) is down-weighted.")

    with t_cmp:
        st.markdown("##### Recovery of the corroborable floating matter "
                    "(objects both sensors saw)")
        disp = {"optical-only (trust all S2)": m_opt,
                "traditional (ICP)": m_trad, "proposed (RANSAC+CPD)": m_prop}
        st.dataframe({"method": list(disp), "precision": [f"{m.precision:.2f}" for m in disp.values()],
                      "recall": [f"{m.recall:.2f}" for m in disp.values()],
                      "F1": [f"{m.f1:.2f}" for m in disp.values()],
                      "TP": [m.tp for m in disp.values()], "FP": [m.fp for m in disp.values()],
                      "FN": [m.fn for m in disp.values()]},
                     width='stretch', hide_index=True)
        rec_p, prec_p, _, ap_p = precision_recall_curve(proposed.fusion.candidates,
                                                        scene.true_debris, match_radius)
        rec_t, prec_t, ap_t = strength_pr_curve(scored["traditional_icp"], scene.true_debris, match_radius)
        rec_o, prec_o, ap_o = strength_pr_curve(scored["optical_only"], scene.true_debris, match_radius)
        c1, c2 = st.columns([3, 2])
        c1.pyplot(visualize.plot_pr_curves({"proposed": (rec_p, prec_p, ap_p),
                  "traditional_icp": (rec_t, prec_t, ap_t),
                  "optical_only": (rec_o, prec_o, ap_o)}), width='stretch')
        with c2:
            st.metric("Proposed AP", f"{ap_p:.3f}")
            st.metric("Traditional AP", f"{ap_t:.3f}", f"{ap_p-ap_t:+.3f}")
            st.metric("Optical-only AP", f"{ap_o:.3f}", f"{ap_p-ap_o:+.3f}")
            st.caption("optical-only cannot separate corroborated debris from "
                       "single-sensor artifacts → precision ceiling. Cross-modal "
                       "validation lifts it; RANSAC+CPD recovers more true matches "
                       "than ICP.")

    with t_about:
        st.markdown(
            f"""
#### Data
* **Sentinel-2 L2A (optical) — real & live.** Surface-reflectance COGs read
  directly from the public AWS bucket `sentinel-cogs`, MGRS tile **44QKJ**
  (covers all Nagpur lakes), January-2024, least-cloudy scene per lake. Current
  scene: **{bs.meta.get('scene','—')}** ({bs.meta.get('date','—')}).
* **Sentinel-1 GRD (SAR).** `src/acquire.py::fetch_sentinel1_ee` fetches VV/VH via
  Google Earth Engine on your machine. In this hosted demo the S1 endpoints are
  blocked by network policy, so the SAR point set is an **independent, noisy,
  mis-registered sample derived from the real optical detections** (plus SAR
  clutter). Swap in the real S1 detector for fully-live operation.

#### Detectors (real Sentinel-2)
* Water mask: **NDWI** = (Green−NIR)/(Green+NIR).
* Floating matter on water: **NDVI** (hyacinth / algal scum) ∪ **FDI** (plastic /
  trash rafts, Biermann 2020) → connected-component centroids = optical point set.

#### Pipeline
RANSAC robust init → rigid CPD refinement → cross-modal validation with the
registration-gated confidence `C`. `traditional (ICP)` shares the identical
fusion stage, so the gap is purely the registration algorithm. See
`docs/METHODOLOGY.md` and `docs/NOVELTY.md`.

#### Honest scope
The optical detections and imagery are **real Sentinel-2 over Nagpur**. Ground
truth here is the set of floating-matter objects visible to **both** sensors; the
SAR view is simulated where live S1 is unreachable. Absolute numbers are
illustrative — validate against field/annotated data (e.g. MARIDA-style labels)
for operational claims.
            """)

# ===========================================================================
# SIMULATION MODE
# ===========================================================================
else:
    with st.sidebar:
        st.header("⚙️ Scene")
        seed = st.number_input("Seed", 0, 9999, 42)
        n_debris = st.slider("Ground-truth debris", 10, 100, 40, 5)
        c1, c2 = st.columns(2)
        p_sar = c1.slider("P(det) SAR", 0.3, 1.0, 0.78, 0.02)
        p_opt = c2.slider("P(det) optical", 0.3, 1.0, 0.82, 0.02)
        c3, c4 = st.columns(2)
        n_fs = c3.slider("SAR clutter", 0, 100, 25, 5)
        n_fo = c4.slider("Optical clutter", 0, 100, 30, 5)
        st.header("🧭 Residual mis-registration")
        scale = st.slider("Scale", 0.90, 1.15, 1.03, 0.01)
        rot = st.slider("Rotation (°)", 0.0, 20.0, 4.0, 0.5)
        tx = st.slider("Shift X", -100.0, 100.0, 30.0, 5.0)
        ty = st.slider("Shift Y", -100.0, 100.0, -18.0, 5.0)
        st.header("🔧 Pipeline")
        rthr = st.slider("RANSAC threshold", 5.0, 40.0, 15.0, 1.0)
        w = st.slider("CPD outlier weight w", 0.0, 0.9, 0.4, 0.05)
        mr = st.slider("Match radius", 5.0, 50.0, MATCH_SIM, 1.0)
        st.header("🎯 Decision")
        conf_thr = st.slider("Confidence threshold T", 0.0, 1.0, 0.30, 0.02)

    scene, proposed, traditional = compute_sim(seed, n_debris, p_sar, p_opt, n_fs, n_fo,
                                               scale, rot, tx, ty, rthr, w, mr)
    rows, scored = render_metrics(scene, proposed, traditional, mr, conf_thr, 0.5)
    m_prop, m_trad, m_opt = rows["proposed"], rows["traditional_icp"], rows["optical_only"]

    st.subheader("Live scorecard")
    k = st.columns(4)
    k[0].metric("Proposed F1", f"{m_prop.f1:.3f}", f"{m_prop.f1-m_trad.f1:+.3f} vs ICP")
    k[1].metric("Proposed precision", f"{m_prop.precision:.3f}",
                f"{m_prop.precision-m_trad.precision:+.3f} vs ICP")
    k[2].metric("Proposed recall", f"{m_prop.recall:.3f}",
                f"{m_prop.recall-m_trad.recall:+.3f} vs ICP")
    k[3].metric("Cross-modal matches", proposed.fusion.n_matches,
                f"Q_reg {proposed.fusion.q_reg:.2f}")

    t_reg, t_conf, t_cmp, t_about = st.tabs(
        ["🧭 Registration", "🎯 Debris confidence", "📊 Comparison", "📄 Method"])
    with t_reg:
        c1, c2 = st.columns([3, 1])
        c1.pyplot(visualize.plot_registration(scene, proposed, traditional), width='stretch')
        c2.markdown(f"""<div class="stage"><b>1 · RANSAC</b><br>putative {proposed.n_putative}
            · inliers {proposed.n_ransac_inliers}</div>
            <div class="stage"><b>2 · CPD</b><br>iters {proposed.cpd_iterations}
            · D<sub>reg</sub> {proposed.fusion.d_registration:.2f}</div>
            <div class="stage"><b>3 · Fusion</b><br>matches {proposed.fusion.n_matches}
            · Q<sub>reg</sub> {proposed.fusion.q_reg:.2f}</div>""", unsafe_allow_html=True)
        c2.info(f"ICP RMSE: {traditional.icp_rmse:.1f}")
    with t_conf:
        c1, c2 = st.columns([3, 1])
        c1.pyplot(visualize.plot_confidence(scene, proposed, conf_thr), width='stretch')
        b = best_f1(scored["proposed"], scene.true_debris, mr)
        c2.success(f"Best-F1 {b['f1']:.3f} at T={b['threshold']:.2f}")
    with t_cmp:
        disp = {"optical-only": m_opt, "traditional (ICP)": m_trad, "proposed": m_prop}
        st.dataframe({"method": list(disp), "precision": [f"{m.precision:.3f}" for m in disp.values()],
                      "recall": [f"{m.recall:.3f}" for m in disp.values()],
                      "F1": [f"{m.f1:.3f}" for m in disp.values()]},
                     width='stretch', hide_index=True)
        rec_p, prec_p, _, ap_p = precision_recall_curve(proposed.fusion.candidates, scene.true_debris, mr)
        rec_t, prec_t, ap_t = strength_pr_curve(scored["traditional_icp"], scene.true_debris, mr)
        rec_o, prec_o, ap_o = strength_pr_curve(scored["optical_only"], scene.true_debris, mr)
        st.pyplot(visualize.plot_pr_curves({"proposed": (rec_p, prec_p, ap_p),
                  "traditional_icp": (rec_t, prec_t, ap_t),
                  "optical_only": (rec_o, prec_o, ap_o)}), width='stretch')
    with t_about:
        st.markdown("Synthetic stress-test of the same pipeline. Switch to "
                    "**Nagpur — real Sentinel-2** for live imagery. See "
                    "`docs/METHODOLOGY.md` and `docs/NOVELTY.md`.")

st.divider()
st.caption("Prototype · runs the same `src/` pipeline as `run_demo.py` · "
           "real Sentinel-2 over Nagpur (tile 44QKJ). Not for operational use.")
