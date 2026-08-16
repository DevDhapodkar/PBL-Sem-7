# Methodology

This document specifies the algorithms in the prototype and the exact form of
the cross-modal debris-confidence score. It expands the three-stage architecture
sketched in the README.

---

## 0. Inputs — object-level point sets

Both sensors are reduced to **object-level detections** *before* registration:

* **SAR point set** `P_SAR = {(x_i, s_i^SAR)}` — centroids of Sentinel-1
  backscatter anomalies with an anomaly strength `s^SAR ∈ [0,1]`.
* **Optical point set** `P_opt = {(y_j, s_j^opt)}` — centroids of Sentinel-2
  Floating-Debris-Index detections with an index strength `s^opt ∈ [0,1]`.

Working at the object level (not pixel level) is deliberate: debris manifests as
*discrete slicks/rafts*, the two sensors have different resolutions and revisit
times, and point-set registration is exactly the tool for aligning two clouds of
discrete objects related by an unknown transform and contaminated by outliers.

Because Sentinel products are geocoded, the two point sets differ by a **residual
similarity transform** (small scale, small rotation, a tens-of-metres shift), not
an arbitrary one. The simulator (`src/data_simulation.py`) injects this residual
plus realistic corruptions: per-object localisation noise, missed detections
(`p_detect < 1`), and **modality-specific false positives** — SAR ships/wind
streaks and optical sun-glint/cloud.

---

## 1. Stage 1 — RANSAC robust initialisation

**Goal:** a robust initial `SAR → optical` similarity transform `T₀`, plus
rejection of grossly wrong correspondences.

1. **Putative correspondences.** For geocoded inputs, form them by **spatial
   mutual-nearest-neighbour** within a radius (`spatial_putative_matches`): a
   genuine cross-modal pair is spatially close after geocoding, so most mutual-NN
   pairs are correct and the rest are clutter-to-clutter accidents.
   *(For large, unknown transforms — raw scenes — the code also provides
   rotation/scale-invariant **shape-context** descriptors and descriptor-space
   matching, `shape_context_descriptors` + `putative_matches`.)*
2. **RANSAC** (`ransac_similarity`): repeatedly sample 2 correspondences, fit a
   similarity with **Umeyama's closed form** (`estimate_similarity`), and count
   inliers (residual `< threshold`). Keep the largest consensus set, then refit
   `T₀` on all its inliers.

RANSAC's explicit inlier/outlier partition is what a least-squares or ICP fit
lacks: it discards the false-positive correspondences *before* they can bias the
transform.

---

## 2. Stage 2 — Coherent Point Drift refinement

**Goal:** refine `T₀` into the final transform `T` by aligning the two *full*
distributions, not just the RANSAC inliers.

CPD (Myronenko & Song, *TPAMI* 2010; `cpd_rigid`) models the source point set as
the centroids of a Gaussian Mixture and moves them coherently to maximise the
likelihood of the target, with a **uniform component of weight `w`** that absorbs
outliers. We use the **rigid (rotation + translation) variant**, warm-started by
`T₀`:

* The EM **E-step** computes soft correspondence probabilities `P` (an `M×N`
  matrix) — no hard assignment, so noisy/near-miss detections contribute
  partially instead of snapping to a wrong neighbour.
* The **M-step** has a closed form for `R, t` via the SVD of the
  weighted cross-covariance.
* The uniform term `c = (2πσ²)^{D/2} · w/(1−w) · M/N` down-weights points with no
  good match — the mechanism that makes CPD robust to the clutter RANSAC left.

**Why rigid, not scale+rigid, here.** RANSAC has *already* recovered the scale.
Re-estimating scale inside CPD invites the well-known **scale-shrink degeneracy**
(the GMM collapses to a point, `σ² → 0`). Fixing scale removes that failure mode;
`test_cpd_rigid_does_not_collapse` guards it. The final transform is the
composition `T = T_CPD ∘ T₀`.

**Division of labour.** RANSAC gives a globally correct but *coarse* transform
from a hard inlier set; CPD gives a *fine* transform using *soft, global*
evidence. Neither alone is sufficient here: CPD from the identity falls into a
wrong local optimum (empirically it collapsed), and RANSAC alone leaves a
few-metre residual that Stage 3 would rather not pay for.

---

## 3. Stage 3 — Cross-modal validation & confidence

**Goal:** decide which detections are corroborated across both sensors and score
each. This is the step that turns registration into *debris validation*.

Register SAR into the optical frame, `x̃_i = T(x_i)`, then match to optical
detections by **mutual-NN within `match_radius`**. For every candidate compute

```
C_i = Q_reg · [ α·g_i + β·s_i^SAR + γ·s_i^opt ]           (matched pair)
C_i = Q_reg · penalty · s_i^(single)                       (single-modality)
```

with weights `α+β+γ = 1` (defaults `0.45 / 0.275 / 0.275`) and

| Symbol | Meaning | Form |
|--------|---------|------|
| `g_i` | geometric agreement of the specific pair | `exp(−d_i² / 2s²)`, `d_i` = post-registration residual |
| `s^SAR, s^opt` | per-object evidence strengths | detector outputs |
| `D_reg` | global registration error | RMSE of matched-pair residuals |
| `N_matches` | number of cross-modal matches | count |
| **`Q_reg`** | **global registration-quality gate** | `exp(−D_reg²/2τ²) · min(1, N_matches / N_support)` |

`Q_reg` is the crux of the *validation* semantics. It multiplies **every**
confidence, so:

* if the two modalities could **not** be aligned (`D_reg` large) → `Q_reg → 0` →
  no confident debris can be manufactured from a bad registration;
* if too few detections corroborate (`N_matches < N_support`) → the support term
  shrinks `Q_reg`.

The support term **saturates** (`min(1, ·)`): it asks *"are there enough coherent
cross-modal matches to trust the alignment?"*, **not** *"what fraction of
detections are debris"* — so a mostly-clutter scene is not unfairly penalised.
`test_fusion_gate_penalises_failed_registration` checks the `Q_reg → 0` behaviour.

Detections seen in **one** modality only are heavily discounted (`penalty = 0.35`)
— they are the likely single-sensor false positives (a ship with no optical
counterpart, sun-glint with no SAR counterpart). A candidate is reported as
debris iff `C_i > T`.

---

## 4. The traditional baseline (for comparison)

`traditional_icp` (`src/baseline.py`) replaces **only** Stage 1+2 with classic
**Iterative Closest Point** (`icp_rigid`): each iteration hard-assigns every
source point to its nearest target and refits a similarity. ICP has **no outlier
model** and is initialised at the identity, so the modality-specific false
positives act as phantom correspondences and pull the transform off; genuine
cross-modal pairs then fail to fall within `match_radius`. Stage 3 is identical
to the proposed method, which isolates the registration algorithm as the only
variable. A second reference, `optical_only`, thresholds the optical strength
alone (no fusion) to show the value of cross-modal validation itself.

---

## 5. Evaluation

`src/evaluate.py` matches each method's reported debris locations to
ground-truth debris (greedy, one-to-one, within `match_radius`) → TP/FP/FN →
precision, recall, F1. For the methods with a tunable score we sweep the
threshold to produce **precision–recall curves** and **average precision** (area
under the PR curve), and report the **best-F1 operating point**. `run_demo.py`
also runs a **clutter-robustness sweep** (metrics vs. increasing false-positive
load) averaged over seeds.

### Parameters (defaults)

| Where | Parameter | Default | Meaning |
|-------|-----------|---------|---------|
| sim | `scale, rotation_deg, translation` | `1.03, 4°, (30,−18)` | residual S1↔S2 mis-registration |
| sim | `p_detect_sar / optical` | `0.78 / 0.82` | per-sensor detection probability |
| sim | `n_false_sar / optical` | `25 / 30` | clutter detections |
| RANSAC | `threshold` | `15` | inlier residual |
| CPD | `w` | `0.4` | uniform-noise (outlier) weight |
| fusion | `match_radius` | `20` | cross-modal spatial tolerance |
| fusion | `α, β, γ` | `0.45, 0.275, 0.275` | confidence weights |
| fusion | `τ, N_support, penalty` | `25, 6, 0.35` | `Q_reg` gate + single-modality discount |
| decision | `T` | `0.30` | confidence threshold |

### References
* A. Myronenko, X. Song. *Point Set Registration: Coherent Point Drift.* IEEE TPAMI, 2010.
* S. Umeyama. *Least-squares estimation of transformation parameters between two point patterns.* IEEE TPAMI, 1991.
* M. Fischler, R. Bolles. *Random Sample Consensus (RANSAC).* Comm. ACM, 1981.
* L. Biermann et al. *Finding Plastic Patches in Coastal Waters using Optical Satellite Data (FDI).* Scientific Reports, 2020.
