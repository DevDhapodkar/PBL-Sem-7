# Cross-Modal Sentinel-1 / Sentinel-2 Marine-Debris Detection

### RANSAC-initialised Coherent Point Drift for object-level SAR ↔ optical validation

A research **prototype** for a PBL project. It implements — and rigorously
benchmarks against a traditional baseline — the pipeline:

```
 Sentinel-1 SAR  ──detector──▶  SAR debris point set ──┐
                                                        │  Stage 1  RANSAC  (robust init + outlier rejection)
 Sentinel-2 optical ─detector─▶ optical debris point set┤  Stage 2  CPD     (probabilistic alignment refinement)
                                                        │  Stage 3  cross-modal validation ▶ confidence  C
                                                        └────────────────────────────────────────────────────
                                                              C > T  ⇒  reported as debris
```

The design principle the prototype follows (and the reason it is more than
"RANSAC → CPD → done") is that **each stage has a distinct, non-redundant job**:

| Stage | Algorithm | Job |
|------|-----------|-----|
| 1 | **RANSAC** | From putative correspondences, estimate a *robust* initial similarity transform and **reject gross outlier** correspondences (ships, sun-glint that happen to line up). |
| 2 | **CPD** (rigid) | Warm-started by the RANSAC transform, **probabilistically refine** the alignment of the two imperfect point distributions; its uniform-noise component absorbs the *remaining* clutter. |
| 3 | **Cross-modal validation** | Match the registered detections, **gate by registration quality**, and emit a per-object debris confidence `C = f(D_reg, N_matches, S_SAR, S_optical)`. |

> **Novelty claim** (see [`docs/NOVELTY.md`](docs/NOVELTY.md)): *independent* debris
> detection in Sentinel-1 and Sentinel-2 → conversion to **object-level point
> sets** → **RANSAC-initialised CPD** cross-modal registration → spatial
> agreement used as a **debris-validation mechanism** with a registration-gated
> confidence score. RANSAC + CPD on their own are **not** novel; this
> application-specific composition and the validation semantics are the
> contribution to investigate — pending the targeted prior-art search outlined
> in the novelty doc.

---

## Quick start

```bash
pip install -r requirements.txt        # numpy, scipy, matplotlib
python run_demo.py                      # runs everything, writes results/
python tests/test_pipeline.py           # 8 unit/integration tests
```

`run_demo.py` runs three methods on the same simulated scenes, prints a metrics
table, and writes four figures + `results/metrics.json`.

---

## What it compares (and why the comparison is fair)

| Method | Registration | Fusion / decision | Role |
|--------|--------------|-------------------|------|
| `optical_only` | — | threshold optical strength | single-modality baseline (no fusion) |
| `traditional_icp` | **ICP** | cross-modal confidence `C` | *the traditional method* |
| `proposed` | **RANSAC + CPD** | cross-modal confidence `C` | this work |

`traditional_icp` and `proposed` share the **identical** Stage-3 confidence
model and decision threshold; they differ **only** in the registration stage.
So any performance gap between them is attributable to **RANSAC+CPD vs ICP** —
which is exactly the comparison the project asks for. `optical_only` shows why
cross-modal validation is worth doing at all.

---

## Headline results (20-seed average, fixed operating point)

| Method | Precision | Recall | F1 | Average Precision |
|--------|:---:|:---:|:---:|:---:|
| optical-only | 0.75 | **0.81** | 0.78 | 0.75 |
| traditional (ICP) | 0.79 | 0.30 | 0.42 | 0.66 |
| **proposed (RANSAC+CPD)** | **0.93** | 0.54 | **0.67** | **0.81** |

* **vs the traditional method** — the proposed pipeline **dominates ICP on every
  metric** (F1 0.67 vs 0.42, AP 0.81 vs 0.66). Because the fusion stage is
  identical, this gap is entirely the registration: ICP has no outlier model, so
  modality-specific false positives bias its transform and genuine cross-modal
  pairs fail to overlap.
* **vs optical-only** — cross-modal validation trades recall for **precision by
  design** (it cannot confirm debris that only one sensor saw). It roughly halves
  the false-alarm rate (precision 0.93 vs 0.75) and wins on the overall
  precision–recall trade-off (**highest AP**).
* **Robustness** — as clutter grows (5 → 75 false detections), the proposed
  method holds precision at **0.72** where optical-only collapses to **0.50**
  (see `results/04_robustness.png`).

Numbers regenerate with `python run_demo.py`; exact values vary slightly with
`--trials`. All are written to `results/metrics.json`.

### Figures (`results/`)
1. `01_registration.png` — point sets before registration, after ICP, after RANSAC+CPD.
2. `02_confidence.png` — per-object debris confidence map; red rings = reported debris.
3. `03_precision_recall.png` — PR curves for all three methods (proposed dominates the high-precision region).
4. `04_robustness.png` — precision & F1 vs clutter load.

---

## Using real Sentinel data

The registration + fusion pipeline consumes **object-level point sets**, so it is
agnostic to where the points come from. To run on real scenes, implement the two
detectors stubbed in [`src/detection.py`](src/detection.py):

* **Sentinel-1 (SAR)** — calibrate σ⁰, speckle-filter, run CFAR / local-contrast
  anomaly detection, extract blob centroids → SAR point set (anomaly magnitude → strength).
* **Sentinel-2 (optical)** — compute the Floating Debris Index (FDI; Biermann et al.,
  2020) with cloud/glint masking, threshold, extract centroids → optical point set
  (index value → strength).

Then wrap the outputs with `detection.detections_from_arrays(...)` into a `Scene`
and call `pipeline.run_proposed(scene)`. Nothing else changes.

---

## Repository layout

```
src/
  data_simulation.py   synthetic S1/S2 detections + ground truth (realistic clutter & residual mis-registration)
  detection.py         detector interface + where REAL Sentinel data plugs in (stubs)
  registration.py      RANSAC, CPD (rigid), ICP, similarity/Umeyama, shape-context & spatial putative matching
  fusion.py            cross-modal validation + registration-gated confidence C = f(D_reg, N_matches, S_SAR, S_opt)
  pipeline.py          proposed pipeline: RANSAC → CPD → fusion
  baseline.py          traditional (ICP) and optical-only baselines
  evaluate.py          precision / recall / F1, PR curves, average precision, best-F1
  visualize.py         all figures
run_demo.py            end-to-end demo + benchmark
tests/test_pipeline.py 8 unit/integration tests
docs/
  METHODOLOGY.md       algorithms + the confidence function, in detail
  NOVELTY.md           novelty claim, prior-art positioning, and the search to run
```

## Caveats

This is a **prototype on simulated data**. The simulator captures the phenomena
that matter for the *algorithm* (independent noisy detections, modality-specific
false positives, missed detections, a residual co-registration error), but it is
**not** radiometrically realistic and does not model debris spectra, sea state,
or SAR speckle statistics. Absolute numbers are illustrative; the **relative**
ordering of the methods is the result. Validation on real Sentinel-1/2 scenes
with annotated debris (e.g. the Marine Debris Archive / MARIDA) is the necessary
next step, alongside the prior-art search in `docs/NOVELTY.md`.
