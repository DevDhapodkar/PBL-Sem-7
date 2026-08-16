# Novelty positioning & prior-art plan

> **Honest framing first.** Using RANSAC and CPD does **not**, by itself, make
> this research novel. Both are decades-old, widely-used algorithms, and
> multimodal SAR↔optical registration is an established field. This document
> states precisely which part of the pipeline is put forward as a contribution,
> what would defeat that claim, and the **targeted prior-art search** that must
> be completed before any paper/patent language calls it novel. **Until that
> search is done, treat novelty as _unverified_.**

---

## 1. What is explicitly *not* claimed as novel

* **RANSAC** for robust transform estimation / outlier rejection — 1981.
* **CPD** for point-set registration — 2010, with mature implementations
  (`pycpd`, `probreg`).
* **SAR↔optical image co-registration**, including feature-based and
  area-based methods, and learned descriptors. A large literature exists.
* **Single-sensor** marine-debris detection: Sentinel-2 FDI/NDVI methods
  (Biermann 2020, MARIDA), and Sentinel-1 SAR slick/anomaly detection.
* **Generic decision-level fusion** of two classifiers.

If a reviewer reads the title as "we applied RANSAC+CPD to Sentinel data",
the work is *not* novel. The contribution has to be more specific.

---

## 2. The specific novelty claim (to be tested)

> **An object-level, cross-modal _validation_ pipeline in which independent
> Sentinel-1 and Sentinel-2 debris detections are converted to point sets,
> registered by RANSAC-initialised CPD, and whose _spatial agreement after
> registration_ is used as the debris-confidence mechanism via a
> registration-quality-gated score
> `C = f(D_reg, N_matches, S_SAR, S_optical)`.**

The claimed contribution is **not** any one algorithm but the **composition and
the semantics**:

1. **Detection-to-point-set at object level, per modality, _independently_.**
   Registration operates on *debris candidates*, not raw image features — the
   entities being aligned are the objects of interest themselves.
2. **RANSAC used as the _initialiser_ for CPD** (not stacked blindly): RANSAC
   supplies a robust coarse transform + inlier set; CPD refines with soft,
   global, outlier-tolerant evidence. Each has a distinct, documented job
   (see `docs/METHODOLOGY.md`).
3. **Registration re-purposed as _validation_.** The output of interest is not
   the transform but the **cross-modal spatial agreement**. `Q_reg` gates the
   confidence so a *failed* registration cannot produce confident debris — the
   alignment doubles as an evidence test.
4. **A confidence score that fuses geometry _and_ radiometry**: registration
   residual, match support, and both sensors' evidence strengths in one gated
   expression, giving a tunable precision/recall operating curve.

The prototype in this repo demonstrates that this composition is (a) buildable
and (b) measurably better than the traditional ICP-based alternative under
identical fusion — a **necessary** condition for a contribution, not a
sufficient one. Sufficiency requires the prior-art search below.

---

## 3. What would _defeat_ or _weaken_ the claim

Look specifically for existing work that already does any of these; the more of
them a single paper covers, the weaker the novelty:

* CPD (or GMM/optimal-transport registration) used for **SAR–optical**
  registration of **detections/objects** (not pixels/keypoints).
* **RANSAC → CPD initialisation** proposed as a registration recipe (in any
  domain) — this coarse-to-fine pairing may already be named.
* Marine/floating-**debris** work that fuses **Sentinel-1 + Sentinel-2** and
  uses **spatial co-location/agreement** as the fusion rule.
* Any pipeline using **registration residual as a detection-confidence gate**
  for change detection or multimodal target validation.
* Ship/oil-slick detection papers that co-locate SAR and optical detections
  (debris and oil/ships share this "agreement" idea — check them too).

If prior art covers items 1+3 together, the novelty collapses to "we applied a
known cross-modal-agreement idea to debris" — still possibly publishable as an
application/benchmark study, but not as a new method.

---

## 4. Targeted prior-art search to run

Run these before writing novelty language. Sources: Google Scholar, IEEE Xplore,
Scopus/Web of Science, arXiv, and patents (Google Patents, Espacenet, USPTO).

**Query bundles**

* `("coherent point drift" OR CPD) AND (SAR OR "synthetic aperture radar") AND (optical OR Sentinel)`
* `RANSAC AND ("coherent point drift" OR CPD) AND registration` *(is the pairing named?)*
* `(Sentinel-1 AND Sentinel-2) AND (debris OR "marine litter" OR "floating plastic") AND (fusion OR "data fusion")`
* `("point set registration" OR "point cloud registration") AND (SAR OR multimodal) AND (detection OR object)`
* `(SAR AND optical) AND ("spatial agreement" OR "co-location" OR "cross-modal validation") AND (detection OR confidence)`
* `("registration residual" OR "registration error") AND (confidence OR "change detection") AND multimodal`
* Adjacent domains to check for the same idea under a different name: **oil-spill**
  and **ship** detection with SAR+optical co-location; multimodal **change
  detection**; medical/remote-sensing **registration-as-validation**.

**Key references to read and cite regardless**

* Myronenko & Song 2010 (CPD); Fischler & Bolles 1981 (RANSAC); Umeyama 1991.
* Biermann et al. 2020 (FDI); MARIDA dataset (Kikaki et al. 2022).
* Recent SAR–optical registration surveys and learned-descriptor methods.
* Any Sentinel-1/2 debris-fusion paper (e.g. combining SAR roughness with optical
  spectral cues) — the closest competitors.

**Decision rule.** For each hit, record how many of §3's items it covers and at
what level (pixel vs object; image vs debris; registration vs validation). Then:

* **0–1 items, none on debris** → novelty of the *composition* is plausible;
  proceed, positioning the paper as a method + Sentinel-1/2 debris benchmark.
* **2+ items in one work, incl. debris or the validation-gate idea** → reframe
  as an application/benchmark study or find a sharper twist (e.g. the
  `Q_reg`-gated confidence, or learned object descriptors for putative matches).

---

## 4b. Preliminary search findings (not exhaustive)

A first, non-exhaustive scan (Aug 2026) — a **starting point**, not the §4 search:

* **CPD for SAR↔optical registration already exists.** "Enhanced CPD" for remote-
  sensing image registration and optical–SAR registration models are published
  (e.g. SPIE *J. Applied Remote Sensing*; MDPI *Remote Sensing* optical–SAR
  multi-constraint registration). ⇒ **"CPD for SAR–optical" is prior art** — the
  claim must rest on the *object-level debris validation* framing, not the
  registration alone. Item §3.1 is partially pre-empted at the *pixel/image*
  level; check whether any of it is done at the *object/detection* level.
* **Sentinel-2 debris detection is mature** (FDI, plastic index, RF/ML on MARIDA;
  large-scale coastal detection). Not novel on its own.
* **Existing debris "fusion" is mostly optical–optical, image-level.** Reported
  fusion pairs Sentinel-2 with high-resolution *optical* (WorldView-2/3,
  PlanetScope) for spatial/spectral **super-resolution** — i.e. pixel-level image
  fusion. **Sentinel-1 SAR + Sentinel-2 fusion for debris, and _object-level
  spatial co-location as the fusion rule_, appear sparsely covered** in this first
  scan. This is the gap the claim targets — to be confirmed by the full §4 search
  (IEEE Xplore, Scopus, patents), which this scan does **not** replace.

Net: the individual ingredients are prior art; the **S1+S2, object-level,
registration-gated cross-modal _validation_** framing is where the residual
novelty most plausibly lives — still to be verified.

## 5. Honest status

* **Built & measured:** the pipeline works and beats the traditional ICP
  baseline under identical fusion (see `README.md`, `results/`).
* **Not yet done:** the prior-art search of §4, and validation on **real**
  Sentinel-1/2 debris data (MARIDA or a purpose-built annotated set).
* **Therefore:** novelty is **claimed but unverified**. Do not assert it in a
  submission until §4 is complete.
