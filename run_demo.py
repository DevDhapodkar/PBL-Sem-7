#!/usr/bin/env python3
"""
End-to-end demo & benchmark for the Sentinel-1/Sentinel-2 debris pipeline.
==========================================================================

Runs four methods on the same simulated scene(s):

    1. optical_only     -- single-modality baseline (no fusion)
    2. no_alignment     -- traditional cross-modal fusion WITHOUT point-set
                           alignment: overlay the two nominally-geocoded detection
                           sets as-is (identity) and take the agreement
    3. traditional_icp  -- cross-modal fusion with classic ICP registration
    4. proposed         -- RANSAC init -> CPD refine -> cross-modal confidence

Crucially, ``no_alignment``, ``traditional_icp`` and ``proposed`` share the
*identical* cross-modal validation / confidence stage and decision threshold --
they differ ONLY in the registration step (none / ICP / RANSAC+CPD). Any gap is
therefore attributable to the alignment method. ``no_alignment`` is the key
comparison: it is exactly the conventional method *without the point-set alignment
that is this project's contribution*. ``optical_only`` shows why cross-modal
validation is worth doing at all.

Outputs (figures + machine-readable metrics) land in ``results/``.

    python run_demo.py                # single seed figures + averaged benchmark
    python run_demo.py --trials 30    # more seeds for tighter averages
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.baseline import run_no_alignment, run_optical_only, run_traditional
from src.data_simulation import simulate_scene
from src.evaluate import (
    best_f1,
    precision_recall_curve,
    score_locations,
    strength_pr_curve,
)
from src.pipeline import run_proposed
from src import visualize

RESULTS = Path(__file__).parent / "results"
MATCH_RADIUS = 20.0
CONF_THRESHOLD = 0.30      # decision threshold on cross-modal confidence C
OPTICAL_THRESHOLD = 0.50   # decision threshold on optical strength (typical FDI op-point)


# ---------------------------------------------------------------------------
def _scored(candidates):
    """Extract (xy, confidence) tuples from fusion candidates."""
    return [(c.xy, c.confidence) for c in candidates]


def evaluate_all(scene, *, verbose=False):
    """Run the three methods on one scene; return results, fixed-point metrics,
    and per-method scored points (for threshold sweeps)."""
    truth = scene.true_debris

    optical = run_optical_only(scene)
    no_align = run_no_alignment(scene, match_radius=MATCH_RADIUS)
    traditional = run_traditional(scene, match_radius=MATCH_RADIUS)
    proposed = run_proposed(scene, match_radius=MATCH_RADIUS,
                            confidence_threshold=CONF_THRESHOLD, verbose=verbose)

    scored = {
        "optical_only": [(d.xy, d.strength) for d in scene.optical],
        "no_alignment": _scored(no_align.fusion.candidates),
        "traditional_icp": _scored(traditional.fusion.candidates),
        "proposed": _scored(proposed.fusion.candidates),
    }
    fixed_thr = {"optical_only": OPTICAL_THRESHOLD,
                 "no_alignment": CONF_THRESHOLD,
                 "traditional_icp": CONF_THRESHOLD,
                 "proposed": CONF_THRESHOLD}

    rows = {}
    for name, pts in scored.items():
        sel = np.array([xy for xy, s in pts if s >= fixed_thr[name]]) \
            if pts else np.empty((0, 2))
        rows[name] = score_locations(sel if len(sel) else np.empty((0, 2)),
                                     truth, MATCH_RADIUS)
    results = {"optical": optical, "no_alignment": no_align,
               "traditional": traditional, "proposed": proposed}
    return results, rows, scored


def robustness_sweep(clutter_levels, trials=8):
    """Precision, recall & F1 vs clutter load, averaged over seeds, per method."""
    keys = ("proposed", "traditional_icp", "no_alignment", "optical_only")
    out = {"x": clutter_levels}
    for metric in ("precision", "recall", "f1"):
        for k in keys:
            out[f"{k}_{metric}"] = []
    for load in clutter_levels:
        acc = {f"{k}_{m}": [] for k in keys for m in ("precision", "recall", "f1")}
        for s in range(trials):
            scene = simulate_scene(n_false_sar=load, n_false_optical=int(load * 1.2),
                                   seed=1000 + s)
            _, rows, _ = evaluate_all(scene)
            for k in keys:
                acc[f"{k}_precision"].append(rows[k].precision)
                acc[f"{k}_recall"].append(rows[k].recall)
                acc[f"{k}_f1"].append(rows[k].f1)
        for key, vals in acc.items():
            out[key].append(float(np.mean(vals)))
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20,
                    help="seeds for the averaged benchmark")
    ap.add_argument("--seed", type=int, default=42, help="seed for the figure scene")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    # ---- 1. headline single scene (for figures) ----------------------------
    print("=" * 72)
    print("SINGLE SCENE  (seed=%d)" % args.seed)
    print("=" * 72)
    scene = simulate_scene(seed=args.seed)
    print(f"ground-truth debris : {len(scene.true_debris)}")
    print(f"SAR detections      : {len(scene.sar)} "
          f"({sum(d.is_true_debris for d in scene.sar)} real / "
          f"{sum(not d.is_true_debris for d in scene.sar)} clutter)")
    print(f"optical detections  : {len(scene.optical)} "
          f"({sum(d.is_true_debris for d in scene.optical)} real / "
          f"{sum(not d.is_true_debris for d in scene.optical)} clutter)\n")

    results, rows, scored = evaluate_all(scene, verbose=True)

    print("\n" + "-" * 72)
    print("Fixed operating point  (proposed & traditional share threshold C>=%.2f)"
          % CONF_THRESHOLD)
    print("-" * 72)
    print(f"{'method':<18}{'TP':>5}{'FP':>5}{'FN':>5}"
          f"{'precision':>11}{'recall':>9}{'F1':>7}")
    for name in ("optical_only", "no_alignment", "traditional_icp", "proposed"):
        m = rows[name]
        print(f"{name:<18}{m.tp:>5}{m.fp:>5}{m.fn:>5}"
              f"{m.precision:>11.3f}{m.recall:>9.3f}{m.f1:>7.3f}")

    # ---- 2. best-F1 operating points + PR curves ---------------------------
    print("\n" + "-" * 72)
    print("Best-F1 operating point  (each method swept over its own threshold)")
    print("-" * 72)
    print(f"{'method':<18}{'bestF1':>8}{'precision':>11}{'recall':>9}{'@thr':>8}")
    bf1 = {}
    for name in ("optical_only", "no_alignment", "traditional_icp", "proposed"):
        b = best_f1(scored[name], scene.true_debris, MATCH_RADIUS)
        bf1[name] = b
        print(f"{name:<18}{b['f1']:>8.3f}{b['precision']:>11.3f}"
              f"{b['recall']:>9.3f}{b['threshold']:>8.2f}")

    rec_p, prec_p, _, ap_p = precision_recall_curve(
        results["proposed"].fusion.candidates, scene.true_debris, MATCH_RADIUS)
    rec_n, prec_n, ap_n = strength_pr_curve(
        scored["no_alignment"], scene.true_debris, MATCH_RADIUS)
    rec_t, prec_t, ap_t = strength_pr_curve(
        scored["traditional_icp"], scene.true_debris, MATCH_RADIUS)
    rec_o, prec_o, ap_o = strength_pr_curve(
        scored["optical_only"], scene.true_debris, MATCH_RADIUS)
    print(f"\nAverage precision (area under PR):  proposed={ap_p:.3f}  "
          f"traditional_icp={ap_t:.3f}  no_alignment={ap_n:.3f}  "
          f"optical_only={ap_o:.3f}")

    # ---- 3. figures --------------------------------------------------------
    visualize.plot_registration(scene, results["proposed"], results["traditional"],
                                RESULTS / "01_registration.png")
    visualize.plot_confidence(scene, results["proposed"], CONF_THRESHOLD,
                              RESULTS / "02_confidence.png")
    visualize.plot_pr_curves(
        {"proposed": (rec_p, prec_p, ap_p),
         "traditional_icp": (rec_t, prec_t, ap_t),
         "no_alignment": (rec_n, prec_n, ap_n),
         "optical_only": (rec_o, prec_o, ap_o)},
        RESULTS / "03_precision_recall.png")

    # ---- 4. multi-seed averaged benchmark ----------------------------------
    print("\n" + "=" * 72)
    print(f"AVERAGED BENCHMARK  ({args.trials} seeds, fixed operating point)")
    print("=" * 72)
    agg = {k: {"precision": [], "recall": [], "f1": [], "ap": []}
           for k in ("optical_only", "no_alignment", "traditional_icp", "proposed")}
    for s in range(args.trials):
        sc = simulate_scene(seed=200 + s)
        res, r, sco = evaluate_all(sc)
        for k in agg:
            agg[k]["precision"].append(r[k].precision)
            agg[k]["recall"].append(r[k].recall)
            agg[k]["f1"].append(r[k].f1)
        agg["proposed"]["ap"].append(
            precision_recall_curve(res["proposed"].fusion.candidates,
                                   sc.true_debris, MATCH_RADIUS)[3])
        for k in ("no_alignment", "traditional_icp", "optical_only"):
            agg[k]["ap"].append(
                strength_pr_curve(sco[k], sc.true_debris, MATCH_RADIUS)[2])

    print(f"{'method':<18}{'precision':>13}{'recall':>13}{'F1':>13}{'AP':>8}")
    print("-" * 65)
    bench = {}
    for k in ("optical_only", "no_alignment", "traditional_icp", "proposed"):
        def ms(x): return (float(np.mean(x)), float(np.std(x)))
        p, ps = ms(agg[k]["precision"]); r, rs = ms(agg[k]["recall"])
        f, fs = ms(agg[k]["f1"]); a, _ = ms(agg[k]["ap"])
        bench[k] = {"precision": [round(p, 3), round(ps, 3)],
                    "recall": [round(r, 3), round(rs, 3)],
                    "f1": [round(f, 3), round(fs, 3)],
                    "ap": round(a, 3)}
        print(f"{k:<18}{p:>7.3f}±{ps:<4.2f}{r:>7.3f}±{rs:<4.2f}"
              f"{f:>7.3f}±{fs:<4.2f}{a:>8.3f}")
    print("-" * 65)

    # ---- 5. robustness-to-clutter sweep (precision headline) ---------------
    print("\nRunning clutter-robustness sweep ...")
    sweep = robustness_sweep([5, 15, 30, 50, 75], trials=8)
    visualize.plot_robustness(sweep, RESULTS / "04_robustness.png")
    print("precision vs clutter load %s :" % sweep["x"])
    for k in ("optical_only", "no_alignment", "traditional_icp", "proposed"):
        print(f"  {k:<16}", [round(v, 3) for v in sweep[f"{k}_precision"]])

    # ---- 6. persist a machine-readable report ------------------------------
    report = {
        "single_scene": {name: rows[name].as_row()
                         for name in ("optical_only", "no_alignment",
                                      "traditional_icp", "proposed")},
        "single_scene_best_f1": bf1,
        "average_precision_single": {"proposed": round(ap_p, 3),
                                     "traditional_icp": round(ap_t, 3),
                                     "no_alignment": round(ap_n, 3),
                                     "optical_only": round(ap_o, 3)},
        "averaged_benchmark": bench,
        "robustness_sweep": sweep,
        "config": {"match_radius": MATCH_RADIUS, "confidence_threshold": CONF_THRESHOLD,
                   "optical_threshold": OPTICAL_THRESHOLD, "trials": args.trials},
    }
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))

    print("\nFigures + metrics written to:", RESULTS)
    for f in sorted(RESULTS.glob("*")):
        print("  -", f.name)


if __name__ == "__main__":
    main()
