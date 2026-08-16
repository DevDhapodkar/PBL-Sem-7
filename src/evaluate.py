"""
Evaluation: score reported debris against ground truth.
=======================================================

A method reports debris *locations* in the optical/common frame. We greedily
match them to ground-truth debris within ``match_radius`` (one-to-one) and count

    TP  reported location matched to an unused ground-truth debris
    FP  reported location with no ground-truth debris nearby
    FN  ground-truth debris that nothing reported

from which precision / recall / F1 follow. For the proposed method we also sweep
the confidence threshold to produce a precision-recall curve and average
precision, quantifying the value of the confidence score itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Metrics:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_row(self) -> dict:
        return {
            "TP": self.tp, "FP": self.fp, "FN": self.fn,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "F1": round(self.f1, 3),
        }


def score_locations(reported: np.ndarray, truth: np.ndarray,
                    match_radius: float = 20.0) -> Metrics:
    """Greedy one-to-one matching of reported -> truth within ``match_radius``."""
    n_truth = len(truth)
    if len(reported) == 0:
        return Metrics(0, 0, n_truth)
    if n_truth == 0:
        return Metrics(0, len(reported), 0)

    tree = cKDTree(truth)
    used = np.zeros(n_truth, bool)
    tp = 0
    for r in reported:
        dists, idxs = tree.query(r, k=min(5, n_truth))
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)
        hit = False
        for d, gi in zip(dists, idxs):
            if d <= match_radius and not used[gi]:
                used[gi] = True
                tp += 1
                hit = True
                break
        # (a reported point whose only near-GT are already used counts as FP)
        _ = hit
    fp = len(reported) - tp
    fn = n_truth - int(used.sum())
    return Metrics(tp, fp, fn)


def _sweep(scored_points, truth, thresholds, match_radius):
    """scored_points: list of (xy, score). Returns recalls, precisions, f1s."""
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        pts = np.array([xy for xy, s in scored_points if s >= t])
        m = score_locations(pts if len(pts) else np.empty((0, 2)), truth, match_radius)
        precisions.append(m.precision)
        recalls.append(m.recall)
        f1s.append(m.f1)
    return np.array(recalls), np.array(precisions), np.array(f1s)


def precision_recall_curve(candidates, truth: np.ndarray,
                           match_radius: float = 20.0, n_steps: int = 60):
    """Sweep confidence thresholds -> (recalls, precisions, thresholds, AP)."""
    confs = np.array([c.confidence for c in candidates])
    if len(confs) == 0:
        return np.array([0.0]), np.array([1.0]), np.array([0.0]), 0.0
    thresholds = np.linspace(0, max(confs.max(), 1e-6), n_steps)
    scored = [(c.xy, c.confidence) for c in candidates]
    recalls, precisions, _ = _sweep(scored, truth, thresholds, match_radius)
    order = np.argsort(recalls)
    ap = float(np.trapezoid(precisions[order], recalls[order]))
    return recalls, precisions, thresholds, ap


def strength_pr_curve(scored_points, truth: np.ndarray,
                      match_radius: float = 20.0, n_steps: int = 60):
    """PR curve for any (xy, score) list (e.g. optical-only strength sweep)."""
    if len(scored_points) == 0:
        return np.array([0.0]), np.array([1.0]), 0.0
    smax = max(s for _, s in scored_points)
    thresholds = np.linspace(0, max(smax, 1e-6), n_steps)
    recalls, precisions, _ = _sweep(scored_points, truth, thresholds, match_radius)
    order = np.argsort(recalls)
    ap = float(np.trapezoid(precisions[order], recalls[order]))
    return recalls, precisions, ap


def best_f1(scored_points, truth: np.ndarray, match_radius: float = 20.0,
            n_steps: int = 60):
    """Best achievable F1 over a threshold sweep, with the threshold + P/R there."""
    if len(scored_points) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.0}
    smax = max(s for _, s in scored_points)
    thresholds = np.linspace(0, max(smax, 1e-6), n_steps)
    recalls, precisions, f1s = _sweep(scored_points, truth, thresholds, match_radius)
    k = int(np.argmax(f1s))
    return {"f1": float(f1s[k]), "precision": float(precisions[k]),
            "recall": float(recalls[k]), "threshold": float(thresholds[k])}
