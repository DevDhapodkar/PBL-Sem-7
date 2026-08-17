#!/usr/bin/env python3
"""
Latest-Sentinel-1 × latest-clear-Sentinel-2 debris overlay for a Nagpur lake.
=============================================================================

Fetches the **most recent low-cloud Sentinel-2** optical scene and the **latest
Sentinel-1 SAR** scene over a chosen Nagpur water body, detects floating debris in
each, then registers the two object point sets with **RANSAC + CPD** and draws a
drift-aware overlay comparing the debris seen by both sensors.

Why registration is needed: Sentinel-1 and Sentinel-2 image at *different times*,
so floating debris physically **drifts** between the two passes (plus a residual
co-registration offset). The RANSAC+CPD step estimates that drift/offset and lines
the two debris clouds up, so genuine debris seen by *both* sensors can be matched.

Usage
-----
    python fetch_and_overlay.py                       # Ambazari, auto S1
    python fetch_and_overlay.py --lake gorewada
    python fetch_and_overlay.py --lake futala --s1 pc # force real S1 (Planetary Computer)
    python fetch_and_overlay.py --s1 simulate         # S1 derived from optical (offline demo)

S1 modes:
  auto      try real Sentinel-1 (Planetary Computer, anonymous); fall back to
            'simulate' if the PC endpoint is unreachable (e.g. restricted network).
  pc        require real Sentinel-1 RTC from Planetary Computer.
  simulate  derive an independent, drifted SAR point set from the real optical
            detections (works fully offline; clearly a stand-in).

Real Sentinel-1 needs:  pip install pystac-client planetary-computer rioxarray
"""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from src.acquire import fetch_latest_clear_s2, get_nagpur_s2
from src.detect_real import (
    build_scene_from_real,
    detect_optical,
    detect_sar_backscatter,
    simulate_sar_pointset,
    water_mask,
)
from src.data_simulation import Scene
from src.nagpur import get_lake
from src.pipeline import run_proposed
from src import visualize_nagpur as vn


def _try_real_s2(lake, allow_live):
    """Most recent clear Sentinel-2 (live); fall back to the bundled cache."""
    if allow_live:
        try:
            bs = fetch_latest_clear_s2(lake)
            print(f"  Sentinel-2 (optical): {bs.meta['scene']}  {bs.meta['date']}  "
                  f"[most recent clear, live]")
            return bs
        except Exception as e:
            print(f"  live S2 fetch failed ({str(e)[:60]}); using bundled cache.")
    bs = get_nagpur_s2(lake)
    print(f"  Sentinel-2 (optical): {bs.meta.get('scene','?')}  "
          f"{bs.meta.get('date','?')}  [bundled real scene]")
    return bs


def _get_s1(mode, lake, bs, opt_dets, water, seed):
    """Return (sar_detections, true_transform, s1_meta) per the chosen S1 mode."""
    if mode in ("auto", "pc"):
        try:
            from src.acquire import fetch_latest_s1_pc
            bs_sar = fetch_latest_s1_pc(lake, bs)
            sar = detect_sar_backscatter(bs_sar, water)
            print(f"  Sentinel-1 (SAR):     {bs_sar.meta['scene']}  "
                  f"{bs_sar.meta['date']}  [Planetary Computer, live]  "
                  f"-> {len(sar)} detections")
            return sar, {"matrix_sar_to_opt": np.eye(3), "note": "live S1"}, bs_sar.meta
        except Exception as e:
            if mode == "pc":
                raise
            print(f"  live S1 unavailable ({str(e)[:70]});")
            print("  -> falling back to S1 derived from the real optical scene.")
    sar, tt, _ = simulate_sar_pointset(opt_dets, water, seed=seed)
    meta = {"source": "Sentinel-1 (simulated from real optical)",
            "date": "n/a (stand-in)", "scene": "derived"}
    print(f"  Sentinel-1 (SAR):     derived from optical [offline stand-in]  "
          f"-> {len(sar)} detections")
    return sar, tt, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lake", default="ambazari", help="Nagpur lake key (see src/nagpur.py)")
    ap.add_argument("--s1", choices=["auto", "pc", "simulate"], default="auto")
    ap.add_argument("--no-live-s2", action="store_true",
                    help="skip the live S2 fetch and use the bundled cache")
    ap.add_argument("--out", default="results/overlay_latest.png")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    lake = get_lake(args.lake)
    print(f"\nNagpur lake: {lake.name}")
    print("Fetching latest imagery …")
    bs = _try_real_s2(lake, allow_live=not args.no_live_s2)

    # --- detect floating matter in the real optical scene ---
    opt_dets, diag = detect_optical(bs)
    water = diag["water"]
    print(f"  optical floating-matter detections: {len(opt_dets)}")

    # --- get the SAR view + detections ---
    sar_dets, tt, s1_meta = _get_s1(args.s1, lake, bs, opt_dets, water, args.seed)

    # --- assemble scene + register with RANSAC+CPD ---
    scene = Scene(true_debris=np.empty((0, 2)), sar=sar_dets, optical=opt_dets,
                  true_transform=tt)
    proposed = run_proposed(scene, putative_radius=16, ransac_threshold=5,
                            match_radius=6)

    # --- estimate drift (translation of the recovered transform) ---
    drift_px = float(np.hypot(*proposed.T_final[:2, 2]))
    drift_m = drift_px * 10.0                       # 10 m pixels
    print("\nRegistration (RANSAC → CPD):")
    print(f"  putative {proposed.n_putative} · RANSAC inliers {proposed.n_ransac_inliers}"
          f" · CPD iters {proposed.cpd_iterations}")
    print(f"  D_reg (RMSE): {proposed.fusion.d_registration:.2f} px "
          f"(~{proposed.fusion.d_registration*10:.0f} m)")
    print(f"  estimated debris drift / offset between passes: "
          f"~{drift_px:.1f} px (~{drift_m:.0f} m)")
    print(f"  debris corroborated by BOTH sensors: {proposed.fusion.n_matches}")

    # --- time gap ---
    try:
        d2 = dt.date.fromisoformat(bs.meta.get("date"))
        d1 = dt.date.fromisoformat(s1_meta.get("date"))
        print(f"  acquisition gap: {abs((d2 - d1).days)} days "
              f"(S2 {d2}  vs  S1 {d1})")
    except Exception:
        pass

    # --- overlay figure ---
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vn.plot_overlay(bs, scene, proposed, path=args.out,
                    s1_date=s1_meta.get("date", "?"), s2_date=bs.meta.get("date", "?"),
                    s1_source=s1_meta.get("source", "Sentinel-1"))
    print(f"\nOverlay written to: {args.out}\n")


if __name__ == "__main__":
    main()
