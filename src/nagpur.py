"""
Nagpur water bodies — areas of interest (AOIs) for the debris pipeline.
=======================================================================

Nagpur (Maharashtra, India) is the "City of Lakes": a set of historic tanks and
reservoirs that are the target of this study. Each lake is given as a centre
(lat, lon) and a half-window in metres. All of them fall inside a **single
Sentinel-2 MGRS tile, 44QKJ** (UTM zone 44N / EPSG:32644), which the acquisition
module uses to read Copernicus data directly.

Floating matter on these inland lakes is dominated by **water hyacinth / algal
scum, plastic and mixed trash rafts, and shoreline litter** — all of which
(a) reflect strongly in the NIR/red-edge (optical cue) and (b) modulate the
water-surface roughness (SAR cue), which is exactly what the cross-modal pipeline
exploits.
"""

from __future__ import annotations

from dataclasses import dataclass

# Sentinel-2 tile / projection that covers the Nagpur lakes
S2_TILE = "44QKJ"
UTM_EPSG = "EPSG:32644"      # UTM zone 44N
TILE_BOUNDS_UTM = (199980, 2290200, 309780, 2400000)  # (left, bottom, right, top)


@dataclass(frozen=True)
class Lake:
    key: str
    name: str
    lat: float
    lon: float
    half_m: float = 900.0     # half-window (metres); full window = 2*half
    note: str = ""


# Major Nagpur water bodies (centres are approximate lake centroids).
LAKES: dict[str, Lake] = {
    "ambazari": Lake("ambazari", "Ambazari Lake", 21.1307, 79.0428, 1000,
                     "largest of Nagpur's lakes; feeds the Nag river"),
    "futala": Lake("futala", "Futala Lake", 21.1497, 79.0447, 700,
                   "popular lake in the city centre"),
    "gorewada": Lake("gorewada", "Gorewada Lake", 21.2035, 79.0295, 1200,
                     "reservoir / drinking-water source in the north"),
    "gandhisagar": Lake("gandhisagar", "Gandhisagar (Shukrawari) Lake",
                        21.1530, 79.1000, 500, "historic tank in the old city"),
    "sonegaon": Lake("sonegaon", "Sonegaon Lake", 21.1005, 79.0525, 700,
                     "lake in the south-west"),
    "telangkhedi": Lake("telangkhedi", "Telangkhedi (Seminary Hills) Lake",
                        21.1553, 79.0525, 500, "near Seminary Hills"),
    "sakkardara": Lake("sakkardara", "Sakkardara Lake", 21.1250, 79.1180, 500,
                       "historic tank in the east"),
    "naik": Lake("naik", "Naik Lake", 21.1360, 79.1075, 400,
                 "small urban tank, prone to weed cover"),
}

DEFAULT_LAKE = "ambazari"


def get_lake(key: str) -> Lake:
    if key not in LAKES:
        raise KeyError(f"unknown lake '{key}'. Options: {list(LAKES)}")
    return LAKES[key]
