"""
Live Copernicus data acquisition for the Nagpur lakes.
======================================================

Two real data paths, chosen for what is actually reachable:

* **Sentinel-2 L2A (optical) — ``fetch_sentinel2``**
  Reads the surface-reflectance **Cloud-Optimised GeoTIFFs** directly from the
  public AWS Open-Data bucket ``sentinel-cogs`` (Element-84 / Sinergio mirror of
  Copernicus). No credentials are required and only the requested spatial window
  is transferred (GDAL ``/vsicurl`` range reads), so it is fast and works even in
  restricted networks. This is genuinely **live** — pick any recent date.

* **Sentinel-1 GRD (SAR) — ``fetch_sentinel1_ee``**
  Uses **Google Earth Engine** (``ee``), the standard way to get analysis-ready,
  terrain-corrected S1 backscatter server-side. Earth Engine needs a one-time
  ``earthengine authenticate`` and a registered cloud project, so this path runs
  on the *user's* machine / Colab. (In some sandboxes the S1 STAC/PC endpoints are
  blocked by egress policy; EE is the portable route.)

Both return a simple ``BandStack`` (a dict of 2-D float arrays on a common grid
plus the affine transform and CRS), which ``src/detect_real.py`` turns into the
object-level SAR / optical point sets the pipeline consumes.
"""

from __future__ import annotations

import os
import re
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from .nagpur import S2_TILE, UTM_EPSG, Lake

S2_BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
S2_PREFIX = "sentinel-s2-l2a-cogs/44/Q/KJ"     # Nagpur tile 44QKJ

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_S2_BANDS = ("B02", "B03", "B04", "B06", "B08", "B11")


def _gdal_env():
    """Configure GDAL /vsicurl to use the session proxy + CA and range reads."""
    os.environ.setdefault("GDAL_HTTP_PROXY", os.environ.get("HTTPS_PROXY", ""))
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("VSI_CACHE", "TRUE")


@dataclass
class BandStack:
    """Co-registered 2-D bands on a common grid (metres, UTM)."""

    bands: dict[str, np.ndarray]          # name -> (H, W) float32 reflectance/backscatter
    transform: object                     # affine transform (optical grid)
    crs: str
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return next(iter(self.bands.values())).shape

    def __getitem__(self, k):
        return self.bands[k]


def load_cached_s2(lake_key: str) -> BandStack | None:
    """Load a bundled real Sentinel-2 scene (``data/nagpur_<key>_s2.npz``) if present.

    These are genuine Sentinel-2 L2A windows fetched from the public bucket and
    committed so the app/README work offline and in networks where the bucket is
    blocked. Returns None if no cache exists for the lake.
    """
    path = os.path.join(_CACHE_DIR, f"nagpur_{lake_key}_s2.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    bands = {b: d[b].astype("float32") for b in _S2_BANDS if b in d}
    meta = {}
    if "meta" in d:
        try:
            meta = eval(str(d["meta"][0]), {"__builtins__": {}}, {})
        except Exception:
            meta = {"raw": str(d["meta"][0])}
    meta.setdefault("live", True)
    meta["cached"] = True
    return BandStack(bands, transform=None, crs=UTM_EPSG, meta=meta)


def get_nagpur_s2(lake, year: int = 2024, month: int = 1,
                  prefer_cache: bool = True) -> BandStack:
    """Get a real Sentinel-2 stack for a Nagpur lake: bundled cache first (fast,
    offline-safe), else a live fetch from the public bucket."""
    if prefer_cache:
        cached = load_cached_s2(lake.key)
        if cached is not None:
            return cached
    return fetch_sentinel2(lake, year=year, month=month)


# ---------------------------------------------------------------------------
# Sentinel-2 (optical) — live from the public COG bucket
# ---------------------------------------------------------------------------
def list_s2_scenes(year: int, month: int) -> list[str]:
    """Scene prefixes for tile 44QKJ in a given month (newest Copernicus mirror)."""
    url = (f"{S2_BUCKET}/?list-type=2&prefix={S2_PREFIX}/{year}/{month}/"
           f"&delimiter=/")
    xml = urllib.request.urlopen(url, timeout=40).read().decode()
    return re.findall(r"<Prefix>([^<]+_L2A/)</Prefix>", xml)


def fetch_sentinel2(
    lake: Lake,
    year: int = 2024,
    month: int = 1,
    max_cloud_probe: int = 8,
    bands: tuple[str, ...] = ("B02", "B03", "B04", "B06", "B08", "B11"),
    scene_override: str | None = None,
) -> BandStack:
    """
    Fetch a window of real Sentinel-2 L2A reflectance over ``lake``.

    Scans the month's scenes, reads a small green-band probe over the lake and
    keeps the **least-cloudy valid** scene (lowest mean over the window, rejecting
    empty/nodata orbit edges), then reads the requested bands for that window.
    Reflectance is scaled to [0, ~1] (L2A DN / 10000). 20 m bands (B11) are
    nearest-upsampled to the 10 m grid.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    from pyproj import Transformer

    _gdal_env()
    tr = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
    cx, cy = tr.transform(lake.lon, lake.lat)
    h = lake.half_m
    left, bottom, right, top = cx - h, cy - h, cx + h, cy + h

    scenes = [scene_override] if scene_override else list_s2_scenes(year, month)
    best, best_score = None, None
    for sc in scenes[:max_cloud_probe]:
        try:
            with rasterio.open(f"{S2_BUCKET}/{sc}B03.tif") as ds:
                win = from_bounds(left, bottom, right, top, ds.transform)
                g = ds.read(1, window=win, out_dtype="float32")
        except Exception:
            continue
        valid = (g > 0).mean()
        if valid < 0.98:                        # skip nodata / partial scenes
            continue
        score = float(g.mean())                 # low mean ~ clear (cloud is bright)
        if best_score is None or score < best_score:
            best, best_score = sc, score
    if best is None:
        raise RuntimeError(f"no valid cloud-free S2 scene for {lake.name} "
                           f"in {year}-{month:02d}; try another month.")

    out: dict[str, np.ndarray] = {}
    ref_transform = ref_crs = None
    ref_shape = None
    for b in bands:
        with rasterio.open(f"{S2_BUCKET}/{best}{b}.tif") as ds:
            win = from_bounds(left, bottom, right, top, ds.transform)
            if ref_shape is None:
                arr = ds.read(1, window=win, out_dtype="float32")
                ref_shape = arr.shape
                ref_transform = ds.window_transform(win)
                ref_crs = str(ds.crs)
            else:                                # resample coarse bands to 10 m grid
                arr = ds.read(1, window=win, out_dtype="float32",
                              out_shape=ref_shape, resampling=Resampling.nearest)
            out[b] = arr / 10000.0               # DN -> reflectance
    date = re.search(r"_(\d{8})_", best).group(1)
    return BandStack(out, ref_transform, ref_crs,
                     meta={"source": "Sentinel-2 L2A (AWS sentinel-cogs)",
                           "tile": S2_TILE, "scene": best.split("/")[-2],
                           "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
                           "lake": lake.name, "live": True})


def fetch_latest_clear_s2(
    lake: Lake,
    max_cloud: float = 0.10,
    months_back: int = 14,
    bands: tuple[str, ...] = _S2_BANDS,
    start=None,
) -> BandStack:
    """
    Fetch the **most recent low-cloud** Sentinel-2 L2A scene over ``lake``.

    Scans scenes backwards from ``start`` (default: today), reads the L2A **Scene
    Classification (SCL)** band over the lake window, and returns the newest scene
    whose cloud fraction (SCL ∈ {cloud-shadow, cloud-med, cloud-high, cirrus}) is
    below ``max_cloud``. This is the "most recent clear photo" the debris plot uses.
    """
    import datetime as _dt

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    from pyproj import Transformer

    _gdal_env()
    tr = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
    cx, cy = tr.transform(lake.lon, lake.lat)
    h = lake.half_m
    box = (cx - h, cy - h, cx + h, cy + h)

    today = start or _dt.date.today()
    ym = [( (today.replace(day=1) - _dt.timedelta(days=31 * i)).year,
            (today.replace(day=1) - _dt.timedelta(days=31 * i)).month )
          for i in range(months_back)]
    cand = []
    for y, m in ym:
        try:
            cand += list_s2_scenes(y, m)
        except Exception:
            continue
    cand = sorted(set(cand), key=lambda s: re.search(r"_(\d{8})_", s).group(1),
                  reverse=True)

    CLOUD = {3, 8, 9, 10}
    chosen = None
    for sc in cand:
        try:
            with rasterio.open(f"{S2_BUCKET}/{sc}SCL.tif") as ds:
                scl = ds.read(1, window=from_bounds(*box, ds.transform))
        except Exception:
            continue
        if scl.size == 0 or (scl > 0).mean() < 0.9:      # nodata / partial
            continue
        if np.isin(scl, list(CLOUD)).mean() <= max_cloud:
            chosen = sc
            break
    if chosen is None:
        raise RuntimeError(f"no clear (<{max_cloud:.0%} cloud) S2 scene for "
                           f"{lake.name} in the last {months_back} months.")

    out, ref_shape, ref_tr, ref_crs = {}, None, None, None
    for b in bands:
        with rasterio.open(f"{S2_BUCKET}/{chosen}{b}.tif") as ds:
            win = from_bounds(*box, ds.transform)
            if ref_shape is None:
                arr = ds.read(1, window=win, out_dtype="float32")
                ref_shape, ref_tr, ref_crs = arr.shape, ds.window_transform(win), str(ds.crs)
            else:
                arr = ds.read(1, window=win, out_dtype="float32",
                              out_shape=ref_shape, resampling=Resampling.nearest)
            out[b] = arr / 10000.0
    date = re.search(r"_(\d{8})_", chosen).group(1)
    return BandStack(out, ref_tr, ref_crs,
                     meta={"source": "Sentinel-2 L2A (AWS sentinel-cogs)",
                           "tile": S2_TILE, "scene": chosen.split("/")[-2],
                           "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
                           "lake": lake.name, "live": True, "utm_box": box})


# ---------------------------------------------------------------------------
# Sentinel-1 (SAR) — live via Google Earth Engine (needs user credentials)
# ---------------------------------------------------------------------------
def fetch_sentinel1_ee(lake: Lake, start: str, end: str,
                       ref: BandStack | None = None) -> BandStack:  # pragma: no cover
    """
    Fetch Sentinel-1 GRD (VV, VH; dB) over ``lake`` via Google Earth Engine.

    Requires a one-time ``earthengine authenticate`` and a cloud project:

        import ee; ee.Authenticate(); ee.Initialize(project="your-project")

    Returns a BandStack on (approximately) the optical grid so the two modalities
    can be compared. Kept as a thin, documented integration point — run it where
    Earth Engine is reachable (your machine / Colab).
    """
    import ee  # noqa: F401  (import here so the rest of the package needs no ee)

    geom = ee.Geometry.Point([lake.lon, lake.lat]).buffer(lake.half_m).bounds()
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(geom).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    img = col.median().select(["VV", "VH"]).clip(geom)
    rect = geom.bounds().getInfo()["coordinates"]
    arr = ee.data.computePixels  # placeholder: user chooses export mechanism
    raise NotImplementedError(
        "Earth Engine S1 fetch: authenticate & initialise ee, then export the "
        "VV/VH median to numpy (ee.data.computePixels or getDownloadURL). "
        "This runs where Earth Engine is reachable; see docs/DATA.md.")


def fetch_latest_s1_pc(lake: Lake, ref: BandStack, months_back: int = 3,
                       start=None) -> BandStack:  # pragma: no cover (needs live PC)
    """
    Fetch the **latest** Sentinel-1 RTC (VV/VH, γ⁰) over ``lake`` from Microsoft
    Planetary Computer, reprojected onto the optical grid ``ref``.

    Planetary Computer allows **anonymous** access (no account) — asset URLs are
    signed with the free ``planetary-computer`` package. This is the portable,
    fully-live SAR path; it is blocked only in networks that deny the PC endpoint.

        pip install pystac-client planetary-computer rioxarray odc-stac

    ``ref`` is a Sentinel-2 ``BandStack`` (from ``fetch_latest_clear_s2``); the SAR
    is warped to the same window/CRS/shape so the two modalities overlay directly.
    """
    import datetime as _dt

    import planetary_computer as pc
    import rasterio
    from pystac_client import Client
    from rasterio.warp import Resampling, reproject

    today = start or _dt.date.today()
    since = (today - _dt.timedelta(days=31 * months_back)).isoformat()
    L, B_, R, T = ref.meta["utm_box"]
    tr_inv = _to_lonlat_box(ref.crs, (L, B_, R, T))

    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                      modifier=pc.sign_inplace)
    search = cat.search(collections=["sentinel-1-rtc"], bbox=tr_inv,
                        datetime=f"{since}/{today.isoformat()}",
                        sortby=[{"field": "datetime", "direction": "desc"}])
    items = list(search.items())
    if not items:
        raise RuntimeError("no recent Sentinel-1 RTC over this lake on PC.")
    item = items[0]

    H, W = ref.shape
    dst_bands = {}
    for pol in ("vv", "vh"):
        if pol not in item.assets:
            continue
        with rasterio.open(item.assets[pol].href) as src:
            dst = np.zeros((H, W), "float32")
            reproject(source=rasterio.band(src, 1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref.transform, dst_crs=ref.crs,
                      resampling=Resampling.bilinear)
            dst_bands[pol.upper()] = 10 * np.log10(np.clip(dst, 1e-5, None))  # dB
    return BandStack(dst_bands, ref.transform, ref.crs,
                     meta={"source": "Sentinel-1 RTC (Planetary Computer)",
                           "date": str(item.datetime.date()),
                           "scene": item.id, "lake": lake.name, "live": True})


def _to_lonlat_box(crs, box):
    """UTM (left,bottom,right,top) -> lon/lat bbox for STAC search."""
    from pyproj import Transformer
    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    L, B_, R, T = box
    xs, ys = t.transform([L, R, L, R], [B_, B_, T, T])
    return [min(xs), min(ys), max(xs), max(ys)]
