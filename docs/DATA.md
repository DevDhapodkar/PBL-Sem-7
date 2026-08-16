# Data acquisition guide

How the prototype gets **real Sentinel-1 / Sentinel-2** data for the Nagpur lakes,
and how to run each path yourself.

All Nagpur lakes fall in **Sentinel-2 MGRS tile `44QKJ`** (UTM zone 44N /
EPSG:32644). Lake centres and windows are in [`src/nagpur.py`](../src/nagpur.py).

---

## 1. Sentinel-2 L2A (optical) — public COG bucket ✅ works with no credentials

Surface-reflectance Cloud-Optimised GeoTIFFs are mirrored on the AWS Open Data
bucket **`sentinel-cogs`** (`s3://sentinel-cogs`, region `us-west-2`, free). The
pipeline reads only the requested spatial window via GDAL `/vsicurl` HTTP range
reads, so it is fast and needs no login.

```python
from src.nagpur import get_lake
from src.acquire import fetch_sentinel2

bs = fetch_sentinel2(get_lake("ambazari"), year=2024, month=1)
# picks the least-cloudy valid scene over the lake that month
print(bs.meta)            # scene id, date, tile, source
print(bs.bands.keys())    # B02 B03 B04 B06 B08 B11 (reflectance 0..~1)
```

Bucket layout used:
`sentinel-cogs/sentinel-s2-l2a-cogs/44/Q/KJ/{year}/{month}/S2X_44QKJ_{date}_0_L2A/{BAND}.tif`.

**Bundled cache.** Five lakes are committed under `data/nagpur_<key>_s2.npz` (real
scenes, fetched once) so the app and README work offline / where the bucket is
blocked. `get_nagpur_s2(lake)` returns the cache if present, else fetches live.

### Alternative S2 sources
* **Copernicus Data Space Ecosystem** (`catalogue.dataspace.copernicus.eu`, STAC +
  OData; free account) — the authoritative source; use `pystac-client` / `sentinelsat`.
* **Element-84 Earth Search** STAC (`earth-search.aws.element84.com`) — search API
  in front of the same `sentinel-cogs` COGs.
* **Microsoft Planetary Computer** — STAC + signed asset URLs.

---

## 2. Sentinel-1 GRD (SAR) — Google Earth Engine ⚙️ needs one-time auth

Earth Engine serves analysis-ready, terrain-corrected S1 backscatter server-side —
the most portable route for VV/VH over an AOI.

```bash
pip install earthengine-api
earthengine authenticate           # one-time browser login
```

```python
import ee
ee.Initialize(project="your-cloud-project")

from src.nagpur import get_lake
from src.acquire import fetch_sentinel1_ee
bs_sar = fetch_sentinel1_ee(get_lake("ambazari"), "2024-01-01", "2024-01-31")
# VV/VH median (dB) over the lake window, on ~the optical grid
```

Then run fully-live cross-modal:

```python
from src.detect_real import build_scene_from_real
from src.pipeline import run_proposed
scene, diag = build_scene_from_real(bs_optical, bs_sar)   # both real
result = run_proposed(scene, putative_radius=14, ransac_threshold=4, match_radius=5)
```

### Alternative S1 sources
* **Copernicus Data Space** (GRD products, free account).
* **Microsoft Planetary Computer** `sentinel-1-grd` / `sentinel-1-rtc` (STAC + signing).
* **Alaska Satellite Facility (ASF)** for GRD/RTC downloads.

---

## 3. Network note (this project's build sandbox)

In the hosted build environment used to develop this repo, the egress policy
**allows** the public `sentinel-cogs` S3 bucket (so **Sentinel-2 is genuinely
live**) but **blocks** the S1 STAC endpoints (Earth Search, Planetary Computer,
Copernicus) and Earth Engine has no credentials. There, the SAR point set is
**derived from the real optical detections** (`simulate_sar_pointset`) — an
independent, noisy, mis-registered sample plus SAR clutter — so the cross-modal
demo still runs on the **real Nagpur optical scene**. On your own machine / Colab,
use §2 for a fully-live S1×S2 result.

---

## 4. Detectors (bands → point sets)

Implemented in [`src/detect_real.py`](../src/detect_real.py):

| Cue | Formula | Flags |
|-----|---------|-------|
| Water mask | NDWI = (B03 − B08)/(B03 + B08) | open water |
| Floating vegetation | NDVI = (B08 − B04)/(B08 + B04) | hyacinth / algal scum |
| Floating debris | FDI = B08 − [B06 + (B11 − B06)·λ] (Biermann 2020) | plastic / trash rafts |
| SAR anomaly | \|VV − localmedian\| > k·localstd on water | roughness-modulating mats |

Connected components → centroids = point sets; index value → detection strength.
