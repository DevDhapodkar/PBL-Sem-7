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

### Most-recent clear scene

`fetch_latest_clear_s2(lake)` scans scenes backwards from today, reads the L2A
**SCL** cloud-classification band over the lake, and returns the newest scene below
a cloud threshold — the "most recent clear photo". Nagpur's July–September is
monsoon (mostly cloudy), so the newest clear scene is often from the dry season.

---

## 2. Sentinel-1 GRD/RTC (SAR)

### 2a. Public AWS GRD bucket — ✅ anonymous, no account, no STAC (recommended)

`src/acquire.py::fetch_latest_s1_aws` reads real Sentinel-1 **GRD** VV/VH straight
from the AWS Open-Data bucket **`sentinel-s1-l1c`** (`s3://sentinel-s1-l1c`, the
Sinergise mirror). This is the most **reachable** real-S1 route: unlike the S1
discovery APIs (Copernicus Data Space, Earth Engine, Planetary Computer, ASF,
Earth Search — all HTTPS services an egress policy can block), this bucket is a
bare S3 store, reachable wherever the Sentinel-2 COG bucket is, and its VV/VH
measurement GeoTIFFs are anonymously readable over HTTP range requests.

```bash
pip install rasterio pyproj            # already in requirements.txt — nothing extra
python fetch_and_overlay.py --lake ambazari --s1 aws   # fully-real S1 × S2 overlay
```

How it works (no spatial index needed — the bucket is partitioned only by date,
`GRD/<Y>/<M>/<D>/IW/DV/<scene>/`):

1. **Discover** the latest scene over the lake — scan days newest-first; within a
   day the ~800 time-sorted frames are *sampled* (every few frames, so no orbit
   pass is skipped) to find the strip crossing India, then the strip's
   time-neighbours are *zoomed* and the frame whose `productInfo.json` footprint
   contains the lake is kept. Sentinel-1 images a given point only every ~6–12
   days, so most days are ruled out in a couple of requests. Result is cached
   per-day in `data/_s1_aws_index.json`.
2. **Read + geocode** — each GRD `measurement/iw-vv.tiff` already carries the
   geolocation grid as ~200 embedded GCPs (EPSG:4326), so only the lake window is
   `reproject`-ed (via those GCPs) onto the Sentinel-2 grid — a few `/vsicurl`
   range reads, ~2 s. Backscatter is returned as `10·log10(DN²)` dB, an
   *uncalibrated γ⁰ proxy* — all the CFAR local-contrast detector needs.

Because the two products are both delivered geocoded on a shared grid, the
overlay registers them with the **bounded translation-only** model
(`run_proposed(registration="translation", max_translation=…)`): the residual is a
small co-registration offset + limited inter-pass debris drift, and a free
rotation/scale would over-fit the handful of real SAR detections.

### 2b. Microsoft Planetary Computer — ✅ anonymous, no account

`src/acquire.py::fetch_latest_s1_pc` searches the `sentinel-1-rtc` collection on
Planetary Computer, signs the asset URLs anonymously with the free
`planetary-computer` package, and warps VV/VH (γ⁰, dB) onto the optical grid.

```bash
pip install pystac-client planetary-computer rioxarray
python fetch_and_overlay.py --lake ambazari --s1 pc     # real S1 × real S2 overlay
```

Portable and fully-live where the PC endpoint is reachable; `fetch_and_overlay.py
--s1 auto` tries the AWS bucket (§2a) first and falls back to this.

### 2b. Google Earth Engine — ⚙️ needs one-time auth

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
**allows public AWS S3 Open-Data buckets** but **blocks** every S1 *discovery API*
(Earth Search, Planetary Computer, Copernicus Data Space, ASF), and Earth Engine
has no credentials. Because the AWS route (§2a) needs only S3 — no discovery
service — **both sensors are genuinely live here**: `--s1 aws` fetches the latest
real Sentinel-1D GRD over the lake and the latest clear real Sentinel-2, and
aligns them. (`--s1 pc` fails in this sandbox with a 403 on the PC endpoint, by
design — that host is not on the allow-list; do not route around it.)

`--s1 simulate` remains as a fully-offline fallback: it derives the SAR point set
from the real optical detections (`simulate_sar_pointset`) — an independent,
noisy, mis-registered sample plus SAR clutter — for environments with no outbound
network at all. It is clearly labelled as a stand-in.

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
