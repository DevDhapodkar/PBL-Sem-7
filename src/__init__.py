"""Cross-modal Sentinel-1/Sentinel-2 debris detection prototype."""

from . import (
    baseline,
    data_simulation,
    detection,
    evaluate,
    fusion,
    nagpur,
    pipeline,
    registration,
    visualize,
)

# acquire / detect_real / visualize_nagpur depend on optional geo packages
# (rasterio, pyproj) — import lazily so the core pipeline works without them.

__all__ = [
    "baseline",
    "data_simulation",
    "detection",
    "evaluate",
    "fusion",
    "nagpur",
    "pipeline",
    "registration",
    "visualize",
]
