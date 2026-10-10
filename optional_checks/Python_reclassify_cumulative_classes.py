#!/usr/bin/env python3
"""

Recomputes, from the downloaded annual z-score rasters, the same classes as
GEE_05 section 7: for each pixel the number of wartime years (2022-2024) in
which the annual z-score was >= T (loss) or <= -T (gain). The mapped classes
keep the DOMINANT direction (more anomalous years; ties by the sign of the
median z) and remove 8-connected patches smaller than 8 cells per direction.
The table also gives the separate loss and gain unions (a pixel can be in
both) and their overlap. Observed areas only: the annual comparisons share one
pre-war baseline, so no independent-year (binomial) expectation is computed.

Areas use 0.25 km2 per complete 500 m cell of the EPSG:3035 grid, which is
independent of the GEE area table; the GEE areas use pixelArea() and differ by
less than 0.01% on this grid.

Inputs per scheme folder:
  Debiased_RF_Main_revision_annualbase_annual_median.tif  (band 7 = z_median)
  Debiased_RF_20XX_AnnualInputSensitivity_revision_annualbase_annual_median.tif
      (band 3 = annual z-score; all three years required)
Outputs per scheme folder:
  cumulative_confidence_classes_local.tif  (1 class_mapped, 2 class_px,
      3 n_loss_years, 4 n_gain_years)
  cumulative_confidence_class_areas_local.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

import sys as _sys
try:                        # run as a file (also via Python_02)
    _SUPPORT = Path(__file__).resolve().parents[1] / "support"  # moved to optional_checks/
except NameError:           # pasted into a notebook cell
    _SUPPORT = Path.cwd() / "support"
_sys.path.insert(0, str(_SUPPORT))
# A console that already ran another copy of the package keeps its support
# modules in memory; drop them so that this package's support folder is used.
for _m in ("folders", "pipeline_checks"):
    _sys.modules.pop(_m, None)
import folders  # noqa: E402  (local folders: support/folders.py)

SCHEMES = ["primary_effort", "recurrent_effort"]

T_EVIDENCE = {"primary_effort": 2.5, "recurrent_effort": 3.0}   # as GEE_05
MIN_PATCH_PX = 8

MAIN_NAME = "Debiased_RF_Main_revision_annualbase_annual_median.tif"
YEAR_NAME = ("Debiased_RF_{y}_AnnualInputSensitivity_revision_"
             "annualbase_annual_median.tif")
Z_BAND_MAIN = 7
Z_BAND_YEAR = 3
NODATA_IN = -9999.0
NODATA_OUT = -9999.0
CELL_KM2 = 0.25

STRUCT8 = np.ones((3, 3), dtype=bool)


def patch_filter(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask, structure=STRUCT8)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= MIN_PATCH_PX
    keep[0] = False
    return keep[lab]


def run_scheme(scheme: str) -> None:
    folder = folders.data_dir(folders.GEE05[scheme])
    main_path = folder / MAIN_NAME
    if not main_path.exists():
        print(f"[skip] {scheme}: {main_path} not found")
        return
    year_paths = [folder / YEAR_NAME.format(y=y) for y in (2022, 2023, 2024)]
    missing = [p.name for p in year_paths if not p.exists()]
    if missing:
        print(f"[skip] {scheme}: year counts need all three annual stacks; "
              f"missing {missing}")
        return

    T = T_EVIDENCE[scheme]
    print(f"=== {scheme}: anomalous year = z_year >= {T} or <= -{T} ===")
    with rasterio.open(main_path) as src:
        z_med = src.read(Z_BAND_MAIN).astype(np.float32)
        profile = src.profile
        if src.crs is None or src.crs.to_epsg() != 3035 or \
                not np.isclose(abs(src.transform.a * src.transform.e), 250000.0):
            raise ValueError("expected the 500 m EPSG:3035 grid")
    valid = np.isfinite(z_med) & (z_med != NODATA_IN)

    pos = np.zeros(z_med.shape, dtype=np.int8)
    neg = np.zeros(z_med.shape, dtype=np.int8)
    for p in year_paths:
        with rasterio.open(p) as s:
            zy = s.read(Z_BAND_YEAR).astype(np.float32)
        ok = np.isfinite(zy) & (zy != NODATA_IN)
        valid &= ok
        pos += ((zy >= T) & ok).astype(np.int8)
        neg += ((zy <= -T) & ok).astype(np.int8)

    loss_wins = (pos > neg) | ((pos == neg) & (pos > 0) & (z_med >= 0))
    gain_wins = (neg > pos) | ((pos == neg) & (neg > 0) & (z_med < 0))
    cls_px = np.where(loss_wins, pos, 0).astype(np.int8)
    cls_px = np.where(gain_wins, -neg, cls_px).astype(np.int8)
    cls_px[~valid] = 0
    keep = patch_filter(valid & (cls_px > 0)) | patch_filter(valid & (cls_px < 0))
    cls_map = np.where(keep, cls_px, 0).astype(np.int8)

    rows = []
    for variant, arr, patch in (("pixel", cls_px, 0), ("mapped", cls_map, MIN_PATCH_PX)):
        for cls in (-3, -2, -1, 1, 2, 3):
            rows.append({"variant": variant, "set": "dominant_class", "class": cls,
                         "direction": "loss" if cls > 0 else "gain",
                         "min_years": abs(cls),
                         "observed_km2": float(np.sum(valid & (arr == cls))) * CELL_KM2,
                         "min_patch_px": patch})
    for name, direction, k, m in (("union_loss_ge1", "loss", 1, pos >= 1),
                                  ("union_loss_ge2", "loss", 2, pos >= 2),
                                  ("union_gain_ge1", "gain", 1, neg >= 1),
                                  ("union_gain_ge2", "gain", 2, neg >= 2)):
        m = m & valid
        for variant, arr, patch in (("pixel", m, 0), ("mapped", patch_filter(m), MIN_PATCH_PX)):
            rows.append({"variant": variant, "set": name, "class": np.nan,
                         "direction": direction, "min_years": k,
                         "observed_km2": float(arr.sum()) * CELL_KM2, "min_patch_px": patch})
    lo, ga = valid & (pos >= 1), valid & (neg >= 1)
    for variant, arr, patch in (("pixel", lo & ga, 0),
                                ("mapped", patch_filter(lo) & patch_filter(ga), MIN_PATCH_PX)):
        rows.append({"variant": variant, "set": "overlap_loss_and_gain", "class": np.nan,
                     "direction": "both", "min_years": 1,
                     "observed_km2": float(arr.sum()) * CELL_KM2, "min_patch_px": patch})
    table = pd.DataFrame(rows)
    table["evidence_z_threshold"] = T
    table.to_csv(folder / "cumulative_confidence_class_areas_local.csv", index=False)
    print(table[table.variant == "mapped"].to_string(index=False))

    profile.update(count=4, dtype="float32", nodata=NODATA_OUT, compress="lzw")
    out_tif = folder / "cumulative_confidence_classes_local.tif"
    with rasterio.open(out_tif, "w", **profile) as dst:
        for i, (arr, name) in enumerate([(cls_map, "class_mapped"), (cls_px, "class_px"),
                                         (pos, "n_loss_years"), (neg, "n_gain_years")], 1):
            out = arr.astype(np.float32)
            out[~valid] = NODATA_OUT
            dst.write(out, i)
            dst.set_band_description(i, name)
    print(f"    written: {out_tif}")


def main() -> None:
    for scheme in SCHEMES:
        run_scheme(scheme)


if __name__ == "__main__":
    main()
