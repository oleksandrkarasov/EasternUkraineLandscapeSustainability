#!/usr/bin/env python3
"""
Areas of loss and gain anomalies inside and outside the pre-2022 occupation line
===============================================================================

Reads GeoTIFFs that GEE_05 has already exported; no Earth Engine run is needed.

    python optional_checks/Python_areas_by_occupation_line.py \\
        --main    <Debiased_RF_Main_revision_annualbase_annual_median*.tif> \\
        --annual  <Debiased_RF_2022_AnnualInputSensitivity_revision_annualbase_annual_median*.tif> \\
        --threshold 2.5

Use --threshold 3.0 with the GeoTIFFs of the recurrent-visitation model.

Bands used
    main GeoTIFF    band 7   z-score of the median comparison (2019-2021 vs 2022-2024)
                    band 26  1 inside the pre-2022 occupation line
    annual GeoTIFF  band 3   z-score of one wartime year against the 2019-2021 median

A pixel is a loss anomaly where z >= threshold and a gain anomaly where
z <= -threshold (no neighbourhood or patch rule), as in Table S6. The row
'all' must reproduce Table S6; if it does not, the files or the threshold are
not the ones of the paper.
"""
import argparse
import glob

import numpy as np
import rasterio
from rasterio.merge import merge

Z_MEDIAN_BAND = 7
ORDLO_BAND = 26
Z_ANNUAL_BAND = 3


def read(paths, bands):
    files = sorted(f for p in paths for f in glob.glob(p))
    if not files:
        raise SystemExit("No file found for: {0}".format(" ".join(paths)))
    with rasterio.open(files[0]) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0
    arr, transform = merge(files, indexes=bands, nodata=nodata)
    return arr.astype("float64"), transform, nodata


def areas(z, ordlo, valid, threshold, pixel_km2):
    rows = []
    for name, zone in (("inside the line", ordlo & valid),
                       ("outside the line", ~ordlo & valid),
                       ("all", valid)):
        rows.append((name, zone.sum() * pixel_km2,
                     ((z >= threshold) & zone).sum() * pixel_km2,
                     ((z <= -threshold) & zone).sum() * pixel_km2))
    return rows


def report(title, rows):
    print("\n" + title)
    print("  {0:<18}{1:>14}{2:>12}{3:>8}{4:>12}{5:>8}".format(
        "zone", "analysed km2", "loss km2", "%", "gain km2", "%"))
    for name, total, loss, gain in rows:
        print("  {0:<18}{1:>14,.0f}{2:>12,.0f}{3:>8.2f}{4:>12,.0f}{5:>8.2f}".format(
            name, total, loss, 100 * loss / total, gain, 100 * gain / total))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", nargs="+", required=True, help="main GeoTIFF of GEE_05 (one file or its tiles)")
    ap.add_argument("--annual", nargs="+", help="annual GeoTIFF of GEE_05 for one wartime year (optional)")
    ap.add_argument("--threshold", type=float, default=2.5, help="2.5 for regular, 3.0 for recurrent visitation")
    a = ap.parse_args()

    arr, transform, nodata = read(a.main, [Z_MEDIAN_BAND, ORDLO_BAND])
    z, ordlo = arr[0], arr[1] == 1
    valid = np.isfinite(z) & (z != nodata)
    pixel_km2 = abs(transform.a * transform.e) / 1e6
    print("Pixel area: {0:.4f} km2; threshold |z| >= {1}".format(pixel_km2, a.threshold))
    report("Median comparison (2019-2021 vs 2022-2024)", areas(z, ordlo, valid, a.threshold, pixel_km2))

    if a.annual:
        arr_y, transform_y, nodata_y = read(a.annual, [Z_ANNUAL_BAND])
        zy = arr_y[0]
        if zy.shape != z.shape or transform_y != transform:
            raise SystemExit("The annual GeoTIFF is not on the grid of the main GeoTIFF.")
        valid_y = np.isfinite(zy) & (zy != nodata_y)
        report("Annual comparison ({0})".format(a.annual[0]), areas(zy, ordlo, valid_y, a.threshold, pixel_km2))


if __name__ == "__main__":
    main()
