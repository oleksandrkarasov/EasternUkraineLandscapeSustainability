# -*- coding: utf-8 -*-
"""Step 1: build the two inputs that cannot be shipped with the code.

    python Python_prepare_inputs.py --acled acled_export.csv --viina events_latest.csv
                                    --worldcover tile1.tif tile2.tif ...

1. data/war_events4326.gpkg  from an ACLED export (CSV or XLSX; its data licence
   does not allow redistribution) and the VIINA event file
   (https://github.com/zhukovyuri/VIINA). Both are restricted to the study area
   (data/SUSTD_aoi.zip) and to 2022-2024. Columns: 'source' (ACLED / VIINA), 'date',
   point geometry (EPSG:4326). The same events are also written as the zipped
   shapefile data/upload_to_GEE/SUSTD_war_events.zip for upload to Earth Engine.
2. data/ESA_WorldCover_2021_Ukraine_AOI.tif  from the ESA WorldCover 2021 v200
   tiles covering the study area (https://esa-worldcover.org/en/data-access;
   tiles N45E036, N45E039, N48E036, N48E039), mosaicked and clipped to the AOI.
   Used by support/Python_correlations_with_events.py (dominant land cover per hexagon).

Either part can be run alone by giving only its arguments.
"""
import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd
import geopandas as gpd

_PKG = Path(__file__).resolve().parent
sys.modules.pop("folders", None)
sys.path.insert(0, str(_PKG / "support"))
import folders  # noqa: E402

YEARS = (2022, 2023, 2024)


def read_events(path, source):
    """Read an ACLED or VIINA table and return points with 'source' and 'date'."""
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    lon = next((cols[c] for c in ("longitude", "lon", "lng", "long") if c in cols), None)
    lat = next((cols[c] for c in ("latitude", "lat") if c in cols), None)
    date = next((cols[c] for c in ("event_date", "date", "event_dt", "time") if c in cols), None)
    if lon is None or lat is None or date is None:
        raise SystemExit(f"{path.name}: need longitude/latitude and a date column (found {list(df.columns)[:15]} ...)")
    raw = df[date].astype(str).str.strip()
    parsed = pd.to_datetime(raw, errors="coerce", format="%Y%m%d") if raw.str.fullmatch(r"\d{8}").mean() > 0.9 \
        else pd.to_datetime(raw, errors="coerce")          # VIINA uses YYYYMMDD, ACLED '24 February 2022'
    out = pd.DataFrame({"source": source,
                        "date": parsed.dt.strftime("%Y-%m-%d"),
                        "lon": pd.to_numeric(df[lon], errors="coerce"),
                        "lat": pd.to_numeric(df[lat], errors="coerce")}).dropna()
    out = out[pd.to_datetime(out["date"]).dt.year.isin(YEARS)]
    return gpd.GeoDataFrame(out[["source", "date"]], geometry=gpd.points_from_xy(out.lon, out.lat), crs="EPSG:4326")


def build_war_events(acled, viina, aoi_path, out_path):
    parts = []
    if acled:
        parts.append(read_events(acled, "ACLED"))
    if viina:
        parts.append(read_events(viina, "VIINA"))
    events = pd.concat(parts, ignore_index=True)
    aoi = gpd.read_file(f"zip://{aoi_path}" if str(aoi_path).endswith(".zip") else aoi_path).to_crs("EPSG:4326")
    events = events[events.within(aoi.geometry.union_all() if hasattr(aoi.geometry, "union_all") else aoi.geometry.unary_union)]
    events.to_file(out_path, driver="GPKG")
    print(f"war events: {len(events):,} points ({events['source'].value_counts().to_dict()}) -> {out_path}")
    # zipped shapefile for Earth Engine (asset name = file name without .zip)
    zip_path = folders.WAR_EVENTS_ZIP
    shp_dir = zip_path.parent / "shapefiles"
    shp_dir.mkdir(parents=True, exist_ok=True)
    shp = shp_dir / (zip_path.stem + ".shp")
    events.to_file(shp, driver="ESRI Shapefile", encoding="UTF-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            f = shp.with_suffix(ext)
            if f.exists():
                z.write(f, arcname=f.name)
    print(f"           -> {zip_path} (upload as Earth Engine asset {zip_path.stem})")


def build_worldcover(tiles, aoi_path, out_path):
    import rasterio
    from rasterio.merge import merge
    from rasterio.mask import mask
    srcs = [rasterio.open(t) for t in tiles]
    mosaic, transform = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform, compress="lzw")
    tmp = out_path.with_suffix(".mosaic.tif")
    with rasterio.open(tmp, "w", **meta) as dst:
        dst.write(mosaic)
    aoi = gpd.read_file(f"zip://{aoi_path}" if str(aoi_path).endswith(".zip") else aoi_path).to_crs(srcs[0].crs)
    with rasterio.open(tmp) as src:
        clipped, ctransform = mask(src, aoi.geometry, crop=True, nodata=0)
        meta.update(height=clipped.shape[1], width=clipped.shape[2], transform=ctransform, nodata=0)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(clipped)
    tmp.unlink()
    for s in srcs:
        s.close()
    print(f"WorldCover: {clipped.shape[1]} x {clipped.shape[2]} pixels -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acled", type=Path, help="ACLED export (CSV/XLSX) for Ukraine, 2022-2024")
    ap.add_argument("--viina", type=Path, help="VIINA events file (CSV)")
    ap.add_argument("--worldcover", type=Path, nargs="*", help="ESA WorldCover 2021 tiles (GeoTIFF)")
    args = ap.parse_args()
    aoi = folders.AOI
    if not aoi.exists():
        raise SystemExit(f"{aoi} is missing")
    if args.acled or args.viina:
        build_war_events(args.acled, args.viina, aoi, folders.WAR_EVENTS)
    if args.worldcover:
        build_worldcover(args.worldcover, aoi, folders.WORLDCOVER)
    if not (args.acled or args.viina or args.worldcover):
        ap.print_help()


if __name__ == "__main__":
    main()
