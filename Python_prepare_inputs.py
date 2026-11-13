# -*- coding: utf-8 -*-
"""Step 1: build the two inputs that cannot be shipped with the code.

    python Python_prepare_inputs.py --acled acled_export.csv --viina events_latest.csv
                                    --worldcover tile1.tif tile2.tif ...

1. data/war_events4326.gpkg  from an ACLED export for Ukraine (CSV or XLSX;
   https://acleddata.com/monitor/ukraine-conflict-monitor; its data licence does
   not allow redistribution) and the VIINA event table (CSV;
   https://github.com/zhukovyuri/VIINA). Each input is one table with longitude,
   latitude and date columns. Both are restricted to the study area
   (data/upload_to_GEE/SUSTD_aoi.zip) and to the period from 24 February 2022 to
   the end of 2024. Columns written: 'source'
   (ACLED / VIINA), 'date' (YYYY-MM-DD), point geometry (EPSG:4326). The same
   events are also written as the zipped shapefile
   data/upload_to_GEE/SUSTD_war_events.zip for upload to Earth Engine.
2. data/ESA_WorldCover_2021_Ukraine_AOI.tif  from the ESA WorldCover 2021 v200
   tiles covering the study area (https://esa-worldcover.org/en/data-access;
   tiles N45E036, N45E039, N48E033, N48E036, N48E039), mosaicked and clipped to
   the AOI. The script stops if the tiles do not cover the whole study area.
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

START_DATE = "2022-02-24"   # first day of the full-scale invasion
END_DATE = "2024-12-31"


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
    out = out[(out["date"] >= START_DATE) & (out["date"] <= END_DATE)]   # ISO dates compare as text
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


def missing_worldcover_tiles(aoi_geom, tile_bounds):
    """Names of the 3-degree WorldCover tiles (south-west corner) that the study
    area needs and the given files do not cover."""
    import math
    from shapely.geometry import box
    from shapely.ops import unary_union
    rest = aoi_geom.difference(unary_union([box(*b) for b in tile_bounds]))
    if rest.is_empty or rest.area < 1e-6:          # square degrees
        return []
    minx, miny, maxx, maxy = rest.bounds
    names = []
    for lat in range(int(math.floor(miny / 3)) * 3, int(math.ceil(maxy / 3)) * 3, 3):
        for lon in range(int(math.floor(minx / 3)) * 3, int(math.ceil(maxx / 3)) * 3, 3):
            if rest.intersection(box(lon, lat, lon + 3, lat + 3)).area >= 1e-6:
                names.append("{0}{1:02d}{2}{3:03d}".format(
                    "N" if lat >= 0 else "S", abs(lat), "E" if lon >= 0 else "W", abs(lon)))
    return names


def build_worldcover(tiles, aoi_path, out_path):
    import rasterio
    from rasterio.merge import merge
    from rasterio.mask import mask
    srcs = [rasterio.open(t) for t in tiles]
    aoi_ll = gpd.read_file(f"zip://{aoi_path}" if str(aoi_path).endswith(".zip") else aoi_path).to_crs(srcs[0].crs)
    aoi_geom = aoi_ll.geometry.union_all() if hasattr(aoi_ll.geometry, "union_all") else aoi_ll.geometry.unary_union
    missing = missing_worldcover_tiles(aoi_geom, [tuple(s.bounds) for s in srcs])
    if missing:
        raise SystemExit("The WorldCover tiles do not cover the whole study area. Missing: " + ", ".join(missing))
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
    ap.add_argument("--acled", type=Path, help="ACLED export (CSV/XLSX) for Ukraine, 24 February 2022 to 2024")
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
