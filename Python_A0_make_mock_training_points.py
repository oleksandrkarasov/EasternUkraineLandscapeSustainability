# -*- coding: utf-8 -*-
"""
Python_A0_make_mock_training_points.py
======================================

Writes SYNTHETIC presences and pseudo-absences in the exact format produced by
Python_A1_prepare_presences_and_backgrounds.py, so that the Earth Engine
pipeline (GEE_02 -> GEE_03 -> GEE_04 -> GEE_05) and the Python post-processing
can be run end to end without the social-media records, which cannot be shared.

The mocked points contain NO information derived from individual social-media
records: no record, user or date is carried over, directly or indirectly. In the
default mode ('synthetic') the locations are drawn from a purely synthetic
propensity surface built from public covariates (land cover, distance to
infrastructure) and a random spatial field. In the optional mode 'density' the
draw is additionally weighted by a coarse, aggregated count surface (H3
resolution 5 hexagons, about 250 km^2, counts below a disclosure threshold
removed) that the authors may publish separately; even then no point is closer
to a real record than the hexagon it falls in.

Results obtained with these points do NOT reproduce the numbers of the
manuscript; they only show that the code runs and what its outputs look like.

Outputs (see support/folders.py)
--------------------------------
Six ZIP shapefiles for Earth Engine (data/mock_points), named as the
assets read by GEE_02 (support/folders.py, TABLES):
  SUSTD_points_regular_presence.zip        SUSTD_points_regular_pseudoabsence.zip
  SUSTD_points_full_presence.zip           SUSTD_points_full_pseudoabsence.zip
  SUSTD_points_recurrent_presence.zip      SUSTD_points_recurrent_pseudoabsence.zip
and, in work/, the GeoPackages read by Python_A2/A3:
  presences_nondvi_matched_<RUN_TAG>_4326.gpkg
  presences_recurrent_core_nondvi_matched_<RUN_TAG>_4326.gpkg
  pa_pool_<RUN_TAG>_3857.gpkg
plus mock_audit_<RUN_TAG>.csv with the counts of every layer.

Inputs
------
--aoi      polygon file of the study area (required; any CRS)
--ordlo    polygon file of the pre-2022 occupation area (optional; sets in_ordlo)
--esa      ESA WorldCover 2021 GeoTIFF (optional but recommended: without it the
           land-cover class is random and GEE_02 will drop the points whose
           real 500 m land cover is outside the six training classes)
--ndvi     summer-median NDVI GeoTIFF (optional; otherwise random values)
--infra    one or more polygon files with 10 m buffers of roads/railways/
           buildings (optional; otherwise distances are random)
--density  CSV with columns h3_5,count (optional; switches to mode 'density')
--n-presences  number of presence cells for 2019-2021 (default 10765, as in the
           manuscript before predictor extraction)
--seed     random seed (default 20250523)

Example
-------
python Python_A0_make_mock_training_points.py            # reads data/aoi.gpkg etc. if present
python Python_A0_make_mock_training_points.py --aoi my_aoi.gpkg --esa WorldCover.tif
"""
import argparse
import shutil
import subprocess
import time
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import sys

# Load support/folders.py from THIS package, never a copy left in memory or on
# sys.path by another package or an earlier notebook cell.
try:
    _PKG = Path(__file__).resolve().parent
except NameError:                       # pasted into a notebook cell
    _PKG = Path.cwd()
if not (_PKG / "support" / "folders.py").exists():
    raise SystemExit(f"support/folders.py not found next to this script ({_PKG}). "
                     "Run the script from the package folder, e.g. cd <package folder>, "
                     "or %run <package folder>/" + Path(__file__).name if '__file__' in dir() else "")
sys.modules.pop("folders", None)
sys.path.insert(0, str(_PKG / "support"))
import folders  # noqa: E402

try:
    import h3  # optional: only used to label H3 cells
except ImportError:  # pragma: no cover
    h3 = None

# ---- constants shared with Python_A1 (do not change) ------------------------
RUN_TAG = "outdoor_sym_buffer_both_effort_weighted_guard0"
PA_SCHEME_TAG = "effort_guard0"
TARGET_ESA_CLASSES = [10, 20, 30, 80, 90, 100]
ESA_CLASS_PROBS = {10: 0.35, 20: 0.05, 30: 0.45, 80: 0.08, 90: 0.05, 100: 0.02}
EFFORT_RADIUS_M = 5000.0
INFRA_BUFFER_M = 10.0          # width of the road/rail/building buffers
MAX_INFRA_DIST_M = 20000.0     # search radius and cap of the distance covariate (as in Python_A1)
METRIC_CRS = "EPSG:3857"
GEE_EXPORT_NAMES = {
    "presences_nondvi_matched": folders.TABLES["presences_regular"],
    "pa_nondvi_matched": folders.TABLES["pa_regular"],
    "presences_nondvi_full": folders.TABLES["presences_full"],
    "pa_nondvi_full": folders.TABLES["pa_full"],
    "presences_recurrent": folders.TABLES["presences_recurrent"],
    "pa_recurrent": folders.TABLES["pa_recurrent"],
}
# Shapefile field names are limited to 10 characters; GEE_02 and Python_A2/A3
# know these aliases.
SHAPEFILE_ALIASES = {
    "dist_infra_m": "dist_inf", "effort_5km": "effort5k", "log_effort": "log_eff",
    "in_matched_window": "match_win", "n_user_days": "n_udays",
    "recurrent_rule": "recur_rule", "match_pair_id": "pair_id",
    "pa_guard_rings": "pa_guard", "time_window": "time_win",
    "n_platforms": "n_platf", "first_year": "first_yr", "last_year": "last_yr",
    "first_date": "first_dt", "last_date": "last_dt", "infra_rule": "infra_rul",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aoi", type=Path, help="AOI polygon (default: data/aoi.gpkg)")
    p.add_argument("--ordlo", type=Path)
    p.add_argument("--esa", type=Path)
    p.add_argument("--ndvi", type=Path)
    p.add_argument("--infra", type=Path, nargs="*", default=[])
    p.add_argument("--density", type=Path)
    p.add_argument("--work-dir", type=Path, default=folders.DATA, help="folder with the input files (default: data/)")
    p.add_argument("--n-presences", type=int, default=10765)
    p.add_argument("--seed", type=int, default=20250523)
    p.add_argument("--no-matching", action="store_true",
                   help="do not run Python_A3/A2 afterwards (the four accessibility-matched tables)")
    return p.parse_args()


# ---- helpers -----------------------------------------------------------------
def sample_raster(raster_path, gdf, block=2048):
    """Sample a single-band raster at point locations (nearest cell), reading
    the raster in square blocks so that large rasters are never loaded whole."""
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(raster_path) as src:
        pts = gdf.to_crs(src.crs)
        xs = pts.geometry.x.values; ys = pts.geometry.y.values
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows = np.asarray(rows); cols = np.asarray(cols)
        values = np.full(len(xs), np.nan)
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        br = rows // block; bc = cols // block
        keys = np.unique(np.stack([br[inside], bc[inside]], axis=1), axis=0)
        for k, (r0, c0) in enumerate(keys):
            sel = inside & (br == r0) & (bc == c0)
            win = Window(int(c0 * block), int(r0 * block),
                         int(min(block, src.width - c0 * block)), int(min(block, src.height - r0 * block)))
            arr = src.read(1, window=win)
            values[sel] = arr[rows[sel] - int(r0 * block), cols[sel] - int(c0 * block)]
            if k % 50 == 0:
                print(f"      raster blocks {k + 1}/{len(keys)}", end="\r")
        nodata = src.nodata
    print()
    if nodata is not None:
        values[values == nodata] = np.nan
    return values


def read_polygons(path, preferred_layer):
    """Read a polygon file; in a multi-layer GeoPackage take the named layer if present."""
    if str(path).endswith(".zip"):
        return gpd.read_file(f"zip://{path}")
    try:
        import pyogrio
        layers = [l[0] for l in pyogrio.list_layers(path)]
    except Exception:
        layers = []
    if preferred_layer in layers:
        return gpd.read_file(path, layer=preferred_layer)
    return gpd.read_file(path)


def random_points_in_polygon(poly, n, rng):
    """Uniform random points inside a (multi)polygon, tested with a prepared geometry."""
    import shapely
    shapely.prepare(poly)
    minx, miny, maxx, maxy = poly.bounds
    xs_out = []; ys_out = []; got = 0
    while got < n:
        k = int((n - got) * 1.6) + 1000
        xs = rng.uniform(minx, maxx, k); ys = rng.uniform(miny, maxy, k)
        keep = shapely.contains_xy(poly, xs, ys)
        xs_out.append(xs[keep]); ys_out.append(ys[keep]); got += int(keep.sum())
    xs = np.concatenate(xs_out)[:n]; ys = np.concatenate(ys_out)[:n]
    return gpd.GeoSeries(shapely.points(xs, ys), crs=METRIC_CRS)


def gaussian_field(xy, centres, scales, weights):
    """Sum of Gaussian bumps: a smooth, random spatial field in [0, 1]."""
    f = np.zeros(len(xy))
    for c, s, w in zip(centres, scales, weights):
        d2 = ((xy[:, 0] - c[0]) ** 2 + (xy[:, 1] - c[1]) ** 2) / (2 * s ** 2)
        f += w * np.exp(-d2)
    f = f - f.min()
    return f / (f.max() + 1e-9)


def weighted_draw(df, n, weight_col, rng):
    """n rows without replacement, probability proportional to the weight
    (Efraimidis-Spirakis keys; rows with zero weight are never drawn)."""
    w = df[weight_col].to_numpy(dtype=float)
    w = np.where(w > 0, w, 0.0)
    n = min(n, int((w > 0).sum()))
    keys = np.full(len(w), -np.inf)
    pos = w > 0
    keys[pos] = np.log(rng.random(pos.sum())) / w[pos]
    return df.iloc[np.argsort(-keys)[:n]]


def h3_cells(gdf_4326, res):
    if h3 is None:
        return None
    if hasattr(h3, "latlng_to_cell"):
        return [h3.latlng_to_cell(g.y, g.x, res) for g in gdf_4326.geometry]
    return [h3.geo_to_h3(g.y, g.x, res) for g in gdf_4326.geometry]


def write_gee_zip(gdf_4326, upload_dir, basename):
    asset_name = folders.ASSET_PREFIX + basename
    shp_dir = upload_dir / "shapefiles"
    shp_dir.mkdir(parents=True, exist_ok=True)
    gee = gdf_4326.rename(columns={k: v for k, v in SHAPEFILE_ALIASES.items() if k in gdf_4326.columns})
    for col in gee.columns:
        if gee[col].dtype == object and col != "geometry":
            gee[col] = gee[col].astype(str)
    shp_path = shp_dir / f"{asset_name}.shp"
    gee.to_file(shp_path, driver="ESRI Shapefile", encoding="UTF-8")
    zip_path = upload_dir / f"{asset_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            f = shp_path.with_suffix(ext)
            if f.exists():
                archive.write(f, arcname=f.name)
    return zip_path


# ---- main ------------------------------------------------------------------------
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    def lap(msg):
        print(f"   [{time.time() - t0:6.0f} s] {msg}", flush=True)
    data_dir = args.work_dir.expanduser().resolve()   # inputs
    work_dir = folders.WORK                           # working files read by A2/A3
    work_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = folders.MOCK_POINTS                  # shipped with the code
    upload_dir.mkdir(parents=True, exist_ok=True)
    # inputs default to the standard file names in data/ when not given
    defaults = {"aoi": ("aoi.gpkg", "SUSTD_aoi.zip"), "ordlo": ("ordlo.gpkg", "SUSTD_ordlo.zip"),
                "esa": ("ESA_WorldCover_2021_Ukraine_AOI.tif",), "ndvi": ("Summer_NDVI_Median_2019_2021.tif",)}
    for key, names in defaults.items():
        for name in names:
            if getattr(args, key) is None and (data_dir / name).exists():
                setattr(args, key, data_dir / name)
    if args.aoi is None:
        raise SystemExit("--aoi is required (no data/aoi.gpkg found)")
    if not args.infra:
        args.infra = [data_dir / n for n in ("railways_buffer.gpkg", "roads_buffer_3857.gpkg",
                      "buildings_overture_buffer.gpkg", "buildings_east_10m_buffer.gpkg") if (data_dir / n).exists()]
    audit = []

    aoi = read_polygons(args.aoi, "aoi").to_crs(METRIC_CRS)
    aoi_poly = aoi.geometry.union_all() if hasattr(aoi.geometry, "union_all") else aoi.geometry.unary_union
    ordlo_poly = None
    if args.ordlo and args.ordlo.exists():
        o = read_polygons(args.ordlo, "ordlo").to_crs(METRIC_CRS)
        ordlo_poly = o.geometry.union_all() if hasattr(o.geometry, "union_all") else o.geometry.unary_union

    # 1. Candidate locations: many more than needed, inside the AOI -------------
    n_pres = args.n_presences
    n_full = int(n_pres * 1.8)            # 2015-2021 layer is about 1.8x the 2019-2021 layer
    n_cand = max(8 * n_full, 150_000)
    print(f"Drawing {n_cand:,} candidate locations inside the AOI ...", flush=True)
    cand = gpd.GeoDataFrame(geometry=random_points_in_polygon(aoi_poly, n_cand, rng), crs=METRIC_CRS)
    lap(f"{len(cand):,} candidates drawn")
    xy = np.column_stack([cand.geometry.x.values, cand.geometry.y.values])

    # 2. Covariates: real where rasters are given, synthetic otherwise -----------
    cand_4326 = cand.to_crs("EPSG:4326")
    if args.esa and args.esa.exists():
        print("Sampling ESA WorldCover ...", flush=True)
        cand["esa"] = sample_raster(args.esa, cand_4326)
        cand = cand[cand["esa"].isin(TARGET_ESA_CLASSES)].copy()
        audit.append(("candidates_in_six_classes", len(cand)))
    else:
        classes = np.array(list(ESA_CLASS_PROBS))
        probs = np.array(list(ESA_CLASS_PROBS.values()))
        cand["esa"] = rng.choice(classes, size=len(cand), p=probs / probs.sum())
        print("   WARNING: no ESA raster given; land-cover classes are random. GEE_02 will\n"
              "   drop points whose real 500 m land cover is outside the six classes.")
    xy = np.column_stack([cand.geometry.x.values, cand.geometry.y.values])
    cand_4326 = cand.to_crs("EPSG:4326")

    if args.ndvi and args.ndvi.exists():
        print("Sampling NDVI ...", flush=True)
        cand["ndvi"] = sample_raster(args.ndvi, cand_4326)
        cand["ndvi"] = cand["ndvi"].fillna(cand["ndvi"].median())
    else:
        base = {10: 0.65, 20: 0.5, 30: 0.45, 80: 0.05, 90: 0.4, 100: 0.3}
        cand["ndvi"] = np.clip(cand["esa"].map(base).values + rng.normal(0, 0.1, len(cand)), -0.2, 0.95)

    if args.infra:
        # same rule as Python_A1: distance to the nearest 10 m buffer (bounded search),
        # plus the buffer width; points inside a buffer are dropped
        print("Reading infrastructure buffers ...", flush=True)
        infra = pd.concat([gpd.read_file(f).to_crs(METRIC_CRS)[["geometry"]] for f in args.infra if Path(f).exists()])
        joined = gpd.sjoin_nearest(cand[["geometry"]], infra, how="left", distance_col="_d", max_distance=MAX_INFRA_DIST_M)
        joined = joined[~joined.index.duplicated(keep="first")]
        d = joined["_d"].reindex(cand.index).fillna(MAX_INFRA_DIST_M)
        cand["dist_infra_m"] = (d + INFRA_BUFFER_M).clip(upper=MAX_INFRA_DIST_M).values
        cand = cand[d.values > 0].copy()
    else:
        cand["dist_infra_m"] = np.maximum(10.01, rng.lognormal(mean=np.log(400), sigma=1.0, size=len(cand)))
    xy = np.column_stack([cand.geometry.x.values, cand.geometry.y.values])

    lap("covariates ready")

    # 3. Synthetic social-media activity field (the 'effort' covariate) ----------
    minx, miny, maxx, maxy = aoi_poly.bounds
    k = 40
    centres = np.column_stack([rng.uniform(minx, maxx, k), rng.uniform(miny, maxy, k)])
    scales = rng.uniform(15_000, 60_000, k)
    weights = rng.gamma(2.0, 1.0, k)
    activity = gaussian_field(xy, centres, scales, weights)
    if args.density and args.density.exists():
        # mode 'density': weight by a coarse aggregated surface (H3 res 5 counts)
        if h3 is None:
            raise SystemExit("mode 'density' needs the h3 package")
        dens = pd.read_csv(args.density)
        dens = dict(zip(dens.iloc[:, 0].astype(str), dens.iloc[:, 1].astype(float)))
        cells5 = h3_cells(cand_4326.loc[cand.index], 5)
        d = np.array([dens.get(c, 0.0) for c in cells5])
        d = np.log1p(d) / (np.log1p(d).max() + 1e-9)
        activity = 0.5 * activity + 0.5 * d
        mode = "density"
    else:
        mode = "synthetic"
    # expected number of social-media records of any kind within 5 km
    cand["effort_5km"] = rng.poisson(lam=np.exp(np.log(5) + 5.5 * activity) - 1).astype(int)
    cand["log_effort"] = np.log1p(cand["effort_5km"])

    # 4. Synthetic visitation propensity and the presence draw --------------------
    lc_pref = {10: 1.0, 20: 0.2, 30: -0.3, 80: 1.2, 90: 0.3, 100: 0.0}
    noise = gaussian_field(xy, np.column_stack([rng.uniform(minx, maxx, 60), rng.uniform(miny, maxy, 60)]),
                           rng.uniform(5_000, 30_000, 60), rng.gamma(1.5, 1.0, 60))
    logit = (cand["esa"].map(lc_pref).values
             - 0.9 * np.log(cand["dist_infra_m"].values / 100.0)
             + 1.5 * cand["log_effort"].values / cand["log_effort"].max()
             + 2.0 * noise + rng.normal(0, 0.7, len(cand)))
    prob = 1 / (1 + np.exp(-(logit - np.median(logit))))
    cand["_w"] = prob

    lap("propensity surface ready")
    # one record per H3 resolution-8 cell (or per 500 m cell if h3 is missing)
    cells8 = h3_cells(cand_4326.loc[cand.index], 8)
    if cells8 is None:
        cells8 = (np.floor(xy[:, 0] / 500).astype(int).astype(str) + "_" + np.floor(xy[:, 1] / 500).astype(int).astype(str))
    cand["h3_index"] = cells8
    cand = cand.drop_duplicates("h3_index")

    if len(cand) < n_full:
        print(f"   WARNING: only {len(cand):,} candidate cells; fewer presences than requested")
    draw_full = weighted_draw(cand, n_full, "_w", rng)
    pres_full = draw_full.copy()
    # exactly n_pres cells fall in 2019-2021 (the main layer); the rest in 2015-2018
    years = np.concatenate([rng.integers(2019, 2022, min(n_pres, len(pres_full))),
                            rng.integers(2015, 2019, max(0, len(pres_full) - n_pres))])
    rng.shuffle(years)
    pres_full["year"] = years
    pres_matched = pres_full[pres_full["year"] >= 2019].copy()
    audit += [("presences_full_2015_2021", len(pres_full)), ("presences_matched_2019_2021", len(pres_matched))]

    def presence_attributes(df, window):
        df = df.copy()
        df["presence"] = 1
        df["variant"] = "nondvi"
        df["infra_rule"] = "buffer_both"
        df["layer"] = rng.choice(["flickr", "twitter", "vk"], size=len(df), p=[0.15, 0.25, 0.60])
        df["time_window"] = window
        df["in_matched_window"] = (df["year"] >= 2019).astype(int)
        df["n_records"] = rng.geometric(0.35, len(df))
        df["n_users"] = np.minimum(df["n_records"], rng.geometric(0.6, len(df)))
        df["n_platforms"] = np.minimum(df["n_users"], rng.integers(1, 3, len(df)))
        df["first_year"] = df["year"]
        df["last_year"] = np.minimum(2021, df["year"] + rng.integers(0, 3, len(df)))
        return df

    pres_matched = presence_attributes(pres_matched, "matched")
    pres_full = presence_attributes(pres_full, "full")

    # recurrent subset: >=2 users, >=3 user-days, span >= 30 days (synthetic counts)
    rec = pres_matched.sample(frac=0.27, random_state=args.seed).copy()
    rec["n_users"] = 2 + rng.geometric(0.5, len(rec))
    rec["n_user_days"] = 3 + rng.geometric(0.3, len(rec))
    rec["n_dates"] = np.minimum(rec["n_user_days"], 2 + rng.geometric(0.4, len(rec)))
    rec["n_months"] = np.minimum(5, 1 + rng.geometric(0.5, len(rec)))
    rec["n_years"] = np.minimum(3, 1 + rng.geometric(0.6, len(rec)))
    rec["span_days"] = 30 + rng.integers(0, 900, len(rec))
    rec["first_date"] = "2019-05-01"
    rec["last_date"] = "2021-09-30"
    rec["recurrent_rule"] = "u2_ud3_d2_span30"
    rec["variant"] = "recurrent_core_nondvi"
    audit.append(("presences_recurrent", len(rec)))

    lap(f"{len(cand):,} unique cells; presences drawn")

    # 5. Pseudo-absences: same rules, effort-weighted draw (target-group background)
    pool = cand[~cand["h3_index"].isin(pres_full["h3_index"])].copy()
    pool["presence"] = 0
    pool["variant"] = "nondvi"
    pool["infra_rule"] = "buffer_both"
    pool["pa_mode"] = "effort_weighted"
    pool["pa_guard_rings"] = 0
    pool["_pa_w"] = np.log1p(pool["effort_5km"]) + 0.05

    def draw_pa(n, window, variant="nondvi", require_effort=False):
        src = pool[pool["effort_5km"] > 0] if require_effort else pool
        out = weighted_draw(src, n, "_pa_w", rng).copy()
        out["time_window"] = window
        out["variant"] = variant
        return out

    pa_matched = draw_pa(len(pres_matched), "matched")
    pa_full = draw_pa(len(pres_full), "full")
    pa_rec = draw_pa(len(rec), "matched", "recurrent_core_nondvi", require_effort=True)
    audit += [("pa_matched", len(pa_matched)), ("pa_full", len(pa_full)), ("pa_recurrent", len(pa_rec))]

    lap("pseudo-absences drawn")

    # 6. in_ordlo and export -------------------------------------------------------
    def finish(df, cols):
        df = gpd.GeoDataFrame(df, geometry="geometry", crs=METRIC_CRS)
        df["in_ordlo"] = df.geometry.within(ordlo_poly).astype(int) if ordlo_poly is not None else 0
        cols = [c for c in cols if c in df.columns] + ["geometry"]
        return df[cols]

    pres_cols = ["presence", "variant", "infra_rule", "layer", "year", "time_window", "in_matched_window",
                 "esa", "ndvi", "dist_infra_m", "effort_5km", "log_effort", "n_records", "n_users",
                 "n_platforms", "first_year", "last_year", "in_ordlo", "h3_index"]
    rec_cols = pres_cols[:12] + ["n_records", "n_user_days", "n_users", "n_dates", "n_months", "n_years",
                                 "span_days", "first_date", "last_date", "n_platforms", "first_year",
                                 "last_year", "recurrent_rule", "in_ordlo", "h3_index"]
    pa_cols = ["presence", "variant", "time_window", "infra_rule", "pa_mode", "pa_guard_rings", "h3_index",
               "esa", "ndvi", "dist_infra_m", "effort_5km", "log_effort", "in_ordlo"]

    layers = {
        "presences_nondvi_matched": finish(pres_matched, pres_cols),
        "pa_nondvi_matched": finish(pa_matched, pa_cols),
        "presences_nondvi_full": finish(pres_full, pres_cols),
        "pa_nondvi_full": finish(pa_full, pa_cols),
        "presences_recurrent": finish(rec, rec_cols),
        "pa_recurrent": finish(pa_rec, pa_cols),
    }
    for key, gdf in layers.items():
        zip_path = write_gee_zip(gdf.to_crs("EPSG:4326"), upload_dir, GEE_EXPORT_NAMES[key])
        print(f"   {len(gdf):>7} rows -> {zip_path.name}")

    # GeoPackages read by Python_A2 / Python_A3
    layers["presences_nondvi_matched"].to_crs("EPSG:4326").to_file(
        work_dir / f"presences_nondvi_matched_{RUN_TAG}_4326.gpkg", driver="GPKG")
    layers["presences_recurrent"].to_crs("EPSG:4326").to_file(
        work_dir / f"presences_recurrent_core_nondvi_matched_{RUN_TAG}_4326.gpkg", driver="GPKG")
    finish(pool, pa_cols).to_file(work_dir / f"pa_pool_{RUN_TAG}_3857.gpkg", driver="GPKG")

    audit.append(("mode", mode))
    pd.DataFrame(audit, columns=["item", "value"]).to_csv(work_dir / f"mock_audit_{RUN_TAG}.csv", index=False)
    with open(work_dir / f"mock_manifest_{RUN_TAG}.json", "w") as f:
        json.dump({"mode": mode, "seed": args.seed, "n_presences_requested": n_pres,
                   "inputs": {k: str(v) for k, v in vars(args).items() if v},
                   "note": "Synthetic points; results do not reproduce the manuscript."}, f, indent=2)
    # 7. Accessibility-matched tables: Python_A3 (regular) and Python_A2 (recurrent)
    #    read the GeoPackages written above and write two ZIP tables each.
    if not args.no_matching:
        for script, key in (("Python_A3_match_nonrecurrent_accessibility.py", "A3"),
                            ("Python_A2_match_recurrent_accessibility.py", "A2")):
            print(f"\nRunning {script} ...", flush=True)
            subprocess.run([sys.executable, str(_PKG / script), "--work-dir", str(work_dir)], check=True)
            for z in sorted(folders.UPLOAD[key].glob(folders.ASSET_PREFIX + "points_*.zip")):
                shutil.copy2(z, upload_dir / z.name)
                print(f"   copied {z.name}")
    n_zip = len(list(upload_dir.glob(folders.ASSET_PREFIX + "points_*.zip")))
    print(f"\nDone: {n_zip} ZIP tables in {upload_dir} (mode: {mode}). Upload each as an Earth Engine\n"
          f"table asset named like the file (without .zip), then run GEE_02.")


if __name__ == "__main__":
    main()
