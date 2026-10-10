#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8
"""
=============================================================================
 VISITATION AND RECURRENT-VISITATION TRAINING DATA -- PIPELINE V13
 Manuscript SUST-D-26-00527 (Sustainability Science)
=============================================================================

 BUNDLE ROLE
 -----------
 This script rebuilds every workflow on one common, high-confidence outdoor
 WorldCover domain and additionally creates a stricter, predeclared
 recurrent-outdoor-visitation analysis. Earlier AUC values from the broader
 domain must not be carried over; all outdoor-only workflows require new CV.
 The recurrent label is
 based on independent platform-user-days, users, dates and temporal span; it is
 never selected by maximising model performance. Run
 Python_A2_match_recurrent_accessibility.py and
 Python_A3_match_nonrecurrent_accessibility.py afterwards to construct
 the recurrent and non-recurrent accessibility-matched backgrounds.

 RECURRENT PRESENCE DEFINITION
 -----------------------------
 A recurrent presence is an H3 cell that, during May-September 2019-2021:
   * contains records in core green/blue WorldCover classes
     10, 20, 30, 80, 90 or 100;
   * contains at least 3 independent platform-user-days;
   * contains at least 2 independent platform-user identifiers;
   * was observed on at least 2 dates; and
   * spans at least 30 days between first and last observation.

 Multiple posts from the same platform user, date and H3 cell contribute one
 user-day. Cells with outdoor records that do not reach the recurrent rule are
 ambiguous and are not used as pseudo-absences. One seeded random independent
 event is retained as the cell's representative point; NDVI is never used to
 choose it.

 WHAT CHANGED IN THIS VERSION AND WHY
 ------------------------------------
 Every change below is tied to a specific referee comment. Reviewer numbering
 follows the decision letter (R3 = the detailed methodological review,
 R2 = the page-by-page review in Review_comments.pdf).

 NOTE ON THE EFFORT COVARIATE WINDOW
     effort_5km counts raw records from the matched 2019-2021 window for both
     classes, because that is the observation window of the main analysis. It
     is therefore only approximately matched for the full 2015-2021 sensitivity
     layer, where some presences predate the effort window. State this if the
     effort covariate is used in the full-window run.

 [R2, Page 13 line 6 / Page 20] Control zones and event datasets.
     Handled in GEE_05 (debiasing), not here.

 NOTE ON OUTPUT NAMES
 --------------------
 Input paths and filenames are unchanged so the script runs on your machine
 as-is. Output filenames now carry RUN_TAG so that the symmetric variants do
 not overwrite each other. GeoPackages are archival outputs in work_dir. The
 six zipped Shapefiles to upload are written to
 outputs/A1_upload_to_GEE and named like their Earth Engine
 assets (ASSET_PREFIX + name, support/folders.py); upload each under its own
 name (ASSET_HINTS at the bottom of this file).
=============================================================================
"""

import os
import sys
import json
import hashlib
import secrets
import re
import zipfile

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree

# =============================================================================
# CONFIGURATION
# =============================================================================

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # notebook cell
    _HERE = os.getcwd()
sys.path.insert(0, os.path.join(_HERE, "support"))
for _m in ("folders", "pipeline_checks"):
    sys.modules.pop(_m, None)
import folders  # noqa: E402

data_dir = str(folders.DATA)            # inputs
work_dir = str(folders.WORK)            # working files (per dataset)
os.makedirs(work_dir, exist_ok=True)
UPLOAD_DIR = str(folders.UPLOAD["A1"])  # six ZIP tables for Earth Engine
ASSET_PREFIX = folders.ASSET_PREFIX

SEED = 20250523
rng = np.random.default_rng(SEED)

# sampling-effort radius, the 50 m pseudo-absence spacing and the
# distance-to-infrastructure covariate) previously ran in EPSG:3857, where
# distances at 47-50 degrees N are inflated by roughly 1.45-1.55x, so
# "effort_5km" was really a ~3.3-3.5 km ground radius and dist_infra_m was
# overstated by ~50%. GEE_05 was already recomputed on EPSG:3035 for exactly
# this reason; the point-preparation pipeline now uses the same equal-area
# metric CRS. Intermediate filenames keep their historical "_3857" suffix so
# that A2/A3 file discovery is unchanged; the CRS stored in each file is the
# authority. Rerunning A1-A3 is required for the corrected attributes.
METRIC_CRS = "EPSG:3035"

# ---- Reviewer-driven switches ------------------------------------------------

#   "buffer_both"          : drop points inside the 10 m buffer union (relaxes
#                            the pseudo-absence rule to match the presences)
#   "cell_neighbours_both" : drop whole H3 cells intersecting the buffers plus
#                            their direct neighbours (tightens the presence
#                            rule to match the pseudo-absences)
#   "legacy_asymmetric"    : original behaviour, kept only for comparison
INFRA_RULE = "buffer_both"

#   "random"          : uniform draw from eligible cells (original)
#   "effort_weighted" : draw weighted by log(1 + number of raw 2019-2021
#                       records within 5 km of the cell centre) + 0.05
#                       (target-group style background), so that the negative
#                       class shares the observation process of the positive
#                       class
PA_SAMPLING_MODE = "effort_weighted"

# pool. 0 = observed cell only (recommended main analysis); 1 = observed cell
# plus direct H3 neighbours (sensitivity only).
PA_GUARD_RINGS = 0

# embedding / NDVI window so the matched subset can be selected later.
PRESENCE_START_YEAR = 2015
PRESENCE_END_YEAR = 2021
MATCHED_WINDOW_START = 2019
MATCHED_WINDOW_END = 2021

HASH_USER_IDS = True

# ---- Fixed analytical parameters --------------------------------------------

SUMMER_MONTHS = [5, 6, 7, 8, 9]
MIN_NDVI = 0.5
# One symmetric, high-confidence outdoor green/blue domain for both training
# classes. Cropland (40), built-up (50) and bare/sparse land (60) are excluded
# from presences and pseudo-absences. Open water (80) is included on both sides.
# Downstream GEE maps every valid embedding pixel without a WorldCover mask.
# All classes outside this six-class training domain are declared transfer
# domains; cropland is the predeclared focal transfer analysis for abandonment.
TARGET_ESA_CLASSES = [10, 20, 30, 80, 90, 100]
RECURRENT_ESA_CLASSES = TARGET_ESA_CLASSES.copy()
RECURRENT_MIN_USER_DAYS = 3
RECURRENT_MIN_USERS = 2
RECURRENT_MIN_DATES = 2
RECURRENT_MIN_SPAN_DAYS = 30
RECURRENT_RULE_TAG = "u2_ud3_d2_span30"
INFRA_BUFFER_M = 10.0        # nominal buffer width used to build the buffer layers
EFFORT_RADIUS_M = 5000.0     # radius for the sampling-effort covariate
MAX_INFRA_DIST_M = 20000.0   # cap for the distance-to-infrastructure covariate
PA_MIN_DIST_M = 50.0         # minimum spacing between pseudo-absence points
PA_POOL_N = 60000            # candidate pool, large enough to survive NDVI filtering
PA_FINAL_N = 20000           # exported pseudo-absences per variant, "fixed" mode

# threshold-dependent metric reported in Table 1 (TSS, sensitivity, specificity,
# and the Brier skill score) depends on it. Restricting presences to the
# matched 2019-2021 window can cut their number sharply, so a fixed 20,000
# pseudo-absences would silently move the prevalence between the main run and
# the sensitivity run and make the two tables non-comparable.
#   "fixed"           : always PA_FINAL_N points (submitted behaviour)
#   "match_presences" : as many pseudo-absences as the matching presence layer
#                       contains, holding prevalence at 0.5 across all runs
PA_TARGET_N_MODE = "match_presences"

RUN_TAG = "outdoor_sym_{0}_{1}_guard{2}".format(
    INFRA_RULE, PA_SAMPLING_MODE, PA_GUARD_RINGS)
PA_SCHEME_TAG = ("effort" if PA_SAMPLING_MODE == "effort_weighted" else "random") + \
                "_guard{0}".format(PA_GUARD_RINGS)

# ---- Input files (unchanged) -------------------------------------------------

flic_file = os.path.join(data_dir, "flic.shp")
twitt_file = os.path.join(data_dir, "twitt.shp")
vk_file = os.path.join(data_dir, "vk_east.shp")

ordlo_gpkg = os.path.join(data_dir, "ordlo.gpkg")
input_h3 = os.path.join(data_dir, "h3_8.gpkg")
esa_raster = os.path.join(data_dir, "ESA_WorldCover_2021_Ukraine_AOI.tif")
ndvi_raster = os.path.join(data_dir, "Summer_NDVI_Median_2019_2021.tif")
roads_raw = os.path.join(data_dir, "roads_east3857.gpkg")

exclusion_files = [
    os.path.join(data_dir, "railways_buffer.gpkg"),
    os.path.join(data_dir, "roads_buffer_3857.gpkg"),
    os.path.join(data_dir, "buildings_overture_buffer.gpkg"),
    os.path.join(data_dir, "buildings_east_10m_buffer.gpkg"),
]

# ---- Intermediate and output files ------------------------------------------

merged_4326 = os.path.join(work_dir, "social_merged_4326.gpkg")
merged_3857 = os.path.join(work_dir, "social_merged_3857.gpkg")
merged_cov_3857 = os.path.join(work_dir, "social_merged_cov_3857.gpkg")
roads_main = os.path.join(work_dir, "roads_main_3857.gpkg")

out_pres_ndvi = os.path.join(
    work_dir, "presences_ndvi050_matched_{0}_4326.gpkg".format(RUN_TAG))
out_pres_nondvi = os.path.join(
    work_dir, "presences_nondvi_matched_{0}_4326.gpkg".format(RUN_TAG))
out_pres_ndvi_full = os.path.join(
    work_dir, "presences_ndvi050_full2015_2021_{0}_4326.gpkg".format(RUN_TAG))
out_pres_nondvi_full = os.path.join(
    work_dir, "presences_nondvi_full2015_2021_{0}_4326.gpkg".format(RUN_TAG))
out_pres_recurrent = os.path.join(
    work_dir, "presences_recurrent_core_nondvi_matched_{0}_4326.gpkg".format(
        RUN_TAG))
out_recurrent_cell_evidence = os.path.join(
    work_dir, "recurrent_cell_evidence_{0}.csv".format(RUN_TAG))
out_recurrent_threshold_audit = os.path.join(
    work_dir, "recurrent_threshold_audit_{0}.csv".format(RUN_TAG))
out_pa_pool = os.path.join(work_dir, "pa_pool_{0}_3857.gpkg".format(RUN_TAG))
out_pa = {
    ("ndvi050", "matched"): os.path.join(
        work_dir, "pa_ndvi050_matched_{0}_4326.gpkg".format(RUN_TAG)),
    ("nondvi", "matched"): os.path.join(
        work_dir, "pa_nondvi_matched_{0}_4326.gpkg".format(RUN_TAG)),
    ("ndvi050", "full"): os.path.join(
        work_dir, "pa_ndvi050_full_{0}_4326.gpkg".format(RUN_TAG)),
    ("nondvi", "full"): os.path.join(
        work_dir, "pa_nondvi_full_{0}_4326.gpkg".format(RUN_TAG)),
}
out_pa_recurrent = os.path.join(
    work_dir, "pa_recurrent_core_nondvi_matched_{0}_4326.gpkg".format(RUN_TAG))

# Earth Engine table upload accepts a ZIP containing the Shapefile sidecars.
# These short, stable names match the asset paths used by GEE_02. The
# GeoPackages above remain the lossless archival outputs; the ZIP files are
# upload copies with deliberately short (<=10-character) field names.
GEE_EXPORT_NAMES = {
    "presences_ndvi050_matched": "presences_ndvi050_matched_outdoor_sym",
    "presences_nondvi_matched": folders.TABLES["presences_regular"],
    "presences_ndvi050_full": "presences_ndvi050_full_outdoor_sym",
    "presences_nondvi_full": folders.TABLES["presences_full"],
    "presences_recurrent": folders.TABLES["presences_recurrent"],
    # pseudo-absence layer cannot hold prevalence at 0.5 for both the matched
    # and the full presence window, because the two presence layers differ in
    # size by about half.
    "pa_ndvi050_matched": "pa_ndvi050_matched_outdoor_sym_{0}".format(PA_SCHEME_TAG),
    "pa_nondvi_matched": folders.TABLES["pa_regular"],
    "pa_ndvi050_full": "pa_ndvi050_full_outdoor_sym_{0}".format(PA_SCHEME_TAG),
    "pa_nondvi_full": folders.TABLES["pa_full"],
    "pa_recurrent": folders.TABLES["pa_recurrent"],
}

# The layers uploaded to Earth Engine (GEE_02); the ndvi050 variants are
# archival GeoPackages only.
UPLOADED_LAYERS = {GEE_EXPORT_NAMES[k] for k in (
    "presences_nondvi_matched", "presences_nondvi_full", "presences_recurrent",
    "pa_nondvi_matched", "pa_nondvi_full", "pa_recurrent")}

# ESRI Shapefile/DBF field names are limited to 10 characters.  Explicit names
# avoid the driver-dependent silent truncation that otherwise occurs when a
# GeoPackage is manually converted in QGIS.
GEE_FIELD_NAMES = {
    "presence": "presence",
    "variant": "variant",
    "infra_rule": "infra_rule",
    "layer": "layer",
    "year": "year",
    "time_window": "time_win",
    "in_matched_window": "match_win",
    "esa": "esa",
    "ndvi": "ndvi",
    "dist_infra_m": "dist_inf",
    "effort_5km": "effort5k",
    "log_effort": "log_eff",
    "n_records": "n_records",
    "n_users": "n_users",
    "n_user_days": "n_udays",
    "n_dates": "n_dates",
    "n_months": "n_months",
    "n_years": "n_years",
    "span_days": "span_days",
    "first_date": "date_first",
    "last_date": "date_last",
    "recurrent_rule": "recur_rule",
    "n_platforms": "n_plat",
    "first_year": "yr_first",
    "last_year": "yr_last",
    "in_ordlo": "in_ordlo",
    "pa_mode": "pa_mode",
    "pa_guard_rings": "pa_guard",
}

audit_csv = os.path.join(work_dir, "sampling_audit_{0}.csv".format(RUN_TAG))
salt_file = os.path.join(work_dir, "DO_NOT_PUBLISH_user_id_salt.txt")

# Which steps to run. Steps are sequential; set to False to skip a finished one.
RUN_STEPS = {
    "merge": True,
    "ordlo_stats": True,
    "sample_covariates": True,
    "presences": True,
    "pseudo_absences": True,
}

# =============================================================================
# =============================================================================

AUDIT = []


def audit(step, layer, n_in, n_out, note=""):
    """Record one filtering step so the whole data flow can be reported."""
    AUDIT.append({
        "step": step,
        "layer": layer,
        "n_in": int(n_in),
        "n_out": int(n_out),
        "n_removed": int(n_in) - int(n_out),
        "note": note,
    })
    print("   [audit] {0} | {1}: {2} -> {3} (removed {4}) {5}".format(
        step, layer, n_in, n_out, int(n_in) - int(n_out), note))


def write_audit():
    df = pd.DataFrame(AUDIT)
    df.to_csv(audit_csv, index=False)
    print("\nAudit log written to {0}".format(audit_csv))
    return df


# =============================================================================
# HELPERS
# =============================================================================

def get_salt():
    """"""
    if os.path.exists(salt_file):
        with open(salt_file, "r") as fh:
            return fh.read().strip()
    salt = secrets.token_hex(16)
    with open(salt_file, "w") as fh:
        fh.write(salt)
    print("   New user-id salt written to {0}. Keep this file out of the "
          "repository.".format(salt_file))
    return salt


def hash_series(series, salt):
    """Salted SHA-256 digest, preserving missing identifiers as missing.

    Treating every missing identifier as the literal string ``nan`` would turn
    unrelated anonymous records into one apparent repeat visitor. That is
    harmless for the non-recurrent one-record occupancy analysis but invalid for
    the recurrent-use rule, so missing users must remain missing.
    """
    text = series.astype("string").str.strip()
    valid = text.notna() & ~text.str.lower().isin(["", "nan", "none", "null"])
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out.loc[valid] = text.loc[valid].map(
        lambda value: hashlib.sha256(
            (salt + value).encode("utf-8")).hexdigest()[:16]
    )
    return out.astype(object).where(out.notna(), None)


def sample_raster_values(gdf, raster_path, column_name, expect_epsg=None):
    """Sample a single-band raster at point locations, reprojecting if needed."""
    print("   Sampling {0} from {1}...".format(column_name, os.path.basename(raster_path)))
    with rasterio.open(raster_path) as src:
        src_epsg = src.crs.to_epsg()
        pts = gdf if (src_epsg is None or gdf.crs.to_epsg() == src_epsg) else gdf.to_crs(src_epsg)
        coords = [(x, y) for x, y in zip(pts.geometry.x, pts.geometry.y)]
        values = [v[0] for v in src.sample(coords)]
        nodata = src.nodata
    arr = np.array(values, dtype="float64")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    gdf[column_name] = arr
    return gdf


def load_exclusions(target_crs):
    """Union of the railway, road and building buffer layers."""
    geoms = []
    for fpath in exclusion_files:
        if not os.path.exists(fpath):
            print("   Warning: missing exclusion layer {0}".format(os.path.basename(fpath)))
            continue
        gdf_ex = gpd.read_file(fpath)
        if gdf_ex.crs != target_crs:
            gdf_ex = gdf_ex.to_crs(target_crs)
        geoms.append(gdf_ex[["geometry"]])
    if not geoms:
        raise RuntimeError("No exclusion layers could be loaded.")
    out = gpd.GeoDataFrame(pd.concat(geoms, ignore_index=True),
                           geometry="geometry", crs=target_crs)
    print("   Exclusion features loaded: {0}".format(len(out)))
    return out


def add_distance_to_infrastructure(gdf_points, gdf_infra):
    """"""
    print("   Computing distance to nearest infrastructure...")
    joined = gpd.sjoin_nearest(
        gdf_points[["geometry"]].copy(),
        gdf_infra[["geometry"]].copy(),
        how="left",
        distance_col="_d",
        max_distance=MAX_INFRA_DIST_M,
    )
    # sjoin_nearest can duplicate rows on ties; keep the first match per point.
    joined = joined[~joined.index.duplicated(keep="first")]
    d = joined["_d"].reindex(gdf_points.index)
    d = d.fillna(MAX_INFRA_DIST_M) + INFRA_BUFFER_M
    gdf_points["dist_infra_m"] = d.clip(upper=MAX_INFRA_DIST_M).values
    return gdf_points


def apply_infra_rule(gdf_points, gdf_infra, gdf_h3, label):
    """

    One exception, and it is deliberate: "legacy_asymmetric" exists only to
    reproduce the submitted design for comparison, and the submitted design was
    asymmetric by definition. It therefore applies the 10 m buffer to presences
    and the cell-plus-neighbours rule to pseudo-absences. Treating both classes
    the same under that label would silently produce a third design that is
    neither the old one nor the new one.
    """
    n_in = len(gdf_points)
    rule = INFRA_RULE
    if INFRA_RULE == "legacy_asymmetric":
        rule = ("buffer_both" if label == "presences" else "cell_neighbours_both")

    if rule == "buffer_both":
        # Drop points that fall inside the buffer union.
        joined = gpd.sjoin(gdf_points, gdf_infra, how="left", predicate="intersects")
        joined = joined[~joined.index.duplicated(keep="first")]
        keep = joined["index_right"].isna()
        out = gdf_points.loc[keep.index[keep]].copy()
        note = ("10 m buffer union"
                if INFRA_RULE != "legacy_asymmetric"
                else "10 m buffer union (legacy presence rule)")

    elif rule == "cell_neighbours_both":
        # Drop whole H3 cells intersecting the buffers, plus their neighbours.
        bad = gpd.sjoin(gdf_h3, gdf_infra, how="inner", predicate="intersects")
        bad_idx = bad.index.unique()
        gdf_bad = gdf_h3.loc[bad_idx]
        nbr = gpd.sjoin(gdf_h3, gdf_bad, how="inner", predicate="touches")
        drop_idx = set(bad_idx).union(set(nbr.index.unique()))
        gdf_safe_cells = gdf_h3.drop(index=list(drop_idx))[["geometry"]]
        joined = gpd.sjoin(gdf_points, gdf_safe_cells, how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
        out = gdf_points.loc[joined["index_right"].notna().values].copy()
        note = ("H3 cell plus direct neighbours"
                if INFRA_RULE != "legacy_asymmetric"
                else "H3 cell plus neighbours (legacy pseudo-absence rule)")

    else:
        raise ValueError("Unknown INFRA_RULE: {0}".format(INFRA_RULE))

    if "index_right" in out.columns:
        out = out.drop(columns=["index_right"])
    audit("infrastructure_exclusion", label, n_in, len(out), note)
    return out


def compute_effort(gdf_points, raw_coords_tree):
    """"""
    print("   Computing sampling effort within {0:.0f} m...".format(EFFORT_RADIUS_M))
    coords = np.column_stack([gdf_points.geometry.x.values, gdf_points.geometry.y.values])
    counts = raw_coords_tree.query_ball_point(coords, r=EFFORT_RADIUS_M, return_length=True)
    gdf_points["effort_5km"] = np.asarray(counts, dtype="int64")
    gdf_points["log_effort"] = np.log1p(gdf_points["effort_5km"].astype("float64"))
    return gdf_points


def h3_id_column(gdf_h3):
    """Return a stable H3 identifier column, creating one if necessary."""
    for col in ["h3_index", "h3_id", "index", "id"]:
        if col in gdf_h3.columns:
            return col
    gdf_h3["h3_cell_id"] = gdf_h3.index.astype(str)
    return "h3_cell_id"


def random_point_in_polygon(geom, generator, max_tries=1000):
    """Generate a reproducible uniform point in a polygon.

    This replaces GeoPandas.sample_points so the seed is honoured on old as
    well as new GeoPandas versions. Multipart cells are handled by rejection
    sampling over their common bounding box.
    """
    from shapely.geometry import Point

    minx, miny, maxx, maxy = geom.bounds
    for _ in range(max_tries):
        p = Point(generator.uniform(minx, maxx), generator.uniform(miny, maxy))
        if geom.covers(p):
            return p
    return geom.representative_point()


def presence_guard_cells(gdf_h3, h3_col, visited_ids, rings):
    """H3 ids excluded from the pseudo-absence pool.

    The main analysis uses rings=0. rings=1 is deliberately only a sensitivity
    variant because a broad geographic moat around presences can make spatial
    discrimination artificially easy.
    """
    visited_ids = set(pd.Series(list(visited_ids)).dropna().astype(str))
    ids_as_str = gdf_h3[h3_col].astype(str)
    visited = gdf_h3.loc[ids_as_str.isin(visited_ids)]
    excluded = set(visited[h3_col].astype(str))
    if rings == 0 or visited.empty:
        return excluded
    frontier = visited[["geometry"]].copy()
    for _ in range(rings):
        hits = gpd.sjoin(gdf_h3[[h3_col, "geometry"]], frontier,
                         how="inner", predicate="touches")
        new_ids = set(hits[h3_col].astype(str))
        excluded.update(new_ids)
        frontier = gdf_h3.loc[gdf_h3[h3_col].astype(str).isin(new_ids),
                               ["geometry"]]
    return excluded


def tag_ordlo(gdf_points):
    """"""
    if not os.path.exists(ordlo_gpkg):
        print("   Warning: ordlo.gpkg not found, in_ordlo not set.")
        gdf_points["in_ordlo"] = 0
        return gdf_points
    gdf_ordlo = gpd.read_file(ordlo_gpkg)
    if gdf_ordlo.crs != gdf_points.crs:
        gdf_ordlo = gdf_ordlo.to_crs(gdf_points.crs)
    joined = gpd.sjoin(gdf_points, gdf_ordlo[["geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    gdf_points["in_ordlo"] = joined["index_right"].notna().astype(int).values
    return gdf_points


def gee_safe_field_map(columns, geometry_name):
    """Return a deterministic, collision-free DBF field-name mapping."""
    mapping = {}
    used = set()
    for column in columns:
        if column == geometry_name:
            continue
        preferred = GEE_FIELD_NAMES.get(column, column)
        preferred = re.sub(r"[^A-Za-z0-9_]", "_", str(preferred))
        base = (preferred or "field")[:10]
        candidate = base
        suffix = 1
        while candidate.lower() in used:
            suffix_text = str(suffix)
            candidate = base[:10 - len(suffix_text)] + suffix_text
            suffix += 1
        used.add(candidate.lower())
        mapping[column] = candidate
    return mapping


def write_gee_shapefile_zip(gdf_4326, gee_basename):
    """Write a GEE-ready zipped Shapefile (only for the layers that are
    uploaded) into UPLOAD_DIR, named like its asset, and a field-name
    crosswalk into work_dir."""
    if gee_basename not in UPLOADED_LAYERS:
        print("   (archival GeoPackage only; not uploaded to Earth Engine)")
        return
    asset_name = ASSET_PREFIX + gee_basename
    shp_dir = os.path.join(work_dir, "gee_shapefiles")
    os.makedirs(shp_dir, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    shp_path = os.path.join(shp_dir, asset_name + ".shp")
    zip_path = os.path.join(UPLOAD_DIR, asset_name + ".zip")
    field_csv = os.path.join(work_dir, gee_basename + "_fields.csv")

    field_map = gee_safe_field_map(gdf_4326.columns, gdf_4326.geometry.name)
    gee_gdf = gdf_4326.rename(columns=field_map)
    gee_gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="UTF-8")

    pd.DataFrame([
        {"source_field": source, "gee_field": target}
        for source, target in field_map.items()
    ]).to_csv(field_csv, index=False)

    sidecars = [".shp", ".shx", ".dbf", ".prj", ".cpg"]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for extension in sidecars:
            path = os.path.splitext(shp_path)[0] + extension
            if os.path.exists(path):
                archive.write(path, arcname=os.path.basename(path))
    print("   GEE upload ZIP: {0}".format(os.path.basename(zip_path)))
    print("   Field crosswalk: {0}".format(os.path.basename(field_csv)))


def export_pair(gdf, path_4326, gee_basename):
    """Write an archival GeoPackage and a GEE-ready zipped Shapefile."""
    out = gdf.to_crs(epsg=4326) if gdf.crs.to_epsg() != 4326 else gdf
    out.to_file(path_4326, driver="GPKG")
    print("   Written: {0} ({1} features)".format(os.path.basename(path_4326), len(out)))
    write_gee_shapefile_zip(out, gee_basename)


# =============================================================================
# STEP 1. MERGE AND STANDARDISE THE THREE PLATFORMS
# =============================================================================

def step_merge():
    print("\n=== STEP 1. Merging and standardising social media datasets ===")
    salt = get_salt() if HASH_USER_IDS else None

    def read_std(path, layer_name, encoding=None, rename_map=None):
        gdf = gpd.read_file(path, encoding=encoding) if encoding else gpd.read_file(path)
        if "fid" in gdf.columns:
            gdf = gdf.drop(columns=["fid"])
        if rename_map:
            gdf = gdf.rename(columns=rename_map)
        gdf["layer"] = layer_name
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs(epsg=4326)
        return gdf

    print("1. Flickr...")
    gdf_flic = read_std(flic_file, "flic")

    print("2. Twitter...")
    gdf_twitt = read_std(twitt_file, "twitt")

    print("3. VKontakte...")
    vk_rename = {"photo_id": "id", "owner_id": "user_id",
                 "lat": "latitude", "long": "longitude", "date": "datetaken_raw"}
    try:
        gdf_vk = read_std(vk_file, "vk", encoding="cp1251", rename_map=vk_rename)
    except Exception as exc:
        print("   cp1251 failed ({0}), retrying with latin1".format(exc))
        gdf_vk = read_std(vk_file, "vk", encoding="latin1", rename_map=vk_rename)
    gdf_vk["datetaken"] = pd.to_datetime(
        gdf_vk["datetaken_raw"], errors="coerce").dt.strftime("%Y-%m-%d")

    target_columns = ["id", "user_id", "datetaken", "latitude", "longitude", "layer", "geometry"]
    merged = pd.concat([gdf_flic[target_columns],
                        gdf_twitt[target_columns],
                        gdf_vk[target_columns]], ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")

    # within the study, so per-user statistics remain possible, but the raw
    # platform identifier is not retained in any exported product.
    if HASH_USER_IDS:
        merged["user_hash"] = hash_series(merged["user_id"], salt)
        merged = merged.drop(columns=["user_id"])
        merged = merged.rename(columns={"user_hash": "user_id"})

    merged = merged.reset_index(drop=True)
    merged.index.name = "fid"
    audit("merge", "all_platforms", len(merged), len(merged), "raw georeferenced records")

    merged.to_file(merged_4326, driver="GPKG")
    merged.to_crs(METRIC_CRS).to_file(merged_3857, driver="GPKG")
    print("Done.")


# =============================================================================
# STEP 2. ORDLO SUMMARY FOR THE SUPPLEMENTARY MATERIAL (unchanged logic)
# =============================================================================

def step_ordlo_stats():
    """
    Unchanged in substance. Note for the rebuttal (R2, Page 10): the decline of
    VK on Ukraine-controlled territory starts with the 2017 ban, not with the
    full-scale invasion. That is a text change; the figure produced here already
    shows the 2017 break, so point to it explicitly in the caption.
    """
    print("\n=== STEP 2. ORDLO comparison for the supplement ===")
    layer_config = {"flic": "Flickr", "twitt": "Twitter", "vk": "VK"}

    gdf_social = gpd.read_file(merged_4326)
    gdf_ordlo = gpd.read_file(ordlo_gpkg)
    gdf_social["date"] = pd.to_datetime(gdf_social["datetaken"], errors="coerce")
    gdf_social["year"] = gdf_social["date"].dt.year

    if gdf_social.crs != gdf_ordlo.crs:
        gdf_ordlo = gdf_ordlo.to_crs(gdf_social.crs)

    joined = gpd.sjoin(gdf_social, gdf_ordlo[["geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    joined["ordlo"] = joined["index_right"].notna().map({True: "yes", False: "no"})

    for layer_code, display_name in layer_config.items():
        df_layer = joined[joined["layer"] == layer_code].copy()
        if len(df_layer) == 0:
            continue
        stats = df_layer.groupby(["year", "ordlo"]).agg(
            total_rows=("user_id", "size"),
            unique_users=("user_id", "nunique"),
        ).reset_index().dropna(subset=["year"])
        stats["year"] = stats["year"].astype(int)
        stats = stats.sort_values("year")

        # Also write the numbers, so the supplement can cite them.
        stats.to_csv(os.path.join(work_dir, "ordlo_stats_{0}.csv".format(layer_code)),
                     index=False)

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        sns.barplot(data=stats, x="year", y="total_rows", hue="ordlo",
                    palette={"yes": "#d62728", "no": "#1f77b4"}, ax=axes[0])
        axes[0].set_title("Total {0} content per year".format(display_name), fontsize=14)
        axes[0].set_ylabel("Number of posts")
        axes[0].set_xlabel("")
        axes[0].legend(title="Inside ORDLO")
        sns.barplot(data=stats, x="year", y="unique_users", hue="ordlo",
                    palette={"yes": "#d62728", "no": "#1f77b4"}, ax=axes[1])
        axes[1].set_title("Unique {0} users per year".format(display_name), fontsize=14)
        axes[1].set_ylabel("Number of unique users")
        axes[1].set_xlabel("Year")
        axes[1].legend(title="Inside ORDLO")
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, "ordlo_stats_{0}.pdf".format(layer_code)),
                    format="pdf", bbox_inches="tight")
        plt.close(fig)
    print("Done.")


# =============================================================================
# STEP 3. COVARIATES ON ALL RAW RECORDS
# =============================================================================

def step_sample_covariates():
    """"""
    print("\n=== STEP 3. Sampling covariates on all raw records ===")
    gdf = gpd.read_file(merged_4326)
    print("   Raw records: {0}".format(len(gdf)))

    gdf = sample_raster_values(gdf, esa_raster, "esa")
    gdf = sample_raster_values(gdf, ndvi_raster, "ndvi")

    gdf["date"] = pd.to_datetime(gdf["datetaken"], errors="coerce")
    gdf["year"] = gdf["date"].dt.year
    gdf["month"] = gdf["date"].dt.month
    gdf = gdf.drop(columns=["date"])

    gdf.to_crs(METRIC_CRS).to_file(merged_cov_3857, driver="GPKG")
    audit("covariate_sampling", "all_platforms", len(gdf), len(gdf),
          "ESA and NDVI attached to every raw record")
    print("Done.")


# =============================================================================
# STEP 4. PRESENCES (both NDVI variants, symmetric infrastructure rule)
# =============================================================================

def step_presences():
    print("\n=== STEP 4. Building presences ===")
    gdf = gpd.read_file(merged_cov_3857)
    gdf_h3 = gpd.read_file(input_h3)
    if gdf_h3.crs != gdf.crs:
        gdf_h3 = gdf_h3.to_crs(gdf.crs)
    gdf_infra = load_exclusions(gdf.crs)

    n_raw = len(gdf)
    audit("start", "presences", n_raw, n_raw, "raw records with covariates")

    # ---- 4a. Land cover filter (applied to both classes) --------------------
    mask_esa = gdf["esa"].isin(TARGET_ESA_CLASSES)
    gdf = gdf.loc[mask_esa].copy()
    audit("esa_filter", "presences", n_raw, len(gdf),
          "common outdoor classes {0}".format(TARGET_ESA_CLASSES))

    # ---- 4b. Temporal filter -----------------------------------------------
    n_before = len(gdf)
    mask_season = gdf["month"].isin(SUMMER_MONTHS)
    mask_year = (gdf["year"] >= PRESENCE_START_YEAR) & (gdf["year"] <= PRESENCE_END_YEAR)
    gdf = gdf.loc[mask_season & mask_year].copy()
    audit("temporal_filter", "presences", n_before, len(gdf),
          "May-Sep, {0}-{1}".format(PRESENCE_START_YEAR, PRESENCE_END_YEAR))

    # The flag is retained for auditing, but the main layer is constructed by
    # filtering on it BEFORE thinning below.
    gdf["in_matched_window"] = (
        (gdf["year"] >= MATCHED_WINDOW_START) &
        (gdf["year"] <= MATCHED_WINDOW_END)
    ).astype(int)
    audit("matched_window_check", "presences", len(gdf),
          int(gdf["in_matched_window"].sum()),
          "records in {0}-{1}, before H3 thinning".format(
              MATCHED_WINDOW_START, MATCHED_WINDOW_END))

    # ---- 4c. Infrastructure exclusion (symmetric) ---------------------------
    gdf = apply_infra_rule(gdf, gdf_infra, gdf_h3, "presences")
    gdf = add_distance_to_infrastructure(gdf, gdf_infra)

    # ---- 4d. Sampling effort ------------------------------------------------
    raw_all = gpd.read_file(merged_cov_3857)
    # Target-group effort is matched to the main 2019-2021 observation window.
    raw_effort = raw_all.loc[
        (raw_all["year"] >= MATCHED_WINDOW_START) &
        (raw_all["year"] <= MATCHED_WINDOW_END)
    ].copy()
    raw_coords = np.column_stack(
        [raw_effort.geometry.x.values, raw_effort.geometry.y.values])
    tree = cKDTree(raw_coords)
    gdf = compute_effort(gdf, tree)

    # ---- 4e. H3 assignment, cell statistics and thinning --------------------
    print("   Assigning records to H3 cells...")
    h3_col = h3_id_column(gdf_h3)
    h3_subset = gdf_h3[[h3_col, "geometry"]].copy()

    joined = gpd.sjoin(gdf, h3_subset, how="inner", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined.drop(columns=["index_right"])
    audit("h3_assignment", "presences", len(gdf), len(joined), "records inside an H3 cell")

    def thin(df, tag):
        """
        Keep one seeded random representative per H3 cell. The representative
        point is not chosen on NDVI in either variant; selecting the greenest
        point would add an outcome-dependent spatial displacement even after
        symmetric filtering.
        """
        out = df.sample(frac=1.0, random_state=SEED).drop_duplicates(
            subset=[h3_col], keep="first").copy()
        note = "random record per cell (seeded)"
        audit("h3_thinning", "presences_{0}".format(tag), len(df), len(out),
              "{0}; {1} records discarded".format(note, len(df) - len(out)))
        return out

    def build_variant(source, tag, matched, apply_ndvi):
        df = source.copy()
        window_tag = "matched" if matched else "full2015_2021"
        if matched:
            n_before = len(df)
            df = df.loc[df["in_matched_window"] == 1].copy()
            audit("temporal_match", "presences_{0}_{1}".format(tag, window_tag),
                  n_before, len(df), "filter applied before H3 thinning")
        if apply_ndvi:
            n_before = len(df)
            df = df.loc[df["ndvi"] >= MIN_NDVI].copy()
            audit("ndvi_filter", "presences_{0}_{1}".format(tag, window_tag),
                  n_before, len(df), "NDVI >= {0}; symmetric".format(MIN_NDVI))

        cell_stats = df.groupby(h3_col).agg(
            n_records=("id", "size"),
            n_users=("user_id", "nunique"),
            n_platforms=("layer", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        ).reset_index()
        out = thin(df, "{0}_{1}".format(tag, window_tag))
        out = out.merge(cell_stats, on=h3_col, how="left",
                        suffixes=("", "_cell"))
        out["time_window"] = window_tag
        return out

    def build_recurrent_variant(source):
        """Build the predeclared recurrent green/blue visitation layer.

        The response unit is still an occupied H3 cell, preserving the spatial
        support of the non-recurrent analysis. Evidence within that cell is
        counted as independent platform-user-days rather than raw posts.
        """
        df = source.loc[source["in_matched_window"] == 1].copy()
        n_before = len(df)
        df = df.loc[df["esa"].isin(RECURRENT_ESA_CLASSES)].copy()
        audit("recurrent_core_esa", "presences_recurrent", n_before, len(df),
              "core green/blue classes {0}".format(RECURRENT_ESA_CLASSES))

        df["visit_date"] = pd.to_datetime(
            df["datetaken"], errors="coerce").dt.normalize()
        user_text = df["user_id"].astype("string").str.strip()
        valid_user = user_text.notna() & ~user_text.str.lower().isin(
            ["", "nan", "none", "null"])
        valid_date = df["visit_date"].notna()
        n_before = len(df)
        df = df.loc[valid_user & valid_date].copy()
        audit("recurrent_identity_date", "presences_recurrent", n_before,
              len(df), "valid pseudonymous user and observation date required")
        if df.empty:
            raise RuntimeError(
                "No records remain for recurrent visitation after user/date "
                "validation. Check the user_id and datetaken fields.")

        df["platform_user"] = (
            df["layer"].astype(str) + ":" + df["user_id"].astype(str))
        df["visit_month"] = df["visit_date"].dt.to_period("M").astype(str)
        df["visit_year"] = df["visit_date"].dt.year.astype(int)
        # Source identifiers can mix numeric and string dtypes across
        # platforms. A string key keeps duplicate selection deterministic
        # without asking pandas to order incomparable Python objects.
        df["_event_key"] = (
            df["layer"].astype(str) + ":" + df["id"].astype(str))

        raw_counts = df.groupby(h3_col).size().rename("n_records").reset_index()
        events = (df.sort_values(
            [h3_col, "platform_user", "visit_date", "_event_key"],
            kind="mergesort")
            .drop_duplicates(
                subset=[h3_col, "platform_user", "visit_date"],
                keep="first")
            .copy())
        audit("user_day_collapse", "presences_recurrent", len(df), len(events),
              "one event per platform-user, date and H3 cell")

        stats = events.groupby(h3_col).agg(
            n_user_days=("id", "size"),
            n_users=("platform_user", "nunique"),
            n_dates=("visit_date", "nunique"),
            n_months=("visit_month", "nunique"),
            n_years=("visit_year", "nunique"),
            n_platforms=("layer", "nunique"),
            first_date=("visit_date", "min"),
            last_date=("visit_date", "max"),
            first_year=("visit_year", "min"),
            last_year=("visit_year", "max"),
        ).reset_index().merge(raw_counts, on=h3_col, how="left")
        stats["span_days"] = (
            stats["last_date"] - stats["first_date"]).dt.days.astype(int)
        stats["recurrent"] = (
            (stats["n_user_days"] >= RECURRENT_MIN_USER_DAYS) &
            (stats["n_users"] >= RECURRENT_MIN_USERS) &
            (stats["n_dates"] >= RECURRENT_MIN_DATES) &
            (stats["span_days"] >= RECURRENT_MIN_SPAN_DAYS)
        )
        stats["cell_status"] = np.where(
            stats["recurrent"], "recurrent_presence", "ambiguous_observed")
        stats["recurrent_rule"] = RECURRENT_RULE_TAG

        recurrent_stats = stats.loc[stats["recurrent"]].copy()
        audit("recurrent_cell_rule", "presences_recurrent", len(stats),
              len(recurrent_stats),
              ">={0} user-days, >={1} users, >={2} dates, span >={3} days"
              .format(RECURRENT_MIN_USER_DAYS, RECURRENT_MIN_USERS,
                      RECURRENT_MIN_DATES, RECURRENT_MIN_SPAN_DAYS))
        if recurrent_stats.empty:
            raise RuntimeError(
                "The recurrent rule retained no H3 cells. Inspect "
                "recurrent_cell_evidence before changing any threshold.")

        # Transparent threshold ladder. These counts are descriptive and must
        # not be selected according to downstream AUC.
        threshold_rules = [
            ("any_core_observation", 1, 1, 1, 0, 1),
            ("two_users_two_userdays_span7", 2, 2, 2, 7, 1),
            ("recommended_" + RECURRENT_RULE_TAG, 2, 3, 2, 30, 1),
            ("strict_u3_ud5_d2_span30", 3, 5, 2, 30, 1),
            ("recommended_plus_two_years", 2, 3, 2, 30, 2),
        ]
        threshold_rows = []
        for name, min_users, min_udays, min_dates, min_span, min_years in threshold_rules:
            keep = (
                (stats["n_users"] >= min_users) &
                (stats["n_user_days"] >= min_udays) &
                (stats["n_dates"] >= min_dates) &
                (stats["span_days"] >= min_span) &
                (stats["n_years"] >= min_years)
            )
            threshold_rows.append({
                "rule": name,
                "rule_expression": (
                    "n_users>={0};n_user_days>={1};n_dates>={2};"
                    "span_days>={3};n_years>={4}".format(
                        min_users, min_udays, min_dates, min_span, min_years)),
                "min_users": min_users,
                "min_user_days": min_udays,
                "min_dates": min_dates,
                "min_span_days": min_span,
                "min_years": min_years,
                "n_cells": int(keep.sum()),
            })
        # One-criterion-at-a-time diagnostics identify what is actually
        # binding without changing the declared response or consulting AUC.
        base_user_days = stats["n_user_days"] >= RECURRENT_MIN_USER_DAYS
        base_users = stats["n_users"] >= RECURRENT_MIN_USERS
        base_dates = stats["n_dates"] >= RECURRENT_MIN_DATES
        base_span = stats["span_days"] >= RECURRENT_MIN_SPAN_DAYS
        binding_rules = [
            (
                "diagnostic_relax_users_only",
                base_user_days & base_dates & base_span,
                "user minimum removed; all other recommended criteria retained",
            ),
            (
                "diagnostic_relax_user_days_only",
                base_users & base_dates & base_span,
                "user-day minimum removed; all other criteria retained",
            ),
            (
                "diagnostic_relax_dates_only",
                base_user_days & base_users & base_span,
                "date minimum removed; all other criteria retained",
            ),
            (
                "diagnostic_relax_span_only",
                base_user_days & base_users & base_dates,
                "span minimum removed; all other criteria retained",
            ),
            (
                "diagnostic_users2_or_single_user_dates3",
                base_user_days & base_dates & base_span & (
                    base_users | (
                        (stats["n_users"] == 1) & (stats["n_dates"] >= 3))),
                "recommended rule except users>=2 OR one user on >=3 dates",
            ),
        ]
        for name, keep, expression in binding_rules:
            threshold_rows.append({
                "rule": name,
                "rule_expression": expression,
                "min_users": np.nan,
                "min_user_days": np.nan,
                "min_dates": np.nan,
                "min_span_days": np.nan,
                "min_years": 1,
                "n_cells": int(keep.sum()),
            })
        pd.DataFrame(threshold_rows).to_csv(
            out_recurrent_threshold_audit, index=False)

        evidence = stats.copy()
        evidence["first_date"] = evidence["first_date"].dt.strftime("%Y-%m-%d")
        evidence["last_date"] = evidence["last_date"].dt.strftime("%Y-%m-%d")
        evidence.to_csv(out_recurrent_cell_evidence, index=False)

        recurrent_ids = set(recurrent_stats[h3_col].astype(str))
        recurrent_events = events.loc[
            events[h3_col].astype(str).isin(recurrent_ids)].copy()
        audit(
            "recurrent_supporting_events", "presences_recurrent",
            len(recurrent_events), len(recurrent_events),
            "{0} independent platform-user-days support {1} recurrent H3 cells"
            .format(len(recurrent_events), len(recurrent_stats)))
        # Convert the cell-level response to exactly one model feature per
        # qualifying spatial unit. This does not remove recurrent cells: it
        # prevents the 7+ events in a typical popular cell from becoming
        # pseudoreplicated training rows with nearly identical predictors.
        out = recurrent_events.sample(
            frac=1.0, random_state=SEED).drop_duplicates(
                subset=[h3_col], keep="first").copy()
        audit(
            "recurrent_cell_representation", "presences_recurrent",
            len(recurrent_stats), len(out),
            "one seeded observed representative per recurrent H3 cell; "
            "no qualifying cell removed")
        out = out.merge(recurrent_stats, on=h3_col, how="left",
                        suffixes=("", "_cell"))
        out["time_window"] = "matched"
        out["recurrent_rule"] = RECURRENT_RULE_TAG
        out["first_date"] = pd.to_datetime(
            out["first_date"]).dt.strftime("%Y-%m-%d")
        out["last_date"] = pd.to_datetime(
            out["last_date"]).dt.strftime("%Y-%m-%d")
        return out

    variants = [
        (build_variant(joined, "ndvi050", True, True),
         out_pres_ndvi, "ndvi050", "matched",
         GEE_EXPORT_NAMES["presences_ndvi050_matched"]),
        (build_variant(joined, "nondvi", True, False),
         out_pres_nondvi, "nondvi", "matched",
         GEE_EXPORT_NAMES["presences_nondvi_matched"]),
        (build_variant(joined, "ndvi050", False, True),
         out_pres_ndvi_full, "ndvi050", "full2015_2021",
         GEE_EXPORT_NAMES["presences_ndvi050_full"]),
        (build_variant(joined, "nondvi", False, False),
         out_pres_nondvi_full, "nondvi", "full2015_2021",
         GEE_EXPORT_NAMES["presences_nondvi_full"]),
    ]

    recurrent = build_recurrent_variant(joined)

    for out_gdf, path, tag, window_tag, gee_basename in variants:
        out_gdf = gpd.GeoDataFrame(out_gdf, geometry="geometry", crs=gdf.crs)
        out_gdf = tag_ordlo(out_gdf)
        out_gdf["presence"] = 1
        out_gdf["variant"] = tag
        out_gdf["time_window"] = window_tag
        out_gdf["infra_rule"] = INFRA_RULE
        keep_cols = ["presence", "variant", "infra_rule", "layer", "year",
                     "time_window", "in_matched_window", "esa", "ndvi", "dist_infra_m",
                     "effort_5km", "log_effort", "n_records", "n_users",
                     "n_platforms", "first_year", "last_year", "in_ordlo",
                     h3_col, "geometry"]
        keep_cols = [c for c in keep_cols if c in out_gdf.columns]
        export_pair(out_gdf[keep_cols], path, gee_basename)

    recurrent = gpd.GeoDataFrame(
        recurrent, geometry="geometry", crs=gdf.crs)
    recurrent = tag_ordlo(recurrent)
    recurrent["presence"] = 1
    recurrent["variant"] = "recurrent_core_nondvi"
    recurrent["time_window"] = "matched"
    recurrent["infra_rule"] = INFRA_RULE
    recurrent_keep = [
        "presence", "variant", "infra_rule", "layer", "year",
        "time_window", "in_matched_window", "esa", "ndvi",
        "dist_infra_m", "effort_5km", "log_effort", "n_records",
        "n_user_days", "n_users", "n_dates", "n_months", "n_years",
        "span_days", "first_date", "last_date", "n_platforms",
        "first_year", "last_year", "recurrent_rule", "in_ordlo", h3_col,
        "geometry"
    ]
    recurrent_keep = [c for c in recurrent_keep if c in recurrent.columns]
    export_pair(recurrent[recurrent_keep], out_pres_recurrent,
                GEE_EXPORT_NAMES["presences_recurrent"])

    print("Done.")


# =============================================================================
# STEP 5. PSEUDO-ABSENCES (same rules as presences)
# =============================================================================

def step_pseudo_absences():
    print("\n=== STEP 5. Building pseudo-absences ===")
    # If h3_8.gpkg were stored in EPSG:4326, EFFORT_RADIUS_M, PA_MIN_DIST_M and
    # dist_infra_m in this step would silently be computed in degrees.
    gdf_h3 = gpd.read_file(input_h3).to_crs(METRIC_CRS)
    target_crs = gdf_h3.crs
    print("   Total H3 cells: {0}".format(len(gdf_h3)))

    gdf_infra = load_exclusions(target_crs)
    raw_all = gpd.read_file(merged_cov_3857)
    if raw_all.crs != target_crs:
        raw_all = raw_all.to_crs(target_crs)
    raw_effort = raw_all.loc[
        (raw_all["year"] >= MATCHED_WINDOW_START) &
        (raw_all["year"] <= MATCHED_WINDOW_END)
    ].copy()
    raw_coords = np.column_stack(
        [raw_effort.geometry.x.values, raw_effort.geometry.y.values])
    tree = cKDTree(raw_coords)
    h3_col = h3_id_column(gdf_h3)

    # ---- 5a. Eligible cells --------------------------------------------------
    # of the pseudo-absence definition unless INFRA_RULE says so; the same
    # point-level rule is applied to both classes further down.
    n_cells = len(gdf_h3)
    if INFRA_RULE == "cell_neighbours_both":
        bad = gpd.sjoin(gdf_h3, gdf_infra, how="inner", predicate="intersects")
        bad_idx = bad.index.unique()
        gdf_bad = gdf_h3.loc[bad_idx]
        nbr = gpd.sjoin(gdf_h3, gdf_bad, how="inner", predicate="touches")
        drop_idx = set(bad_idx).union(set(nbr.index.unique()))
        gdf_cells = gdf_h3.drop(index=list(drop_idx)).copy()
        note = "cells intersecting infrastructure plus neighbours removed"
    else:
        gdf_cells = gdf_h3.copy()
        note = "no cell-level removal; point-level rule applied to both classes"
    audit("pa_cell_eligibility", "pseudo_absences", n_cells, len(gdf_cells), note)

    # cell. It therefore cannot be drawn from a cell used as a presence. The
    # full 2015-2021 non-NDVI layer is used so a known visit is respected even
    # when the main model uses the narrower 2019-2021 window.
    visited = gpd.read_file(out_pres_nondvi_full)
    if h3_col not in visited.columns:
        raise RuntimeError(
            "Presence output does not contain the H3 id column {!r}. "
            "Run step_presences with the same H3 grid first.".format(h3_col))
    excluded_ids = presence_guard_cells(
        gdf_h3, h3_col, visited[h3_col].astype(str), PA_GUARD_RINGS)
    n_before_guard = len(gdf_cells)
    gdf_cells = gdf_cells.loc[
        ~gdf_cells[h3_col].astype(str).isin(excluded_ids)].copy()
    audit("presence_cell_guard", "pseudo_absences", n_before_guard,
          len(gdf_cells),
          "exclude known presence cells; neighbour rings={0}".format(
              PA_GUARD_RINGS))

    # ---- 5b. Cell selection --------------------------------------------------
    if PA_SAMPLING_MODE == "effort_weighted":
        # the density of raw georeferenced records, so the negative class is
        # exposed to the same observation process as the positive class and
        # "no record detected" is no longer confounded with low platform
        # penetration.
        print("   Computing effort weights at cell centroids...")
        centroids = gdf_cells.geometry.centroid
        cent_coords = np.column_stack([centroids.x.values, centroids.y.values])
        eff = tree.query_ball_point(cent_coords, r=EFFORT_RADIUS_M, return_length=True)
        weights = np.log1p(np.asarray(eff, dtype="float64")) + 0.05  # 0.05 keeps zero-effort cells reachable
        weights = weights / weights.sum()
        n_draw = min(PA_POOL_N, len(gdf_cells))
        idx = rng.choice(len(gdf_cells), size=n_draw, replace=False, p=weights)
        gdf_subset = gdf_cells.iloc[idx].copy()
        note = "effort-weighted draw, radius {0:.0f} m".format(EFFORT_RADIUS_M)
    elif PA_SAMPLING_MODE == "random":
        n_draw = min(PA_POOL_N, len(gdf_cells))
        gdf_subset = gdf_cells.sample(n=n_draw, random_state=SEED).copy()
        note = "uniform draw (original behaviour)"
    else:
        raise ValueError("Unknown PA_SAMPLING_MODE: {0}".format(PA_SAMPLING_MODE))
    audit("pa_cell_draw", "pseudo_absences", len(gdf_cells), len(gdf_subset), note)

    # ---- 5c. One random point per selected cell -----------------------------
    print("   Generating one reproducible random point per selected cell...")
    gdf_subset = gdf_subset.reset_index(drop=True)
    points = [random_point_in_polygon(geom, rng)
              for geom in gdf_subset.geometry]
    gdf_points = gpd.GeoDataFrame(
        {h3_col: gdf_subset[h3_col].values}, geometry=points, crs=target_crs)
    print("   Points generated: {0}".format(len(gdf_points)))

    # ---- 5d. Minimum spacing -------------------------------------------------
    coords = np.column_stack([gdf_points.geometry.x.values, gdf_points.geometry.y.values])
    pair_tree = cKDTree(coords)
    pairs = pair_tree.query_pairs(PA_MIN_DIST_M)
    drop = set()
    for i, j in pairs:
        if i not in drop and j not in drop:
            drop.add(j)
    keep = [i for i in range(len(coords)) if i not in drop]
    n_before = len(gdf_points)
    gdf_points = gdf_points.iloc[keep].reset_index(drop=True)
    audit("pa_min_distance", "pseudo_absences", n_before, len(gdf_points),
          "minimum spacing {0:.0f} m".format(PA_MIN_DIST_M))

    # ---- 5e. Covariates, identical to the presence side ---------------------
    gdf_points = sample_raster_values(gdf_points, esa_raster, "esa")
    gdf_points = sample_raster_values(gdf_points, ndvi_raster, "ndvi")
    gdf_points = compute_effort(gdf_points, tree)
    gdf_points = add_distance_to_infrastructure(gdf_points, gdf_infra)
    gdf_points = tag_ordlo(gdf_points)

    # ---- 5f. Same land cover and infrastructure rules as the presences ------
    n_before = len(gdf_points)
    gdf_points = gdf_points.loc[gdf_points["esa"].isin(TARGET_ESA_CLASSES)].copy()
    audit("esa_filter", "pseudo_absences", n_before, len(gdf_points),
          "identical to the presence rule")

    gdf_points = apply_infra_rule(gdf_points, gdf_infra, gdf_h3, "pseudo_absences")

    gdf_points["presence"] = 0
    gdf_points["infra_rule"] = INFRA_RULE
    gdf_points["pa_mode"] = PA_SAMPLING_MODE
    gdf_points["pa_guard_rings"] = PA_GUARD_RINGS
    gdf_points.to_file(out_pa_pool, driver="GPKG")
    print("   Candidate pool written: {0} points".format(len(gdf_points)))

    # ---- 5g. Four exported layers, one per presence layer -------------------
    # separable the two classes were under the submitted asymmetric design,
    # where every presence had NDVI >= 0.5 by construction while the background
    # was unfiltered.
    frac_pool_green = float((gdf_points["ndvi"] >= MIN_NDVI).mean())
    print("\n   [R3-7 diagnostic] {0:.1%} of the pseudo-absence pool has "
          "NDVI >= {1}.".format(frac_pool_green, MIN_NDVI))
    print("   Under the submitted asymmetric design the remaining {0:.1%} were "
          "separable".format(1 - frac_pool_green))
    print("   from EVERY presence on the NDVI threshold alone. Quote this "
          "figure in the response.")
    audit("ndvi_asymmetry_diagnostic", "pseudo_absences",
          len(gdf_points), int((gdf_points["ndvi"] >= MIN_NDVI).sum()),
          "pool fraction with NDVI >= {0}".format(MIN_NDVI))

    def target_count(presence_path, tag):
        """"""
        if PA_TARGET_N_MODE == "fixed":
            return PA_FINAL_N
        try:
            n_pres = len(gpd.read_file(presence_path))
        except Exception as exc:
            print("   Could not read {0} ({1}); falling back to PA_FINAL_N."
                  .format(os.path.basename(presence_path), exc))
            return PA_FINAL_N
        print("   Matching pseudo-absence count for {0} to {1} presences."
              .format(tag, n_pres))
        return n_pres

    def finalise(pool, tag, window, path, gee_basename, target_n):
        """
        The pool is shuffled ONCE per NDVI variant and the layers are taken from
        the top, so the matched-window layer is a strict subset of the
        full-window layer. The two runs then differ only in size, not in which
        background locations happen to have been drawn.
        """
        if len(pool) > target_n:
            out = pool.head(target_n).copy()
        else:
            print("   Warning: only {0} points available for {1}/{2} "
                  "(target {3}). Prevalence will not be 0.5; report it."
                  .format(len(pool), tag, window, target_n))
            out = pool.copy()
        out["variant"] = tag
        out["time_window"] = window
        audit("pa_final_draw", "pseudo_absences_{0}_{1}".format(tag, window),
              len(pool), len(out), "exported layer")
        keep_cols = ["presence", "variant", "time_window", "infra_rule",
                     "pa_mode", "pa_guard_rings", h3_col, "esa", "ndvi",
                     "dist_infra_m", "effort_5km", "log_effort", "in_ordlo",
                     "geometry"]
        keep_cols = [c for c in keep_cols if c in out.columns]
        export_pair(out[keep_cols], path, gee_basename)

    # One shuffled pool per NDVI variant, so the layers below are nested.
    pool_ndvi = gdf_points.loc[gdf_points["ndvi"] >= MIN_NDVI].sample(
        frac=1.0, random_state=SEED).copy()
    pool_nondvi = gdf_points.sample(frac=1.0, random_state=SEED).copy()
    pool_recurrent = gdf_points.loc[
        gdf_points["esa"].isin(RECURRENT_ESA_CLASSES) &
        (gdf_points["effort_5km"] > 0)
    ].sample(frac=1.0, random_state=SEED).copy()
    audit("recurrent_background_domain", "pseudo_absences_recurrent",
          len(gdf_points), len(pool_recurrent),
          "core green/blue classes and non-zero target-group effort")

    layers = [
        (pool_ndvi, "ndvi050", "matched", out_pres_ndvi),
        (pool_nondvi, "nondvi", "matched", out_pres_nondvi),
        (pool_ndvi, "ndvi050", "full", out_pres_ndvi_full),
        (pool_nondvi, "nondvi", "full", out_pres_nondvi_full),
    ]
    for pool, tag, window, pres_path in layers:
        key = "pa_{0}_{1}".format(tag, window)
        finalise(pool, tag, window, out_pa[(tag, window)],
                 GEE_EXPORT_NAMES[key],
                 target_count(pres_path, "{0}/{1}".format(tag, window)))

    finalise(
        pool_recurrent,
        "recurrent_core_nondvi",
        "matched",
        out_pa_recurrent,
        GEE_EXPORT_NAMES["pa_recurrent"],
        target_count(out_pres_recurrent, "recurrent_core_nondvi/matched"),
    )

    print("Done.")


# =============================================================================
# MAIN
# =============================================================================

# Asset IDs read by GEE_02 (asset name = ZIP name in UPLOAD_DIR).
ASSET_HINTS = {key: "{0}/{1}{2}".format(folders.ASSET_ROOT, ASSET_PREFIX, GEE_EXPORT_NAMES[key])
               for key in ("presences_nondvi_matched", "pa_nondvi_matched",
                           "presences_nondvi_full", "pa_nondvi_full",
                           "presences_recurrent", "pa_recurrent")}

if __name__ == "__main__":
    if RUN_STEPS["merge"]:
        step_merge()
    if RUN_STEPS["ordlo_stats"]:
        step_ordlo_stats()
    if RUN_STEPS["sample_covariates"]:
        step_sample_covariates()
    if RUN_STEPS["presences"]:
        step_presences()
    if RUN_STEPS["pseudo_absences"]:
        step_pseudo_absences()

    write_audit()

    # ---- Final summary -------------------------------------------------------
    # These are the numbers that go into the revised Methods and into Table 1's
    # prevalence statement, so they are printed together rather than left to be
    print("\n" + "=" * 74)
    print("FINAL COUNTS AND PREVALENCE")
    print("=" * 74)
    pairs = [
        ("ndvi050 / matched 2019-2021", out_pres_ndvi, out_pa[("ndvi050", "matched")]),
        ("nondvi  / matched 2019-2021", out_pres_nondvi, out_pa[("nondvi", "matched")]),
        ("ndvi050 / full 2015-2021", out_pres_ndvi_full, out_pa[("ndvi050", "full")]),
        ("nondvi  / full 2015-2021", out_pres_nondvi_full, out_pa[("nondvi", "full")]),
        ("recurrent core / matched", out_pres_recurrent, out_pa_recurrent),
    ]
    summary_rows = []
    for label, pres_path, pa_path in pairs:
        try:
            n_p = len(gpd.read_file(pres_path))
            n_a = len(gpd.read_file(pa_path))
        except Exception:
            continue
        prev = n_p / (n_p + n_a)
        summary_rows.append({"pairing": label, "n_presence": n_p,
                             "n_absence": n_a, "prevalence": round(prev, 4)})
        print("  {0:30s} presences {1:>7}  pseudo-absences {2:>7}  "
              "prevalence {3:.3f}".format(label, n_p, n_a, prev))
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(work_dir, "final_counts_{0}.csv".format(RUN_TAG)),
            index=False)

    print("\nThe matched-window layer (2019-2021) is the main analysis; the")
    print("full 2015-2021 layer is the sensitivity run that quantifies the cost")
    print("of the temporal matching. Only the *_nondvi_* and recurrent layers")
    print("are uploaded to Earth Engine.")

    print("\nUpload the six ZIP files in {0} as Earth Engine table assets "
          "(asset name = ZIP name). The asset IDs used by GEE_02 are:".format(UPLOAD_DIR))
    print(json.dumps(ASSET_HINTS, indent=2))


# In[ ]:
