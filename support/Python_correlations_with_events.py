#!/usr/bin/env python
# coding: utf-8
"""
=============================================================================
 ZONAL STATISTICS, SPEARMAN CORRELATIONS WITH DOCUMENTED WAR EVENTS AND FIGURE 8 -- REVISED VERSION
 (file renamed from Python_correlations_and_validation.py; the event data are
 COMPARISON data for RQ3, not a validation of the detections)
 Manuscript SUST-D-26-00527 (Sustainability Science)
=============================================================================

 WHAT CHANGED IN THIS VERSION AND WHY
 ------------------------------------
 Reviewer numbering follows the decision letter (R3 = detailed methodological
 review, R2 = page-by-page review).

     This version computes MAGNITUDE columns for all four intensity metrics,
     so that in every cell of Figure 8 a larger value means more of the change
     type named in the column, and a positive rho always means "more war events
     co-occur with more of that change". Both conventions are exported, and
     sign_convention_check.csv shows the two side by side so the change can be
     documented in the response letter. Every correlation row also carries a
     plain-language 'interpretation' string.

 COMBINED DATASETS. A combined
     ACLED plus VIINA column but the figures dropped it. The combined column is
     now optional in the figures, and DEDUP_COMBINED removes events reported by
     both sources at the same place and time, which naive concatenation
     double-counts.

"""

import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.stats import spearmanr, rankdata
from pathlib import Path
from collections import Counter

import sys as _sys
try:                        # run as a file (also via Python_02)
    _SUPPORT = Path(__file__).resolve().parent
except NameError:           # pasted into a notebook cell
    _SUPPORT = Path.cwd() / "support"
_sys.path.insert(0, str(_SUPPORT))
# A console that already ran another copy of the package keeps its support
# modules in memory; drop them so that this package's support folder is used.
for _m in ("folders", "pipeline_checks"):
    _sys.modules.pop(_m, None)
import folders  # noqa: E402  (local folders: support/folders.py)

# =============================================================================
# CONFIGURATION
# =============================================================================

WAR_EVENTS_PATH = str(folders.WAR_EVENTS)
HEX_PATH        = str(folders.HEXAGON_GRID)
ESA_PATH        = str(folders.WORLDCOVER)

# One GEE_05 folder per visitation model, one output folder per model
# (support/folders.py). The analysis runs once per GEE_05 folder it finds.
ANALYSIS_SCHEMES = ["recurrent_effort", "primary_effort"]
SCHEME_LABELS = {
    "recurrent_effort":
        "Recurrent outdoor visitation model (2019-2021) [comparison]",
    "primary_effort": "Regular outdoor visitation model (2019-2021) [main]",
}
RF_MAIN_NAME = "Debiased_RF_Main_revision_annualbase_annual_median.tif"
RF_CONF_NAME = "Debiased_RF_Confidence_revision_annualbase_annual_median.tif"
RF_PATH = None    # set per scheme in main()
CONF_PATH = None  # set per scheme in main()
OUT_DIR = None    # set per scheme in main()
CURRENT_SCHEME_LABEL = None

# Band indices in the revised main stack (1-based, as rasterio expects).
ZSCORE_BAND = 7
NOVEL_BAND  = 25     # extrapolation flag; set to None for the old raster
ORDLO_BAND  = 26     # pre-2022 occupation line; set to None for the old raster

Z_VALID_MIN = -15.0
Z_VALID_MAX =  15.0
Z_NODATA    = -9999.0

MIN_N = 30

# ---- Reviewer-driven switches -----------------------------------------------

# 'raw' reproduces the submitted Figure 8 exactly, for the response letter.
SIGN_CONVENTION = 'magnitude'

# detections. GEE_05 uses ACLED only to show that spatial control areas are
# impossible. HEADLINE_SOURCE selects the source quoted in abstract_statement.txt.
FEASIBILITY_CHECK_SOURCE = 'ACLED'
HEADLINE_SOURCE          = 'VIINA'

# [R2 p.20] Combine the two sources, removing events reported by both.
DEDUP_COMBINED     = True
DEDUP_TOLERANCE_M  = 1000      # events closer than this, same date, count once

EXCLUDE_NOVEL_HEX = True
NOVEL_MAX_FRAC    = 0.50

COMPUTE_CI   = True
N_BOOT       = 500
BOOT_BLOCK_KM = 50             # spatial blocks for the bootstrap
FDR_ALPHA    = 0.05
SEED         = 20250523

rng = np.random.default_rng(SEED)

ESA_LEGEND = {
    10: 'Tree cover', 20: 'Shrubland', 30: 'Grassland',
    40: 'Cropland', 50: 'Built-up', 60: 'Bare / sparse vegetation',
    70: 'Snow and ice', 80: 'Water bodies',
    90: 'Herbaceous wetland', 95: 'Mangroves',
    100: 'Moss and lichen',
}
VALID_LC_CODES = set(ESA_LEGEND.keys())


# =============================================================================
# RASTER I/O HELPERS (unchanged)
# =============================================================================

def _hex_window(src, geom, transform):
    minx, miny, maxx, maxy = geom.bounds
    c0 = max(0, int(np.floor((minx - transform.c) / transform.a)))
    r0 = max(0, int(np.floor((maxy - transform.f) / transform.e)))
    c1 = min(src.width,  int(np.ceil((maxx - transform.c) / transform.a)))
    r1 = min(src.height, int(np.ceil((miny - transform.f) / transform.e)))
    if r1 <= r0 or c1 <= c0:
        return None, None
    win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
    return win, src.window_transform(win)


def _masked_pixels(src, geom, transform, nodata, band=1):
    win, wt = _hex_window(src, geom, transform)
    if win is None:
        return None
    arr = src.read(band, window=win).astype(np.float32)
    mask = rasterize([(geom, 1)], out_shape=arr.shape,
                     transform=wt, fill=0, dtype='uint8')
    inside = arr[mask == 1]
    if nodata is not None:
        inside = inside[inside != nodata]
    inside = inside[np.isfinite(inside)]
    return inside if len(inside) > 0 else None


def extract_band_tif(raster_path, band_idx, tmp_path, clamp=True):
    with rasterio.open(raster_path) as src:
        arr = src.read(band_idx).astype(np.float32)
        nd = src.nodata
        profile = src.profile.copy()
    invalid = np.isnan(arr)
    if nd is not None:
        invalid |= (arr == nd)
    if clamp:
        invalid |= (arr < Z_VALID_MIN) | (arr > Z_VALID_MAX)
    arr[invalid] = Z_NODATA
    profile.update(count=1, dtype='float32', nodata=Z_NODATA,
                   compress='deflate', tiled=True,
                   blockxsize=256, blockysize=256)
    with rasterio.open(tmp_path, 'w', **profile) as dst:
        dst.write(arr, 1)


# =============================================================================
# ZONAL STATISTICS
# =============================================================================

def zscore_zonal(hexgdf, zscore_tif):
    """"""
    keys = ['z_median_damage', 'z_median_rewild',
            'z_max_damage', 'z_max_rewild', 'z_count',
            'z_n_pos', 'z_n_neg']
    out = {k: [] for k in keys}
    nan_row = {k: np.nan for k in keys}
    nan_row['z_count'] = 0
    nan_row['z_n_pos'] = 0
    nan_row['z_n_neg'] = 0

    with rasterio.open(zscore_tif) as src:
        crs, nd, transform = src.crs, src.nodata, src.transform
        hex_r = hexgdf.to_crs(crs) if hexgdf.crs != crs else hexgdf
        total = len(hex_r)

        for ci, (_, row) in enumerate(hex_r.iterrows()):
            if (ci + 1) % 500 == 0:
                print(f"      {ci + 1}/{total}")
            geom = row.geometry
            if geom is None or geom.is_empty:
                for k in keys: out[k].append(nan_row[k])
                continue
            try:
                px = _masked_pixels(src, geom, transform, nd)
                if px is None:
                    for k in keys: out[k].append(nan_row[k])
                    continue
                pos = px[px > 0]
                neg = px[px < 0]
                out['z_count'].append(len(px))
                out['z_n_pos'].append(int(len(pos)))
                out['z_n_neg'].append(int(len(neg)))
                out['z_max_damage'].append(float(np.max(px)))
                out['z_max_rewild'].append(float(np.min(px)))
                out['z_median_damage'].append(
                    float(np.median(pos)) if len(pos) > 0 else 0.0)
                out['z_median_rewild'].append(
                    float(np.median(neg)) if len(neg) > 0 else 0.0)
            except Exception:
                for k in keys: out[k].append(nan_row[k])
    return out


def add_magnitude_columns(hexgdf):
    """"""
    hexgdf['mag_median_damage'] = hexgdf['z_median_damage'].clip(lower=0)
    hexgdf['mag_median_rewild'] = (-hexgdf['z_median_rewild']).clip(lower=0)
    hexgdf['mag_max_damage']    = hexgdf['z_max_damage'].clip(lower=0)
    hexgdf['mag_max_rewild']    = (-hexgdf['z_max_rewild']).clip(lower=0)
    return hexgdf


def confidence_zonal(hexgdf, conf_path):
    dmg, rew, val = [], [], []
    with rasterio.open(conf_path) as src:
        crs, nd, transform = src.crs, src.nodata, src.transform
        hex_r = hexgdf.to_crs(crs) if hexgdf.crs != crs else hexgdf
        total = len(hex_r)
        for ci, (_, row) in enumerate(hex_r.iterrows()):
            if (ci + 1) % 500 == 0:
                print(f"      {ci + 1}/{total}")
            geom = row.geometry
            if geom is None or geom.is_empty:
                dmg.append(0); rew.append(0); val.append(0)
                continue
            try:
                px = _masked_pixels(src, geom, transform, nd)
                if px is None:
                    dmg.append(0); rew.append(0); val.append(0)
                    continue
                val.append(len(px))
                dmg.append(int(np.sum((px >= 1) & (px <= 6))))
                rew.append(int(np.sum((px <= -1) & (px >= -6))))
            except Exception:
                dmg.append(0); rew.append(0); val.append(0)
    return dmg, rew, val


def band_fraction_zonal(hexgdf, raster_path, band_idx):
    """"""
    fracs = []
    with rasterio.open(raster_path) as src:
        crs, nd, transform = src.crs, src.nodata, src.transform
        hex_r = hexgdf.to_crs(crs) if hexgdf.crs != crs else hexgdf
        for _, row in hex_r.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                fracs.append(np.nan); continue
            try:
                px = _masked_pixels(src, geom, transform, nd, band=band_idx)
                fracs.append(float(np.mean(px > 0.5)) if px is not None else np.nan)
            except Exception:
                fracs.append(np.nan)
    return fracs


def majority_landcover(hexgdf, esa_path):
    results = []
    with rasterio.open(esa_path) as src:
        crs, nd, transform = src.crs, src.nodata, src.transform
        hex_r = hexgdf.to_crs(crs) if hexgdf.crs != crs else hexgdf
        total = len(hex_r)
        for ci, (_, row) in enumerate(hex_r.iterrows()):
            if (ci + 1) % 500 == 0:
                print(f"      {ci + 1}/{total}")
            geom = row.geometry
            if geom is None or geom.is_empty:
                results.append(np.nan); continue
            try:
                win, wt = _hex_window(src, geom, transform)
                if win is None:
                    results.append(np.nan); continue
                arr = src.read(1, window=win)
                mask = rasterize([(geom, 1)], out_shape=arr.shape,
                                 transform=wt, fill=0, dtype='uint8')
                inside = arr[mask == 1]
                inside = inside[inside != 0]
                if nd is not None and nd != 0:
                    inside = inside[inside != nd]
                inside = inside[np.isin(inside, list(VALID_LC_CODES))]
                if len(inside) == 0:
                    results.append(np.nan)
                else:
                    results.append(int(Counter(inside).most_common(1)[0][0]))
            except Exception:
                results.append(np.nan)
    return results


# =============================================================================
# EVENT COUNTING, WITH ROLES AND DEDUPLICATION
# =============================================================================

def find_source_column(df):
    for c in df.columns:
        if c.lower() in ('layer', 'source', 'dataset'):
            return c
    for c in df.columns:
        if 'layer' in c.lower() or 'source' in c.lower():
            return c
    return None


def normalise_source(value):
    """Map the layer names in the merged asset onto ACLED / VIINA."""
    v = str(value).lower()
    if 'acled' in v:
        return 'ACLED'
    if 'viina' in v:
        return 'VIINA'
    return str(value)


def dedup_events(events, source_col):
    """
    [R2 p.20] The same incident reported by both ACLED and VIINA is two rows in
    a naive concatenation. Events from different sources within
    DEDUP_TOLERANCE_M of each other on the same date are collapsed to one.
    Without a date column the tolerance is applied on location alone, which is
    more aggressive and is reported as such.
    """
    date_col = None
    for c in events.columns:
        if c.lower() in ('date', 'event_date', 'timestamp', 'datetime'):
            date_col = c
            break

    # ~48-50 deg N are ~1.5x smaller on the ground than nominal, so the
    # stated 1 km tolerance was really ~660 m.
    ev = events.to_crs(3035).copy()
    ev['_x'] = (ev.geometry.x / DEDUP_TOLERANCE_M).round().astype(int)
    ev['_y'] = (ev.geometry.y / DEDUP_TOLERANCE_M).round().astype(int)
    if date_col:
        ev['_d'] = pd.to_datetime(ev[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
        keys = ['_x', '_y', '_d']
    else:
        print("    WARNING: no date column found, deduplicating on location only.")
        keys = ['_x', '_y']

    before = len(ev)
    ev = ev.drop_duplicates(subset=keys, keep='first')
    print(f"    Combined deduplication: {before} -> {len(ev)} events "
          f"({before - len(ev)} duplicates removed)")
    return ev.to_crs(events.crs).drop(columns=['_x', '_y'] +
                                      (['_d'] if date_col else []))


def count_events(hexgdf, events_path):
    events = gpd.read_file(events_path)
    print(f"    {len(events)} events loaded.")
    if events.crs != hexgdf.crs:
        events = events.to_crs(hexgdf.crs)

    source_col = find_source_column(events)
    if source_col is None:
        raise RuntimeError(
            "No source/layer column in the war events file. The independence "
            "argument in comment 13 cannot be answered without one.")
    events['source_role'] = events[source_col].map(normalise_source)
    print(f"    Source column '{source_col}', values: "
          f"{sorted(events['source_role'].unique())}")

    joined = gpd.sjoin(events, hexgdf[['hex_id', 'geometry']],
                       how='inner', predicate='within')

    acled = (joined[joined['source_role'] == 'ACLED']
             .groupby('hex_id').size().rename('n_acled'))
    viina = (joined[joined['source_role'] == 'VIINA']
             .groupby('hex_id').size().rename('n_viina'))

    if DEDUP_COMBINED:
        ev_dedup = dedup_events(events, source_col)
        joined_c = gpd.sjoin(ev_dedup, hexgdf[['hex_id', 'geometry']],
                             how='inner', predicate='within')
        combined = joined_c.groupby('hex_id').size().rename('n_combined')
    else:
        combined = joined.groupby('hex_id').size().rename('n_combined')

    for s in (acled, viina, combined):
        hexgdf = hexgdf.join(s, on='hex_id')
    hexgdf = hexgdf.fillna({'n_acled': 0, 'n_viina': 0, 'n_combined': 0})
    for c in ['n_acled', 'n_viina', 'n_combined']:
        hexgdf[c] = hexgdf[c].astype(int)

    print(f"    ACLED: {hexgdf['n_acled'].sum()}, "
          f"VIINA: {hexgdf['n_viina'].sum()}, "
          f"Combined: {hexgdf['n_combined'].sum()}")
    return hexgdf


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

if SIGN_CONVENTION == 'magnitude':
    Z_METRICS = [
        ('Median loss',      'mag_median_damage', 'intensity'),
        ('Median gain',  'mag_median_rewild', 'intensity'),
        ('Max loss',         'mag_max_damage',    'intensity'),
        ('Max gain',     'mag_max_rewild',    'intensity'),
    ]
else:
    Z_METRICS = [
        ('Median loss',      'z_median_damage',   'intensity'),
        ('Median gain',  'z_median_rewild',   'intensity'),
        ('Max loss',         'z_max_damage',      'intensity'),
        ('Max gain',     'z_max_rewild',      'intensity'),
    ]

PX_METRICS = [
    ('Loss pixels',     'conf_damage_px', 'scope'),
    ('Gain pixels', 'conf_rewild_px', 'scope'),
]

SUBSET_METRICS = {
    'all_valid': Z_METRICS + PX_METRICS,
    'z_active':  Z_METRICS,
    'px_active': PX_METRICS,
}

# Plain-language reading of a positive coefficient, written into every row so
def interpretation_text(metric_label, convention):
    # damage/disturbance) and 'visitation-gain anomaly' (wilding-like
    # signal). Machine column names keep damage/rewild for continuity.
    change = ('visitation-gain anomalies'
              if 'gain' in metric_label.lower()
              else 'visitation-loss anomalies')
    if convention == 'magnitude':
        return (f"positive rho: hexagons with more documented war events also "
                f"show stronger {change}")
    if 'gain' in metric_label.lower() and 'pixels' not in metric_label.lower():
        return (f"NEGATIVE rho: hexagons with more documented war events show "
                f"stronger {change}, because this column stores signed "
                f"negative z-scores")
    return (f"positive rho: hexagons with more documented war events also show "
            f"stronger {change}")


# =============================================================================
# SUBSETS
# =============================================================================

def make_subsets(hexgdf, ev_col):
    """
    all_valid : raster coverage. Events and change may both be zero.
    z_active  : raster coverage and events > 0.
    px_active : raster coverage, events > 0 and at least one confident pixel.

    px_active is always the smallest, which is why the pixel columns of
    Figure 8 carry much smaller n than the intensity columns. That is now
    stated in the caption rather than left to be inferred.
    """
    has_raster = (hexgdf['z_count'] > 0) & (hexgdf['conf_valid_px'] > 0)
    if EXCLUDE_NOVEL_HEX and 'novel_frac' in hexgdf.columns:
        has_raster = has_raster & (hexgdf['novel_frac'].fillna(0) <= NOVEL_MAX_FRAC)
    has_events = hexgdf[ev_col] > 0
    has_change = (hexgdf['conf_damage_px'] > 0) | (hexgdf['conf_rewild_px'] > 0)
    return {
        'all_valid': has_raster,
        'z_active':  has_raster & has_events,
        'px_active': has_raster & has_events & has_change,
    }


# =============================================================================
# UNCERTAINTY: SPATIALLY BLOCKED BOOTSTRAP AND BH CORRECTION
# =============================================================================

def blocked_bootstrap_rho(x, y, block_ids, n_boot=N_BOOT):
    """

"""
    if len(x) < MIN_N:
        return np.nan, np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    blocks, inv = np.unique(block_ids, return_inverse=True)
    idx_by_block = [np.where(inv == i)[0] for i in range(len(blocks))]

    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([idx_by_block[b] for b in pick])
        a, b = rankdata(x[idx]), rankdata(y[idx])
        sa, sb = a.std(), b.std()
        out[i] = np.nan if (sa == 0 or sb == 0) else np.corrcoef(a, b)[0, 1]
    out = out[np.isfinite(out)]
    if len(out) == 0:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def bh_correct(pvals, alpha=FDR_ALPHA):
    """"""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return out
    ps = p[ok]
    order = np.argsort(ps)
    ranked = ps[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty(n)
    tmp[order] = adj
    out[ok] = tmp
    return out


def spatial_block_ids(hexdf, block_km=BOOT_BLOCK_KM):
    """Coarse spatial blocks from hexagon centroids, for the bootstrap."""
    lat0 = float(hexdf['cent_lat'].mean())
    x = (hexdf['cent_lon'] - hexdf['cent_lon'].min()) * 111.32 * np.cos(np.deg2rad(lat0))
    y = (hexdf['cent_lat'] - hexdf['cent_lat'].min()) * 110.57
    return (np.floor(x / block_km).astype(int) * 100000 +
            np.floor(y / block_km).astype(int)).to_numpy()


# =============================================================================
# CORRELATIONS
# =============================================================================

def compute_correlations(hexgdf, event_cols):
    rows = []
    total_hex = int(((hexgdf['z_count'] > 0) & (hexgdf['conf_valid_px'] > 0)).sum())

    strata = [('All', None)]
    if 'ordlo_frac' in hexgdf.columns:
        # only so that the table stays readable.
        strata += [('Outside ORDLO', hexgdf['ordlo_frac'].fillna(0) < 0.5),
                   ('Inside ORDLO',  hexgdf['ordlo_frac'].fillna(0) >= 0.5)]

    for ev_label, ev_col in event_cols.items():
        # Both sources are independent of the detected anomalies: no event
        # data enter the model or the change detection.
        independent = True
        subsets = make_subsets(hexgdf, ev_col)

        for subset_name, shared_mask in subsets.items():
            metrics = SUBSET_METRICS[subset_name]

            # ---- pooled land cover, by stratum -----------------------------
            for stratum_name, stratum_mask in strata:
                m = shared_mask if stratum_mask is None else (shared_mask & stratum_mask)
                sub = hexgdf[m]
                if len(sub) < MIN_N:
                    continue
                blocks = spatial_block_ids(sub)
                for m_label, m_col, m_group in metrics:
                    rho, pval = spearmanr(sub[ev_col].values, sub[m_col].values)
                    lo, hi = (blocked_bootstrap_rho(sub[ev_col].values,
                                                    sub[m_col].values, blocks)
                              if COMPUTE_CI else (np.nan, np.nan))
                    rows.append({
                        'subset': subset_name,
                        'event_source': ev_label,
                        'event_role': 'comparison',
                        'independent_of_pipeline': independent,
                        'stratum': stratum_name,
                        'metric': m_label,
                        'metric_column': m_col,
                        'sign_convention': SIGN_CONVENTION,
                        'group': m_group,
                        'land_cover': 'All',
                        'lc_code': -1,
                        'n': len(sub),
                        'n_share_of_valid': round(len(sub) / max(total_hex, 1), 3),
                        'spearman_rho': rho,
                        'p_value': pval,
                        'rho_ci_lo': lo,
                        'rho_ci_hi': hi,
                        'interpretation': interpretation_text(m_label, SIGN_CONVENTION),
                    })

            # ---- per land cover, pooled strata ------------------------------
            for lc_code, lc_name in ESA_LEGEND.items():
                lc_sub = hexgdf[shared_mask & (hexgdf['lc_code'] == lc_code)]
                if len(lc_sub) < MIN_N:
                    continue
                blocks = spatial_block_ids(lc_sub)
                for m_label, m_col, m_group in metrics:
                    rho, pval = spearmanr(lc_sub[ev_col].values, lc_sub[m_col].values)
                    lo, hi = (blocked_bootstrap_rho(lc_sub[ev_col].values,
                                                    lc_sub[m_col].values, blocks)
                              if COMPUTE_CI else (np.nan, np.nan))
                    rows.append({
                        'subset': subset_name,
                        'event_source': ev_label,
                        'event_role': 'comparison',
                        'independent_of_pipeline': independent,
                        'stratum': 'All',
                        'metric': m_label,
                        'metric_column': m_col,
                        'sign_convention': SIGN_CONVENTION,
                        'group': m_group,
                        'land_cover': lc_name,
                        'lc_code': lc_code,
                        'n': len(lc_sub),
                        'n_share_of_valid': round(len(lc_sub) / max(total_hex, 1), 3),
                        'spearman_rho': rho,
                        'p_value': pval,
                        'rho_ci_lo': lo,
                        'rho_ci_hi': hi,
                        'interpretation': interpretation_text(m_label, SIGN_CONVENTION),
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df['p_fdr'] = bh_correct(df['p_value'].values)
        df['significant_fdr'] = df['p_fdr'] < FDR_ALPHA
        df['spatial_ci_excludes_zero'] = (df['rho_ci_lo'] > 0) | (df['rho_ci_hi'] < 0)
        df['subgroup_flag'] = df['n_share_of_valid'] < 0.10
    return df


def sign_convention_check(hexgdf, event_cols):
    """"""
    pairs = [('Median gain', 'z_median_rewild', 'mag_median_rewild'),
             ('Max gain',    'z_max_rewild',    'mag_max_rewild'),
             ('Median loss',     'z_median_damage', 'mag_median_damage'),
             ('Max loss',        'z_max_damage',    'mag_max_damage')]
    rows = []
    for ev_label, ev_col in event_cols.items():
        mask = make_subsets(hexgdf, ev_col)['z_active']
        sub = hexgdf[mask]
        if len(sub) < MIN_N:
            continue
        for label, raw_col, mag_col in pairs:
            r_raw, p_raw = spearmanr(sub[ev_col].values, sub[raw_col].values)
            r_mag, p_mag = spearmanr(sub[ev_col].values, sub[mag_col].values)
            rows.append({'event_source': ev_label, 'metric': label, 'n': len(sub),
                         'rho_raw_signed': r_raw, 'p_raw': p_raw,
                         'rho_magnitude': r_mag, 'p_magnitude': p_mag,
                         'sign_flipped': np.sign(r_raw) != np.sign(r_mag)})
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def run_scheme():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / '_tmp'
    tmp_dir.mkdir(exist_ok=True)

    print("=" * 74)
    print(f"ZONAL STATISTICS AND SPEARMAN CORRELATIONS -- "
          f"{CURRENT_SCHEME_LABEL}")
    print(f"Sign convention: {SIGN_CONVENTION} | "
          f"comparison sources: {HEADLINE_SOURCE}, "
          f"{FEASIBILITY_CHECK_SOURCE}")
    print("=" * 74)

    # ---- 1. Hexagons ----
    print("\n[1] Loading hexagons...")
    hexgdf = gpd.read_file(HEX_PATH)
    if 'hex_id' not in hexgdf.columns:
        hexgdf['hex_id'] = hexgdf.index
    cent = hexgdf.geometry.centroid.to_crs(4326) if hexgdf.crs.to_epsg() != 4326 \
        else hexgdf.geometry.centroid
    hexgdf['cent_lon'] = cent.x.values
    hexgdf['cent_lat'] = cent.y.values
    print(f"    {len(hexgdf)} hexagons loaded.")

    # ---- 2. War events, with roles ----
    print("\n[2] Counting war events per hexagon...")
    hexgdf = count_events(hexgdf, WAR_EVENTS_PATH)

    # ---- 3. Z-score zonal statistics ----
    print(f"\n[3] Z-score zonal statistics (band {ZSCORE_BAND})...")
    z_tmp = str(tmp_dir / 'zscore_rf.tif')
    extract_band_tif(RF_PATH, ZSCORE_BAND, z_tmp)
    for k, v in zscore_zonal(hexgdf, z_tmp).items():
        hexgdf[k] = v
    hexgdf = add_magnitude_columns(hexgdf)
    print(f"    {(hexgdf['z_count'] > 0).sum()} hexagons with valid z data")

    # ---- 4. Confidence pixel counts ----
    print("\n[4] Confidence raster zonal pixel counts...")
    dmg_c, rew_c, val_c = confidence_zonal(hexgdf, CONF_PATH)
    hexgdf['conf_damage_px'] = dmg_c
    hexgdf['conf_rewild_px'] = rew_c
    hexgdf['conf_valid_px']  = val_c

    # ---- 5. Extrapolation and occupation line fractions ----
    if NOVEL_BAND is not None:
        print(f"\n[5a] Extrapolation flag fraction (band {NOVEL_BAND})...")
        n_tmp = str(tmp_dir / 'novel.tif')
        extract_band_tif(RF_PATH, NOVEL_BAND, n_tmp, clamp=False)
        hexgdf['novel_frac'] = band_fraction_zonal(hexgdf, n_tmp, 1)
        print(f"    hexagons above {NOVEL_MAX_FRAC:.0%} novel: "
              f"{(hexgdf['novel_frac'].fillna(0) > NOVEL_MAX_FRAC).sum()}")
    if ORDLO_BAND is not None:
        print(f"\n[5b] Occupation line fraction (band {ORDLO_BAND})...")
        o_tmp = str(tmp_dir / 'ordlo.tif')
        extract_band_tif(RF_PATH, ORDLO_BAND, o_tmp, clamp=False)
        hexgdf['ordlo_frac'] = band_fraction_zonal(hexgdf, o_tmp, 1)

    # ---- 6. Majority land cover ----
    print("\n[6] Majority land cover...")
    hexgdf['lc_code'] = majority_landcover(hexgdf, ESA_PATH)
    hexgdf['lc_name'] = hexgdf['lc_code'].map(ESA_LEGEND).fillna('No data')

    # ---- 7. Export the hexagon dataset ----
    print("\n[7] Exporting hexagon dataset...")
    export_cols = [
        'hex_id', 'cent_lon', 'cent_lat',
        'n_combined', 'n_acled', 'n_viina',
        'lc_code', 'lc_name',
        'z_median_damage', 'z_median_rewild', 'z_max_damage', 'z_max_rewild',
        'mag_median_damage', 'mag_median_rewild', 'mag_max_damage', 'mag_max_rewild',
        'z_count', 'z_n_pos', 'z_n_neg',
        'conf_damage_px', 'conf_rewild_px', 'conf_valid_px',
    ]
    for c in ['novel_frac', 'ordlo_frac']:
        if c in hexgdf.columns:
            export_cols.append(c)

    hexgdf[export_cols].to_csv(out_dir / 'hexagon_stats.csv',
                               index=False, float_format='%.4f')
    hexgdf[export_cols + ['geometry']].to_file(out_dir / 'hexagon_stats.gpkg',
                                               driver='GPKG')

    # ---- 8. Correlations ----
    print("\n[8] Computing Spearman correlations...")
    event_cols = {
        'ACLED':    'n_acled',
        'VIINA':    'n_viina',
        'Combined': 'n_combined',
    }
    for ev_label, ev_col in event_cols.items():
        ss = make_subsets(hexgdf, ev_col)
        print(f"    {ev_label}: all_valid={ss['all_valid'].sum()}, "
              f"z_active={ss['z_active'].sum()}, px_active={ss['px_active'].sum()}")

    corr_df = compute_correlations(hexgdf, event_cols)
    corr_df.to_csv(out_dir / 'spearman_correlations.csv',
                   index=False, float_format='%.6f')

    check = sign_convention_check(hexgdf, event_cols)
    check.to_csv(out_dir / 'sign_convention_check.csv',
                 index=False, float_format='%.6f')
    print("\n    Sign convention check (raw signed against magnitude):")
    print(check.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # ---- 9. Reporting summary ----
    print("\n[9] Headline numbers for the abstract and the response letter...")
    val = corr_df[(corr_df['event_source'] == HEADLINE_SOURCE) &
                  (corr_df['subset'] == 'z_active') &
                  (corr_df['stratum'] == 'All')]
    grass = val[val['land_cover'] == 'Grassland']
    allrows = val[val['land_cover'] == 'All']

    lines = []
    lines.append(f"HEADLINE NUMBERS, {HEADLINE_SOURCE} "
                 f"(subset 'z_active': hexagons with at least one event; "
                 f"Figure 8 uses 'all_valid')")
    lines.append("=" * 60)
    lines.append("VIINA and ACLED are comparison data for RQ3: no event data "
                 "enter the model or the change detection, and the "
                 "correlations do not validate the detections.")
    lines.append(f"Sign convention: {SIGN_CONVENTION}")
    lines.append("")
    if not allrows.empty:
        lines.append("Pooled land cover:")
        for _, r in allrows.iterrows():
            lines.append(f"  {r['metric']:22s} rho = {r['spearman_rho']:+.2f} "
                         f"[{r['rho_ci_lo']:+.2f}, {r['rho_ci_hi']:+.2f}]  "
                         f"n = {int(r['n'])}  p_FDR = {r['p_fdr']:.3g}")
    if not grass.empty:
        lines.append("")
        lines.append("Grassland hexagons only (a small subgroup):")
        for _, r in grass.iterrows():
            lines.append(f"  {r['metric']:22s} rho = {r['spearman_rho']:+.2f} "
                         f"[{r['rho_ci_lo']:+.2f}, {r['rho_ci_hi']:+.2f}]  "
                         f"n = {int(r['n'])} "
                         f"({r['n_share_of_valid']:.1%} of valid hexagons)  "
                         f"p_FDR = {r['p_fdr']:.3g}")
    lines.append("")
    lines.append("Correlation with documented events describes co-occurrence; "
                 "it is not a validation of the detections.")

    text = "\n".join(lines)
    (out_dir / 'abstract_statement.txt').write_text(text, encoding='utf-8')
    print("\n" + text)

    # ---- 10. Cleanup ----
    for f in tmp_dir.glob('*.tif'):
        f.unlink()
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    print("\n" + "=" * 74)
    print("Done. Outputs in:", out_dir)
    print("=" * 74)


def main():
    global RF_PATH, CONF_PATH, OUT_DIR, CURRENT_SCHEME_LABEL
    ran = 0
    candidates = [
        (scheme, folders.data_dir(folders.GEE05[scheme]), folders.HEXAGONS[scheme],
         SCHEME_LABELS[scheme])
        for scheme in ANALYSIS_SCHEMES
    ]
    for scheme, in_dir, out_dir, label in candidates:
        rf = in_dir / RF_MAIN_NAME
        # v13b9c: prefer the cumulative year-count confidence classes
        # (GEE_05 export, or the local reclassifier's copy) for the
        # damage/war-wilding extent columns; band 1 = class_mapped, signed
        # -3..+3, so confidence_zonal works unchanged. Fall back to the old
        # single-band confidence raster.
        conf = None
        for cand in ("Cumulative_Confidence_Classes.tif",
                     "cumulative_confidence_classes.tif",
                     RF_CONF_NAME):
            if (in_dir / cand).exists():
                conf = in_dir / cand
                break
        if not rf.exists() or conf is None:
            print(f"[skip] {label}: rasters not found in {in_dir}")
            continue
        print(f"    extent columns from: {conf.name}")
        RF_PATH = str(rf)
        CONF_PATH = str(conf)
        OUT_DIR = str(out_dir)
        CURRENT_SCHEME_LABEL = label
        run_scheme()
        ran += 1
    if ran == 0:
        raise FileNotFoundError(
            "No GEE_05 rasters found for any scheme. Download the folders "
            "from Drive first.")


if __name__ == '__main__':
    main()
