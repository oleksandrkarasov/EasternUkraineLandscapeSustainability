#!/usr/bin/env python
# coding: utf-8
"""
=============================================================================
 EMPIRICAL NULL DIAGNOSTICS FOR THE Z FIELD
 Manuscript SUST-D-26-00527 (Sustainability Science)
=============================================================================

   1. Those are normal tail probabilities, and normality of the z field was
      never demonstrated; the former global recalibration fixed the mean and
      SD of the reference distribution and said nothing about the tails.
   2. The number of pixels tested is not accounted for. At 500 m resolution the
      eligible area contains on the order of a hundred thousand pixels, so even
      a correct 0.3 per cent per-pixel rate implies a large absolute number of
      false positives.
   3. Spatial autocorrelation surviving the 1.5 km smoothing is not addressed,
      so the effective number of independent tests is far below the pixel count.

 INPUTS (all written by GEE_05)
   ControlZ_Sample.csv                    pre-war pseudo-z with coordinates
   ControlZ_Histogram.csv                 optional, full-population histogram
   Areas_Excess_by_Zclass.csv             detected areas and pre-war benchmarks per z class
   Areas_Headline_annualbase_annual_median.csv  stricter version, detected
   Areas_Stricter_Null_v13b12.csv         stricter version on the pre-war maps
   Sensitivity_*_v13b12.csv               one table per sensitivity variant
   Analysis_Grid_and_Scale_Parameters.csv optional grid/scale audit
=============================================================================
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

try:                        # run as a file (also via Python_02)
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:           # pasted into a notebook cell: the working folder
    _HERE = os.getcwd()
for _d in (_HERE, os.path.join(_HERE, "support")):
    if os.path.isfile(os.path.join(_d, "pipeline_checks.py")):
        sys.path.insert(0, _d)
        break
else:
    raise FileNotFoundError(
        f"pipeline_checks.py not found in {_HERE} or its support folder; run this "
        "script from the unzipped package (support folder) or via Python_02.")
# A console that already ran another copy of the package keeps its support
# modules in memory; drop them so that this package's support folder is used.
for _m in ("folders", "pipeline_checks"):
    sys.modules.pop(_m, None)
from pipeline_checks import CODE_VERSION, jaccard, safe_ratio  # noqa: E402
import folders  # noqa: E402  (local folders: support/folders.py)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

# One GEE_05 folder per visitation model (support/folders.py). This script
# runs the full null-diagnostic suite once per folder it finds.
ANALYSIS_SCHEMES = ["recurrent_effort", "primary_effort"]
SCHEME_LABELS = {
    "recurrent_effort":
        "Recurrent outdoor visitation model (2019-2021) [comparison]",
    "primary_effort": "Regular outdoor visitation model (2019-2021) [main]",
}
IN_DIR = None   # set per scheme in main()
OUT_DIR = None  # set per scheme in main()

Z_GRID      = [2.0, 2.5, 3.0, 3.5, 4.0]

CORR_FINE_MAX_KM = 40       # 1-km bins, all sample points of a year
CORR_FINE_BIN_KM = 1.0
CORR_MAX_KM      = 150      # 5-km bins beyond 40 km, subsample per year
CORR_COARSE_BIN_KM = 5.0
CORR_COARSE_N    = 4000     # points per year for the long distances
RANGE_CORR       = 0.05     # range = first of two consecutive bins <= this

N_BOOT        = 1000        # blocked bootstrap replicates
SEED          = 20250523

rng = np.random.default_rng(SEED)


def _read(name, required=True):
    path = os.path.join(IN_DIR, name)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Missing input: {path}")
        print(f"    (optional input not found: {name})")
        return None
    return pd.read_csv(path)


# =============================================================================
# A. SHAPE OF THE EMPIRICAL NULL
# =============================================================================

def null_shape(z):
    """
    Moments and tail behaviour. If the excess kurtosis is positive the tails are
    heavier than normal and the per-pixel confidence wording overstates the
    evidence, whatever the mean and SD have been forced to.
    """
    m = {
        'n': len(z),
        'mean': float(np.mean(z)),
        'sd': float(np.std(z, ddof=1)),
        'median': float(np.median(z)),
        'skewness': float(stats.skew(z)),
        'excess_kurtosis': float(stats.kurtosis(z, fisher=True)),
        'p0_1': float(np.percentile(z, 0.1)),
        'p1': float(np.percentile(z, 1)),
        'p99': float(np.percentile(z, 99)),
        'p99_9': float(np.percentile(z, 99.9)),
        'min': float(np.min(z)),
        'max': float(np.max(z)),
    }
    # Anderson-Darling is more sensitive in the tails than Kolmogorov-Smirnov,
    # which is what matters here.
    ad = stats.anderson(z[rng.choice(len(z), size=min(len(z), 20000),
                                     replace=False)], dist='norm')
    m['anderson_darling_stat'] = float(ad.statistic)
    m['anderson_darling_crit_1pct'] = float(ad.critical_values[-1])
    m['normal_rejected_1pct'] = bool(ad.statistic > ad.critical_values[-1])
    return m


def qq_plot(z, out_png):
    """QQ plot against the normal, with the working thresholds marked."""
    zs = z[rng.choice(len(z), size=min(len(z), 20000), replace=False)]
    osm, osr = stats.probplot(zs, dist='norm', fit=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(osm, osr, '.', ms=2, alpha=0.5, color='#4575b4')
    lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
    ax.plot(lim, lim, 'r-', lw=1.5)
    for t in (2.5, 3.0):
        ax.axhline(t, color='grey', ls='--', lw=0.8)
        ax.axhline(-t, color='grey', ls='--', lw=0.8)
    ax.set_xlabel('Theoretical normal quantiles')
    ax.set_ylabel('Pre-war z-score (pseudo-z)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, facecolor='white')
    fig.savefig(out_png.replace('.png', '.pdf'), facecolor='white')
    plt.close(fig)


# =============================================================================
# B, D. EXCEEDANCE AND SPATIALLY BLOCKED BOOTSTRAP
# =============================================================================

def to_km(lon, lat):
    """Local equirectangular projection, adequate over a few hundred km."""
    lat0 = float(np.mean(lat))
    x = (lon - float(np.mean(lon))) * 111.32 * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * 110.57
    return x, y


def exceedance_table(z, x_km, y_km, block_km):
    """"""
    bx = np.floor((x_km - x_km.min()) / block_km).astype(int)
    by = np.floor((y_km - y_km.min()) / block_km).astype(int)
    block = bx * 100000 + by
    blocks = np.unique(block)
    idx_by_block = {b: np.where(block == b)[0] for b in blocks}
    print(f"    bootstrap blocks: {len(blocks)} of {block_km:.0f} km")

    boot = np.empty((N_BOOT, len(Z_GRID)))
    for i in range(N_BOOT):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([idx_by_block[b] for b in pick])
        zb = np.abs(z[idx])
        boot[i, :] = [np.mean(zb >= t) for t in Z_GRID]

    rows = []
    for j, t in enumerate(Z_GRID):
        nominal = 2 * stats.norm.sf(t)
        observed = float(np.mean(np.abs(z) >= t))
        rows.append({
            'z_threshold': t,
            'nominal_two_sided': nominal,
            'observed_two_sided': observed,
            'inflation_factor': observed / nominal if nominal > 0 else np.nan,
            'boot_lo_2p5': float(np.percentile(boot[:, j], 2.5)),
            'boot_hi_97p5': float(np.percentile(boot[:, j], 97.5)),
            'block_km': block_km,
            'n_blocks': len(blocks),
        })
    return pd.DataFrame(rows)


# =============================================================================
# C. EMPIRICAL RANGE OF THE Z FIELD
# =============================================================================

def _pairs_by_bin(z, x, y, rmax, width, n_sub=None):
    """Sum of products, squared differences and counts of standardised z for
    all point pairs of ONE pre-war year within rmax km, in bins of `width`."""
    if n_sub is not None and len(z) > n_sub:
        sel = rng.choice(len(z), size=n_sub, replace=False)
        z, x, y = z[sel], x[sel], y[sel]
    zz = (z - z.mean()) / z.std()
    pairs = cKDTree(np.column_stack([x, y])).query_pairs(rmax, output_type='ndarray')
    nb = int(round(rmax / width))
    out = np.zeros((3, nb))
    if len(pairs) == 0:
        return out
    d = np.hypot(x[pairs[:, 0]] - x[pairs[:, 1]], y[pairs[:, 0]] - y[pairs[:, 1]])
    b = np.minimum((d / width).astype(int), nb - 1)
    a, c = zz[pairs[:, 0]], zz[pairs[:, 1]]
    np.add.at(out[0], b, a * c)
    np.add.at(out[1], b, 0.5 * (a - c) ** 2)
    np.add.at(out[2], b, 1)
    return out


def correlogram(zs, zcol, x_km, y_km, out_png):
    """"""
    years = zs['null_year'].to_numpy() if 'null_year' in zs.columns else np.zeros(len(zs))
    z_all = zs[zcol].to_numpy(float)
    fine = np.zeros((3, int(CORR_FINE_MAX_KM / CORR_FINE_BIN_KM)))
    coarse = np.zeros((3, int(CORR_MAX_KM / CORR_COARSE_BIN_KM)))
    per_year = {}
    for yv in np.unique(years):
        m = years == yv
        f1 = _pairs_by_bin(z_all[m], x_km[m], y_km[m], CORR_FINE_MAX_KM, CORR_FINE_BIN_KM)
        c1 = _pairs_by_bin(z_all[m], x_km[m], y_km[m], CORR_MAX_KM, CORR_COARSE_BIN_KM,
                           n_sub=CORR_COARSE_N)
        fine += f1
        coarse += c1
        per_year[yv] = (f1, c1)
    rows = []
    for i in range(fine.shape[1]):
        if fine[2, i] > 0:
            row = {'dist_lo_km': i * CORR_FINE_BIN_KM, 'dist_hi_km': (i + 1) * CORR_FINE_BIN_KM,
                   'n_pairs': int(fine[2, i]), 'correlation': fine[0, i] / fine[2, i],
                   'semivariance': fine[1, i] / fine[2, i]}
            for yv, (f1, _) in per_year.items():
                row[f'correlation_{yv}'] = f1[0, i] / f1[2, i] if f1[2, i] > 0 else np.nan
            rows.append(row)
    first_coarse = int(CORR_FINE_MAX_KM / CORR_COARSE_BIN_KM)
    for i in range(first_coarse, coarse.shape[1]):
        if coarse[2, i] > 0:
            row = {'dist_lo_km': i * CORR_COARSE_BIN_KM, 'dist_hi_km': (i + 1) * CORR_COARSE_BIN_KM,
                   'n_pairs': int(coarse[2, i]), 'correlation': coarse[0, i] / coarse[2, i],
                   'semivariance': coarse[1, i] / coarse[2, i]}
            for yv, (_, c1) in per_year.items():
                row[f'correlation_{yv}'] = c1[0, i] / c1[2, i] if c1[2, i] > 0 else np.nan
            rows.append(row)
    cg = pd.DataFrame(rows)
    cg['dist_mid_km'] = 0.5 * (cg.dist_lo_km + cg.dist_hi_km)
    low = (cg.correlation <= RANGE_CORR).to_numpy()
    hit = np.where(low[:-1] & low[1:])[0]
    range_km = float(cg.dist_hi_km.iloc[hit[0]]) if len(hit) else np.nan

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(cg.dist_mid_km, cg.correlation, 'o-', ms=3, color='#4575b4')
    ax.axhline(RANGE_CORR, color='grey', ls='--', lw=0.8)
    ax.axhline(0, color='#444444', lw=0.6)
    if np.isfinite(range_km):
        ax.axvline(range_km, color='#d73027', ls='--', lw=1.0)
        ax.text(range_km, 0.9, f'  range {range_km:.0f} km', color='#d73027', fontsize=9)
    ax.set_xlabel('Distance between pixels of the same pre-war year (km)')
    ax.set_ylabel('Correlation of pre-war z-scores')
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, facecolor='white')
    fig.savefig(out_png.replace('.png', '.pdf'), facecolor='white')
    plt.close(fig)
    return cg, range_km


# =============================================================================
# E-G. PRE-WAR BENCHMARKS, STRICTER RULES ON THE PRE-WAR MAPS, SENSITIVITY
# =============================================================================

def benchmark_table(excess, headline=None, strict_null=None):
    """
    Pre-war benchmark ratio: benchmark area (share of pre-war leave-one-out
    z-scores at or beyond the threshold in that direction x analysed area)
    divided by the detected area, per comparison, threshold and direction,
    from Areas_Excess_by_Zclass. When the tables are present, the stricter
    version is added with its benchmark measured on the pre-war maps (mean
    over the three pre-war years). A ratio is a reference, not a false
    discovery rate; a ratio above one means no excess over the benchmark. A
    zero detected area gives NA with status 'no_detections'.
    """
    rows = []
    for _, r in excess.iterrows():
        for side in ('loss', 'gain'):
            d = r[f'observed_{side}_km2']
            e = r[f'prewar_benchmark_{side}_km2']
            ratio, status = safe_ratio(e, d)
            rows.append({'comparison': r['comparison'], 'rule': 'pixel exceedance',
                         'z_threshold': float(r['z_class']), 'direction': side,
                         'detected_km2': d, 'prewar_benchmark_km2': e,
                         'benchmark_ratio': ratio, 'status': status})
    if headline is not None and strict_null is not None:
        h = headline.set_index('variant')
        for variant, key in (('primary_all_valid_input_pixels', ''),
                             ('quality_screen_with_novelty_sensitivity', '_screened')):
            if variant not in h.index:
                continue
            for side, det in (('loss', 'damage_km2'), ('gain', 'rewild_km2')):
                d = h.loc[variant, det]
                e = float(strict_null[f'stricter_{side}{key}_km2'].mean())
                ratio, status = safe_ratio(e, d)
                rows.append({'comparison': 'median_headline',
                             'rule': 'stricter version' + (' (all three screens)' if key else ''),
                             'z_threshold': float(strict_null['z_threshold'].iloc[0]),
                             'direction': side, 'detected_km2': d, 'prewar_benchmark_km2': e,
                             'benchmark_ratio': ratio, 'status': status})
    return pd.DataFrame(rows)


def stricter_prewar_summary(strict_null):
    """What the neighbour, closing and patch rules do to the pre-war
    exceedances (|z| at or beyond the stricter threshold), per pre-war year
    and on average (GEE_05 Areas_Stricter_Null_v13b12).

    postfilter_to_exceedance_area_ratio_<side> = result area / exceedance
        area. The closing can add cells, so this is an AREA RATIO, not the
        share of exceedances kept.
    retained_share_<side> = (result AND exceedance) / exceedance.
    added_km2 / removed_km2 = result cells that were not exceedances /
        exceedances not in the result; no_closing_km2 = the rules without
        the closing (a subset of the exceedances).
    """
    t = strict_null.copy()
    years = sorted(pd.to_numeric(t['null_year'], errors='coerce').dropna().astype(int).tolist())
    if years != [2019, 2020, 2021]:
        raise ValueError(f"Areas_Stricter_Null: expected one row for each of 2019, 2020, 2021; "
                         f"found {years}. The mean would not be a three-year mean.")
    num = t.drop(columns=['null_year']).select_dtypes('number')
    if not np.isfinite(num.to_numpy(float)).all():
        raise ValueError("Areas_Stricter_Null: missing or non-finite values; a mean would skip them.")
    t['null_year'] = t['null_year'].astype(str)
    mean = t.select_dtypes('number').mean().to_dict()
    mean['null_year'] = 'mean'
    t = pd.concat([t, pd.DataFrame([mean])], ignore_index=True)
    for side in ('loss', 'gain'):
        ratio, retained, st = [], [], []
        for _, r in t.iterrows():
            a, s1 = safe_ratio(r[f'stricter_{side}_km2'], r[f'pixel_{side}_km2'])
            b, _ = safe_ratio(r[f'stricter_{side}_in_exceedance_km2'], r[f'pixel_{side}_km2'])
            ratio.append(a)
            retained.append(b)
            st.append('ok' if s1 == 'ok' else ('no_exceedances' if s1 == 'no_detections' else s1))
        t[f'postfilter_to_exceedance_area_ratio_{side}'] = ratio
        t[f'retained_share_{side}'] = retained
        t[f'status_{side}'] = st
    return t


def sensitivity_summary(frames):
    """Long table of the Sensitivity_*_v13b12 exports with the ratio of each
    area to the main-parameter area and the Jaccard overlap (NA + status when
    undefined)."""
    s = pd.concat(frames, ignore_index=True, sort=False)
    ratio = s.apply(lambda r: safe_ratio(r.variant_km2, r.main_km2), axis=1)
    jac = s.apply(lambda r: jaccard(r.variant_km2, r.main_km2, r.overlap_with_main_km2), axis=1)
    s['ratio_to_main'] = [x[0] for x in ratio]
    s['ratio_status'] = [x[1] for x in ratio]
    s['jaccard_with_main'] = [x[0] for x in jac]
    s['jaccard_status'] = [x[1] for x in jac]
    return s


# =============================================================================
# MAIN
# =============================================================================

def run_scheme(scheme_label):
    print("=" * 74)
    print(f"EMPIRICAL NULL DIAGNOSTICS -- {scheme_label}")
    print("=" * 74)

    zs = _read("ControlZ_Sample.csv")
    zcol = 'zscore_median' if 'zscore_median' in zs.columns else zs.columns[0]
    zs = zs[np.isfinite(zs[zcol]) & np.isfinite(zs['lon']) & np.isfinite(zs['lat'])]
    z = zs[zcol].to_numpy(dtype=float)
    x_km, y_km = to_km(zs['lon'].to_numpy(float), zs['lat'].to_numpy(float))
    print(f"\n[A] Pre-war pseudo-z sample: {len(z):,} pixels")

    # ---- A. Shape --------------------------------------------------------
    moments = null_shape(z)
    pd.DataFrame([moments]).to_csv(
        os.path.join(OUT_DIR, "controlz_moments.csv"), index=False)
    for k, v in moments.items():
        print(f"    {k:28s}: {v}")
    qq_plot(z, os.path.join(OUT_DIR, "controlz_qqplot.png"))
    print("    QQ plot written.")

    # Consume the full-population histogram as a check that the random sample
    # used for QQ/spatial diagnostics represents the full control-zone raster.
    histogram = _read("ControlZ_Histogram.csv", required=False)
    if histogram is not None and {"bin_centre", "count"}.issubset(histogram.columns):
        histogram["bin_centre"] = pd.to_numeric(
            histogram["bin_centre"], errors="coerce")
        histogram["count"] = pd.to_numeric(histogram["count"], errors="coerce")
        histogram = histogram.dropna(subset=["bin_centre", "count"])
        histogram = histogram[histogram["count"] > 0]
        full_n = float(histogram["count"].sum())
        full_mean = float(np.average(
            histogram["bin_centre"], weights=histogram["count"]))
        full_var = float(np.average(
            (histogram["bin_centre"] - full_mean) ** 2,
            weights=histogram["count"]))
        histogram_check = pd.DataFrame([{
            "full_histogram_n": full_n,
            "full_histogram_mean_approx": full_mean,
            "full_histogram_sd_approx": np.sqrt(full_var),
            "sample_n": len(z),
            "sample_mean": float(np.mean(z)),
            "sample_sd": float(np.std(z, ddof=1)),
            "sample_minus_histogram_mean": float(np.mean(z) - full_mean),
        }])
        histogram_check.to_csv(
            os.path.join(OUT_DIR, "controlz_histogram_sample_check.csv"),
            index=False)
        print("    Full histogram consumed; sample-representativeness check written.")

    # ---- C. Spatial correlation (needed before the bootstrap block size) --
    print("\n[C] Correlogram of the pre-war z field (pairs within each year)...")
    cg, range_km = correlogram(zs, zcol, x_km, y_km,
                               os.path.join(OUT_DIR, "controlz_correlogram.png"))
    cg.to_csv(os.path.join(OUT_DIR, "controlz_correlogram.csv"), index=False)
    print(cg[cg.dist_hi_km.isin([1, 2, 5, 10, 20, 40, 60, 100, 150])]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"    range (first two consecutive bins with correlation <= {RANGE_CORR}): "
          f"{range_km} km (read it together with the curve)")

    # Semivariogram coordinates are converted from longitude/latitude to
    # approximate ground kilometres. GEE_05 is recomputed on EPSG:3035, so the
    # configured metric values can be compared directly at this precision.
    scales = _read("Analysis_Grid_and_Scale_Parameters.csv", required=False)
    if scales is not None:
        required = {
            "parameter", "spatial_kind", "configured_unit",
            "value_as_configured_on_analysis_grid",
            "ground_equivalent_centroid",
        }
        missing = required.difference(scales.columns)
        if missing:
            raise ValueError(
                "Analysis_Grid_and_Scale_Parameters.csv is missing columns: "
                f"{sorted(missing)}")
        linear = scales[scales["spatial_kind"] == "linear"].copy()
        linear["ground_equivalent_centroid"] = pd.to_numeric(
            linear["ground_equivalent_centroid"], errors="coerce")
        linear["ground_equivalent_centroid_km"] = np.where(
            linear["configured_unit"] == "m",
            linear["ground_equivalent_centroid"] / 1000.0,
            linear["ground_equivalent_centroid"],
        )
        linear["correlogram_range_ground_km"] = range_km
        linear["range_to_parameter_ratio"] = (
            range_km / linear["ground_equivalent_centroid_km"].replace(0, np.nan)
        )
        linear.to_csv(
            os.path.join(OUT_DIR, "controlz_range_vs_ground_scales.csv"),
            index=False,
        )
        print("    Ground-scale comparison table written.")

    # ---- B, D. Exceedance with blocked bootstrap -------------------------
    print("\n[B/D] Exceedance rates with spatially blocked bootstrap...")
    block_km = max(range_km, 10) if np.isfinite(range_km) else CORR_MAX_KM
    exceed = exceedance_table(z, x_km, y_km, block_km)
    exceed.to_csv(os.path.join(OUT_DIR, "controlz_exceedance.csv"), index=False)
    print(exceed.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # ---- E/F. Pre-war benchmark ratios and the stricter rules ---------------
    print("\n[E] Pre-war benchmark ratios...")
    excess = _read("Areas_Excess_by_Zclass.csv", required=False)
    headline = _read("Areas_Headline_annualbase_annual_median.csv", required=False)
    strict_null = _read(f"Areas_Stricter_Null_{CODE_VERSION}.csv", required=False)
    for name, tab in (("Areas_Excess_by_Zclass.csv", excess),
                      ("Areas_Headline_annualbase_annual_median.csv", headline)):
        if tab is not None and ('code_version' not in tab.columns or
                                not (tab['code_version'] == CODE_VERSION).all()):
            raise ValueError(f"{name} is not a GEE_05 {CODE_VERSION} export: rerun "
                             f"GEE_05 {CODE_VERSION} and download its outputs.")
    if excess is not None:
        bt = benchmark_table(excess, headline, strict_null)
        bt.to_csv(os.path.join(OUT_DIR, "prewar_benchmark_ratios.csv"), index=False)
        print(bt[bt.comparison == 'median_headline'].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
    if strict_null is not None:
        print("\n[F] Stricter rules applied to the pre-war maps (GEE_05):")
        sn = stricter_prewar_summary(strict_null)
        sn.to_csv(os.path.join(OUT_DIR, "stricter_version_prewar_summary.csv"), index=False)
        print(sn.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    paths = sorted(glob.glob(os.path.join(IN_DIR, f"Sensitivity_*_{CODE_VERSION}.csv")))
    if paths:
        frames = []
        for path in paths:
            frame = pd.read_csv(path)
            frame["source_file"] = os.path.basename(path)
            frames.append(frame)
        ss = sensitivity_summary(frames)
        if (ss.jaccard_status == 'invalid_areas').any():
            print("    WARNING: impossible areas (negative, or overlap larger than an area) in "
                  f"{int((ss.jaccard_status == 'invalid_areas').sum())} sensitivity rows; "
                  "Python_03 will reject them.")
        ss.to_csv(os.path.join(OUT_DIR, "change_detection_sensitivity_summary.csv"),
                  index=False)
        print("\n[G] Sensitivity summary (area, ratio to main, Jaccard):")
        show = ss[ss.statistic.isin(["median_loss", "year_2024_loss", "year_2022_gain",
                                     "union_loss_ge1_mapped", "stricter_loss"])]
        print(show[["family", "variant", "design", "statistic", "variant_km2",
                    "ratio_to_main", "jaccard_with_main"]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
    else:
        print("\n[G] No Sensitivity_*_v13b12.csv tables found.")

    # ---- Reporting note ---------------------------------------------------
    note = (
        "HOW THESE DIAGNOSTICS RELATE TO THE MANUSCRIPT (v13b12)\n"
        "-------------------------------------------------------\n"
        "The pre-war leave-one-out z-scores describe pre-war variability.\n"
        "Their mean, SD, tails and spatial correlation (within years) are\n"
        "reported as properties of that reference. Areas derived from them\n"
        "(prewar_benchmark_ratios.csv, stricter_version_prewar_summary.csv)\n"
        "are benchmarks, not calibrated no-war expectations, and their ratios\n"
        "to detected areas are not false discovery rates. The closing of the\n"
        "stricter version can add cells, so the stricter/exceedance area ratio\n"
        "is not a retained share (retained_share_* is).\n"
    )
    with open(os.path.join(OUT_DIR, "REPORTING_NOTE.txt"), "w", encoding="utf-8") as fh:
        fh.write(note)
    print("\n" + note)
    print(f"All outputs written to {OUT_DIR}")


def main():
    global IN_DIR, OUT_DIR, rng
    ran = 0
    for scheme in ANALYSIS_SCHEMES:
        candidate = str(folders.data_dir(folders.GEE05[scheme]))
        if not os.path.isdir(candidate):
            print(f"[skip] {SCHEME_LABELS[scheme]}: folder not found "
                  f"({candidate})")
            continue
        IN_DIR = candidate
        OUT_DIR = os.path.join(candidate, "null_diagnostics")
        os.makedirs(OUT_DIR, exist_ok=True)
        rng = np.random.default_rng(SEED)
        run_scheme(SCHEME_LABELS[scheme])
        ran += 1
    if ran == 0:
        raise FileNotFoundError(
            "No GEE_05 output folder found. Expected "
            + " or ".join(str(folders.GEE05[sch]) for sch in ANALYSIS_SCHEMES))


if __name__ == "__main__":
    main()
