#!/usr/bin/env python
# coding: utf-8
"""
=============================================================================
 FIGURE 8 AND SUPPORTING SCATTERPLOTS
 Manuscript SUST-D-26-00527 (Sustainability Science)
=============================================================================

 Figure 8 shows the Spearman correlations between documented war events
 (VIINA, ACLED) and six anomaly metrics per H3 resolution-6 hexagon. All
 coefficients, intervals and sample sizes are read from
 spearman_correlations.csv (support/Python_correlations_with_events.py);
 nothing is recalculated here.

 LAYOUT (v52, presentation only)
 -------------------------------
 - One panel per event source (VIINA above ACLED); rows = all hexagons,
   hexagons outside and inside the pre-2022 occupation line, and the majority
   land-cover groups (cropland, grassland, tree cover), as in Table S7.
 - Loss columns (median intensity, maximum intensity, extent) left of the
   gain columns, separated by a white line and headed 'Loss' and 'Gain'.
 - The number of hexagons once per row, in the row label, when it is the same
   for all six metrics; otherwise (the 'active' variant) in each cell.
 - Each cell gives the coefficient and its spatially blocked 95% bootstrap
   interval, all in regular weight: no bold, no 'ci' corner labels and no
   subgroup triangles. The interval itself shows whether it includes zero,
   and the row n shows how many hexagons a row rests on.
 - INCLUDE_ORDLO_ROWS = True (default) draws the rows inside and outside the
   pre-2022 occupation line; set it to False to leave them to Table S7.

"""

import argparse
import re
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from scipy.stats import spearmanr
from pathlib import Path

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

# =============================================================================
# CONFIGURATION
# =============================================================================

# One hexagon-analysis folder per visitation model (written by the
# correlations script; support/folders.py); Figure 8 is produced once per scheme.
ANALYSIS_SCHEMES = ["recurrent_effort", "primary_effort"]
SCHEME_LABELS = {
    "recurrent_effort":
        "Recurrent outdoor visitation model (2019-2021) [comparison]",
    "primary_effort": "Regular outdoor visitation model (2019-2021) [main]",
}
DATA_DIR: Path = None   # set per scheme in main()
HEX_CSV: Path = None
CORR_CSV: Path = None
OUT_DIR: Path = None
CURRENT_SCHEME_LABEL = ""

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import folders  # noqa: E402  (support/folders.py)

FIG_WIDTH_MM = 234
FIG_DPI      = 900
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4

MIN_N = 30

# Must match the correlations script.
SIGN_CONVENTION        = 'magnitude'
FEASIBILITY_CHECK_SOURCE = 'ACLED'
HEADLINE_SOURCE          = 'VIINA'   # source quoted in the draft caption

# One panel per source, top to bottom.
PLOT_EVENT_SOURCES = ['VIINA', 'ACLED']

# Rows of each panel: (stratum, land_cover, label), land-cover groups by size.
ROWS = [('All', 'All', 'All hexagons'),
        ('All', 'Cropland', 'Cropland'),
        ('All', 'Grassland', 'Grassland'),
        ('All', 'Tree cover', 'Tree cover')]
INCLUDE_ORDLO_ROWS = True
ORDLO_ROWS = [('Outside ORDLO', 'All', 'Outside the pre-2022\noccupation line'),
              ('Inside ORDLO', 'All', 'Inside the pre-2022\noccupation line')]

# Loss columns first, then gain columns (internal metric keys of the CSV).
HEATMAP_COLS = ['Median loss', 'Max loss', 'Loss pixels',
                'Median gain', 'Max gain', 'Gain pixels']
COLUMN_LABELS = ['Median\nintensity', 'Maximum\nintensity', 'Extent\n(cells)',
                 'Median\nintensity', 'Maximum\nintensity', 'Extent\n(cells)']

# (7.5 here: the true minus sign is wider than the hyphen used before).
RHO_FONTSIZE = 10
CI_FONTSIZE  = 7.5
ROW_H_IN     = 0.55     # height of one heatmap row (inches)

CLR_DAMAGE = '#d73027'
CLR_REWILD = '#4575b4'

# Column names follow the sign convention chosen in the correlations script.
if SIGN_CONVENTION == 'magnitude':
    SCATTER_METRICS = [
        ('a', 'Max loss',         'mag_max_damage',    CLR_DAMAGE, 'z'),
        ('b', 'Max gain',     'mag_max_rewild',    CLR_REWILD, 'z'),
        ('c', 'Loss pixels',      'conf_damage_px',    CLR_DAMAGE, 'px'),
        ('d', 'Gain pixels',  'conf_rewild_px',    CLR_REWILD, 'px'),
    ]
else:
    SCATTER_METRICS = [
        ('a', 'Max loss',         'z_max_damage',      CLR_DAMAGE, 'z'),
        ('b', 'Max gain',     'z_max_rewild',      CLR_REWILD, 'z'),
        ('c', 'Loss pixels',      'conf_damage_px',    CLR_DAMAGE, 'px'),
        ('d', 'Gain pixels',  'conf_rewild_px',    CLR_REWILD, 'px'),
    ]

Z_COL_NAMES  = {'Median loss', 'Median gain', 'Max loss', 'Max gain'}
PX_COL_NAMES = {'Loss pixels', 'Gain pixels'}

ESA_LEGEND = {
    10: 'Tree cover', 20: 'Shrubland', 30: 'Grassland',
    40: 'Cropland', 50: 'Built-up', 60: 'Bare / sparse vegetation',
    70: 'Snow and ice', 80: 'Water bodies',
    90: 'Herbaceous wetland', 95: 'Mangroves',
    100: 'Moss and lichen',
}


# =============================================================================
# SETUP AND HELPERS
# =============================================================================

def setup_font():
    """Matplotlib default font; only the sizes and resolution are set."""
    plt.rcParams.update({'axes.titlesize': 10, 'axes.labelsize': 10,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.dpi': FIG_DPI})


def sig_label(pval):
    """Stars for the ordinary p-value of the supporting scatterplots only."""
    if not np.isfinite(pval):
        return 'ns'
    if pval < 0.001: return '***'
    if pval < 0.01:  return '**'
    if pval < 0.05:  return '*'
    return 'ns'


def number(x):
    """Two decimals with a true minus sign, no '-0.00' (as in Table S7)."""
    s = f'{x:.2f}'
    if s == '-0.00':
        s = '0.00'
    return s.replace('-', '\u2212')


def save_fig(fig, stem, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{stem}.png', dpi=FIG_DPI,
                bbox_inches='tight', facecolor='white')
    fig.savefig(out_dir / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"      -> {stem}.png / {stem}.pdf")


def safe_slug(name):
    return name.lower().replace(' ', '_').replace('/', '_')


def get_mask(hexdf, ev_col, subset, lc_code=None):
    """Replicates the subset logic of the correlations script."""
    base = (hexdf['z_count'] > 0) & (hexdf['conf_valid_px'] > 0)
    if 'novel_frac' in hexdf.columns:
        base = base & (hexdf['novel_frac'].fillna(0) <= 0.50)
    if subset in ('z_active', 'px_active'):
        base = base & (hexdf[ev_col] > 0)
    if subset == 'px_active':
        base = base & ((hexdf['conf_damage_px'] > 0) | (hexdf['conf_rewild_px'] > 0))
    if lc_code is not None:
        base = base & (hexdf['lc_code'] == lc_code)
    return base


def bootstrap_settings():
    """(replicates, block size in km) as set in the correlations script, or
    (None, None) if that script is not beside this one."""
    p = _SUPPORT / 'Python_correlations_with_events.py'
    if not p.is_file():
        return None, None
    src = p.read_text(encoding='utf-8', errors='ignore')
    nb = re.search(r'^N_BOOT\s*=\s*(\d+)', src, re.M)
    bk = re.search(r'^BOOT_BLOCK_KM\s*=\s*([\d.]+)', src, re.M)
    return (int(nb.group(1)) if nb else None,
            float(bk.group(1)) if bk else None)


# =============================================================================
# HEATMAP
# =============================================================================

def heatmap_table(corr_sub):
    """Checked long table of the cells to plot (source, row, metric)."""
    need = {'event_source', 'stratum', 'land_cover', 'metric', 'n',
            'spearman_rho', 'rho_ci_lo', 'rho_ci_hi'}
    missing = need - set(corr_sub.columns)
    if missing:
        raise ValueError(f'{CORR_CSV}: missing columns {sorted(missing)}')
    if 'sign_convention' in corr_sub.columns and \
            not corr_sub['sign_convention'].eq(SIGN_CONVENTION).all():
        raise ValueError(f'Figure 8 expects the {SIGN_CONVENTION!r} sign convention '
                         'of the correlations script.')
    # Rows inside / outside the occupation line follow 'All hexagons', as in Table S7.
    rows = ROWS[:1] + (ORDLO_ROWS if INCLUDE_ORDLO_ROWS else []) + ROWS[1:]
    keep = pd.Series(False, index=corr_sub.index)
    for stratum, lc, _ in rows:
        keep |= (corr_sub['stratum'] == stratum) & (corr_sub['land_cover'] == lc)
    t = corr_sub[keep & corr_sub['event_source'].isin(PLOT_EVENT_SOURCES)
                 & corr_sub['metric'].isin(HEATMAP_COLS)].copy()
    key = ['event_source', 'stratum', 'land_cover', 'metric']
    if t.duplicated(key).any():
        raise ValueError(f'{CORR_CSV}: duplicate Figure 8 entries; use the output of a single run.')
    for _, r in t.iterrows():
        rho, lo, hi = float(r.spearman_rho), float(r.rho_ci_lo), float(r.rho_ci_hi)
        if np.isfinite(rho) and np.isfinite(lo) and np.isfinite(hi) and not (-1 <= lo <= hi <= 1):
            raise ValueError(f'Invalid interval: {r.event_source}, {r.stratum}, {r.land_cover}, {r.metric}')
    t['row_label'] = ''
    for stratum, lc, label in rows:
        t.loc[(t['stratum'] == stratum) & (t['land_cover'] == lc), 'row_label'] = label
    t['interval_excludes_zero'] = (t['rho_ci_lo'] > 0) | (t['rho_ci_hi'] < 0)
    return t, [r for r in rows if (t['row_label'] == r[2]).any()]


def make_heatmap(corr_sub, save_stem, out_dir):
    if corr_sub.empty:
        return
    t, rows = heatmap_table(corr_sub)
    sources = [s for s in PLOT_EVENT_SOURCES if (t['event_source'] == s).any()]
    if not sources or not rows:
        return
    idx = t.set_index(['event_source', 'row_label', 'metric'])

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    n_rows = len(rows)
    title_in, bottom_in = 0.62, 0.75
    fig_h = len(sources) * (n_rows * ROW_H_IN + title_in) + bottom_in
    fig, axes = plt.subplots(len(sources), 1, figsize=(FIG_WIDTH_IN, fig_h),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    left, right = 0.16, 0.88
    fig.subplots_adjust(left=left, right=right, top=1 - title_in / fig_h,
                        bottom=bottom_in / fig_h,
                        hspace=title_in / (n_rows * ROW_H_IN))

    for ax, src in zip(axes, sources):
        rho = np.full((n_rows, len(HEATMAP_COLS)), np.nan)
        for i, (_, _, label) in enumerate(rows):
            for j, m in enumerate(HEATMAP_COLS):
                if (src, label, m) in idx.index:
                    rho[i, j] = idx.loc[(src, label, m), 'spearman_rho']
        im = ax.imshow(rho, cmap=cmap, norm=norm, aspect='auto')

        ylabels = []
        for i, (_, _, label) in enumerate(rows):
            ns = [int(idx.loc[(src, label, m), 'n']) for m in HEATMAP_COLS
                  if (src, label, m) in idx.index]
            same_n = len(set(ns)) == 1
            ylabels.append(f'{label}\n(n = {ns[0]:,})' if same_n else label)
            for j, m in enumerate(HEATMAP_COLS):
                if (src, label, m) not in idx.index or not np.isfinite(rho[i, j]):
                    ax.text(j, i, '\u2013', ha='center', va='center', fontsize=RHO_FONTSIZE,
                            color='grey')
                    continue
                r = idx.loc[(src, label, m)]
                bg = cmap(norm(rho[i, j]))
                lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
                tc = 'white' if lum < 0.45 else 'black'
                has_ci = np.isfinite(r.rho_ci_lo) and np.isfinite(r.rho_ci_hi)
                y_rho, y_ci, y_n = (i - 0.14, i + 0.20, None) if same_n else (i - 0.24, i + 0.27, i + 0.02)
                ax.text(j, y_rho, number(rho[i, j]), ha='center', va='center',
                        fontsize=RHO_FONTSIZE, color=tc)
                if has_ci:
                    ax.text(j, y_ci, f'[{number(r.rho_ci_lo)}, {number(r.rho_ci_hi)}]',
                            ha='center', va='center', fontsize=CI_FONTSIZE, color=tc)
                if y_n is not None:
                    ax.text(j, y_n, f'(n = {int(r.n):,})', ha='center', va='center',
                            fontsize=CI_FONTSIZE, color=tc)

        ax.axvline(2.5, color='white', lw=3)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.tick_params(length=0, pad=6)
        ax.set_title(src, loc='left', fontsize=10, pad=20)
        ax.text(0.25, 1.02, 'Loss', transform=ax.transAxes, ha='center', va='bottom',
                fontsize=10)
        ax.text(0.75, 1.02, 'Gain', transform=ax.transAxes, ha='center', va='bottom',
                fontsize=10)

    axes[-1].set_xticks(range(len(HEATMAP_COLS)))
    axes[-1].set_xticklabels(COLUMN_LABELS, fontsize=9)

    # Colour bar in its own axes so that the heatmap columns keep their width.
    y0, y1 = axes[-1].get_position().y0, axes[0].get_position().y1
    h = 0.8 * (y1 - y0)
    cax = fig.add_axes([right + 0.02, y0 + 0.5 * (y1 - y0 - h), 0.016, h])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Spearman's rho", fontsize=12)
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    save_fig(fig, save_stem, out_dir)
    t.drop(columns=['row_label']).to_csv(out_dir / f'{save_stem}_values.csv', index=False)
    print(f"      -> {save_stem}_values.csv ({len(t)} coefficients, unchanged from {CORR_CSV.name})")


# =============================================================================
# 2x2 SCATTER
# =============================================================================

def make_scatter_2x2(hexdf, ev_col, ev_label, mode,
                     lc_code=None, save_stem=None, out_dir=None):
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.82))

    for idx, (panel_lbl, metric_name, m_col, color, mtype) in enumerate(SCATTER_METRICS):
        ax = axes[idx // 2, idx % 2]
        subset_key = 'all_valid' if mode == 'all_valid' else (
            'z_active' if mtype == 'z' else 'px_active')
        sub = hexdf[get_mask(hexdf, ev_col, subset_key, lc_code)]

        ax.text(0.03, 0.97, panel_lbl, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left')
        ax.set_xlabel(f'log10({ev_label} events + 1)', fontsize=8)
        ax.set_ylabel(metric_name, fontsize=8)

        if len(sub) < 5 or m_col not in sub.columns:
            ax.text(0.5, 0.5, 'n < 5', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='grey')
            continue

        x = np.log10(sub[ev_col].values + 1)
        y = sub[m_col].values
        rho, pval = spearmanr(sub[ev_col].values, y)

        ax.scatter(x, y, c=color, alpha=0.25, s=10,
                   edgecolors='none', rasterized=True)
        if len(sub) > 10:
            coeffs = np.polyfit(x, y, 1)
            xf = np.linspace(x.min(), x.max(), 100)
            ax.plot(xf, np.polyval(coeffs, xf), 'k-', lw=1.2, alpha=0.7)

        p_str = 'p < 0.001' if pval < 0.001 else f'p = {pval:.3f}'
        txt = f'rho = {rho:+.3f} {sig_label(pval)}\n{p_str}\nn = {len(sub)}'
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=10,
                va='top', ha='right', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='grey', alpha=0.85))

    fig.tight_layout(pad=1.5)
    save_fig(fig, save_stem, out_dir)


# =============================================================================
# CAPTION
# =============================================================================

def write_caption(corr_df, out_dir):
    """Draft caption in the wording of the manuscript (Figure 8, Table S7)."""
    allv = corr_df[(corr_df['subset'] == 'all_valid') &
                   (corr_df['event_source'] == HEADLINE_SOURCE) &
                   (corr_df['stratum'] == 'All')]
    n_all = allv.loc[allv['land_cover'] == 'All', 'n']
    n_all = int(n_all.iloc[0]) if len(n_all) else None
    grass = allv.loc[allv['land_cover'] == 'Grassland', 'n']
    grass_n = int(grass.iloc[0]) if len(grass) else None
    n_boot, block_km = bootstrap_settings()
    boot = (f' ({block_km:g} km blocks, {n_boot} replicates)'
            if n_boot and block_km else '')

    if SIGN_CONVENTION == 'magnitude':
        sign_sentence = (
            "Positive correlations indicate that event density increases with the "
            "intensity or extent of the named anomaly.")
    else:
        sign_sentence = (
            "Visitation-loss metrics are positive z-scores and visitation-gain "
            "intensity metrics are negative z-scores, so for the two gain-intensity "
            "columns a negative coefficient means stronger gain anomalies.")

    caption = (
        "Figure 8. Spearman correlations between documented war events (VIINA and "
        "ACLED) and six anomaly metrics calculated for H3 resolution-6 hexagons. Loss "
        "intensity is the median and maximum of the positive z-scores of the median "
        "comparison and gain intensity the median and maximum magnitude of its "
        "negative z-scores, both including values below the detection threshold; "
        "extent is the number of cells assigned to loss or gain in Figure 7. "
        + ("Rows show all hexagons, hexagons outside and inside the pre-2022 "
           "occupation line, and hexagons by majority land cover; "
           if INCLUDE_ORDLO_ROWS else
           "Rows show all hexagons and hexagons by majority land cover; ")
        + "n is the number of hexagons in each row. "
        f"{sign_sentence} Brackets show spatially blocked 95% bootstrap confidence "
        f"intervals{boot}. The "
        "intervals are not adjusted for the number of comparisons. Hexagons in which "
        "more than half of the cells had embeddings outside the range of the training "
        "data are excluded" + (f" (n = {n_all:,})." if n_all else "."))

    lines = ["DRAFT CAPTION FOR FIGURE 8", "=" * 60, "",
             f"Analysis: {CURRENT_SCHEME_LABEL}.", "", caption]
    if grass_n:
        lines += ["",
                  f"Note for the text: the grassland coefficients rest on {grass_n} "
                  f"hexagons and are subgroup results. State that n wherever a value "
                  f"is quoted."]
    text = "\n".join(lines) + "\n"
    (out_dir / 'figure8_caption.txt').write_text(text, encoding='utf-8')
    print("\n" + text)


# =============================================================================
# MAIN
# =============================================================================

def run_scheme():
    print("=" * 74)
    print(f"FIGURE 8 PLOTTING -- {CURRENT_SCHEME_LABEL}")
    print("=" * 74)

    print("\n[0] Font setup...")
    setup_font()

    print("\n[1] Loading data...")
    corr_df = pd.read_csv(CORR_CSV)
    hexdf = pd.read_csv(HEX_CSV)
    print(f"    Correlations: {len(corr_df)} rows | Hexagons: {len(hexdf)} rows")
    if 'stratum' not in corr_df.columns:
        corr_df['stratum'] = 'All'

    print("\n[2] Heatmaps...")
    av_dir = OUT_DIR / 'all_valid'
    act_dir = OUT_DIR / 'active'

    print("  all_valid:")
    make_heatmap(corr_df[corr_df['subset'] == 'all_valid'], 'figure8_all_valid', av_dir)

    print("  active (z_active + px_active):")
    z_act = corr_df[(corr_df['subset'] == 'z_active') & (corr_df['metric'].isin(Z_COL_NAMES))]
    px_act = corr_df[(corr_df['subset'] == 'px_active') & (corr_df['metric'].isin(PX_COL_NAMES))]
    make_heatmap(pd.concat([z_act, px_act], ignore_index=True), 'figure8_active', act_dir)

    print("\n[3] Scatterplots...")
    event_cols = [(s, {'ACLED': 'n_acled', 'VIINA': 'n_viina',
                       'Combined': 'n_combined'}[s]) for s in PLOT_EVENT_SOURCES]

    for mode, mode_dir in [('all_valid', av_dir), ('active', act_dir)]:
        gate = 'all_valid' if mode == 'all_valid' else 'z_active'
        for ev_label, ev_col in event_cols:
            ev_tag = ev_label.lower()
            make_scatter_2x2(hexdf, ev_col, ev_label, mode,
                             save_stem=f'scatter_{ev_tag}_all', out_dir=mode_dir)
            for lc_code, lc_name in sorted(ESA_LEGEND.items()):
                n_lc = get_mask(hexdf, ev_col, gate, lc_code=lc_code).sum()
                if n_lc < MIN_N:
                    continue
                make_scatter_2x2(hexdf, ev_col, ev_label, mode, lc_code=lc_code,
                                 save_stem=f'scatter_{ev_tag}_{safe_slug(lc_name)}',
                                 out_dir=mode_dir)

    print("\n[4] Caption...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_caption(corr_df, OUT_DIR)

    print("\n" + "=" * 74)
    print(f"Done. Figures in: {OUT_DIR} ({FIG_DPI} dpi, {FIG_WIDTH_MM} mm wide)")
    print("=" * 74)


def main():
    """Both models from support/folders.py, or one hexagon-analysis folder given
    with --folder (it must hold spearman_correlations.csv and hexagon_stats.csv;
    the figures go to <folder>/figures)."""
    global DATA_DIR, HEX_CSV, CORR_CSV, OUT_DIR, CURRENT_SCHEME_LABEL
    ap = argparse.ArgumentParser(description='Figure 8 and supporting scatterplots')
    ap.add_argument('--folder', type=Path, help='one hexagon-analysis folder')
    ap.add_argument('--label', default=SCHEME_LABELS['primary_effort'])
    args, _ = ap.parse_known_args()      # tolerate notebook / Python_02 arguments
    if args.folder:
        candidates = [(None, args.folder, args.label)]
    else:
        candidates = [(scheme, folders.HEXAGONS[scheme], SCHEME_LABELS[scheme])
                      for scheme in ANALYSIS_SCHEMES]
    ran = 0
    for scheme, folder, label in candidates:
        if not (folder / 'spearman_correlations.csv').exists():
            print(f"[skip] {label}: no correlation outputs in {folder}")
            continue
        DATA_DIR = folder
        HEX_CSV = folder / 'hexagon_stats.csv'
        CORR_CSV = folder / 'spearman_correlations.csv'
        OUT_DIR = folder / 'figures'
        CURRENT_SCHEME_LABEL = label
        run_scheme()
        ran += 1
    if ran == 0:
        raise FileNotFoundError(
            "No hexagon-analysis outputs found; run "
            "Python_correlations_with_events.py first.")


if __name__ == '__main__':
    main()
