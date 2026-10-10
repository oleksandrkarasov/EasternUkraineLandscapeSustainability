#!/usr/bin/env python
# coding: utf-8
"""
=============================================================================
 SUPPLEMENTARY FIGURES S5-S9
 Manuscript SUST-D-26-00527 (Sustainability Science)
=============================================================================

 Figure S5  Spatial correlation of the cross-validation errors (Python_01
            output: residual_spatial_correlogram.csv, residual_spatial_range.csv)
 Figure S6  Pre-war z-scores against normal quantiles, both models
            (GEE_05 output: ControlZ_Sample.csv, column zscore_median)
 Figure S7  Annual z-scores of the regular-visitation model, 2022, 2023, 2024
            against the 2019-2021 median (GEE_05 annual GeoTIFFs, band 3)
 Figure S8  Pixels whose embeddings were outside the range of the training
            data in at least one year 2019-2024 (extrapolation check,
            Reviewer 3 comment 10),
            with the loss anomalies of the median comparison and the 2022 gain
            anomalies, both models (GEE_05 main GeoTIFF bands 7 and 25; annual
            GeoTIFF of 2022, band 3)
 Figure S9  Recurrent-visitation model: median z-score and number of wartime
            years with an anomaly in the dominant direction (GEE_05 main
            GeoTIFF band 7; Cumulative_Confidence_Classes.tif band 1)

 The maps use the class colours of Figures 6 and 7. Before plotting, the
 script recomputes the mapped areas from the GeoTIFFs and compares them with
 the GEE_05 area tables (raster_area_check.csv). A mismatch means that a band
 or a file does not match the tables of the manuscript, so check the printed
 table before using the maps.

 INPUTS: the two GEE_05 download folders (one per model) and the Python_01
 metrics folder (a folder or its .zip). If Python_03 has run, its
 run_manifest_v13b12.json (package folder) ties the figures to the GEE_05
 runs it accepted: each folder must hold that run (GEE05_RunManifest), and
 every file that Python_03 recorded must be unchanged (SHA-256). Optional
 line layers (for example the front lines of Figures 6 and 7) can be added in
 OVERLAYS. Run it from the package folder (python
 support\\Python_plot_supplementary_figures.py, or %run in a notebook).

 Machine-readable names keep earlier tokens: damage/rewild = loss/gain
 anomaly, primary_effort = regular-visitation model, ControlZ = pre-war
 z-scores (see README).

 v52 (5 October 2026), presentation only: no change to thresholds, classes,
 masks, areas or provenance checks, nor to font, sizes, colours or
 resolution. Legend wording: 'Small change' for -1 < z < 1; 'No mapped
 anomaly' for class 0 of the year-count map (it also holds removed small
 patches); 'flagged embeddings' in Figure S8, whose panel titles now read
 '... loss in median comparison' and '... gain in 2022'. More space between
 the rows of Figure S8, so that lower-row titles no longer touch the upper-row
 longitude labels; slightly more space between the maps and the legend
 (BELOW_MAP_IN) and between neighbouring panels (PANEL_GAP) in S7-S9.
=============================================================================
"""

import hashlib
import io
import json
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy import ndimage

import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin
from pyproj import Transformer

warnings.filterwarnings('ignore', category=RuntimeWarning)

import sys as _sys
try:
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:                                   # notebook cell
    _sys.path.insert(0, str(Path.cwd() / 'support'))
# A console that already ran another copy of the package keeps its support
# modules in memory; drop them so that this package's support folder is used.
for _m in ("folders", "pipeline_checks"):
    _sys.modules.pop(_m, None)
import folders  # noqa: E402  (local folders: support/folders.py)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Folders: support/folders.py. A GEE_05 folder is used only if it holds that
# model's GEE05_RunManifest_v13b12.csv and main GeoTIFF and, when Python_03
# has run, the GEE_05 run that Python_03 accepted (run_manifest_v13b12.json
# in Python03_manuscript_numbers).
SCHEME_DIRS = {s: [folders.data_dir(folders.GEE05[s])] for s in ('primary_effort', 'recurrent_effort')}
SEARCH_ROOTS = [folders.OUTPUTS]
# Folder that holds run_manifest_v13b12.json written by Python_03.
BUNDLE_DIR_OVERRIDE = None
PY03_DIR = folders.PYTHON03
# Python_01 output with the residual correlogram.
CV_METRICS = [folders.PYTHON01]

OUT_DIR = folders.FIGURES
FIGURES = ['S5', 'S6', 'S7', 'S8', 'S9']        # drop items to skip figures

# Same font as Figure 8; Inter or Arial are used if it is not installed.
DPI = 300
FIG_WIDTH_IN = 176 / 25.4                       # journal print width, 176 mm

# Display projection. EPSG:3857 draws north up with straight graticule lines,
# as in the QGIS maps of Figures 6 and 7. The analysis grid is EPSG:3035.
PLOT_CRS = 'EPSG:3857'
PLOT_RES_M = 750                                # ~500 m on the ground at 49 N

# Optional line layers drawn on every map, e.g.
# {'path': r'C:\...\frontline_2024_09.gpkg', 'label': 'Approximate front line, Sep 2024',
#  'color': '#222222', 'ls': (0, (3, 2)), 'lw': 0.6}
OVERLAYS = []

LABELS = {'primary_effort': 'Regular visitation', 'recurrent_effort': 'Recurrent visitation'}
THRESHOLD = {'primary_effort': 2.5, 'recurrent_effort': 3.0}   # GEE_05 T_EVIDENCE_BY_SCHEME
YEARS = ['2022', '2023', '2024']

# GEE_05 file names and 1-based band numbers (see the export stack in GEE_05,
# section 10). Band descriptions are used instead when the file carries them.
CODE_VERSION = 'v13b12'
TAG = 'revision_annualbase_annual_median'
MAIN_FILE = f'Debiased_RF_Main_{TAG}'
YEAR_FILE = 'Debiased_RF_{y}_AnnualInputSensitivity_' + TAG
CLASS_FILE = 'Cumulative_Confidence_Classes'
MAIN_BANDS = {'zscore_median': 7, 'damage': 11, 'rewild': 12, 'extrapolation_flag': 25,
              'in_ordlo': 26, 'quality_screen_with_novelty': 30}
YEAR_Z_BAND = 3                                 # zscore_<year>_annualbase
CLASS_BAND = 1                                  # class_mapped
NODATA = -9999

C_REG, C_REC = '#1f5f8b', '#c26a1b'             # S5/S6 line colours
C_OUTLINE, C_ORDLO = '#8a8a8a', '#333333'

# =============================================================================
# HELPERS
# =============================================================================


def setup_font():
    """Matplotlib default font; only sizes, spines and resolution are set."""
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': 0.8, 'savefig.dpi': DPI})


def minus(s):
    return s.replace('-', '\u2212')


GEE05_MANIFEST = f'GEE05_RunManifest_{CODE_VERSION}.csv'
PY03_MANIFEST = f'run_manifest_{CODE_VERSION}.json'
_DIRS = {}          # model -> (folder, GEE_05 run_id), filled by scheme_dir
_PY03 = []          # [(path, manifest)] once read


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def python03_manifest():
    """(path, content) of run_manifest_v13b12.json written by Python_03, or
    (None, None). It must report status 'passed'."""
    if _PY03:
        return _PY03[0]
    places = [Path(BUNDLE_DIR_OVERRIDE)] if BUNDLE_DIR_OVERRIDE else [Path(PY03_DIR)]
    places = list(dict.fromkeys(p.resolve() for p in places))
    found = (None, None)
    for p in places:
        f = p / PY03_MANIFEST
        if f.is_file():
            m = json.loads(f.read_text(encoding='utf-8'))
            if m.get('status') != 'passed':
                raise RuntimeError(f'{f} has status "{m.get("status")}": Python_03 did not accept the '
                                   'GEE_05 numbers. Fix and rerun Python_03 before drawing the figures.')
            found = (f, m)
            break
    if found[0] is None:
        print(f'  note: {PY03_MANIFEST} (Python_03) not found in: ' + '; '.join(map(str, places)) +
              '. Run Python_03 first: the figures must show the GEE_05 run it accepted.')
    else:
        print(f'  Python_03 manifest: {found[0]}')
    _PY03.append(found)
    return found


def has_raster(folder, stem):
    return bool(list(folder.glob(stem + '.tif')) or list(folder.glob(stem + '-*.tif')))


def folder_run(folder):
    """(run_id, model, code_version, exported GeoTIFF names) from the folder's
    GEE05_RunManifest, or None if the folder has none."""
    p = folder / GEE05_MANIFEST
    if not p.is_file():
        return None
    m = pd.read_csv(p)
    if len(m) != 1:
        raise ValueError(f'{p} has {len(m)} rows (expected one)')
    r = m.iloc[0]
    geotiffs = [g for g in str(r.get('exported_geotiffs', '')).split(';') if g and g != 'nan']
    return str(r.run_id), str(r.analysis_scheme), str(r.code_version), geotiffs


def scheme_dir(scheme):
    """"""
    if scheme in _DIRS:
        return _DIRS[scheme][0]
    _, py03 = python03_manifest()
    want = py03['outputs'][scheme].get('gee05_run_id') if py03 else None
    tried = []

    def usable(d):
        info = folder_run(d)
        if info is None:
            tried.append(f'{d} (no {GEE05_MANIFEST})')
        elif info[1] != scheme:
            tried.append(f'{d} (GEE_05 run of {info[1]})')
        elif info[2] != CODE_VERSION:
            tried.append(f'{d} (code version {info[2]})')
        elif want and info[0] != want:
            tried.append(f'{d} (run {info[0]}; Python_03 accepted {want})')
        elif not has_raster(d, MAIN_FILE):
            tried.append(f'{d} (no {MAIN_FILE}.tif)')
        else:
            return info
        return None

    found = []
    for d in SCHEME_DIRS.get(scheme, []):
        d = Path(d)
        if not d.is_dir():
            tried.append(f'{d} (does not exist)')
        elif usable(d):
            found = [d]
            break
        else:
            print(f'  note: SCHEME_DIRS folder not used: {tried[-1]}')
    if not found and py03 and py03['outputs'][scheme].get('folder'):
        d = Path(py03['outputs'][scheme]['folder'])
        if not d.is_dir():
            tried.append(f'{d} (folder checked by Python_03; does not exist any more)')
        else:
            found = [p for p in [d, *sorted(q.parent for q in d.glob(f'*/{GEE05_MANIFEST}'))] if usable(p)]
    if not found:
        hits = set()
        for root in map(Path, SEARCH_ROOTS):
            for depth in range(4):
                hits.update(q.parent for q in root.glob('/'.join(['*'] * depth + [GEE05_MANIFEST])))
        found = [d for d in sorted(hits, key=lambda p: (len(p.parts), str(p))) if usable(d)]
    if not found:
        raise FileNotFoundError(
            f'No GEE_05 folder of {scheme} ({LABELS[scheme]}) found. Set SCHEME_DIRS to the folder '
            f'that holds {GEE05_MANIFEST} and {MAIN_FILE}.tif of this model. Folders looked at:\n  ' +
            '\n  '.join(tried or ['none under ' + '; '.join(map(str, SEARCH_ROOTS))]))
    runs = {folder_run(d)[0] for d in found}
    if len(runs) > 1:
        raise RuntimeError(f'GEE_05 folders of {scheme} from different runs ({", ".join(sorted(runs))}): ' +
                           '; '.join(map(str, found)) + '. Set SCHEME_DIRS to the folder of the run '
                           'used in the manuscript (the run that Python_03 accepted).')
    d = max(found, key=lambda p: sum(has_raster(p, g) for g in folder_run(p)[3]))
    run_id = folder_run(d)[0]
    _DIRS[scheme] = (d, run_id)
    print(f'  {LABELS[scheme]}: {d}\n    GEE_05 run {run_id}' +
          (' (accepted by Python_03)' if want else ''))
    missing = [g for g in folder_run(d)[3] if not has_raster(d, g)]
    if missing:
        print('    note: GeoTIFFs exported by this run but not in the folder: ' + ', '.join(missing))
    return d


def check_file(scheme, path):
    """The file must be the one Python_03 checked (same SHA-256), when
    Python_03 recorded a file of that name for this model."""
    _, py03 = python03_manifest()
    if not py03:
        return
    files = py03['outputs'][scheme].get('files', {})
    known = {v['sha256'] for k, v in files.items() if k.replace('\\', '/').split('/')[-1] == path.name}
    if not known:
        print(f'    note: {path.name} was not in the folder when Python_03 ran (not compared)')
    elif sha256(path) not in known:
        raise RuntimeError(f'{path} differs from the file of that name that Python_03 checked for '
                           f'{scheme}. Use the folder of the accepted GEE_05 run, or rerun Python_03.')


def read_table(scheme, name):
    """GEE_05 table of a model, checked for model, code version and run_id."""
    d = scheme_dir(scheme)
    path = d / f'{name}.csv'
    check_file(scheme, path)
    t = pd.read_csv(path)
    for col, value in (('analysis_scheme', scheme), ('code_version', CODE_VERSION),
                       ('run_id', _DIRS[scheme][1])):
        if col in t.columns and not (t[col].astype(str) == value).all():
            raise ValueError(f'{path}: {col} is not {value} in every row')
        if col in ('code_version', 'run_id') and col not in t.columns:
            raise ValueError(f'{path} has no {col} column: not a GEE_05 {CODE_VERSION} export')
    return t


def read_csv_anywhere(name, places):
    """Read a CSV from the first folder or zip in `places` that contains it."""
    for p in places:
        p = Path(p)
        if p.is_dir():
            hits = sorted(p.rglob(name))
            if hits:
                return pd.read_csv(hits[0])
        elif p.suffix == '.zip' and p.exists():
            with zipfile.ZipFile(p) as z:
                hits = [n for n in z.namelist() if n.endswith('/' + name) or n == name]
                if hits:
                    return pd.read_csv(io.BytesIO(z.read(hits[0])))
    raise FileNotFoundError(f'{name} not found in: ' + '; '.join(str(p) for p in places))


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'{name}.{ext}', dpi=DPI, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f'  wrote {OUT_DIR / name}.png / .pdf')


class Raster:
    """Selected bands of a GEE export (single file or Drive tiles), EPSG:3035."""

    def __init__(self, folder, stem, bands):
        files = sorted(folder.glob(stem + '.tif')) or sorted(folder.glob(stem + '-*.tif'))
        if not files:
            raise FileNotFoundError(f'{stem}.tif not found in {folder}')
        with rasterio.open(files[0]) as src:
            desc = list(src.descriptions)
            self.crs = src.crs
            nodata = src.nodata if src.nodata is not None else NODATA
        idx = {}
        for name, number in bands.items():
            if name in desc:
                idx[name] = desc.index(name) + 1
            else:
                idx[name] = number
                if any(desc):
                    print(f'    note: band "{name}" not named in {files[0].name}; using band {number}')
        order = list(idx.values())
        if len(files) == 1:
            with rasterio.open(files[0]) as src:
                arr = src.read(order).astype('float32')
                self.transform = src.transform
        else:
            arr, self.transform = merge([str(f) for f in files], indexes=order, nodata=nodata)
            arr = arr.astype('float32')
        arr[(arr == nodata) | (arr <= NODATA + 1)] = np.nan
        self.data = {name: arr[i] for i, name in enumerate(idx)}
        self.files = files
        self.shape = arr.shape[1:]
        print(f'    {files[0].name}' + (f' (+{len(files) - 1} tiles)' if len(files) > 1 else '') +
              f': bands {dict(zip(idx, order))}')


class Canvas:
    """Common display grid in PLOT_CRS for all maps."""

    def __init__(self, ref):
        h, w = ref.shape
        left, top = ref.transform.c, ref.transform.f
        right, bottom = left + w * ref.transform.a, top + h * ref.transform.e
        b = transform_bounds(ref.crs, PLOT_CRS, left, bottom, right, top, densify_pts=51)
        pad = 5000
        self.x0, self.y0, self.x1, self.y1 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
        self.w = int(np.ceil((self.x1 - self.x0) / PLOT_RES_M))
        self.h = int(np.ceil((self.y1 - self.y0) / PLOT_RES_M))
        self.transform = from_origin(self.x0, self.y1, PLOT_RES_M, PLOT_RES_M)
        self.extent = (self.x0, self.x0 + self.w * PLOT_RES_M, self.y1 - self.h * PLOT_RES_M, self.y1)
        self.to_plot = Transformer.from_crs('EPSG:4326', PLOT_CRS, always_xy=True)
        self.valid = None
        self.view = self.extent

    def set_valid(self, valid, pad_m=8000):
        """Study-area mask on the canvas; the map view is cropped to it."""
        self.valid = valid
        rows, cols = np.where(valid)
        x0, y1 = self.extent[0], self.extent[3]
        self.view = (x0 + cols.min() * PLOT_RES_M - pad_m, x0 + (cols.max() + 1) * PLOT_RES_M + pad_m,
                     y1 - (rows.max() + 1) * PLOT_RES_M - pad_m, y1 - rows.min() * PLOT_RES_M + pad_m)
        self.aspect = (self.view[3] - self.view[2]) / (self.view[1] - self.view[0])

    def warp(self, arr, src, fill=np.nan):
        out = np.full((self.h, self.w), fill, dtype='float32')
        reproject(arr.astype('float32'), out, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=self.transform, dst_crs=PLOT_CRS,
                  resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=fill)
        return out


def clean_mask(a, min_px=400):
    """0/1 mask with NaN outside: drop slivers (< min_px cells) so that only the
    occupation line itself is drawn, not small rings along the state border."""
    out = a.copy()
    valid = np.isfinite(a)
    for value in (1, 0):
        lab, n = ndimage.label(valid & (a == value))
        if n:
            sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
            small = np.isin(lab, np.where(sizes < min_px)[0] + 1)
            out[small] = 1 - value
    return out


def draw_map(ax, cv, classes, colors, ordlo=None, scalebar=False, lat_labels=True):
    """classes: float array on the canvas with integer codes 0..n-1, NaN outside."""
    ax.set_facecolor('white')
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), len(colors))
    ax.imshow(np.ma.masked_invalid(classes), cmap=cmap, norm=norm, extent=cv.extent,
              interpolation='nearest', origin='upper', zorder=1)
    xs = np.linspace(cv.extent[0], cv.extent[1], cv.w)
    ys = np.linspace(cv.extent[3], cv.extent[2], cv.h)
    ax.contour(xs, ys, cv.valid.astype(float), levels=[0.5], colors=C_OUTLINE,
               linewidths=0.5, zorder=3)
    if ordlo is not None:
        ax.contour(xs, ys, clean_mask(ordlo), levels=[0.5], colors=C_ORDLO, linewidths=0.7, zorder=4)
    for ov in OVERLAYS:
        import geopandas as gpd
        g = gpd.read_file(ov['path']).to_crs(PLOT_CRS)
        g.plot(ax=ax, color=ov.get('color', '#222222'), linewidth=ov.get('lw', 0.6),
               linestyle=ov.get('ls', '-'), zorder=5)
    # degree graticule as frame ticks (north-up display)
    v = cv.view
    lons = [l for l in range(30, 45) if v[0] < cv.to_plot.transform(l, 49)[0] < v[1]]
    lats = [l for l in range(44, 54) if v[2] < cv.to_plot.transform(37, l)[1] < v[3]]
    ax.set_xticks([cv.to_plot.transform(l, 49)[0] for l in lons])
    ax.set_xticklabels([f'{l}°E' for l in lons], fontsize=6)
    ax.set_yticks([cv.to_plot.transform(37, l)[1] for l in lats])
    ax.set_yticklabels([f'{l}°N' for l in lats] if lat_labels else [], fontsize=6, rotation=90, va='center')
    ax.tick_params(length=2, width=0.5, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color('#999999')
    ax.set_xlim(v[0], v[1])
    ax.set_ylim(v[2], v[3])
    ax.set_aspect('equal')
    if scalebar:
        lat_c = Transformer.from_crs(PLOT_CRS, 'EPSG:4326', always_xy=True).transform(
            0.5 * (v[0] + v[1]), 0.5 * (v[2] + v[3]))[1]
        length = 50000 / np.cos(np.radians(lat_c))          # 50 km on the ground
        x1 = v[1] - 0.05 * (v[1] - v[0])
        y = v[2] + 0.04 * (v[3] - v[2])
        ax.plot([x1 - length, x1], [y, y], color='#222222', lw=1.2, solid_capstyle='butt', zorder=6)
        ax.text(x1 - length / 2, y + 0.012 * (v[3] - v[2]), '50 km',
                ha='center', va='bottom', fontsize=6, zorder=6)


def z_class_scheme(t):
    """Class codes, colours and labels for z-score maps (Figure 6 colours)."""
    tt = f'{t:g}'
    rows = [('le', -4, '#01665e', 'Gain anomaly, z ≤ −4'),
            ('rng', (-4, -3), '#35978f', 'Gain anomaly, −4 < z ≤ −3')]
    if t < 3:
        rows.append(('rng', (-3, -t), '#7ccbc3', f'Gain anomaly, −3 < z ≤ −{tt}'))
    rows += [('rng', (-t, -1), '#eaf4f2', f'Gain tendency, −{tt} < z ≤ −1'),
             ('mid', (-1, 1), '#f7f7f7', 'Small change, −1 < z < 1'),
             ('rng2', (1, t), '#f8f1e0', f'Loss tendency, 1 ≤ z < {tt}')]
    if t < 3:
        rows.append(('rng2', (t, 3), '#dfc27d', f'Loss anomaly, {tt} ≤ z < 3'))
    rows += [('rng2', (3, 4), '#bf812d', 'Loss anomaly, 3 ≤ z < 4'),
             ('ge', 4, '#543005', 'Loss anomaly, z ≥ 4')]
    return rows


def classify_z(z, t):
    rows = z_class_scheme(t)
    out = np.full(z.shape, np.nan, dtype='float32')
    for code, (kind, v, _, _) in enumerate(rows):
        if kind == 'le':
            m = z <= v
        elif kind == 'ge':
            m = z >= v
        elif kind == 'rng':
            m = (z > v[0]) & (z <= v[1])
        elif kind == 'mid':
            m = (z > v[0]) & (z < v[1])
        else:
            m = (z >= v[0]) & (z < v[1])
        out[m & np.isfinite(z)] = code
    return out, [r[2] for r in rows], [minus(r[3]) for r in rows]


YEAR_CLASSES = [(-3, '#01665e', 'Gain anomaly in 3 wartime years'),
                (-2, '#35978f', 'Gain anomaly in 2 wartime years'),
                (-1, '#99d8e0', 'Gain anomaly in 1 wartime year'),
                (0, '#f7f7f7', 'No mapped anomaly'),
                (1, '#dfc27d', 'Loss anomaly in 1 wartime year'),
                (2, '#bf812d', 'Loss anomaly in 2 wartime years'),
                (3, '#543005', 'Loss anomaly in 3 wartime years')]


def legend_below(target, handles, ncol, y=-0.02, fontsize=6.5):
    kw = dict(handles=handles, loc='upper center', ncol=ncol, frameon=False, fontsize=fontsize,
              handlelength=1.6, handleheight=1.0, columnspacing=1.2, borderaxespad=0)
    if hasattr(target, 'add_axes'):
        target.legend(bbox_to_anchor=(0.5, y), bbox_transform=target.transFigure, **kw)
    else:
        target.legend(bbox_to_anchor=(0.5, y), **kw)


def patch(color, label):
    return Patch(facecolor=color, edgecolor='#bbbbbb', linewidth=0.3, label=label)


def ordlo_handle():
    return Line2D([], [], color=C_ORDLO, lw=0.8, label='Pre-2022 occupation line')


def overlay_handles():
    return [Line2D([], [], color=o.get('color', '#222222'), lw=o.get('lw', 0.6),
                   ls=o.get('ls', '-'), label=o['label']) for o in OVERLAYS if 'label' in o]


# =============================================================================
# DATA AND AREA CHECK
# =============================================================================


def load_scheme(scheme):
    d = scheme_dir(scheme)
    main = Raster(d, MAIN_FILE, MAIN_BANDS)
    years = {}
    for y in YEARS:
        stem = YEAR_FILE.format(y=y)
        if has_raster(d, stem):
            years[y] = Raster(d, stem, {f'zscore_{y}_annualbase': YEAR_Z_BAND})
        elif scheme == 'primary_effort' and 'S7' in FIGURES:
            raise FileNotFoundError(f'{stem}.tif not found in {d} (needed for Figure S7)')
        elif y == '2022' and 'S8' in FIGURES:
            raise FileNotFoundError(f'{stem}.tif not found in {d} (needed for Figure S8)')
        else:
            years[y] = None
            print(f'    note: {stem}.tif not in the folder; its areas are not checked')
    cls = Raster(d, CLASS_FILE, {'class_mapped': CLASS_BAND,
                                 'n_loss_years': 3, 'n_gain_years': 4})
    for r in [main, cls, *[v for v in years.values() if v is not None]]:
        for f in r.files:
            check_file(scheme, f)
    tables = {n: read_table(scheme, n) for n in
              ('Areas_Excess_by_Zclass', 'Cumulative_Confidence_Class_Areas',
               f'Areas_Headline_{TAG.replace("revision_", "")}', f'Areas_by_Screen_{CODE_VERSION}')}
    return dict(dir=d, main=main, years=years, cls=cls, tables=tables)


def _patch8(mask, min_px=8):
    """Keep 8-connected patches of at least min_px cells (as GEE_05)."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return mask
    keep = np.bincount(lab.ravel()) >= min_px
    keep[0] = False
    return keep[lab]


def area_check(scheme, S):
    """

    Cells on the AOI outline: GEE counts such a cell only by the part that
    lies inside the outline (weights in steps of 1/255), while the GeoTIFF
    stores the cell whole. Raster areas can therefore be slightly larger than
    the tables, mostly for the analysed area itself. The analysed area is
    checked against GEE's cell count (eligible_px in GEE05_RunManifest): the
    raster must hold at least these cells, and the extra cells must not
    exceed the cells on the edge of the mapped area."""
    t = THRESHOLD[scheme]
    ex = S['tables']['Areas_Excess_by_Zclass']
    for name, tab in S['tables'].items():
        if 'code_version' not in tab.columns or not (tab.code_version == 'v13b12').all():
            raise ValueError(f'{name}.csv is not a GEE_05 v13b12 export: rerun GEE_05 v13b12')
    cum_all = S['tables']['Cumulative_Confidence_Class_Areas']
    cum_map = cum_all[cum_all.variant == 'mapped']
    cum = cum_map[cum_map['set'] == 'dominant_class']
    head = S['tables'][f'Areas_Headline_{TAG.replace("revision_", "")}'].set_index('variant')
    m = S['main']
    if m.crs is None or m.crs.to_epsg() != 3035 or not np.isclose(abs(m.transform.a * m.transform.e), 250000.0):
        raise ValueError('expected the 500 m EPSG:3035 analysis grid')
    px_km2 = 0.25
    z = m.data['zscore_median']
    row = ex[(ex.comparison == 'median_headline') & (np.isclose(ex.z_class, t))].iloc[0]
    rows = []

    def add(item, raster_px, table_km2):
        rows.append(dict(model=LABELS[scheme], item=item, raster_km2=raster_px * px_km2,
                         table_km2=table_km2))

    add('Analysed area (mapping mask)', np.sum(np.isfinite(z)), row.eligible_ground_area_km2)
    edge_note = None
    run = pd.read_csv(S['dir'] / GEE05_MANIFEST).iloc[0]
    if 'eligible_px' in run.index and np.isfinite(float(run.eligible_px)):
        fin = np.pad(np.isfinite(z), 1, constant_values=False)
        inner = (fin[1:-1, 1:-1] & fin[:-2, 1:-1] & fin[2:, 1:-1] & fin[1:-1, :-2] & fin[1:-1, 2:])
        n_px, n_gee = int(fin.sum()), int(run.eligible_px)
        n_edge = n_px - int(inner.sum())
        edge_note = n_gee <= n_px <= n_gee + n_edge
        print(f'    analysed area: {n_px:,} cells in the GeoTIFF, {n_gee:,} counted by GEE; '
              f'{n_px - n_gee:,} extra cells, {n_edge:,} cells on the edge of the mapped area')
    add(f'Median comparison, z >= {t:g}', np.sum(z >= t), row.observed_loss_km2)
    add(f'Median comparison, z <= -{t:g}', np.sum(z <= -t), row.observed_gain_km2)
    for y in YEARS:
        r = ex[(ex.comparison == f'year_{y}') & (np.isclose(ex.z_class, t))].iloc[0]
        if S['years'][y] is None:          # GeoTIFF not downloaded: not checked
            add(f'{y}, z >= {t:g}', np.nan, r.observed_loss_km2)
            add(f'{y}, z <= -{t:g}', np.nan, r.observed_gain_km2)
            continue
        zy = S['years'][y].data[f'zscore_{y}_annualbase']
        add(f'{y}, z >= {t:g}', np.sum(zy >= t), r.observed_loss_km2)
        add(f'{y}, z <= -{t:g}', np.sum(zy <= -t), r.observed_gain_km2)
    c = S['cls'].data['class_mapped']
    for k in (-3, -2, -1, 1, 2, 3):
        r = cum[pd.to_numeric(cum['class'], errors='coerce') == k]
        add(f'Year-count class {k:+d}', np.sum(c == k), float(r.observed_km2.iloc[0]) if len(r) else np.nan)
    nl = S['cls'].data['n_loss_years']
    ng = S['cls'].data['n_gain_years']
    ok = np.isfinite(nl) & np.isfinite(ng)
    unions = {}
    for name, m in (('union_loss_ge1', ok & (nl >= 1)), ('union_loss_ge2', ok & (nl >= 2)),
                    ('union_gain_ge1', ok & (ng >= 1)), ('union_gain_ge2', ok & (ng >= 2))):
        unions[name] = _patch8(m)
        r = cum_map[cum_map['set'] == name]
        add(f'Year counts, {name} (patch-filtered)', unions[name].sum(),
            float(r.observed_km2.iloc[0]) if len(r) else np.nan)
    r = cum_map[cum_map['set'] == 'overlap_loss_and_gain']
    add('Year counts, loss and gain overlap (patch-filtered)',
        (unions['union_loss_ge1'] & unions['union_gain_ge1']).sum(),
        float(r.observed_km2.iloc[0]) if len(r) else np.nan)
    dmg = S['main'].data['damage'] == 1
    rew = S['main'].data['rewild'] == 1
    scr = S['main'].data['quality_screen_with_novelty'] == 1
    add('Stricter version, loss', dmg.sum(), head.loc['primary_all_valid_input_pixels', 'damage_km2'])
    add('Stricter version, gain', rew.sum(), head.loc['primary_all_valid_input_pixels', 'rewild_km2'])
    add('Stricter version, loss, after the three exclusion checks',
        (dmg & scr).sum(), head.loc['quality_screen_with_novelty_sensitivity', 'damage_km2'])
    add('Stricter version, gain, after the three exclusion checks',
        (rew & scr).sum(), head.loc['quality_screen_with_novelty_sensitivity', 'rewild_km2'])
    # Extrapolation check (Figure S8, Table 2): anomalies in pixels whose
    # embeddings were like the training data in every year 2019-2024.
    scr_t = S['tables'][f'Areas_by_Screen_{CODE_VERSION}'].set_index('statistic')
    like = S['main'].data['extrapolation_flag'] == 0
    col = 'no_annual_extrapolation_km2'
    add(f'Median comparison, z >= {t:g}, embeddings like the training data',
        np.sum((z >= t) & like), float(scr_t.loc['median_loss', col]))
    for y, side, stat in (('2022', -1, 'year_2022_gain'), ('2023', 1, 'year_2023_loss'),
                          ('2024', 1, 'year_2024_loss')):
        if S['years'][y] is None:
            add(f'{y}, embeddings like the training data', np.nan, float(scr_t.loc[stat, col]))
            continue
        zy = S['years'][y].data[f'zscore_{y}_annualbase']
        hit = (zy >= t) if side > 0 else (zy <= -t)
        add(f'{y}, z {">=" if side > 0 else "<="} {"" if side > 0 else "-"}{t:g}, embeddings like the '
            'training data', np.sum(hit & like), float(scr_t.loc[stat, col]))
    out = pd.DataFrame(rows)
    out['difference_pct'] = 100 * (out.raster_km2 - out.table_km2) / out.table_km2.where(out.table_km2 > 0)
    ok = (out.raster_km2 - out.table_km2).abs() <= np.maximum(1.0, 0.005 * out.table_km2)
    out['status'] = np.select([out.raster_km2.isna(), out.table_km2.isna(), ok],
                              ['not checked (GeoTIFF not in folder)', 'no table row', 'ok'],
                              'MISMATCH')
    if edge_note is not None:
        out.loc[0, 'status'] = 'ok (AOI edge cells)' if edge_note else 'MISMATCH'
    return out


# =============================================================================
# FIGURE S5 AND S6
# =============================================================================


def figure_s5():
    cg = read_csv_anywhere('residual_spatial_correlogram.csv', CV_METRICS)
    rg = read_csv_anywhere('residual_spatial_range.csv', CV_METRICS)
    rg = rg[rg.analysis_type == 'main_fixed'].set_index('sampling_scheme')
    lat = rg.mean_latitude_deg.mean()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN * 0.9, 3.4))
    for sch, col, ytxt in (('primary_effort', C_REG, 0.118), ('recurrent_effort', C_REC, 0.100)):
        s = cg[cg.sampling_scheme == sch].sort_values('distance_mid_km')
        rng = float(rg.loc[sch, 'empirical_residual_range_km'])
        ax.plot(s.distance_mid_km, s.residual_correlation, '-o', color=col, ms=3.5, lw=1.4,
                label=LABELS[sch])
        ax.axvline(rng, color=col, ls=':', lw=1.2)
        ax.text(rng + 1, ytxt, f'{rng:g} km', color=col, fontsize=8)
    ax.axhline(0.05, color='#777777', ls='--', lw=0.8)
    ax.axhline(0, color='#444444', lw=0.6)
    ax.text(148, 0.053, 'correlation 0.05', ha='right', va='bottom', fontsize=7.5, color='#555555')
    for deg in (0.30, 0.45, 0.60):          # east-west block width at the mean latitude
        w = deg * 111.32 * np.cos(np.radians(lat))
        ax.plot([w, w], [-0.05, -0.038], color='#888888', lw=1.2)
        ax.text(w + 1.0, -0.049, f'{deg:.2f}°', rotation=90, fontsize=6.8,
                color='#666666', ha='left', va='bottom')
    ax.text(w + 4.5, -0.049, 'block width, east–west', fontsize=6.8, color='#666666',
            ha='left', va='bottom')
    ax.set_xlim(0, 150)
    ax.set_ylim(-0.05, 0.14)
    ax.set_xlabel('Distance between records (km)')
    ax.set_ylabel('Correlation of out-of-fold residuals')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout()
    save(fig, 'Figure_S5')


def figure_s6():
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN * 0.9, 3.2), sharex=True, sharey=True)
    for ax, sch, col, lab in ((axes[0], 'primary_effort', C_REG, 'a  Regular visitation'),
                              (axes[1], 'recurrent_effort', C_REC, 'b  Recurrent visitation')):
        z = read_table(sch, 'ControlZ_Sample').zscore_median.dropna().values
        z = np.sort(z)
        n = len(z)
        q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        idx = np.unique(np.concatenate([np.linspace(0, n - 1, 1500).astype(int),
                                        np.arange(0, 60), np.arange(n - 60, n)]))
        ax.plot(q[idx], z[idx], '.', ms=2.2, color=col)
        ax.plot([-4.2, 4.2], [-4.2, 4.2], color='#444444', lw=0.8)
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(-6, 4.2)
        ax.set_title(lab, loc='left', fontsize=9, fontweight='bold')
        ax.text(0.97, 0.05, minus(f'SD {z.std(ddof=1):.2f}\nskewness {stats.skew(z):.2f}\nn = {n:,}'),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5, color='#333333')
        ax.set_xlabel('Normal quantiles')
    axes[0].set_ylabel('Pre-war z-scores')
    fig.tight_layout()
    save(fig, 'Figure_S6')


# =============================================================================
# MAPS: FIGURES S7-S9
# =============================================================================


# Layout of the map figures (inches / figure fractions). BELOW_MAP_IN is the
# space between the bottom of the maps and the legend anchor plus 0.06; it was
# 0.18, which let the longitude labels touch the first legend row. PANEL_GAP
# was 0.012, which let the last longitude label of one panel touch the first
# label of the next.
BELOW_MAP_IN = 0.26
PANEL_GAP = 0.02


def map_figure(ncols, cv, legend_rows):
    """Figure with `ncols` map panels; returns fig and axes."""
    gap = PANEL_GAP
    left, right = 0.035, 0.995
    pw = (right - left - gap * (ncols - 1)) / ncols
    ph_in = FIG_WIDTH_IN * pw * cv.aspect
    top_in, legend_in = 0.22, 0.16 * legend_rows + 0.12
    fig_h = ph_in + top_in + legend_in + BELOW_MAP_IN
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h))
    axes = [fig.add_axes([left + i * (pw + gap), (legend_in + BELOW_MAP_IN) / fig_h, pw, ph_in / fig_h])
            for i in range(ncols)]
    return fig, axes, (legend_in + 0.06) / fig_h


def panel_title(ax, text):
    ax.set_title(text, loc='left', fontsize=8, fontweight='bold', pad=3)


def figure_s7(S, cv):
    sch = 'primary_effort'
    t = THRESHOLD[sch]
    main = S[sch]['main']
    ordlo = cv.warp(main.data['in_ordlo'], main)
    _, colors, labels = classify_z(np.array([0.0]), t)
    fig, axes, ly = map_figure(3, cv, legend_rows=4)
    for i, (ax, y) in enumerate(zip(axes, YEARS)):
        r = S[sch]['years'][y]
        cls, _, _ = classify_z(r.data[f'zscore_{y}_annualbase'], t)
        draw_map(ax, cv, cv.warp(cls, r), colors, ordlo, scalebar=(i == 2), lat_labels=(i == 0))
        panel_title(ax, f'{"abc"[i]}  {y} vs 2019–2021 median')
    hs = [patch(c, l) for c, l in zip(colors, labels)] + [ordlo_handle()] + overlay_handles()
    legend_below(fig, hs, ncol=3, y=ly)
    save(fig, 'Figure_S7')


def map_grid(nrows, ncols, cv, legend_rows):
    """Figure with nrows x ncols map panels; returns fig, axes[row][col] and
    the legend position."""
    gap = PANEL_GAP
    left, right = 0.035, 0.995
    pw = (right - left - gap * (ncols - 1)) / ncols
    ph_in = FIG_WIDTH_IN * pw * cv.aspect
    title_in, legend_in = 0.40, 0.18 * legend_rows + 0.14
    fig_h = nrows * (ph_in + title_in) + legend_in + BELOW_MAP_IN
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h))
    axes = []
    for r in range(nrows):
        bottom_in = legend_in + BELOW_MAP_IN + (nrows - 1 - r) * (ph_in + title_in)
        axes.append([fig.add_axes([left + c * (pw + gap), bottom_in / fig_h, pw, ph_in / fig_h])
                     for c in range(ncols)])
    return fig, axes, (legend_in + 0.06) / fig_h


def figure_s8(S, cv):
    """Extrapolation check. A pixel is flagged when at least one annual
    embedding 2019-2024 was outside the range of the training data (GEE_04 diagnostic:
    standardised distance to the training mean, number of bands outside the
    training 2.5-97.5 percentile range or largest standardised band value
    above the 99th percentile of the training records). Pixels without a flag
    value count as flagged, as in the GEE_05 area table."""
    colors = ['#f7f7f7', '#cdcdcd', '#8c510a', '#dfc27d', '#01665e', '#9fd3cc']
    labels = ['No anomaly of the plotted type',
              'No anomaly of the plotted type; flagged embeddings',
              'Loss anomaly',
              'Loss anomaly; flagged embeddings',
              'Gain anomaly',
              'Gain anomaly; flagged embeddings']
    fig, axes, ly = map_grid(2, 2, cv, legend_rows=3)
    for r, sch in enumerate(('primary_effort', 'recurrent_effort')):
        t = THRESHOLD[sch]
        m = S[sch]['main']
        yr = S[sch]['years']['2022']
        if yr.shape != m.shape or yr.transform != m.transform:
            raise ValueError(f'{sch}: the 2022 GeoTIFF is not on the grid of the main GeoTIFF')
        ok = np.isfinite(m.data['zscore_median'])
        like = m.data['extrapolation_flag'] == 0
        n_missing = int(np.sum(ok & ~np.isfinite(m.data['extrapolation_flag'])))
        if n_missing:
            print(f'    note: {n_missing} cells without an extrapolation flag value (shown as flagged)')
        share = np.sum(ok & ~like) / np.sum(ok)
        print(f'    {LABELS[sch]}: {100 * share:.1f}% of the mapped cells flagged')
        panels = ((m.data['zscore_median'] >= t, 2, f'{LABELS[sch]}: loss in median comparison'),
                  (yr.data['zscore_2022_annualbase'] <= -t, 4, f'{LABELS[sch]}: gain in 2022'))
        for c, (hit, code, title) in enumerate(panels):
            cls = np.where(ok, 0, np.nan).astype('float32')
            cls[ok & ~like] = 1
            cls[ok & hit & like] = code
            cls[ok & hit & ~like] = code + 1
            draw_map(axes[r][c], cv, cv.warp(cls, m), colors, cv.warp(m.data['in_ordlo'], m),
                     scalebar=(r == 1 and c == 1), lat_labels=(c == 0))
            panel_title(axes[r][c], f'{"abcd"[2 * r + c]}  {title}')
    hs = [patch(col, lab) for col, lab in zip(colors, labels)] + [ordlo_handle()] + overlay_handles()
    legend_below(fig, hs, ncol=2, y=ly)
    save(fig, 'Figure_S8')


def figure_s9(S, cv):
    sch = 'recurrent_effort'
    t = THRESHOLD[sch]
    m = S[sch]['main']
    ordlo = cv.warp(m.data['in_ordlo'], m)
    fig, axes, ly = map_figure(2, cv, legend_rows=5)
    cls, colors, labels = classify_z(m.data['zscore_median'], t)
    draw_map(axes[0], cv, cv.warp(cls, m), colors, ordlo, lat_labels=True)
    panel_title(axes[0], 'a  Median 2019–2021 vs 2022–2024')
    hs_a = [patch(c, l) for c, l in zip(colors, labels)]
    c = S[sch]['cls'].data['class_mapped']
    present = [k for k, _, _ in YEAR_CLASSES if k == 0 or np.any(c == k)]
    rows = [r for r in YEAR_CLASSES if r[0] in present]
    code = np.full(c.shape, np.nan, dtype='float32')
    for i, (k, _, _) in enumerate(rows):
        code[c == k] = i
    draw_map(axes[1], cv, cv.warp(code, S[sch]['cls']), [r[1] for r in rows], ordlo,
             scalebar=True, lat_labels=False)
    panel_title(axes[1], 'b  Number of wartime years with an anomaly')
    hs_b = ([patch(col, minus(lab.format(t=f'{t:g}'))) for _, col, lab in rows] +
            [ordlo_handle()] + overlay_handles())
    for ax, hs in ((axes[0], hs_a), (axes[1], hs_b)):
        ax.legend(handles=hs, loc='upper left', bbox_to_anchor=(0.0, -0.05), ncol=1, frameon=False,
                  fontsize=6.5, handlelength=1.6, handleheight=1.0, borderaxespad=0)
    save(fig, 'Figure_S9')


# =============================================================================
# MAIN
# =============================================================================


def main():
    setup_font()
    print(f'Output folder: {OUT_DIR}')
    if 'S5' in FIGURES:
        print('Figure S5')
        figure_s5()
    if 'S6' in FIGURES:
        print('Figure S6')
        figure_s6()
    if not {'S7', 'S8', 'S9'} & set(FIGURES):
        return
    print('Reading GeoTIFFs')
    S = {s: load_scheme(s) for s in ('primary_effort', 'recurrent_effort')}
    chk = pd.concat([area_check(s, S[s]) for s in S], ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chk.to_csv(OUT_DIR / 'raster_area_check.csv', index=False)
    with pd.option_context('display.width', 160, 'display.max_columns', 10):
        print(chk.round(2).to_string(index=False))
    n = chk.status.value_counts()
    print('Area check: ' + ', '.join(f'{k} {v}' for k, v in n.items()))
    if n.get('MISMATCH', 0) or n.get('no table row', 0):
        print('WARNING: some raster areas differ from the GEE_05 tables by more than 0.5% (at least 1 km2) '
              '(see raster_area_check.csv). Check file names and band numbers before using the maps.')
    if n.get('not checked (GeoTIFF not in folder)', 0):
        print('NOTE: some areas were not checked because a GeoTIFF of that GEE_05 run is not in the '
              'folder. Download it from the run\'s Drive folder (export_folder in '
              f'{GEE05_MANIFEST}) into the same model folder and run this script again.')
    ref = S['primary_effort']['main']
    cv = Canvas(ref)
    cv.set_valid(np.isfinite(cv.warp(ref.data['zscore_median'], ref)))
    for name, fn in (('S7', figure_s7), ('S8', figure_s8), ('S9', figure_s9)):
        if name in FIGURES:
            print(f'Figure {name}')
            fn(S, cv)


if __name__ == '__main__':
    main()
