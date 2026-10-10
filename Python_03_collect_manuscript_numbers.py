#!/usr/bin/env python3
"""

    python Python_03_collect_manuscript_numbers.py

Reads the folders of support/folders.py (GEE05_regular, GEE05_recurrent,
GEE04_final_models, Python01_cv_metrics) and writes to
outputs/Python03_manuscript_numbers:
  checks_v13b12.csv                one row per check: pass / fail, detail
  manuscript_numbers_v13b12.csv    one row per number (only if every check
                                   passes): model, section, item, value in
                                   manuscript v17 where printed, new value,
                                   status missing / new / changed / unchanged
  manuscript_numbers_v13b12_FAILED.csv  the same table when a check fails
  run_manifest_v13b12.json         run identifiers, parameters, code and
                                   output hashes (SHA-256)
Exit status 1 when any check fails. Nothing is rounded. Pre-war columns are
BENCHMARKS, not calibrated no-war expectations (see the README).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Folder of the unzipped code package (the folder that contains support\).
# Leave None when you run this file from inside the package. Set it when you
# paste the script into a notebook cell or keep a copy of the script
# elsewhere.
BUNDLE_DIR_OVERRIDE = None


def _bundle_dir() -> Path:
    """The package folder: BUNDLE_DIR_OVERRIDE, else this script's folder,
    else (notebook cell, no __file__) the working folder; it must contain
    support/pipeline_checks.py."""
    candidates = [Path(BUNDLE_DIR_OVERRIDE)] if BUNDLE_DIR_OVERRIDE else []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    candidates.append(Path.cwd())
    for folder in candidates:
        if (folder / "support" / "pipeline_checks.py").is_file():
            return folder
    raise FileNotFoundError(
        "support\\pipeline_checks.py was not found (looked in: "
        + "; ".join(str(c) for c in candidates) + "). Keep the unzipped package "
        "together and run the script from its folder, e.g. in PowerShell: "
        "cd <package folder>; python Python_03_collect_manuscript_numbers.py ; in a notebook: "
        "%run <package folder>\\Python_03_collect_manuscript_numbers.py , or set BUNDLE_DIR_OVERRIDE near the top "
        "of this script to the package folder.")


BUNDLE_DIR = _bundle_dir()
sys.path.insert(0, str(BUNDLE_DIR / "support"))
CODE_DIR = BUNDLE_DIR      # code files are hashed from here
# A console that already ran another copy of the package keeps its support
# modules in memory; drop them so that this package's support folder is used.
for _m in ("folders", "pipeline_checks"):
    sys.modules.pop(_m, None)
from pipeline_checks import (  # noqa: E402
    CODE_VERSION, DRIVE_DUPLICATE, SENS_STAT_NAMES, area_close, area_leq, cell_close, find_unique,
    jaccard, safe_ratio, valid_overlap)

import folders  # noqa: E402  (local folders: support/folders.py)

# GEE_04 download (GEE04_RunManifest of the run whose assets GEE_05 read).
BASE_DIR = folders.GEE04
FOLDER_STEM = "debiased_rf_revision_v13b_direct_embeddings"   # used by the tests only
SCHEME_DIRS = {s: [folders.data_dir(folders.GEE05[s])] for s in ("primary_effort", "recurrent_effort")}
# Python_01 output folder (its default --output-dir); its overall verdict
# (CV_CHECKS_VERDICT_v13b12.json, CHECKS_FAILED.txt) is read from there.
CV_METRICS_DIR = folders.PYTHON01
THRESHOLD = {"primary_effort": 2.5, "recurrent_effort": 3.0}
TAG = "annualbase_annual_median"
SFX = f"_{CODE_VERSION}"
OUT_OK = folders.PYTHON03 / f"manuscript_numbers{SFX}.csv"
OUT_FAILED = folders.PYTHON03 / f"manuscript_numbers{SFX}_FAILED.csv"
OUT_CHECKS = folders.PYTHON03 / f"checks{SFX}.csv"
OUT_MANIFEST = folders.PYTHON03 / f"run_manifest{SFX}.json"

SENSITIVITY_TABLES = ["main_main", "bias_sigma_50000m", "bias_sigma_200000m",
                      "dropped_year_2019", "dropped_year_2020", "dropped_year_2021",
                      "sd_floor_0p01", "sd_floor_0p04"]
SENS_COLS = ["family", "variant", "design", "statistic", "variant_km2",
             "main_km2", "overlap_with_main_km2"]
SENS_KEYS = {  # table -> (family, variant, designs)
    "main_main": ("main", "main", ["main"]),
    "bias_sigma_50000m": ("bias_sigma", "50000m", ["three_years"]),
    "bias_sigma_200000m": ("bias_sigma", "200000m", ["three_years"]),
    **{f"dropped_year_{y}": ("dropped_year", f"without_{y}",
                             ["variance_only", "baseline_and_variance"])
       for y in (2019, 2020, 2021)},
    "sd_floor_0p01": ("sd_floor", "0.01", ["three_years"]),
    "sd_floor_0p04": ("sd_floor", "0.04", ["three_years"]),
}
Z_CLASSES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
EXCESS_COMPARISONS = ["median_headline", "composite_sensitivity", "year_2022", "year_2023",
                      "year_2024"]
UNION_SETS = ["union_loss_ge1", "union_loss_ge2", "union_gain_ge1", "union_gain_ge2",
              "overlap_loss_and_gain"]
HEADLINE_VARIANTS = ["primary_all_valid_input_pixels", "composite_representation_sensitivity",
                     "quality_screen_with_novelty_sensitivity",
                     "quality_screen_without_novelty_sensitivity", "stricter_no_closing"]
DIAG_PERIODS = ["median", "composite", "2022", "2023", "2024", "null_2019", "null_2020",
                "null_2021", "null_pooled_sample"]
PREWAR_YEARS = [2019, 2020, 2021]
SCREEN_STATS = ["domain", "median_loss", "median_gain", "year_2022_loss", "year_2022_gain",
                "year_2023_loss", "year_2023_gain", "year_2024_loss", "year_2024_gain",
                "class_loss_mapped", "class_gain_mapped", "class_loss2_mapped",
                "union_loss_mapped", "union_gain_mapped", "stricter_loss", "stricter_gain"]
SCREEN_COLS = ["all_pixels_km2", "wartime_sd_km2", "interior_propensity_km2",
               "no_annual_extrapolation_km2", "all_three_screens_km2", "at_sd_floor_km2"]
GRID_Z = [2.5, 3.0, 3.5, 4.0]
GRID_PATCH = [4, 8, 16]
# name: (required columns, key columns, columns that must be finite)
TABLES = {
    "Areas_Excess_by_Zclass.csv": (
        ["comparison", "z_class", "observed_loss_km2", "observed_gain_km2",
         "prewar_benchmark_loss_km2", "prewar_benchmark_gain_km2",
         "eligible_ground_area_km2"],
        ["comparison", "z_class"],
        ["observed_loss_km2", "observed_gain_km2", "prewar_benchmark_loss_km2",
         "prewar_benchmark_gain_km2", "eligible_ground_area_km2"],
        [(c, z) for c in EXCESS_COMPARISONS for z in Z_CLASSES]),
    "Cumulative_Confidence_Class_Areas.csv": (
        ["variant", "set", "class", "observed_km2"], ["variant", "set", "class"],
        ["observed_km2"],
        [(v, "dominant_class", k) for v in ("pixel", "mapped") for k in (-3, -2, -1, 1, 2, 3)]
        + [(v, u, float("nan")) for v in ("pixel", "mapped") for u in UNION_SETS]),
    f"Areas_Headline_{TAG}.csv": (
        ["variant", "damage_km2", "rewild_km2"], ["variant"],
        ["damage_km2", "rewild_km2"], [(v,) for v in HEADLINE_VARIANTS]),
    f"Debiased_RF_Diagnostics_revision_{TAG}.csv": (
        ["period", "z_cal_mean"], ["period"], [], [(p,) for p in DIAG_PERIODS]),
    f"Areas_Stricter_Null{SFX}.csv": (
        ["null_year", "z_threshold", "stricter_loss_km2", "stricter_gain_km2",
         "stricter_loss_screened_km2", "pixel_loss_km2", "pixel_gain_km2",
         "stricter_loss_in_exceedance_km2", "stricter_gain_in_exceedance_km2",
         "stricter_loss_added_km2", "stricter_gain_added_km2",
         "pixel_loss_removed_km2", "pixel_gain_removed_km2",
         "stricter_no_closing_loss_km2", "stricter_no_closing_gain_km2"],
        ["null_year"],
        ["stricter_loss_km2", "stricter_gain_km2", "stricter_loss_screened_km2",
         "stricter_gain_screened_km2", "pixel_loss_km2", "pixel_gain_km2",
         "stricter_loss_in_exceedance_km2", "stricter_gain_in_exceedance_km2",
         "stricter_loss_added_km2", "stricter_gain_added_km2",
         "pixel_loss_removed_km2", "pixel_gain_removed_km2",
         "stricter_no_closing_loss_km2", "stricter_no_closing_gain_km2"],
        [(y,) for y in PREWAR_YEARS]),
    f"Areas_by_Screen{SFX}.csv": (
        ["statistic"] + SCREEN_COLS, ["statistic"], SCREEN_COLS, [(x,) for x in SCREEN_STATS]),
    f"SD_Floor_Diagnostics{SFX}.csv": (["share_pixels_at_floor"], [], ["share_pixels_at_floor"], 1),
    "Areas_vs_Threshold_and_PatchSize.csv": (
        ["z_threshold", "min_patch_px", "is_main_parameters", "damage_km2", "rewild_km2"],
        ["z_threshold", "min_patch_px"], ["damage_km2", "rewild_km2"],
        [(z, p) for z in GRID_Z for p in GRID_PATCH]),
    f"GEE05_RunManifest{SFX}.csv": (
        ["run_id", "period_probability_run_id", "annual_probability_run_id",
         "novelty_run_id", "period_probability_code_version"], [], [], 1),
    **{f"Sensitivity_{n}{SFX}.csv": (
        SENS_COLS, ["family", "variant", "design", "statistic"],
        ["variant_km2", "main_km2", "overlap_with_main_km2"],
        [(SENS_KEYS[n][0], SENS_KEYS[n][1], dz, st)
         for dz in SENS_KEYS[n][2] for st in SENS_STAT_NAMES])
       for n in SENSITIVITY_TABLES},
}

V17 = {
    "primary_effort": {
        "median_loss": 1744.97, "median_gain": 6.5, "2022_loss": 0.0, "2022_gain": 5648.08,
        "2023_loss": 1537.23, "2023_gain": 49.75, "2024_loss": 4925.33, "2024_gain": 0.0,
        "median_loss_bench": 125.09, "median_gain_bench": 610.53,
        "dom1_loss": 4572.56, "dom2_loss": 926.5, "dom1_gain": 5516.08, "dom2_gain": 5.0,
        "stricter_loss": 459.25, "stricter_loss_screens": 321.0,
        "drop2019_loss": 61.0, "drop2020_loss": 868.25, "drop2021_loss": 61.5,
        "sigma50000_loss": 989.75, "sigma200000_loss": 377.5,
    },
    "recurrent_effort": {
        "median_loss": 398.25, "median_gain": 0.0, "2022_loss": 0.0, "2022_gain": 1275.5,
        "2023_loss": 270.25, "2023_gain": 20.75, "2024_loss": 680.0, "2024_gain": 0.0,
        "median_loss_bench": 35.59, "median_gain_bench": 257.01,
        "dom1_loss": 578.75, "dom2_loss": 182.5, "dom1_gain": 1288.25, "dom2_gain": 0.0,
        "stricter_loss": 396.75, "stricter_loss_screens": 202.0,
        "drop2019_loss": 294.0, "drop2020_loss": 519.0, "drop2021_loss": 71.0,
        "sigma50000_loss": 544.0, "sigma200000_loss": 372.25,
    },
}


class Checks:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, scheme, check, ok, detail=""):
        self.rows.append({"model": scheme, "check": check,
                          "status": "pass" if ok else "fail", "detail": detail})
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if r["status"] == "fail"]


close = area_close      # same pixels summed by different reductions


def _norm(v) -> str:
    """Key value as text: numbers in %g form (2.0 -> '2'), NaN -> 'nan'."""
    try:
        f = float(v)
        return "nan" if math.isnan(f) else f"{f:g}"
    except (TypeError, ValueError):
        return str(v).strip()


def folder(scheme: str):
    for d in SCHEME_DIRS[scheme]:
        if d.is_dir():
            return d
    return None


def load_table(d: Path, name: str, scheme: str, chk: Checks):
    """Read one GEE_05 table and check it; None when a check fails. Checks:
    one copy, columns, code version, model, one run_id, EXACTLY the expected
    rows (keys and count; a missing or extra row fails), finite required
    values and non-negative areas."""
    req, keys, finite, expected = TABLES[name]
    label = f"{name}"
    path = d / name
    dupes = [p.name for p in d.glob(f"{Path(name).stem}*{Path(name).suffix}")
             if p.name != name and DRIVE_DUPLICATE.sub(r"\1", p.name) == name]
    if not chk.add(scheme, f"{label}: one copy", not dupes, "; ".join(dupes)):
        return None
    if not chk.add(scheme, f"{label}: present", path.exists(), str(path)):
        return None
    t = pd.read_csv(path)
    miss = [c for c in req + ["run_id", "code_version", "analysis_scheme"] if c not in t.columns]
    if not chk.add(scheme, f"{label}: columns", not miss, f"missing {miss}" if miss else ""):
        return None
    ok = chk.add(scheme, f"{label}: code version", (t.code_version == CODE_VERSION).all(),
                 ";".join(sorted(t.code_version.astype(str).unique())))
    ok &= chk.add(scheme, f"{label}: model", (t.analysis_scheme == scheme).all(),
                  ";".join(sorted(t.analysis_scheme.astype(str).unique())))
    ok &= chk.add(scheme, f"{label}: one run_id", t.run_id.nunique() == 1,
                  ";".join(sorted(t.run_id.astype(str).unique())))
    if keys:
        dup = int(t.duplicated(keys).sum())
        ok &= chk.add(scheme, f"{label}: unique keys {keys}", dup == 0, f"{dup} duplicate keys")
        got = {tuple(_norm(v) for v in r) for r in t[keys].itertuples(index=False)}
        want = {tuple(_norm(v) for v in k) for k in expected}
        missing, extra = sorted(want - got), sorted(got - want)
        ok &= chk.add(scheme, f"{label}: expected rows ({len(want)})",
                      not missing and not extra and len(t) == len(want),
                      f"{len(t)} rows; missing {missing[:6]}{' ...' if len(missing) > 6 else ''}"
                      f"; unexpected {extra[:6]}")
    else:
        ok &= chk.add(scheme, f"{label}: {expected} row(s)", len(t) == expected, f"{len(t)} rows")
    for c in finite:
        v = pd.to_numeric(t[c], errors="coerce")
        bad = int((~np.isfinite(v)).sum())
        ok &= chk.add(scheme, f"{label}: finite {c}", bad == 0, f"{bad} missing/non-finite")
    for c in [c for c in t.columns if c.endswith("_km2")]:
        v = pd.to_numeric(t[c], errors="coerce")
        neg = int((v < -1e-9).sum())
        if neg:
            ok &= chk.add(scheme, f"{label}: non-negative {c}", False, f"{neg} negative")
    return t if ok else None


def bounds_checks(scheme: str, tab: dict, chk: Checks) -> None:
    """"""
    def leq(label, a, b):
        chk.add(scheme, f"bounds: {label}", area_leq(a, b), f"{a} <= {b}")

    ex = tab["Areas_Excess_by_Zclass.csv"]
    if ex is not None:
        for comp, g in ex.groupby("comparison"):
            g = g.sort_values("z_class")
            for col in ("observed_loss_km2", "observed_gain_km2", "prewar_benchmark_loss_km2",
                        "prewar_benchmark_gain_km2"):
                v = g[col].to_numpy(float)
                chk.add(scheme, f"bounds: {comp} {col} does not increase with the z class",
                        all(area_leq(v[i + 1], v[i]) for i in range(len(v) - 1)), str(list(v)))
            for col in ("observed_loss_km2", "observed_gain_km2"):
                chk.add(scheme, f"bounds: {comp} {col} <= analysed area",
                        all(area_leq(a, b) for a, b in zip(g[col], g.eligible_ground_area_km2)))
    cum = tab["Cumulative_Confidence_Class_Areas.csv"]
    if cum is not None:
        c = cum.assign(k=[_norm(x) for x in cum["class"]]).set_index(["variant", "set", "k"]).observed_km2
        for v in ("pixel", "mapped"):
            leq(f"{v} union_loss_ge2 <= ge1", c[(v, "union_loss_ge2", "nan")], c[(v, "union_loss_ge1", "nan")])
            leq(f"{v} union_gain_ge2 <= ge1", c[(v, "union_gain_ge2", "nan")], c[(v, "union_gain_ge1", "nan")])
            ov = c[(v, "overlap_loss_and_gain", "nan")]
            leq(f"{v} overlap <= loss union", ov, c[(v, "union_loss_ge1", "nan")])
            leq(f"{v} overlap <= gain union", ov, c[(v, "union_gain_ge1", "nan")])
            dl = sum(c[(v, "dominant_class", str(k))] for k in (1, 2, 3))
            dg = sum(c[(v, "dominant_class", str(k))] for k in (-1, -2, -3))
            leq(f"{v} dominant loss classes <= loss union", dl, c[(v, "union_loss_ge1", "nan")])
            leq(f"{v} dominant gain classes <= gain union", dg, c[(v, "union_gain_ge1", "nan")])
        for u in UNION_SETS:
            leq(f"mapped {u} <= pixel {u}", c[("mapped", u, "nan")], c[("pixel", u, "nan")])
        for k in (-3, -2, -1, 1, 2, 3):
            leq(f"mapped class {k} <= pixel class {k}", c[("mapped", "dominant_class", str(k))],
                c[("pixel", "dominant_class", str(k))])
    head = tab[f"Areas_Headline_{TAG}.csv"]
    if head is not None:
        h = head.set_index("variant")
        for col in ("damage_km2", "rewild_km2"):
            leq(f"{col}: all three screens <= two screens",
                h.loc["quality_screen_with_novelty_sensitivity", col],
                h.loc["quality_screen_without_novelty_sensitivity", col])
            leq(f"{col}: two screens <= all pixels",
                h.loc["quality_screen_without_novelty_sensitivity", col],
                h.loc["primary_all_valid_input_pixels", col])
            leq(f"{col}: without closing <= with closing", h.loc["stricter_no_closing", col],
                h.loc["primary_all_valid_input_pixels", col])
        p = h.loc["primary_all_valid_input_pixels"]
        for pre in ("damage", "rewild"):
            cols = [f"{pre}_km2", f"{pre}_exceedance_km2", f"{pre}_in_exceedance_km2"]
            if all(x in h.columns for x in cols):
                leq(f"{pre}: in-exceedance <= result", p[cols[2]], p[cols[0]])
                leq(f"{pre}: in-exceedance <= exceedances", p[cols[2]], p[cols[1]])
                leq(f"{pre}: without closing <= exceedances", h.loc["stricter_no_closing", f"{pre}_km2"],
                    p[cols[1]])
    sn = tab[f"Areas_Stricter_Null{SFX}.csv"]
    if sn is not None:
        for _, r in sn.iterrows():
            y = _norm(r.null_year)
            for side in ("loss", "gain"):
                res, pix = r[f"stricter_{side}_km2"], r[f"pixel_{side}_km2"]
                inn = r[f"stricter_{side}_in_exceedance_km2"]
                leq(f"pre-war {y} {side}: in-exceedance <= result", inn, res)
                leq(f"pre-war {y} {side}: in-exceedance <= exceedances", inn, pix)
                chk.add(scheme, f"bounds: pre-war {y} {side}: in-exceedance + removed = exceedances",
                        area_close(inn + r[f"pixel_{side}_removed_km2"], pix))
                leq(f"pre-war {y} {side}: without closing <= exceedances",
                    r[f"stricter_no_closing_{side}_km2"], pix)
                leq(f"pre-war {y} {side}: without closing <= result",
                    r[f"stricter_no_closing_{side}_km2"], res)
                leq(f"pre-war {y} {side}: screened <= result", r[f"stricter_{side}_screened_km2"], res)
    sc = tab[f"Areas_by_Screen{SFX}.csv"]
    if sc is not None:
        s2 = sc.set_index("statistic")
        for st, r in s2.iterrows():
            for col in SCREEN_COLS[1:]:
                leq(f"screens {st}: {col} <= all_pixels_km2", r[col], r.all_pixels_km2)
            for col in SCREEN_COLS[1:4]:
                leq(f"screens {st}: all three <= {col}", r.all_three_screens_km2, r[col])
            for col in SCREEN_COLS:
                leq(f"screens {st}: {col} <= domain", r[col], s2.loc["domain", col])
    grid = tab["Areas_vs_Threshold_and_PatchSize.csv"]
    if grid is not None:
        for col in ("damage_km2", "rewild_km2"):
            for p_, g in grid.groupby("min_patch_px"):
                v = g.sort_values("z_threshold")[col].to_numpy(float)
                chk.add(scheme, f"bounds: threshold table {col}, patch {p_:g}: no increase with z",
                        all(area_leq(v[i + 1], v[i]) for i in range(len(v) - 1)), str(list(v)))
            for z_, g in grid.groupby("z_threshold"):
                v = g.sort_values("min_patch_px")[col].to_numpy(float)
                chk.add(scheme, f"bounds: threshold table {col}, z {z_:g}: no increase with patch size",
                        all(area_leq(v[i + 1], v[i]) for i in range(len(v) - 1)), str(list(v)))
    for n in SENSITIVITY_TABLES:
        t = tab[f"Sensitivity_{n}{SFX}.csv"]
        if t is None:
            continue
        bad = [f"{r.design}/{r.statistic}: A={r.variant_km2}, B={r.main_km2}, I={r.overlap_with_main_km2}"
               for r in t.itertuples() if not valid_overlap(r.variant_km2, r.main_km2,
                                                            r.overlap_with_main_km2)]
        chk.add(scheme, f"bounds: Sensitivity {n}: 0 <= overlap <= min(variant, main)",
                not bad, "; ".join(bad[:5]))


REQUIRED_CV_SCHEMES = ["primary_effort", "access_matched", "recurrent_effort",
                       "recurrent_access_matched"]
PY01_VERDICT = f"CV_CHECKS_VERDICT{SFX}.json"
PY01_MARKER = "CHECKS_FAILED.txt"


def python01_verdict_checks(scheme: str, m: pd.Series, chk: Checks, info: dict) -> None:
    """"""
    d = CV_METRICS_DIR
    marker = d / PY01_MARKER
    chk.add(scheme, "Python_01: no CHECKS_FAILED.txt", not marker.exists(),
            marker.read_text(encoding="utf-8")[:300] if marker.exists() else "")
    vp = d / PY01_VERDICT
    if not chk.add(scheme, f"Python_01: {PY01_VERDICT} present", vp.exists(), str(vp)):
        return
    try:
        v = json.loads(vp.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        chk.add(scheme, "Python_01: verdict readable", False, str(err))
        return
    info["python01_verdict"] = {k: v.get(k) for k in ("created_utc", "passed", "input_dir")}
    chk.add(scheme, "Python_01: verdict code version", v.get("code_version") == CODE_VERSION,
            str(v.get("code_version")))
    chk.add(scheme, "Python_01: overall verdict passed", v.get("passed") is True,
            "; ".join(v.get("failed_checks", []))[:300])
    fm = v.get("final_models", {}).get(scheme)
    if chk.add(scheme, "Python_01: final-model check of this model in the verdict", fm is not None):
        chk.add(scheme, "Python_01: final-model check passed", fm.get("passed") is True)
        chk.add(scheme, "Python_01 checked the GEE_04 run used by GEE_05",
                str(fm.get("gee04_run_id")) == str(m.period_probability_run_id),
                f"{fm.get('gee04_run_id')} vs {m.period_probability_run_id}")
        if "period_probability_presence_score_mode" in m.index:
            chk.add(scheme, "probability assets use the verified score mode",
                    str(m.period_probability_presence_score_mode) == str(fm.get("presence_score_mode")),
                    f"{m.period_probability_presence_score_mode} vs {fm.get('presence_score_mode')}")
        else:
            chk.add(scheme, "GEE05 manifest records the asset score mode", False)
    cp = v.get("cv_partitions", {})
    missing = sorted(set(REQUIRED_CV_SCHEMES) - set(cp.get("schemes", [])))
    chk.add(scheme, "Python_01: every CV partition checked and passed",
            cp.get("n", 0) > 0 and cp.get("n_passed") == cp.get("n") and not missing,
            f"{cp.get('n_passed')} of {cp.get('n')} passed; schemes missing {missing}")
    files = v.get("files", {})
    need = {"final_vs_cv_training_rows.csv", "cv_scores_vs_manifest.csv"}
    chk.add(scheme, "Python_01: verdict covers both check tables", need.issubset(files),
            str(sorted(files)))
    for name, digest in files.items():
        path = d / name
        ok = path.exists() and sha256(path) == digest
        chk.add(scheme, f"Python_01: {name} unchanged since the verdict", ok, str(path))
        if ok and name == "cv_scores_vs_manifest.csv":
            t = pd.read_csv(path)
            chk.add(scheme, "Python_01: cv_scores_vs_manifest.csv all partitions passed",
                    len(t) > 0 and t["passed"].astype(str).str.lower().isin(["true", "1"]).all())
        if ok and name == "final_vs_cv_training_rows.csv":
            t = pd.read_csv(path)
            t = t[t.sampling_scheme == scheme]
            chk.add(scheme, "Python_01: final_vs_cv_training_rows.csv passed for this model",
                    len(t) == 1 and str(t["passed"].iloc[0]).lower() in ("true", "1"))


def collect(scheme: str, chk: Checks):
    d = folder(scheme)
    rows: list[dict] = []
    info: dict = {"folder": str(d) if d else None}
    if not chk.add(scheme, "GEE_05 output folder", d is not None, str(SCHEME_DIRS[scheme][0])):
        return rows, info
    T = THRESHOLD[scheme]

    def add(section, item, new, key=None, status=None):
        try:
            val = float(new)
        except (TypeError, ValueError):
            val = float("nan")
        rows.append({"model": scheme, "section": section, "item": item,
                     "v17": V17[scheme].get(key, np.nan) if key else np.nan,
                     "new": val, "value_status": status or ("ok" if math.isfinite(val) else "missing")})

    tab = {name: load_table(d, name, scheme, chk) for name in TABLES}
    bounds_checks(scheme, tab, chk)

    # ---- run identity -------------------------------------------------------------
    run_ids = {n: t.run_id.iloc[0] for n, t in tab.items() if t is not None}
    man = tab[f"GEE05_RunManifest{SFX}.csv"]
    if man is not None:
        m = man.iloc[0]
        info.update({"gee05_run_id": m.run_id,
                     "gee05_parameters": {k: (None if pd.isna(v) else v) for k, v in m.items()}})
        other = sorted({v for v in run_ids.values() if v != m.run_id})
        chk.add(scheme, "all tables carry the manifest run_id", not other, ";".join(other))
        asset_runs = {m.period_probability_run_id, m.annual_probability_run_id, m.novelty_run_id}
        chk.add(scheme, "probability and novelty assets from one GEE_04 run",
                len(asset_runs) == 1 and all(isinstance(x, str) for x in asset_runs),
                ";".join(map(str, asset_runs)))
        chk.add(scheme, "probability assets written by GEE_04 v13b12",
                m.period_probability_code_version == CODE_VERSION,
                str(m.period_probability_code_version))
        info["gee04_run_id_on_assets"] = m.period_probability_run_id
        # The GEE_04 manifest of the run whose assets GEE_05 read, selected by
        # its run_id (not by name or date): copies of that manifest (several
        # downloads of the same Drive folder) must be identical; manifests of
        # other GEE_04 runs are listed and ignored.
        name04 = f"GEE04_RunManifest_{scheme}{SFX}.csv"
        _, st04, c04 = find_unique([BASE_DIR], name04)
        c04 = c04 if c04 else []
        runs04 = {}
        for pth in c04:
            try:
                runs04[pth] = str(pd.read_csv(pth).iloc[0].get("run_id"))
            except Exception as err:          # unreadable copy
                runs04[pth] = f"unreadable ({err})"
        want = str(m.period_probability_run_id)
        match = [pth for pth, r in runs04.items() if r == want]
        other = [f"{pth} ({r})" for pth, r in runs04.items() if r != want]
        if chk.add(scheme, f"{name04} of the GEE_04 run used by GEE_05 found", bool(match),
                   f"wanted {want}; found " + ("; ".join(f"{p_} ({r})" for p_, r in runs04.items())
                                               or "no copy")):
            digests = {sha256(pth) for pth in match}
            chk.add(scheme, f"{name04}: copies of that run are identical", len(digests) == 1,
                    "; ".join(map(str, match)))
            info["gee04_manifest"] = [str(pth) for pth in match]
            info["gee04_manifests_of_other_runs_ignored"] = other
            chk.add(scheme, "GEE_04 manifest run_id = run_id on the assets", True, want)
        python01_verdict_checks(scheme, m, chk, info)

    ex = tab["Areas_Excess_by_Zclass.csv"]
    cum = tab["Cumulative_Confidence_Class_Areas.csv"]
    head = tab[f"Areas_Headline_{TAG}.csv"]
    diag = tab[f"Debiased_RF_Diagnostics_revision_{TAG}.csv"]
    strict = tab[f"Areas_Stricter_Null{SFX}.csv"]
    screens = tab[f"Areas_by_Screen{SFX}.csv"]
    floor = tab[f"SD_Floor_Diagnostics{SFX}.csv"]
    grid = tab["Areas_vs_Threshold_and_PatchSize.csv"]
    sens = {n: tab[f"Sensitivity_{n}{SFX}.csv"] for n in SENSITIVITY_TABLES}

    # ---- headline values used by the acceptance checks ---------------------------------
    headline_stat: dict[str, float] = {}

    def excess_row(comp):
        r = ex[(ex.comparison == comp) & np.isclose(ex.z_class, T)]
        chk.add(scheme, f"Areas_Excess_by_Zclass: row {comp} at T={T:g}", len(r) == 1, f"{len(r)} rows")
        return r.iloc[0] if len(r) == 1 else None

    if ex is not None:
        add("Table 2", "Analysed area (mapping mask), km2", ex.eligible_ground_area_km2.iloc[0])
        for comp, label, key in (("median_headline", "Median", "median"), ("year_2022", "2022", "2022"),
                                 ("year_2023", "2023", "2023"), ("year_2024", "2024", "2024")):
            r = excess_row(comp)
            if r is None:
                continue
            stat = "median" if comp == "median_headline" else comp
            for side in ("loss", "gain"):
                headline_stat[f"{stat}_{side}"] = r[f"observed_{side}_km2"]
                add("Table 2", f"{label}, {side} detected (z beyond {T:g})",
                    r[f"observed_{side}_km2"], f"{key}_{side}")
                add("Table 2", f"{label}, {side} pre-war benchmark", r[f"prewar_benchmark_{side}_km2"],
                    f"{key}_{side}_bench" if key == "median" else None)
                ratio, st = safe_ratio(r[f"prewar_benchmark_{side}_km2"], r[f"observed_{side}_km2"])
                add("Table 2", f"{label}, {side} benchmark / detected", ratio, status=st)
        r = ex[(ex.comparison == "year_2022") & np.isclose(ex.z_class, 3.0)]
        if len(r) == 1:
            add("Table 2", "2022 at |z| >= 3, gain detected", r.observed_gain_km2.iloc[0])
            add("Table 2", "2022 at |z| >= 3, gain pre-war benchmark", r.prewar_benchmark_gain_km2.iloc[0])
    if diag is not None:
        dg = diag.set_index("period")
        for k in ("median", "2022", "2023", "2024", "null_2019", "null_2020", "null_2021"):
            ok = k in dg.index
            chk.add(scheme, f"Diagnostics: period {k}", ok)
            if ok:
                add("Table 2", f"mean z-score, {k}", dg.loc[k, "z_cal_mean"])

    if cum is not None:
        mp = cum[cum.variant == "mapped"]
        dom = mp[mp["set"] == "dominant_class"].copy()
        dom["class"] = pd.to_numeric(dom["class"], errors="coerce")
        dom = dom.set_index("class")
        ok = chk.add(scheme, "Year-count classes -3..3 present",
                     set(dom.index) >= {-3, -2, -1, 1, 2, 3}, str(sorted(dom.index)))
        if ok:
            for k in (1, 2, 3):
                add("Table 2", f"dominant class, loss in {k} year(s)", dom.loc[k, "observed_km2"], f"dom{k}_loss")
                add("Table 2", f"dominant class, gain in {k} year(s)", dom.loc[-k, "observed_km2"], f"dom{k}_gain")
            headline_stat["dominant_loss_mapped"] = dom.loc[[1, 2, 3], "observed_km2"].sum()
            headline_stat["dominant_gain_mapped"] = dom.loc[[-1, -2, -3], "observed_km2"].sum()
            headline_stat["dominant_loss_ge2_mapped"] = dom.loc[[2, 3], "observed_km2"].sum()
            headline_stat["dominant_gain_ge2_mapped"] = dom.loc[[-2, -3], "observed_km2"].sum()
        for st in ("union_loss_ge1", "union_loss_ge2", "union_gain_ge1", "union_gain_ge2",
                   "overlap_loss_and_gain"):
            r = mp[mp["set"] == st]
            if chk.add(scheme, f"Year counts: {st} (mapped) present", len(r) == 1, f"{len(r)} rows"):
                add("Table 2", f"{st} (patch-filtered)", r.observed_km2.iloc[0])
                headline_stat[f"{st}_mapped"] = r.observed_km2.iloc[0]

    if head is not None:
        h = head.set_index("variant")
        need = ["primary_all_valid_input_pixels", "quality_screen_with_novelty_sensitivity",
                "stricter_no_closing"]
        if chk.add(scheme, "Areas_Headline rows", all(v in h.index for v in need)):
            p = h.loc["primary_all_valid_input_pixels"]
            headline_stat["stricter_loss"] = p.damage_km2
            headline_stat["stricter_gain"] = p.rewild_km2
            add("Table 2", "stricter version, loss", p.damage_km2, "stricter_loss")
            add("Table 2", "stricter version, gain", p.rewild_km2)
            add("Table 2", "stricter version, loss, all three screens",
                h.loc["quality_screen_with_novelty_sensitivity", "damage_km2"], "stricter_loss_screens")
            add("Table 2", "stricter version without closing, loss", h.loc["stricter_no_closing", "damage_km2"])
            add("Table 2", "stricter version without closing, gain", h.loc["stricter_no_closing", "rewild_km2"])
            for side, pre in (("loss", "damage"), ("gain", "rewild")):
                cols = [f"{pre}_exceedance_km2", f"{pre}_in_exceedance_km2",
                        f"{pre}_added_by_closing_km2", f"{pre}_exceedance_removed_km2"]
                if not chk.add(scheme, f"Areas_Headline: split columns ({side})",
                               all(c in h.columns for c in cols)):
                    continue
                exc, inn, addd, rem = (p[c] for c in cols)
                chk.add(scheme, f"stricter {side}: in-exceedance + added = result",
                        close(inn + addd, p[f"{pre}_km2"]), f"{inn}+{addd} vs {p[f'{pre}_km2']}")
                chk.add(scheme, f"stricter {side}: in-exceedance + removed = exceedances",
                        close(inn + rem, exc), f"{inn}+{rem} vs {exc}")
                add("Methods", f"stricter {side}: pixel exceedances (z beyond 3)", exc)
                add("Methods", f"stricter {side}: cells added by the closing", addd)
                add("Methods", f"stricter {side}: exceedances removed by the rules", rem)
                r_, s_ = safe_ratio(inn, exc)
                add("Methods", f"stricter {side}: retained share of the exceedances", r_, status=s_)
                r_, s_ = safe_ratio(p[f"{pre}_km2"], exc)
                add("Methods", f"stricter {side}: result / exceedance area ratio (not a share)", r_, status=s_)

    if strict is not None:
        mean = strict.select_dtypes("number").mean()
        for side in ("loss", "gain"):
            for yr, r in strict.iterrows():
                chk.add(scheme, f"pre-war {r.null_year} stricter {side}: in-exceedance + added = result",
                        close(r[f"stricter_{side}_in_exceedance_km2"] + r[f"stricter_{side}_added_km2"],
                              r[f"stricter_{side}_km2"]))
            add("Table 2", f"stricter rules on pre-war maps, {side} (mean of 3 years; benchmark)",
                mean[f"stricter_{side}_km2"])
            r_, s_ = safe_ratio(mean[f"stricter_{side}_in_exceedance_km2"], mean[f"pixel_{side}_km2"])
            add("Methods", f"pre-war maps, {side}: retained share of exceedances", r_, status=s_)
            r_, s_ = safe_ratio(mean[f"stricter_{side}_km2"], mean[f"pixel_{side}_km2"])
            add("Methods", f"pre-war maps, {side}: result / exceedance area ratio (not a share)",
                r_, status=s_)
            add("Methods", f"pre-war maps, {side}: cells added by the closing (mean)",
                mean[f"stricter_{side}_added_km2"])
        add("Table 2", "stricter rules on pre-war maps, loss, all three screens (benchmark)",
            mean["stricter_loss_screened_km2"])
        if head is not None and "primary_all_valid_input_pixels" in head.variant.values:
            p = head.set_index("variant").loc["primary_all_valid_input_pixels"]
            for side, pre in (("loss", "damage"), ("gain", "rewild")):
                r_, s_ = safe_ratio(mean[f"stricter_{side}_km2"], p[f"{pre}_km2"])
                add("Table 2", f"stricter version, {side}: pre-war benchmark / detected", r_, status=s_)

    # ---- acceptance: sensitivity tables against the headline ----------------------------
    main = sens["main_main"]
    main_by_stat = {}
    if main is not None:
        mm = main.set_index("statistic")
        main_by_stat = mm.main_km2.to_dict()
        for st, r in mm.iterrows():
            chk.add(scheme, f"Sensitivity main: {st} variant = main = overlap",
                    close(r.variant_km2, r.main_km2) and close(r.overlap_with_main_km2, r.main_km2),
                    f"{r.variant_km2} / {r.main_km2} / {r.overlap_with_main_km2}")
        for st, val in headline_stat.items():
            if st in mm.index:
                # different GEE_05 reduction: equal within one cell (CELL_KM2)
                d = float(mm.loc[st, "main_km2"]) - float(val)
                chk.add(scheme, f"Sensitivity main reproduces headline: {st}",
                        cell_close(mm.loc[st, "main_km2"], val),
                        f"{mm.loc[st, 'main_km2']} vs {val} (difference {d:.4f} km2)")
            else:
                chk.add(scheme, f"Sensitivity main has statistic {st}", False)
    for n, t in sens.items():
        if t is None or n == "main_main" or not main_by_stat:
            continue
        for (design, st), g in t.groupby(["design", "statistic"]):
            chk.add(scheme, f"Sensitivity {n}/{design}: main area of {st} = main table",
                    close(g.main_km2.iloc[0], main_by_stat.get(st, np.nan)))
    if grid is not None and "stricter_loss" in headline_stat:
        g = grid[grid.is_main_parameters == 1]
        if chk.add(scheme, "Threshold/patch table: one main-parameter row", len(g) == 1, f"{len(g)} rows"):
            for side, col in (("loss", "damage_km2"), ("gain", "rewild_km2")):
                a, b = g[col].iloc[0], headline_stat[f"stricter_{side}"]
                chk.add(scheme, f"Threshold/patch main row = stricter {side}",
                        cell_close(a, b), f"{a} vs {b}")

    # ---- sensitivity numbers --------------------------------------------------------------
    for n, t in sens.items():
        if t is None or n == "main_main":
            continue
        for _, r in t.iterrows():
            base = f"{r.family} {r.variant} ({r.design}), {r.statistic}"
            key = None
            if r.statistic == "stricter_loss":
                if r.family == "dropped_year" and r.design == "variance_only":
                    key = f"drop{str(r.variant).replace('without_', '')}_loss"
                elif r.family == "bias_sigma":
                    key = f"sigma{str(r.variant).replace('m', '')}_loss"
            add("Table S5", base + ", km2", r.variant_km2, key)
            ratio, st = safe_ratio(r.variant_km2, r.main_km2)
            add("Table S5", base + ", ratio to main", ratio, status=st)
            jac, st = jaccard(r.variant_km2, r.main_km2, r.overlap_with_main_km2)
            add("Table S5", base + ", Jaccard with main", jac, status=st)

    if floor is not None:
        add("Methods", "SD floor: share of pixels at the floor", floor.share_pixels_at_floor.iloc[0])
    if screens is not None:
        sc = screens.set_index("statistic")
        for st in ("median_loss", "year_2024_loss", "year_2022_gain", "union_loss_mapped",
                   "union_gain_mapped", "class_loss2_mapped", "stricter_loss"):
            if chk.add(scheme, f"Areas_by_Screen: statistic {st}", st in sc.index):
                for col in ("all_pixels_km2", "all_three_screens_km2", "at_sd_floor_km2"):
                    add("Screens", f"{st}, {col.replace('_km2', '')}", sc.loc[st, col])
    return rows, info


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(infos: dict, status: str) -> dict:
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "code_version": CODE_VERSION, "status": status, "code": {}, "outputs": {}}
    for p in sorted(CODE_DIR.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".js", ".md", ".txt"):
            out["code"][str(p.relative_to(CODE_DIR))] = sha256(p)
    for scheme, info in infos.items():
        entry = dict(info)
        d = Path(info["folder"]) if info.get("folder") else None
        if d is not None:
            entry["files"] = {str(p.relative_to(d)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                              for p in sorted(d.rglob("*"))
                              if p.is_file() and p.suffix.lower() in (".csv", ".tif", ".tiff")}
        out["outputs"][scheme] = entry
    return out


def change_status(v17, new, value_status):
    """missing (no value and no reason), a stated reason such as
    no_detections (an undefined ratio), new (not printed in v17), changed or
    unchanged. No missing value is ever treated as zero."""
    if not math.isfinite(new):
        return "missing" if value_status in ("ok", "missing") else value_status
    if not math.isfinite(v17):
        return "new"
    return "unchanged" if np.isclose(v17, new, rtol=0.005, atol=0.05) else "changed"


def main() -> int:
    chk = Checks()
    rows, infos = [], {}
    for scheme in THRESHOLD:
        r, info = collect(scheme, chk)
        rows += r
        infos[scheme] = info
    out = pd.DataFrame(rows, columns=["model", "section", "item", "v17", "new", "value_status"])
    out["status"] = [change_status(a, b, s) for a, b, s in zip(out.v17, out.new, out.value_status)]
    # A value is allowed to be NA only when its status says why (e.g. no_detections).
    for r in out[out.status == "missing"].itertuples():
        chk.add(r.model, f"value present: {r.item}", False, "missing value")
    checks = pd.DataFrame(chk.rows)
    Path(OUT_CHECKS).parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(OUT_CHECKS, index=False)
    failed = chk.failed
    status = "failed" if failed else "passed"
    OUT_MANIFEST.write_text(json.dumps(manifest(infos, status), indent=1, default=str),
                            encoding="utf-8")
    print(f"CHECKS: {len(checks) - len(failed)} passed, {len(failed)} failed  ({OUT_CHECKS})")
    if failed:
        out.to_csv(OUT_FAILED, index=False)
        if OUT_OK.exists():
            OUT_OK.unlink()
            print(f"  removed the earlier {OUT_OK.name}, which no longer matches the outputs")
        for f in failed[:40]:
            print(f"  FAIL [{f['model']}] {f['check']}  {f['detail']}")
        if len(failed) > 40:
            print(f"  ... {len(failed) - 40} more in {OUT_CHECKS.name}")
        print(f"\nNumbers NOT accepted; diagnostic copy: {OUT_FAILED}")
        return 1
    if OUT_FAILED.exists():
        OUT_FAILED.unlink()
    out.to_csv(OUT_OK, index=False)
    print(f"Numbers: {OUT_OK}\nManifest: {OUT_MANIFEST}")
    print("\nChanged against manuscript v17:")
    with pd.option_context("display.width", 160, "display.max_colwidth", 90):
        print(out[out.status == "changed"][["model", "item", "v17", "new"]].to_string(
            index=False, float_format=lambda v: f"{v:,.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
