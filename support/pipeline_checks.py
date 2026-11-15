#!/usr/bin/env python3
"""

Imported by Python_01, Python_03 and the support scripts; tested by
tests/test_pipeline_checks.py. Every function is pure (no file writing).

"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CODE_VERSION = "v13b12"
FP_RTOL = 1e-9          # fingerprints are sums of 64 doubles
FP_ATOL = 1e-6
PROB_TOL = 1e-6         # PROBABILITY vs label-1 column
# Share of clear rows (|p0 - p1| > margin) on which the argmax of the label-1
# column must match CLASSIFICATION == 1, on the fitting rows and on the mapped
# sample, while the other column matches on at most 1 - this share. The class
# output and the class probabilities of a random forest are not computed the
# same way (the class is decided by the trees' votes, the probabilities are
# 32 of 14,438 fitting rows (0.22%) of the regular model and 3 of 4,097 of
# the recurrent model did, and none of the 5,811 / 5,914 mapped pixels. 0.99
# still rejects any unclear assignment; the decisive test remains that
# PROBABILITY equals the identified column (< PROB_TOL) in both predicted
# groups.
LABEL_AGREEMENT = 0.99
AREA_ABS_TOL = 0.01     # km2; the same pixels summed by different reductions
AREA_REL_TOL = 1e-6
# One 500 m cell of the EPSG:3035 grid. The headline areas of GEE_05
# (Areas_Excess_by_Zclass, areaKm2: selfMask) and the main rows of the
# sensitivity and threshold/patch tables (finishMasks + sumAreasKm2:
# unmask(0).gt(0)) count the same z maps but treat cells cut by the AOI edge
# 2 Oct 2026: 0.076 km2 in the 2024 loss of the regular model).
CELL_KM2 = 0.25

# The 19 statistics of every Sensitivity_*_v13b12 table (GEE_05 SENS_STAT_NAMES).
SENS_STAT_NAMES = [
    "median_loss", "median_gain",
    "year_2022_loss", "year_2022_gain", "year_2023_loss", "year_2023_gain",
    "year_2024_loss", "year_2024_gain",
    "dominant_loss_mapped", "dominant_gain_mapped",
    "dominant_loss_ge2_mapped", "dominant_gain_ge2_mapped",
    "union_loss_ge1_mapped", "union_loss_ge2_mapped",
    "union_gain_ge1_mapped", "union_gain_ge2_mapped",
    "overlap_loss_and_gain_mapped",
    "stricter_loss", "stricter_gain",
]


# ---------------------------------------------------------------------------
# ratios
# ---------------------------------------------------------------------------

def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def safe_ratio(numerator, denominator) -> tuple[float, str]:
    """numerator / denominator, or (NaN, status). Status is 'ok',
    'no_detections' (denominator 0) or 'missing' (a value is absent)."""
    if not (_finite(numerator) and _finite(denominator)):
        return float("nan"), "missing"
    den = float(denominator)
    if den == 0:
        return float("nan"), "no_detections"
    return float(numerator) / den, "ok"


def _tol(*values) -> float:
    return AREA_ABS_TOL + AREA_REL_TOL * max([abs(float(v)) for v in values] + [0.0])


def area_close(a, b) -> bool:
    return _finite(a) and _finite(b) and abs(float(a) - float(b)) <= _tol(a, b)


def cell_close(a, b) -> bool:
    """a = b within one grid cell; for the same z map summed by the two
    GEE_05 reductions (see CELL_KM2)."""
    return (_finite(a) and _finite(b)
            and abs(float(a) - float(b)) <= CELL_KM2 + _tol(a, b))


def area_leq(a, b) -> bool:
    """a <= b within the area tolerance (both finite)."""
    return _finite(a) and _finite(b) and float(a) <= float(b) + _tol(a, b)


def valid_overlap(a, b, i) -> bool:
    """A >= 0, B >= 0 and 0 <= I <= min(A, B), within the tolerance."""
    return (all(_finite(v) for v in (a, b, i)) and area_leq(0, a) and area_leq(0, b)
            and area_leq(0, i) and area_leq(i, a) and area_leq(i, b))


def jaccard(variant_km2, main_km2, overlap_km2) -> tuple[float, str]:
    """|A and B| / |A or B|. NaN with status 'missing' (a value absent),
    'invalid_areas' (negative area or overlap larger than an area: the input
    is impossible and is reported, not clipped) or 'both_empty'."""
    if not all(_finite(v) for v in (variant_km2, main_km2, overlap_km2)):
        return float("nan"), "missing"
    if not valid_overlap(variant_km2, main_km2, overlap_km2):
        return float("nan"), "invalid_areas"
    union = float(variant_km2) + float(main_km2) - float(overlap_km2)
    if union <= _tol(variant_km2, main_km2):
        return float("nan"), "both_empty"
    return min(max(float(overlap_km2) / union, 0.0), 1.0), "ok"


# ---------------------------------------------------------------------------
# row identity
# ---------------------------------------------------------------------------

def multiset_compare(a: Iterable, b: Iterable) -> dict:
    """Compare two lists of keys as multisets (duplicates count)."""
    ca, cb = Counter(a), Counter(b)
    only_a = ca - cb
    only_b = cb - ca
    return {
        "n_a": int(sum(ca.values())),
        "n_b": int(sum(cb.values())),
        "n_distinct_a": len(ca),
        "n_distinct_b": len(cb),
        "n_keys_repeated_a": sum(1 for v in ca.values() if v > 1),
        "n_keys_repeated_b": sum(1 for v in cb.values() if v > 1),
        "n_unmatched_a": int(sum(only_a.values())),
        "n_unmatched_b": int(sum(only_b.values())),
        "identical": ca == cb,
        "examples_unmatched_a": [str(k) for k in list(only_a)[:5]],
        "examples_unmatched_b": [str(k) for k in list(only_b)[:5]],
    }


def coordinate_keys(frame: pd.DataFrame, digits: int = 7) -> list:
    lon = pd.to_numeric(frame["lon"], errors="coerce").round(digits)
    lat = pd.to_numeric(frame["lat"], errors="coerce").round(digits)
    pres = pd.to_numeric(frame["presence"], errors="coerce")
    return list(zip(lon, lat, pres))


def _ids(frame: pd.DataFrame) -> pd.Series:
    return frame["row_id"].astype(str).str.strip()


def _fp_equal(a, b) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.isfinite(a) & np.isfinite(b) & np.isclose(a, b, rtol=FP_RTOL, atol=FP_ATOL)


def index_manifest(manifest: pd.DataFrame | None, label: str = "CV row manifest"):
    """The manifest indexed by row_id, or (None, reasons). row_id must be
    present, non-missing and unique; presence and fingerprint present."""
    reasons: list[str] = []
    need = {"row_id", "presence", "emb_fingerprint"}
    if manifest is None:
        return None, [f"{label} missing"]
    if not need.issubset(manifest.columns):
        return None, [f"{label} lacks {sorted(need - set(manifest.columns))}"]
    if manifest["row_id"].isna().any():
        reasons.append(f"{label}: {int(manifest['row_id'].isna().sum())} rows without row_id")
    ids = _ids(manifest)
    if ids.duplicated().any():
        reasons.append(f"{label}: row_id not unique ({int(ids.duplicated().sum())} repeats)")
    if reasons:
        return None, reasons
    return manifest.assign(_id=ids).set_index("_id"), []


def compare_to_manifest(rows: pd.DataFrame, man: pd.DataFrame, fold_rule: str | None = None,
                        fold_col: str = "fold_id") -> dict:
    """Align rows to the indexed manifest by row_id and count disagreements:
    rows without row_id, row_ids not in the manifest, presence labels,
    predictor fingerprints (missing counts as a mismatch) and, with
    fold_rule 'same' / 'different', the row's fold against the manifest's
    spatial_fold (held-out rows must be in their own fold; training rows of a
    fold must come from the other folds)."""
    out = {"n_rows": int(len(rows))}
    has_id = rows["row_id"].notna() if "row_id" in rows.columns else pd.Series(False, index=rows.index)
    out["n_without_row_id"] = int((~has_id).sum())
    r = rows[has_id].assign(_id=lambda f: _ids(f))
    known = r["_id"].isin(man.index)
    out["n_not_in_manifest"] = int((~known).sum())
    r = r[known]
    m = man.loc[r["_id"]]
    lab_r = pd.to_numeric(r["presence"], errors="coerce").to_numpy(float)
    lab_m = pd.to_numeric(m["presence"], errors="coerce").to_numpy(float)
    out["n_label_mismatch"] = int((~(np.isfinite(lab_r) & (lab_r == lab_m))).sum())
    fp_r = pd.to_numeric(r.get("emb_fingerprint"), errors="coerce") if "emb_fingerprint" in r \
        else pd.Series(np.nan, index=r.index)
    out["n_fingerprint_mismatch"] = int((~_fp_equal(fp_r, pd.to_numeric(m["emb_fingerprint"],
                                                                         errors="coerce"))).sum())
    if fold_rule:
        f_r = pd.to_numeric(r.get(fold_col), errors="coerce").to_numpy(float)
        f_m = pd.to_numeric(m.get("spatial_fold"), errors="coerce").to_numpy(float)
        same = np.isfinite(f_r) & np.isfinite(f_m) & (f_r == f_m)
        bad = ~same if fold_rule == "same" else (same | ~np.isfinite(f_r) | ~np.isfinite(f_m))
        out["n_fold_mismatch"] = int(bad.sum())
    return out


def _content_reasons(prefix: str, c: dict) -> list[str]:
    msgs = {"n_without_row_id": "rows without row_id", "n_not_in_manifest": "row_ids not in the manifest",
            "n_label_mismatch": "presence labels differ from the manifest",
            "n_fingerprint_mismatch": "predictor fingerprints differ from the manifest",
            "n_fold_mismatch": "folds disagree with the manifest"}
    return [f"{prefix}: {c[k]} {msg}" for k, msg in msgs.items() if c.get(k)]


def row_identity_check(final: pd.DataFrame | None, manifest: pd.DataFrame | None,
                       cv_test: pd.DataFrame | None) -> dict:
    """Final-model rows and the pooled main blocked CV test scores, each
    against the CV row manifest by row_id: every manifest row exactly once,
    and for every row the presence label, the predictor fingerprint and (for
    the CV scores) the held-out fold equal the manifest's. Separate counts
    for final-vs-manifest (final_*) and scores-vs-manifest (cv_*). Score rows
    without row_id (an older export or a mix of versions) fail."""
    out: dict = {"passed": False, "reasons": []}
    reasons = out["reasons"]
    man, why = index_manifest(manifest)
    reasons += why
    need = {"row_id", "presence", "emb_fingerprint"}
    if final is None:
        reasons.append("final-model fitted-point table missing")
    elif not need.issubset(final.columns):
        reasons.append(f"final-model table lacks {sorted(need - set(final.columns))}")
    if cv_test is None or len(cv_test) == 0:
        reasons.append("CV test scores missing")
    if reasons:
        return out
    man_ids = pd.Series(man.index)

    fin_ids = _ids(final.dropna(subset=["row_id"]))
    cmp = multiset_compare(fin_ids, man_ids)
    out.update({
        "n_final_rows": int(len(final)), "n_manifest_rows": cmp["n_b"],
        "n_final_ids_repeated": cmp["n_keys_repeated_a"],
        "n_final_only": cmp["n_unmatched_a"], "n_manifest_not_in_final": cmp["n_unmatched_b"],
        "examples_final_only": ";".join(cmp["examples_unmatched_a"]),
        "examples_manifest_not_in_final": ";".join(cmp["examples_unmatched_b"]),
    })
    if cmp["n_keys_repeated_a"]:
        reasons.append("final-model row_id not unique (an ID must identify one row)")
    if not cmp["identical"]:
        reasons.append("final-model rows and CV rows differ (see counts)")
    c = compare_to_manifest(final, man)
    out.update({f"final_{k}": v for k, v in c.items()})
    reasons += _content_reasons("final-model rows", c)

    has_id = cv_test["row_id"].notna() if "row_id" in cv_test.columns else pd.Series(False, index=cv_test.index)
    if not has_id.all():
        n_missing = int((~has_id).sum())
        cc = multiset_compare(coordinate_keys(cv_test), coordinate_keys(manifest))
        out.update({"cv_scores_method": "coordinate_multiset (diagnostic only)",
                    "cv_n_rows_without_row_id": n_missing,
                    "cv_n_coordinate_unmatched": cc["n_unmatched_a"] + cc["n_unmatched_b"]})
        reasons.append(f"{n_missing} CV test rows lack row_id (older GEE_03 export or a mix "
                       "of versions); rerun GEE_03 v13b12")
    else:
        out["cv_scores_method"] = "row_id"
        cc = multiset_compare(_ids(cv_test), man_ids)
        out.update({"cv_n_test_rows": cc["n_a"], "cv_n_scored_more_than_once": cc["n_keys_repeated_a"],
                    "cv_n_not_scored": cc["n_unmatched_b"]})
        if not cc["identical"]:
            reasons.append("CV test scores do not cover the manifest rows exactly once")
        c = compare_to_manifest(cv_test, man, fold_rule="same")
        out.update({f"cv_{k}": v for k, v in c.items()})
        reasons += _content_reasons("CV test scores", c)
        man_runs = set(manifest["run_id"].astype(str)) if "run_id" in manifest.columns else set()
        cv_runs = (set(cv_test["run_id"].astype(str)) if "run_id" in cv_test.columns
                   and cv_test["run_id"].notna().all() else None)
        if cv_runs is None or len(cv_runs) != 1 or cv_runs != man_runs or len(man_runs) != 1:
            reasons.append("CV test scores do not all come from the GEE_03 run of the manifest")
    out["passed"] = not reasons
    return out


PARTITION_COLS = ["analysis_type", "model", "repeat_id", "block_size", "min_leaf",
                  "variables_per_split"]
# Partitions whose folds are the manifest's main blocked folds.
MAIN_FOLD_TYPES = {"main_fixed", "regularisation_sensitivity"}


def cv_scores_vs_manifest(scores: pd.DataFrame, manifest: pd.DataFrame | None) -> pd.DataFrame:
    """Every CV partition of one scheme (model x analysis type x setting)
    against the scheme's row manifest. Held-out rows: every manifest row
    exactly once, labels and fingerprints equal, and for the main blocked
    folds the fold equal to the manifest's. Training rows: in the manifest,
    labels and fingerprints equal, never in their own held-out fold (for the
    main blocked folds: never from the manifest's fold). Metadata: one run_id
    equal to the manifest's, one code version, one score mode and label-1
    index. One row per partition with counts, 'passed' and 'problems'."""
    man, why = index_manifest(manifest)
    rows = []
    if man is None:
        return pd.DataFrame([{"passed": False, "problems": " | ".join(why)}])
    man_ids = pd.Series(man.index)
    man_run = set(manifest["run_id"].astype(str)) if "run_id" in manifest.columns else set()
    keys = [c for c in PARTITION_COLS if c in scores.columns]
    for key, g in scores.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        problems: list[str] = []
        test = g[g["set"] == "test"]
        train = g[g["set"] == "train"]
        main_folds = row.get("analysis_type") in MAIN_FOLD_TYPES
        if test["row_id"].isna().any() if "row_id" in test else True:
            problems.append("held-out rows without row_id")
        else:
            cc = multiset_compare(_ids(test), man_ids)
            row.update({"n_test": cc["n_a"], "n_test_repeated": cc["n_keys_repeated_a"],
                        "n_not_scored": cc["n_unmatched_b"]})
            if not cc["identical"]:
                problems.append("held-out rows do not cover the manifest exactly once")
        ct = compare_to_manifest(test, man, fold_rule="same" if main_folds else None)
        row.update({f"test_{k}": v for k, v in ct.items()})
        problems += _content_reasons("held-out rows", ct)
        ctr = compare_to_manifest(train, man, fold_rule="different" if main_folds else None)
        row.update({f"train_{k}": v for k, v in ctr.items()})
        problems += _content_reasons("training rows", ctr)
        # leakage: a row may not be both held out and trained on in one fold
        if "row_id" in g.columns and "fold_id" in g.columns:
            te = set(zip(test["fold_id"], _ids(test.dropna(subset=["row_id"]))))
            tr = set(zip(train["fold_id"], _ids(train.dropna(subset=["row_id"]))))
            row["n_rows_trained_and_held_out_in_one_fold"] = len(te & tr)
            if te & tr:
                problems.append("rows used for training and testing in the same fold")
        for col, label in (("run_id", "run_id"), ("code_version", "code version"),
                           ("presence_score_mode", "score mode"), ("class1_index", "label-1 index")):
            vals = set(g[col].astype(str)) if col in g.columns and g[col].notna().all() else None
            row[col] = ";".join(sorted(vals)) if vals else ""
            if vals is None or len(vals) != 1:
                problems.append(f"{label} missing or not single-valued")
        if row.get("run_id") and set([row["run_id"]]) != man_run:
            problems.append("scores are not from the GEE_03 run of the manifest")
        if row.get("code_version") and row["code_version"] != CODE_VERSION:
            problems.append(f"code version {row['code_version']} is not {CODE_VERSION}")
        row["passed"] = not problems
        row["problems"] = " | ".join(problems)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# probability mode
# ---------------------------------------------------------------------------

def probability_mode_verdict(check: dict | None) -> dict:
    """

    1. Label-1 column: the MULTIPROBABILITY column whose argmax matches
       CLASSIFICATION == 1 on (nearly) all rows with a clear winner, on the
       fitting rows AND the mapped sample; the other column must not.
    2. PROBABILITY equals that column (max |diff| < PROB_TOL) in BOTH
       predicted groups of both samples, each group non-empty.
    Agreement with the other column, or group means, never pass the check.
    If PRESENCE_SCORE_MODE is MULTIPROBABILITY_CLASS1, the configured index
    must equal the verified label-1 column instead of step 2.
    """
    out = {"probability_output_is_presence_probability": False,
           "label1_column": np.nan, "reasons": []}
    reasons = out["reasons"]
    if check is None:
        reasons.append("SDM_ProbabilityModeCheck table missing")
        return out

    def num(key):
        v = check.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    col = None
    for c in (0, 1):
        ok = all(num(f"{s}_share_argmax_col{c}_is_label1") >= LABEL_AGREEMENT
                 and num(f"{s}_share_argmax_col{1 - c}_is_label1") <= 1 - LABEL_AGREEMENT
                 and num(f"{s}_n_clear") > 0 for s in ("train", "mapped"))
        if ok:
            col = c
    if col is None:
        reasons.append("label-1 column of MULTIPROBABILITY not identified on both samples")
        return out
    out["label1_column"] = col
    for s_ in ("train", "mapped"):
        out[f"{s_}_share_label1_column_matches_class"] = num(f"{s_}_share_argmax_col{col}_is_label1")
    mode = str(check.get("presence_score_mode", "PROBABILITY"))
    out["presence_score_mode"] = mode
    if mode == "MULTIPROBABILITY_CLASS1":
        conf = num("configured_class1_index")
        if conf != col:
            reasons.append(f"configured CLASS1_INDEX {conf:g} is not the label-1 column {col}")
        out["probability_output_is_presence_probability"] = not reasons
        return out
    for s in ("train", "mapped"):
        for g in (1, 0):
            n = num(f"{s}_pred{g}_n")
            d = num(f"{s}_pred{g}_max_abs_diff_col{col}")
            out[f"{s}_pred{g}_n"] = n
            out[f"{s}_pred{g}_max_abs_diff_label1"] = d
            if not (_finite(n) and n > 0):
                reasons.append(f"{s}: no rows predicted {g}; the check needs both groups")
            elif not (_finite(d) and d < PROB_TOL):
                reasons.append(f"{s}: PROBABILITY differs from the label-1 column "
                               f"where the predicted label is {g} (max {d:.3g})")
    out["probability_output_is_presence_probability"] = not reasons
    return out


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

DRIVE_DUPLICATE = re.compile(r" \(\d+\)(\.[A-Za-z]+)?$")


def find_unique(roots: Iterable[Path], name: str, max_depth: int = 3):
    """Exactly one file called `name` under roots (depth <= max_depth).
    Returns (path | None, status, candidates); status is 'ok', 'missing' or
    'ambiguous'. Drive duplicates ('name (1).csv') count as candidates."""
    stem, suffix = Path(name).stem, Path(name).suffix
    cands: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for depth in range(max_depth + 1):
            pattern = "/".join(["*"] * depth + [f"{stem}*{suffix}"])
            for p in root.glob(pattern):
                base = DRIVE_DUPLICATE.sub(r"\1", p.name)
                if p.is_file() and base == name:
                    cands.append(p.resolve())
    cands = sorted(set(cands))
    if not cands:
        return None, "missing", []
    if len(cands) > 1:
        return None, "ambiguous", cands
    return cands[0], "ok", cands


def version_rank(name: str) -> int:
    """"""
    m = re.findall(r"v13b(\d+)", name)
    return max(int(x) for x in m) if m else 0
