#!/usr/bin/env python3
"""

* discrimination: ROC AUC and average precision;
* paired per-fold t-tests of the embedding model against the environmental
  baseline and the NDVI-only model (model_comparison_paired_tests.csv);
* proper scoring rules: log loss, Brier score and Brier skill score;
* calibration: calibration intercept, slope, ECE and reliability tables;
* fixed and training-selected threshold metrics;
* fold-specific and summarised train-test gaps;
* a predeclared regularisation sensitivity (leaf 5/20/50 and mtry 8/16/21);
* random-fold versus blocked-fold spatial transferability penalties;
* repeated spatial assignments and block-size sensitivity summaries;
* accessibility/effort/NDVI confound checks;
* marginal prediction summaries by observed land cover and NDVI, when the
  final-map interpretability sample from GEE_04 is present;
* RF variable-importance tables from GEE_04, when present;
* manuscript-ready methods notes and diagnostic plots.

"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, t, ttest_rel
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)

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
        "cd <package folder>; python Python_01_calculate_cv_metrics.py ; in a notebook: "
        "%run <package folder>\\Python_01_calculate_cv_metrics.py , or set BUNDLE_DIR_OVERRIDE near the top "
        "of this script to the package folder.")


BUNDLE_DIR = _bundle_dir()
sys.path.insert(0, str(BUNDLE_DIR / "support"))
# A console that already ran another copy of the package keeps its support
# modules in memory; drop them so that this package's support folder is used.
for _m in ("folders", "pipeline_checks"):
    sys.modules.pop(_m, None)
from pipeline_checks import (
    cv_scores_vs_manifest,
    find_unique,
    probability_mode_verdict,
    row_identity_check,
    version_rank,
)


import folders  # noqa: E402  (local folders: support/folders.py)

# GEE_03 tables (score files, fold audits, row manifests), GEE_04 tables
# (final-model checks) and the output folder; see support/folders.py.
DEFAULT_INPUT_DIR = folders.GEE03
DEFAULT_GEE04_DIR = folders.GEE04
DEFAULT_OUTPUT_DIR = folders.PYTHON01
GEE04_DIR = DEFAULT_GEE04_DIR      # set from --gee04-dir in main()
FIXED_THRESHOLD = 0.50
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
CALIBRATION_BINS = 10
EXPECTED_FOLDS = 5
LEAF_CANDIDATES = [5, 20, 50]
MTRY_CANDIDATES = [8, 16, 21]
FIXED_MIN_LEAF = 20
FIXED_MTRY = 16
# of predictors (embeddings 16/64, baseline_simple 3/11, ndvi_null 1/1). Must
# match GEE_03.
EXPECTED_MTRY_BY_MODEL = {
    "embeddings": FIXED_MTRY,
    "baseline_simple": 3,
    "ndvi_null": 1,
}
SPATIAL_PAIR_SAMPLE = 2_000_000
SPATIAL_MAX_DISTANCE_KM = 150.0
SPATIAL_BIN_WIDTH_KM = 10.0

# ---------------------------------------------------------------------------
# pipeline remains greppable; a *_label column beside each carries the wording
# used in the paper, and LABELS_GLOSSARY.csv maps one to the other.
# ---------------------------------------------------------------------------
SCHEME_LABELS = {
    "primary_effort": "Regular outdoor visitation model (2019-2021) [main]",
    "primary_effort_full":
        "Regular outdoor visitation model, full 2015-2021 window "
        "[sensitivity]",
    "access_matched":
        "Regular outdoor visitation model, accessibility-matched "
        "[sensitivity]",
    "recurrent_effort":
        "Recurrent outdoor visitation model (2019-2021) [comparison]",
    "recurrent_access_matched":
        "Recurrent outdoor visitation model, accessibility-matched "
        "[sensitivity]",
}
# residual range in an earlier run (regular 20 km -> 0.30 deg; recurrent 30 km
# -> 0.45 deg, since 0.30 deg spans only ~22 km east-west at 48.9 deg N). In the
# final run the recurrent range is 40 km (residual_spatial_range.csv): 0.45 deg
# exceeds it north-south but not east-west, so 0.60 deg is reported as a check.
SCHEME_MAIN_BLOCK_DEG = {
    "primary_effort": 0.30,
    "primary_effort_full": 0.30,
    "access_matched": 0.30,
    "recurrent_effort": 0.45,
    "recurrent_access_matched": 0.45,
}
# grows 0.30 -> 0.45 deg, so small shifts are expected; a deviation beyond
# STABILITY_TOLERANCE_AUC signals a code or data problem, not noise.
STABILITY_REFERENCE_AUC = {
    ("primary_effort", "embeddings"): 0.815,
    ("primary_effort", "baseline_simple"): 0.720,
    ("primary_effort", "ndvi_null"): 0.582,
    ("recurrent_effort", "embeddings"): 0.886,
    ("recurrent_effort", "baseline_simple"): 0.761,
    ("recurrent_effort", "ndvi_null"): 0.611,
    ("access_matched", "embeddings"): 0.693,
    ("recurrent_access_matched", "embeddings"): 0.769,
    ("primary_effort_full", "embeddings"): 0.776,
}
STABILITY_TOLERANCE_AUC = 0.02

# and retained co-located records, used by the coverage check and reports.
DEDUP_REMOVED_TEST_COUNTS: dict = {}
COLOCATED_TEST_COUNTS: dict = {}
COLOCATED_EXAMPLES = None
MODEL_LABELS = {
    "embeddings": "Satellite-embedding model (64 predictors)",
    "baseline_simple": "Environmental baseline (11 predictors)",
    "ndvi_null": "NDVI-only null (1 predictor)",
}
METRIC_GLOSSARY = [
    ("sampling_scheme", "analysis / outcome definition",
     "primary_effort = outdoor visitation (2019-2021); recurrent_effort = "
     "recurrent outdoor visitation (2019-2021); *_access_matched = the same "
     "outcome with the background matched on accessibility and sampling "
     "effort; primary_effort_full = full 2015-2021 presence window."),
    ("model", "predictor set",
     "embeddings = satellite-embedding model (64 predictors); "
     "baseline_simple = environmental baseline (11 predictors); "
     "ndvi_null = NDVI-only null."),
    ("prob", "predicted visitation-propensity score",
     "Out-of-fold Random Forest probability that a 500 m cell belongs to the "
     "visited class, under the case-control design (~50% prevalence)."),
    ("Mean_Prob_Presence", "mean predicted visitation-propensity score at "
     "visited cells", "Higher is better; compare with Mean_Prob_Absence."),
    ("Mean_Prob_Absence", "mean predicted visitation-propensity score at "
     "background cells", "Lower is better."),
    ("AUC", "area under the ROC curve",
     "Probability that a randomly chosen visited cell receives a higher "
     "propensity score than a randomly chosen background cell."),
    ("Brier_Skill_Score", "Brier skill score",
     "Improvement in squared-error accuracy of the propensity scores over "
     "always predicting the sample prevalence."),
    ("TSS_fixed", "true skill statistic at the fixed 0.50 threshold",
     "Sensitivity + specificity - 1 at 0.50; specific to the ~50% prevalence "
     "design and not transferable to a real visitation rate."),
    ("Calibration_Slope", "calibration slope",
     "Ideal 1; values above 1 mean the propensity scores are under-dispersed "
     "(shrunk toward the middle)."),
    ("ECE_10bin", "expected calibration error (10 bins)",
     "Mean absolute gap between predicted score and observed visited "
     "fraction across ten equal-frequency bins."),
    ("block_size", "spatial cross-validation block size (degrees)",
     "Side length of the checkerboard blocks whose whole contents share a "
     "fold; the headline uses 0.30 degrees."),
    ("in_ordlo", "inside the pre-2022 occupation line",
     "1 = cell lies within territory occupied before 24 Feb 2022."),
    ("dist_infra_m", "distance to the nearest road, railway or building (m)",
     "Accessibility covariate; not a model input."),
    ("log_effort", "log(1 + geotagged records within 5 km)",
     "Social-media sampling-effort covariate; not a model input."),
]


def with_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Insert manuscript-terminology columns beside the technical keys."""
    if frame is None or len(frame) == 0:
        return frame
    out = frame.copy()
    if "sampling_scheme" in out.columns and "scheme_label" not in out.columns:
        position = out.columns.get_loc("sampling_scheme") + 1
        out.insert(position, "scheme_label",
                   out["sampling_scheme"].map(SCHEME_LABELS)
                   .fillna(out["sampling_scheme"]))
    if "model" in out.columns and "model_label" not in out.columns:
        position = out.columns.get_loc("model") + 1
        out.insert(position, "model_label",
                   out["model"].map(MODEL_LABELS).fillna(out["model"]))
    return out


def write_labels_glossary(output_dir: Path) -> None:
    rows = [{"technical_term": key, "manuscript_term": term,
             "explanation": note} for key, term, note in METRIC_GLOSSARY]
    rows += [{"technical_term": key, "manuscript_term": label,
              "explanation": "analysis scheme"}
             for key, label in SCHEME_LABELS.items()]
    rows += [{"technical_term": key, "manuscript_term": label,
              "explanation": "predictor set"}
             for key, label in MODEL_LABELS.items()]
    pd.DataFrame(rows).to_csv(output_dir / "LABELS_GLOSSARY.csv", index=False)


FILE_PATTERNS = [
    "CV_MainBlocked_*.csv",
    "CV_RandomFold_*.csv",
    "CV_Regularisation_*.csv",
    "CV_Repeated_*.csv",
    "CV_Block_*.csv",
]


def _without_jupyter_kernel_args(arguments: list[str]) -> list[str]:
    """Remove only the kernel-file argument injected by Jupyter/IPython."""
    if "ipykernel" not in sys.modules:
        return arguments
    cleaned: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-f" and index + 1 < len(arguments):
            kernel_file = re.split(r"[\\/]", arguments[index + 1])[-1].lower()
            if kernel_file.startswith("kernel-") and kernel_file.endswith(".json"):
                index += 2
                continue
        if argument.startswith("-f="):
            kernel_file = re.split(r"[\\/]", argument[3:])[-1].lower()
            if kernel_file.startswith("kernel-") and kernel_file.endswith(".json"):
                index += 1
                continue
        cleaned.append(argument)
        index += 1
    return cleaned


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse spatial-CV raw probability exports.")
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Directory containing downloaded GEE CSV files (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument(
        "--gee04-dir", type=Path, default=DEFAULT_GEE04_DIR,
        help=f"Directory containing the GEE_04 tables (default: {DEFAULT_GEE04_DIR})")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for metrics and plots (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run deterministic metric unit checks without reading files")
    supplied = list(sys.argv[1:] if arguments is None else arguments)
    return parser.parse_args(_without_jupyter_kernel_args(supplied))


def auc_safe(y: Iterable[int], probability: Iterable[float]) -> float:
    y_array = np.asarray(y, dtype=int)
    p_array = np.asarray(probability, dtype=float)
    valid = np.isfinite(y_array) & np.isfinite(p_array)
    if valid.sum() < 2 or np.unique(y_array[valid]).size < 2:
        return np.nan
    return float(roc_auc_score(y_array[valid], p_array[valid]))


def average_precision_safe(
    y: Iterable[int], probability: Iterable[float]
) -> float:
    y_array = np.asarray(y, dtype=int)
    p_array = np.asarray(probability, dtype=float)
    valid = np.isfinite(y_array) & np.isfinite(p_array)
    if valid.sum() < 2 or np.unique(y_array[valid]).size < 2:
        return np.nan
    return float(average_precision_score(y_array[valid], p_array[valid]))


def threshold_metrics(
    y: Iterable[int], probability: Iterable[float], threshold: float
) -> dict[str, float]:
    y_array = np.asarray(y, dtype=int)
    p_array = np.asarray(probability, dtype=float)
    prediction = (p_array >= threshold).astype(int)
    tp = int(np.sum((y_array == 1) & (prediction == 1)))
    tn = int(np.sum((y_array == 0) & (prediction == 0)))
    fp = int(np.sum((y_array == 0) & (prediction == 1)))
    fn = int(np.sum((y_array == 1) & (prediction == 0)))
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tss": float(sensitivity + specificity - 1),
        "accuracy": float((tp + tn) / len(y_array)),
    }


def maximum_tss_threshold(
    y: Iterable[int], probability: Iterable[float]
) -> tuple[float, float]:
    candidates = [
        (float(threshold), threshold_metrics(y, probability, threshold)["tss"])
        for threshold in THRESHOLD_GRID
    ]
    # In a tie, choose the threshold closest to 0.5, then the lower threshold.
    candidates.sort(key=lambda item: (-item[1], abs(item[0] - 0.5), item[0]))
    return candidates[0]


def calibration_intercept_slope(
    y: Iterable[int], probability: Iterable[float]
) -> tuple[float, float]:
    """Logistic recalibration intercept and slope.

    Ideal values are 0 and 1. Probabilities are clipped before the logit to
    avoid infinities. The values are conditional on the approximately 50:50
    sampled prevalence and are not population-prevalence calibration claims.
    """
    y_array = np.asarray(y, dtype=float)
    p_array = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    valid = np.isfinite(y_array) & np.isfinite(p_array)
    y_array = y_array[valid]
    p_array = p_array[valid]
    if len(y_array) < 20 or np.unique(y_array).size < 2:
        return np.nan, np.nan
    logit_p = np.log(p_array / (1 - p_array))

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        linear = parameters[0] + parameters[1] * logit_p
        return float(np.sum(np.logaddexp(0.0, linear) - y_array * linear))

    result = minimize(
        negative_log_likelihood,
        x0=np.array([0.0, 1.0]),
        method="BFGS",
        options={"maxiter": 1000, "gtol": 1e-8},
    )
    if not np.all(np.isfinite(result.x)):
        return np.nan, np.nan
    return float(result.x[0]), float(result.x[1])


def equal_frequency_calibration(
    y: Iterable[int], probability: Iterable[float], bins: int = CALIBRATION_BINS
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "presence": np.asarray(y, dtype=int),
        "prob": np.asarray(probability, dtype=float),
    }).dropna()
    if frame.empty:
        return pd.DataFrame()
    number_of_bins = min(bins, max(1, frame["prob"].nunique()))
    frame["calibration_bin"] = pd.qcut(
        frame["prob"].rank(method="first"), number_of_bins,
        labels=False, duplicates="drop")
    rows = []
    for calibration_bin, group in frame.groupby("calibration_bin", sort=True):
        n = len(group)
        observed = float(group["presence"].mean())
        predicted = float(group["prob"].mean())
        # Wilson 95% interval for the observed class proportion.
        z = 1.959963984540054
        denominator = 1 + z * z / n
        centre = (observed + z * z / (2 * n)) / denominator
        half_width = z * math.sqrt(
            observed * (1 - observed) / n + z * z / (4 * n * n)
        ) / denominator
        rows.append({
            "calibration_bin": int(calibration_bin) + 1,
            "n": n,
            "mean_predicted": predicted,
            "observed_fraction": observed,
            "observed_ci_low": max(0.0, centre - half_width),
            "observed_ci_high": min(1.0, centre + half_width),
            "absolute_error": abs(observed - predicted),
        })
    return pd.DataFrame(rows)


def probabilistic_metrics(
    y: Iterable[int], probability: Iterable[float]
) -> dict[str, float]:
    y_array = np.asarray(y, dtype=int)
    p_array = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    prevalence = float(np.mean(y_array))
    brier = float(np.mean((p_array - y_array) ** 2))
    brier_reference = float(np.mean((prevalence - y_array) ** 2))
    brier_skill = 1 - brier / brier_reference if brier_reference > 0 else np.nan
    calibration = equal_frequency_calibration(y_array, p_array)
    ece = float(np.average(
        calibration["absolute_error"], weights=calibration["n"]
    )) if not calibration.empty else np.nan
    intercept, slope = calibration_intercept_slope(y_array, p_array)
    return {
        "n": int(len(y_array)),
        "n_presence": int(np.sum(y_array == 1)),
        "n_absence": int(np.sum(y_array == 0)),
        "prevalence": prevalence,
        "AUC": auc_safe(y_array, p_array),
        "Average_Precision": average_precision_safe(y_array, p_array),
        "Log_Loss": float(log_loss(y_array, p_array, labels=[0, 1])),
        "Brier_Score": brier,
        "Brier_Reference": brier_reference,
        "Brier_Skill_Score": brier_skill,
        "Calibration_Intercept": intercept,
        "Calibration_Slope": slope,
        "ECE_10bin": ece,
        "Mean_Prob_Presence": float(np.mean(p_array[y_array == 1])),
        "Mean_Prob_Absence": float(np.mean(p_array[y_array == 0])),
    }


def infer_sampling_scheme(text: str) -> str:
    lower = text.lower()
    # primary_effort patterns, which its run tag also contains.
    if "primary_effort_full" in lower or "full2015_2021" in lower:
        return "primary_effort_full"
    is_recurrent = "recurrent" in lower
    is_access_matched = "accessmatch" in lower or "access_matched" in lower
    if is_recurrent and is_access_matched:
        return "recurrent_access_matched"
    if is_recurrent and ("effort_guard0" in lower or "recurrent_effort" in lower):
        return "recurrent_effort"
    if is_access_matched:
        return "access_matched"
    if "effort_guard0" in lower or "primary_effort" in lower:
        return "primary_effort"
    return "unspecified"


def infer_analysis_type(filename: str) -> str:
    lower = filename.lower()
    if "regularisation" in lower or "regularization" in lower:
        return "regularisation_sensitivity"
    if "randomfold" in lower:
        return "random_cv"
    if "repeated" in lower or "repeat" in lower:
        return "repeated_cv"
    if "cv_block" in lower or "_block" in lower:
        return "block_sensitivity"
    if "mainblocked" in lower or "mainfixed" in lower or "rawscores" in lower:
        return "main_fixed"
    return "unknown"


def extract_number(pattern: str, text: str, default: float = np.nan) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1).replace("p", ".")) if match else default


def canonicalise_raw(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    frame = frame.copy()
    filename = source.name
    if "prob" not in frame.columns and "classification" in frame.columns:
        frame = frame.rename(columns={"classification": "prob"})
    required = ["presence", "prob"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{filename} lacks required columns: {missing}")

    defaults: dict[str, object] = {
        "analysis_type": infer_analysis_type(filename),
        "model": "embeddings",
        "sampling_scheme": infer_sampling_scheme(filename),
        "run_tag": filename,
        "repeat_id": "det",
        "block_size": extract_number(r"block(\d+p?\d*)", filename, 0.30),
        "min_leaf": extract_number(r"leaf(\d+)", filename, FIXED_MIN_LEAF),
        "variables_per_split": extract_number(
            r"mtry(\d+)", filename, FIXED_MTRY),
        "outer_fold": extract_number(r"outer(\d+)", filename),
        "inner_fold": np.nan,
        "fold_id": np.nan,
        "set": "test",
        "block_id": np.nan,
        "lon": np.nan,
        "lat": np.nan,
    }
    if "sample_variant" in frame.columns and "sampling_scheme" not in frame.columns:
        # sample_variant is NDVI filtering, not the PA scheme. Infer the latter
        # from run_tag or filename instead.
        pass
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

    combined_text = filename + " " + " ".join(
        frame["run_tag"].dropna().astype(str).unique().tolist()[:5])
    if (frame["sampling_scheme"].astype(str) == "unspecified").all():
        frame["sampling_scheme"] = infer_sampling_scheme(combined_text)
    if (frame["analysis_type"].astype(str) == "unknown").all():
        frame["analysis_type"] = infer_analysis_type(filename)

    numeric_columns = [
        "presence", "prob", "block_size", "min_leaf",
        "variables_per_split", "outer_fold",
        "inner_fold", "fold_id", "block_id", "lon", "lat", "ndvi", "esa",
        "dist_infra_m", "log_effort", "in_ordlo", "emb_fingerprint",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[np.isfinite(frame["presence"]) & np.isfinite(frame["prob"])].copy()
    frame["presence"] = frame["presence"].astype(int)
    frame["prob"] = frame["prob"].clip(0, 1)
    frame["source_file"] = filename
    return frame


def discover_files(input_dir: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in FILE_PATTERNS:
        files.update(input_dir.glob(pattern))
    return sorted(files)


def load_raw_scores(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = discover_files(input_dir)
    if not files:
        raise FileNotFoundError(
            f"No raw CV exports found in {input_dir}. Patterns: {FILE_PATTERNS}")
    frames = []
    inventory = []
    for source in files:
        try:
            frame = canonicalise_raw(pd.read_csv(source), source)
        except Exception as error:
            inventory.append({
                "source_file": source.name, "status": "rejected",
                "rows": 0, "message": str(error),
            })
            continue
        frames.append(frame)
        inventory.append({
            "source_file": source.name, "status": "loaded",
            "rows": len(frame), "message": "",
        })
    if not frames:
        raise RuntimeError("All discovered files were rejected; inspect run_inventory.csv")
    combined = pd.concat(frames, ignore_index=True, sort=False)

    # an older export that happens to share a broad filename pattern.
    is_v13b = combined["source_file"].astype(str).str.contains(
        "v13b", case=False, regex=False)
    if not is_v13b.any():
        raise RuntimeError(
            "No v13b raw-score exports were found. Download the files created "
            "by GEE_03; older runs are intentionally not mixed into v13b.")
    legacy_count = int((~is_v13b).sum())
    combined = combined.loc[is_v13b].copy()
    if legacy_count:
        inventory.append({
            "source_file": "VERSION_GUARD",
            "status": "excluded_non_v13b",
            "rows": legacy_count,
            "message": "Older score rows were excluded from the v13b analysis.",
        })

    # Never fill missing outdoor-only folds with older broad-domain folds.
    # Once at least one explicitly tagged outdoor export exists for a
    # non-recurrent scheme, every legacy row for that scheme is excluded. An
    # incomplete new run must remain visibly incomplete rather than becoming a
    # scientifically invalid hybrid dataset with an apparently complete CV.
    run_tag_text = combined["run_tag"].fillna("").astype(str).str.lower()
    rows_removed_for_domain = 0
    for scheme in ["primary_effort", "access_matched"]:
        scheme_rows = combined["sampling_scheme"].eq(scheme)
        has_outdoor = bool((scheme_rows & run_tag_text.str.contains("outdoor")).any())
        if has_outdoor:
            remove = scheme_rows & ~run_tag_text.str.contains("outdoor")
            rows_removed_for_domain += int(remove.sum())
            combined = combined.loc[~remove].copy()
            run_tag_text = combined["run_tag"].fillna("").astype(str).str.lower()
    if rows_removed_for_domain:
        inventory.append({
            "source_file": "DOMAIN_GUARD",
            "status": "excluded_legacy_broad_domain",
            "rows": rows_removed_for_domain,
            "message": (
                "Outdoor-tagged non-recurrent exports were present; legacy "
                "broad-domain rows were not mixed into their missing folds."),
        })

    # Among duplicate browser/Drive downloads, keep the most recent file for
    # each complete logical experiment.
    # files of one experiment the preferred naming (CV_* prefix) and the
    # newest code version in the name are kept. If two or more files remain
    # (for example a second download 'name (1).csv' or a repeated export),
    # their contents must be identical; conflicting files stop the run. The
    # download time never decides between files.
    source_metadata = {
        source.name: {
            "priority": 2 if source.name.startswith((
                "CV_MainBlocked_", "CV_RandomFold_", "CV_Regularisation_",
                "CV_Repeated_", "CV_Block_")) else 1,
            "version": version_rank(source.name),
        }
        for source in files
    }
    combined["_source_priority"] = combined["source_file"].map(
        lambda name: source_metadata[name]["priority"])
    combined["_source_version"] = combined["source_file"].map(
        lambda name: source_metadata[name]["version"])
    dataset_keys = [
        "analysis_type", "model", "sampling_scheme", "repeat_id",
        "block_size", "min_leaf", "variables_per_split", "outer_fold",
        "inner_fold", "fold_id",
    ]
    signature_values = combined[dataset_keys].copy()
    signature_values = signature_values.fillna("__NA__").astype(str)
    combined["_dataset_signature"] = signature_values.agg("|".join, axis=1)
    meta = combined[["_dataset_signature", "source_file", "_source_priority",
                     "_source_version"]].drop_duplicates()
    best = meta.groupby("_dataset_signature")[["_source_priority", "_source_version"]] \
        .transform("max")
    meta = meta[(meta["_source_priority"] == best["_source_priority"])
                & (meta["_source_version"] == best["_source_version"])]

    def _content(signature: str, source: str) -> pd.DataFrame:
        part = combined[(combined["_dataset_signature"] == signature)
                        & (combined["source_file"] == source)]
        drop = [c for c in part.columns if c.startswith("_") or c == "source_file"]
        if "run_tag" in part.columns and (part["run_tag"] == source).all():
            drop.append("run_tag")          # default filled from the file name
        part = part.drop(columns=drop)
        part = part[sorted(part.columns)]
        return part.sort_values(list(part.columns), na_position="last",
                                kind="mergesort").reset_index(drop=True)

    chosen_source: dict[str, str] = {}
    conflicts: list[str] = []
    for signature, group in meta.groupby("_dataset_signature", sort=True):
        names = sorted(group["source_file"])
        chosen_source[signature] = names[0]
        if len(names) == 1:
            continue
        reference = _content(signature, names[0])
        differing = [n for n in names[1:] if not reference.equals(_content(signature, n))]
        if differing:
            conflicts.append(f"{names[0]} vs {', '.join(differing)}")
        else:
            inventory.append({
                "source_file": "; ".join(names), "status": "identical_copies",
                "rows": len(reference),
                "message": f"Identical copies of one experiment; {names[0]} used."})
    if conflicts:
        raise RuntimeError(
            "Conflicting CV score files for the same experiment and code version "
            "(different contents). Keep one download of each GEE_03 export and "
            "rerun:\n  " + "\n  ".join(conflicts[:20]))
    keep_mask = combined["source_file"].eq(
        combined["_dataset_signature"].map(pd.Series(chosen_source)))
    combined = combined.loc[keep_mask].copy()
    # file of a newer code version, rows of older versions are dropped (a
    # missing new fold then stops the run in required_main_cv_status).
    newest = combined.groupby("sampling_scheme")["_source_version"].transform("max")
    older = combined["_source_version"] < newest
    if older.any():
        inventory.append({
            "source_file": "VERSION_GUARD_v13b12",
            "status": "excluded_older_code_version",
            "rows": int(older.sum()),
            "message": ("Rows of older code versions were dropped for schemes "
                        "that have newer score files: "
                        + ", ".join(sorted(combined.loc[older, "sampling_scheme"]
                                           .astype(str).unique())) + "."),
        })
        combined = combined.loc[~older].copy()
    combined = combined.drop(columns=[
        "_source_priority", "_source_version", "_dataset_signature"])

    # the server-side audit reconcile exactly (for every model and fold
    # scheme, unique rows + within-file "duplicates" == audited collection
    # size), and the duplicate profile identifies what these rows are:
    # DISTINCT co-located records. H3 resolution-8 cells (~460 m across) are
    # smaller than the 500 m analysis pixel, so ~2-5% of records share a
    # pixel with another record; the export snaps coordinates to the pixel
    # centre, and the 64 embedding predictors are then identical, so the two
    # records also receive identical embedding-model scores (baseline/null
    # scores differ because their NDVI input varies within the pixel). Under
    # any blocked partition co-located records share a fold, so the old
    # identity-based filter always collapsed them (and only one time in five
    # under random folds) -- they are legitimate records, not duplicates.
    # The filter therefore now removes only rows identical in EVERY exported
    # column within one file (true byte-duplicates, which a single Earth
    # Engine export should not contain at all), and reports what it did.
    duplicate_mask = combined.duplicated(subset=list(combined.columns),
                                         keep="first")
    before = len(combined)
    removed = combined.loc[duplicate_mask].copy()
    combined = combined.loc[~duplicate_mask].reset_index(drop=True)
    global DEDUP_REMOVED_TEST_COUNTS
    if len(removed):
        removed_test = removed[removed["set"] == "test"]
        DEDUP_REMOVED_TEST_COUNTS = (removed_test.groupby(
            ["sampling_scheme", "analysis_type", "model"], dropna=False)
            .size().to_dict())
    else:
        DEDUP_REMOVED_TEST_COUNTS = {}
    # Document the co-located groups (identity-tied rows that are now
    # RETAINED): up to 300 example rows so the mechanism is auditable, and a
    # per-partition count for the SI data-flow table.
    legacy_identity = [
        "analysis_type", "model", "sampling_scheme", "repeat_id",
        "block_size", "min_leaf", "variables_per_split", "outer_fold",
        "inner_fold", "fold_id",
        "set", "presence", "prob", "block_id", "lon", "lat", "source_file",
    ]
    legacy_identity = [c for c in legacy_identity if c in combined.columns]
    tie_mask = combined.duplicated(subset=legacy_identity, keep=False)
    global COLOCATED_EXAMPLES, COLOCATED_TEST_COUNTS
    ties = combined.loc[tie_mask]
    COLOCATED_EXAMPLES = (ties.sort_values(legacy_identity).head(300).copy()
                          if len(ties) else pd.DataFrame())
    extra = combined.duplicated(subset=legacy_identity, keep="first")
    extra_test = combined.loc[extra & combined["set"].eq("test")]
    COLOCATED_TEST_COUNTS = (extra_test.groupby(
        ["sampling_scheme", "analysis_type", "model"], dropna=False)
        .size().to_dict()) if len(extra_test) else {}
    inventory.append({
        "source_file": "ALL_FILES", "status": "deduplicated",
        "rows": len(combined),
        "message": (
            f"Removed {before - len(combined)} byte-identical duplicate rows "
            f"within files; retained {int(extra.sum())} distinct co-located "
            "records (same 500 m pixel, identical exported coordinates and "
            "embedding scores) that earlier versions wrongly collapsed."),
    })
    return combined, pd.DataFrame(inventory)


PARTITION_KEYS = [
    "analysis_type", "model", "sampling_scheme", "repeat_id", "block_size",
    "min_leaf", "variables_per_split", "outer_fold", "inner_fold", "fold_id",
]


def partition_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key_values, group in raw.groupby(PARTITION_KEYS, dropna=False, sort=True):
        metadata = dict(zip(PARTITION_KEYS, key_values))
        training = group[group["set"] == "train"]
        threshold_train = np.nan
        if not training.empty and training["presence"].nunique() == 2:
            threshold_train, _ = maximum_tss_threshold(
                training["presence"], training["prob"])

        for set_name, evaluated in group.groupby("set", sort=True):
            if evaluated["presence"].nunique() < 2:
                continue
            metrics = probabilistic_metrics(
                evaluated["presence"], evaluated["prob"])
            fixed = threshold_metrics(
                evaluated["presence"], evaluated["prob"], FIXED_THRESHOLD)
            metrics.update({
                "threshold_fixed": FIXED_THRESHOLD,
                "sensitivity_fixed": fixed["sensitivity"],
                "specificity_fixed": fixed["specificity"],
                "TSS_fixed": fixed["tss"],
                "accuracy_fixed": fixed["accuracy"],
                "threshold_train_selected": threshold_train,
            })
            if np.isfinite(threshold_train):
                selected = threshold_metrics(
                    evaluated["presence"], evaluated["prob"], threshold_train)
                metrics.update({
                    "sensitivity_train_selected": selected["sensitivity"],
                    "specificity_train_selected": selected["specificity"],
                    "TSS_train_selected": selected["tss"],
                })
            else:
                metrics.update({
                    "sensitivity_train_selected": np.nan,
                    "specificity_train_selected": np.nan,
                    "TSS_train_selected": np.nan,
                })
            rows.append({**metadata, "set": set_name, **metrics})
    return pd.DataFrame(rows)


def mean_sd_ci(group: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(group, errors="coerce").dropna().to_numpy(float)
    n = len(values)
    mean = float(np.mean(values)) if n else np.nan
    sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
    if n > 1:
        half = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n))
        low, high = mean - half, mean + half
    else:
        low = high = np.nan
    return {"mean": mean, "sd": sd, "ci95_low": low, "ci95_high": high, "n": n}


def summarise_partitions(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics_to_summarise = [
        "AUC", "Average_Precision", "Log_Loss", "Brier_Score",
        "Brier_Skill_Score", "Calibration_Intercept", "Calibration_Slope",
        "ECE_10bin", "TSS_fixed", "sensitivity_fixed", "specificity_fixed",
        "TSS_train_selected",
    ]
    group_keys = [
        "analysis_type", "model", "sampling_scheme", "repeat_id",
        "block_size", "min_leaf", "variables_per_split", "set",
    ]
    rows = []
    for key_values, group in metrics.groupby(group_keys, dropna=False, sort=True):
        metadata = dict(zip(group_keys, key_values))
        for metric in metrics_to_summarise:
            summary = mean_sd_ci(group[metric])
            rows.append({**metadata, "metric": metric, **summary})
    return pd.DataFrame(rows)


def paired_model_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    """Paired two-sided t-tests on the five blocked test-fold AUCs.

    For each visitation model with a model comparison (regular and
    recurrent), the embedding model is compared with the environmental
    baseline and with the NDVI-only model, fold by fold. These are the
    p-values quoted in the Results.
    """
    test = metrics[(metrics["analysis_type"] == "main_fixed")
                   & (metrics["set"] == "test")]
    rows = []
    for scheme, group in test.groupby("sampling_scheme", sort=True):
        auc = group.pivot_table(index="fold_id", columns="model",
                                values="AUC", aggfunc="first")
        if "embeddings" not in auc.columns:
            continue
        for other in ["baseline_simple", "ndvi_null"]:
            if other not in auc.columns:
                continue
            pair = auc[["embeddings", other]].dropna()
            if len(pair) < 2:
                continue
            result = ttest_rel(pair["embeddings"], pair[other])
            rows.append({
                "sampling_scheme": scheme,
                "comparison": f"embeddings_vs_{other}",
                "n_folds": int(len(pair)),
                "mean_auc_embeddings": float(pair["embeddings"].mean()),
                "mean_auc_other": float(pair[other].mean()),
                "mean_auc_difference": float(
                    (pair["embeddings"] - pair[other]).mean()),
                "t_statistic": float(result.statistic),
                "p_value_two_sided": float(result.pvalue),
                # the t-test treats dependent pairs as independent; report the
                # fold-by-fold pattern beside it.
                "n_folds_embeddings_higher": int(
                    (pair["embeddings"] > pair[other]).sum()),
                "min_fold_difference": float(
                    (pair["embeddings"] - pair[other]).min()),
                "max_fold_difference": float(
                    (pair["embeddings"] - pair[other]).max()),
                "note": "folds share training data; p-value indicative only",
            })
    return pd.DataFrame(rows)


def pooled_test_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    test = raw[raw["set"].isin(["test", "validation"])].copy()
    group_keys = [
        "analysis_type", "model", "sampling_scheme", "repeat_id",
        "block_size", "min_leaf", "variables_per_split",
    ]
    rows = []
    for key_values, group in test.groupby(group_keys, dropna=False, sort=True):
        if group["presence"].nunique() < 2:
            continue
        rows.append({
            **dict(zip(group_keys, key_values)),
            **probabilistic_metrics(group["presence"], group["prob"]),
        })
    return pd.DataFrame(rows)


def train_test_gaps(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = [column for column in PARTITION_KEYS if column != "inner_fold"]
    selected = metrics[metrics["set"].isin(["train", "test"])].copy()
    # Sentinels preserve main-CV rows whose outer_fold is intentionally
    # undefined. An explicit train/test merge is used instead of pivot_table:
    # pandas pivot tables with dropna=False can manufacture a Cartesian product
    # of unobserved model/scheme combinations.
    numeric_sentinels = {"outer_fold": -999, "fold_id": -999}
    for column, sentinel in numeric_sentinels.items():
        if column in selected.columns:
            selected[column] = selected[column].fillna(sentinel)
    for column in ["analysis_type", "model", "sampling_scheme", "repeat_id"]:
        if column in selected.columns:
            selected[column] = selected[column].fillna("unspecified")
    metric_columns = [
        "AUC", "Brier_Score", "Brier_Skill_Score", "Log_Loss",
        "Calibration_Slope", "ECE_10bin",
    ]
    train = selected[selected["set"].eq("train")][
        index + metric_columns].drop_duplicates(index)
    test = selected[selected["set"].eq("test")][
        index + metric_columns].drop_duplicates(index)
    if train.empty or test.empty:
        return pd.DataFrame(), pd.DataFrame()
    train = train.rename(columns={
        metric: f"{metric}_train" for metric in metric_columns})
    test = test.rename(columns={
        metric: f"{metric}_test" for metric in metric_columns})
    wide = train.merge(test, on=index, how="inner", validate="one_to_one")
    if "outer_fold" in wide.columns:
        wide["outer_fold"] = wide["outer_fold"].replace(-999, np.nan)
    if "fold_id" in wide.columns:
        wide["fold_id"] = wide["fold_id"].replace(-999, np.nan)
    required_pairs = {
        "AUC_train_minus_test": ("AUC_train", "AUC_test"),
        "Brier_test_minus_train": ("Brier_Score_test", "Brier_Score_train"),
        "BSS_train_minus_test": ("Brier_Skill_Score_train", "Brier_Skill_Score_test"),
        "LogLoss_test_minus_train": ("Log_Loss_test", "Log_Loss_train"),
        "CalibrationSlope_train_minus_test": (
            "Calibration_Slope_train", "Calibration_Slope_test"),
        "ECE_test_minus_train": ("ECE_10bin_test", "ECE_10bin_train"),
    }
    for output, (left, right) in required_pairs.items():
        if left in wide.columns and right in wide.columns:
            wide[output] = wide[left] - wide[right]

    gap_columns = list(required_pairs)
    summary_keys = [
        "analysis_type", "model", "sampling_scheme", "block_size",
        "min_leaf", "variables_per_split"
    ]
    rows = []
    for key_values, group in wide.groupby(summary_keys, dropna=False, sort=True):
        metadata = dict(zip(summary_keys, key_values))
        row: dict[str, object] = metadata
        for gap in gap_columns:
            if gap in group.columns:
                stats = mean_sd_ci(group[gap])
                row[f"{gap}_mean"] = stats["mean"]
                row[f"{gap}_sd"] = stats["sd"]
        row["n_partitions"] = len(group)
        rows.append(row)
    return wide, pd.DataFrame(rows)


def confound_checks(raw: pd.DataFrame) -> pd.DataFrame:
    subset = raw[
        (raw["analysis_type"] == "main_fixed")
        & (raw["model"] == "embeddings")
        & (raw["set"] == "test")
    ].copy()
    rows = []
    for scheme, group in subset.groupby("sampling_scheme", sort=True):
        y = group["presence"].to_numpy(int)
        rows.append({
            "sampling_scheme": scheme,
            "variable": "model_probability",
            "auc_alone": auc_safe(y, group["prob"]),
            "spearman_with_model_probability": 1.0,
            "n": len(group),
        })
        definitions = [
            ("ndvi", +1, "summer_median_NDVI"),
            ("dist_infra_m", -1, "distance_to_infrastructure"),
            ("log_effort", +1, "log_sampling_effort"),
        ]
        for column, auc_sign, label in definitions:
            if column not in group.columns:
                continue
            valid = np.isfinite(group[column]) & np.isfinite(group["prob"])
            if valid.sum() < 100:
                continue
            values = group.loc[valid, column].to_numpy(float)
            correlation = spearmanr(
                values, group.loc[valid, "prob"].to_numpy(float),
                nan_policy="omit").statistic
            rows.append({
                "sampling_scheme": scheme,
                "variable": label,
                "auc_alone": auc_safe(
                    group.loc[valid, "presence"].to_numpy(int), auc_sign * values),
                "spearman_with_model_probability": float(correlation),
                "n": int(valid.sum()),
            })
    return pd.DataFrame(rows)


STRATIFIED_MIN_PER_CLASS = 30


def stratified_performance(raw: pd.DataFrame) -> pd.DataFrame:
    """

    Answers R3-2 at the model level (does performance hold inside the
    pre-2022 occupation line, where the quasi-peaceful-baseline assumption is
    weakest?) and the R2 platform-bias question (each platform's presences
    against the common pseudo-absence background). WorldCover strata expose
    class-specific behaviour inside the training domain. Pure post-processing
    of the pooled out-of-fold scores; no Earth Engine re-run involved.
    """
    subset = raw[
        (raw["analysis_type"] == "main_fixed")
        & (raw["model"] == "embeddings")
        & (raw["set"] == "test")
    ].copy()
    rows: list[dict[str, object]] = []

    def add_row(scheme: str, stratum_type: str, stratum_value: object,
                frame: pd.DataFrame) -> None:
        y = frame["presence"].to_numpy(int)
        n_presence = int((y == 1).sum())
        n_absence = int((y == 0).sum())
        if n_presence < STRATIFIED_MIN_PER_CLASS \
                or n_absence < STRATIFIED_MIN_PER_CLASS:
            return
        metrics = probabilistic_metrics(frame["presence"], frame["prob"])
        rows.append({
            "sampling_scheme": scheme,
            "stratum_type": stratum_type,
            "stratum": str(stratum_value),
            "n": metrics["n"],
            "n_presence": n_presence,
            "n_absence": n_absence,
            "prevalence": metrics["prevalence"],
            "AUC": metrics["AUC"],
            "Brier_Skill_Score": metrics["Brier_Skill_Score"],
            "Calibration_Slope": metrics["Calibration_Slope"],
            "ECE_10bin": metrics["ECE_10bin"],
        })

    for scheme, group in subset.groupby("sampling_scheme", sort=True):
        if "in_ordlo" in group.columns:
            in_ordlo = pd.to_numeric(
                group["in_ordlo"], errors="coerce").fillna(0).astype(int)
            for value in sorted(in_ordlo.unique()):
                add_row(scheme, "in_ordlo", value, group[in_ordlo == value])
        if "esa" in group.columns:
            esa = pd.to_numeric(group["esa"], errors="coerce")
            for value in sorted(esa.dropna().unique()):
                add_row(scheme, "worldcover_class", int(value),
                        group[esa == value])
        # Platform is a presence-side attribute; pseudo-absences share one
        # label. Each platform's presences are scored against the FULL
        # background, so the strata differ only in the positive class.
        if "platform" in group.columns:
            absences = group[group["presence"] == 0]
            presences_only = group[group["presence"] == 1]
            for platform, platform_presences in presences_only.groupby(
                    "platform", sort=True):
                add_row(scheme, "platform_presences_vs_all_absences",
                        platform,
                        pd.concat([platform_presences, absences],
                                  ignore_index=True))
    return pd.DataFrame(rows)


def complete_case_dropout(raw: pd.DataFrame) -> pd.DataFrame:
    """

    Before the identical-rows fix, each compared model silently used its own
    complete-case subset; this table documents the pooled out-of-fold counts
    per model and the presence/absence rows dropped relative to the
    largest-row model in the scheme. After the fix and re-run, the within-
    scheme differences collapse to zero and this table demonstrates that;
    the uploaded-versus-complete accounting (WHERE the masked-embedding
    dropout falls) comes from GEE_06's CompleteCase_Audit export.
    """
    subset = raw[(raw["analysis_type"] == "main_fixed")
                 & (raw["set"] == "test")]
    counts = (subset.groupby(["sampling_scheme", "model"])
              .agg(n=("presence", "size"),
                   n_presence=("presence", lambda v: int((v == 1).sum())),
                   n_absence=("presence", lambda v: int((v == 0).sum())))
              .reset_index())
    rows: list[dict[str, object]] = []
    for scheme, group in counts.groupby("sampling_scheme", sort=True):
        reference = group.loc[group["n"].idxmax()]
        for _, row in group.iterrows():
            presence_reference = max(int(reference["n_presence"]), 1)
            absence_reference = max(int(reference["n_absence"]), 1)
            rows.append({
                "sampling_scheme": scheme,
                "model": row["model"],
                "n": int(row["n"]),
                "n_presence": int(row["n_presence"]),
                "n_absence": int(row["n_absence"]),
                "reference_model": reference["model"],
                "presence_rows_dropped_vs_reference":
                    int(reference["n_presence"]) - int(row["n_presence"]),
                "absence_rows_dropped_vs_reference":
                    int(reference["n_absence"]) - int(row["n_absence"]),
                "presence_dropout_fraction": round(
                    (int(reference["n_presence"]) - int(row["n_presence"]))
                    / presence_reference, 4),
                "absence_dropout_fraction": round(
                    (int(reference["n_absence"]) - int(row["n_absence"]))
                    / absence_reference, 4),
            })
    return pd.DataFrame(rows)


def matching_gate_check(input_dir: Path, raw: pd.DataFrame) -> pd.DataFrame:
    """

    The A2/A3 gates are evaluated on eligible input presences BEFORE raster
    sampling, while the CV tables count complete-case rows AFTER it, so the
    two retention figures legitimately differ. This table puts both side by
    side so the manuscript can report the gate actually passed (with its max
    |SMD|) and the complete-case retention separately.
    """
    rows: list[dict[str, object]] = []
    # written by Python_A2/A3 into data/
    meta = {p.resolve() for d in (input_dir, folders.DOWNLOADS) if d.is_dir()
            for p in d.glob("*accessibility_matching_metadata.json")}
    for path in sorted(meta):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            rows.append({"source": path.name,
                         "status": f"unreadable: {error}"})
            continue
        rows.append({
            "source": path.name,
            "status": "gate_metadata",
            "n_presence_input": payload.get("n_presence_input"),
            "n_presence_matched": payload.get("n_presence_matched"),
            "presence_retention_fraction":
                payload.get("presence_retention_fraction"),
            "maximum_absolute_smd": payload.get("maximum_absolute_smd"),
            "smd_acceptance_limit": payload.get("smd_acceptance_limit"),
            "minimum_presence_retention":
                payload.get("minimum_presence_retention"),
            "selected_bins_per_covariate":
                payload.get("selected_bins_per_covariate"),
        })
    test = raw[
        (raw["analysis_type"] == "main_fixed")
        & (raw["model"] == "embeddings")
        & (raw["set"] == "test")
    ]
    # stage, i.e. on the rows actually evaluated in cross-validation.
    for matched in ("access_matched", "recurrent_access_matched"):
        rows_m = test[test["sampling_scheme"] == matched]
        if rows_m.empty:
            continue
        pres, absn = rows_m[rows_m["presence"] == 1], rows_m[rows_m["presence"] == 0]
        smd = {}
        for name, fn in (("log_effort", lambda f: pd.to_numeric(f["log_effort"], errors="coerce")),
                         ("log_distance", lambda f: np.log1p(pd.to_numeric(f["dist_infra_m"], errors="coerce")))):
            a, b = fn(pres).to_numpy(float), fn(absn).to_numpy(float)
            pooled = np.sqrt((np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)) / 2.0)
            smd[name] = float((np.nanmean(a) - np.nanmean(b)) / pooled) if pooled > 0 else np.nan
        pair_sides = rows_m.groupby("match_pair_id")["presence"].agg(lambda v: set(v))
        n_complete = int(sum(1 for v in pair_sides if v == {0, 1}))
        rows.append({
            "source": f"cv_modelling_stage:{matched}",
            "status": "balance_and_pairs_at_modelling_stage",
            "n_presence_input": int(len(pres)),
            "n_presence_matched": int(len(absn)),
            "presence_retention_fraction": None,
            "maximum_absolute_smd": float(np.nanmax(np.abs(list(smd.values())))),
            "smd_acceptance_limit": 0.10,
            "minimum_presence_retention": None,
            "selected_bins_per_covariate": None,
            "smd_log_effort": smd["log_effort"],
            "smd_log_distance": smd["log_distance"],
            "n_complete_pairs": n_complete,
            "n_incomplete_pairs": int(len(pair_sides) - n_complete),
        })
    presence_counts = test.groupby("sampling_scheme")["presence"].agg(
        lambda v: int((v == 1).sum()))
    for matched, base in [
        ("access_matched", "primary_effort"),
        ("recurrent_access_matched", "recurrent_effort"),
    ]:
        if matched in presence_counts.index \
                and base in presence_counts.index \
                and presence_counts[base] > 0:
            rows.append({
                "source": f"cv_complete_case:{matched}_vs_{base}",
                "status": "complete_case_rows",
                "n_presence_input": int(presence_counts[base]),
                "n_presence_matched": int(presence_counts[matched]),
                "presence_retention_fraction": round(
                    float(presence_counts[matched])
                    / float(presence_counts[base]), 4),
                "maximum_absolute_smd": None,
                "smd_acceptance_limit": None,
                "minimum_presence_retention": None,
                "selected_bins_per_covariate": None,
            })
    return pd.DataFrame(rows)


def test_train_distance_check(raw: pd.DataFrame) -> pd.DataFrame:
    """"""
    rows = []
    test = raw[(raw["analysis_type"] == "main_fixed") & (raw["model"] == "embeddings")
               & (raw["set"] == "test")]
    for scheme, g in test.groupby("sampling_scheme", sort=True):
        lon = pd.to_numeric(g["lon"], errors="coerce").to_numpy(float)
        lat = pd.to_numeric(g["lat"], errors="coerce").to_numpy(float)
        ok = np.isfinite(lon) & np.isfinite(lat)
        lat0 = float(np.nanmean(lat))
        xy = np.column_stack([lon * 111.320 * math.cos(math.radians(lat0)), lat * 110.574])
        fold = g["fold_id"].to_numpy()
        dist = np.full(len(g), np.nan)
        for f in np.unique(fold[ok]):
            te, tr = ok & (fold == f), ok & (fold != f)
            if tr.sum() and te.sum():
                dist[te] = cKDTree(xy[tr]).query(xy[te])[0]
        d = dist[np.isfinite(dist)]
        if len(d) == 0:
            continue
        rows.append({"sampling_scheme": scheme, "n_test_records": int(len(d)),
                     "p05_km": float(np.percentile(d, 5)), "median_km": float(np.median(d)),
                     "p95_km": float(np.percentile(d, 95)),
                     "share_within_1km": float(np.mean(d <= 1)),
                     "share_within_5km": float(np.mean(d <= 5))})
    return pd.DataFrame(rows)


ROW_CHECK_TAG = "v13b12"
# read by Python_03. It is removed when a run starts and rewritten at the
# end; CHECKS_FAILED.txt exists while a run is unfinished or has failed.
VERDICT_NAME = f"CV_CHECKS_VERDICT_{ROW_CHECK_TAG}.json"
MARKER_NAME = "CHECKS_FAILED.txt"


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_verdict(output_dir: Path, input_dir: Path, final_rows: pd.DataFrame,
                  cv_check: pd.DataFrame, failed_checks: list[str]) -> dict:
    """Overall verdict covering both final-model checks and every CV
    partition, with the run identifiers checked and the SHA-256 of the two
    check tables (Python_03 recomputes them, so a table replaced later is
    detected)."""
    from datetime import datetime, timezone
    finals = {}
    for r in final_rows.itertuples():
        finals[r.sampling_scheme] = {
            "passed": bool(r.passed),
            "gee04_run_id": str(getattr(r, "gee04_run_id", "")),
            "gee03_run_id": str(getattr(r, "gee03_run_id", "")),
            "presence_score_mode": str(getattr(r, "presence_score_mode", "")),
        }
    runs = {}
    if "run_id" in cv_check.columns:
        for scheme, g in cv_check.groupby("sampling_scheme"):
            runs[scheme] = sorted(g["run_id"].dropna().astype(str).unique())
    verdict = {
        "code_version": ROW_CHECK_TAG,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": not failed_checks,
        "input_dir": str(input_dir),
        "final_models": finals,
        "cv_partitions": {
            "n": int(len(cv_check)),
            "n_passed": int(cv_check["passed"].astype(bool).sum()),
            "schemes": sorted(cv_check["sampling_scheme"].dropna().astype(str).unique()),
            "gee03_run_ids": runs,
        },
        "files": {name: _sha256(output_dir / name) for name in
                  ("final_vs_cv_training_rows.csv", "cv_scores_vs_manifest.csv")},
        "failed_checks": failed_checks,
    }
    (output_dir / VERDICT_NAME).write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    return verdict


def final_training_row_check(input_dir: Path, raw: pd.DataFrame) -> pd.DataFrame:
    """

"""
    rows: list[dict[str, object]] = []
    test = raw[(raw["analysis_type"] == "main_fixed") & (raw["model"] == "embeddings")
               & (raw["set"] == "test")]
    for scheme in ("primary_effort", "recurrent_effort"):
        row: dict[str, object] = {"sampling_scheme": scheme}
        problems: list[str] = []
        tables = {}
        for key, name in (
                ("final", f"SDM_FullModelPointPredictions_{scheme}_{ROW_CHECK_TAG}.csv"),
                ("manifest", f"CV_RowManifest_{scheme}_{ROW_CHECK_TAG}.csv"),
                ("mode", f"SDM_ProbabilityModeCheck_{scheme}_{ROW_CHECK_TAG}.csv"),
                ("rowcheck", f"SDM_TrainingRowCheck_{scheme}_{ROW_CHECK_TAG}.csv")):
            roots = [input_dir] if key == "manifest" else [input_dir, GEE04_DIR]
            path, status, cands = find_unique(roots, name)
            row[f"file_{key}"] = str(path) if path else f"{status}: " + "; ".join(
                str(c) for c in cands)
            if status != "ok":
                problems.append(f"{name} {status}")
                tables[key] = None
            else:
                tables[key] = pd.read_csv(path)
        cv = test[test["sampling_scheme"] == scheme]
        ident = row_identity_check(tables["final"], tables["manifest"], cv)
        problems += ident.pop("reasons")
        row.update({f"identity_{k}" if k == "passed" else k: v for k, v in ident.items()})
        run_ids = {}
        for key in ("final", "manifest", "mode", "rowcheck"):
            t = tables[key]
            if t is not None and "run_id" in t.columns:
                run_ids[key] = sorted(t["run_id"].astype(str).unique())
        g4 = {k: run_ids.get(k) for k in ("final", "mode", "rowcheck")}
        if any(v is None for k, v in g4.items() if tables[k] is not None) or \
                len({tuple(v) for v in g4.values() if v is not None}) > 1 or \
                any(v is not None and len(v) != 1 for v in g4.values()):
            problems.append("fitted points, probability check and row check do not "
                            "all come from one GEE_04 run")
        row["gee04_run_id"] = ";".join(run_ids.get("final", []))
        row["gee03_run_id"] = ";".join(run_ids.get("manifest", []))
        if "run_id" in cv.columns and cv["run_id"].notna().any():
            cv_runs = sorted(cv["run_id"].dropna().astype(str).unique())
            row["cv_scores_run_id"] = ";".join(cv_runs)
            if run_ids.get("manifest") and cv_runs != run_ids["manifest"]:
                problems.append("CV scores and CV row manifest come from different GEE_03 runs")
        mode = tables["mode"].iloc[0].to_dict() if tables["mode"] is not None else None
        # that the probability check verified for the final model.
        if mode is not None:
            want_mode = str(mode.get("presence_score_mode"))
            for label, t in (("CV test scores", cv), ("CV row manifest", tables["manifest"])):
                if t is None:
                    continue
                if "presence_score_mode" not in t.columns:
                    problems.append(f"{label} lack presence_score_mode (older GEE_03 export)")
                    continue
                got = set(t["presence_score_mode"].astype(str))
                if got != {want_mode}:
                    problems.append(f"{label} use score mode {sorted(got)}, the GEE_04 "
                                    f"check verified {want_mode}")
                if want_mode == "MULTIPROBABILITY_CLASS1":
                    idx = set(pd.to_numeric(t.get("class1_index"), errors="coerce").dropna())
                    if idx != {float(mode.get("configured_class1_index"))}:
                        problems.append(f"{label} use CLASS1_INDEX {sorted(idx)}, GEE_04 uses "
                                        f"{mode.get('configured_class1_index')}")
            row["presence_score_mode"] = want_mode
        verdict = probability_mode_verdict(mode)
        problems += verdict.pop("reasons")
        row.update(verdict)
        if tables["rowcheck"] is not None:
            c = tables["rowcheck"].iloc[0].to_dict()
            for k in ("n_rows_sampled_table", "n_rows_current_rule",
                      "n_rows_v34_filter_reconstruction",
                      "n_rows_v34_reconstruction_and_embeddings_complete",
                      "n_rows_current_rule_not_in_v34_reconstruction",
                      "n_rows_v34_reconstruction_not_in_current_rule",
                      "v34_filter_reconstructed"):
                row[k] = c.get(k)
            row["v34_note"] = ("reconstruction of the v34 filter from the same "
                               "sampled table; not a record of the rows the "
                               "archived v34 assets used")
        row["passed"] = not problems
        row["problems"] = " | ".join(problems)
        rows.append(row)
    return pd.DataFrame(rows)


def cv_manifest_check(input_dir: Path, raw: pd.DataFrame,
                      verified_modes: set[str]) -> pd.DataFrame:
    """"""
    frames = []
    for scheme, g in raw.groupby("sampling_scheme", sort=True):
        name = f"CV_RowManifest_{scheme}_{ROW_CHECK_TAG}.csv"
        path, status, cands = find_unique([input_dir], name)
        if status != "ok":
            frames.append(pd.DataFrame([{
                "sampling_scheme": scheme, "passed": False,
                "problems": f"{name} {status}: " + "; ".join(map(str, cands))}]))
            continue
        res = cv_scores_vs_manifest(g, pd.read_csv(path))
        res.insert(0, "sampling_scheme", scheme)
        if verified_modes and "presence_score_mode" in res.columns:
            bad = ~res["presence_score_mode"].isin(verified_modes)
            res.loc[bad, "passed"] = False
            res.loc[bad, "problems"] = (res.loc[bad, "problems"].fillna("") +
                                        " | score mode differs from the mode verified in GEE_04")
        frames.append(res)
    return pd.concat(frames, ignore_index=True, sort=False)


def calibration_outputs(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = raw[
        raw["set"].eq("test")
        & raw["analysis_type"].eq("main_fixed")
    ].copy()
    group_keys = [
        "analysis_type", "model", "sampling_scheme", "min_leaf",
        "variables_per_split"
    ]
    curve_parts = []
    summary_rows = []
    for key_values, group in subset.groupby(group_keys, dropna=False, sort=True):
        metadata = dict(zip(group_keys, key_values))
        if group["presence"].nunique() < 2:
            continue
        curve = equal_frequency_calibration(group["presence"], group["prob"])
        for column, value in metadata.items():
            curve[column] = value
        curve_parts.append(curve)
        summary_rows.append({
            **metadata,
            **probabilistic_metrics(group["presence"], group["prob"]),
        })
    curves = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame()
    return pd.DataFrame(summary_rows), curves


def residual_spatial_correlogram(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Approximate residual spatial correlation from sampled point pairs.

"""
    correlogram_parts: list[pd.DataFrame] = []
    range_rows: list[dict[str, object]] = []
    for selected_scheme in ["primary_effort", "recurrent_effort"]:
        subset = raw[
            (raw["analysis_type"] == "main_fixed")
            & (raw["model"] == "embeddings")
            & (raw["sampling_scheme"] == selected_scheme)
            & (raw["set"] == "test")
            & np.isfinite(raw["lon"])
            & np.isfinite(raw["lat"])
        ].copy()
        if len(subset) < 500:
            continue

        subset = subset.sort_values("source_file").drop_duplicates(
            subset=["lon", "lat", "presence"], keep="last"
        ).reset_index(drop=True)
        longitude = subset["lon"].to_numpy(float)
        latitude = subset["lat"].to_numpy(float)
        latitude0 = float(np.mean(latitude))
        x = longitude * 111.320 * math.cos(math.radians(latitude0))
        y_coordinate = latitude * 110.574
        residual = subset["presence"].to_numpy(float) \
            - subset["prob"].to_numpy(float)
        residual_centre = float(np.mean(residual))
        residual_variance = float(np.var(residual, ddof=1))
        if residual_variance <= 0:
            continue

        rng = np.random.default_rng(20250523)
        number_of_points = len(subset)
        first = rng.integers(0, number_of_points, SPATIAL_PAIR_SAMPLE)
        second = rng.integers(0, number_of_points, SPATIAL_PAIR_SAMPLE)
        unequal = first != second
        first, second = first[unequal], second[unequal]
        distance = np.hypot(x[first] - x[second],
                            y_coordinate[first] - y_coordinate[second])
        within = (distance > 0) & (distance <= SPATIAL_MAX_DISTANCE_KM)
        first, second, distance = first[within], second[within], distance[within]
        pair_product = (
            (residual[first] - residual_centre)
            * (residual[second] - residual_centre)
            / residual_variance)
        semivariance = 0.5 * (residual[first] - residual[second]) ** 2
        edges = np.arange(
            0.0, SPATIAL_MAX_DISTANCE_KM + SPATIAL_BIN_WIDTH_KM,
            SPATIAL_BIN_WIDTH_KM)
        bin_index = np.digitize(distance, edges, right=False) - 1
        rows = []
        for index in range(len(edges) - 1):
            selected = bin_index == index
            if selected.sum() < 500:
                continue
            rows.append({
                "sampling_scheme": selected_scheme,
                "distance_low_km": edges[index],
                "distance_high_km": edges[index + 1],
                "distance_mid_km": (edges[index] + edges[index + 1]) / 2.0,
                "n_sampled_pairs": int(selected.sum()),
                "residual_correlation": float(np.mean(pair_product[selected])),
                "semivariance": float(np.mean(semivariance[selected])),
            })
        scheme_correlogram = pd.DataFrame(rows)
        if scheme_correlogram.empty:
            continue
        correlogram_parts.append(scheme_correlogram)

        empirical_range = np.nan
        below = scheme_correlogram["residual_correlation"].abs().to_numpy() \
            <= 0.05
        for index in range(len(below) - 1):
            if below[index] and below[index + 1]:
                empirical_range = float(
                    scheme_correlogram.iloc[index]["distance_high_km"])
                break
        range_rows.append({
            "sampling_scheme": selected_scheme,
            "analysis_type": "main_fixed",
            "n_oof_points": number_of_points,
            "n_random_pairs_requested": SPATIAL_PAIR_SAMPLE,
            "n_pairs_within_max_distance": len(distance),
            "max_distance_km": SPATIAL_MAX_DISTANCE_KM,
            "bin_width_km": SPATIAL_BIN_WIDTH_KM,
            "range_definition":
                "first_of_two_consecutive_bins_with_abs_correlation_le_0.05",
            "empirical_residual_range_km": empirical_range,
            "main_block_size_degrees":
                SCHEME_MAIN_BLOCK_DEG.get(selected_scheme, 0.30),
            "main_block_north_south_extent_km":
                SCHEME_MAIN_BLOCK_DEG.get(selected_scheme, 0.30) * 110.574,
            "main_block_east_west_extent_km_at_mean_latitude":
                SCHEME_MAIN_BLOCK_DEG.get(selected_scheme, 0.30)
                * 111.320 * math.cos(math.radians(latitude0)),
            "block_exceeds_range_both_dimensions": bool(
                np.isfinite(empirical_range)
                and min(
                    SCHEME_MAIN_BLOCK_DEG.get(selected_scheme, 0.30)
                    * 110.574,
                    SCHEME_MAIN_BLOCK_DEG.get(selected_scheme, 0.30)
                    * 111.320 * math.cos(math.radians(latitude0)),
                ) >= empirical_range),
            "mean_latitude_deg": latitude0,
        })

    if not correlogram_parts:
        return pd.DataFrame(), pd.DataFrame()
    return (pd.concat(correlogram_parts, ignore_index=True),
            pd.DataFrame(range_rows))


def repeated_summary(pooled: pd.DataFrame) -> pd.DataFrame:
    repeated = pooled[pooled["analysis_type"] == "repeated_cv"]
    if repeated.empty:
        return pd.DataFrame()
    rows = []
    for (scheme, min_leaf), group in repeated.groupby(
        ["sampling_scheme", "min_leaf"], sort=True
    ):
        row: dict[str, object] = {
            "sampling_scheme": scheme,
            "min_leaf": int(min_leaf),
            "n_repeats": len(group),
        }
        for metric in [
            "AUC", "Brier_Skill_Score", "Brier_Score", "Log_Loss",
            "Calibration_Intercept", "Calibration_Slope", "ECE_10bin",
        ]:
            stats = mean_sd_ci(group[metric])
            for name, value in stats.items():
                row[f"{metric}_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def block_summary(pooled: pd.DataFrame) -> pd.DataFrame:
    block = pooled[pooled["analysis_type"] == "block_sensitivity"].copy()
    headline_030 = pooled[
        pooled["analysis_type"].eq("main_fixed")
        & pooled["model"].eq("embeddings")
        & pooled["sampling_scheme"].isin(
            ["primary_effort", "recurrent_effort"])
        & pooled["min_leaf"].eq(FIXED_MIN_LEAF)
        & pooled["variables_per_split"].eq(FIXED_MTRY)
    ].copy()
    if not headline_030.empty:
        headline_030["analysis_type"] = "block_sensitivity"
        block = pd.concat([block, headline_030], ignore_index=True, sort=False)
    keep = [
        "sampling_scheme", "block_size", "min_leaf", "n", "AUC",
        "Brier_Skill_Score", "Brier_Score", "Log_Loss",
        "Calibration_Intercept", "Calibration_Slope", "ECE_10bin",
    ]
    return block[[column for column in keep if column in block.columns]]


def regularisation_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarise the diagnostic leaf/mtry cross plus the predeclared model."""
    sensitivity_rows = metrics["analysis_type"].eq(
        "regularisation_sensitivity")
    headline_rows = (
        metrics["analysis_type"].eq("main_fixed")
        & metrics["model"].eq("embeddings")
        & metrics["min_leaf"].eq(FIXED_MIN_LEAF)
        & metrics["variables_per_split"].eq(FIXED_MTRY)
        & metrics["sampling_scheme"].isin(
            ["primary_effort", "recurrent_effort"])
    )
    subset = metrics[(sensitivity_rows | headline_rows)
                     & metrics["set"].eq("test")]
    rows = []
    for (scheme, minimum_leaf, mtry), group in subset.groupby(
        ["sampling_scheme", "min_leaf", "variables_per_split"], sort=True
    ):
        row: dict[str, object] = {
            "sampling_scheme": scheme,
            "min_leaf": int(minimum_leaf),
            "variables_per_split": int(mtry),
            "n_folds": len(group),
            "role": "predeclared_final" if (
                int(minimum_leaf) == FIXED_MIN_LEAF
                and int(mtry) == FIXED_MTRY
            ) else "regularisation_sensitivity_only",
        }
        for metric in [
            "AUC", "Brier_Skill_Score", "Brier_Score", "Log_Loss",
            "Calibration_Slope", "ECE_10bin", "TSS_fixed",
        ]:
            stats = mean_sd_ci(group[metric])
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_sd"] = stats["sd"]
        rows.append(row)
    return pd.DataFrame(rows)


def input_validation(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = [
        "analysis_type", "model", "sampling_scheme", "repeat_id",
        "block_size", "min_leaf", "variables_per_split", "outer_fold",
        "inner_fold", "fold_id", "set",
    ]
    for key_values, group in raw.groupby(group_keys, dropna=False, sort=True):
        rows.append({
            **dict(zip(group_keys, key_values)),
            "n": len(group),
            "n_presence": int((group["presence"] == 1).sum()),
            "n_absence": int((group["presence"] == 0).sum()),
            "prevalence": float(group["presence"].mean()),
            "n_unique_blocks": int(group["block_id"].nunique(dropna=True)),
            "probability_min": float(group["prob"].min()),
            "probability_max": float(group["prob"].max()),
            "both_classes_present": int(group["presence"].nunique() == 2),
        })
    return pd.DataFrame(rows)


def required_main_cv_status(raw: pd.DataFrame) -> pd.DataFrame:
    """Audit the exact main-CV matrix; never publish a partial model set."""
    requirements = []
    core_models = ["embeddings", "baseline_simple", "ndvi_null"]
    schemes = [
        "primary_effort", "access_matched", "recurrent_effort",
        "recurrent_access_matched",
    ]
    for scheme in schemes:
        requirements.append((
            "main_fixed", scheme, "embeddings", "blocked_required"))
    # and the random-fold transferability check.
    for comparison_scheme in ["primary_effort", "recurrent_effort"]:
        for model in core_models[1:]:
            requirements.append((
                "main_fixed", comparison_scheme, model,
                "model_comparison_required"))
        requirements.append((
            "random_cv", comparison_scheme, "embeddings",
            "random_comparison_required"))
    # must never halt the metrics run.
    requirements.append((
        "main_fixed", "primary_effort_full", "embeddings",
        "full_window_sensitivity_optional"))

    rows = []
    for analysis_type, scheme, model, role in requirements:
        # requiring 16 for the 11-predictor baseline or the 1-predictor NDVI
        # null would flag a complete run as missing.
        expected_mtry = EXPECTED_MTRY_BY_MODEL.get(model, FIXED_MTRY)
        selected = raw[
            raw["analysis_type"].eq(analysis_type)
            & raw["sampling_scheme"].eq(scheme)
            & raw["model"].eq(model)
            & raw["min_leaf"].eq(FIXED_MIN_LEAF)
            & raw["variables_per_split"].eq(expected_mtry)
        ]
        test_folds = sorted(
            pd.to_numeric(selected.loc[selected["set"].eq("test"), "fold_id"],
                          errors="coerce").dropna().astype(int).unique().tolist())
        train_folds = sorted(
            pd.to_numeric(selected.loc[selected["set"].eq("train"), "fold_id"],
                          errors="coerce").dropna().astype(int).unique().tolist())
        complete = test_folds == list(range(EXPECTED_FOLDS)) and \
            train_folds == list(range(EXPECTED_FOLDS))
        rows.append({
            "analysis_type": analysis_type,
            "sampling_scheme": scheme,
            "model": model,
            "expected_variables_per_split": expected_mtry,
            "role": role,
            "test_folds_found": ",".join(map(str, test_folds)),
            "train_folds_found": ",".join(map(str, train_folds)),
            "expected_folds": "0,1,2,3,4",
            "complete": int(complete),
            "n_rows": len(selected),
        })
    return pd.DataFrame(rows)


def spatial_transferability_penalty(pooled: pd.DataFrame) -> pd.DataFrame:
    """Random-fold minus blocked-fold performance; blocked remains headline."""
    main = pooled[
        pooled["analysis_type"].isin(["main_fixed", "random_cv"])
        & pooled["model"].eq("embeddings")
        & pooled["min_leaf"].eq(FIXED_MIN_LEAF)
        & pooled["variables_per_split"].eq(FIXED_MTRY)
    ].copy()
    rows = []
    for scheme, group in main.groupby("sampling_scheme", sort=True):
        by_type = group.set_index("analysis_type")
        if not {"main_fixed", "random_cv"}.issubset(by_type.index):
            continue
        blocked = by_type.loc["main_fixed"]
        random = by_type.loc["random_cv"]
        rows.append({
            "sampling_scheme": scheme,
            "model": "embeddings",
            "blocked_AUC_headline": blocked["AUC"],
            "random_fold_AUC": random["AUC"],
            "spatial_transferability_penalty_AUC": random["AUC"] - blocked["AUC"],
            "blocked_BSS": blocked["Brier_Skill_Score"],
            "random_fold_BSS": random["Brier_Skill_Score"],
            "spatial_transferability_penalty_BSS": (
                random["Brier_Skill_Score"] - blocked["Brier_Skill_Score"]),
            "interpretation": "random_minus_blocked;blocked_is_headline",
        })
    return pd.DataFrame(rows)


def scoring_coverage_check(
    pooled: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """

"""
    audit_path = output_dir / "fold_assignment_audit.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else None
    audit_col = None
    if audit is not None:
        audit_col = ("n_rows_complete" if "n_rows_complete" in audit.columns
                     else "n_rows" if "n_rows" in audit.columns else None)
        fold_columns = [c for c in audit.columns
                        if c.startswith("n_test_fold")]
        if fold_columns and "per_fold_expected" not in audit.columns:
            audit["per_fold_expected"] = audit[fold_columns].astype(int) \
                .apply(lambda row: ";".join(map(str, row)), axis=1)
    rows: list[dict[str, object]] = []
    for scheme, group in pooled[
        pooled["analysis_type"].isin(["main_fixed", "random_cv"])
    ].groupby("sampling_scheme", sort=True):
        # within files must equal the audited collection size; the comparison
        # therefore uses the effective (restored) counts.
        counts = {
            f"{row.analysis_type}:{row.model}": int(row.n)
            + int(DEDUP_REMOVED_TEST_COUNTS.get(
                (scheme, row.analysis_type, row.model), 0))
            for row in group.itertuples()
        }
        values = sorted(set(counts.values()))
        expected = None
        if audit is not None and audit_col is not None:
            match = audit[audit["scheme"] == scheme]
            if not match.empty:
                expected = int(match.iloc[0][audit_col])
        consistent = len(values) == 1 and (
            expected is None or values[0] == expected)
        per_fold_expected = None
        if audit is not None and "per_fold_expected" in audit.columns:
            match = audit[audit["scheme"] == scheme]
            if not match.empty:
                per_fold_expected = match.iloc[0]["per_fold_expected"]
        rows.append({
            "per_fold_expected_from_audit": per_fold_expected,
            "sampling_scheme": scheme,
            "scheme_label": SCHEME_LABELS.get(scheme, scheme),
            "distinct_row_counts": ";".join(map(str, values)),
            "per_run_counts": "; ".join(
                f"{key}={value}" for key, value in sorted(counts.items())),
            "audit_n_rows_complete": expected,
            "consistent": bool(consistent),
        })
    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "scoring_coverage_check.csv", index=False)
    failures = report[~report["consistent"]]
    if len(failures):
        print("\n" + "!" * 74)
        print("SCORING-COVERAGE CHECK FAILED for "
              f"{len(failures)} scheme(s): models are NOT scored on "
              "identical rows.")
        print(failures[["sampling_scheme", "distinct_row_counts",
                        "audit_n_rows_complete"]].to_string(index=False))
        print("Do not report the model comparison until this passes "
              "(see scoring_coverage_check.csv).")
        print("!" * 74)
    else:
        print("\nScoring-coverage check PASSED: every model and the "
              "random-fold run score identical rows in every scheme.")
    return report


def results_stability_check(
    pooled: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """

"""
    main = pooled[pooled["analysis_type"] == "main_fixed"]
    rows: list[dict[str, object]] = []
    for (scheme, model), reference in sorted(STABILITY_REFERENCE_AUC.items()):
        observed = main[(main["sampling_scheme"] == scheme)
                        & (main["model"] == model)]
        auc = float(observed["AUC"].iloc[0]) if len(observed) else np.nan
        delta = auc - reference if np.isfinite(auc) else np.nan
        rows.append({
            "sampling_scheme": scheme,
            "scheme_label": SCHEME_LABELS.get(scheme, scheme),
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "reference_AUC_v13b3": reference,
            "observed_AUC": round(auc, 4) if np.isfinite(auc) else np.nan,
            "delta": round(delta, 4) if np.isfinite(delta) else np.nan,
            "tolerance": STABILITY_TOLERANCE_AUC,
            "within_tolerance": bool(
                np.isfinite(delta) and abs(delta) <= STABILITY_TOLERANCE_AUC),
        })
    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "results_stability_check.csv", index=False)
    flagged = report[~report["within_tolerance"]
                     & np.isfinite(report["delta"].astype(float))]
    print("\nResults-stability check against the v13b3 reference:")
    print(report[["scheme_label", "model_label", "reference_AUC_v13b3",
                  "observed_AUC", "delta", "within_tolerance"]]
          .to_string(index=False))
    if len(flagged):
        print("INVESTIGATE: the flagged rows moved more than "
              f"{STABILITY_TOLERANCE_AUC} AUC against the reference; do not "
              "report them until the cause is understood.")
    return report


def plot_calibration(curves: pd.DataFrame, output_dir: Path) -> None:
    if curves.empty:
        return
    subset = curves[
        (curves["analysis_type"] == "main_fixed")
        & (curves["model"] == "embeddings")
    ]
    if subset.empty:
        return
    fig, axis = plt.subplots(figsize=(7.0, 6.2))
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.45", label="Ideal")
    colours = {
        "primary_effort": "#1b9e77",
        "primary_effort_full": "#66a61e",
        "access_matched": "#d95f02",
        "recurrent_effort": "#7570b3",
        "recurrent_access_matched": "#e7298a",
    }
    labels = dict(SCHEME_LABELS)
    for scheme, group in subset.groupby("sampling_scheme", sort=True):
        group = group.sort_values("mean_predicted")
        axis.errorbar(
            group["mean_predicted"], group["observed_fraction"],
            yerr=[
                group["observed_fraction"] - group["observed_ci_low"],
                group["observed_ci_high"] - group["observed_fraction"],
            ],
            marker="o", linewidth=1.7, capsize=2.5,
            color=colours.get(scheme), label=labels.get(scheme, scheme),
        )
    axis.set(xlim=(0, 1), ylim=(0, 1),
             xlabel="Mean predicted visitation-propensity score",
             ylabel="Observed fraction of visited cells")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_main_models.png", dpi=300)
    fig.savefig(output_dir / "calibration_main_models.pdf")
    plt.close(fig)


def plot_sensitivities(
    regularisation: pd.DataFrame, block: pd.DataFrame, pooled: pd.DataFrame,
    output_dir: Path,
) -> None:
    if regularisation.empty and block.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    if not regularisation.empty:
        ordered = regularisation.sort_values(
            ["min_leaf", "variables_per_split"]).reset_index(drop=True)
        labels = [
            f"leaf {int(row.min_leaf)}\nmtry {int(row.variables_per_split)}"
            for row in ordered.itertuples(index=False)
        ]
        positions = np.arange(len(ordered))
        colours = np.where(
            ordered["role"].eq("predeclared_final"), "#d95f02", "#4c78a8")
        axes[0].bar(positions, ordered["AUC_mean"], color=colours, alpha=0.85)
        axes[0].errorbar(
            positions, ordered["AUC_mean"], yerr=ordered["AUC_sd"],
            fmt="none", color="0.2", capsize=3)
        axes[0].set(
            xticks=positions, xticklabels=labels,
            ylabel="Spatial-CV AUC", title="Regularisation sensitivity")
        axes[0].tick_params(axis="x", labelsize=8)
    else:
        axes[0].text(0.5, 0.5, "Regularisation exports not yet available",
                     ha="center", va="center")
        axes[0].set_axis_off()
    if not block.empty:
        for scheme, group in block.groupby("sampling_scheme", sort=True):
            group = group.sort_values("block_size")
            axes[1].plot(group["block_size"], group["AUC"],
                         marker="o", label=scheme)
        axes[1].set(xlabel="Block size (degrees)", ylabel="Pooled out-of-fold AUC",
                    title="Block-size sensitivity")
        axes[1].legend(frameon=False)
    else:
        axes[1].text(0.5, 0.5, "block-size exports not yet available",
                     ha="center", va="center")
        axes[1].set_axis_off()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "regularisation_and_block_sensitivity.png", dpi=300)
    fig.savefig(output_dir / "regularisation_and_block_sensitivity.pdf")
    plt.close(fig)


def plot_residual_correlogram(
    correlogram: pd.DataFrame, range_table: pd.DataFrame, output_dir: Path
) -> None:
    if correlogram.empty:
        return
    scheme_colours = {"primary_effort": "#4c78a8",
                      "recurrent_effort": "#7570b3"}
    fig, axis = plt.subplots(figsize=(7.6, 5.0))
    axis.axhline(0, color="0.35", linewidth=1)
    axis.axhline(0.05, color="0.65", linewidth=0.8, linestyle="--")
    axis.axhline(-0.05, color="0.65", linewidth=0.8, linestyle="--")
    for scheme, group in correlogram.groupby("sampling_scheme", sort=True):
        colour = scheme_colours.get(scheme, "#4c78a8")
        axis.plot(group["distance_mid_km"], group["residual_correlation"],
                  marker="o", linewidth=1.8, color=colour,
                  label=SCHEME_LABELS.get(scheme, scheme))
        if not range_table.empty:
            row = range_table[range_table["sampling_scheme"] == scheme]
            if not row.empty and np.isfinite(
                    row.iloc[0]["empirical_residual_range_km"]):
                spatial_range = float(
                    row.iloc[0]["empirical_residual_range_km"])
                axis.axvline(spatial_range, color=colour, linestyle=":",
                             linewidth=1.6, alpha=0.9)
                axis.text(spatial_range, axis.get_ylim()[1], f" {spatial_range:.0f} km",
                          va="top", fontsize=8, color=colour)
    if not range_table.empty:
        for _, row in range_table.iterrows():
            colour = scheme_colours.get(row["sampling_scheme"], "0.3")
            extent = float(min(
                row["main_block_east_west_extent_km_at_mean_latitude"],
                row["main_block_north_south_extent_km"]))
            axis.axvline(extent, color=colour, linestyle="--",
                         linewidth=1.1, alpha=0.7)
            axis.text(extent, axis.get_ylim()[0],
                      f" {row['main_block_size_degrees']:.2f}"
                      f"deg block\n min {extent:.0f} km",
                      va="bottom", fontsize=7, color=colour)
    axis.set(
        xlabel="Pair distance (km)",
        ylabel="Out-of-fold residual correlation",
        title="Residual spatial correlogram "
              "(0.30-degree blocks exceed the empirical range)",
        xlim=(0, SPATIAL_MAX_DISTANCE_KM))
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "residual_spatial_correlogram.png", dpi=300)
    fig.savefig(output_dir / "residual_spatial_correlogram.pdf")
    plt.close(fig)


def write_methods_note(
    output_dir: Path, repeated: pd.DataFrame
) -> None:
    note = f"""REPRODUCIBLE VALIDATION NOTE

Two sampling estimands are retained. The symmetric effort-weighted background
is the primary analysis of realised visitation propensity. The background
matched on sampling effort and distance to infrastructure is a sensitivity
analysis of discrimination conditional on comparable accessibility. Training
is symmetric in WorldCover classes 10, 20, 30, 80, 90 and 100. WorldCover is a
training-domain filter only: final probability and change rasters cover every
valid model-input pixel inside the AOI, including cropland, built-up and
bare/sparse land. Pixels outside the six training classes are declared transfer
predictions, not directly validated visitation probabilities. Same-run novelty,
land-cover, probability-interior and wartime-variability diagnostics accompany
those predictions but do not cut holes in the primary rasters. Cropland remains
the predeclared focal transfer analysis for post-2022 agricultural abandonment;
other land-cover classes are reported transparently rather than receiving
class-specific masking exceptions.
The accessibility-matched background is selected from the effort-weighted
candidate pool and therefore inherits its broad spatial composition; balance
is enforced on the prespecified log-effort and log-distance marginals using a
maximum absolute standardised mean difference of 0.10.

Before generating v13b held-out predictions, the final Random Forest was fixed
at 500 trees, minLeafPopulation={FIXED_MIN_LEAF} and
variablesPerSplit={FIXED_MTRY}: all 64 embedding bands are predictors, and each
split evaluates a random subset of {FIXED_MTRY} of them (3 for the
11-predictor environmental baseline, 1 for the NDVI-only model). The NDVI
predictor is the summer-median Sentinel-2 NDVI (May-September 2019-2021) at
each record location; the other baseline predictors are 500 m values. No CV result
selects or replaces that setting. A small prespecified diagnostic set compares
(leaf,mtry) values (5,8), (5,16), (20,8), (20,21) and (50,16) on the same
spatial folds; these rows are regularisation sensitivity only.

Spatially blocked five-fold CV is the headline. Whole blocks of equal size in
degrees are assigned greedily to class-balanced folds. The block size is the
smallest 0.05-degree step whose east-west and north-south extent exceeded the
model's empirical residual range (first of two consecutive 10-km bins with
absolute out-of-fold residual correlation <= 0.05) in an earlier run: 0.30
degrees for the regular model (range 20 km) and 0.45 degrees for the recurrent
model (range 30 km then). The ranges of this run are in
residual_spatial_range.csv; where 0.45 degrees no longer exceeds the recurrent
range east-west, the 0.60-degree blocks below are the check. For both models the embedding model is also validated with three
seeded random block-to-fold assignments, smaller and larger blocks (0.15 and
0.60 degrees for the regular model; 0.30 and 0.60 for the recurrent model),
ordinary random folds (spatial transferability penalty = random-fold AUC minus
blocked AUC; matched pairs share a random fold) and the regularisation set.
The embedding model is compared with the environmental baseline and the
NDVI-only model on identical complete-case rows, by paired t-tests on the five
test-fold AUCs (model_comparison_paired_tests.csv).

Discrimination is reported using ROC AUC and average precision. Calibration is
reported using log loss, Brier score, Brier skill score (reference: a constant
prediction equal to the fold's share of presences), calibration intercept,
calibration slope, ten-bin expected calibration error and reliability curves.
Because the training sample has approximately 50% prevalence by design,
calibration and threshold-dependent metrics apply to this sampling design;
they do not estimate the absolute real-world probability of visitation.

The fixed classification threshold is {FIXED_THRESHOLD:.2f}. When a max-TSS
threshold is shown, it is selected exclusively from the corresponding training
partition and applied unchanged to its held-out test partition.

Train-test AUC gap is AUC_train minus AUC_test. Held-out folds are scored in
full. Training performance is estimated from a fixed-seed subsample of at most
3,000 presences and 3,000 pseudo-absences per fold after the model has been
fitted to the complete training partition (an Earth Engine memory limit that
affects this diagnostic only). Model fitting uses every eligible record; no
record cap or thinning is applied to the fitted models.

The embedding representation is fixed before validation: each annual Google
Satellite Embedding mosaic is materialised on the fixed 500 m EPSG:3035 grid
(GEE_01), and the same 64 bands are used for sampling, CV, final maps and the
extrapolation flag. Rows lacking any required predictor are removed before CV
(counts: fold_assignment_audit.csv and the GEE_06 audit). Out-of-fold
performance is additionally reported by the pre-2022 occupation line, by
WorldCover class and by platform. A full 2015-2021 presence-window model is
fitted as a labelled sensitivity scheme. All tabular outputs carry
manuscript-terminology label columns beside the technical keys;
LABELS_GLOSSARY.csv maps one to the other.

Co-located records. H3 resolution-8 response cells (average edge ~0.53 km,
area ~0.74 km2) and 500 m analysis pixels do not nest, so a small share of
records (~2-5% depending on the layer) shares a pixel with another record;
their exported coordinates coincide with the pixel centre and their
embedding-based scores are identical. These are distinct records and are
retained in all metrics (earlier pipeline versions collapsed them as
apparent duplicates); their counts are reported in
colocated_record_counts.csv and belong in the SI data-flow table.
"""
    (output_dir / "MANUSCRIPT_METHODS_NOTE.txt").write_text(note, encoding="utf-8")


def interpretability_outputs(input_dir: Path, output_dir: Path) -> None:
    """Summarise the optional same-run sample exported by GEE_04.

    These are marginal descriptive summaries, not causal partial-dependence
    curves. They expose extrapolation-like behaviour in familiar variables
    while RF importance reports the contribution of the latent embedding bands.
    """
    files = sorted({path for d in (input_dir, folders.data_dir(GEE04_DIR)) if d.is_dir()
                    for path in d.glob("SDM_InterpretabilitySample_*.csv")
                    if "v13b" in path.name.lower()})
    if not files:
        return
    sample = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    required = {"predicted_probability", "ndvi_2019_2021", "lc"}
    missing = required.difference(sample.columns)
    if missing:
        print("WARNING: interpretability sample is missing columns: "
              + ", ".join(sorted(missing)))
        return
    for column in required:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    for column in ["extrapolation_flag", "in_training_landcover_domain"]:
        if column in sample:
            sample[column] = pd.to_numeric(sample[column], errors="coerce")
    sample = sample.dropna(subset=list(required))
    sample = sample[sample["predicted_probability"].between(0, 1)]

    legend = {
        10: "Tree cover", 20: "Shrubland", 30: "Grassland",
        40: "Cropland", 50: "Built-up",
        60: "Bare / sparse vegetation", 70: "Snow and ice",
        80: "Permanent water bodies", 90: "Herbaceous wetland",
        95: "Mangroves",
        100: "Moss and lichen",
    }
    sample = sample[sample["lc"].round().isin(legend)].copy()
    sample["lc"] = sample["lc"].round().astype(int)
    sample["landcover"] = sample["lc"].map(legend)
    sample["prediction_role"] = np.where(
        sample["lc"].isin({10, 20, 30, 80, 90, 100}),
        "training_landcover_domain", "transfer_landcover_domain")
    grouping = ["lc", "landcover", "prediction_role"]
    if "sampling_scheme" in sample:
        grouping.insert(0, "sampling_scheme")
    aggregations = {
        "n": ("predicted_probability", "size"),
        "mean_probability": ("predicted_probability", "mean"),
        "sd_probability": ("predicted_probability", "std"),
        "median_probability": ("predicted_probability", "median"),
    }
    if "extrapolation_flag" in sample:
        aggregations["novelty_flag_fraction"] = (
            "extrapolation_flag", "mean")
    by_landcover = (sample.dropna(subset=["landcover"])
                    .groupby(grouping, as_index=False)
                    .agg(**aggregations))
    with_labels(by_landcover).to_csv(
        output_dir / "interpretability_by_landcover.csv", index=False)

    edges = np.round(np.arange(-0.2, 1.0001, 0.1), 10)
    sample["ndvi_bin"] = pd.cut(
        sample["ndvi_2019_2021"], bins=edges, include_lowest=True)
    ndvi_grouping = ["ndvi_bin"]
    if "sampling_scheme" in sample:
        ndvi_grouping.insert(0, "sampling_scheme")
    by_ndvi = (sample.dropna(subset=["ndvi_bin"])
               .groupby(ndvi_grouping, observed=True)
               .agg(n=("predicted_probability", "size"),
                    mean_probability=("predicted_probability", "mean"),
                    sd_probability=("predicted_probability", "std"),
                    median_probability=("predicted_probability", "median"))
               .reset_index())
    by_ndvi["ndvi_bin"] = by_ndvi["ndvi_bin"].astype(str)
    with_labels(by_ndvi).to_csv(
        output_dir / "interpretability_by_ndvi.csv", index=False)


def variable_importance_outputs(input_dir: Path, output_dir: Path) -> None:
    """Combine same-run RF importance exports and write a compact plot."""
    files = sorted({path for d in (input_dir, folders.data_dir(GEE04_DIR)) if d.is_dir()
                    for path in d.glob("SDM_VariableImportance_*.csv")
                    if "v13b" in path.name.lower()})
    if not files:
        return
    parts = []
    for path in files:
        frame = pd.read_csv(path)
        if {"predictor", "importance"}.issubset(frame.columns):
            frame["source_file"] = path.name
            parts.append(frame)
    if not parts:
        return
    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined["importance"] = pd.to_numeric(
        combined["importance"], errors="coerce")
    combined = combined.dropna(subset=["predictor", "importance"])
    with_labels(combined).to_csv(
        output_dir / "variable_importance.csv", index=False)
    grouping = "sampling_scheme" if "sampling_scheme" in combined else None
    groups = combined.groupby(grouping, sort=True) if grouping else [("model", combined)]
    for scheme, group in groups:
        top = group.nlargest(20, "importance").sort_values("importance")
        if top.empty:
            continue
        fig, axis = plt.subplots(figsize=(7.2, 6.0))
        axis.barh(top["predictor"].astype(str), top["importance"], color="#4c78a8")
        axis.set(
            xlabel="Random Forest importance",
            ylabel="Predictor",
            title=f"Top predictors: {scheme}")
        axis.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        safe_scheme = re.sub(r"[^A-Za-z0-9_-]+", "_", str(scheme))
        fig.savefig(
            output_dir / f"variable_importance_{safe_scheme}.png", dpi=300)
        fig.savefig(output_dir / f"variable_importance_{safe_scheme}.pdf")
        plt.close(fig)


def run_self_test() -> None:
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert np.isclose(auc_safe(y, perfect), 1.0)
    assert np.isclose(threshold_metrics(y, perfect, 0.5)["tss"], 1.0)
    metrics = probabilistic_metrics(y, perfect)
    assert metrics["Brier_Skill_Score"] > 0
    reversed_probability = 1 - perfect
    assert np.isclose(auc_safe(y, reversed_probability), 0.0)
    synthetic_rows = []
    for fold in range(5):
        for set_name, shift in [("train", 0.10), ("test", 0.0)]:
            for index in range(40):
                outcome = index % 2
                base = 0.72 if outcome else 0.28
                synthetic_rows.append({
                    "analysis_type": "main_fixed",
                    "model": "embeddings",
                    "sampling_scheme": "primary_effort",
                    "repeat_id": "det",
                    "block_size": 0.3,
                    "min_leaf": 5,
                    "variables_per_split": 8,
                    "outer_fold": np.nan,
                    "inner_fold": np.nan,
                    "fold_id": fold,
                    "set": set_name,
                    "presence": outcome,
                    "prob": np.clip(
                        base + (shift if outcome else -shift), 0.01, 0.99),
                })
    synthetic = pd.DataFrame(synthetic_rows)
    partition = partition_metrics(synthetic)
    gaps, gap_summary = train_test_gaps(partition)
    assert len(partition) == 10
    assert len(gaps) == 5
    assert len(gap_summary) == 1
    print("Self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    global GEE04_DIR
    input_dir = folders.data_dir(args.input_dir.expanduser()).resolve()
    GEE04_DIR = args.gee04_dir.expanduser()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # until this run has passed every check (a crash leaves it in place).
    (output_dir / VERDICT_NAME).unlink(missing_ok=True)
    (output_dir / MARKER_NAME).write_text(
        "Python_01 started and has not finished; any earlier verdict is void.\n",
        encoding="utf-8")

    raw, inventory = load_raw_scores(input_dir)
    inventory.to_csv(output_dir / "run_inventory.csv", index=False)
    write_labels_glossary(output_dir)
    # alongside the score files. n_fold_assigned must equal n_rows.
    audit_files = sorted(input_dir.glob("FoldAssignment_Audit_*.csv"))
    if audit_files:
        audits = pd.concat(
            (pd.read_csv(path) for path in audit_files), ignore_index=True)
        audits.to_csv(output_dir / "fold_assignment_audit.csv", index=False)
        rows_col = ("n_rows_complete" if "n_rows_complete" in audits.columns
                    else "n_rows")
        bad = audits[
            pd.to_numeric(audits.get("n_fold_assigned"), errors="coerce")
            != pd.to_numeric(audits.get(rows_col), errors="coerce")]
        if len(bad):
            print("WARNING: fold-assignment audit shows unassigned rows:")
            print(bad.to_string(index=False))
        else:
            print("Fold-assignment audit clean: every complete-case row "
                  "carries a blocked fold.")
    else:
        print("NOTE: FoldAssignment_Audit_*.csv not found in the input "
              "directory; download them with the score files to verify the "
              "v13b3 fold fix.")
    validation = input_validation(raw)
    with_labels(validation).to_csv(
        output_dir / "cv_input_validation.csv", index=False)
    required_status = required_main_cv_status(raw)
    with_labels(required_status).to_csv(
        output_dir / "required_main_cv_status.csv", index=False)
    incomplete = required_status[
        required_status["complete"].eq(0)
        & ~required_status["role"].str.contains("optional")
    ]
    if not incomplete.empty:
        missing = "; ".join(
            f"{row.analysis_type}/{row.sampling_scheme}/{row.model} "
            f"(test={row.test_folds_found or 'none'}, "
            f"train={row.train_folds_found or 'none'})"
            for row in incomplete.itertuples(index=False)
        )
        raise RuntimeError(
            "Main CV is incomplete, so metrics were intentionally stopped. "
            "Download every GEE_03 fold CSV into the input directory. Missing: "
            + missing
        )

    metrics = partition_metrics(raw)
    with_labels(metrics).to_csv(
        output_dir / "metrics_per_partition.csv", index=False)
    summary = summarise_partitions(metrics)
    with_labels(summary).to_csv(
        output_dir / "metrics_summary.csv", index=False)
    paired = paired_model_tests(metrics)
    with_labels(paired).to_csv(
        output_dir / "model_comparison_paired_tests.csv", index=False)
    if not paired.empty:
        print("\nPaired per-fold t-tests (blocked test-fold AUC):")
        print(paired.to_string(index=False,
                               float_format=lambda value: f"{value:.4f}"))
    pooled = pooled_test_metrics(raw)
    with_labels(pooled).to_csv(
        output_dir / "pooled_out_of_fold_metrics.csv", index=False)

    gaps, gap_summary = train_test_gaps(metrics)
    with_labels(gaps).to_csv(
        output_dir / "train_test_gap_per_partition.csv", index=False)
    with_labels(gap_summary).to_csv(
        output_dir / "train_test_gap_summary.csv", index=False)

    regularisation = regularisation_summary(metrics)
    with_labels(regularisation).to_csv(
        output_dir / "regularisation_sensitivity.csv", index=False)

    repeated = repeated_summary(pooled)
    with_labels(repeated).to_csv(
        output_dir / "repeated_cv_summary.csv", index=False)
    block_sizes = block_summary(pooled)
    with_labels(block_sizes).to_csv(
        output_dir / "block_size_sensitivity.csv", index=False)

    calibration_summary, calibration_curves = calibration_outputs(raw)
    with_labels(calibration_summary).to_csv(
        output_dir / "calibration_summary.csv", index=False)
    with_labels(calibration_curves).to_csv(
        output_dir / "calibration_curves.csv", index=False)
    confounds = confound_checks(raw)
    with_labels(confounds).to_csv(
        output_dir / "confound_checks.csv", index=False)
    if COLOCATED_EXAMPLES is not None and len(COLOCATED_EXAMPLES):
        with_labels(COLOCATED_EXAMPLES).to_csv(
            output_dir / "colocated_record_examples.csv", index=False)
        colocated_rows = [
            {"sampling_scheme": key[0], "analysis_type": key[1],
             "model": key[2], "n_colocated_extra_test_rows": int(value)}
            for key, value in sorted(COLOCATED_TEST_COUNTS.items())]
        with_labels(pd.DataFrame(colocated_rows)).to_csv(
            output_dir / "colocated_record_counts.csv", index=False)
        print("Co-located records retained (see colocated_record_counts.csv); "
              "examples in colocated_record_examples.csv. Records sharing a "
              "500 m pixel share their embedding values but not necessarily "
              "'ndvi', which is the point-level Sentinel-2 NDVI (see the NDVI "
              "note in GEE_02).")
    stratified = stratified_performance(raw)
    with_labels(stratified).to_csv(
        output_dir / "stratified_performance.csv", index=False)
    dropout = complete_case_dropout(raw)
    with_labels(dropout).to_csv(
        output_dir / "complete_case_dropout.csv", index=False)
    gates = matching_gate_check(input_dir, raw)
    with_labels(gates).to_csv(
        output_dir / "matching_gate_check.csv", index=False)
    test_train_distance_check(raw).to_csv(
        output_dir / "test_train_distance_check.csv", index=False)
    final_rows = final_training_row_check(input_dir, raw)
    final_rows.to_csv(output_dir / "final_vs_cv_training_rows.csv", index=False)
    print("\nFinal model rows, CV rows and probability mode (v13b12):")
    print(final_rows[["sampling_scheme", "passed", "problems"]].to_string(index=False))
    failed_checks = [f"{r.sampling_scheme}: {r.problems}"
                     for r in final_rows.itertuples() if not r.passed]
    verified = set(final_rows.get("presence_score_mode", pd.Series(dtype=str))
                   .dropna().astype(str))
    if len(verified) > 1:
        failed_checks.append(f"the two final models use different score modes {sorted(verified)}")
    cv_check = cv_manifest_check(input_dir, raw, verified)
    cv_check.to_csv(output_dir / "cv_scores_vs_manifest.csv", index=False)
    bad = cv_check[~cv_check["passed"].astype(bool)]
    print(f"\nCV scores vs row manifests: {len(cv_check) - len(bad)} of "
          f"{len(cv_check)} partitions passed (cv_scores_vs_manifest.csv)")
    failed_checks += [f"{r.sampling_scheme} {getattr(r, 'analysis_type', '')} "
                      f"{getattr(r, 'model', '')}: {r.problems}" for r in bad.itertuples()]
    if not stratified.empty:
        ordlo_rows = stratified[stratified["stratum_type"] == "in_ordlo"]
        if not ordlo_rows.empty:
            print("\nOut-of-fold AUC by pre-2022 occupation line (R3-2):")
            print(ordlo_rows[[
                "sampling_scheme", "stratum", "n_presence", "n_absence",
                "AUC", "Brier_Skill_Score",
            ]].to_string(index=False,
                         float_format=lambda value: f"{value:.4f}"))
    spatial_correlogram, spatial_range = residual_spatial_correlogram(raw)
    with_labels(spatial_correlogram).to_csv(
        output_dir / "residual_spatial_correlogram.csv", index=False)
    with_labels(spatial_range).to_csv(
        output_dir / "residual_spatial_range.csv", index=False)

    scoring_coverage_check(pooled, output_dir)
    results_stability_check(pooled, output_dir)
    main_comparison = pooled[
        pooled["analysis_type"].isin(["main_fixed", "random_cv"])
    ].copy()
    with_labels(main_comparison).to_csv(
        output_dir / "main_model_comparison.csv", index=False)
    transfer_penalty = spatial_transferability_penalty(pooled)
    with_labels(transfer_penalty).to_csv(
        output_dir / "spatial_transferability_penalty.csv", index=False)
    plot_calibration(calibration_curves, output_dir)
    plot_sensitivities(regularisation, block_sizes, pooled, output_dir)
    plot_residual_correlogram(spatial_correlogram, spatial_range, output_dir)
    interpretability_outputs(input_dir, output_dir)
    variable_importance_outputs(input_dir, output_dir)
    write_methods_note(output_dir, pd.DataFrame())

    print("=" * 78)
    print("SPATIAL VALIDATION OUTPUTS")
    print("=" * 78)
    print(f"Loaded {len(raw):,} unique raw-score rows from {input_dir}")
    if not main_comparison.empty:
        columns = [
            "analysis_type", "model", "sampling_scheme", "min_leaf",
            "variables_per_split", "n", "AUC",
            "Brier_Skill_Score", "Calibration_Intercept",
            "Calibration_Slope", "ECE_10bin",
        ]
        print("\nBlocked and random-fold pooled metrics:")
        print(main_comparison[columns].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"))
    if not gap_summary.empty:
        print("\nTrain-test gap summary:")
        gap_columns = [
            "analysis_type", "model", "sampling_scheme", "min_leaf",
            "variables_per_split",
            "AUC_train_minus_test_mean", "Brier_test_minus_train_mean",
        ]
        gap_columns = [column for column in gap_columns
                       if column in gap_summary.columns]
        print(gap_summary[gap_columns].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nAll outputs written to {output_dir}")
    marker = output_dir / MARKER_NAME
    write_verdict(output_dir, input_dir, final_rows, cv_check, failed_checks)
    if failed_checks:
        marker.write_text("\n".join(failed_checks) + "\n", encoding="utf-8")
        print("\n" + "!" * 74)
        print("ROW-IDENTITY / PROBABILITY-MODE CHECK FAILED "
              "(final_vs_cv_training_rows.csv, cv_scores_vs_manifest.csv):")
        print("\n".join(f"  - {x}" for x in failed_checks))
        print("The CV metrics above are diagnostic until this passes.")
        print("!" * 74)
        sys.exit(1)
    marker.unlink(missing_ok=True)
    print("Row-identity and probability-mode checks PASSED for both models and "
          f"every CV partition ({VERDICT_NAME}).")


if __name__ == "__main__":
    main()
