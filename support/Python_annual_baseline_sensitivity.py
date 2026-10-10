#!/usr/bin/env python3
"""Summarise the two annual-baseline representations exported by GEE_05.

The Earth Engine table contains both modes from the same run:

* annual_median: median(prob_2019, prob_2020, prob_2021) versus each annual
  wartime probability; this is the primary representation-consistent mode;
* composite: the three-year composite probability versus each annual wartime
  probability; this reproduces the earlier mixed-representation calculation.

Outputs report absolute and relative changes in true ground area and in exact
500-m output-grid cell counts. A 10% flag is supplied as a transparent
screening rule, not as an automatic scientific conclusion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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


# One GEE_05 folder per visitation model (support/folders.py).
ANALYSIS_SCHEMES = ["recurrent_effort", "primary_effort"]
SCHEME_LABELS = {
    "recurrent_effort":
        "Recurrent outdoor visitation model (2019-2021) [comparison]",
    "primary_effort": "Regular outdoor visitation model (2019-2021) [main]",
}
INPUT_DIR: Path = None   # set per scheme in main()
OUTPUT_DIR: Path = None
AREA_INPUT: Path = None
CONFIDENCE_INPUT: Path = None
RELATIVE_SCREENING_THRESHOLD = 0.10


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")


def comparison_table(
    frame: pd.DataFrame,
    index_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for value in value_columns:
        wide = frame.pivot_table(
            index=index_columns,
            columns="annual_baseline_mode",
            values=value,
            aggfunc="first",
        ).reset_index()
        require_columns(wide, {"annual_median", "composite"}, Path(value))
        denominator = wide["composite"].abs()
        relative = np.where(
            denominator > 0,
            (wide["annual_median"] - wide["composite"]) / denominator,
            np.nan,
        )
        part = wide[index_columns].copy()
        part["metric"] = value
        part["annual_median"] = wide["annual_median"]
        part["composite"] = wide["composite"]
        part["annual_median_minus_composite"] = (
            wide["annual_median"] - wide["composite"]
        )
        part["relative_change_vs_composite"] = relative
        part["absolute_relative_change_ge_10pct"] = (
            np.abs(relative) >= RELATIVE_SCREENING_THRESHOLD
        ).astype(int)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def run_scheme(scheme_label: str) -> None:
    if not AREA_INPUT.exists():
        raise FileNotFoundError(f"Missing Earth Engine export: {AREA_INPUT}")
    if not CONFIDENCE_INPUT.exists():
        raise FileNotFoundError(
            f"Missing Earth Engine export: {CONFIDENCE_INPUT}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    areas = pd.read_csv(AREA_INPUT)
    require_columns(
        areas,
        {
            "annual_baseline_mode", "row_type", "year_or_period",
            "damage_km2", "rewild_km2", "damage_px", "rewild_px",
        },
        AREA_INPUT,
    )
    for column in ["damage_km2", "rewild_km2", "damage_px", "rewild_px"]:
        areas[column] = pd.to_numeric(areas[column], errors="coerce")
    area_comparison = comparison_table(
        areas,
        ["row_type", "year_or_period"],
        ["damage_km2", "rewild_km2", "damage_px", "rewild_px"],
    )
    area_comparison.to_csv(
        OUTPUT_DIR / "annual_baseline_area_comparison.csv", index=False)

    confidence = pd.read_csv(CONFIDENCE_INPUT)
    require_columns(
        confidence,
        {"annual_baseline_mode", "confidence_class", "area_km2"},
        CONFIDENCE_INPUT,
    )
    confidence["confidence_class"] = pd.to_numeric(
        confidence["confidence_class"], errors="coerce")
    confidence["area_km2"] = pd.to_numeric(
        confidence["area_km2"], errors="coerce")
    confidence_comparison = comparison_table(
        confidence,
        ["confidence_class"],
        ["area_km2"],
    )
    confidence_comparison.to_csv(
        OUTPUT_DIR / "annual_baseline_confidence_comparison.csv", index=False)

    area_count_max = float(
        area_comparison["relative_change_vs_composite"].abs().max())
    confidence_nonzero = confidence_comparison[
        confidence_comparison["confidence_class"] != 0
    ]
    confidence_max = float(
        confidence_nonzero["relative_change_vs_composite"].abs().max())
    flagged_area_rows = int(
        area_comparison["absolute_relative_change_ge_10pct"].sum())
    flagged_confidence_rows = int(
        confidence_nonzero["absolute_relative_change_ge_10pct"].sum())

    report = f"""ANNUAL BASELINE REPRESENTATION SENSITIVITY
Analysis: {scheme_label}

Primary mode: annual_median
Comparator: composite

Maximum absolute relative change in annual/persistence area or count: {area_count_max:.1%}
Area/count rows changing by at least 10%: {flagged_area_rows} of {len(area_comparison)}

Maximum absolute relative change among non-zero confidence classes: {confidence_max:.1%}
Non-zero confidence rows changing by at least 10%: {flagged_confidence_rows} of {len(confidence_nonzero)}

Interpretation:
The 10% column is a prespecified screening threshold, not a significance test.
Report the actual annual and persistence-area differences; the pixel-count
rows are an independent arithmetic check and must not be converted to area by
multiplying by 0.25 km2. If several annual, persistence or non-zero confidence
classes cross 10%, describe the submitted
annual persistence/confidence result as sensitive to representation mismatch
and use annual_median for the corrected analysis. If differences are small,
state that the sensitivity test did not materially change the mapped areas.
"""
    (OUTPUT_DIR / "ANNUAL_BASELINE_SENSITIVITY_NOTE.txt").write_text(
        report, encoding="utf-8")
    print(report)
    print(f"Outputs written to {OUTPUT_DIR}")


def main() -> None:
    global INPUT_DIR, OUTPUT_DIR, AREA_INPUT, CONFIDENCE_INPUT
    ran = 0
    for scheme in ANALYSIS_SCHEMES:
        folder, label = folders.data_dir(folders.GEE05[scheme]), SCHEME_LABELS[scheme]
        if not folder.is_dir():
            print(f"[skip] {label}: folder not found ({folder})")
            continue
        INPUT_DIR = folder
        OUTPUT_DIR = folder / "annual_baseline_sensitivity"
        AREA_INPUT = folder / "AnnualBaselineSensitivity_Areas.csv"
        CONFIDENCE_INPUT = folder / "AnnualBaselineSensitivity_Confidence.csv"
        print("=" * 70)
        print(f"ANNUAL BASELINE SENSITIVITY -- {label}")
        print("=" * 70)
        run_scheme(label)
        ran += 1
    if ran == 0:
        raise FileNotFoundError("No GEE_05 output folder found.")


if __name__ == "__main__":
    main()
