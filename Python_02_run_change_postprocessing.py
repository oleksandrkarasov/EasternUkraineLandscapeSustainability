#!/usr/bin/env python3
"""Run all four post-change local analyses in their required order.

HOW THE FOLDER LOGIC WORKS
--------------------------
GEE_05 is executed once per visitation model and writes one Drive folder per
model. Download them into (support/folders.py):

    outputs/GEE05_regular/
    outputs/GEE05_recurrent/

Inputs are found by FIXED file names inside each folder; outputs go to:

    GEE05_<model>\\annual_baseline_sensitivity\\       (step 1)
    GEE05_<model>\\null_diagnostics\\                  (step 2)
    outputs/Python02_hexagons_<model>/         (steps 3-4)

So there is nothing to configure per run: one `python
Python_02_run_change_postprocessing.py` processes BOTH models. The preflight
below only makes that logic visible and fails early with a readable message
instead of deep inside a support script.

The preflight also aborts on stale Google Drive duplicates: if GEE_05 was
re-run into the same Drive folder, Drive keeps the old export beside the new
one and the download contains e.g. "ControlZ_Sample (1).csv". The support
scripts read the un-numbered name, which would silently mix pipeline
versions. Delete the old Drive folders (or download into fresh local
folders) and re-download, then run again.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pandas as pd

try:
    _SUPPORT = Path(__file__).resolve().parent / "support"
except NameError:
    _SUPPORT = Path.cwd() / "support"
sys.path.insert(0, str(_SUPPORT))
# A console that already ran another copy of the package keeps its support
# modules in memory; drop them so that this package's support folder is used.
for _m in ("folders", "pipeline_checks"):
    sys.modules.pop(_m, None)
import folders  # noqa: E402  (local folders: support/folders.py)

# Where the bundle (and its support\ subfolder) lives. When this code runs as
# a file, that is the file's own folder. In a Jupyter/IPython cell __file__
# does not exist, so fall back to BUNDLE_DIR_OVERRIDE if set, else the
# notebook's working directory — and verify support\ is actually there.
BUNDLE_DIR_OVERRIDE = None  # set to the package folder when running from a notebook

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = (Path(BUNDLE_DIR_OVERRIDE) if BUNDLE_DIR_OVERRIDE
                  else Path.cwd())

if not (SCRIPT_DIR / "support").is_dir():
    raise FileNotFoundError(
        f"No 'support' subfolder in {SCRIPT_DIR}. Running in a notebook? "
        "Set BUNDLE_DIR_OVERRIDE at the top of this script to the unzipped "
        "bundle folder (the one containing support\\), or run from a "
        "terminal: cd <bundle folder> && python "
        "Python_02_run_change_postprocessing.py")

STEPS = [
    SCRIPT_DIR / "support" / "Python_annual_baseline_sensitivity.py",
    SCRIPT_DIR / "support" / "Python_null_diagnostics.py",
    SCRIPT_DIR / "support" / "Python_correlations_with_events.py",
    SCRIPT_DIR / "support" / "Python_plot_figure8.py",
]

ANALYSIS_SCHEMES = ["recurrent_effort", "primary_effort"]
SCHEME_LABELS = {
    "recurrent_effort":
        "Recurrent outdoor visitation model (2019-2021) [comparison]",
    "primary_effort":
        "Regular outdoor visitation model (2019-2021) [main]",
}

# Files each scheme folder must contain before post-processing can run.
REQUIRED_PER_SCHEME = [
    "Debiased_RF_Main_revision_annualbase_annual_median.tif",
    "Cumulative_Confidence_Classes.tif",
    "Cumulative_Confidence_Class_Areas.csv",
    "Areas_Excess_by_Zclass.csv",
    "Areas_Stricter_Null_v13b12.csv",
    "GEE05_RunManifest_v13b12.csv",
    "Sensitivity_main_main_v13b12.csv",
    "ControlZ_Sample.csv",
    "Areas_vs_Threshold_and_PatchSize.csv",
    "AnnualBaselineSensitivity_Areas.csv",
    "AnnualBaselineSensitivity_Confidence.csv",
    "Areas_Headline_annualbase_annual_median.csv",
]

# Extra standalone inputs used by step 3 (hex correlations).
EXTRA_INPUTS = [folders.WAR_EVENTS, folders.HEXAGON_GRID, folders.WORLDCOVER]


def preflight() -> None:
    print("PREFLIGHT — folder discovery")
    print(f"  Output folder : {folders.OUTPUTS}")
    problems: list[str] = []
    found_any = False

    for scheme in ANALYSIS_SCHEMES:
        folder = folders.data_dir(folders.GEE05[scheme])
        label = SCHEME_LABELS[scheme]
        if not folder.is_dir():
            print(f"  [absent] {label}\n           ({folder}) — this scheme "
                  "will be SKIPPED by every step")
            continue
        found_any = True
        print(f"  [found ] {label}\n           ({folder})")

        for name in REQUIRED_PER_SCHEME:
            alternatives = name if isinstance(name, tuple) else (name,)
            if not any((folder / alt).exists() for alt in alternatives):
                problems.append(
                    f"{folder.name}: missing {' or '.join(alternatives)}")

        run_ids = set()
        for name in [n for n in REQUIRED_PER_SCHEME if n.endswith(".csv")]:
            path = folder / name
            if not path.exists():
                continue
            head = pd.read_csv(path, usecols=lambda c: c in ("code_version", "run_id"))
            if "code_version" not in head.columns or \
                    not (head["code_version"] == "v13b12").all():
                problems.append(
                    f"{folder.name}: {name} is not a GEE_05 v13b12 export "
                    "-- rerun GEE_05 v13b12 and download its outputs.")
            if "run_id" in head.columns:
                run_ids.update(head["run_id"].astype(str).unique())
        if len(run_ids) > 1:
            problems.append(
                f"{folder.name}: tables from {len(run_ids)} different GEE_05 runs "
                f"({', '.join(sorted(run_ids))}) -- download one complete run.")

        # Stale Drive duplicates: 'name (1).csv' etc. beside the read name.
        dupes = sorted(p.name for p in folder.glob("* (*.???")) + sorted(
            p.name for p in folder.glob("* (*).*"))
        dupes = sorted(set(dupes))
        if dupes:
            problems.append(
                f"{folder.name}: Google Drive duplicate exports present "
                f"({', '.join(dupes[:4])}{' ...' if len(dupes) > 4 else ''}) "
                "— the scripts read the un-numbered file and cannot tell "
                "which pipeline version it is. Clean the Drive folder and "
                "re-download into a fresh local folder.")

    if not found_any:
        problems.append(
            "No scheme folder found. Expected "
            + " and/or ".join(str(folders.GEE05[s]) for s in ANALYSIS_SCHEMES))

    for extra in EXTRA_INPUTS:
        if not extra.exists():
            print(f"  [warn  ] step-3 input not found: {extra}")
            print("           (hex correlations / Figure 8 will fail; "
                  "steps 1-2 still run)")

    if problems:
        print("\nPREFLIGHT FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("Preflight OK.\n")


def main() -> None:
    preflight()
    for index, script in enumerate(STEPS, start=1):
        if not script.exists():
            raise FileNotFoundError(f"Missing support script: {script}")
        print("=" * 78)
        print(f"POST-PROCESSING STEP {index}/{len(STEPS)}: {script.name}")
        print("=" * 78)
        runpy.run_path(str(script), run_name="__main__")
    print("All change-detection post-processing steps completed.")


if __name__ == "__main__":
    main()
