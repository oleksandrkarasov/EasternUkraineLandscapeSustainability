"""Folders, Earth Engine asset names and the names of the point tables: the
only place to change them. All folders are inside the package folder
(override with the environment variable VP_ROOT):

    data/                 shipped inputs (SUSTD_aoi.zip, SUSTD_ordlo.zip, h3_hexagons_res6.gpkg,
                          mock_points/) + the files prepared by Python_prepare_inputs.py
    work/                 working files of Python_A0/A1-A3 (not needed to run the pipeline)
    outputs/
        A1_upload_to_GEE/, A2_upload_to_GEE/, A3_upload_to_GEE/   Python_A0/A1-A3 (authors' use)
        GEE03_cross_validation/   <- download the Drive folder of GEE_03 here
        GEE04_final_models/       <- Drive folder of GEE_04
        GEE05_regular/, GEE05_recurrent/   <- the two Drive folders of GEE_05
        Python01_cv_metrics/, Python02_hexagons_regular/, Python02_hexagons_recurrent/,
        Python03_manuscript_numbers/, Supplementary_figures/     written by the Python steps
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("VP_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
WORK = ROOT / "work"
OUTPUTS = ROOT / "outputs"
DOWNLOADS = WORK  # older name used by some scripts

# Earth Engine: every asset of the pipeline is ASSET_ROOT/ASSET_PREFIX<name>;
# the same two values appear at the top of each GEE script.
ASSET_ROOT = "projects/YOUR-PROJECT/assets"   # your Earth Engine project
ASSET_PREFIX = "SUSTD_"

# Point tables uploaded to Earth Engine (ZIP name = asset name = ASSET_PREFIX + value).
# Written by Python_A1 or Python_A0 (first six) and Python_A2/A3 (last four); read by GEE_02.
TABLES = {
    "presences_regular":            "points_regular_presence",
    "pa_regular":                   "points_regular_pseudoabsence",
    "presences_full":               "points_full_presence",          # 2015-2021 sensitivity
    "pa_full":                      "points_full_pseudoabsence",
    "presences_recurrent":          "points_recurrent_presence",
    "pa_recurrent":                 "points_recurrent_pseudoabsence",
    "presences_regular_matched":    "points_regular_matched_presence",      # accessibility-matched
    "pa_regular_matched":           "points_regular_matched_pseudoabsence",
    "presences_recurrent_matched":  "points_recurrent_matched_presence",
    "pa_recurrent_matched":         "points_recurrent_matched_pseudoabsence",
}

UPLOAD = {"A1": OUTPUTS / "A1_upload_to_GEE",
          "A2": OUTPUTS / "A2_upload_to_GEE",
          "A3": OUTPUTS / "A3_upload_to_GEE"}

GEE03 = OUTPUTS / "GEE03_cross_validation"
GEE04 = OUTPUTS / "GEE04_final_models"
GEE05 = {"primary_effort": OUTPUTS / "GEE05_regular",
         "recurrent_effort": OUTPUTS / "GEE05_recurrent"}

PYTHON01 = OUTPUTS / "Python01_cv_metrics"
HEXAGONS = {"primary_effort": OUTPUTS / "Python02_hexagons_regular",
            "recurrent_effort": OUTPUTS / "Python02_hexagons_recurrent"}
PYTHON03 = OUTPUTS / "Python03_manuscript_numbers"
FIGURES = OUTPUTS / "Supplementary_figures"

# Shipped inputs (data/): study area and pre-2022 occupation area as zipped
# shapefiles (ready to upload to Earth Engine), hexagon grid, mocked point tables.
AOI = DATA / "SUSTD_aoi.zip"
ORDLO = DATA / "SUSTD_ordlo.zip"
HEXAGON_GRID = DATA / "h3_hexagons_res6.gpkg"
MOCK_POINTS = DATA / "mock_points"
# Built by Python_prepare_inputs.py from files the user downloads.
WORLDCOVER = DATA / "ESA_WorldCover_2021_Ukraine_AOI.tif"
WAR_EVENTS = DATA / "war_events4326.gpkg"
WAR_EVENTS_ZIP = DATA / "upload_to_GEE" / "SUSTD_war_events.zip"

ALL_OUTPUT_FOLDERS = [*UPLOAD.values(), GEE03, GEE04, *GEE05.values(), PYTHON01,
                      *HEXAGONS.values(), PYTHON03, FIGURES]


def data_dir(folder: Path) -> Path:
    """The folder itself or, if the Drive zip put the files into one inner
    folder, that inner folder."""
    folder = Path(folder)
    if not folder.is_dir() or any(folder.glob("*.csv")) or any(folder.glob("*.tif")):
        return folder
    inner = [d for d in folder.iterdir() if d.is_dir()
             and (any(d.glob("*.csv")) or any(d.glob("*.tif")))]
    return inner[0] if len(inner) == 1 else folder
