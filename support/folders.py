"""Folders, Earth Engine asset names and the names of the point tables: the
only place to change them. All folders are inside the package folder
(override with the environment variable VP_ROOT):

    data/                 shipped inputs (upload_to_GEE/SUSTD_aoi.zip, upload_to_GEE/SUSTD_ordlo.zip,
                          h3_hexagons_res6.gpkg, mock_points/) + the files that the user prepares
                          with Python_prepare_inputs.py (war events, WorldCover)
    work/                 working files of Python_A0/A1-A3 (not needed to run the pipeline)
    gee_exports/          one folder per Earth Engine script. Each run of a GEE script
                          exports to a Drive folder named <step>_<start time>; download
                          that folder and extract its files into the step folder:
        GEE00_summer_ndvi/        <- GEE_00 (authors' preprocessing)
        GEE03_cross_validation/   <- GEE_03
        GEE04_final_models/       <- GEE_04
        GEE05_regular/, GEE05_recurrent/   <- GEE_05, one run per model
        GEE06_complete_case_audit/         <- optional_checks/GEE_06
    outputs/              written by the Python steps:
        A1_upload_to_GEE/, A2_upload_to_GEE/, A3_upload_to_GEE/   Python_A0/A1-A3 (authors' use)
        Python01_cv_metrics/, Python02_hexagons_regular/, Python02_hexagons_recurrent/,
        Python03_manuscript_numbers/, Supplementary_figures/

    python support/folders.py     creates the step folders and lists them
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("VP_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
WORK = ROOT / "work"
GEE_EXPORTS = ROOT / "gee_exports"
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

GEE00 = GEE_EXPORTS / "GEE00_summer_ndvi"
GEE03 = GEE_EXPORTS / "GEE03_cross_validation"
GEE04 = GEE_EXPORTS / "GEE04_final_models"
GEE05 = {"primary_effort": GEE_EXPORTS / "GEE05_regular",
         "recurrent_effort": GEE_EXPORTS / "GEE05_recurrent"}
GEE06 = GEE_EXPORTS / "GEE06_complete_case_audit"
NDVI_FILENAME = "Summer_NDVI_Median_2019_2021.tif"

PYTHON01 = OUTPUTS / "Python01_cv_metrics"
HEXAGONS = {"primary_effort": OUTPUTS / "Python02_hexagons_regular",
            "recurrent_effort": OUTPUTS / "Python02_hexagons_recurrent"}
PYTHON03 = OUTPUTS / "Python03_manuscript_numbers"
FIGURES = OUTPUTS / "Supplementary_figures"

# Shipped inputs: study area and pre-2022 occupation area as zipped shapefiles
# in data/upload_to_GEE/ (ready to upload to Earth Engine), hexagon grid and
# mocked point tables in data/.
AOI = DATA / "upload_to_GEE" / "SUSTD_aoi.zip"
ORDLO = DATA / "upload_to_GEE" / "SUSTD_ordlo.zip"
HEXAGON_GRID = DATA / "h3_hexagons_res6.gpkg"
MOCK_POINTS = DATA / "mock_points"
# Not shipped. Built by Python_prepare_inputs.py from files that the user
# downloads (ACLED and VIINA war events, ESA WorldCover tiles).
WORLDCOVER = DATA / "ESA_WorldCover_2021_Ukraine_AOI.tif"
WAR_EVENTS = DATA / "war_events4326.gpkg"
WAR_EVENTS_ZIP = DATA / "upload_to_GEE" / "SUSTD_war_events.zip"

ALL_OUTPUT_FOLDERS = [GEE00, GEE03, GEE04, *GEE05.values(), GEE06, *UPLOAD.values(),
                      PYTHON01, *HEXAGONS.values(), PYTHON03, FIGURES]


def create_step_folders() -> None:
    for folder in ALL_OUTPUT_FOLDERS:
        folder.mkdir(parents=True, exist_ok=True)


def source_ndvi(work_dir: Path) -> Path:
    """The NDVI raster of GEE_00: from its step folder if it was downloaded
    there, otherwise from the given folder."""
    export = data_dir(GEE00) / NDVI_FILENAME
    return export if export.is_file() else Path(work_dir) / NDVI_FILENAME


def data_dir(folder: Path) -> Path:
    """The folder itself or, if the Drive zip put the files into one inner
    folder, that inner folder."""
    folder = Path(folder)
    if not folder.is_dir() or any(folder.glob("*.csv")) or any(folder.glob("*.tif")):
        return folder
    inner = [d for d in folder.iterdir() if d.is_dir()
             and (any(d.glob("*.csv")) or any(d.glob("*.tif")))]
    return inner[0] if len(inner) == 1 else folder


if __name__ == "__main__":
    create_step_folders()
    print("Extract each Earth Engine download into its step folder:")
    for step, folder in (
        ("GEE_00", GEE00), ("GEE_03", GEE03), ("GEE_04", GEE04),
        ("GEE_05 regular", GEE05["primary_effort"]),
        ("GEE_05 recurrent", GEE05["recurrent_effort"]), ("GEE_06", GEE06),
    ):
        print(f"  {step:16} {folder}")
    print(f"Python outputs: {OUTPUTS}")
