# -*- coding: utf-8 -*-
"""Step 0: check the package and create the output folders.

    python setup_project.py

Lists which input files are present in data/, which still have to be prepared
(Python_prepare_inputs.py) and which Earth Engine uploads the next steps expect.
"""
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if not (_PKG / "support" / "folders.py").exists():
    raise SystemExit("Run this script from the package folder (support/folders.py not found).")
sys.modules.pop("folders", None)
sys.path.insert(0, str(_PKG / "support"))
import folders  # noqa: E402

SHIPPED = ["SUSTD_aoi.zip", "SUSTD_ordlo.zip", "h3_hexagons_res6.gpkg"]
PREPARED = ["ESA_WorldCover_2021_Ukraine_AOI.tif", "war_events4326.gpkg"]
TABLES = [folders.ASSET_PREFIX + n + ".zip" for n in folders.TABLES.values()]


def main():
    for d in [folders.WORK, *folders.ALL_OUTPUT_FOLDERS]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Package folder: {folders.ROOT}\n")
    print("Shipped inputs (data/):")
    for n in SHIPPED:
        print(f"  [{'x' if (folders.DATA / n).exists() else ' '}] {n}")
    print("Inputs to prepare with Python_prepare_inputs.py (data/):")
    for n in PREPARED:
        print(f"  [{'x' if (folders.DATA / n).exists() else ' '}] {n}")
    print(f"  [{'x' if folders.WAR_EVENTS_ZIP.exists() else ' '}] upload_to_GEE/{folders.WAR_EVENTS_ZIP.name}")
    print("Point tables to upload to Earth Engine (data/mock_points/):")
    for n in TABLES:
        print(f"  [{'x' if (folders.MOCK_POINTS / n).exists() else ' '}] {n}")
    print("\nEarth Engine assets expected by the GEE scripts (asset name = file name without .zip):")
    print("  " + ", ".join(folders.ASSET_PREFIX + n for n in ("aoi", "ordlo", "war_events"))
          + " and the ten tables above (all zipped shapefiles).")
    print(f"\nOutput folders created under {folders.OUTPUTS}. Next: README.md, step 2.")


if __name__ == "__main__":
    main()
