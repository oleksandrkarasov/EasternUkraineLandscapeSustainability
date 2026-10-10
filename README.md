# Code for "War-related anomalies in outdoor visitation propensity in Eastern Ukraine"
### Sustainability Science, manuscript SUST-D-26-00527

This package lets you run the complete analysis of the paper — Random Forest
models of outdoor visitation propensity on AlphaEarth satellite embeddings,
spatially blocked cross-validation, annual propensity maps for 2019–2024,
change detection against pre-war variability, and the comparison with
documented war events — on **synthetic training points** that stand in for the
social-media records, which cannot be shared (Ethics statement of the paper).
Every output file, table and band keeps the name used in the Supplementary
materials (Tables S3–S4), so you can follow the paper's SI while you run it.
Because the training points are synthetic, **the numbers you obtain are not
those of the paper**; the pipeline, its intermediate products and the shape of
every table and figure are.

The analysis alternates between Google Earth Engine (folder `GEE/`, run in the
Code Editor) and Python (run from this folder). Allow a working day for a full
run; the Earth Engine exports (steps 3–6 and 8) take most of it.

## What you need

- A Google Earth Engine account with a Cloud project you can write assets to,
  and about 2 GB of asset storage.
- Python 3.10 or newer: `pip install -r requirements.txt`.
- Two downloads that the licences do not let us ship (step 1): an ACLED data
  export for Ukraine 2022–2024 (https://acleddata.com, free registration) and
  the VIINA event file (https://github.com/zhukovyuri/VIINA), plus the four
  ESA WorldCover 2021 tiles covering the area (https://esa-worldcover.org/en/data-access;
  N45E036, N45E039, N48E036, N48E039).

Everything else is in `data/` (see `data/README.md`): the study-area polygon
`aoi.gpkg`, the pre-2022 occupation area `ordlo.gpkg`, the hexagon grid
`h3_hexagons_res6.gpkg`, and the ten point tables in `data/mock_points/`.

## Step by step

**0. Check the package.** `python setup_project.py` — creates the `outputs/`
folders and lists which inputs are present. Then, in every file of `GEE/`,
replace `projects/YOUR-PROJECT/assets` in the configuration block at the top
with your own project (the same value sits in `support/folders.py`).

**1. Prepare the two downloaded inputs.**
`python Python_prepare_inputs.py --acled <acled.csv> --viina <events_latest.csv> --worldcover <tile1.tif> <tile2.tif> ...`
writes `data/war_events4326.gpkg` (events 2022–2024 inside the study area, with
`source` = ACLED / VIINA and `date`) and `data/ESA_WorldCover_2021_Ukraine_AOI.tif`.

**2. Upload 13 table assets to Earth Engine** (Assets → New → Shape files),
each with the asset name equal to the file name without `.zip`/`.gpkg`:

| File | Asset name | Read by |
|---|---|---|
| `data/aoi.gpkg` | `SUSTD_aoi` | all GEE scripts |
| `data/ordlo.gpkg` | `SUSTD_ordlo` | GEE_05 |
| `data/war_events4326.gpkg` (from step 1) | `SUSTD_war_events` | GEE_05 |
| `data/mock_points/SUSTD_points_regular_presence.zip`, `..._pseudoabsence.zip` | same as file | GEE_02 (main model) |
| `..._points_full_presence/pseudoabsence.zip` | same | GEE_02 (2015–2021 sensitivity) |
| `..._points_recurrent_presence/pseudoabsence.zip` | same | GEE_02 (recurrent-visitation model) |
| `..._points_regular_matched_presence/pseudoabsence.zip` | same | GEE_02 (accessibility-matched) |
| `..._points_recurrent_matched_presence/pseudoabsence.zip` | same | GEE_02 (accessibility-matched, recurrent) |

(GeoPackages can be uploaded after conversion to zipped shapefiles, e.g. with
QGIS or `ogr2ogr -f "ESRI Shapefile"`.)

**3. `GEE/GEE_01_create_embedding_composites.js`** — six tasks; writes the
annual AlphaEarth embeddings 2019–2024 on the 500 m EPSG:3035 grid as assets
`SUSTD_EmbeddingAnnual_<year>_EPSG3035_500m_v13b`. Wait until all six finish.

**4. `GEE/GEE_02_sample_all_training_tables.js`** — five tasks; samples the
64 embedding bands and the 11 baseline predictors at the uploaded points and
writes one table asset per model (`SUSTD_SDM_Sampled_all_<scheme>_outdoor_v13b`).

**5. `GEE/GEE_03_cross_validation_and_sensitivities.js`** — about 40 tasks;
out-of-fold scores of the embedding, environmental-baseline and NDVI-only
models for all schemes and settings → Drive folder `GEE03_cross_validation`.
Download it into `outputs/GEE03_cross_validation/`.

**6. `GEE/GEE_04_final_probability_products.js`** — final models; annual and
period propensity maps (Figure 5; Table S3 files), extrapolation flags,
variable importance (Table S9), land-cover sample (Table S8) → Drive folder
`GEE04_final_models`; also assets `SUSTD_SDM_Probability_*`, `SUSTD_SDM_Novelty_*`
read by GEE_05. Download the folder into `outputs/GEE04_final_models/`.

**7. `python Python_01_calculate_cv_metrics.py`** — Table 2; Table S2a–d;
Table S9; Figure S5; the matching and co-location counts of Table S1
→ `outputs/Python01_cv_metrics/`.

**8. `GEE/GEE_05_debiased_change_detection.js`, twice** — set
`CHANGE_SCHEME = 'primary_effort'` (regular visitation; Drive folder
`GEE05_regular`), run, then `'recurrent_effort'` (`GEE05_recurrent`). Produces
the z-score and year-count rasters (Figures 6–7, S7, S9; Table S4 files), all
areas (Table S6), the sensitivity analyses (Table S5), the embedding screen
(Figure S8) and the pre-war z-score sample (Figure S6). Download both folders
into `outputs/GEE05_regular/` and `outputs/GEE05_recurrent/`.

**9. `python Python_02_run_change_postprocessing.py`** — runs the four
`support/` steps: baseline-representation sensitivity (Table S5d), pre-war
z-score diagnostics (Figure S6; Text S2), hexagon statistics and Spearman
correlations with the war events (Table S7, with the spatial block bootstrap)
and Figure 8 → `outputs/Python02_hexagons_regular/`, `.../_recurrent/`.

**10. `python Python_03_collect_manuscript_numbers.py`** — every
change-detection number of the Results and SI in one table
(`manuscript_numbers_*.csv`) with consistency checks → `outputs/Python03_manuscript_numbers/`.

**11. `python support/Python_plot_supplementary_figures.py`** — Figures S5–S9
and `raster_area_check.csv` (areas recomputed from the GeoTIFFs against the
tables) → `outputs/Supplementary_figures/`.

Figures 1–4 of the paper are drawn by hand; the map layouts of Figures 2 and
5–7 were composed in a GIS from the GeoTIFFs of steps 6 and 8 (Figure 2 uses
the Dynamic World land-cover product, loaded directly in the GIS).

## How the training points were made (not needed to run the pipeline)

`Python_A1_prepare_presences_and_backgrounds.py` turned the Flickr, Twitter and
VKontakte records into presences and pseudo-absences (Methods "Social media
data"; Table S1; Figures S1–S3); `Python_A2`/`A3` built the accessibility-matched
samples (Text S1). They document the preprocessing but cannot be run without
the records. `Python_A0_make_mock_training_points.py` produced the synthetic
tables in `data/mock_points/`: locations drawn from a propensity surface built
only from public covariates (land-cover class, distance to infrastructure) and
a random spatial field, then processed with the same thinning, pseudo-absence
and matching rules as the real data; no post, user or date enters them.

`optional_checks/` contains scripts used to verify the pipeline that produce
nothing shown in the paper; `README_detailed.md` documents every output file
and check. Random steps use seed 20250523.
