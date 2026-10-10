# Code for manuscript SUST-D-26-00527 — detailed notes

See `README.md` for the step-by-step run with the shipped synthetic point tables. In this package
`GEE_02b`, `GEE_06`, `support/Python_reclassify_cumulative_classes.py` and
`tests/` live in `optional_checks/`; `Python_A0_make_mock_training_points.py`
writes synthetic training points in the format of `Python_A1`. The point tables
uploaded to Earth Engine are named in `support/folders.py` (`TABLES`).

Code accompanying *"War-related anomalies in outdoor visitation propensity in
Eastern Ukraine: insights from an Earth observation foundation model and
social media"* (Sustainability Science).

The workflow has two homes. **Python** (this folder) prepares the training
points before Earth Engine and turns the Earth Engine exports into the tables
and figures of the manuscript. **Google Earth Engine** JavaScript (folder
`GEE/`) does the raster work: embedding composites, sampling of training
tables, cross-validation, final probability maps and change detection.
Python scripts need Python 3.10+ and `pip install -r requirements.txt`.

Keep the unzipped package together: `Python_01`, `Python_02` and `Python_03`
import `support\pipeline_checks.py`, so run them from the package folder
(PowerShell: `cd <package folder>` then `python Python_01_calculate_cv_metrics.py`;
Jupyter: `%run <package folder>\Python_01_calculate_cv_metrics.py`).
If you paste a script into a notebook cell or keep a copy elsewhere, set
`BUNDLE_DIR_OVERRIDE` near its top to the package folder.

All local folders live under one root (`~/visitation_propensity`, or the
environment variable `VP_ROOT`): inputs and working files in `data/`, every
other folder under `outputs/`. They, and the prefix of the Earth Engine assets,
are set in one file, `support/folders.py` (see "Folders and assets" below).
All random steps use fixed seeds (20250523). The package contains code only: the
social-media records cannot be shared (see the Ethics statement of the
manuscript), and the Earth Engine assets and output rasters are not included.
Every table exported by GEE_03, GEE_04 and GEE_05 carries `run_id` and
`code_version` (v13b12); each Earth Engine run also exports a run manifest
with its parameters and input assets. `Python_03_collect_manuscript_numbers.py`
checks that one model's tables all come from one run and writes a manifest
with the run identifiers and SHA-256 hashes of the code and of every output
file used for the manuscript. `tests/test_pipeline_checks.py` runs the
checks on small synthetic tables (`python tests/test_pipeline_checks.py`).

## Folders and assets

**Assets.** Every Earth Engine asset made by the pipeline is called
`<ASSET_ROOT>/SUSTD_<name>`: the uploaded point tables, the
embedding composites (GEE_01), the sampled tables (GEE_02) and the
probability and novelty maps (GEE_04). The prefix is `ASSET_PREFIX` at the
top of every GEE script and in `support/folders.py`. Earlier assets keep their
names and are never read, so nothing has to be deleted; for a further run,
change the prefix in all of these places. Inputs that are not made by the
pipeline keep their names (`ordlo`, `war_events`, the AOI).

**Files to upload.** Python_A1–A3 write the ZIP files for Earth Engine to one
folder each, named like their assets; upload every ZIP of a folder as a table
asset under its file name (Assets → New → Shape files).

**Folders.** Each Earth Engine script exports to one Google Drive folder.
Download it and put its files into the local folder of the same name in
`<root>/outputs` (if the zip creates an inner
folder of the same name, that is fine; if Drive splits a folder into several
zips, put all files into the one folder). The Python scripts read and write
these folders without further settings.

| Script | Google Drive folder | Local folder (`<root>/outputs/...`) |
|---|---|---|
| Python_A1 | — | `A1_upload_to_GEE` (6 ZIP files, written) |
| Python_A2 | — | `A2_upload_to_GEE` (2 ZIP files, written) |
| Python_A3 | — | `A3_upload_to_GEE` (2 ZIP files, written) |
| GEE_03 | `GEE03_cross_validation` | `GEE03_cross_validation` |
| GEE_04 | `GEE04_final_models` | `GEE04_final_models` |
| GEE_05, `CHANGE_SCHEME = 'primary_effort'` | `GEE05_regular` | `GEE05_regular` |
| GEE_05, `CHANGE_SCHEME = 'recurrent_effort'` | `GEE05_recurrent` | `GEE05_recurrent` |
| Python_01 | — | `Python01_cv_metrics` (written) |
| Python_02 | — | `Python02_hexagons_regular`, `Python02_hexagons_recurrent` (written; also subfolders in the GEE05 folders) |
| Python_03 | — | `Python03_manuscript_numbers` (written) |
| `support/Python_plot_supplementary_figures.py` | — | `Supplementary_figures` (written) |

GEE_01 and GEE_02 write Earth Engine assets only; GEE_06 (optional audit)
writes the Drive folder `GEE06_complete_case_audit`, which no script reads.

## Run order

| Step | Script | What it does (manuscript section) |
|---|---|---|
| 1 | `Python_A1_prepare_presences_and_backgrounds.py` | Merges the Flickr, Twitter and VKontakte records and pseudonymises user IDs; keeps records in six ESA WorldCover classes (10, 20, 30, 80, 90, 100), May–September 2015–2021, outside a 10 m buffer around roads, railways and buildings; thins to one random record per H3 resolution-8 cell (2019–2021 main layer and 2015–2021 sensitivity layer); labels recurrent cells; draws pseudo-absences by the same rules, sampled according to overall social-media activity (target-group background; `effort` in the code). Writes GeoPackages and zipped shapefiles for upload. (*Social media data*) |
| 2 | `Python_A2_match_recurrent_accessibility.py`, `Python_A3_match_nonrecurrent_accessibility.py` | Accessibility-matched pseudo-absences for the recurrent and the regular model: coarsened exact matching on log social-media activity (`log_effort`) and log distance to infrastructure, exact on the pre-2022 occupation line, with balance (SMD ≤ 0.10) and retention (≥ 70% / ≥ 65%) gates. Sensitivity analysis only. (*Accessibility diagnostics*) |
| 3 | `GEE/GEE_01_create_embedding_composites.js` | Materialises the 64-band annual AlphaEarth embeddings 2019–2024 on a fixed 500 m EPSG:3035 grid. (*Satellite data*) |
| 4 | `GEE/GEE_02_sample_all_training_tables.js` | Samples the pre-war (2019–2021 median) embeddings and the 11 baseline predictors at all training points; points whose 500 m cell is outside the six-class training domain are dropped. One table per sampling scheme. (*Modelling*) |
| 4b | `GEE/GEE_02b_check_ndvi_source.js` | One-off, read-only check of tables made before the NDVI-name fix (see below). |
| 5 | `GEE/GEE_03_cross_validation_and_sensitivities.js` | Five-fold spatially blocked CV of the embedding model, the environmental baseline and the NDVI-only model on identical complete-case rows; repeated block assignments, block-size, random-fold and Random Forest-setting sensitivities; fold audit. Matched schemes keep complete pairs only. Every score row carries `row_id` (feature ID of the sampled table) and a fingerprint of its 64 predictor values; `CV_RowManifest_<scheme>_v13b12` lists the rows of each scheme. |
| 6 | `GEE/GEE_04_final_probability_products.js` | Fits the final models on the CV complete-case rule and predicts every year 2019–2024 for the whole study area (no land-cover mask); extrapolation flags for the median and all six annual embeddings. Checks: the probability output mode (`SDM_ProbabilityModeCheck_*`), the training rows with `row_id` and fingerprint (`SDM_FullModelPointPredictions_*`), and a labelled reconstruction of the previous (v34) row filters (`SDM_TrainingRowCheck_*`, `SDM_TrainingRowSets_*`). (*Figure 5*) |
| 7 | `Python_01_calculate_cv_metrics.py` | Table 1 and the model-performance results: AUC, BSS, TSS, sensitivity, specificity, train–test gap; paired t-tests between models (with fold-by-fold differences; folds share training data, so p-values are indicative); NDVI and distance ranking checks; performance by occupation line; residual correlogram and block-size check; covariate balance and pair completeness of the matched schemes on the rows actually evaluated. Final checks: the final-model rows and the main blocked CV scores equal the CV row manifest by `row_id`, label, fingerprint and (scores) held-out fold, and the exported score is the verified label-1 probability (`final_vs_cv_training_rows.csv`); every CV partition of every scheme and model agrees with its manifest in labels, fingerprints, folds, run and score mode, with no row trained on and held out in one fold (`cv_scores_vs_manifest.csv`). Two different copies of one CV export stop the run. One overall verdict, `CV_CHECKS_VERDICT_v13b12.json`, covers all of these; it is removed when a run starts, and `CHECKS_FAILED.txt` stays until a run has passed. Exit status 1 if a check fails. |
| 8 | `GEE/GEE_05_debiased_change_detection.js` | Change detection against pre-war year-to-year variation: z-scores for the median and annual comparisons, pre-war areas (areas over which the pre-war changes exceed the threshold), the stricter version, the year-count map and table (sole producer), screens, SD-floor diagnostics, one sensitivity table per variant (`Sensitivity_*_v13b12`) and a run manifest. Run once per `CHANGE_SCHEME`. (*Change detection*; *Figures 6–7*) |
| 9 | `Python_02_run_change_postprocessing.py` | Checks the downloaded GEE_05 folders (one v13b12 run per model) and runs the four `support/` steps for both models. |
| 10 | `Python_03_collect_manuscript_numbers.py` | Checks every table (presence, columns, unique keys, finite values, model, one run) and that the main-parameter sensitivity table reproduces the headline areas for both signs (within one 500 m cell, 0.25 km², because GEE_05 sums the two with different masking of cells cut by the AOI edge); then collects every change-detection number with its status against manuscript v17 (`missing`, `new`, `changed`, `unchanged`). Writes `checks_v13b12.csv`, `manuscript_numbers_v13b12.csv` (only if every check passes; otherwise `..._FAILED.csv` and exit status 1) and `run_manifest_v13b12.json`. |
| — | `GEE/GEE_06_complete_case_audit.js` | Optional audit of rows removed for missing predictor values. |

### The four steps inside `Python_02`

1. `support/Python_annual_baseline_sensitivity.py` — compares the
   annual-median comparison with the composite-based alternative.
2. `support/Python_null_diagnostics.py` — distribution of the pre-war
   leave-one-out z-scores (mean, SD, tails, QQ plot), their spatial
   correlation (pairs within each pre-war year), ratios of the pre-war area to the detected area per
   comparison, threshold and direction (NA with status `no_detections` when
   nothing is detected), what the stricter rules do to the pre-war
   exceedances (area ratio, retained share, cells added by the closing and
   exceedances removed; measured on the pre-war maps by GEE_05), and the
   sensitivity summary (area, ratio to the main-parameter area and Jaccard
   overlap for every variant and statistic). These pre-war areas are references, not
   calibrated no-war expectations or false discovery rates. (The earlier
   Monte Carlo of the filter rules was removed in v13b11.)
3. `support/Python_correlations_with_events.py` — H3 resolution-6
   hexagon statistics and Spearman correlations with VIINA and ACLED events
   (comparison data for RQ3; the correlations describe co-occurrence with
   documented events and are not a validation of the detections)
   (spatially blocked bootstrap intervals with ranks recomputed in each
   replicate; ordinary Spearman p-values with Benjamini–Hochberg correction,
   which does not account for spatial dependence). Intensity = median and
   maximum of all positive (loss side) or negative (gain side) z-scores of
   the median comparison in the hexagon, whether or not they exceed the
   threshold; extent = cells mapped as loss or gain in the dominant-direction
   year-count map. Hexagons in which more than half of the cells carry the
   extrapolation flag (any annual embedding flagged) are excluded.
4. `support/Python_plot_figure8.py` — Figure 8 (corner labels mark
   coefficients whose spatially blocked interval excludes zero;
   `SIG_FROM` switches to BH p-values).

`support/Python_plot_supplementary_figures.py` draws Figures S5–S9 of the
Supplementary materials. Run it after Python_03, from the package folder; it
reads the two GEE_05 folders, the Python_01 metrics folder and Python_03's
`run_manifest_v13b12.json`. A model folder is used only if its
`GEE05_RunManifest_v13b12.csv` names that model and the GEE_05 run that
Python_03 accepted (folders from `support/folders.py`). Every file it
reads that Python_03 also recorded must be unchanged (SHA-256), and every table
must carry that run_id. Before drawing the maps it recomputes the mapped areas
from the GeoTIFFs and compares them with the GEE_05 area tables
(`raster_area_check.csv`: median and annual areas, year-count classes, loss
and gain unions and their overlap, and the stricter version, both signs),
which confirms that the files and band numbers match the tables of the
manuscript; Figure S8 maps the pixels whose embeddings were outside the range of the
training data in at least one year (GEE_05 main GeoTIFF band 25) with the median loss and
2022 gain anomalies, and the area check compares these areas with
`Areas_by_Screen_v13b12.csv`. The pre-2022 occupation line in the maps is
band 26 of the main GeoTIFF (asset `ordlo`, the same polygons as
`data/ordlo.gpkg`). GEE counts a cell on the AOI outline only by the part inside the
outline, whereas the GeoTIFF stores it whole, so raster areas can be slightly
larger than the tables; the analysed area is therefore checked against GEE's
cell count (`eligible_px`) and reads "ok (AOI edge cells)". A GeoTIFF that is
not in the folder (the annual files are needed only for Figure S7, regular
model) is reported as not checked.

`support/Python_reclassify_cumulative_classes.py` is an optional local check
of the GEE_05 year-count classes and unions (0.25 km² per cell, independent of
the GEE area table).

## Where the manuscript numbers come from

| Manuscript | Output |
|---|---|
| Data flow (1,052,929 → 10,765 cells; 2,949 recurrent cells) | `sampling_audit_*.csv` (Python_A1) |
| 7,667 / 7,381 records, Table 1 | GEE_03 fold audit; `metrics_summary.csv`, `pooled_out_of_fold_metrics.csv` (Python_01) |
| Paired t-tests | `model_comparison_paired_tests.csv` (Python_01) |
| Matched schemes: rows, pairs and balance | `matching_gate_check.csv` (Python_01) |
| NDVI (0.41) and distance (0.771) ranking checks | `confound_checks.csv` (Python_01) |
| Occupation-line performance | `stratified_performance.csv` (Python_01) |
| Pre-war z-scores: mean/SD/tails, spatial correlation | `null_diagnostics/*` (Python_02 step 2) |
| Median and annual loss/gain areas; pre-war areas; thresholds 2.5 / 3.0 | `Areas_Excess_by_Zclass.csv` (GEE_05) |
| Year counts: dominant-direction classes, unions, overlap | `Cumulative_Confidence_Class_Areas.csv` (GEE_05) |
| Stricter version with and without closing, cells added by the closing; after the three exclusion checks | `Areas_Headline_*.csv` (GEE_05) |
| Stricter rules applied to the pre-war changes | `Areas_Stricter_Null_v13b12.csv` (GEE_05); `null_diagnostics/stricter_version_prewar_summary.csv` |
| Exclusion checks (embeddings outside the range of the training data; variable wartime predictions; pre-war propensity near 0 or 1) and SD floor, per statistic; Figure S8 and the extrapolation rows of Table 2 | `Areas_by_Screen_v13b12.csv` (column `no_annual_extrapolation_km2`), `SD_Floor_Diagnostics_v13b12.csv` (GEE_05) |
| Sensitivity to the 50 / 200 km scale, a dropped pre-war year (two designs) and the SD floor, for the median, annual, year-count and stricter areas, with overlap | `Sensitivity_*_v13b12.csv` (GEE_05); `null_diagnostics/change_detection_sensitivity_summary.csv` |
| Threshold and patch-size sensitivity of the stricter version | `Areas_vs_Threshold_and_PatchSize.csv` (GEE_05) |
| Probability output mode; final-model rows = CV rows; CV scores = row manifests | `final_vs_cv_training_rows.csv`, `cv_scores_vs_manifest.csv` (Python_01) |
| All change-detection numbers in one table; checks; manifest | `manuscript_numbers_v13b12.csv`, `checks_v13b12.csv`, `run_manifest_v13b12.json` (Python_03) |
| Correlations, Figure 8 | `spearman_correlations.csv`, Figure 8 files (support steps 3–4) |
| Figures S5 and S6 | `residual_spatial_correlogram.csv`, `residual_spatial_range.csv` (Python_01); `ControlZ_Sample.csv` (GEE_05) |
| Figures S7–S9 | annual and main GeoTIFFs, `Cumulative_Confidence_Classes.tif` (GEE_05); plotted by `support/Python_plot_supplementary_figures.py` |

## Running the pipeline

PowerShell in the package folder.

1. `python Python_A1_prepare_presences_and_backgrounds.py` — uses
   `data/ordlo.gpkg` (pre-2022 occupation line) for `in_ordlo` and
   Figures S1–S3 (`ordlo_stats_*.pdf`). Upload the six ZIP files in
   `outputs/A1_upload_to_GEE`.
2. `python Python_A2_match_recurrent_accessibility.py` and
   `python Python_A3_match_nonrecurrent_accessibility.py` — upload the two ZIP
   files in `A2_upload_to_GEE` and the two in `A3_upload_to_GEE`.
3. GEE_01 (assets `SUSTD_EmbeddingAnnual_<year>_EPSG3035_500m_v13b`), then
   GEE_02 (assets `SUSTD_SDM_Sampled_all_<scheme>_outdoor_v13b`).
4. GEE_03 (155 score tasks, 5 fold audits, 5 row manifests, 1 run manifest)
   → `GEE03_cross_validation`. GEE_04 (`RUN_MODE 'all'` for both models; start
   the six image-to-asset tasks first) → assets `SUSTD_SDM_Probability_*` and
   `SUSTD_SDM_Novelty_*`, tables and GeoTIFFs → `GEE04_final_models`.
5. `python Python_01_calculate_cv_metrics.py` — must end with
   "Row-identity and probability-mode checks PASSED".
6. GEE_05 with `CHANGE_SCHEME = 'primary_effort'`, then
   `'recurrent_effort'` (start every task; the eight `Sensitivity_*` tasks one
   at a time) → `GEE05_regular`, `GEE05_recurrent`.
7. `python Python_02_run_change_postprocessing.py`.
8. `python Python_03_collect_manuscript_numbers.py` — must print "0 failed".
9. `python support\Python_plot_supplementary_figures.py` — every row of
   `raster_area_check.csv` must read ok.

## Notes on specific choices

**Random Forest settings.** All 64 embedding bands are predictors. The setting
`variablesPerSplit = 16` means that each split in each tree chooses among 16
randomly selected bands (the usual Random Forest "mtry"); this keeps the trees
diverse and is not a memory measure. It was fixed before validation (500 trees,
minimum leaf population 20); leaf sizes 5/20/50 and mtry 8/16/21 are reported
as a sensitivity only. The simpler models use mtry 3 (11 predictors) and 1
(NDVI only).

**NDVI.** The NDVI predictor of the environmental baseline and the NDVI-only
model, and the NDVI in the ranking check, is the summer-median Sentinel-2 NDVI
(May–September 2019–2021; `Summer_NDVI_Median_2019_2021.tif`) sampled by
Python_A1 at each record location — the same NDVI that defined the presence
filter of the submitted version. The other baseline predictors are 500 m
values. Earlier versions of GEE_02 also carried a MODIS 500 m band named
`ndvi`; `GEE_02b` confirmed that the point attribute is what the tables hold,
and the MODIS band is now named `ndvi_modis` (reference only, not a model
input).

**Training domain and prediction domain.** WorldCover limits the training
points only. Final probability and change maps cover every pixel with valid
embeddings, including cropland; such pixels are predictions outside the
training domain and are flagged by the extrapolation check where their
embeddings are unusual relative to the training data.

**Memory-related limits.** These do not affect the fitted models: one forest
per exported task, persistent sampled tables, and training AUC computed on a
random subsample of at most 3,000 records per class (train–test gap only).

**Historical names.** Machine-readable names keep earlier tokens for pipeline
continuity: `damage`/`rewild` = loss/gain anomaly; `confidence` = signed
dominant-direction year-count class; `ControlZ_*`/`controlz_*` = pre-war
leave-one-out z-scores (a temporal reference; no spatial control areas are
used); `prewar_benchmark_*` columns = pre-war areas; `Areas_Excess_by_Zclass` = detected areas with pre-war areas;
`quality_screen_*` = the three exclusion checks (the manuscript reports the extrapolation check on its own); `primary_effort` = regular-visitation
model; `effort`/`log_effort` = local social-media activity (all records within 5 km, 2019–2021); `window` = period.
