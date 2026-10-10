// CHANGE DETECTION. RUN AFTER GEE_04, ONCE PER VISITATION MODEL.
/**
 * Compares the annual visitation-propensity predictions of GEE_04 with the
 * year-to-year variability of the three pre-war years: pre-war contrasts
 * (each year against the mean of the other two) are smoothed and summarised
 * as a geographically weighted standard deviation, wartime differences are
 * expressed as z-scores against it, and loss/gain anomalies, year counts,
 * the stricter version, the embedding screen and the sensitivity analyses of
 * Tables S5-S6 are exported (Figures 6-7, S7-S9; Table S4 files).
 * File and band names keep the terms damage (= loss anomaly), rewild (= gain
 * anomaly) and confidence (= number of years with an anomaly).
 */

// ---- Configuration (identical in every GEE script) ------------------------
// Replace ASSET_ROOT with your own Earth Engine project. Every asset of the
// pipeline is named <ASSET_ROOT>/<ASSET_PREFIX><name>: the point tables
// uploaded from outputs/A1-A3_upload_to_GEE, the study area ('aoi'), the
// pre-2022 occupation area ('ordlo') and the war events ('war_events')
// uploaded from data/, and everything written by GEE_01, GEE_02 and GEE_04.
var ASSET_ROOT = 'projects/YOUR-PROJECT/assets';
var ASSET_PREFIX = 'SUSTD_';
var OWN = ASSET_ROOT + '/' + ASSET_PREFIX;
var AOI_ASSET = OWN + 'aoi';

// Run this script twice, once per visitation model:
//   'primary_effort'   -> regular outdoor visitation (Drive folder GEE05_regular)
//   'recurrent_effort' -> recurrent outdoor visitation (GEE05_recurrent)
var CHANGE_SCHEME = 'primary_effort';
var CHANGE_SCHEME_LABELS = {
  recurrent_effort:
    'Recurrent outdoor visitation model (2019-2021) [comparison]',
  primary_effort: 'Regular outdoor visitation model (2019-2021) [main]'
};
if (!CHANGE_SCHEME_LABELS[CHANGE_SCHEME]) {
  throw new Error('CHANGE_SCHEME must be recurrent_effort or primary_effort');
}
var CHANGE_SCHEME_LABEL = CHANGE_SCHEME_LABELS[CHANGE_SCHEME];

var refScale         = 500;
var coarseScale      = 5000;
var EXPORT_CRS       = 'EPSG:3035';
var EXPORT_TRANSFORM = [refScale, 0, 0, 0, -refScale, 0];
var SMOOTH_SIGMA_M   = 1500;
var BIAS_SIGMA_M     = 100000;  // main analysis, 100 km
var Z_THRESH         = 3.0;
var Z_NEIGHBOR       = 2.0;
var MIN_PATCH_PX     = 8;
var MAX_WAR_SD       = 0.10;
var PROB_LO          = 0.05;
var PROB_HI          = 0.95;
var ATTRIBUTION_DIST_KM = 30;
var DEN_CLAMP_FRAC   = 0.10;
var MIN_LOCAL_SD     = 0.02;  // floor of the unadjusted reference SD
var APPLY_CLOSING    = true;  // one-pixel closing in the stricter version
var MIN_LOCAL_SD_LIST = [0.01, 0.04];  // SD-floor sensitivity

// Var(mean-of-two-others minus year) = 1.5 sigma^2 (the LOO pseudo-diff that
// localSD estimates); Var(median of 3 iid normals) = (1 - sqrt(3)/pi)
// factors match the four-decimal values in the text).
var LOO_VARIANCE_FACTOR      = 1.5;
var MEDIAN3_VARIANCE_FACTOR  = 1 - Math.sqrt(3) / Math.PI;
var MEDIAN_NULL_SD_FACTOR = Math.sqrt(
  2 * MEDIAN3_VARIANCE_FACTOR / LOO_VARIANCE_FACTOR);  // 0.7735
var PERYEAR_NULL_SD_FACTOR = Math.sqrt(
  (1 + MEDIAN3_VARIANCE_FACTOR) / LOO_VARIANCE_FACTOR);  // 0.9827
// leave-one-out change (mean of the other n-1 years minus the year) has
// variance (1 + 1/(n-1)) sigma^2: 1.5 for three years, 2 for two years.
function looVarianceFactor(nYears) { return 1 + 1 / (nYears - 1); }
function medianNullSdFactor(nYears) {
  return Math.sqrt(2 * MEDIAN3_VARIANCE_FACTOR / looVarianceFactor(nYears));
}
// Baseline built from the n kept years (median of 3, mean of 2) against the
// wartime median of three years.
function baselineVarianceFactor(nYears) {
  return nYears === 3 ? MEDIAN3_VARIANCE_FACTOR : 1 / nYears;
}
function droppedBaselineSdFactor(nYears) {
  return Math.sqrt((baselineVarianceFactor(nYears) + MEDIAN3_VARIANCE_FACTOR) /
                   looVarianceFactor(nYears));
}
if (Math.abs(medianNullSdFactor(3) - MEDIAN_NULL_SD_FACTOR) > 1e-12) {
  throw new Error('medianNullSdFactor(3) must equal MEDIAN_NULL_SD_FACTOR');
}
// wartime side, over the leave-one-out variance of the reference built from
// nYears. Wartime side: 0.4487 for the median of three years, 1 for one year.
//   median, 3 reference years:            (0.4487 + 0.4487) / 1.5 -> 0.7735
//   one year, 3 reference years:          (0.4487 + 1) / 1.5      -> 0.9827
//   median, 2 years, baseline unchanged:  (0.4487 + 0.4487) / 2   -> 0.6698
//   one year, 2 years, baseline unchanged:(0.4487 + 1) / 2        -> 0.8511
//   median, 2 years, baseline = mean of 2:(0.5 + 0.4487) / 2      -> 0.6887
//   one year, 2 years, baseline = mean of 2:(0.5 + 1) / 2         -> 0.8660
var WARTIME_VARIANCE = {median: MEDIAN3_VARIANCE_FACTOR, year: 1};
function comparisonSdFactor(baselineVariance, wartimeVariance, nYears) {
  return Math.sqrt((baselineVariance + wartimeVariance) /
                   looVarianceFactor(nYears));
}
if (Math.abs(comparisonSdFactor(MEDIAN3_VARIANCE_FACTOR, 1, 3) -
             PERYEAR_NULL_SD_FACTOR) > 1e-12 ||
    Math.abs(comparisonSdFactor(0.5, MEDIAN3_VARIANCE_FACTOR, 2) -
             droppedBaselineSdFactor(2)) > 1e-12) {
  throw new Error('comparisonSdFactor disagrees with the named factors');
}

// analysis_scheme (exportTable below); GEE05_RunManifest records the rest.
var CODE_VERSION = 'v13b12';
var OUT_SFX = '_' + CODE_VERSION;
var RUN_STARTED_UTC = new Date().toISOString();
var RUN_ID = 'GEE05_' + CODE_VERSION + '_' + CHANGE_SCHEME + '_' +
  RUN_STARTED_UTC.replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');
var EXPORTED_TABLES = [];
function exportTable(params) {
  var idCols = ['run_id', 'code_version', 'analysis_scheme'];
  params.collection = ee.FeatureCollection(params.collection)
    .map(function(f) {
      return f.set({'run_id': RUN_ID, 'code_version': CODE_VERSION,
                    'analysis_scheme': CHANGE_SCHEME});
    });
  if (params.selectors) {
    params.selectors = idCols.concat(params.selectors.filter(function(s) {
      return idCols.indexOf(s) < 0;
    }));
  }
  EXPORTED_TABLES.push(params.description);
  Export.table.toDrive(params);
}
var EXPORTED_IMAGES = [];
function exportImage(params) {
  EXPORTED_IMAGES.push(params.description);
  Export.image.toDrive(params);
}
/** Ratio status: 'no_detections' when the detected area is 0 (v13b12). */
function detectionStatus(areaKm2Value) {
  return ee.Algorithms.If(ee.Number(areaKm2Value).gt(0), 'detections',
                          'no_detections');
}
var Z_CLASS_LIST = [1.5, 2, 2.5, 3, 3.5, 4];
var T_EVIDENCE_BY_SCHEME = {primary_effort: 2.5, recurrent_effort: 3.0};
var T_EVIDENCE = T_EVIDENCE_BY_SCHEME[CHANGE_SCHEME];
var MIN_PATCH_PX_CLASSES = 8;  // year-count classes: drop 8-connected patches below this
var ANNUAL_BASELINE_MODE = 'annual_median'; // 'annual_median' | 'composite'
var RUN_ANNUAL_BASELINE_SENSITIVITY = true;
var ANNUAL_OUTPUT_TAG = 'annualbase_' + ANNUAL_BASELINE_MODE;

// Distance-transform search radius. Pixels farther than this are safely set
// to the radius because every downstream decision uses <=30, >=50 or >=75 km.
var DISTANCE_NEIGHBORHOOD_PX = 1024;
var DISTANCE_CAP_KM = DISTANCE_NEIGHBORHOOD_PX * refScale / 1000;

if (['annual_median', 'composite'].indexOf(ANNUAL_BASELINE_MODE) < 0) {
  throw new Error('ANNUAL_BASELINE_MODE must be annual_median or composite.');
}

// Reproducible distance rule; the manual polygons are intentionally absent.
var CONTROL_MODE          = 'rule';
var CONTROL_MIN_DIST_KM   = 50;  // pixels at least this far from any CONTROL_SOURCE event
var CONTROL_EXCLUDE_ORDLO = true;  // also exclude the pre-2022 occupation line
var CONTROL_DIST_LIST     = [30, 50, 75];  // sensitivity over the control set

// The asset is expected to carry a property naming the source of each event.
// If it does not, set SOURCE_PROPERTY to null; the script then falls back to
// the merged collection for both roles and prints a warning, and the
// independence argument must be made in the text instead.
var SOURCE_PROPERTY = 'source';   // e.g. 'ACLED' / 'VIINA'; set to null if absent
var CONTROL_SOURCE  = 'ACLED';  // control-area feasibility check only; no role in z or maps
var VALID_SOURCE    = 'VIINA';  // comparison source for the exported distance diagnostic (band 20);
                                  // both sources are comparison data, not validation (see header)

var APPLY_GLOBAL_RECAL = false;  // main analysis: test the null; do not force mean=0, SD=1
var THRESHOLD_MODE     = 'normal';   // 'normal' | 'empirical'
var EMPIRICAL_PCTL     = 99.9;    // pooled temporal-null percentile of |pseudo-z| when 'empirical'

// These checks are sensitivity domains only. They never mask the primary
// probability, change or detection rasters, and no WorldCover class receives
// a special exception.
var noveltyAsset =
  OWN + 'SDM_Novelty_' + CHANGE_SCHEME +
  '_outdoor_final_v13b12';
var NOVELTY_FLAG_BAND = 'extrapolation_flag_annual';

var BIAS_SIGMA_LIST = [50000, 100000, 200000];
var Z_THRESH_LIST   = [2.5, 3.0, 3.5, 4.0];
var MIN_PATCH_LIST  = [4, 8, 16];
var ATTRIBUTION_DIST_LIST = [15, 30, 50];

// ---- Run flags ---------------------------------------------------------
var RUN = {
  mainOutputs:      true,
  controlZoneExport:true,
  nullDiagnostics:  true,  // histogram, sample, flagged fraction
  areaSensitivity:  true,  // area vs threshold and patch size
  sigmaSensitivity: true,  // creates ONE independent export task per sigma
  nullYearSensitivity: true,  // ONE task per dropped pre-war year
  mainSensitivityRow: true,  // Sensitivity_main_main (acceptance check)
  ordloSplit:       true,
  yearlyOutputs:    true,
  strictNull:       true,  // stricter rules on the pre-war maps
  sdFloorSensitivity: true,  // ONE task per alternative floor
  screens:          true,  // Areas_by_Screen
  sdFloorDiagnostics: true,
  yearCounts:       true,  // year-count map and table (sole producer)
  annualBaselineSensitivity: true,
  landCover:        true,
  diagnosticsTable: true
};

// **** ASSET PATHS ******************************************************
// GEE_04 exports probabilities in two unambiguous files. The period asset
// is the PRIMARY composite-input model used for the median-period analysis;
// the annual asset is used only for explicitly labelled yearly sensitivities.
// models are refitted on the rows of their cross-validation).
var PROBABILITY_ASSET_SUFFIX = {primary_effort: 'v13b12',
                                recurrent_effort: 'v13b12'};
var periodProbabilityAsset =
  OWN + 'SDM_Probability_PeriodComposites_' +
  CHANGE_SCHEME + '_outdoor_final_' + PROBABILITY_ASSET_SUFFIX[CHANGE_SCHEME];
var annualProbabilityAsset =
  OWN + 'SDM_Probability_Annual2019_2024_' +
  CHANGE_SCHEME + '_outdoor_final_' + PROBABILITY_ASSET_SUFFIX[CHANGE_SCHEME];
var aoiAsset       = AOI_ASSET;
var warEventsAsset = OWN + 'war_events';   // data/war_events4326.gpkg, uploaded
var ordloAsset     = OWN + 'ordlo';        // data/ordlo.gpkg, uploaded
var esaAsset       = 'ESA/WorldCover/v200';

// One Google Drive folder per visitation model; download them into
// outputs/GEE05_regular and outputs/GEE05_recurrent (support/folders.py).
var exportFolder = 'GEE05_' +
  ({primary_effort: 'regular', recurrent_effort: 'recurrent'})[CHANGE_SCHEME];

// =====================================================================
// 0. SETUP
// =====================================================================

var roi = ee.FeatureCollection(aoiAsset).geometry();
var refProj = ee.Projection(EXPORT_CRS, EXPORT_TRANSFORM);
var coarseProj = ee.Projection(
  EXPORT_CRS, [coarseScale, 0, 0, 0, -coarseScale, 0]);
var lcImage = ee.ImageCollection(esaAsset).filterBounds(roi).mosaic().clip(roi)
  .select('Map').reproject({crs: refProj}).rename('lc');
var YEARS = ['2022', '2023', '2024'];
// Nominal area is metadata only; every reported area uses pixelArea().
var NOMINAL_PROJECTED_PIXEL_KM2 = (refScale * refScale) / 1e6;
var groundPixelAreaKm2 = ee.Image.pixelArea().divide(1e6)
  .rename('ground_area_km2');

// Geographic bounds are kept as provenance only; EPSG:3035 is equal-area and
// metric, so no latitude correction of areas is needed.
var wgs84 = ee.Projection('EPSG:4326');
var roiWgs84 = roi.transform(wgs84, 1000, true);
var aoiCentroidLat = ee.Number(
  ee.List(roiWgs84.centroid(1000).coordinates()).get(1));
var aoiBoundsRing = ee.List(
  ee.List(roiWgs84.bounds(1000, wgs84).coordinates()).get(0));
var aoiBoundsLatitudes = aoiBoundsRing.map(function(point) {
  return ee.Number(ee.List(point).get(1));
});
var aoiSouthLat = ee.Number(aoiBoundsLatitudes.reduce(ee.Reducer.min()));
var aoiNorthLat = ee.Number(aoiBoundsLatitudes.reduce(ee.Reducer.max()));

var GROUND_FACTOR_SOUTH = ee.Number(1);
var GROUND_FACTOR_CENTROID = ee.Number(1);
var GROUND_FACTOR_NORTH = ee.Number(1);

function groundEquivalent(value, spatialKind, factor) {
  value = ee.Number(value);
  return value;
}

var smoothKernel = ee.Kernel.gaussian({
  radius: SMOOTH_SIGMA_M * 2.5, sigma: SMOOTH_SIGMA_M,
  units: 'meters', normalize: true
});

/** True ground area of a 0/1 mask in km2. Never use 500 m * 500 m here. */
function areaKm2(mask) {
  var a = mask.selfMask().multiply(groundPixelAreaKm2).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: roi, crs: refProj, scale: refScale,
    maxPixels: 1e13
  }).values().get(0);
  return ee.Number(ee.Algorithms.If(a, a, 0));
}

/** Count of pixels in a 0/1 mask. */
function pixelCount(mask) {
  var c = mask.selfMask().reduceRegion({
    reducer: ee.Reducer.count(), geometry: roi, crs: refProj, scale: refScale,
    maxPixels: 1e13
  }).values().get(0);
  return ee.Number(ee.Algorithms.If(c, c, 0));
}

/** One reduction returns true areas and exact pixel counts for both classes. */
function twoClassAreaCount(damageMask, rewildMask) {
  var damage01 = damageMask.unmask(0).toDouble();
  var rewild01 = rewildMask.unmask(0).toDouble();
  return ee.Image.cat([
    damage01.multiply(groundPixelAreaKm2).rename('damage_km2'),
    rewild01.multiply(groundPixelAreaKm2).rename('rewild_km2'),
    damage01.rename('damage_px'),
    rewild01.rename('rewild_px')
  ]).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: roi, crs: refProj, scale: refScale,
    maxPixels: 1e13
  });
}

/**
 */
function sumAreasKm2(masks) {
  var names = Object.keys(masks);
  return ee.Image.cat(names.map(function(n) {
    return masks[n].unmask(0).gt(0).multiply(groundPixelAreaKm2).rename(n);
  })).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: roi, crs: refProj, scale: refScale,
    maxPixels: 1e13
  });
}

// =====================================================================
// 1. LOAD PREDICTIONS; BUILD THE MAPPING DOMAIN AND DIAGNOSTIC SCREENS
// =====================================================================

var NODATA_VALUE = -9999;

// GeoTIFF NoData metadata is not always retained when a file is ingested as a
// GEE asset. Mask the sentinel explicitly before any convolution or threshold.
// This prevents -9999 values at the AOI edge or genuine source-data gaps from
// contaminating the 1.5-km smoothed probability surface.
function maskUploadedNoData(image) {
  var allBandsValid = image.neq(NODATA_VALUE).reduce(ee.Reducer.min());
  return image.updateMask(allBandsValid);
}

var periodProbabilityRaw = maskUploadedNoData(ee.Image(periodProbabilityAsset));
var annualProbabilityRaw = maskUploadedNoData(ee.Image(annualProbabilityAsset));
print('Uploaded period-probability bands:', periodProbabilityRaw.bandNames());
print('Uploaded annual-probability bands:', annualProbabilityRaw.bandNames());

// Select by zero-based position because GEE may rename uploaded GeoTIFF bands
// to b1, b2, ... . The manifest exported by GEE_04 documents this order.
var periodProbability = periodProbabilityRaw.select(
  [0, 1, 2, 3],
  ['prob_prewar', 'prob_wartime', 'prob_war_mean', 'prob_war_sd']);
var annualProbability = annualProbabilityRaw.select(
  [0, 1, 2, 3, 4, 5],
  ['prob_2019', 'prob_2020', 'prob_2021',
   'prob_2022', 'prob_2023', 'prob_2024']);

var prewar  = periodProbability.select('prob_prewar');
var wartime = periodProbability.select('prob_wartime');
var warSD   = periodProbability.select('prob_war_sd');

// REPRESENTATION CONSISTENCY FOR THE ANNUAL ANALYSES.
// The composite prewar probability is fitted and predicted on a three-year
// median embedding, while each annual wartime probability is predicted from a
// single, noisier annual embedding. A Random Forest shrinks predictions from
// noisier inputs towards the middle of the training distribution, so
// composite_prewar minus annual_wartime carries a systematic component that is
// positive where suitability is high and negative where it is low. That bias is
// organised by suitability rather than by geography, so the 100 km control-zone
// bias field cannot remove it, and it feeds straight into the persistence
// evidence line of the confidence index.
// GEE_04 exports prob_2019 to prob_2021, so the annual analyses can use
// a baseline in the SAME representation as the wartime years they are compared
// with. The composite baseline remains available for a like-for-like check.
var prewarAnnualMedian = annualProbability
  .select(['prob_2019', 'prob_2020', 'prob_2021'])
  .reduce(ee.Reducer.median()).rename('prob_prewar_annual_median');
var warAnnualMedian = annualProbability
  .select(['prob_2022', 'prob_2023', 'prob_2024'])
  .reduce(ee.Reducer.median()).rename('prob_wartime_annual_median');

print('Baseline used for the per-year differences:', ANNUAL_BASELINE_MODE);

var probYear = {};
YEARS.forEach(function(y) {
  probYear[y] = annualProbability.select('prob_' + y);
});

// predictions valid, so every comparison, pseudo-z and denominator is defined
// on the same pixels. WorldCover, novelty, probability-interior and
// wartime-variability checks are not part of this mask.
var validMappingDomain = prewar.mask().and(wartime.mask())
  .and(warSD.mask())
  .and(annualProbability.mask().reduce(ee.Reducer.min()))
  .rename('valid_mapping_domain');

// propensity screen uses the smoothed 2019-2021 annual-median prediction, the
// baseline of the comparisons, instead of the composite-model prediction).
var prewarAnnualMedianSmooth = prewarAnnualMedian.convolve(smoothKernel);
var probabilityInterior = prewarAnnualMedianSmooth.gt(PROB_LO)
  .and(prewarAnnualMedianSmooth.lt(PROB_HI)).rename('probability_interior');
var wartimeSdAccepted = warSD.lte(MAX_WAR_SD).rename('wartime_sd_accepted');
var eligible_nonovel = validMappingDomain
  .and(warSD.lte(MAX_WAR_SD))
  .and(probabilityInterior)
  .rename('quality_screen_no_novelty');

// outside its valid source coverage and do not use it as the main raster mask.
var noveltyImage = ee.Image(noveltyAsset);
if (!noveltyImage.bandNames().contains(NOVELTY_FLAG_BAND).getInfo()) {
  throw new Error('Novelty asset lacks ' + NOVELTY_FLAG_BAND +
                  ': rerun GEE_04 v13b12 first.');
}
var extrapFlag = maskUploadedNoData(noveltyImage)
  .select([NOVELTY_FLAG_BAND], ['extrapolation_flag'])
  .reproject({crs: refProj}).toByte();
var noveltyAccepted = extrapFlag.eq(0).rename('novelty_accepted');
var eligible_novel = eligible_nonovel.and(noveltyAccepted)
  .rename('quality_screen_with_novelty');

// Main analysis mask: valid input coverage only. This produces continuous
// surfaces across all WorldCover classes wherever embeddings are available.
var aoi_mask = validMappingDomain;
// Reused throughout the sensitivity tables; computing these once avoids many
// identical full-AOI reductions.
var ELIGIBLE_GROUND_AREA_KM2 = areaKm2(aoi_mask);
var ELIGIBLE_PIXEL_COUNT = pixelCount(aoi_mask);
var ELIGIBLE_NOMINAL_PROJECTED_AREA_KM2 =
  ELIGIBLE_PIXEL_COUNT.multiply(NOMINAL_PROJECTED_PIXEL_KM2);
var NOMINAL_TO_TRUE_ELIGIBLE_AREA_RATIO =
  ELIGIBLE_NOMINAL_PROJECTED_AREA_KM2.divide(
    ELIGIBLE_GROUND_AREA_KM2.max(1e-12));

// Audit of the pixel-count × 0.25 km² arithmetic. This table uses
// the same eligible grid for both values, so their difference is projection
// arithmetic rather than a difference in classification or masking.
if (RUN.mainOutputs) {
  exportTable({
    collection: ee.FeatureCollection([ee.Feature(null, {
      'analysis_crs': EXPORT_CRS,
      'nominal_scale_m': refScale,
      'eligible_px': ELIGIBLE_PIXEL_COUNT,
      'eligible_true_ground_km2': ELIGIBLE_GROUND_AREA_KM2,
      'eligible_nominal_projected_km2':
        ELIGIBLE_NOMINAL_PROJECTED_AREA_KM2,
      'nominal_to_true_area_ratio': NOMINAL_TO_TRUE_ELIGIBLE_AREA_RATIO,
      'aoi_south_latitude': aoiSouthLat,
      'aoi_centroid_latitude': aoiCentroidLat,
      'aoi_north_latitude': aoiNorthLat,
      'ground_factor_south_cos_lat': GROUND_FACTOR_SOUTH,
      'ground_factor_centroid_cos_lat': GROUND_FACTOR_CENTROID,
      'ground_factor_north_cos_lat': GROUND_FACTOR_NORTH
    })]),
    description: 'Area_Projection_Diagnostic',
    folder: exportFolder,
    fileFormat: 'CSV'
  });

  // Linear and areal analysis parameters. In EPSG:3035 the ground-equivalent
  // columns equal the configured values.
  var scaleParameterSpecs = [
    ['analysis_pixel_size', refScale, 'm', 'linear'],
    ['coarse_grid_size', coarseScale, 'm', 'linear'],
    ['neighbourhood_3x3_width', refScale * 3, 'm', 'linear'],
    ['smoothing_sigma', SMOOTH_SIGMA_M, 'm', 'linear'],
    ['bias_variance_sigma', BIAS_SIGMA_M, 'm', 'linear'],
    ['attribution_radius', ATTRIBUTION_DIST_KM, 'km', 'linear'],
    ['control_zone_minimum_distance', CONTROL_MIN_DIST_KM, 'km', 'linear'],
    ['distance_transform_cap', DISTANCE_CAP_KM, 'km', 'linear'],
    ['minimum_patch_area_' + MIN_PATCH_PX + '_cells',
      MIN_PATCH_PX * NOMINAL_PROJECTED_PIXEL_KM2, 'km2', 'areal']
  ];
  BIAS_SIGMA_LIST.forEach(function(value) {
    scaleParameterSpecs.push([
      'bias_sigma_sensitivity_' + value + 'm', value, 'm', 'linear']);
  });
  CONTROL_DIST_LIST.forEach(function(value) {
    scaleParameterSpecs.push([
      'control_distance_sensitivity_' + value + 'km', value, 'km', 'linear']);
  });
  MIN_PATCH_LIST.forEach(function(value) {
    scaleParameterSpecs.push([
      'patch_sensitivity_' + value + '_cells',
      value * NOMINAL_PROJECTED_PIXEL_KM2, 'km2', 'areal']);
  });

  var scaleParameterRows = scaleParameterSpecs.map(function(row) {
    return ee.Feature(null, {
      'parameter': row[0],
      'spatial_kind': row[3],
      'configured_unit': row[2],
      'value_as_configured_on_analysis_grid': row[1],
      'ground_equivalent_south': groundEquivalent(
        row[1], row[3], GROUND_FACTOR_SOUTH),
      'ground_equivalent_centroid': groundEquivalent(
        row[1], row[3], GROUND_FACTOR_CENTROID),
      'ground_equivalent_north': groundEquivalent(
        row[1], row[3], GROUND_FACTOR_NORTH),
      'aoi_south_latitude': aoiSouthLat,
      'aoi_centroid_latitude': aoiCentroidLat,
      'aoi_north_latitude': aoiNorthLat,
      'ground_factor_south_cos_lat': GROUND_FACTOR_SOUTH,
      'ground_factor_centroid_cos_lat': GROUND_FACTOR_CENTROID,
      'ground_factor_north_cos_lat': GROUND_FACTOR_NORTH,
      'analysis_crs': EXPORT_CRS,
      'correction_status': 'recomputed_on_equal_area_metric_grid',
      'method': 'EPSG3035_configured_value;pixelArea_for_area_outputs'
    });
  });

  exportTable({
    collection: ee.FeatureCollection(scaleParameterRows),
    description: 'Analysis_Grid_and_Scale_Parameters',
    folder: exportFolder,
    fileFormat: 'CSV'
  });
}

// =====================================================================
// =====================================================================

var warAll = ee.FeatureCollection(warEventsAsset).filterBounds(roi);
var warControl, warValid;

if (SOURCE_PROPERTY) {
  warControl = warAll.filter(ee.Filter.eq(SOURCE_PROPERTY, CONTROL_SOURCE));
  warValid   = warAll.filter(ee.Filter.eq(SOURCE_PROPERTY, VALID_SOURCE));
} else {
  print('WARNING: SOURCE_PROPERTY is null. Control zones, the confidence index '
      + 'and the comparison with events all use the same merged event set, so '
      + 'the comparison is not independent. Split the asset to fix this.');
  warControl = warAll;
  warValid   = warAll;
}
print('Control-role events:', warControl.size());
print('Comparison events (VALID_SOURCE):', warValid.size());

// Fail early if the source labels do not match the event asset. Otherwise an
// empty collection would generate a superficially plausible all-far raster.
if (SOURCE_PROPERTY) {
  if (warControl.size().getInfo() === 0) {
    throw new Error('No control-role events found. Check SOURCE_PROPERTY and CONTROL_SOURCE.');
  }
  if (warValid.size().getInfo() === 0) {
    throw new Error('No comparison events found. Check SOURCE_PROPERTY and VALID_SOURCE.');
  }
}

function distanceKmTo(fc) {
  var raster = ee.Image(0).byte().paint(fc, 1).reproject(refProj);
  // fastDistanceTransform measures distance to the nearest non-zero pixel.
  // Do not invert the event raster: raster.not() would instead measure the
  // distance to non-event pixels and return zero across almost the whole AOI.
  // Arguments are named: the second positional argument is UNITS, not metric.
  // The returned numerical unit is kilometres on the EPSG:3035 analysis grid.
  var distanceKm = raster.fastDistanceTransform({
      neighborhood: DISTANCE_NEIGHBORHOOD_PX,
      units: 'pixels',
      metric: 'squared_euclidean'
    })
    .sqrt().reproject(refProj).multiply(refScale / 1000);
  // fastDistanceTransform only searches within the specified neighbourhood.
  // Anything still masked is farther than the cap, which is sufficient for
  // all attribution and control-zone thresholds in this script.
  return distanceKm.unmask(DISTANCE_CAP_KM);
}

var distControlKm = distanceKmTo(warControl).rename('dist_control_km');
// Sanity check: if the transform is working, the maximum distance to a control
// event must be far greater than ATTRIBUTION_DIST_KM. A value near zero or near
// 1 km means the transform silently failed and the attribution mask is passing
// everything, which is the failure this pipeline has already had twice.
var distControlStats = distControlKm.reduceRegion({
  reducer: ee.Reducer.minMax(), geometry: roi, scale: coarseScale,
  maxPixels: 1e9, bestEffort: true
});
print('Control-event distance min/max (km); max must be >> '
      + ATTRIBUTION_DIST_KM + ':', distControlStats);
var distValidKm   = distanceKmTo(warValid).rename('dist_valid_km');

// =====================================================================
// =====================================================================

var ordloMask = ee.Image(0).byte().paint(ee.FeatureCollection(ordloAsset), 1)
  .reproject(refProj).eq(1).rename('in_ordlo');

/**
 * Rule-based control zones. Stating the rule is the whole point: any reader
 * can regenerate exactly this mask from the published event data.
 */
function buildControlMask(minDistKm) {
  var ruleMask = distControlKm.gte(minDistKm);
  if (CONTROL_EXCLUDE_ORDLO) { ruleMask = ruleMask.and(ordloMask.not()); }

  return ruleMask.and(aoi_mask).rename('control');
}

var controlMask = buildControlMask(CONTROL_MIN_DIST_KM);

if (RUN.controlZoneExport) {
  // beside the (empty) rule mask, because the distance surface IS the
  // documentation of why no spatial control zone exists in this AOI.
  exportImage({
    image: distControlKm.toFloat()
      .addBands(controlMask.unmask(0).toFloat()).clip(roi),
    description: 'ControlZones_' + CONTROL_MODE + '_' +
      CONTROL_MIN_DIST_KM + 'km',
    folder: exportFolder, region: roi, crs: EXPORT_CRS,
    crsTransform: EXPORT_TRANSFORM,
    maxPixels: 1e13, fileFormat: 'GeoTIFF'
  });

  // The count and area the referee asked for.
  var ctrlMeta = ee.FeatureCollection([ee.Feature(null, {
    'analysis_scheme': CHANGE_SCHEME,
    'analysis_scheme_label': CHANGE_SCHEME_LABEL,
    'control_mode': CONTROL_MODE,
    'min_dist_km': CONTROL_MIN_DIST_KM,
    'distance_units': 'EPSG3035_grid_km',
    'min_dist_projected_km': CONTROL_MIN_DIST_KM,
    'min_dist_ground_south_km': groundEquivalent(
      CONTROL_MIN_DIST_KM, 'linear', GROUND_FACTOR_SOUTH),
    'min_dist_ground_centroid_km': groundEquivalent(
      CONTROL_MIN_DIST_KM, 'linear', GROUND_FACTOR_CENTROID),
    'min_dist_ground_north_km': groundEquivalent(
      CONTROL_MIN_DIST_KM, 'linear', GROUND_FACTOR_NORTH),
    'attribution_projected_km': ATTRIBUTION_DIST_KM,
    'attribution_ground_centroid_km': groundEquivalent(
      ATTRIBUTION_DIST_KM, 'linear', GROUND_FACTOR_CENTROID),
    'exclude_ordlo': CONTROL_EXCLUDE_ORDLO ? 1 : 0,
    'control_source': SOURCE_PROPERTY ? CONTROL_SOURCE : 'merged',
    'n_control_role_events': warControl.size(),
    'distance_neighborhood_px': DISTANCE_NEIGHBORHOOD_PX,
    'distance_cap_km': DISTANCE_CAP_KM,
    'distance_cap_projected_km': DISTANCE_CAP_KM,
    'distance_cap_ground_centroid_km': groundEquivalent(
      DISTANCE_CAP_KM, 'linear', GROUND_FACTOR_CENTROID),
    'distance_min_km': distControlStats.get('dist_control_km_min'),
    'distance_max_km': distControlStats.get('dist_control_km_max'),
    'distance_min_projected_km':
      distControlStats.get('dist_control_km_min'),
    'distance_max_projected_km':
      distControlStats.get('dist_control_km_max'),
    'distance_min_ground_centroid_approx_km': groundEquivalent(
      distControlStats.get('dist_control_km_min'),
      'linear', GROUND_FACTOR_CENTROID),
    'distance_max_ground_centroid_approx_km': groundEquivalent(
      distControlStats.get('dist_control_km_max'),
      'linear', GROUND_FACTOR_CENTROID),
    'n_pixels_within_attribution_distance': pixelCount(
      distControlKm.lte(ATTRIBUTION_DIST_KM).and(aoi_mask)),
    'fraction_within_attribution_distance': pixelCount(
      distControlKm.lte(ATTRIBUTION_DIST_KM).and(aoi_mask))
      .divide(ELIGIBLE_PIXEL_COUNT),
    'n_pixels_beyond_control_distance': pixelCount(
      distControlKm.gte(CONTROL_MIN_DIST_KM).and(aoi_mask)),
    'fraction_beyond_control_distance': pixelCount(
      distControlKm.gte(CONTROL_MIN_DIST_KM).and(aoi_mask))
      .divide(ELIGIBLE_PIXEL_COUNT),
    'n_control_pixels': pixelCount(controlMask),
    'control_area_km2': areaKm2(controlMask),
    'spatial_controls_used_for_calibration': 0,
    'null_definition': 'temporal_prewar_loo_pseudo_z_v13b8',
    'n_eligible_pixels': ELIGIBLE_PIXEL_COUNT,
    'eligible_ground_area_km2': ELIGIBLE_GROUND_AREA_KM2,
    'nominal_projected_pixel_km2': NOMINAL_PROJECTED_PIXEL_KM2,
    'nominal_to_true_eligible_area_ratio':
      NOMINAL_TO_TRUE_ELIGIBLE_AREA_RATIO,
    'scale_correction_status': 'recomputed_on_equal_area_metric_grid'
  })]);
  exportTable({
    collection: ctrlMeta,
    description: 'ControlZones_Metadata',
    folder: exportFolder, fileFormat: 'CSV'
  });
}

// =====================================================================
// 4. HELPERS: NORMALISED CONVOLUTION AND GLOBAL RENORMALISATION
// =====================================================================

function clampedNormConv(values, mask, sigmaM) {
  var valCoarse = values.updateMask(mask).unmask(0)
    .setDefaultProjection(refProj)
    .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 256})
    .reproject(coarseProj);
  var maskCoarse = mask.toFloat().unmask(0)
    .setDefaultProjection(refProj)
    .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 256})
    .reproject(coarseProj);

  var kernel = ee.Kernel.gaussian({
    radius: sigmaM * 2.5, sigma: sigmaM, units: 'meters', normalize: false
  });

  var num = valCoarse.reduceNeighborhood({reducer: ee.Reducer.sum(), kernel: kernel});
  var den = maskCoarse.reduceNeighborhood({reducer: ee.Reducer.sum(), kernel: kernel});

  var denMax = den.reduceRegion({
    reducer: ee.Reducer.max(), geometry: roi, scale: coarseScale,
    maxPixels: 1e8, bestEffort: true
  }).values().get(0);
  denMax = ee.Number(denMax).max(1);

  return num.divide(den.max(denMax.multiply(DEN_CLAMP_FRAC)))
    .resample('bilinear').reproject(refProj);
}

function globalZNorm(zImage, ctrlMask, bandName) {
  // uses the pooled temporal-null moments instead).
  var stats = zImage.updateMask(ctrlMask).reduceRegion({
    reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true),
    geometry: roi, scale: coarseScale, maxPixels: 1e9, bestEffort: true
  });
  var mu = ee.Number(stats.get(ee.String(bandName).cat('_mean')));
  var sd = ee.Number(stats.get(ee.String(bandName).cat('_stdDev'))).max(0.5);
  return zImage.subtract(mu).divide(sd);
}

// =====================================================================
// =====================================================================
// The reference is built from
// TIME: each pre-war year y in {2019, 2020, 2021} yields a pre-war contrast,
// mean(other two pre-war years) - year y, smoothed with the same kernel as
// the wartime changes. The spatially pooled mean of the contrasts is the
// bias field; their pooled squared deviation is the local variance (its
// expectation, 1.5 sigma^2, matches the design factors, so the divisor 3 is
// is zero by construction and cannot capture a pre-war trend. A trend that
// continues into 2022-2024 appears as wartime change (for example 0.2, 0.3,
// 0.4 continuing to 0.5, 0.6, 0.7 gives z = -3.2 in the median comparison);
// stability of the no-war process is an ASSUMPTION of this reference.

var PREWAR_YEARS = ['2019', '2020', '2021'];
var probPrewarYear = {};
PREWAR_YEARS.forEach(function(y) {
  probPrewarYear[y] = annualProbability.select('prob_' + y);
});

function buildTemporalNull(biasSigma, smoothSig, yearList, minSd) {
  var years = yearList || PREWAR_YEARS;
  // floor of the n-year leave-one-out SD is MIN_LOCAL_SD x
  // sqrt(looVarianceFactor(n) / 1.5) (unchanged for three years).
  var floorSd = (minSd === undefined ? MIN_LOCAL_SD : minSd) *
    Math.sqrt(looVarianceFactor(years.length) / LOO_VARIANCE_FACTOR);
  var kernel = (smoothSig > 0)
    ? ee.Kernel.gaussian({radius: smoothSig * 2.5, sigma: smoothSig,
                          units: 'meters', normalize: true})
    : null;
  var diffs = [];
  years.forEach(function(y) {
    var sum = null;
    var m = 0;
    years.forEach(function(o) {
      if (o !== y) {
        sum = (sum === null) ? probPrewarYear[o]
          : sum.add(probPrewarYear[o]);
        m += 1;
      }
    });
    var d = sum.divide(m).subtract(probPrewarYear[y]);
    if (kernel) { d = d.convolve(kernel); }
    diffs.push(d.rename('pseudo_diff_' + y));
  });

  var meanDiff = null;
  diffs.forEach(function(d) {
    meanDiff = (meanDiff === null) ? d : meanDiff.add(d);
  });
  meanDiff = meanDiff.divide(diffs.length);
  var bias = clampedNormConv(meanDiff, aoi_mask, biasSigma)
    .rename('bias_field');

  var sqSum = null;
  diffs.forEach(function(d) {
    var dev = d.subtract(bias).pow(2);
    sqSum = (sqSum === null) ? dev : sqSum.add(dev);
  });
  var localSDRaw = clampedNormConv(sqSum.divide(diffs.length), aoi_mask,
      biasSigma)
    .sqrt().rename('local_sd_unfloored');
  var localSD = localSDRaw.max(floorSd).rename('local_sd');

  var pseudoZ = [];
  for (var i = 0; i < diffs.length; i++) {
    pseudoZ.push(diffs[i].subtract(bias).divide(localSD)
      .rename('pseudo_z_' + years[i]));
  }
  return {bias: bias, localSD: localSD, localSDRaw: localSDRaw,
          floorSd: floorSd, diffs: diffs, pseudoZ: pseudoZ, years: years};
}

/**
 * Pooled random sample of the null pseudo-z as one FeatureCollection with
 * columns zscore_median (pseudo-z), zscore_raw_median (pseudo-diff / local
 * SD, i.e. without the bias correction), in_ordlo, lon, lat, null_year.
 * Reused for the sample export, the empirical threshold, the pooled moment
 * summaries and the per-zone null validity stats.
 */
function sampleNullZ(nullFields, perYearPixels, extraMask) {
  function sampleOneYear(i) {
    var y = nullFields.years[i];
    var img = nullFields.pseudoZ[i].rename('zscore_median')
      .addBands(nullFields.diffs[i].divide(nullFields.localSD)
        .rename('zscore_raw_median'))
      .addBands(ordloMask.rename('in_ordlo'))
      .updateMask(extraMask ? aoi_mask.and(extraMask) : aoi_mask);
    return img.sample({
      region: roi, scale: refScale, numPixels: perYearPixels,
      seed: 20250523 + i, geometries: true, tileScale: 8
    }).map(function(f) {
      var c = f.geometry().coordinates();
      return f.set({'lon': c.get(0), 'lat': c.get(1), 'null_year': y});
    });
  }
  var out = sampleOneYear(0);
  for (var j = 1; j < nullFields.years.length; j++) {
    out = out.merge(sampleOneYear(j));
  }
  return out;
}

var NULL_MAIN = buildTemporalNull(BIAS_SIGMA_M, SMOOTH_SIGMA_M, null);
// pixels in the mapping domain, all pre-war years pooled.
var NULL_VALID_PX = ee.Number(0);
NULL_MAIN.pseudoZ.forEach(function(pz) {
  NULL_VALID_PX = NULL_VALID_PX.add(pixelCount(pz.abs().gte(0).and(aoi_mask)));
});
var nullZSample = sampleNullZ(NULL_MAIN, 20000, null);
var nullPooledMoments = nullZSample.reduceColumns(
  ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true),
  ['zscore_median']);
var NULL_POOLED = {
  mean: ee.Number(nullPooledMoments.get('mean')),
  sd:   ee.Number(nullPooledMoments.get('stdDev')).max(0.5)
};

// =====================================================================
// =====================================================================

function correctPeriod(probWar, periodLabel, opts) {
  opts = opts || {};
  var biasSigma  = opts.biasSigma  || BIAS_SIGMA_M;
  var smoothSig  = (opts.smoothSigma === undefined) ? SMOOTH_SIGMA_M : opts.smoothSigma;
  var recalibrate = (opts.applyGlobalRecal === undefined)
    ? APPLY_GLOBAL_RECAL : opts.applyGlobalRecal;
  // Baseline defaults to the composite prewar surface; the annual periods pass
  // the annual-median baseline so both sides share one representation.
  var baseline   = opts.baseline || prewar;

  var rawDiff = baseline.subtract(probWar).rename('raw_diff');

  // STEP 1: smoothing (smoothSigma = 0 disables it, for sensitivity runs)
  var smoothDiff = (smoothSig > 0)
    ? rawDiff.convolve(ee.Kernel.gaussian({
        radius: smoothSig * 2.5, sigma: smoothSig, units: 'meters', normalize: true}))
      .rename('smooth_diff')
    : rawDiff.rename('smooth_diff');

  // (section 4B). opts.nullFields overrides for the sensitivity runs; a
  // non-default biasSigma rebuilds the null at that pooling scale so the
  // wartime correction and its null always share one parameterisation.
  var nullFields = opts.nullFields ||
    ((biasSigma === BIAS_SIGMA_M && smoothSig === SMOOTH_SIGMA_M)
      ? NULL_MAIN : buildTemporalNull(biasSigma, smoothSig, null));
  var biasField = nullFields.bias;
  var corrected = smoothDiff.subtract(biasField).rename('corrected');
  // in units of THIS comparison's null SD (1 for the composite sensitivity,
  // 0.7735 for median-vs-median, 0.9827 for year-vs-median3).
  var sdFactor = (opts.sdFactor === undefined) ? 1 : opts.sdFactor;
  var localSD = nullFields.localSD.multiply(sdFactor).rename('local_sd');
  var zRaw = corrected.divide(localSD).rename('zscore_raw_' + periodLabel);

  // POOLED temporal-null moments so the null is tested, not fitted.
  var zScore = recalibrate
    ? zRaw.subtract(NULL_POOLED.mean).divide(NULL_POOLED.sd)
        .rename('zscore_' + periodLabel)
    : zRaw.rename('zscore_' + periodLabel);

  return {
    diff_raw:       rawDiff.rename('raw_' + periodLabel),
    diff_smooth:    smoothDiff.rename('smooth_' + periodLabel),
    bias_field:     biasField.rename('bias_' + periodLabel),
    diff_corrected: corrected.rename('diff_' + periodLabel),
    local_sd:       localSD,
    z_score_raw:    zRaw,
    z_score:        zScore,
    war_debiased:   probWar.add(biasField).rename('prob_' + periodLabel + '_debiased'),
    recalibrated:   recalibrate
  };
}

// the same representation on both sides, calibrated by the exact design
// factor. The composite-vs-composite difference is retained as a labelled,
// deliberately conservative sensitivity (factor 1).
var resMedian = correctPeriod(warAnnualMedian, 'median', {
  baseline: prewarAnnualMedian, sdFactor: MEDIAN_NULL_SD_FACTOR});
var resComposite = correctPeriod(wartime, 'composite', {});
var resYearAnnualMedian = {};
var resYearComposite = {};
YEARS.forEach(function(y) {
  resYearAnnualMedian[y] = correctPeriod(
    probYear[y], y + '_annualbase', {baseline: prewarAnnualMedian,
      sdFactor: PERYEAR_NULL_SD_FACTOR});
  resYearComposite[y] = correctPeriod(
    probYear[y], y + '_compositebase', {baseline: prewar});
});
var resYear = (ANNUAL_BASELINE_MODE === 'annual_median')
  ? resYearAnnualMedian : resYearComposite;

// =====================================================================
// =====================================================================

function detectChanges(zScore, zThr, zNbr, minPatch, closing, domain) {
  closing = (closing === undefined || closing === null) ? APPLY_CLOSING : closing;
  // domain (default: the mapping mask).
  var domain01 = (domain || aoi_mask).unmask(0).gt(0);
  var zIn = zScore.updateMask(domain01);
  // Centre-inclusive 3x3 mean over the domain cells only (not an independent
  // check by eight neighbours; adjacent pixels are strongly dependent after
  // the 1.5 km smoothing).
  var zNeighMean = zIn.reduceNeighborhood({
    reducer: ee.Reducer.mean(),
    kernel: ee.Kernel.square({radius: 1, units: 'pixels'}),
    skipMasked: true
  });

  // sensitivity loops) or as ee.Number (Z_USED/ZN_USED in the main run).
  // JavaScript unary minus on an ee.Number yields NaN, so `zScore.lt(-zThr)`
  // silently corrupted (or errored out) the rewilding branch of the MAIN
  // detection. Both thresholds are now coerced to constant images and negated
  // server-side, which is correct for both input types.
  var zThrImg = ee.Image.constant(zThr);
  var zNbrImg = ee.Image.constant(zNbr);
  // Candidates: 1 inside the domain where both rules hold, 0 everywhere else.
  var damage = zIn.gte(zThrImg).and(zNeighMean.gte(zNbrImg))
    .unmask(0).and(domain01).toByte();
  var rewild = zIn.lte(zThrImg.multiply(-1))
    .and(zNeighMean.lte(zNbrImg.multiply(-1)))
    .unmask(0).and(domain01).toByte();

  // One-pixel morphological closing (dilation then erosion, circle radius
  // 1 pixel = cross) on the 0/1 grid. It can fill a pixel that failed both
  // criteria and join nearby candidates, so the result need not be a subset
  // of the exceedances. Cells outside the domain are removed before the patch
  // rule, so they can neither be added nor join a patch.
  if (closing) {
    var closeKernel = ee.Kernel.circle({radius: 1, units: 'pixels'});
    damage = damage.focal_max({kernel: closeKernel})
      .focal_min({kernel: closeKernel}).and(domain01).toByte();
    rewild = rewild.focal_max({kernel: closeKernel})
      .focal_min({kernel: closeKernel}).and(domain01).toByte();
  }

  damage = damage.selfMask().connectedPixelCount({maxSize: 256, eightConnected: true})
    .gte(minPatch).unmask(0).and(domain01).toByte();
  rewild = rewild.selfMask().connectedPixelCount({maxSize: 256, eightConnected: true})
    .gte(minPatch).unmask(0).and(domain01).toByte();

  return {damage: damage, rewild: rewild};
}

// normal tail. With THRESHOLD_MODE = 'empirical' the cut-off is the
// EMPIRICAL_PCTL percentile of the pooled |pseudo-z| sample, so the nominal
// false positive rate is the observed pre-war exceedance rate by
// construction.
var nullAbsPct = nullZSample.map(function(f) {
  return f.set('abs_z', ee.Number(f.get('zscore_median')).abs());
}).reduceColumns(
  ee.Reducer.percentile([95, 99, 99.5, EMPIRICAL_PCTL],
                        ['p95', 'p99', 'p995', 'pEmp']),
  ['abs_z']);
var zThreshEmpirical = ee.Number(nullAbsPct.get('pEmp'));
print('Temporal-null |pseudo-z| percentiles (pooled sample):', nullAbsPct);

var Z_USED = (THRESHOLD_MODE === 'empirical') ? zThreshEmpirical : ee.Number(Z_THRESH);
var ZN_USED = (THRESHOLD_MODE === 'empirical')
  ? zThreshEmpirical.multiply(Z_NEIGHBOR / Z_THRESH) : ee.Number(Z_NEIGHBOR);

var detMedian = detectChanges(resMedian.z_score, Z_USED, ZN_USED, MIN_PATCH_PX);
var damage_primary = detMedian.damage.rename('damage');
var rewild_primary = detMedian.rewild.rename('rewild');
var detComposite = detectChanges(
  resComposite.z_score, Z_USED, ZN_USED, MIN_PATCH_PX);

function detectAnnualSet(resultSet) {
  var damage = {};
  var rewild = {};
  YEARS.forEach(function(y) {
    var det = detectChanges(
      resultSet[y].z_score, Z_USED, ZN_USED, MIN_PATCH_PX);
    damage[y] = det.damage;
    rewild[y] = det.rewild;
  });
  return {damage: damage, rewild: rewild};
}

var detectionAnnualMedian = detectAnnualSet(resYearAnnualMedian);
var detectionComposite = detectAnnualSet(resYearComposite);
var selectedAnnualDetection = (ANNUAL_BASELINE_MODE === 'annual_median')
  ? detectionAnnualMedian : detectionComposite;
var damageYear = selectedAnnualDetection.damage;
var rewildYear = selectedAnnualDetection.rewild;

// =====================================================================
// 7. PERSISTENCE, ATTRIBUTION AND YEAR COUNTS
// =====================================================================

function persistenceFromAnnualDetections(detection, suffix) {
  var damageCount = detection.damage['2022']
    .add(detection.damage['2023']).add(detection.damage['2024'])
    .rename('dmg_persist_count_' + suffix);
  var rewildCount = detection.rewild['2022']
    .add(detection.rewild['2023']).add(detection.rewild['2024'])
    .rename('rew_persist_count_' + suffix);
  var damageMask = damageCount.gte(2).selfMask()
    .connectedPixelCount({maxSize: 256, eightConnected: true})
    .gte(MIN_PATCH_PX).unmask(0).rename('damage_persist_' + suffix);
  var rewildMask = rewildCount.gte(2).selfMask()
    .connectedPixelCount({maxSize: 256, eightConnected: true})
    .gte(MIN_PATCH_PX).unmask(0).rename('rewild_persist_' + suffix);
  return {
    damageCount: damageCount,
    rewildCount: rewildCount,
    damage: damageMask,
    rewild: rewildMask
  };
}

var persistenceAnnualMedian = persistenceFromAnnualDetections(
  detectionAnnualMedian, 'annualbase');
var persistenceComposite = persistenceFromAnnualDetections(
  detectionComposite, 'compositebase');
var selectedPersistence = (ANNUAL_BASELINE_MODE === 'annual_median')
  ? persistenceAnnualMedian : persistenceComposite;

var damagePersistCount = selectedPersistence.damageCount
  .rename('dmg_persist_count');
var rewildPersistCount = selectedPersistence.rewildCount
  .rename('rew_persist_count');
var damagePersist = selectedPersistence.damage.rename('damage_persist');
var rewildPersist = selectedPersistence.rewild.rename('rewild_persist');

// are kept out of everything upstream of the comparison with events.
var nearWar = distControlKm.lte(ATTRIBUTION_DIST_KM);
var damageAttrib = damage_primary.and(nearWar).rename('damage_attributed');
var rewildAttrib = rewild_primary.and(nearWar).rename('rewild_attributed');

var absZ = resMedian.z_score.abs();
// ee.Image.constant so the threshold is promoted explicitly.
var flagStrongZ = absZ.gt(ee.Image.constant(ee.Number(Z_USED).add(1)))
  .toByte();  // one unit above the working threshold
var flagNearWar = nearWar.toByte();

// of war years (2022-2024) in which the calibrated per-year z exceeded the
// model's evidence threshold in the same direction. Signed classes:
// +1/+2/+3 visitation-LOSS anomaly present in 1/2/3 war years ("damage" in
// interpretive terms), -1/-2/-3 the same for visitation-GAIN anomalies
// (wilding-like signal), 0 no evidence year. Cumulative and inclusive: the
// transient 2022 gain pulse appears at -1 instead of vanishing into the
// median. Mixed pixels (evidence on both sides, e.g. 2022 gain then 2023-24
// loss) take the side with more evidence years; ties follow the sign of the
// median z; both raw counts are exported. The attribution-distance gate is
// dropped (saturated: 92-100% of the AOI is within any radius). Error
// baseline, so their exceedances are correlated across years (about 0.31
// under independent Gaussian years) and a binomial expectation is not valid;
// Cumulative_Confidence_Class_Areas reports observed areas only.
function cumulativeConfidence(resultSet, suffix) {
  var posN = ee.Image(0);
  var negN = ee.Image(0);
  YEARS.forEach(function(y) {
    posN = posN.add(resultSet[y].z_score.gte(T_EVIDENCE));
    negN = negN.add(resultSet[y].z_score.lte(-T_EVIDENCE));
  });
  var lossWins = posN.gt(negN)
    .or(posN.eq(negN).and(posN.gt(0)).and(resMedian.z_score.gte(0)));
  var gainWins = negN.gt(posN)
    .or(posN.eq(negN).and(negN.gt(0)).and(resMedian.z_score.lt(0)));
  return {
    cls: posN.multiply(lossWins).subtract(negN.multiply(gainWins))
      .toInt8().rename('confidence_' + suffix),
    posN: posN.toInt8(),
    negN: negN.toInt8()
  };
}

var confidenceAnnualMedianSet = cumulativeConfidence(
  resYearAnnualMedian, 'annualbase');
var confidenceCompositeSet = cumulativeConfidence(
  resYearComposite, 'compositebase');
var confidenceAnnualMedian = confidenceAnnualMedianSet.cls;
var confidenceComposite = confidenceCompositeSet.cls;
var confidenceSet = (ANNUAL_BASELINE_MODE === 'annual_median')
  ? confidenceAnnualMedianSet : confidenceCompositeSet;
var confidence = confidenceSet.cls.rename('confidence');

// the only producer). Mapped classes keep the dominant direction and remove
// 8-connected patches < MIN_PATCH_PX_CLASSES per direction; the unions keep
// every pixel with at least one anomalous year in that direction (a pixel can
// be in both), with the same patch rule applied to each union separately.
function patchKeepDirection(mask01) {
  return mask01.selfMask()
    .connectedPixelCount({maxSize: 256, eightConnected: true})
    .gte(MIN_PATCH_PX_CLASSES).unmask(0);
}
var classPx = confidenceSet.cls.updateMask(aoi_mask).rename('class_px');
var classMapped = classPx.multiply(patchKeepDirection(classPx.gt(0))
  .max(patchKeepDirection(classPx.lt(0)))).rename('class_mapped');
var nLossYears = confidenceSet.posN.updateMask(aoi_mask).rename('n_loss_years');
var nGainYears = confidenceSet.negN.updateMask(aoi_mask).rename('n_gain_years');
var unionLossPx = nLossYears.gte(1);
var unionGainPx = nGainYears.gte(1);
var unionLossMapped = unionLossPx.and(patchKeepDirection(unionLossPx));
var unionGainMapped = unionGainPx.and(patchKeepDirection(unionGainPx));

// uses the objects of the headline tables (Areas_Excess_by_Zclass at T,
// Cumulative_Confidence_Class_Areas 'mapped', Areas_Headline); the function
// statisticMasks rebuilds the same quantities from any median and annual z,
// so its main-parameter output must reproduce MAIN_STAT_MASKS exactly.
var SENS_STAT_NAMES = [
  'median_loss', 'median_gain',
  'year_2022_loss', 'year_2022_gain', 'year_2023_loss', 'year_2023_gain',
  'year_2024_loss', 'year_2024_gain',
  'dominant_loss_mapped', 'dominant_gain_mapped',
  'dominant_loss_ge2_mapped', 'dominant_gain_ge2_mapped',
  'union_loss_ge1_mapped', 'union_loss_ge2_mapped',
  'union_gain_ge1_mapped', 'union_gain_ge2_mapped',
  'overlap_loss_and_gain_mapped',
  'stricter_loss', 'stricter_gain'
];
function finishMasks(m) {
  var out = {};
  SENS_STAT_NAMES.forEach(function(k) {
    out[k] = m[k].unmask(0).gt(0).and(aoi_mask.unmask(0)).toByte();
  });
  return out;
}
var MAIN_STAT_MASKS = finishMasks({
  median_loss: resMedian.z_score.gte(T_EVIDENCE),
  median_gain: resMedian.z_score.lte(-T_EVIDENCE),
  year_2022_loss: resYearAnnualMedian['2022'].z_score.gte(T_EVIDENCE),
  year_2022_gain: resYearAnnualMedian['2022'].z_score.lte(-T_EVIDENCE),
  year_2023_loss: resYearAnnualMedian['2023'].z_score.gte(T_EVIDENCE),
  year_2023_gain: resYearAnnualMedian['2023'].z_score.lte(-T_EVIDENCE),
  year_2024_loss: resYearAnnualMedian['2024'].z_score.gte(T_EVIDENCE),
  year_2024_gain: resYearAnnualMedian['2024'].z_score.lte(-T_EVIDENCE),
  dominant_loss_mapped: classMapped.gt(0),
  dominant_gain_mapped: classMapped.lt(0),
  dominant_loss_ge2_mapped: classMapped.gte(2),
  dominant_gain_ge2_mapped: classMapped.lte(-2),
  union_loss_ge1_mapped: unionLossMapped,
  union_loss_ge2_mapped: nLossYears.gte(2).and(patchKeepDirection(nLossYears.gte(2))),
  union_gain_ge1_mapped: unionGainMapped,
  union_gain_ge2_mapped: nGainYears.gte(2).and(patchKeepDirection(nGainYears.gte(2))),
  overlap_loss_and_gain_mapped: unionLossMapped.and(unionGainMapped),
  stricter_loss: damage_primary,
  stricter_gain: rewild_primary
});

/** Same statistics from a median z and a {year: z} dictionary (v13b12). */
function statisticMasks(zMed, zYear) {
  var posN = ee.Image(0);
  var negN = ee.Image(0);
  var m = {
    median_loss: zMed.gte(T_EVIDENCE),
    median_gain: zMed.lte(-T_EVIDENCE)
  };
  YEARS.forEach(function(y) {
    m['year_' + y + '_loss'] = zYear[y].gte(T_EVIDENCE);
    m['year_' + y + '_gain'] = zYear[y].lte(-T_EVIDENCE);
    posN = posN.add(zYear[y].gte(T_EVIDENCE));
    negN = negN.add(zYear[y].lte(-T_EVIDENCE));
  });
  var lossWins = posN.gt(negN)
    .or(posN.eq(negN).and(posN.gt(0)).and(zMed.gte(0)));
  var gainWins = negN.gt(posN)
    .or(posN.eq(negN).and(negN.gt(0)).and(zMed.lt(0)));
  var cls = posN.multiply(lossWins).subtract(negN.multiply(gainWins))
    .toInt8().updateMask(aoi_mask);
  var clsMapped = cls.multiply(patchKeepDirection(cls.gt(0))
    .max(patchKeepDirection(cls.lt(0))));
  var nLoss = posN.toInt8().updateMask(aoi_mask);
  var nGain = negN.toInt8().updateMask(aoi_mask);
  var ul1 = nLoss.gte(1), ul2 = nLoss.gte(2);
  var ug1 = nGain.gte(1), ug2 = nGain.gte(2);
  m.dominant_loss_mapped = clsMapped.gt(0);
  m.dominant_gain_mapped = clsMapped.lt(0);
  m.dominant_loss_ge2_mapped = clsMapped.gte(2);
  m.dominant_gain_ge2_mapped = clsMapped.lte(-2);
  m.union_loss_ge1_mapped = ul1.and(patchKeepDirection(ul1));
  m.union_loss_ge2_mapped = ul2.and(patchKeepDirection(ul2));
  m.union_gain_ge1_mapped = ug1.and(patchKeepDirection(ug1));
  m.union_gain_ge2_mapped = ug2.and(patchKeepDirection(ug2));
  m.overlap_loss_and_gain_mapped = m.union_loss_ge1_mapped
    .and(m.union_gain_ge1_mapped);
  var det = detectChanges(zMed, Z_USED, ZN_USED, MIN_PATCH_PX);
  m.stricter_loss = det.damage;
  m.stricter_gain = det.rewild;
  return finishMasks(m);
}

// Direct one-run comparison of the two annual baseline representations. This
// avoids overwriting same-named files or manually calculating areas from six
// rasters. The selected main mode is identified in every row.
if (RUN_ANNUAL_BASELINE_SENSITIVITY && RUN.annualBaselineSensitivity) {
  var annualBaselineAreaRows = [];

  function addAnnualBaselineAreaRows(mode, detection, persistence) {
    YEARS.forEach(function(y) {
      var stats = twoClassAreaCount(
        detection.damage[y].and(aoi_mask),
        detection.rewild[y].and(aoi_mask));
      annualBaselineAreaRows.push(ee.Feature(null, {
        'annual_baseline_mode': mode,
        'is_primary_mode': mode === ANNUAL_BASELINE_MODE ? 1 : 0,
        'row_type': 'annual_detection',
        'year_or_period': y,
        'damage_km2': stats.get('damage_km2'),
        'rewild_km2': stats.get('rewild_km2'),
        'damage_px': stats.get('damage_px'),
        'rewild_px': stats.get('rewild_px'),
        'eligible_ground_area_km2': ELIGIBLE_GROUND_AREA_KM2,
        'eligible_px': ELIGIBLE_PIXEL_COUNT,
        'z_threshold': Z_USED,
        'z_neighbour_threshold': ZN_USED,
        'min_patch_px': MIN_PATCH_PX
      }));
    });
    var persistenceStats = twoClassAreaCount(
      persistence.damage.and(aoi_mask), persistence.rewild.and(aoi_mask));
    annualBaselineAreaRows.push(ee.Feature(null, {
      'annual_baseline_mode': mode,
      'is_primary_mode': mode === ANNUAL_BASELINE_MODE ? 1 : 0,
      'row_type': 'persistence_2_of_3',
      'year_or_period': '2022_2024',
      'damage_km2': persistenceStats.get('damage_km2'),
      'rewild_km2': persistenceStats.get('rewild_km2'),
      'damage_px': persistenceStats.get('damage_px'),
      'rewild_px': persistenceStats.get('rewild_px'),
      'eligible_ground_area_km2': ELIGIBLE_GROUND_AREA_KM2,
      'eligible_px': ELIGIBLE_PIXEL_COUNT,
      'z_threshold': Z_USED,
      'z_neighbour_threshold': ZN_USED,
      'min_patch_px': MIN_PATCH_PX
    }));
  }

  addAnnualBaselineAreaRows(
    'annual_median', detectionAnnualMedian, persistenceAnnualMedian);
  addAnnualBaselineAreaRows(
    'composite', detectionComposite, persistenceComposite);

  exportTable({
    collection: ee.FeatureCollection(annualBaselineAreaRows),
    description: 'AnnualBaselineSensitivity_Areas',
    folder: exportFolder,
    fileFormat: 'CSV'
  });

function groupedConfidenceAreas(mode, confidenceImage) {
    // Official GEE grouped-reduction pattern: band 0 is the quantity to sum
    // (true ground area), band 1 is the integer grouping class. This replaces
    // seven separate full-AOI reductions per mode with one reduction per mode.
    var grouped = groundPixelAreaKm2.updateMask(aoi_mask)
      .addBands(confidenceImage.rename('confidence_class'))
      .reduceRegion({
        reducer: ee.Reducer.sum().group({
          groupField: 1,
          groupName: 'confidence_class'
        }),
        geometry: roi,
        crs: refProj,
        scale: refScale,
        maxPixels: 1e13
      });
    return ee.FeatureCollection(ee.List(grouped.get('groups')).map(function(row) {
      row = ee.Dictionary(row);
      return ee.Feature(null, {
        'annual_baseline_mode': mode,
        'is_primary_mode': mode === ANNUAL_BASELINE_MODE ? 1 : 0,
        'confidence_class': row.get('confidence_class'),
        'area_km2': row.get('sum'),
        'eligible_ground_area_km2': ELIGIBLE_GROUND_AREA_KM2,
        'eligible_px': ELIGIBLE_PIXEL_COUNT
      });
    }));
  }
  var annualBaselineConfidenceRows = groupedConfidenceAreas(
    'annual_median', confidenceAnnualMedian).merge(
      groupedConfidenceAreas('composite', confidenceComposite));

  exportTable({
    collection: annualBaselineConfidenceRows,
    description: 'AnnualBaselineSensitivity_Confidence',
    folder: exportFolder,
    fileFormat: 'CSV'
  });
}

// =====================================================================
// =====================================================================

if (RUN.nullDiagnostics) {

  // 8a. Empirical distribution of the null pseudo-z, as a histogram: one
  // set of 120 bins per pre-war year (pool by summing counts per bin; the
  // downstream check already sums over rows). Guarded against a null
  // reducer result so this task can never hard-fail again.
  function nullHistogramForYear(i) {
    var y = NULL_MAIN.years[i];
    var band = 'pseudo_z_' + y;
    var h = NULL_MAIN.pseudoZ[i].updateMask(aoi_mask).reduceRegion({
      reducer: ee.Reducer.fixedHistogram({min: -6, max: 6, steps: 120}),
      geometry: roi, scale: refScale, maxPixels: 1e10, bestEffort: true
    });
    var values = h.get(band);
    var lst = ee.List(ee.Algorithms.If(
      values, ee.Array(values).toList(), ee.List([])));
    return ee.FeatureCollection(lst.map(function(row) {
      row = ee.List(row);
      return ee.Feature(null, {
        'bin_centre': row.get(0), 'count': row.get(1), 'null_year': y});
    }));
  }
  var histFC = nullHistogramForYear(0);
  for (var hi = 1; hi < NULL_MAIN.years.length; hi++) {
    histFC = histFC.merge(nullHistogramForYear(hi));
  }
  exportTable({
    collection: histFC,
    description: 'ControlZ_Histogram',
    folder: exportFolder, fileFormat: 'CSV',
    selectors: ['bin_centre', 'count', 'null_year']
  });

  // 8b. The pooled null pseudo-z sample with coordinates, for the QQ plot,
  // the empirical FDR and spatial-dependence diagnostics in Python_02.
  // Column names are unchanged from the control-zone version so the support
  // scripts run as-is; null_year is appended for per-year checks.
  exportTable({
    collection: nullZSample,
    description: 'ControlZ_Sample',
    folder: exportFolder, fileFormat: 'CSV',
    selectors: ['zscore_median', 'zscore_raw_median', 'in_ordlo',
                'lon', 'lat', 'null_year']
  });

  // 8c. Observed flagged fraction against the null exceedance: the fraction
  // of pre-war pseudo-z pixels beyond the working threshold is the empirical
  // false-positive rate of the per-pixel rule (before the neighbourhood and
  // patch filters).
  var nEligible = ELIGIBLE_PIXEL_COUNT;
  var nDamagePrimary = pixelCount(damage_primary);
  var nRewildPrimary = pixelCount(rewild_primary);
  var nNullPx = NULL_VALID_PX;
  var nNullOverThreshold = ee.Number(0);
  NULL_MAIN.pseudoZ.forEach(function(pz) {
    nNullOverThreshold = nNullOverThreshold.add(pixelCount(
      pz.abs().gte(ee.Image.constant(Z_USED)).and(aoi_mask)));
  });
  var flaggedFC = ee.FeatureCollection([ee.Feature(null, {
    'threshold_mode': THRESHOLD_MODE,
    'z_threshold_used': Z_USED,
    'z_threshold_empirical_pctl': EMPIRICAL_PCTL,
    'z_threshold_empirical_value': zThreshEmpirical,
    'global_recalibration': APPLY_GLOBAL_RECAL ? 1 : 0,
    'null_definition': 'temporal_prewar_loo_pseudo_z_v13b8',
    'n_eligible_px': nEligible,
    'n_damage_px': nDamagePrimary,
    'n_rewild_px': nRewildPrimary,
    'frac_damage': nDamagePrimary.divide(nEligible),
    'frac_rewild': nRewildPrimary.divide(nEligible),
    'n_null_px': nNullPx,
    'n_null_px_over_thresh': nNullOverThreshold,
    'prewar_share_two_sided': nNullOverThreshold.divide(nNullPx.max(1))
  })]);
  exportTable({
    collection: flaggedFC,
    description: 'Detection_FlaggedFraction',
    folder: exportFolder, fileFormat: 'CSV'
  });
}

// =====================================================================
// =====================================================================

// 9a. The first row is the primary all-valid-input mapping domain. The next
// two rows are quality-screen sensitivities, not alternative primary maps.
if (RUN.mainOutputs) {
  var headlineAllValidStats = twoClassAreaCount(
    damage_primary.and(aoi_mask), rewild_primary.and(aoi_mask));
  var headlineNovelStats = twoClassAreaCount(
    damage_primary.and(eligible_novel), rewild_primary.and(eligible_novel));
  var headlineNoNovelStats = twoClassAreaCount(
    damage_primary.and(eligible_nonovel),
    rewild_primary.and(eligible_nonovel));
  var detNoClosing = detectChanges(resMedian.z_score, Z_USED, ZN_USED,
                                   MIN_PATCH_PX, false);
  var headlineNoClosingStats = twoClassAreaCount(
    detNoClosing.damage.and(aoi_mask), detNoClosing.rewild.and(aoi_mask));
  // stricter z threshold (z >= Z_USED / z <= -Z_USED): cells that were
  // exceedances, cells added by the closing, and exceedances removed.
  var excLossMain = resMedian.z_score.gte(ee.Image.constant(Z_USED))
    .unmask(0).and(aoi_mask.unmask(0));
  var excGainMain = resMedian.z_score.lte(ee.Image.constant(Z_USED).multiply(-1))
    .unmask(0).and(aoi_mask.unmask(0));
  var splitMain = sumAreasKm2({
    loss_exceedance: excLossMain,
    loss_in_exceedance: damage_primary.and(excLossMain),
    loss_added: damage_primary.and(excLossMain.not()),
    loss_removed: excLossMain.and(damage_primary.not()),
    gain_exceedance: excGainMain,
    gain_in_exceedance: rewild_primary.and(excGainMain),
    gain_added: rewild_primary.and(excGainMain.not()),
    gain_removed: excGainMain.and(rewild_primary.not())
  });
  var headline = ee.FeatureCollection([
    ee.Feature(null, {
      'analysis_scheme': CHANGE_SCHEME,
      'analysis_scheme_label': CHANGE_SCHEME_LABEL,
      'variant': 'primary_all_valid_input_pixels',
      'annual_baseline_mode': ANNUAL_BASELINE_MODE,
      'is_primary_domain': 1,
      'worldcover_mask_applied': 0,
      'probability_interior_required': 0,
      'wartime_sd_required': 0,
      'novelty_required': 0,
      'damage_km2': headlineAllValidStats.get('damage_km2'),
      'rewild_km2': headlineAllValidStats.get('rewild_km2'),
      'damage_px': headlineAllValidStats.get('damage_px'),
      'rewild_px': headlineAllValidStats.get('rewild_px'),
      'damage_persist_km2': areaKm2(damagePersist.and(aoi_mask)),
      'rewild_persist_km2': areaKm2(rewildPersist.and(aoi_mask)),
      'damage_near_war_km2': areaKm2(damageAttrib.and(aoi_mask)),
      'rewild_near_war_km2': areaKm2(rewildAttrib.and(aoi_mask)),
      'headline_comparison': 'annual_median_vs_annual_median',
      'z_null_sd_factor': MEDIAN_NULL_SD_FACTOR,
      // the stricter z threshold before any rule.
      'damage_exceedance_km2': splitMain.get('loss_exceedance'),
      'damage_in_exceedance_km2': splitMain.get('loss_in_exceedance'),
      'damage_added_by_closing_km2': splitMain.get('loss_added'),
      'damage_exceedance_removed_km2': splitMain.get('loss_removed'),
      'rewild_exceedance_km2': splitMain.get('gain_exceedance'),
      'rewild_in_exceedance_km2': splitMain.get('gain_in_exceedance'),
      'rewild_added_by_closing_km2': splitMain.get('gain_added'),
      'rewild_exceedance_removed_km2': splitMain.get('gain_removed'),
      'stricter_z_threshold': Z_USED,
      'stricter_neighbour_threshold': ZN_USED,
      'closing_applied': APPLY_CLOSING ? 1 : 0
    }),
    ee.Feature(null, {
      'variant': 'composite_representation_sensitivity',
      'annual_baseline_mode': 'not_applicable',
      'is_primary_domain': 0,
      'worldcover_mask_applied': 0,
      'probability_interior_required': 0,
      'wartime_sd_required': 0,
      'novelty_required': 0,
      'damage_km2': areaKm2(detComposite.damage.and(aoi_mask)),
      'rewild_km2': areaKm2(detComposite.rewild.and(aoi_mask)),
      'damage_px': pixelCount(detComposite.damage.and(aoi_mask)),
      'rewild_px': pixelCount(detComposite.rewild.and(aoi_mask)),
      'headline_comparison': 'composite_vs_composite_conservative',
      'z_null_sd_factor': 1
    }),
    ee.Feature(null, {
      'variant': 'quality_screen_with_novelty_sensitivity',
      'annual_baseline_mode': ANNUAL_BASELINE_MODE,
      'is_primary_domain': 0,
      'worldcover_mask_applied': 0,
      'probability_interior_required': 1,
      'wartime_sd_required': 1,
      'novelty_required': 1,
      'damage_km2': headlineNovelStats.get('damage_km2'),
      'rewild_km2': headlineNovelStats.get('rewild_km2'),
      'damage_px': headlineNovelStats.get('damage_px'),
      'rewild_px': headlineNovelStats.get('rewild_px'),
      'damage_persist_km2': areaKm2(damagePersist.and(eligible_novel)),
      'rewild_persist_km2': areaKm2(rewildPersist.and(eligible_novel)),
      'damage_near_war_km2': areaKm2(damageAttrib.and(eligible_novel)),
      'rewild_near_war_km2': areaKm2(rewildAttrib.and(eligible_novel))
    }),
    ee.Feature(null, {
      'variant': 'quality_screen_without_novelty_sensitivity',
      'annual_baseline_mode': ANNUAL_BASELINE_MODE,
      'is_primary_domain': 0,
      'worldcover_mask_applied': 0,
      'probability_interior_required': 1,
      'wartime_sd_required': 1,
      'novelty_required': 0,
      'damage_km2': headlineNoNovelStats.get('damage_km2'),
      'rewild_km2': headlineNoNovelStats.get('rewild_km2'),
      'damage_px': headlineNoNovelStats.get('damage_px'),
      'rewild_px': headlineNoNovelStats.get('rewild_px'),
      'damage_persist_km2': areaKm2(damagePersist.and(eligible_nonovel)),
      'rewild_persist_km2': areaKm2(rewildPersist.and(eligible_nonovel)),
      'damage_near_war_km2': areaKm2(damageAttrib.and(eligible_nonovel)),
      'rewild_near_war_km2': areaKm2(rewildAttrib.and(eligible_nonovel))
    }),
    ee.Feature(null, {
      'variant': 'stricter_no_closing',
      'annual_baseline_mode': ANNUAL_BASELINE_MODE,
      'is_primary_domain': 1,
      'worldcover_mask_applied': 0,
      'probability_interior_required': 0,
      'wartime_sd_required': 0,
      'novelty_required': 0,
      'damage_km2': headlineNoClosingStats.get('damage_km2'),
      'rewild_km2': headlineNoClosingStats.get('rewild_km2'),
      'damage_px': headlineNoClosingStats.get('damage_px'),
      'rewild_px': headlineNoClosingStats.get('rewild_px')
    })
  ]);
  exportTable({
    collection: headline,
    description: 'Areas_Headline_' + ANNUAL_OUTPUT_TAG,
    folder: exportFolder, fileFormat: 'CSV'
  });
}

// columns). For each z class and direction: the detected area (pixels at or
// beyond the class, no neighbour/patch rule) and the PRE-WAR BENCHMARK, i.e.
// the share of pre-war leave-one-out z-scores at or beyond the class (pooled
// over the three pre-war maps) x the analysed area. The same benchmark share
// is used for every comparison; because the pre-war contrasts and the
// wartime comparisons have different temporal definitions, it is not a
// calibrated no-war expectation, and benchmark/detected is a reference-area
// ratio, not a false discovery rate.
if (RUN.mainOutputs) {
  var nullPosCount = {};
  var nullNegCount = {};
  Z_CLASS_LIST.forEach(function(t) {
    var p = ee.Number(0);
    var q = ee.Number(0);
    NULL_MAIN.pseudoZ.forEach(function(pz) {
      p = p.add(pixelCount(pz.gte(t).and(aoi_mask)));
      q = q.add(pixelCount(pz.lte(-t).and(aoi_mask)));
    });
    nullPosCount[t] = p;
    nullNegCount[t] = q;
  });
  var nNullTotal = NULL_VALID_PX;

  var excessComparisons = [
    {tag: 'median_headline', z: resMedian.z_score,
     factor: MEDIAN_NULL_SD_FACTOR},
    {tag: 'composite_sensitivity', z: resComposite.z_score, factor: 1},
    {tag: 'year_2022', z: resYearAnnualMedian['2022'].z_score,
     factor: PERYEAR_NULL_SD_FACTOR},
    {tag: 'year_2023', z: resYearAnnualMedian['2023'].z_score,
     factor: PERYEAR_NULL_SD_FACTOR},
    {tag: 'year_2024', z: resYearAnnualMedian['2024'].z_score,
     factor: PERYEAR_NULL_SD_FACTOR}
  ];
  var excessRows = [];
  excessComparisons.forEach(function(cmp) {
    Z_CLASS_LIST.forEach(function(t) {
      var shareLoss = nullPosCount[t].divide(nNullTotal.max(1));
      var shareGain = nullNegCount[t].divide(nNullTotal.max(1));
      var obsLoss = areaKm2(cmp.z.gte(t).and(aoi_mask));
      var obsGain = areaKm2(cmp.z.lte(-t).and(aoi_mask));
      var benchLoss = shareLoss.multiply(ELIGIBLE_GROUND_AREA_KM2);
      var benchGain = shareGain.multiply(ELIGIBLE_GROUND_AREA_KM2);
      // detected area gives NA with status 'no_detections').
      excessRows.push(ee.Feature(null, {
        'comparison': cmp.tag,
        'z_sd_factor': cmp.factor,
        'z_class': t,
        'observed_loss_km2': obsLoss,
        'prewar_benchmark_loss_km2': benchLoss,
        'status_loss': detectionStatus(obsLoss),
        'observed_gain_km2': obsGain,
        'prewar_benchmark_gain_km2': benchGain,
        'status_gain': detectionStatus(obsGain),
        'prewar_share_loss': shareLoss,
        'prewar_share_gain': shareGain,
        'eligible_ground_area_km2': ELIGIBLE_GROUND_AREA_KM2,
        'note': 'pixel exceedance (>=); prewar_benchmark = share of pre-war ' +
                'leave-one-out z-scores >= class x analysed area; a benchmark, ' +
                'not a calibrated no-war expectation or FDR'
      }));
    });
  });
  exportTable({
    collection: ee.FeatureCollection(excessRows),
    description: 'Areas_Excess_by_Zclass',
    folder: exportFolder, fileFormat: 'CSV',
    selectors: ['comparison', 'z_sd_factor', 'z_class',
                'observed_loss_km2', 'prewar_benchmark_loss_km2', 'status_loss',
                'observed_gain_km2', 'prewar_benchmark_gain_km2', 'status_gain',
                'prewar_share_loss', 'prewar_share_gain',
                'eligible_ground_area_km2', 'note']
  });
}

if (RUN.areaSensitivity) {
  var rows = [];
  var nNullPxForSensitivity = NULL_VALID_PX;
  Z_THRESH_LIST.forEach(function(zt) {
    var nullOverCount = ee.Number(0);
    NULL_MAIN.pseudoZ.forEach(function(pz) {
      nullOverCount = nullOverCount.add(pixelCount(
        pz.abs().gte(zt).and(aoi_mask)));
    });
    var fracNullOverThreshold =
      nullOverCount.divide(nNullPxForSensitivity);
    MIN_PATCH_LIST.forEach(function(mp) {
      // areas counted inside the mapping mask.
      var isMain = (zt === Z_THRESH && mp === MIN_PATCH_PX &&
                    THRESHOLD_MODE === 'normal');
      var det = isMain
        ? detectChanges(resMedian.z_score, Z_USED, ZN_USED, MIN_PATCH_PX)
        : detectChanges(resMedian.z_score, zt, zt * (Z_NEIGHBOR / Z_THRESH), mp);
      var detStats = twoClassAreaCount(det.damage.and(aoi_mask),
                                       det.rewild.and(aoi_mask));
      rows.push(ee.Feature(null, {
        'z_threshold': zt,
        'z_neighbour': zt * (Z_NEIGHBOR / Z_THRESH),
        'min_patch_px': mp,
        'is_main_parameters': isMain ? 1 : 0,
        'nominal_projected_patch_km2':
          mp * NOMINAL_PROJECTED_PIXEL_KM2,
        'damage_km2': detStats.get('damage_km2'),
        'rewild_km2': detStats.get('rewild_km2'),
        'damage_px': detStats.get('damage_px'),
        'rewild_px': detStats.get('rewild_px'),
        'prewar_share_two_sided': fracNullOverThreshold
      }));
    });
  });
  exportTable({
    collection: ee.FeatureCollection(rows),
    description: 'Areas_vs_Threshold_and_PatchSize',
    folder: exportFolder, fileFormat: 'CSV',
    selectors: ['z_threshold', 'z_neighbour', 'min_patch_px',
                'is_main_parameters',
                'nominal_projected_patch_km2', 'damage_km2', 'rewild_km2',
                'damage_px', 'rewild_px', 'prewar_share_two_sided']
  });
}

// Uses the primary detection masks unchanged; only the proximity radius that
// links a detected patch to documented events varies.
if (RUN.areaSensitivity) {
  var attribRows = ATTRIBUTION_DIST_LIST.map(function(ak) {
    var near = distControlKm.lte(ak);
    return ee.Feature(null, {
      'attribution_dist_km': ak,
      'is_primary_value': ak === ATTRIBUTION_DIST_KM ? 1 : 0,
      'damage_attributed_km2': areaKm2(
        damage_primary.and(near).and(aoi_mask)),
      'rewild_attributed_km2': areaKm2(
        rewild_primary.and(near).and(aoi_mask)),
      'damage_km2_total': areaKm2(damage_primary.and(aoi_mask)),
      'rewild_km2_total': areaKm2(rewild_primary.and(aoi_mask)),
      'fraction_eligible_within_radius': pixelCount(
        near.and(aoi_mask)).divide(ELIGIBLE_PIXEL_COUNT)
    });
  });
  exportTable({
    collection: ee.FeatureCollection(attribRows),
    description: 'Areas_vs_AttributionDistance',
    folder: exportFolder, fileFormat: 'CSV',
    selectors: ['attribution_dist_km', 'is_primary_value',
                'damage_attributed_km2', 'rewild_attributed_km2',
                'damage_km2_total', 'rewild_km2_total',
                'fraction_eligible_within_radius']
  });
}

// Families: main parameters (acceptance check), Gaussian sigma of the
// reference scale, dropped pre-war year (two designs) and SD floor. For each
// variant the median, annual (at T), year-count and stricter areas are
// recomputed with SD factors matched to that variant's reference years and
// baseline (comparisonSdFactor), and every area is reported with the
// main-parameter area of the same statistic and the area of their overlap
// (same pixels in both). Areas only: Python computes the ratio to the main
// area and the Jaccard index (NA with a status when undefined).
// Dropped-year designs:
//   'variance_only': baseline stays the 2019-2021 median; only the reference
//     variance comes from the two kept years (factors 0.6698 median, 0.8511
//     annual; floor converted to the two-year scale).
//   'baseline_and_variance': the year is also removed from the baseline
//     (mean of the two kept years; factors 0.6887 median, 0.8660 annual).
// reference, which shrank the z-scores by sqrt(1.5 / 2) = 0.87.
if (ANNUAL_BASELINE_MODE !== 'annual_median') {
  throw new Error('The sensitivity tables assume ANNUAL_BASELINE_MODE = ' +
                  'annual_median (the main analysis).');
}

function extend(a, b) {
  var o = {};
  for (var k in a) { o[k] = a[k]; }
  for (var j in b) { o[j] = b[j]; }
  return o;
}

/** Median and annual z of one variant, factors matched to its design. */
function variantZ(nullFields, baseline, baselineVariance, nYears, tag) {
  var fMed = comparisonSdFactor(baselineVariance, WARTIME_VARIANCE.median, nYears);
  var fYear = comparisonSdFactor(baselineVariance, WARTIME_VARIANCE.year, nYears);
  var opts = {baseline: baseline, nullFields: nullFields};
  var zMed = correctPeriod(warAnnualMedian, 'med_' + tag,
                           extend(opts, {sdFactor: fMed})).z_score;
  var zYear = {};
  YEARS.forEach(function(y) {
    zYear[y] = correctPeriod(probYear[y], y + '_' + tag,
                             extend(opts, {sdFactor: fYear})).z_score;
  });
  return {zMed: zMed, zYear: zYear, fMed: fMed, fYear: fYear};
}

/** Rows (one per statistic) of one variant; meta = descriptive columns. */
function sensitivityRows(vz, meta) {
  var vm = statisticMasks(vz.zMed, vz.zYear);
  var bands = {};
  SENS_STAT_NAMES.forEach(function(st) {
    bands['v__' + st] = vm[st];
    bands['m__' + st] = MAIN_STAT_MASKS[st];
    bands['o__' + st] = vm[st].and(MAIN_STAT_MASKS[st]);
  });
  var sums = sumAreasKm2(bands);
  return SENS_STAT_NAMES.map(function(st) {
    return ee.Feature(null, extend(meta, {
      'statistic': st,
      'direction': st.indexOf('overlap') === 0 ? 'both'
        : (st.indexOf('loss') >= 0 ? 'loss' : 'gain'),
      'variant_km2': sums.get('v__' + st),
      'main_km2': sums.get('m__' + st),
      'overlap_with_main_km2': sums.get('o__' + st),
      'variant_status': detectionStatus(sums.get('v__' + st)),
      'main_status': detectionStatus(sums.get('m__' + st)),
      'z_sd_factor_median': vz.fMed,
      'z_sd_factor_annual': vz.fYear,
      'evidence_z_threshold': T_EVIDENCE,
      'stricter_z_threshold': Z_USED,
      'stricter_neighbour_threshold': ZN_USED,
      'min_patch_px': MIN_PATCH_PX,
      'eligible_ground_area_km2': ELIGIBLE_GROUND_AREA_KM2
    }));
  });
}

function nullMoments(nf, n) {
  return sampleNullZ(nf, n, null).reduceColumns(
    ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true),
    ['zscore_median']);
}

var SENS_SELECTORS = [
  'family', 'variant', 'design', 'is_main_parameters', 'statistic',
  'direction', 'variant_km2', 'main_km2', 'overlap_with_main_km2',
  'variant_status', 'main_status', 'bias_sigma_m', 'dropped_prewar_year',
  'reference_years', 'baseline', 'sd_floor', 'z_sd_factor_median',
  'z_sd_factor_annual', 'evidence_z_threshold', 'stricter_z_threshold',
  'stricter_neighbour_threshold', 'min_patch_px', 'z_null_mean', 'z_null_sd',
  'eligible_ground_area_km2'
];
function exportSensitivity(name, rows) {
  exportTable({
    collection: ee.FeatureCollection(rows),
    description: 'Sensitivity_' + name + OUT_SFX,
    folder: exportFolder, fileFormat: 'CSV', selectors: SENS_SELECTORS
  });
}

// Main parameters: must reproduce the headline areas exactly.
if (RUN.mainSensitivityRow) {
  exportSensitivity('main_main', sensitivityRows(
    variantZ(NULL_MAIN, prewarAnnualMedian, MEDIAN3_VARIANCE_FACTOR, 3, 'main'),
    {'family': 'main', 'variant': 'main', 'design': 'main',
     'is_main_parameters': 1, 'bias_sigma_m': BIAS_SIGMA_M,
     'dropped_prewar_year': 'none', 'reference_years': PREWAR_YEARS.join('_'),
     'baseline': 'median_2019_2021', 'sd_floor': NULL_MAIN.floorSd}));
}

// Gaussian sigma of the reference scale. One task per value.
if (RUN.sigmaSensitivity) {
  BIAS_SIGMA_LIST.forEach(function(bs) {
    if (bs === BIAS_SIGMA_M) { return; }  // = main parameters
    var nf = buildTemporalNull(bs, SMOOTH_SIGMA_M, null);
    var mom = nullMoments(nf, 7000);
    exportSensitivity('bias_sigma_' + bs + 'm', sensitivityRows(
      variantZ(nf, prewarAnnualMedian, MEDIAN3_VARIANCE_FACTOR, 3, 'sig' + bs),
      {'family': 'bias_sigma', 'variant': bs + 'm', 'design': 'three_years',
       'is_main_parameters': 0, 'bias_sigma_m': bs,
       'dropped_prewar_year': 'none', 'reference_years': PREWAR_YEARS.join('_'),
       'baseline': 'median_2019_2021', 'sd_floor': nf.floorSd,
       'z_null_mean': mom.get('mean'), 'z_null_sd': mom.get('stdDev')}));
  });
}

// Dropped pre-war year, two designs per task. One task per dropped year.
if (RUN.nullYearSensitivity) {
  PREWAR_YEARS.forEach(function(dropYear) {
    var kept = PREWAR_YEARS.filter(function(y) { return y !== dropYear; });
    var nf = buildTemporalNull(BIAS_SIGMA_M, SMOOTH_SIGMA_M, kept);
    var keptBaseline = annualProbability
      .select(kept.map(function(y) { return 'prob_' + y; }))
      .reduce(ee.Reducer.mean()).rename('prob_prewar_kept_mean');
    var mom = nullMoments(nf, 10000);
    var meta = {'family': 'dropped_year', 'is_main_parameters': 0,
                'bias_sigma_m': BIAS_SIGMA_M, 'dropped_prewar_year': dropYear,
                'reference_years': kept.join('_'), 'sd_floor': nf.floorSd,
                'z_null_mean': mom.get('mean'), 'z_null_sd': mom.get('stdDev')};
    var rowsVar = sensitivityRows(
      variantZ(nf, prewarAnnualMedian, MEDIAN3_VARIANCE_FACTOR, kept.length,
               'drop' + dropYear + 'v'),
      extend(meta, {'variant': 'without_' + dropYear, 'design': 'variance_only',
                    'baseline': 'median_2019_2021'}));
    var rowsBase = sensitivityRows(
      variantZ(nf, keptBaseline, 1 / kept.length, kept.length,
               'drop' + dropYear + 'b'),
      extend(meta, {'variant': 'without_' + dropYear,
                    'design': 'baseline_and_variance',
                    'baseline': 'mean_' + kept.join('_')}));
    exportSensitivity('dropped_year_' + dropYear, rowsVar.concat(rowsBase));
  });
}

// SD floor of the unadjusted reference SD (main 0.02). One task per value.
if (RUN.sdFloorSensitivity) {
  MIN_LOCAL_SD_LIST.forEach(function(floorValue) {
    var nfFloor = {bias: NULL_MAIN.bias,
                   localSD: NULL_MAIN.localSDRaw.max(floorValue)};
    exportSensitivity('sd_floor_' + String(floorValue).replace('.', 'p'),
      sensitivityRows(
        // Band names may not contain '.', so 0.01 -> 'floor0p01'.
        variantZ(nfFloor, prewarAnnualMedian, MEDIAN3_VARIANCE_FACTOR, 3,
                 'floor' + String(floorValue).replace('.', 'p')),
        {'family': 'sd_floor', 'variant': String(floorValue),
         'design': 'three_years', 'is_main_parameters': 0,
         'bias_sigma_m': BIAS_SIGMA_M, 'dropped_prewar_year': 'none',
         'reference_years': PREWAR_YEARS.join('_'),
         'baseline': 'median_2019_2021', 'sd_floor': floorValue}));
  });
}

// (benchmark). The stricter rules are applied to each pre-war leave-one-out
// z map with the same function and domain as the median comparison. Beside
// the result the table gives the pixel exceedance at the same threshold and
// the split of the result into cells that were exceedances
// (stricter_*_in_exceedance_km2), cells added by the closing
// (stricter_*_added_km2) and exceedances removed by the rules
// (pixel_*_removed_km2), plus the result without the closing (a subset of
// the exceedances). The closing can add cells, so stricter / pixel is an area
// ratio, not a retained share; the retained share is
// stricter_*_in_exceedance_km2 / pixel_*_km2. Like the other pre-war
// benchmarks (section 9a-bis), this is not a calibrated no-war expectation
// for the median comparison.
if (RUN.strictNull) {
  var strictRows = [];
  NULL_MAIN.pseudoZ.forEach(function(pz, i) {
    var det = detectChanges(pz, Z_USED, ZN_USED, MIN_PATCH_PX);
    var detNc = detectChanges(pz, Z_USED, ZN_USED, MIN_PATCH_PX, false);
    var dom01 = aoi_mask.unmask(0);
    var excL = pz.gte(ee.Image.constant(Z_USED)).unmask(0).and(dom01);
    var excG = pz.lte(ee.Image.constant(Z_USED).multiply(-1)).unmask(0).and(dom01);
    var a = sumAreasKm2({
      stricter_loss: det.damage, stricter_gain: det.rewild,
      stricter_loss_screened: det.damage.and(eligible_novel.unmask(0)),
      stricter_gain_screened: det.rewild.and(eligible_novel.unmask(0)),
      pixel_loss: excL, pixel_gain: excG,
      loss_in: det.damage.and(excL), gain_in: det.rewild.and(excG),
      loss_added: det.damage.and(excL.not()),
      gain_added: det.rewild.and(excG.not()),
      loss_removed: excL.and(det.damage.not()),
      gain_removed: excG.and(det.rewild.not()),
      nc_loss: detNc.damage, nc_gain: detNc.rewild
    });
    strictRows.push(ee.Feature(null, {
      'null_year': NULL_MAIN.years[i],
      'z_threshold': Z_USED,
      'z_neighbour': ZN_USED,
      'min_patch_px': MIN_PATCH_PX,
      'closing_applied': APPLY_CLOSING ? 1 : 0,
      'stricter_loss_km2': a.get('stricter_loss'),
      'stricter_gain_km2': a.get('stricter_gain'),
      'stricter_loss_screened_km2': a.get('stricter_loss_screened'),
      'stricter_gain_screened_km2': a.get('stricter_gain_screened'),
      'pixel_loss_km2': a.get('pixel_loss'),
      'pixel_gain_km2': a.get('pixel_gain'),
      'stricter_loss_in_exceedance_km2': a.get('loss_in'),
      'stricter_gain_in_exceedance_km2': a.get('gain_in'),
      'stricter_loss_added_km2': a.get('loss_added'),
      'stricter_gain_added_km2': a.get('gain_added'),
      'pixel_loss_removed_km2': a.get('loss_removed'),
      'pixel_gain_removed_km2': a.get('gain_removed'),
      'stricter_no_closing_loss_km2': a.get('nc_loss'),
      'stricter_no_closing_gain_km2': a.get('nc_gain'),
      'eligible_ground_area_km2': ELIGIBLE_GROUND_AREA_KM2
    }));
  });
  exportTable({
    collection: ee.FeatureCollection(strictRows),
    description: 'Areas_Stricter_Null' + OUT_SFX,
    folder: exportFolder, fileFormat: 'CSV',
    selectors: ['null_year', 'z_threshold', 'z_neighbour', 'min_patch_px',
                'closing_applied', 'stricter_loss_km2', 'stricter_gain_km2',
                'stricter_loss_screened_km2', 'stricter_gain_screened_km2',
                'pixel_loss_km2', 'pixel_gain_km2',
                'stricter_loss_in_exceedance_km2',
                'stricter_gain_in_exceedance_km2',
                'stricter_loss_added_km2', 'stricter_gain_added_km2',
                'pixel_loss_removed_km2', 'pixel_gain_removed_km2',
                'stricter_no_closing_loss_km2', 'stricter_no_closing_gain_km2',
                'eligible_ground_area_km2']
  });
}

// Areas (km2) of the reported statistics: all pixels, each screen alone
// (wartime SD of annual predictions <= MAX_WAR_SD; smoothed 2019-2021
// annual-median propensity in (PROB_LO, PROB_HI); no annual embedding
// 2019-2024 flagged), all three together, and pixels where the unadjusted
// reference SD is at the floor.
var STAT_MASKS = {
  median_loss: resMedian.z_score.gte(T_EVIDENCE),
  median_gain: resMedian.z_score.lte(-T_EVIDENCE),
  year_2022_loss: resYearAnnualMedian['2022'].z_score.gte(T_EVIDENCE),
  year_2022_gain: resYearAnnualMedian['2022'].z_score.lte(-T_EVIDENCE),
  year_2023_loss: resYearAnnualMedian['2023'].z_score.gte(T_EVIDENCE),
  year_2023_gain: resYearAnnualMedian['2023'].z_score.lte(-T_EVIDENCE),
  year_2024_loss: resYearAnnualMedian['2024'].z_score.gte(T_EVIDENCE),
  year_2024_gain: resYearAnnualMedian['2024'].z_score.lte(-T_EVIDENCE),
  class_loss_mapped: classMapped.gt(0),
  class_gain_mapped: classMapped.lt(0),
  class_loss2_mapped: classMapped.gte(2),
  union_loss_mapped: unionLossMapped,
  union_gain_mapped: unionGainMapped,
  stricter_loss: damage_primary,
  stricter_gain: rewild_primary
};
var STAT_NAMES = Object.keys(STAT_MASKS);
var atFloor = NULL_MAIN.localSDRaw.lt(NULL_MAIN.floorSd);
var SCREEN_MASKS = {
  all_pixels: aoi_mask,
  wartime_sd: aoi_mask.and(wartimeSdAccepted),
  interior_propensity: aoi_mask.and(probabilityInterior),
  no_annual_extrapolation: aoi_mask.and(noveltyAccepted),
  all_three_screens: aoi_mask.and(wartimeSdAccepted)
    .and(probabilityInterior).and(noveltyAccepted),
  at_sd_floor: aoi_mask.and(atFloor)
};
var SCREEN_NAMES = Object.keys(SCREEN_MASKS);

if (RUN.screens || RUN.sdFloorDiagnostics) {
  var bands = [];
  STAT_NAMES.forEach(function(st) {
    SCREEN_NAMES.forEach(function(sc) {
      bands.push(STAT_MASKS[st].unmask(0).and(SCREEN_MASKS[sc].unmask(0))
        .multiply(groundPixelAreaKm2).rename(st + '__' + sc));
    });
  });
  SCREEN_NAMES.forEach(function(sc) {
    bands.push(SCREEN_MASKS[sc].unmask(0).multiply(groundPixelAreaKm2)
      .rename('domain__' + sc));
  });
  var screenSums = ee.Image.cat(bands).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: roi, crs: refProj, scale: refScale,
    maxPixels: 1e13
  });
  var screenRows = ['domain'].concat(STAT_NAMES).map(function(st) {
    var props = {'analysis_scheme': CHANGE_SCHEME, 'statistic': st,
                 'evidence_z_threshold': T_EVIDENCE};
    SCREEN_NAMES.forEach(function(sc) {
      props[sc + '_km2'] = screenSums.get(st + '__' + sc);
    });
    return ee.Feature(null, props);
  });
  exportTable({
    collection: ee.FeatureCollection(screenRows),
    description: 'Areas_by_Screen' + OUT_SFX,
    folder: exportFolder, fileFormat: 'CSV',
    selectors: ['analysis_scheme', 'statistic', 'evidence_z_threshold']
      .concat(SCREEN_NAMES.map(function(sc) { return sc + '_km2'; }))
  });
}

if (RUN.sdFloorDiagnostics) {
  var sdPct = NULL_MAIN.localSDRaw.updateMask(aoi_mask).reduceRegion({
    reducer: ee.Reducer.percentile([0, 1, 5, 25, 50, 75, 95, 99, 100]),
    geometry: roi, crs: refProj, scale: refScale, maxPixels: 1e13
  });
  var nFloorPx = pixelCount(atFloor.and(aoi_mask));
  exportTable({
    collection: ee.FeatureCollection([ee.Feature(null, {
      'analysis_scheme': CHANGE_SCHEME,
      'sd_floor': NULL_MAIN.floorSd,
      'n_pixels_at_floor': nFloorPx,
      'share_pixels_at_floor': nFloorPx.divide(ELIGIBLE_PIXEL_COUNT),
      'area_at_floor_km2': areaKm2(atFloor.and(aoi_mask)),
      'sd_unfloored_p0': sdPct.get('local_sd_unfloored_p0'),
      'sd_unfloored_p1': sdPct.get('local_sd_unfloored_p1'),
      'sd_unfloored_p5': sdPct.get('local_sd_unfloored_p5'),
      'sd_unfloored_p25': sdPct.get('local_sd_unfloored_p25'),
      'sd_unfloored_p50': sdPct.get('local_sd_unfloored_p50'),
      'sd_unfloored_p75': sdPct.get('local_sd_unfloored_p75'),
      'sd_unfloored_p95': sdPct.get('local_sd_unfloored_p95'),
      'sd_unfloored_p99': sdPct.get('local_sd_unfloored_p99'),
      'sd_unfloored_p100': sdPct.get('local_sd_unfloored_p100'),
      'note': 'anomaly areas at the floor: Areas_by_Screen_v13b12, column at_sd_floor_km2'
    })]),
    description: 'SD_Floor_Diagnostics' + OUT_SFX,
    folder: exportFolder, fileFormat: 'CSV'
  });
}

// Descriptive comparison. The pre-war z-scores are locally normalised, so a
// similar spread on both sides does not by itself establish temporal
// stability of either side.
if (RUN.ordloSplit) {
  // CONTROL_EXCLUDE_ORDLO is true, so `controlMask.and(ordloMask)` is empty by
  // construction and the 'inside_ordlo' null statistics were always null. The
  // ORDLO comparison must therefore use the distance rule alone (still
  // restricted to eligible pixels); the row records which rule was used.
  // 2019-2021 baseline is equally valid on both sides of the line, the
  // pre-war pseudo-z should be ~N(0,1) both inside and outside ORDLO.
  var ordloRows = [
    {tag: 'inside_ordlo',  m: ordloMask},
    {tag: 'outside_ordlo', m: ordloMask.not()}
  ].map(function(o) {
    var zone = o.m.and(aoi_mask);
    var zStats = sampleNullZ(NULL_MAIN, 15000, o.m).reduceColumns(
      ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true)
        .combine(ee.Reducer.count(), '', true),
      ['zscore_median']);
    return ee.Feature(null, {
      'zone': o.tag,
      'control_rule_for_z_stats': 'temporal_prewar_null_pseudo_z_pooled',
      'eligible_km2': areaKm2(zone),
      'damage_km2': areaKm2(damage_primary.and(zone)),
      'rewild_km2': areaKm2(rewild_primary.and(zone)),
      'damage_share': areaKm2(damage_primary.and(zone)).divide(areaKm2(zone)),
      'rewild_share': areaKm2(rewild_primary.and(zone)).divide(areaKm2(zone)),
      'control_z_mean': zStats.get('mean'),
      'control_z_sd': zStats.get('stdDev'),
      'control_n': zStats.get('count')
    });
  });
  exportTable({
    collection: ee.FeatureCollection(ordloRows),
    description: 'Detection_by_ORDLO',
    folder: exportFolder, fileFormat: 'CSV',
    selectors: ['zone', 'control_rule_for_z_stats', 'eligible_km2',
                'damage_km2', 'rewild_km2',
                'damage_share', 'rewild_share', 'control_z_mean',
                'control_z_sd', 'control_n']
  });
}

// 9f. Land-cover breakdown in true ground km2. WorldCover is descriptive only:
// all standard classes are retained in the mapping domain. Classes absent from
// the AOI legitimately produce zero-area rows.
var ANALYSIS_ESA_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100];
var lcRows = ANALYSIS_ESA_CLASSES.map(function(cls) {
  var classAoi = lcImage.eq(cls).and(aoi_mask);
  return ee.Feature(null, {
    'LC_class': cls,
    'annual_baseline_mode': ANNUAL_BASELINE_MODE,
    'class_area_km2': areaKm2(classAoi),
    'damage_km2': areaKm2(damage_primary.and(classAoi)),
    'rewild_km2': areaKm2(rewild_primary.and(classAoi)),
    'damage_persist_km2': areaKm2(damagePersist.and(classAoi)),
    'rewild_persist_km2': areaKm2(rewildPersist.and(classAoi))
  });
});
if (RUN.landCover) exportTable({
  collection: ee.FeatureCollection(lcRows),
  description: 'Areas_by_LandCover_' + ANNUAL_OUTPUT_TAG,
  folder: exportFolder, fileFormat: 'CSV',
  selectors: ['LC_class', 'annual_baseline_mode', 'class_area_km2',
              'damage_km2', 'rewild_km2',
              'damage_persist_km2', 'rewild_persist_km2']
});

// =====================================================================
// 10. MAIN RASTER EXPORTS
// =====================================================================

if (RUN.mainOutputs) {
  var outputStack = ee.Image.cat([
    prewar.rename('prob_prewar'),  // 01
    wartime.rename('prob_wartime_raw'),  // 02
    resMedian.war_debiased,  // 03
    resMedian.diff_raw,  // 04
    resMedian.diff_smooth,  // 05
    resMedian.diff_corrected,  // 06
    resMedian.z_score,  // 07
    resMedian.z_score_raw,  // 08 uncalibrated z
    resMedian.bias_field,  // 09
    resMedian.local_sd,  // 10
    damage_primary.toFloat(),  // 11
    rewild_primary.toFloat(),  // 12
    damageAttrib.toFloat(),  // 13
    rewildAttrib.toFloat(),  // 14
    damagePersistCount.toFloat(),  // 15
    rewildPersistCount.toFloat(),  // 16
    damagePersist.toFloat(),  // 17
    rewildPersist.toFloat(),  // 18
    distControlKm,  // 19 control-role events
    distValidKm,  // 20 distance to VALID_SOURCE (comparison) events
    confidence.toFloat(),  // 21
    lcImage.toFloat(),  // 22
    warSD.rename('prob_war_sd'),  // 23
    controlMask.toFloat(),  // 24
    extrapFlag.toFloat(),  // 25
    ordloMask.toFloat(),  // 26
    probabilityInterior.toFloat(),  // 27 diagnostic only
    wartimeSdAccepted.toFloat(),  // 28 diagnostic only
    eligible_nonovel.toFloat(),  // 29 quality sensitivity
    eligible_novel.toFloat(),  // 30 quality sensitivity
    resComposite.z_score
      .rename('zscore_composite'),  // 31 sensitivity
    detComposite.damage
      .rename('damage_composite').toFloat(),  // 32 sensitivity
    detComposite.rewild
      .rename('rewild_composite').toFloat(),  // 33 sensitivity
    NULL_MAIN.localSD.rename('local_sd_prewar'),  // 34 unadjusted, floored
    NULL_MAIN.localSDRaw.rename('local_sd_prewar_unfloored'),  // 35
    aoi_mask.toFloat().rename('mapping_mask')  // 36
  ]).updateMask(aoi_mask).clip(roi).float();

  exportImage({
    image: outputStack.unmask(NODATA_VALUE),
    description: 'Debiased_RF_Main_revision_' + ANNUAL_OUTPUT_TAG,
    folder: exportFolder, region: roi, crs: EXPORT_CRS,
    crsTransform: EXPORT_TRANSFORM,
    maxPixels: 1e13, fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: NODATA_VALUE}
  });

}

// is retired). Five Float32 bands, so -9999 is representable in every band:
// 1 class_mapped, 2 class_px, 3 n_loss_years, 4 n_gain_years,
// 5 zscore_median. Observed areas only (no expected areas; see section 7).
if (RUN.yearCounts) {
  exportImage({
    image: ee.Image.cat([
      classMapped, classPx, nLossYears, nGainYears,
      resMedian.z_score.rename('zscore_median')
    ]).updateMask(aoi_mask).clip(roi).toFloat().unmask(NODATA_VALUE),
    description: 'Cumulative_Confidence_Classes',
    folder: exportFolder, region: roi, crs: EXPORT_CRS,
    crsTransform: EXPORT_TRANSFORM,
    maxPixels: 1e13, fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: NODATA_VALUE}
  });

  var ycRows = [];
  [['pixel', classPx, 0], ['mapped', classMapped, MIN_PATCH_PX_CLASSES]]
    .forEach(function(v) {
      [-3, -2, -1, 1, 2, 3].forEach(function(cls) {
        ycRows.push(ee.Feature(null, {
          'analysis_scheme': CHANGE_SCHEME, 'variant': v[0],
          'set': 'dominant_class', 'class': cls,
          'direction': cls > 0 ? 'loss' : 'gain', 'min_years': Math.abs(cls),
          'observed_km2': areaKm2(v[1].eq(cls).and(aoi_mask)),
          'evidence_z_threshold': T_EVIDENCE, 'min_patch_px': v[2],
          'definition': 'dominant direction (more anomalous years; ties by ' +
                        'the sign of the median z), exactly this many years'
        }));
      });
    });
  var unionSets = [
    ['union_loss_ge1', 'loss', 1, nLossYears.gte(1)],
    ['union_loss_ge2', 'loss', 2, nLossYears.gte(2)],
    ['union_gain_ge1', 'gain', 1, nGainYears.gte(1)],
    ['union_gain_ge2', 'gain', 2, nGainYears.gte(2)]
  ];
  unionSets.forEach(function(u) {
    [['pixel', u[3], 0],
     ['mapped', u[3].and(patchKeepDirection(u[3])), MIN_PATCH_PX_CLASSES]]
      .forEach(function(v) {
        ycRows.push(ee.Feature(null, {
          'analysis_scheme': CHANGE_SCHEME, 'variant': v[0], 'set': u[0],
          'class': null, 'direction': u[1], 'min_years': u[2],
          'observed_km2': areaKm2(v[1].and(aoi_mask)),
          'evidence_z_threshold': T_EVIDENCE, 'min_patch_px': v[2],
          'definition': 'at least this many anomalous years in this ' +
                        'direction, whatever the other direction'
        }));
      });
  });
  [['pixel', unionLossPx.and(unionGainPx), 0],
   ['mapped', unionLossMapped.and(unionGainMapped), MIN_PATCH_PX_CLASSES]]
    .forEach(function(v) {
      ycRows.push(ee.Feature(null, {
        'analysis_scheme': CHANGE_SCHEME, 'variant': v[0],
        'set': 'overlap_loss_and_gain', 'class': null, 'direction': 'both',
        'min_years': 1, 'observed_km2': areaKm2(v[1].and(aoi_mask)),
        'evidence_z_threshold': T_EVIDENCE, 'min_patch_px': v[2],
        'definition': 'mixed history: at least one loss year and one gain year'
      }));
    });
  exportTable({
    collection: ee.FeatureCollection(ycRows),
    description: 'Cumulative_Confidence_Class_Areas',
    folder: exportFolder, fileFormat: 'CSV',
    selectors: ['analysis_scheme', 'variant', 'set', 'class', 'direction',
                'min_years', 'observed_km2', 'evidence_z_threshold',
                'min_patch_px', 'definition']
  });
}

// Annual files: band 3 is the annual z-score used for Table 2 and the year
// counts; damage_<year>/rewild_<year> apply the STRICTER rules to each year
// (diagnostic, not reported).
if (RUN.yearlyOutputs) {
  YEARS.forEach(function(y) {
    var yearStack = ee.Image.cat([
      resYear[y].diff_raw,
      resYear[y].diff_corrected,
      resYear[y].z_score,
      resYear[y].z_score_raw,
      resYear[y].bias_field,
      resYear[y].war_debiased,
      damageYear[y].rename('damage_' + y).toFloat(),
      rewildYear[y].rename('rewild_' + y).toFloat()
    ]).updateMask(aoi_mask).clip(roi).float();

    exportImage({
      image: yearStack.unmask(NODATA_VALUE),
      description: 'Debiased_RF_' + y + '_AnnualInputSensitivity_revision_' +
        ANNUAL_OUTPUT_TAG,
      folder: exportFolder, region: roi, crs: EXPORT_CRS,
      crsTransform: EXPORT_TRANSFORM,
      maxPixels: 1e13, fileFormat: 'GeoTIFF',
      formatOptions: {cloudOptimized: true, noData: NODATA_VALUE}
    });
  });
}

// =====================================================================
// 11. CONTROL-ZONE DIAGNOSTICS TABLE (extended)
// =====================================================================

var diagReducer = ee.Reducer.mean().setOutputs(['mean'])
  .combine(ee.Reducer.stdDev().setOutputs(['sd']), '', true)
  .combine(ee.Reducer.percentile({
    percentiles: [2.5, 25, 50, 75, 97.5],
    outputNames: ['p2_5', 'p25', 'p50', 'p75', 'p97_5']
  }), '', true);

// pseudo-z (the calibration check: mean ~0, SD ~1, and how heavy the tails
// are); 'wartime_full_domain' rows summarise the wartime z over the whole
// valid domain (descriptive: war effects included by construction).
function diagConfigProps(extra) {
  var base = {
    'analysis_scheme': CHANGE_SCHEME,
    'analysis_scheme_label': CHANGE_SCHEME_LABEL,
    'null_definition': 'temporal_prewar_loo_pseudo_z_v13b8',
    'bias_sigma_m': BIAS_SIGMA_M,
    'bias_sigma_units': 'EPSG3035_grid_m',
    'smooth_sigma_m': SMOOTH_SIGMA_M,
    'smooth_sigma_units': 'EPSG3035_grid_m',
    'scale_correction_status': 'recomputed_on_equal_area_metric_grid',
    'global_recalibration': APPLY_GLOBAL_RECAL ? 1 : 0
  };
  for (var k in extra) { base[k] = extra[k]; }
  return base;
}

var diagList = ['median', 'composite'].concat(YEARS).map(function(period) {
  var res = (period === 'median') ? resMedian
    : (period === 'composite') ? resComposite : resYear[period];

  var zRawBand = ee.String(res.z_score_raw.bandNames().get(0));
  var zCalBand = ee.String(res.z_score.bandNames().get(0));
  var corrBand = ee.String(res.diff_corrected.bandNames().get(0));

  var sdStats = res.local_sd.updateMask(aoi_mask).reduceRegion({
    reducer: ee.Reducer.mean(), geometry: roi, scale: coarseScale,
    maxPixels: 1e9, bestEffort: true});
  var zRawStats = res.z_score_raw.updateMask(aoi_mask).reduceRegion({
    reducer: diagReducer, geometry: roi, scale: coarseScale,
    maxPixels: 1e9, bestEffort: true});
  var zCalStats = res.z_score.updateMask(aoi_mask).reduceRegion({
    reducer: diagReducer, geometry: roi, scale: coarseScale,
    maxPixels: 1e9, bestEffort: true});
  var corrStats = res.diff_corrected.updateMask(aoi_mask).reduceRegion({
    reducer: diagReducer, geometry: roi, scale: coarseScale,
    maxPixels: 1e9, bestEffort: true});

  return ee.Feature(null, diagConfigProps({
    'period': period,
    'stat_domain': 'wartime_full_domain',
    'annual_baseline_mode': (period === 'median' || period === 'composite')
      ? 'not_applicable' : ANNUAL_BASELINE_MODE,
    'mean_local_sd': sdStats.get('local_sd'),
    'corr_mean': corrStats.get(corrBand.cat('_mean')),
    'corr_sd':   corrStats.get(corrBand.cat('_sd')),
    'z_raw_mean':  zRawStats.get(zRawBand.cat('_mean')),
    'z_raw_sd':    zRawStats.get(zRawBand.cat('_sd')),
    'z_raw_p2_5':  zRawStats.get(zRawBand.cat('_p2_5')),
    'z_raw_p97_5': zRawStats.get(zRawBand.cat('_p97_5')),
    'z_cal_mean':  zCalStats.get(zCalBand.cat('_mean')),
    'z_cal_sd':    zCalStats.get(zCalBand.cat('_sd')),
    'z_cal_p2_5':  zCalStats.get(zCalBand.cat('_p2_5')),
    'z_cal_p97_5': zCalStats.get(zCalBand.cat('_p97_5'))
  }));
});

NULL_MAIN.years.forEach(function(y, ni) {
  var band = 'pseudo_z_' + y;
  var st = NULL_MAIN.pseudoZ[ni].updateMask(aoi_mask).reduceRegion({
    reducer: diagReducer, geometry: roi, scale: coarseScale,
    maxPixels: 1e9, bestEffort: true});
  diagList.push(ee.Feature(null, diagConfigProps({
    'period': 'null_' + y,
    'stat_domain': 'temporal_null',
    'annual_baseline_mode': 'not_applicable',
    'z_cal_mean':  st.get(band + '_mean'),
    'z_cal_sd':    st.get(band + '_sd'),
    'z_cal_p2_5':  st.get(band + '_p2_5'),
    'z_cal_p97_5': st.get(band + '_p97_5')
  })));
});
diagList.push(ee.Feature(null, diagConfigProps({
  'period': 'null_pooled_sample',
  'stat_domain': 'temporal_null',
  'annual_baseline_mode': 'not_applicable',
  'z_cal_mean': NULL_POOLED.mean,
  'z_cal_sd': ee.Number(nullPooledMoments.get('stdDev'))
})));

if (RUN.diagnosticsTable) exportTable({
  collection: ee.FeatureCollection(diagList),
  description: 'Debiased_RF_Diagnostics_revision_' + ANNUAL_OUTPUT_TAG,
  folder: exportFolder, fileFormat: 'CSV'
});

// =====================================================================
// =====================================================================
// One row: run identity, the input assets with their Earth Engine version
// parameters, and the tables and GeoTIFFs this run exported. Python_03
// checks that every table of the scheme folder carries this run_id and that
// the three probability/novelty assets come from one GEE_04 run.
function assetProps(prefix, id, kind) {
  var obj = kind === 'table' ? ee.FeatureCollection(id) : ee.Image(id);
  var out = {};
  out[prefix + '_asset'] = id;
  out[prefix + '_asset_version'] = obj.get('system:version');
  if (kind !== 'table') {
    out[prefix + '_run_id'] = obj.get('run_id');
    out[prefix + '_code_version'] = obj.get('code_version');
    out[prefix + '_presence_score_mode'] = obj.get('presence_score_mode');
  }
  return out;
}
var manifestProps = extend(extend(extend(extend(extend(extend({
  'run_started_utc': RUN_STARTED_UTC,
  'change_scheme_label': CHANGE_SCHEME_LABEL,
  'export_folder': exportFolder,
  'exported_tables': EXPORTED_TABLES.join(';'),
  'exported_geotiffs': EXPORTED_IMAGES.join(';'),
  'n_exported_tables': EXPORTED_TABLES.length + 1,
  'run_flags': JSON.stringify(RUN),
  'probability_asset_suffix': PROBABILITY_ASSET_SUFFIX[CHANGE_SCHEME],
  'smooth_sigma_m': SMOOTH_SIGMA_M, 'bias_sigma_m': BIAS_SIGMA_M,
  'coarse_scale_m': coarseScale, 'den_clamp_frac': DEN_CLAMP_FRAC,
  'min_local_sd': MIN_LOCAL_SD, 'z_thresh': Z_THRESH,
  'z_neighbour': Z_NEIGHBOR, 'min_patch_px': MIN_PATCH_PX,
  'apply_closing': APPLY_CLOSING ? 1 : 0, 'threshold_mode': THRESHOLD_MODE,
  'evidence_z_threshold': T_EVIDENCE,
  'min_patch_px_year_counts': MIN_PATCH_PX_CLASSES,
  'median_sd_factor': MEDIAN_NULL_SD_FACTOR,
  'annual_sd_factor': PERYEAR_NULL_SD_FACTOR,
  'max_war_sd': MAX_WAR_SD, 'prob_lo': PROB_LO, 'prob_hi': PROB_HI,
  'annual_baseline_mode': ANNUAL_BASELINE_MODE,
  'global_recalibration': APPLY_GLOBAL_RECAL ? 1 : 0,
  'bias_sigma_list_m': BIAS_SIGMA_LIST.join(','),
  'sd_floor_list': MIN_LOCAL_SD_LIST.join(','),
  'z_thresh_list': Z_THRESH_LIST.join(','),
  'min_patch_list': MIN_PATCH_LIST.join(','),
  'eligible_ground_area_km2': ELIGIBLE_GROUND_AREA_KM2,
  'eligible_px': ELIGIBLE_PIXEL_COUNT
}, assetProps('period_probability', periodProbabilityAsset, 'image')),
  assetProps('annual_probability', annualProbabilityAsset, 'image')),
  assetProps('novelty', noveltyAsset, 'image')),
  assetProps('aoi', aoiAsset, 'table')),
  assetProps('war_events', warEventsAsset, 'table')),
  assetProps('ordlo', ordloAsset, 'table'));
exportTable({
  collection: ee.FeatureCollection([ee.Feature(null, manifestProps)]),
  description: 'GEE05_RunManifest' + OUT_SFX,
  folder: exportFolder, fileFormat: 'CSV'
});
print('v13b12 run_id:', RUN_ID, '| tables:', EXPORTED_TABLES.length,
      '| GeoTIFFs:', EXPORTED_IMAGES.length);
print('GEE_04 run_id on the probability assets (must be one run):',
      ee.Image(periodProbabilityAsset).get('run_id'),
      ee.Image(annualProbabilityAsset).get('run_id'),
      ee.Image(noveltyAsset).get('run_id'));

print('GEE_05 analysis scheme:', CHANGE_SCHEME, '=', CHANGE_SCHEME_LABEL);
print('Exports go to Drive folder:', exportFolder);
print('Run this script once per scheme (flip CHANGE_SCHEME and run again).');
print('v13b8: null calibrated from the PRE-WAR YEARS (leave-one-out '
      + 'pseudo-changes 2019-2021); spatial control zones are documentation '
      + 'only -- see ControlZones_Metadata (max event distance ~40 km).');
print('v13b10: cumulative year-count confidence at |z_year| >= ' + T_EVIDENCE
      + ' (classes +/-1..3 = anomalous years in the dominant direction); '
      + 'Cumulative_Confidence_Classes = 5 x Float32; observed areas only.');
print('SD factors (median / annual): main', MEDIAN_NULL_SD_FACTOR.toFixed(4),
      '/', PERYEAR_NULL_SD_FACTOR.toFixed(4), '| without a year, variance only',
      comparisonSdFactor(MEDIAN3_VARIANCE_FACTOR, MEDIAN3_VARIANCE_FACTOR, 2)
        .toFixed(4), '/',
      comparisonSdFactor(MEDIAN3_VARIANCE_FACTOR, 1, 2).toFixed(4),
      '| baseline and variance',
      comparisonSdFactor(0.5, MEDIAN3_VARIANCE_FACTOR, 2).toFixed(4), '/',
      comparisonSdFactor(0.5, 1, 2).toFixed(4), '| closing:', APPLY_CLOSING,
      '| SD floor:', MIN_LOCAL_SD);
print('v13b9: headline = annual-median vs annual-median, null SD factor '
      + MEDIAN_NULL_SD_FACTOR.toFixed(4) + '; per-year factor '
      + PERYEAR_NULL_SD_FACTOR.toFixed(4)
      + '; composite comparison retained as conservative sensitivity; '
      + 'graded excess areas in Areas_Excess_by_Zclass.');
print('GEE_05 configured. Control mode:', CONTROL_MODE,
      '| threshold mode:', THRESHOLD_MODE,
      '| global recalibration:', APPLY_GLOBAL_RECAL,
      '| primary mapping mask: valid input coverage only',
      '| WorldCover/novelty/probability/SD primary masks:', false,
      '| annual baseline:', ANNUAL_BASELINE_MODE,
      '| annual baseline sensitivity:', RUN_ANNUAL_BASELINE_SENSITIVITY);
print('EPSG:3035 scale audit (configured values are retained):',
      ee.Dictionary({
        'south_latitude': aoiSouthLat,
        'centroid_latitude': aoiCentroidLat,
        'north_latitude': aoiNorthLat,
        'linear_ground_factor_south': GROUND_FACTOR_SOUTH,
        'linear_ground_factor_centroid': GROUND_FACTOR_CENTROID,
        'linear_ground_factor_north': GROUND_FACTOR_NORTH,
        'configured_500m_pixel_ground_centroid_m': groundEquivalent(
          refScale, 'linear', GROUND_FACTOR_CENTROID),
        'configured_50km_control_ground_centroid_km': groundEquivalent(
          CONTROL_MIN_DIST_KM, 'linear', GROUND_FACTOR_CENTROID)
      }));
