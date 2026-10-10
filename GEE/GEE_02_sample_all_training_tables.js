// SAMPLING — RUN AFTER GEE_01 HAS FINISHED.
/**
 * Samples every uploaded presence/pseudo-absence design against one combined
 * pre-war predictor stack and writes one complete-case table asset per scheme.
 * Embedding, baseline and NDVI-null comparisons therefore use exactly the same
 * records. Later CV is table-only, which is the main memory safeguard.
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
var REF_SCALE = 500;
var EXPORT_CRS = 'EPSG:3035';
var EXPORT_TRANSFORM = [500, 0, 0, 0, -500, 0];
var PREWAR_YEARS = [2019, 2020, 2021];
var OUTDOOR_CLASSES = [10, 20, 30, 80, 90, 100];
var INCLUDE_FULL_WINDOW_SENSITIVITY = true;

var SCHEMES = {
  primary_effort: {
    presenceAsset: OWN + 'points_regular_presence',
    absenceAsset: OWN + 'points_regular_pseudoabsence',
    runTag: 'nondvi_outdoor_effort_guard0_matched2019_2021',
    recurrent: false,
    paired: false
  },
  primary_effort_full: {
    presenceAsset: OWN + 'points_full_presence',
    absenceAsset: OWN + 'points_full_pseudoabsence',
    runTag: 'nondvi_outdoor_effort_guard0_full2015_2021',
    recurrent: false,
    paired: false
  },
  access_matched: {
    presenceAsset: OWN + 'points_regular_matched_presence',
    absenceAsset: OWN + 'points_regular_matched_pseudoabsence',
    runTag: 'nondvi_outdoor_accessmatch_guard0_matched2019_2021',
    recurrent: false,
    paired: true
  },
  recurrent_effort: {
    presenceAsset: OWN + 'points_recurrent_presence',
    absenceAsset: OWN + 'points_recurrent_pseudoabsence',
    runTag: 'recurrent_core_nondvi_effort_guard0_matched2019_2021',
    recurrent: true,
    paired: false
  },
  recurrent_access_matched: {
    presenceAsset: OWN + 'points_recurrent_matched_presence',
    absenceAsset: OWN + 'points_recurrent_matched_pseudoabsence',
    runTag: 'recurrent_core_nondvi_accessmatch_guard0_matched2019_2021',
    recurrent: true,
    paired: true
  }
};

var aoi = ee.FeatureCollection(AOI_ASSET);
var roi = aoi.geometry();
var refProjection = ee.Projection(EXPORT_CRS, EXPORT_TRANSFORM);
var modisNdvi = ee.ImageCollection('MODIS/061/MOD13Q1');
var worldCover = ee.ImageCollection('ESA/WorldCover/v200')
  .filterBounds(roi).mosaic().clip(roi).select('Map').rename('lc');
var dem = ee.Image('USGS/SRTMGL1_003');

var EMBEDDING_BANDS = [];
for (var bandIndex = 0; bandIndex < 64; bandIndex++) {
  EMBEDDING_BANDS.push('A' + (bandIndex < 10 ? '0' : '') + bandIndex);
}
var MODEL_EMBEDDING_BANDS = EMBEDDING_BANDS;
var POINT_PROPERTIES = [
  'presence', 'variant', 'layer', 'year', 'in_matched_window', 'esa',
  'ndvi', 'dist_infra_m', 'effort_5km', 'log_effort', 'n_records',
  'n_users', 'n_user_days', 'n_dates', 'n_months', 'n_years',
  'span_days', 'recurrent_rule', 'match_pair_id', 'in_ordlo', 'pa_mode',
  'pa_guard_rings', 'time_window'
];

function firstExistingProperty(feature, names, fallback) {
  var propertyNames = feature.propertyNames();
  var value = fallback;
  for (var i = names.length - 1; i >= 0; i--) {
    value = ee.Algorithms.If(propertyNames.contains(names[i]),
      feature.get(names[i]), value);
  }
  return value;
}

function normalisePoint(feature, isPresence, schemeName) {
  return feature.set({
    presence: isPresence ? 1 : 0,
    variant: firstExistingProperty(feature, ['variant'], 'nondvi'),
    layer: firstExistingProperty(feature, ['layer'],
      isPresence ? 'presence' : 'pseudo_absence'),
    year: firstExistingProperty(feature, ['year'], null),
    in_matched_window: firstExistingProperty(
      feature, ['in_matched_window', 'match_win', 'in_matched'], 1),
    esa: firstExistingProperty(feature, ['esa'], null),
    // Point-level Sentinel-2 summer-median NDVI from Python_A1: the NDVI
    // predictor of the baseline and NDVI-only models.
    ndvi: firstExistingProperty(feature, ['ndvi'], null),
    dist_infra_m: firstExistingProperty(
      feature, ['dist_infra_m', 'dist_inf', 'dist_infra'], null),
    effort_5km: firstExistingProperty(
      feature, ['effort_5km', 'effort5k'], null),
    log_effort: firstExistingProperty(
      feature, ['log_effort', 'log_eff'], null),
    n_records: firstExistingProperty(feature, ['n_records'], null),
    n_users: firstExistingProperty(feature, ['n_users'], null),
    n_user_days: firstExistingProperty(
      feature, ['n_user_days', 'n_udays'], null),
    n_dates: firstExistingProperty(feature, ['n_dates'], null),
    n_months: firstExistingProperty(feature, ['n_months'], null),
    n_years: firstExistingProperty(feature, ['n_years'], null),
    span_days: firstExistingProperty(feature, ['span_days'], null),
    recurrent_rule: firstExistingProperty(
      feature, ['recurrent_rule', 'recur_rule'], null),
    match_pair_id: firstExistingProperty(
      feature, ['match_pair_id', 'pair_id'], null),
    in_ordlo: firstExistingProperty(feature, ['in_ordlo'], 0),
    pa_mode: firstExistingProperty(feature, ['pa_mode'],
      isPresence ? 'observed' : schemeName),
    pa_guard_rings: firstExistingProperty(
      feature, ['pa_guard_rings', 'pa_guard'], isPresence ? null : 0),
    time_window: firstExistingProperty(
      feature, ['time_window', 'time_win'], 'matched')
  });
}

function toReferenceGrid(image) {
  return image.reproject({crs: EXPORT_CRS, crsTransform: EXPORT_TRANSFORM});
}

function annualEmbeddingAsset(year) {
  return ee.Image(OWN + 'EmbeddingAnnual_' + year +
    '_EPSG3035_500m_v13b').select(EMBEDDING_BANDS);
}

function prewarEmbedding() {
  var annual = ee.ImageCollection.fromImages(PREWAR_YEARS.map(function(year) {
    return annualEmbeddingAsset(year);
  }));
  return annual.median().select(EMBEDDING_BANDS)
    .setDefaultProjection(refProjection).toFloat();
}

function summerNdvi(years) {
  var startYear = Math.min.apply(null, years);
  var endYear = Math.max.apply(null, years);
  var filteredNdvi = modisNdvi
    .filterBounds(roi)
    .filterDate(
      ee.Date.fromYMD(startYear, 1, 1),
      ee.Date.fromYMD(endYear + 1, 1, 1))
    .filter(ee.Filter.calendarRange(5, 9, 'month'));
  var modisProjection = ee.Image(filteredNdvi.first())
    .select('NDVI').projection();
  var ndvi = filteredNdvi.map(function (image) {
      var quality = image.select('SummaryQA').lte(1);
      return image.select('NDVI').multiply(0.0001)
        .updateMask(quality).rename('ndvi');
    }).median()
    // ImageCollection reductions do not retain a usable default projection.
    // Restore the native MODIS grid before reduceResolution aggregates 250 m
    // pixels to the fixed 500 m analysis grid.
    .setDefaultProjection(modisProjection);
  // MODIS is 250 m, so a direct 2 x 2 mean is bounded and inexpensive.
  return ndvi.reduceResolution({
      reducer: ee.Reducer.mean(), bestEffort: false, maxPixels: 16
    })
    .reproject({crs: EXPORT_CRS, crsTransform: EXPORT_TRANSFORM})
    .rename('ndvi_modis').toFloat();
}

worldCover = toReferenceGrid(worldCover).toInt16();
var analysisDomain = worldCover.remap(
  OUTDOOR_CLASSES, OUTDOOR_CLASSES.map(function () { return 1; }), 0
).eq(1).rename('analysis_domain');
var embeddingPrewar = prewarEmbedding()
  .updateMask(analysisDomain);
var ndviPrewar = summerNdvi([2019, 2020, 2021])
  .updateMask(analysisDomain);

var LAND_COVER_CLASSES = [10, 20, 30, 80, 90, 100];
var LAND_COVER_BANDS = [
  'lc_tree', 'lc_shrub', 'lc_grass', 'lc_water', 'lc_wetland',
  'lc_moss_lichen'
];
var landCoverIndicators = ee.Image.cat(LAND_COVER_CLASSES.map(
  function (landCoverClass, index) {
    return worldCover.eq(landCoverClass)
      .rename(LAND_COVER_BANDS[index]).toFloat();
  }
));
var waterMask = worldCover.eq(80);
var WATER_DISTANCE_NEIGHBORHOOD_PX = 1024;
var WORLD_COVER_PIXEL_M = REF_SCALE;
var WATER_DISTANCE_CAP_KM =
  WATER_DISTANCE_NEIGHBORHOOD_PX * WORLD_COVER_PIXEL_M / 1000;
var distanceToWaterKm = waterMask.fastDistanceTransform({
  neighborhood: WATER_DISTANCE_NEIGHBORHOOD_PX,
  units: 'pixels',
  metric: 'squared_euclidean'
}).sqrt().multiply(WORLD_COVER_PIXEL_M / 1000)
  .unmask(WATER_DISTANCE_CAP_KM)
  .rename('dist_water_km').toFloat();
var elevation = toReferenceGrid(dem.resample('bilinear'))
  .rename('elevation').toFloat();
var slope = toReferenceGrid(ee.Terrain.slope(dem).resample('bilinear'))
  .rename('slope').toFloat();
var tpi = elevation.subtract(elevation.focal_mean({
  radius: 3, kernelType: 'square', units: 'pixels'
})).rename('tpi').toFloat();

// Raster part of the baseline stack. 'ndvi_modis' is sampled for reference
// only; the baseline models use the point attribute 'ndvi' (see header).
var BASELINE_BANDS = ['ndvi_modis'].concat(LAND_COVER_BANDS)
  .concat(['dist_water_km', 'elevation', 'slope', 'tpi']);
var baselineStack = ee.Image.cat([
  ndviPrewar, landCoverIndicators, distanceToWaterKm, elevation, slope, tpi
]).updateMask(analysisDomain);
var embeddingStack = embeddingPrewar.updateMask(analysisDomain);
var ALL_PREDICTOR_BANDS = MODEL_EMBEDDING_BANDS.concat(BASELINE_BANDS);
var allPredictorStack = embeddingStack.addBands(baselineStack)
  .select(ALL_PREDICTOR_BANDS).updateMask(analysisDomain);

function retainCompleteMatchedPairs(sampled) {
  var p = sampled.filter(ee.Filter.eq('presence', 1));
  var a = sampled.filter(ee.Filter.eq('presence', 0));
  var pairCondition = ee.Filter.equals({
    leftField: 'match_pair_id', rightField: 'match_pair_id'
  });
  // saveFirst returns the original feature with one matched counterpart. This
  // avoids materialising a large aggregate_array and ee.Filter.inList for the
  // non-recurrent design while preserving one presence and one background per
  // complete pair.
  function matchedSide(left, right) {
    return ee.FeatureCollection(ee.Join.saveFirst('_matched_pair').apply({
      primary: left,
      secondary: right,
      condition: pairCondition
    })).filter(ee.Filter.notNull(['_matched_pair'])).map(function (feature) {
      feature = ee.Feature(feature);
      return feature.select(feature.propertyNames().remove('_matched_pair'));
    });
  }
  return matchedSide(p, a).merge(matchedSide(a, p));
}

function exportScheme(schemeName) {
  var scheme = SCHEMES[schemeName];
  var presences = ee.FeatureCollection(scheme.presenceAsset)
    .filterBounds(roi)
    .map(function (f) { return normalisePoint(f, true, schemeName); })
    .filter(ee.Filter.inList('esa', OUTDOOR_CLASSES));
  var absences = ee.FeatureCollection(scheme.absenceAsset)
    .filterBounds(roi)
    .map(function (f) { return normalisePoint(f, false, schemeName); })
    .filter(ee.Filter.inList('esa', OUTDOOR_CLASSES));
  if (scheme.recurrent) {
    presences = presences.filter(ee.Filter.and(
      ee.Filter.gte('n_user_days', 3),
      ee.Filter.gte('n_users', 2),
      ee.Filter.gte('n_dates', 2),
      ee.Filter.gte('span_days', 30)
    ));
  }
  var allPoints = presences.merge(absences);

  function sampleCompletePredictors() {
    var sampled = allPredictorStack.sampleRegions({
      collection: allPoints,
      properties: POINT_PROPERTIES,
      projection: refProjection,
      geometries: true,
      tileScale: 16
    }).map(function (feature) {
      return feature.set({
        sampling_scheme: schemeName,
        run_tag: scheme.runTag,
        sampled_model: 'complete_case_all_predictors',
        reference_scale_m: REF_SCALE,
        analysis_crs: EXPORT_CRS,
        embedding_representation:
          'original64_annual500m_prewar_temporal_median',
        training_domain: 'WorldCover_10_20_30_80_90_100',
        pipeline_version: 'v13b'
      });
    });
    return scheme.paired ? retainCompleteMatchedPairs(sampled) : sampled;
  }

  var sampledAll = sampleCompletePredictors();
  var sampleAsset = OWN + 'SDM_Sampled_all_' + schemeName + '_outdoor_v13b';
  print(schemeName + ' point counts before raster completeness:',
    allPoints.aggregate_histogram('presence'));
  print(schemeName + ' complete-case predictor counts:',
    sampledAll.aggregate_histogram('presence'));
  Export.table.toAsset({
    collection: sampledAll,
    description: 'SDM_Sampled_all_' + schemeName + '_outdoor_v13b',
    assetId: sampleAsset
  });
}

exportScheme('primary_effort');
exportScheme('access_matched');
exportScheme('recurrent_effort');
exportScheme('recurrent_access_matched');
if (INCLUDE_FULL_WINDOW_SENSITIVITY) {
  exportScheme('primary_effort_full');
}

print('Start the five table-to-asset tasks (' + OWN + 'SDM_Sampled_all_*). ' +
      'Wait for all to finish before GEE_03.');
