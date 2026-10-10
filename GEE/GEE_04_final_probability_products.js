// FINAL MODELS. RUN ONCE AFTER GEE_03.
/**
 * Fits the two final estimands with the hyperparameters declared before CV:
 *   500 trees, minLeafPopulation=20, variablesPerSplit=16 (all 64 embedding
 *   bands are inputs; 16 randomly chosen bands are evaluated at each split).
 * No held-out score is used to choose these settings.
 *
 * Outputs are produced for:
 *   1) primary_effort — ordinary outdoor-visitation presences;
 *   2) recurrent_effort — recurrently observed outdoor visitation.
 *
 * Predictor representation (identical in sampling, CV and mapping):
 *   - the 64 original embedding bands materialised directly on the fixed
 *     500 m EPSG:3035 grid;
 *   - no embedding spatial means, SDs, heterogeneity or focal context.
 *
 * WorldCover classes [10,20,30,80,90,100] defined the symmetric training
 * domain in Python_A1/GEE_02. THEY DO NOT MASK THESE MAPS. Cropland, built-up
 * land and all other locations with valid embeddings remain in probability
 * and change rasters. Such pixels are transfer predictions and are accompanied
 * by same-run novelty and land-cover-domain diagnostics.
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

// Google Drive folder of all exports; download it into
// outputs/GEE04_final_models (support/folders.py).
var EXPORT_FOLDER = 'GEE04_final_models';
var EXPORT_CRS = 'EPSG:3035';
var EXPORT_TRANSFORM = [500, 0, 0, 0, -500, 0];
var REF_SCALE = 500;
var NODATA_VALUE = -9999;
var SEED = 20250523;
var N_TREES = 500;
var FIXED_MIN_LEAF = 20;
var FIXED_VARIABLES_PER_SPLIT = 16;
var NOVELTY_PERCENTILE = 99;
var TRAINING_ESA_CLASSES = [10, 20, 30, 80, 90, 100];

// RUN MODE per scheme:
//   'checks'  -- row checks, probability-mode check and fitted points only;
//   'novelty' -- checks + the novelty asset;
//   'all'     -- checks + novelty + final probability assets and GeoTIFFs
// GEE_05 come from the final models refitted on their CV rows.
var RUN_MODE = {primary_effort: 'all', recurrent_effort: 'all'};
var NEW_SUFFIX = 'v13b12';
var CODE_VERSION = 'v13b12';
var RUN_STARTED_UTC = new Date().toISOString();
var RUN_ID = 'GEE04_' + CODE_VERSION + '_' +
  RUN_STARTED_UTC.replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');

// GEE_03). 'PROBABILITY' = the classifier's PROBABILITY output (main
// setting).
var PRESENCE_SCORE_MODE = 'PROBABILITY';
var CLASS1_INDEX = 1;
var CHECK_SAMPLE_PIXELS = 3000;  // mapped pixels per embedding for the check
var CLEAR_WINNER_MARGIN = 0.05;  // |p0 - p1| above which the argmax is clear

function exportTable(params, schemeName) {
  params.collection = ee.FeatureCollection(params.collection)
    .map(function (f) {
      return f.set({run_id: RUN_ID, code_version: CODE_VERSION,
                    analysis_scheme: schemeName});
    });
  Export.table.toDrive(params);
}

var SCHEMES = {
  primary_effort: {
    runTag: 'nondvi_outdoor_effort_guard0_matched2019_2021',
    recurrent: false
  },
  recurrent_effort: {
    runTag: 'recurrent_core_nondvi_effort_guard0_matched2019_2021',
    recurrent: true
  }
};

var EMBEDDING_BANDS = [];
for (var bandIndex = 0; bandIndex < 64; bandIndex++) {
  EMBEDDING_BANDS.push('A' + (bandIndex < 10 ? '0' : '') + bandIndex);
}
var MODEL_BANDS = EMBEDDING_BANDS;
var FINGERPRINT_WEIGHTS = ee.Array(EMBEDDING_BANDS.map(function (b, i) {
  return i + 1;
}));
function embFingerprint(feature) {
  return ee.Array(feature.toArray(EMBEDDING_BANDS)).multiply(FINGERPRINT_WEIGHTS)
    .reduce(ee.Reducer.sum(), [0]).get([0]);
}
var YEARS = [2019, 2020, 2021, 2022, 2023, 2024];

var aoi = ee.FeatureCollection(AOI_ASSET);
var roi = aoi.geometry();
var refProjection = ee.Projection(EXPORT_CRS, EXPORT_TRANSFORM);

function annualEmbeddingAsset(year) {
  return ee.Image(OWN + 'EmbeddingAnnual_' + year +
    '_EPSG3035_500m_v13b').select(EMBEDDING_BANDS).toFloat();
}

var annualFeatures = {};
YEARS.forEach(function (year) {
  annualFeatures[year] = annualEmbeddingAsset(year);
});

function periodFeatures(years) {
  var periodCollection = ee.ImageCollection.fromImages(years.map(function (year) {
    return annualEmbeddingAsset(year);
  }));
  return periodCollection.median().select(EMBEDDING_BANDS)
    .setDefaultProjection(refProjection).toFloat();
}

var prewarFeatures = periodFeatures([2019, 2020, 2021]);
var wartimeFeatures = periodFeatures([2022, 2023, 2024]);

var worldCover = ee.ImageCollection('ESA/WorldCover/v200')
  .filterBounds(roi).mosaic().select('Map').rename('lc')
  .reproject({crs: refProjection});
var trainingLandcoverDomain = worldCover.remap(
  TRAINING_ESA_CLASSES,
  TRAINING_ESA_CLASSES.map(function () { return 1; }), 0
).rename('in_training_landcover_domain').toByte();
// MODIS 500 m NDVI for the interpretability sample only (map-wide summary of
// predictions against greenness); it is not a model input.
var prewarNdviCollection = ee.ImageCollection('MODIS/061/MOD13Q1')
  .filterBounds(roi).filterDate('2019-01-01', '2022-01-01')
  .filter(ee.Filter.calendarRange(5, 9, 'month'));
var prewarNdviProjection = ee.Image(prewarNdviCollection.first())
  .select('NDVI').projection();
var prewarNdvi = prewarNdviCollection.map(function (image) {
    return image.select('NDVI').multiply(0.0001)
      .updateMask(image.select('SummaryQA').lte(1)).rename('ndvi');
  }).median()
  // Restore the native MODIS projection lost by ImageCollection.median().
  .setDefaultProjection(prewarNdviProjection)
  .reduceResolution({
    reducer: ee.Reducer.mean(), bestEffort: false, maxPixels: 16
  }).reproject({crs: EXPORT_CRS, crsTransform: EXPORT_TRANSFORM})
  .rename('ndvi_2019_2021').toFloat();

function exportGeoTiff(image, description) {
  Export.image.toDrive({
    image: image.clip(roi).unmask(NODATA_VALUE).toFloat(),
    description: description,
    folder: EXPORT_FOLDER,
    region: roi,
    crs: EXPORT_CRS,
    crsTransform: EXPORT_TRANSFORM,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: NODATA_VALUE}
  });
}


function assetExists(id) {
  try {
    var info = ee.data.getAsset(id);
    return info !== null && info !== undefined;
  } catch (err) {
    return false;
  }
}
function assertAssetAbsent(id) {
  if (assetExists(id)) {
    throw new Error('Asset ' + id + ' already exists (an earlier run). Use a ' +
      'new ASSET_PREFIX in all GEE scripts and in support/folders.py, so that ' +
      'GEE_05 reads the assets of this run.');
  }
}

function exportAsset(image, description, assetId, schemeName) {
  assertAssetAbsent(assetId);
  Export.image.toAsset({
    image: image.clip(roi).set({run_id: RUN_ID, code_version: CODE_VERSION,
                                analysis_scheme: schemeName,
                                presence_score_mode: PRESENCE_SCORE_MODE,
                                class1_index: CLASS1_INDEX}),
    description: description,
    assetId: assetId,
    region: roi,
    crs: EXPORT_CRS,
    crsTransform: EXPORT_TRANSFORM,
    maxPixels: 1e13
  });
}

function runFinalModel(schemeName) {
  var scheme = SCHEMES[schemeName];
  var BASELINE_BANDS_FOR_COMPLETE_CASE = [
    'ndvi', 'lc_tree', 'lc_shrub', 'lc_grass', 'lc_water', 'lc_wetland',
    'lc_moss_lichen', 'dist_water_km', 'elevation', 'slope', 'tpi'
  ];
  var mode = RUN_MODE[schemeName];
  var sfx = '_' + NEW_SUFFIX;
  if (['checks', 'novelty', 'all'].indexOf(mode) < 0) {
    throw new Error('RUN_MODE must be checks, novelty or all');
  }
  var completeCaseBands = MODEL_BANDS.concat(BASELINE_BANDS_FOR_COMPLETE_CASE);
  function withCount(fc, props, name) {
    return fc.map(function (feature) {
      return feature.set(name, feature.toDictionary(props).size());
    });
  }
  var sampledTable = ee.FeatureCollection(OWN + 'SDM_Sampled_all_' +
    schemeName + '_outdoor_v13b').filterBounds(roi)
    .map(function (feature) { return feature.set('row_id', feature.id()); });
  var requiredProperties = completeCaseBands.concat(['presence']);
  var embeddingProperties = MODEL_BANDS.concat(['presence']);
  var counted = withCount(withCount(sampledTable, requiredProperties,
    'n_complete_rule'), embeddingProperties, 'n_complete_emb');
  var training = counted
    .filter(ee.Filter.eq('n_complete_rule', requiredProperties.length))
    .map(function (feature) {
      return feature.set('emb_fingerprint', embFingerprint(feature));
    });

  var v34Bands = schemeName === 'primary_effort'
    ? MODEL_BANDS.concat(BASELINE_BANDS_FOR_COMPLETE_CASE) : MODEL_BANDS;
  var v34Reconstruction = sampledTable
    .filter(ee.Filter.notNull(v34Bands.concat(['presence'])));
  var v34Ids = v34Reconstruction.map(function (feature) {
    return ee.Feature(null, {row_id: feature.get('row_id')});
  });
  var rowSets = ee.Join.saveFirst({matchKey: '_v34', outer: true}).apply(
    counted, v34Ids,
    ee.Filter.equals({leftField: 'row_id', rightField: 'row_id'}))
    .map(function (feature) {
      feature = ee.Feature(feature);
      var coordinates = feature.geometry().coordinates();
      var inRule = ee.Number(feature.get('n_complete_rule'))
        .eq(requiredProperties.length);
      var embComplete = ee.Number(feature.get('n_complete_emb'))
        .eq(embeddingProperties.length);
      var inV34 = ee.Number(ee.Algorithms.If(
        ee.Algorithms.IsEqual(feature.get('_v34'), null), 0, 1));
      return ee.Feature(null, {
        row_id: feature.get('row_id'),
        presence: feature.get('presence'),
        lon: coordinates.get(0),
        lat: coordinates.get(1),
        in_current_rule: inRule,
        embeddings_complete: embComplete,
        in_v34_filter_reconstruction: inV34,
        in_v34_reconstruction_and_embeddings_complete: inV34.and(embComplete)
      });
    });
  exportTable({
    collection: rowSets,
    description: 'SDM_TrainingRowSets_' + schemeName + '_' + NEW_SUFFIX,
    folder: EXPORT_FOLDER,
    fileFormat: 'CSV',
    selectors: ['run_id', 'code_version', 'analysis_scheme', 'row_id',
                'presence', 'lon', 'lat', 'in_current_rule',
                'embeddings_complete', 'in_v34_filter_reconstruction',
                'in_v34_reconstruction_and_embeddings_complete']
  }, schemeName);

  print(schemeName + ' training counts (no cap):',
    training.aggregate_histogram('presence'));

  var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: N_TREES,
    variablesPerSplit: FIXED_VARIABLES_PER_SPLIT,
    minLeafPopulation: FIXED_MIN_LEAF,
    seed: SEED
  }).train(training, 'presence', MODEL_BANDS)
    .setOutputMode('PROBABILITY');
  var multiClassifier = classifier.setOutputMode('MULTIPROBABILITY');
  var classClassifier = classifier.setOutputMode('CLASSIFICATION');

  function scoreFeatures(fc, name) {
    if (PRESENCE_SCORE_MODE === 'PROBABILITY') {
      return fc.classify(classifier, name);
    }
    return fc.classify(multiClassifier, '_multi').map(function (feature) {
      return feature.set(name,
        ee.Array(feature.get('_multi')).get([CLASS1_INDEX]));
    });
  }

  function modeCheckRows(fc) {
    return fc.classify(classifier, 'p_single')
      .classify(multiClassifier, 'p_multi')
      .classify(classClassifier, 'p_class')
      .map(function (feature) {
        var arr = ee.Array(feature.get('p_multi'));
        var m0 = ee.Number(arr.get([0]));
        var m1 = ee.Number(arr.get([1]));
        var single = ee.Number(feature.get('p_single'));
        var label1 = ee.Number(feature.get('p_class')).eq(1);
        return feature.set({
          d_col0: single.subtract(m0).abs(),
          d_col1: single.subtract(m1).abs(),
          pred_label: label1,
          clear_winner: m0.subtract(m1).abs().gt(CLEAR_WINNER_MARGIN),
          argmax_col1_matches_label1: m1.gt(m0).eq(label1),
          argmax_col0_matches_label1: m0.gt(m1).eq(label1)
        });
      });
  }
  function modeCheckSummary(rows, prefix) {
    var clear = rows.filter(ee.Filter.eq('clear_winner', 1));
    var out = {};
    out[prefix + '_n'] = rows.size();
    out[prefix + '_n_clear'] = clear.size();
    out[prefix + '_share_argmax_col1_is_label1'] =
      clear.aggregate_mean('argmax_col1_matches_label1');
    out[prefix + '_share_argmax_col0_is_label1'] =
      clear.aggregate_mean('argmax_col0_matches_label1');
    [1, 0].forEach(function (g) {
      var grp = rows.filter(ee.Filter.eq('pred_label', g));
      out[prefix + '_pred' + g + '_n'] = grp.size();
      out[prefix + '_pred' + g + '_max_abs_diff_col0'] = grp.aggregate_max('d_col0');
      out[prefix + '_pred' + g + '_max_abs_diff_col1'] = grp.aggregate_max('d_col1');
    });
    return out;
  }
  var checkedTraining = modeCheckRows(training);
  var mappedSample = prewarFeatures.sample({
    region: roi, projection: refProjection, numPixels: CHECK_SAMPLE_PIXELS,
    seed: SEED + 11, geometries: false, tileScale: 16
  }).merge(annualFeatures[2024].sample({
    region: roi, projection: refProjection, numPixels: CHECK_SAMPLE_PIXELS,
    seed: SEED + 12, geometries: false, tileScale: 16
  }));
  var checkedMapped = modeCheckRows(mappedSample);
  var checkProps = {
    sampling_scheme: schemeName,
    presence_score_mode: PRESENCE_SCORE_MODE,
    configured_class1_index: CLASS1_INDEX,
    clear_winner_margin: CLEAR_WINNER_MARGIN,
    presence_labels: training.aggregate_array('presence').distinct().sort(),
    mean_probability_presences: checkedTraining
      .filter(ee.Filter.eq('presence', 1)).aggregate_mean('p_single'),
    mean_probability_pseudo_absences: checkedTraining
      .filter(ee.Filter.eq('presence', 0)).aggregate_mean('p_single'),
    note: 'label-1 column = the column whose argmax matches CLASSIFICATION ' +
          '== 1; PROBABILITY must equal it in both predicted groups; the ' +
          'verdict is computed in Python_01'
  };
  var trainSummary = modeCheckSummary(checkedTraining, 'train');
  var mappedSummary = modeCheckSummary(checkedMapped, 'mapped');
  for (var k1 in trainSummary) { checkProps[k1] = trainSummary[k1]; }
  for (var k2 in mappedSummary) { checkProps[k2] = mappedSummary[k2]; }
  exportTable({
    collection: ee.FeatureCollection([ee.Feature(null, checkProps)]),
    description: 'SDM_ProbabilityModeCheck_' + schemeName + '_' + NEW_SUFFIX,
    folder: EXPORT_FOLDER,
    fileFormat: 'CSV'
  }, schemeName);

  var nNew = training.size();
  var rowCheck = ee.FeatureCollection([ee.Feature(null, {
    sampling_scheme: schemeName,
    n_rows_sampled_table: sampledTable.size(),
    n_rows_current_rule: nNew,
    n_presence_current_rule: training.filter(ee.Filter.eq('presence', 1)).size(),
    n_rows_v34_filter_reconstruction: v34Reconstruction.size(),
    n_rows_v34_reconstruction_and_embeddings_complete: rowSets.filter(
      ee.Filter.eq('in_v34_reconstruction_and_embeddings_complete', 1)).size(),
    n_rows_current_rule_not_in_v34_reconstruction: rowSets.filter(ee.Filter.and(
      ee.Filter.eq('in_current_rule', 1),
      ee.Filter.eq('in_v34_filter_reconstruction', 0))).size(),
    n_rows_v34_reconstruction_not_in_current_rule: rowSets.filter(ee.Filter.and(
      ee.Filter.eq('in_current_rule', 0),
      ee.Filter.eq('in_v34_filter_reconstruction', 1))).size(),
    v34_filter_reconstructed: schemeName === 'primary_effort'
      ? 'ee.Filter.notNull(64 embeddings + 11 baseline predictors + presence)'
      : 'ee.Filter.notNull(64 embeddings + presence)',
    current_rule: 'toDictionary count: 64 embeddings + 11 baseline ' +
                  'predictors + presence all non-null',
    run_mode: mode
  })]);
  exportTable({
    collection: rowCheck,
    description: 'SDM_TrainingRowCheck_' + schemeName + '_' + NEW_SUFFIX,
    folder: EXPORT_FOLDER,
    fileFormat: 'CSV'
  }, schemeName);
  print(schemeName + ': rows current rule / v34 filter reconstruction:', nNew,
        v34Reconstruction.size());

  function probability(image, name) {
    if (PRESENCE_SCORE_MODE === 'PROBABILITY') {
      return image.classify(classifier).rename(name).toFloat();
    }
    return image.classify(multiClassifier).arrayGet([CLASS1_INDEX])
      .rename(name).toFloat();
  }

  var probPrewar = probability(prewarFeatures, 'prob_prewar');
  var probWartime = probability(wartimeFeatures, 'prob_wartime');
  var annualProbabilities = {};
  YEARS.forEach(function (year) {
    annualProbabilities[year] = probability(
      annualFeatures[year], 'prob_' + year);
  });
  var prewarAnnualStack = ee.Image.cat([
    annualProbabilities[2019], annualProbabilities[2020],
    annualProbabilities[2021]
  ]);
  var warAnnualStack = ee.Image.cat([
    annualProbabilities[2022], annualProbabilities[2023],
    annualProbabilities[2024]
  ]);
  var probWarMean = warAnnualStack.reduce(ee.Reducer.mean())
    .rename('prob_war_mean').toFloat();
  var probWarSd = warAnnualStack.reduce(ee.Reducer.stdDev())
    .rename('prob_war_sd').toFloat();
  var diffComposite = probPrewar.subtract(probWartime)
    .rename('diff_prewar_minus_wartime').toFloat();
  var prewarAnnualMedian = prewarAnnualStack.reduce(ee.Reducer.median())
    .rename('prob_prewar_annual_median').toFloat();
  var warAnnualMedian = warAnnualStack.reduce(ee.Reducer.median())
    .rename('prob_wartime_annual_median').toFloat();
  var annualMedianDifference = prewarAnnualMedian.subtract(warAnnualMedian)
    .rename('diff_annual_medians').toFloat();

  var periodStack = ee.Image.cat([
    probPrewar, probWartime, probWarMean, probWarSd, diffComposite
  ]).clip(roi).toFloat();
  var annualStack = ee.Image.cat(YEARS.map(function (year) {
    return annualProbabilities[year];
  })).clip(roi).toFloat();
  var annualMedianSensitivity = ee.Image.cat([
    prewarAnnualMedian, warAnnualMedian, annualMedianDifference
  ]).clip(roi).toFloat();

  // Same-run novelty envelope from the exact 64 predictors and fitting table.
  var noveltyReducer = ee.Reducer.mean()
    .combine(ee.Reducer.stdDev(), '', true)
    .combine(ee.Reducer.percentile([2.5, 97.5], ['p2_5', 'p97_5']), '', true);
  var bandStats = MODEL_BANDS.map(function (band) {
    return training.reduceColumns(noveltyReducer, [band]);
  });
  function statImage(key) {
    return ee.Image.constant(ee.List(bandStats.map(function (stats) {
      return ee.Dictionary(stats).get(key);
    }))).rename(MODEL_BANDS).toFloat();
  }
  var trainingMean = statImage('mean');
  var trainingSd = statImage('stdDev').max(1e-6);
  var trainingLow = statImage('p2_5');
  var trainingHigh = statImage('p97_5');

  function noveltyMetrics(image, prefix) {
    var z = image.select(MODEL_BANDS).subtract(trainingMean)
      .divide(trainingSd);
    var distance = z.pow(2).reduce(ee.Reducer.sum())
      .divide(MODEL_BANDS.length).sqrt().rename(prefix + '_distance_std');
    var outside = image.select(MODEL_BANDS).lt(trainingLow)
      .or(image.select(MODEL_BANDS).gt(trainingHigh))
      .reduce(ee.Reducer.sum()).rename(prefix + '_bands_outside');
    var maxAbsZ = z.abs().reduce(ee.Reducer.max())
      .rename(prefix + '_max_abs_z');
    return ee.Image.cat([distance, outside, maxAbsZ]).toFloat();
  }

  var trainingNoveltyImage = noveltyMetrics(prewarFeatures, 'train');
  var noveltyAtTraining = trainingNoveltyImage.sampleRegions({
    collection: training,
    properties: ['presence'],
    projection: refProjection,
    geometries: false,
    tileScale: 16
  });
  function noveltyThreshold(property) {
    return ee.Number(noveltyAtTraining.reduceColumns(
      ee.Reducer.percentile([NOVELTY_PERCENTILE]), [property])
      .get('p' + NOVELTY_PERCENTILE));
  }
  var thresholdDistance = noveltyThreshold('train_distance_std');
  var thresholdOutside = noveltyThreshold('train_bands_outside');
  var thresholdMaxZ = noveltyThreshold('train_max_abs_z');
  function flagNovelty(metrics, prefix) {
    return metrics.select(prefix + '_distance_std').gt(thresholdDistance)
      .or(metrics.select(prefix + '_bands_outside').gt(thresholdOutside))
      .or(metrics.select(prefix + '_max_abs_z').gt(thresholdMaxZ));
  }

  var noveltyPrewar = noveltyMetrics(prewarFeatures, 'prewar');
  var noveltyWartime = noveltyMetrics(wartimeFeatures, 'wartime');
  var novelty2019 = noveltyMetrics(annualFeatures[2019], 'annual_2019');
  var novelty2020 = noveltyMetrics(annualFeatures[2020], 'annual_2020');
  var novelty2021 = noveltyMetrics(annualFeatures[2021], 'annual_2021');
  var novelty2022 = noveltyMetrics(annualFeatures[2022], 'annual_2022');
  var novelty2023 = noveltyMetrics(annualFeatures[2023], 'annual_2023');
  var novelty2024 = noveltyMetrics(annualFeatures[2024], 'annual_2024');
  var flagPrewar = flagNovelty(noveltyPrewar, 'prewar');
  var flagWartime = flagNovelty(noveltyWartime, 'wartime');
  var noveltyOverall = flagPrewar.or(flagWartime)
    .rename('extrapolation_flag').toByte();
  var noveltyAnnualAny = flagNovelty(novelty2019, 'annual_2019')
    .or(flagNovelty(novelty2020, 'annual_2020'))
    .or(flagNovelty(novelty2021, 'annual_2021'))
    .or(flagNovelty(novelty2022, 'annual_2022'))
    .or(flagNovelty(novelty2023, 'annual_2023'))
    .or(flagNovelty(novelty2024, 'annual_2024'))
    .rename('extrapolation_flag_annual').toByte();
  var noveltyStack = ee.Image.cat([
    noveltyOverall,
    flagPrewar.rename('flag_prewar'),
    flagWartime.rename('flag_wartime'),
    flagNovelty(novelty2022, 'annual_2022').rename('flag_annual_2022'),
    flagNovelty(novelty2023, 'annual_2023').rename('flag_annual_2023'),
    flagNovelty(novelty2024, 'annual_2024').rename('flag_annual_2024'),
    noveltyPrewar, noveltyWartime, novelty2022, novelty2023, novelty2024,
    flagNovelty(novelty2019, 'annual_2019').rename('flag_annual_2019'),
    flagNovelty(novelty2020, 'annual_2020').rename('flag_annual_2020'),
    flagNovelty(novelty2021, 'annual_2021').rename('flag_annual_2021'),
    noveltyAnnualAny,
    novelty2019, novelty2020, novelty2021
  ]).clip(roi).toFloat();

  // The novelty asset is exported in 'novelty' and 'all' mode, the
  // probability assets and GeoTIFFs in 'all' mode, all with the suffix
  var noveltyAsset = OWN + 'SDM_Novelty_' + schemeName +
    '_outdoor_final' + sfx;
  if (mode === 'novelty' || mode === 'all') {
    exportAsset(noveltyStack, 'FinalNovelty_' + schemeName + sfx, noveltyAsset,
      schemeName);
    exportGeoTiff(noveltyAnnualAny,
      'SDM_extrapolation_flag_annual_SINGLE_' + schemeName + sfx);
  }
  if (mode === 'all') {
    var periodAsset = OWN + 'SDM_Probability_PeriodComposites_' +
      schemeName + '_outdoor_final' + sfx;
    var annualAsset = OWN + 'SDM_Probability_Annual2019_2024_' +
      schemeName + '_outdoor_final' + sfx;
    exportAsset(periodStack, 'FinalPeriod_' + schemeName + sfx, periodAsset,
      schemeName);
    exportAsset(annualStack, 'FinalAnnual_' + schemeName + sfx, annualAsset,
      schemeName);
    exportGeoTiff(periodStack, 'SDM_Period_' + schemeName + sfx);
    exportGeoTiff(annualStack, 'SDM_Annual_' + schemeName + sfx);
    exportGeoTiff(annualMedianSensitivity,
      'SDM_AnnualMedianSensitivity_' + schemeName + sfx);
    // Standalone files avoid ambiguous multiband handling in desktop GIS.
    exportGeoTiff(probPrewar, 'SDM_prob_prewar_SINGLE_' + schemeName + sfx);
    exportGeoTiff(probWartime, 'SDM_prob_wartime_SINGLE_' + schemeName + sfx);
    exportGeoTiff(diffComposite, 'SDM_diff_SINGLE_' + schemeName + sfx);
    exportGeoTiff(noveltyOverall,
      'SDM_extrapolation_flag_SINGLE_' + schemeName + sfx);
  }

  var importance = ee.Dictionary(classifier.explain().get('importance'));
  var importanceTable = ee.FeatureCollection(MODEL_BANDS.map(function (band) {
    return ee.Feature(null, {
      predictor: band,
      importance: importance.get(band),
      sampling_scheme: schemeName,
      run_tag: scheme.runTag,
      feature_representation: 'original64_direct500m'
    });
  }));
  if (mode === 'all') exportTable({
    collection: importanceTable,
    description: 'SDM_VariableImportance_' + schemeName + sfx,
    folder: EXPORT_FOLDER,
    fileFormat: 'CSV'
  }, schemeName);

  var interpretabilitySample = ee.Image.cat([
    probPrewar.rename('predicted_probability'),
    prewarNdvi,
    worldCover.toFloat(),
    noveltyOverall.toFloat(),
    trainingLandcoverDomain.toFloat()
  ]).sample({
    region: roi,
    projection: refProjection,
    numPixels: 50000,
    seed: SEED,
    geometries: false,
    tileScale: 16
  }).map(function (feature) {
    return feature.set({
      sampling_scheme: schemeName,
      run_tag: scheme.runTag,
      prediction_domain: 'all_valid_embedding_pixels'
    });
  });
  if (mode === 'all') exportTable({
    collection: interpretabilitySample,
    description: 'SDM_InterpretabilitySample_' + schemeName + sfx,
    folder: EXPORT_FOLDER,
    fileFormat: 'CSV'
  }, schemeName);

  var noveltyThresholds = ee.FeatureCollection([ee.Feature(null, {
    sampling_scheme: schemeName,
    run_tag: scheme.runTag,
    percentile: NOVELTY_PERCENTILE,
    distance_std_threshold: thresholdDistance,
    bands_outside_threshold: thresholdOutside,
    max_abs_z_threshold: thresholdMaxZ,
    predictor_count: MODEL_BANDS.length
  })]);
  if (mode !== 'checks') exportTable({
    collection: noveltyThresholds,
    description: 'SDM_NoveltyThresholds_' + schemeName + sfx,
    folder: EXPORT_FOLDER,
    fileFormat: 'CSV'
  }, schemeName);

  var fittedPoints = scoreFeatures(training, 'classification').map(function (feature) {
    var coordinates = feature.geometry().coordinates();
    return ee.Feature(null, {
      row_id: feature.get('row_id'),
      emb_fingerprint: feature.get('emb_fingerprint'),
      presence: feature.get('presence'),
      fitted_probability: feature.get('classification'),
      lon: coordinates.get(0),
      lat: coordinates.get(1),
      esa: feature.get('esa'),
      ndvi: feature.get('ndvi'),
      dist_infra_m: feature.get('dist_infra_m'),
      log_effort: feature.get('log_effort'),
      in_ordlo: feature.get('in_ordlo'),
      sampling_scheme: schemeName
    });
  });
  exportTable({
    collection: fittedPoints,
    description: 'SDM_FullModelPointPredictions_' + schemeName + sfx,
    folder: EXPORT_FOLDER,
    fileFormat: 'CSV'
  }, schemeName);

  var bandManifestRows = [
    ['period', 1, 'prob_prewar', 'median 2019-2021 feature representation'],
    ['period', 2, 'prob_wartime', 'median 2022-2024 feature representation'],
    ['period', 3, 'prob_war_mean', 'mean annual probability 2022-2024'],
    ['period', 4, 'prob_war_sd', 'SD of annual probability 2022-2024'],
    ['period', 5, 'diff_prewar_minus_wartime', 'prewar minus wartime'],
    ['annual', 1, 'prob_2019', 'annual probability'],
    ['annual', 2, 'prob_2020', 'annual probability'],
    ['annual', 3, 'prob_2021', 'annual probability'],
    ['annual', 4, 'prob_2022', 'annual probability'],
    ['annual', 5, 'prob_2023', 'annual probability'],
    ['annual', 6, 'prob_2024', 'annual probability'],
    ['novelty', 1, 'extrapolation_flag', 'prewar or wartime median embedding flagged'],
    ['novelty', 22, 'flag_annual_2019', 'annual 2019 embedding flagged (v13b11)'],
    ['novelty', 23, 'flag_annual_2020', 'annual 2020 embedding flagged (v13b11)'],
    ['novelty', 24, 'flag_annual_2021', 'annual 2021 embedding flagged (v13b11)'],
    ['novelty', 25, 'extrapolation_flag_annual',
     'any annual embedding 2019-2024 flagged (v13b11; used by GEE_05)']
  ];
  var bandManifest = ee.FeatureCollection(bandManifestRows.map(function (row) {
    return ee.Feature(null, {
      file_group: row[0], band_index_one_based: row[1], band_name: row[2],
      definition: row[3], sampling_scheme: schemeName,
      crs: EXPORT_CRS, pixel_size_m: REF_SCALE, nodata: NODATA_VALUE,
      worldcover_prediction_mask_applied: 0
    });
  }));
  if (mode !== 'checks') exportTable({
    collection: bandManifest,
    description: 'SDM_BandManifest_' + schemeName + sfx,
    folder: EXPORT_FOLDER,
    fileFormat: 'CSV'
  }, schemeName);

  var metadata = ee.FeatureCollection([ee.Feature(null, {
    pipeline_version: 'v13b_direct_embeddings',
    presence_score_mode: PRESENCE_SCORE_MODE,
    class1_index_if_multiprobability: CLASS1_INDEX,
    sampling_scheme: schemeName,
    run_tag: scheme.runTag,
    seed: SEED,
    number_of_trees: N_TREES,
    min_leaf: FIXED_MIN_LEAF,
    variables_per_split: FIXED_VARIABLES_PER_SPLIT,
    hyperparameter_source: 'predeclared_before_v13b_validation',
    model_fitting_sample_cap: 'none',
    predictor_representation: 'original64_direct500m',
    aggregation: 'annual_tile_mosaic_nearest_neighbour_at_500m_export',
    export_crs: EXPORT_CRS,
    pixel_size_m: REF_SCALE,
    worldcover_training_filter_applied_symmetrically: 1,
    training_esa_classes: TRAINING_ESA_CLASSES.join(','),
    worldcover_prediction_mask_applied: 0,
    mapping_domain: 'all_valid_embedding_pixels_within_AOI',
    outside_training_landcover_role: 'transfer_prediction_with_novelty_diagnostic',
    quantity: scheme.recurrent ?
      'recurrently_observed_outdoor_visitation_propensity' :
      'outdoor_visitation_propensity'
  })]);
  if (mode === 'all') exportTable({
    collection: metadata,
    description: 'SDM_FinalMapMetadata_' + schemeName + sfx,
    folder: EXPORT_FOLDER,
    fileFormat: 'CSV'
  }, schemeName);

  // outputs of this run, and the settings. GEE_05 reads run_id from the
  // assets and Python_03 compares the two manifests.
  var sampledAssetId = OWN + 'SDM_Sampled_all_' + schemeName +
    '_outdoor_v13b';
  var manifestProps = {
    sampling_scheme: schemeName,
    run_started_utc: RUN_STARTED_UTC,
    run_mode: mode,
    presence_score_mode: PRESENCE_SCORE_MODE,
    class1_index_if_multiprobability: CLASS1_INDEX,
    number_of_trees: N_TREES, min_leaf: FIXED_MIN_LEAF,
    variables_per_split: FIXED_VARIABLES_PER_SPLIT, seed: SEED,
    novelty_percentile: NOVELTY_PERCENTILE,
    sampled_table_asset: sampledAssetId,
    sampled_table_asset_version: ee.FeatureCollection(sampledAssetId)
      .get('system:version'),
    n_rows_sampled_table: sampledTable.size(),
    n_training_rows: training.size(),
    novelty_asset: noveltyAsset,
    period_probability_asset: mode === 'all' ? OWN +
      'SDM_Probability_PeriodComposites_' + schemeName + '_outdoor_final' +
      sfx : 'not_exported_in_this_mode',
    annual_probability_asset: mode === 'all' ? OWN +
      'SDM_Probability_Annual2019_2024_' + schemeName + '_outdoor_final' +
      sfx : 'not_exported_in_this_mode'
  };
  YEARS.forEach(function (year) {
    var id = OWN + 'EmbeddingAnnual_' + year + '_EPSG3035_500m_v13b';
    manifestProps['embedding_' + year + '_asset'] = id;
    manifestProps['embedding_' + year + '_asset_version'] =
      ee.Image(id).get('system:version');
  });
  exportTable({
    collection: ee.FeatureCollection([ee.Feature(null, manifestProps)]),
    description: 'GEE04_RunManifest_' + schemeName + '_' + NEW_SUFFIX,
    folder: EXPORT_FOLDER,
    fileFormat: 'CSV'
  }, schemeName);
}

runFinalModel('primary_effort');
runFinalModel('recurrent_effort');
print('GEE_04 run_id:', RUN_ID);

print('GEE_04 v13b12 run modes:', RUN_MODE, '| presence score:',
      PRESENCE_SCORE_MODE,
      '-- start the image-to-Asset tasks first, then the table and GeoTIFF tasks.');
print('Prediction mask:', 'none beyond AOI and valid embedding inputs.');
