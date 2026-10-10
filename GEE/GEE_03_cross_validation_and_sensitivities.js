// MAIN VALIDATION. RUN ONCE AFTER GEE_02.
/**
 * Exports fixed-model five-fold scores for all sampling schemes. The
 * environmental baseline (11 predictors) and the NDVI-only model are fitted
 * for both the regular (primary_effort) and the recurrent (recurrent_effort)
 * model, on exactly the same complete-case rows as the embedding model; the
 * sensitivity schemes (accessibility-matched, full 2015-2021 window) fit the
 * embedding model only.
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
// outputs/GEE03_cross_validation (support/folders.py).
var EXPORT_FOLDER = 'GEE03_cross_validation';
var SCHEMES_TO_RUN = ['primary_effort', 'access_matched', 'recurrent_effort',
                      'recurrent_access_matched', 'primary_effort_full'];
var CODE_VERSION = 'v13b12';
var FILE_TAG = '_' + CODE_VERSION;
var RUN_STARTED_UTC = new Date().toISOString();
var RUN_ID = 'GEE03_' + CODE_VERSION + '_' +
  RUN_STARTED_UTC.replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');
var PRESENCE_SCORE_MODE = 'PROBABILITY';   // or 'MULTIPROBABILITY_CLASS1'
var CLASS1_INDEX = 1;
var SEED = 20250523;
var N_TREES = 500;
var N_FOLDS = 5;
var MAIN_BLOCK_SIZE = 0.30;
// Block size per scheme, in degrees. At the AOI mean latitude (48.9 deg N) a
// block of B degrees spans ~73.2*B km east-west and ~110.6*B km north-south:
// 0.30 deg = 22.0 x 33.2 km (regular model; residual correlation range 20 km),
// 0.45 deg = 32.9 x 49.8 km (recurrent model; range 40 km, which exceeds the
// east-west width, so 0.60 deg is evaluated as well, see Text S1).
// Each scheme's sensitivity grid brackets its own main size.
var MAIN_BLOCK_SIZE_BY_SCHEME = {
  primary_effort: 0.30,
  primary_effort_full: 0.30,
  access_matched: 0.30,
  recurrent_effort: 0.45,
  recurrent_access_matched: 0.45
};
var BLOCK_SIZE_SENSITIVITY_BY_SCHEME = {
  primary_effort: [0.15, 0.60],
  recurrent_effort: [0.30, 0.60]
};
var N_REPEATED_ASSIGNMENTS = 3;
var FIXED_MIN_LEAF = 20;
var FIXED_VARIABLES_PER_SPLIT = 16;
var BASELINE_VARIABLES_PER_SPLIT = 3;
var NDVI_NULL_VARIABLES_PER_SPLIT = 1;
var MAX_TRAIN_SCORE_SAMPLES_PER_CLASS = 3000;
var REGULARISATION_SENSITIVITY = [
  {leaf: 5,  mtry: 8,  role: 'old_default'},
  {leaf: 5,  mtry: 16, role: 'leaf_contrast'},
  {leaf: 20, mtry: 8,  role: 'mtry_contrast'},
  {leaf: 20, mtry: 21, role: 'mtry_contrast'},
  {leaf: 50, mtry: 16, role: 'leaf_contrast'}
];

// blocked folds). Requires the primary_effort_full table from GEE_02.
var RUN_FULL_WINDOW_SENSITIVITY = true;
var COMPARISON_SCHEMES = ['primary_effort', 'recurrent_effort'];
var DIAGNOSTIC_SCHEMES = ['primary_effort', 'recurrent_effort'];
var SCHEME_LABELS = {
  primary_effort: 'Regular outdoor visitation model (2019-2021) [main]',
  primary_effort_full:
    'Regular outdoor visitation model, full 2015-2021 window [sensitivity]',
  access_matched:
    'Regular outdoor visitation model, accessibility-matched [sensitivity]',
  recurrent_effort:
    'Recurrent outdoor visitation model (2019-2021) [comparison]',
  recurrent_access_matched:
    'Recurrent outdoor visitation model, accessibility-matched [sensitivity]'
};
var MODEL_LABELS = {
  embeddings: 'Satellite-embedding model (64 predictors)',
  baseline_simple: 'Environmental baseline (11 predictors)',
  ndvi_null: 'NDVI-only null (1 predictor)'
};

var SCHEMES = {
  primary_effort: {
    runTag: 'nondvi_outdoor_effort_guard0_matched2019_2021', paired: false
  },
  primary_effort_full: {
    runTag: 'nondvi_outdoor_effort_guard0_full2015_2021', paired: false
  },
  access_matched: {
    runTag: 'nondvi_outdoor_accessmatch_guard0_matched2019_2021', paired: true
  },
  recurrent_effort: {
    runTag: 'recurrent_core_nondvi_effort_guard0_matched2019_2021', paired: false
  },
  recurrent_access_matched: {
    runTag: 'recurrent_core_nondvi_accessmatch_guard0_matched2019_2021',
    paired: true
  }
};

var EMBEDDING_BANDS = [];
for (var bandIndex = 0; bandIndex < 64; bandIndex++) {
  EMBEDDING_BANDS.push('A' + (bandIndex < 10 ? '0' : '') + bandIndex);
}
var MODEL_EMBEDDING_BANDS = EMBEDDING_BANDS;
var FINGERPRINT_WEIGHTS = ee.Array(EMBEDDING_BANDS.map(function (b, i) {
  return i + 1;
}));
function embFingerprint(feature) {
  return ee.Array(feature.toArray(EMBEDDING_BANDS)).multiply(FINGERPRINT_WEIGHTS)
    .reduce(ee.Reducer.sum(), [0]).get([0]);
}
var LAND_COVER_BANDS = [
  'lc_tree', 'lc_shrub', 'lc_grass', 'lc_water', 'lc_wetland',
  'lc_moss_lichen'
];
var BASELINE_BANDS = ['ndvi'].concat(LAND_COVER_BANDS)
  .concat(['dist_water_km', 'elevation', 'slope', 'tpi']);

var aoi = ee.FeatureCollection(AOI_ASSET);
var roi = aoi.geometry();
var boundsCoordinates = ee.List(roi.bounds().coordinates().get(0));
var X_MIN = ee.Number(ee.List(boundsCoordinates.get(0)).get(0));
var Y_MIN = ee.Number(ee.List(boundsCoordinates.get(0)).get(1));

function modelDefinitions(schemeName) {
  var sampleAsset = OWN + 'SDM_Sampled_all_' +
    schemeName + '_outdoor_v13b';
  return {
    embeddings: {
      sampleAsset: sampleAsset,
      bands: MODEL_EMBEDDING_BANDS,
      mtry: FIXED_VARIABLES_PER_SPLIT
    },
    baseline_simple: {
      sampleAsset: sampleAsset,
      bands: BASELINE_BANDS,
      mtry: BASELINE_VARIABLES_PER_SPLIT
    },
    ndvi_null: {
      sampleAsset: sampleAsset,
      bands: ['ndvi'],
      mtry: NDVI_NULL_VARIABLES_PER_SPLIT
    }
  };
}

function addBlockId(feature, blockSize) {
  var coordinates = feature.geometry().coordinates();
  var blockX = ee.Number(coordinates.get(0)).subtract(X_MIN)
    .divide(blockSize).floor();
  var blockY = ee.Number(coordinates.get(1)).subtract(Y_MIN)
    .divide(blockSize).floor();
  var blockId = blockX.multiply(100000).add(blockY).toInt64();
  return feature.set({
    block_id: blockId,
    block_key: blockId.format('%.0f'),
    block_size: blockSize
  });
}

function blockSummary(pointsWithBlocks) {
  var presenceCounts = ee.Dictionary(pointsWithBlocks
    .filter(ee.Filter.eq('presence', 1)).aggregate_histogram('block_key'));
  var absenceCounts = ee.Dictionary(pointsWithBlocks
    .filter(ee.Filter.eq('presence', 0)).aggregate_histogram('block_key'));
  var totalCounts = ee.Dictionary(
    pointsWithBlocks.aggregate_histogram('block_key'));
  return ee.FeatureCollection(totalCounts.keys().sort().map(function (key) {
    key = ee.String(key);
    var blockId = ee.Number.parse(key);
    var nPresence = ee.Number(presenceCounts.get(key, 0));
    var nAbsence = ee.Number(absenceCounts.get(key, 0));
    var nTotal = ee.Number(totalCounts.get(key, 0));
    return ee.Feature(null, {
      block_key: key,
      block_id: blockId,
      n_presence: nPresence,
      n_absence: nAbsence,
      n_total: nTotal,
      deterministic_order: nTotal.multiply(1e9).subtract(blockId)
    });
  }));
}

function assignBalancedFolds(blocks, foldProperty, numberOfFolds,
                             orderProperty, ascending) {
  var sorted = blocks.sort(orderProperty, ascending);
  var blockList = sorted.toList(sorted.size());
  var numberOfBlocks = sorted.size();
  var totalPresence = ee.Number(sorted.aggregate_sum('n_presence')).max(1);
  var totalAbsence = ee.Number(sorted.aggregate_sum('n_absence')).max(1);
  var initialState = ee.Dictionary({
    foldPresence: ee.List.repeat(0, numberOfFolds),
    foldAbsence: ee.List.repeat(0, numberOfFolds),
    assignments: ee.List([])
  });
  var finalState = ee.Dictionary(ee.List.sequence(0, numberOfBlocks.subtract(1))
    .iterate(function (index, stateObject) {
      var state = ee.Dictionary(stateObject);
      var block = ee.Feature(blockList.get(index));
      var blockPresence = ee.Number(block.get('n_presence'));
      var blockAbsence = ee.Number(block.get('n_absence'));
      var foldPresence = ee.List(state.get('foldPresence'));
      var foldAbsence = ee.List(state.get('foldAbsence'));
      var choice = ee.Dictionary(ee.List.sequence(0, numberOfFolds - 1)
        .iterate(function (fold, bestObject) {
          fold = ee.Number(fold);
          var best = ee.Dictionary(bestObject);
          var score = ee.Number(foldPresence.get(fold)).add(blockPresence)
            .divide(totalPresence)
            .add(ee.Number(foldAbsence.get(fold)).add(blockAbsence)
              .divide(totalAbsence));
          return ee.Algorithms.If(score.lt(ee.Number(best.get('score'))),
            ee.Dictionary({fold: fold, score: score}), best);
        }, ee.Dictionary({fold: 0, score: 1e30})));
      var selectedFold = ee.Number(choice.get('fold'));
      return ee.Dictionary({
        foldPresence: foldPresence.set(selectedFold,
          ee.Number(foldPresence.get(selectedFold)).add(blockPresence)),
        foldAbsence: foldAbsence.set(selectedFold,
          ee.Number(foldAbsence.get(selectedFold)).add(blockAbsence)),
        assignments: ee.List(state.get('assignments')).add(
          ee.List([index, selectedFold]))
      });
    }, initialState));
  var assignments = ee.List(finalState.get('assignments'));
  return ee.FeatureCollection(ee.List.sequence(0, numberOfBlocks.subtract(1))
    .map(function (index) {
      return ee.Feature(blockList.get(index)).set(
        foldProperty,
        ee.Number(ee.List(assignments.get(index)).get(1)));
    }));
}

function addBlockedFolds(sourcePoints, blockSize, foldProperty,
                         randomOrderSeed) {
  var pointsWithBlocks = sourcePoints.map(function (feature) {
    return addBlockId(feature, blockSize);
  });
  var blocks = blockSummary(pointsWithBlocks);
  var orderProperty = 'deterministic_order';
  var ascending = false;
  if (randomOrderSeed !== null) {
    blocks = blocks.randomColumn(
      'random_fold_order', randomOrderSeed, 'uniform', ['block_key']);
    orderProperty = 'random_fold_order';
    ascending = true;
  }
  var assignedBlocks = assignBalancedFolds(
    blocks, foldProperty, N_FOLDS, orderProperty, ascending);
  var lookup = ee.Dictionary.fromLists(
    assignedBlocks.aggregate_array('block_key'),
    assignedBlocks.aggregate_array(foldProperty));
  return pointsWithBlocks.map(function (feature) {
    var key = ee.String(feature.get('block_key'));
    var balanced = ee.Number(lookup.get(key, -1));
    var fallback = ee.Number(feature.get('block_id')).mod(N_FOLDS);
    var usedFallback = balanced.lt(0);
    var fold = ee.Number(
      ee.Algorithms.If(usedFallback, fallback, balanced)).toInt();
    return feature
      .set(foldProperty, fold)
      .set(foldProperty + '_fallback', usedFallback.toInt());
  });
}

function retainCompletePairs(rows) {
  var p = rows.filter(ee.Filter.eq('presence', 1));
  var a = rows.filter(ee.Filter.eq('presence', 0));
  var pairCondition = ee.Filter.equals({
    leftField: 'match_pair_id', rightField: 'match_pair_id'
  });
  function matchedSide(left, right) {
    return ee.FeatureCollection(ee.Join.saveFirst('_matched_pair').apply({
      primary: left, secondary: right, condition: pairCondition
    })).filter(ee.Filter.notNull(['_matched_pair'])).map(function (feature) {
      feature = ee.Feature(feature);
      return feature.select(feature.propertyNames().remove('_matched_pair'));
    });
  }
  return matchedSide(p, a).merge(matchedSide(a, p));
}

function addRandomFolds(sourcePoints, paired, randomSeed) {
  var rowKeys = paired
    ? ['match_pair_id']
    : ['presence', 'system:index'];
  return sourcePoints.randomColumn(
    'random_fold_u', randomSeed, 'uniform', rowKeys).map(function (feature) {
      return feature.set(
        'random_fold', ee.Number(feature.get('random_fold_u'))
          .multiply(N_FOLDS).floor().min(N_FOLDS - 1).toInt());
    });
}

function trainForest(training, bands, minimumLeaf, variablesPerSplit,
                     modelSeed) {
  return ee.Classifier.smileRandomForest({
    numberOfTrees: N_TREES,
    variablesPerSplit: variablesPerSplit,
    minLeafPopulation: minimumLeaf,
    seed: modelSeed
  }).train(training, 'presence', bands).setOutputMode(
    PRESENCE_SCORE_MODE === 'PROBABILITY' ? 'PROBABILITY' : 'MULTIPROBABILITY');
}
function presenceScore(feature) {
  return PRESENCE_SCORE_MODE === 'PROBABILITY'
    ? feature.get('classification')
    : ee.Array(feature.get('classification')).get([CLASS1_INDEX]);
}

function balancedTrainingScoreSample(training, foldSeed) {
  return training.filter(ee.Filter.eq('presence', 1))
    .randomColumn('score_random', SEED + foldSeed + 31000)
    .sort('score_random').limit(MAX_TRAIN_SCORE_SAMPLES_PER_CLASS)
    .merge(training.filter(ee.Filter.eq('presence', 0))
      .randomColumn('score_random', SEED + foldSeed + 41000)
      .sort('score_random').limit(MAX_TRAIN_SCORE_SAMPLES_PER_CLASS));
}

function tagScores(points, classifier, metadata) {
  return points.classify(classifier).map(function (feature) {
    var coordinates = feature.geometry().coordinates();
    return ee.Feature(null, {
      analysis_type: metadata.analysisType,
      pipeline_version: 'v13b',
      feature_set: metadata.model === 'embeddings'
        ? 'original64_direct500m' : metadata.model,
      model: metadata.model,
      sampling_scheme: metadata.schemeName,
      run_tag: metadata.runTag,
      repeat_id: metadata.repeatId,
      block_size: metadata.blockSize,
      min_leaf: metadata.minimumLeaf,
      variables_per_split: metadata.variablesPerSplit,
      outer_fold: null,
      inner_fold: null,
      fold_id: metadata.fold,
      set: metadata.setName,
      score_population: metadata.setName === 'train'
        ? 'balanced_training_subsample' : 'full_held_out_fold',
      presence: feature.get('presence'),
      prob: presenceScore(feature),
      row_id: feature.get('row_id'),
      emb_fingerprint: feature.get('emb_fingerprint'),
      run_id: RUN_ID,
      code_version: CODE_VERSION,
      presence_score_mode: PRESENCE_SCORE_MODE,
      class1_index: CLASS1_INDEX,
      block_id: feature.get('block_id'),
      match_pair_id: feature.get('match_pair_id'),
      lon: coordinates.get(0),
      lat: coordinates.get(1),
      ndvi: feature.get('ndvi'),
      esa: feature.get('esa'),
      dist_infra_m: feature.get('dist_infra_m'),
      log_effort: feature.get('log_effort'),
      in_ordlo: feature.get('in_ordlo'),
      platform: feature.get('layer')
    });
  });
}

var SCORE_SELECTORS = [
  'analysis_type', 'pipeline_version', 'feature_set', 'model',
  'sampling_scheme', 'run_tag', 'repeat_id',
  'block_size', 'min_leaf', 'variables_per_split', 'outer_fold',
  'inner_fold', 'fold_id', 'set', 'score_population', 'presence', 'prob',
  'row_id', 'emb_fingerprint', 'run_id', 'code_version',
  'presence_score_mode', 'class1_index',
  'block_id', 'match_pair_id', 'lon', 'lat', 'ndvi', 'esa',
  'dist_infra_m', 'log_effort', 'in_ordlo', 'platform'
];

function exportOneFold(points, modelName, bands, foldProperty, fold,
                       analysisType, schemeName, runTag, repeatId, blockSize,
                       modelSeedOffset, minimumLeaf, variablesPerSplit) {
  minimumLeaf = minimumLeaf === null ? FIXED_MIN_LEAF : minimumLeaf;
  variablesPerSplit = variablesPerSplit === null
    ? FIXED_VARIABLES_PER_SPLIT : variablesPerSplit;
  // predictor bands, whatever the caller passed.
  variablesPerSplit = Math.min(variablesPerSplit, bands.length);
  var training = points.filter(ee.Filter.neq(foldProperty, fold));
  var testing = points.filter(ee.Filter.eq(foldProperty, fold));
  var classifier = trainForest(
    training, bands, minimumLeaf, variablesPerSplit,
    SEED + modelSeedOffset + fold * 100);
  var metadata = {
    analysisType: analysisType,
    model: modelName,
    schemeName: schemeName,
    runTag: runTag,
    repeatId: repeatId,
    blockSize: blockSize,
    minimumLeaf: minimumLeaf,
    variablesPerSplit: variablesPerSplit,
    fold: fold,
    setName: 'test'
  };
  var scores = tagScores(testing, classifier, metadata);
  metadata.setName = 'train';
  scores = scores.merge(tagScores(
    balancedTrainingScoreSample(training, fold), classifier, metadata));
  var prefix = analysisType === 'main_fixed' ? 'CV_MainBlocked_' :
    (analysisType === 'random_cv' ? 'CV_RandomFold_' :
      (analysisType === 'repeated_cv' ? 'CV_Repeated_' :
        (analysisType === 'block_sensitivity' ? 'CV_Block_' :
          'CV_Regularisation_')));
  var suffix = '_leaf' + minimumLeaf + '_mtry' + variablesPerSplit + '_';
  if (analysisType === 'repeated_cv') {
    suffix += 'repeat' + repeatId + '_';
  }
  if (analysisType === 'block_sensitivity') {
    suffix += 'block' + String(blockSize).replace('.', 'p') + '_';
  }
  Export.table.toDrive({
    collection: scores,
    description: prefix + modelName + '_' + schemeName + suffix +
      'fold' + fold + FILE_TAG,
    folder: EXPORT_FOLDER,
    fileFormat: 'CSV',
    selectors: SCORE_SELECTORS
  });
}

function runScheme(schemeName) {
  var scheme = SCHEMES[schemeName];
  var models = modelDefinitions(schemeName);
  var isComparisonScheme = COMPARISON_SCHEMES.indexOf(schemeName) >= 0;
  var modelNames = isComparisonScheme
    ? ['embeddings', 'baseline_simple', 'ndvi_null']
    : ['embeddings'];
  var completeCaseBands = isComparisonScheme
    ? MODEL_EMBEDDING_BANDS.concat(BASELINE_BANDS)
    : MODEL_EMBEDDING_BANDS;
  var mainBlockSize = MAIN_BLOCK_SIZE_BY_SCHEME[schemeName] ||
    MAIN_BLOCK_SIZE;
  var blockSensitivityList =
    BLOCK_SIZE_SENSITIVITY_BY_SCHEME[schemeName] || [];
  var requiredProperties = completeCaseBands.concat(['presence']);
  var requiredCount = requiredProperties.length;
  var schemeSampled = ee.FeatureCollection(
      modelDefinitions(schemeName).embeddings.sampleAsset)
    .filterBounds(roi)
    .map(function (feature) {
      return feature.set({
        row_id: feature.id(),
        n_nonnull_predictors: feature.toDictionary(requiredProperties).size()
      });
    })
    .filter(ee.Filter.eq('n_nonnull_predictors', requiredCount))
    .map(function (feature) {
      return feature.set('emb_fingerprint', embFingerprint(feature));
    });
  if (scheme.paired) {
    print(schemeName + ' complete-case rows before pair check:',
      schemeSampled.aggregate_histogram('presence'));
    schemeSampled = retainCompletePairs(schemeSampled);
    print(schemeName + ' rows in complete pairs (v13b11):',
      schemeSampled.aggregate_histogram('presence'));
  }
  modelNames.forEach(function (modelName) {
    var definition = models[modelName];
    var sampled = schemeSampled;
    print(schemeName + '/' + modelName + ' complete-case counts:',
      sampled.aggregate_histogram('presence'));
    var blocked = addBlockedFolds(
      sampled, mainBlockSize, 'spatial_fold', null);
    if (modelName === 'embeddings') {
      var rawSize = ee.FeatureCollection(
        modelDefinitions(schemeName).embeddings.sampleAsset)
        .filterBounds(roi).size();
      var foldAudit = ee.FeatureCollection([ee.Feature(null, {
        scheme: schemeName,
        scheme_label: SCHEME_LABELS[schemeName] || schemeName,
        n_rows_sampled_table: rawSize,
        n_rows_complete: blocked.size(),
        n_rows_removed_incomplete: rawSize.subtract(blocked.size()),
        n_fold_assigned: blocked.filter(
          ee.Filter.inList('spatial_fold', [0, 1, 2, 3, 4])).size(),
        n_fallback_rows: blocked.filter(
          ee.Filter.eq('spatial_fold_fallback', 1)).size(),
        main_block_size_degrees: mainBlockSize,
        n_blocks: blocked.aggregate_count_distinct('block_key'),
        n_test_fold0: blocked.filter(ee.Filter.eq('spatial_fold', 0)).size(),
        n_test_fold1: blocked.filter(ee.Filter.eq('spatial_fold', 1)).size(),
        n_test_fold2: blocked.filter(ee.Filter.eq('spatial_fold', 2)).size(),
        n_test_fold3: blocked.filter(ee.Filter.eq('spatial_fold', 3)).size(),
        n_test_fold4: blocked.filter(ee.Filter.eq('spatial_fold', 4)).size(),
        pipeline_version: 'v13b5'
      })]);
      Export.table.toDrive({
        collection: foldAudit.map(function (f) {
          return f.set({run_id: RUN_ID, code_version: CODE_VERSION});
        }),
        description: 'FoldAssignment_Audit_' + schemeName + FILE_TAG,
        folder: EXPORT_FOLDER,
        fileFormat: 'CSV',
        selectors: ['scheme', 'scheme_label', 'n_rows_sampled_table',
                    'n_rows_complete', 'n_rows_removed_incomplete',
                    'n_fold_assigned', 'n_fallback_rows',
                    'main_block_size_degrees', 'n_blocks',
                    'n_test_fold0', 'n_test_fold1', 'n_test_fold2',
                    'n_test_fold3', 'n_test_fold4', 'pipeline_version',
                    'run_id', 'code_version']
      });
      Export.table.toDrive({
        collection: blocked.map(function (feature) {
          var c = feature.geometry().coordinates();
          return ee.Feature(null, {
            row_id: feature.get('row_id'),
            emb_fingerprint: feature.get('emb_fingerprint'),
            presence: feature.get('presence'),
            match_pair_id: feature.get('match_pair_id'),
            spatial_fold: feature.get('spatial_fold'),
            lon: c.get(0), lat: c.get(1),
            sampling_scheme: schemeName,
            run_id: RUN_ID, code_version: CODE_VERSION,
            presence_score_mode: PRESENCE_SCORE_MODE,
            class1_index: CLASS1_INDEX
          });
        }),
        description: 'CV_RowManifest_' + schemeName + FILE_TAG,
        folder: EXPORT_FOLDER,
        fileFormat: 'CSV',
        selectors: ['run_id', 'code_version', 'sampling_scheme', 'row_id',
                    'emb_fingerprint', 'presence', 'match_pair_id',
                    'spatial_fold', 'lon', 'lat', 'presence_score_mode',
                    'class1_index']
      });
    }
    for (var fold = 0; fold < N_FOLDS; fold++) {
      exportOneFold(
        blocked, modelName, definition.bands, 'spatial_fold', fold,
        'main_fixed', schemeName, scheme.runTag, 'det', mainBlockSize, 0,
        null, definition.mtry);
    }
    if (modelName === 'embeddings' &&
        DIAGNOSTIC_SCHEMES.indexOf(schemeName) >= 0) {
      var randomPoints = addRandomFolds(sampled, scheme.paired, SEED + 90000);
      for (var randomFold = 0; randomFold < N_FOLDS; randomFold++) {
        exportOneFold(
          randomPoints, modelName, definition.bands, 'random_fold', randomFold,
          'random_cv', schemeName, scheme.runTag, 'random', null, 7000,
          null, definition.mtry);
      }
    }
    if (modelName === 'embeddings' &&
        DIAGNOSTIC_SCHEMES.indexOf(schemeName) >= 0) {
      for (var repeat = 0; repeat < N_REPEATED_ASSIGNMENTS; repeat++) {
        var repeatProperty = 'repeat_fold_' + repeat;
        var repeatPoints = addBlockedFolds(
          sampled, mainBlockSize, repeatProperty,
          SEED + 200000 + repeat * 1000 +
            (schemeName === 'recurrent_effort' ? 500000 : 0));
        for (var repeatFold = 0; repeatFold < N_FOLDS; repeatFold++) {
          exportOneFold(
            repeatPoints, modelName, definition.bands, repeatProperty,
            repeatFold, 'repeated_cv', schemeName, scheme.runTag, repeat,
            mainBlockSize, 20000 + repeat * 1000, null, definition.mtry);
        }
      }
      blockSensitivityList.forEach(function (blockSize, blockIndex) {
        var blockProperty = 'block_fold_' + blockIndex;
        var blockPoints = addBlockedFolds(
          sampled, blockSize, blockProperty, null);
        for (var blockFold = 0; blockFold < N_FOLDS; blockFold++) {
          exportOneFold(
            blockPoints, modelName, definition.bands, blockProperty,
            blockFold, 'block_sensitivity', schemeName, scheme.runTag,
            'det', blockSize, 50000 + blockIndex * 1000, null,
            definition.mtry);
        }
      });

      REGULARISATION_SENSITIVITY.forEach(function (configuration, configIndex) {
        for (var regularisationFold = 0;
             regularisationFold < N_FOLDS; regularisationFold++) {
          exportOneFold(
            blocked, modelName, definition.bands, 'spatial_fold',
            regularisationFold, 'regularisation_sensitivity', schemeName,
            scheme.runTag, configuration.role, mainBlockSize,
            80000 + configIndex * 1000, configuration.leaf,
            configuration.mtry);
        }
      });
    }
  });
}

SCHEMES_TO_RUN.forEach(function (schemeName) {
  if (schemeName === 'primary_effort_full' && !RUN_FULL_WINDOW_SENSITIVITY) {
    return;
  }
  runScheme(schemeName);
});

Export.table.toDrive({
  collection: ee.FeatureCollection([ee.Feature(null, {
    pipeline_version: 'v13b_direct_embeddings',
    run_id: RUN_ID,
    code_version: CODE_VERSION,
    run_started_utc: RUN_STARTED_UTC,
    schemes_in_run: SCHEMES_TO_RUN.join(','),
    presence_score_mode: PRESENCE_SCORE_MODE,
    analysis_stage: 'fixed_spatial_validation_and_prespecified_sensitivities',
    // 2 comparison schemes x 15 main + 2 diagnostic schemes x 55
    // (random 5 + repeated 15 + block 10 + regularisation 25) + 5
    // access_matched + 5 recurrent_access_matched (+ 5 full window).
    expected_score_tasks: RUN_FULL_WINDOW_SENSITIVITY ? 155 : 150,
    fold_assignment: 'total_with_deterministic_fallback_and_audit_v13b4',
    complete_case_method:
      'per_feature_toDictionary_property_count_v13b4',
    comparison_schemes: COMPARISON_SCHEMES.join(','),
    diagnostic_schemes: DIAGNOSTIC_SCHEMES.join(','),
    complete_case_rule:
      'primary_effort: embeddings+baseline non-null (identical rows); ' +
      'other schemes: embeddings non-null',
    full_window_sensitivity_scheme:
      RUN_FULL_WINDOW_SENSITIVITY ? 'primary_effort_full' : 'disabled',
    number_of_trees: N_TREES,
    min_leaf: FIXED_MIN_LEAF,
    variables_per_split: FIXED_VARIABLES_PER_SPLIT,
    variables_per_split_baseline_simple: BASELINE_VARIABLES_PER_SPLIT,
    variables_per_split_ndvi_null: NDVI_NULL_VARIABLES_PER_SPLIT,
    blocked_cv_is_headline: 1,
    random_cv_role: 'nonspatial_comparison_only',
    hyperparameter_selection_from_cv: 0,
    regularisation_sensitivity_role: 'diagnostic_only',
    repeated_spatial_assignments: N_REPEATED_ASSIGNMENTS,
    block_sizes_degrees:
      'regular model: 0.30 main + 0.15/0.60 sensitivity; ' +
      'recurrent model: 0.45 main + 0.30/0.60 sensitivity',
    main_block_size_rule:
      'smallest 0.05-deg step whose extent exceeds the model own ' +
      'empirical residual range in both map dimensions',
    training_sample_cap: 'none',
    prediction_score_training_cap_per_class: MAX_TRAIN_SCORE_SAMPLES_PER_CLASS
  })]),
  description: SCHEMES_TO_RUN.length === 5
    ? 'MainCV_RunManifest_outdoor' + FILE_TAG
    : 'MainCV_RunManifest_outdoor' + FILE_TAG + '_' + SCHEMES_TO_RUN.join('_'),
  folder: EXPORT_FOLDER,
  fileFormat: 'CSV'
});

print('v13b12 -- VALIDATION RUN ' + RUN_ID + '. Start ALL score tasks ' +
      'once (files carry _v13b12). Download into a FRESH folder so no older ' +
      'v13b* score files are mixed in. Acceptance checks: (1) every ' +
      'FoldAssignment_Audit CSV shows n_rows_complete == n_fold_assigned, ' +
      'n_fallback_rows == 0, per-fold counts summing to n_rows_complete; ' +
      '(2) Python_01 prints "Scoring-coverage check PASSED" and its ' +
      'stability check reports no deviation beyond 0.02 AUC from the ' +
      'v13b3 reference values.');
print('Main block sizes: regular outdoor visitation model 0.30 deg; ' +
      'recurrent outdoor visitation model 0.45 deg (residual range 40 km; ' +
      'the 0.60 deg sensitivity exceeds it in both directions).');
print('Schemes in this run:', SCHEMES_TO_RUN.join(', '));
print('Created score tasks plus one manifest. Start in batches if needed.');
