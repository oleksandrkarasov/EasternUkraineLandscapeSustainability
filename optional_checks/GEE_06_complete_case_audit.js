/**
 * WHY THIS SCRIPT EXISTS
 * ----------------------
 * GEE_03 keeps only rows with a value for every predictor (Methods: "records
 * with a missing value for any predictor were also removed"). Missing values
 * need not be class-symmetric -- an early pipeline version lost ~6.7% of
 * presences but only ~1.5% of pseudo-absences to missing embeddings -- so this
 * audit reports, per scheme and class, how many rows each rule removes and
 * where the rows with incomplete embeddings are. Check the current
 * CompleteCase_Audit CSV before quoting any number; 'ndvi' in these tables is
 * the point-level Sentinel-2 NDVI (see the NDVI note in GEE_02).
 *
 * Outputs (all small table exports; nothing heavy):
 *   CompleteCase_Audit_v13b2.csv
 *     One row per scheme x pipeline stage with presence/absence counts:
 *       uploaded_eligible  points in the uploaded assets after the same
 *                          pre-filters GEE_02 applies (bounds, outdoor ESA
 *                          classes, recurrent thresholds where relevant)
 *       sampled_rows       rows in the persistent sampled table
 *       complete_embeddings / complete_baseline / complete_ndvi /
 *       complete_all       rows passing each non-null rule
 *   CompleteCase_MissingEmbeddings_<scheme>.csv (one per scheme)
 *     Every sampled row whose 64 embedding bands are incomplete, with
 *     presence, esa, ndvi, in_ordlo, year, layer and coordinates, so the
 *     spatial and land-cover structure of the dropout can be described.
 *
 */

var ASSET_ROOT = 'projects/YOUR-PROJECT/assets';   // your Earth Engine project
// Every asset made by this pipeline (GEE_01, GEE_02, GEE_04 and the ZIP
// uploads of Python_A1-A3) is named <ASSET_ROOT>/<ASSET_PREFIX><name>. Use
// the same prefix in all GEE scripts and in support/folders.py; a new prefix
// gives a complete new set of assets, so nothing has to be deleted.
var ASSET_PREFIX = 'SUSTD_';
var OWN = ASSET_ROOT + '/' + ASSET_PREFIX;  // root + prefix of pipeline assets
var AOI_ASSET = OWN + 'aoi';
var EXPORT_FOLDER = 'GEE06_complete_case_audit';  // optional audit, not read by Python
var OUTDOOR_CLASSES = [10, 20, 30, 80, 90, 100];
var INCLUDE_FULL_WINDOW_SENSITIVITY = true;  // match GEE_02/GEE_03

var roi = ee.FeatureCollection(AOI_ASSET).geometry();

var EMBEDDING_BANDS = [];
for (var bandIndex = 0; bandIndex < 64; bandIndex++) {
  EMBEDDING_BANDS.push('A' + (bandIndex < 10 ? '0' : '') + bandIndex);
}
var BASELINE_BANDS = [
  'ndvi', 'lc_tree', 'lc_shrub', 'lc_grass', 'lc_water', 'lc_wetland',
  'lc_moss_lichen', 'dist_water_km', 'elevation', 'slope', 'tpi'
];

var SCHEMES = {
  primary_effort: {
    presenceAsset: OWN + 'points_regular_presence',
    absenceAsset: OWN + 'points_regular_pseudoabsence',
    recurrent: false, paired: false
  },
  access_matched: {
    presenceAsset: OWN + 'points_regular_matched_presence',
    absenceAsset: OWN + 'points_regular_matched_pseudoabsence',
    recurrent: false, paired: true
  },
  recurrent_effort: {
    presenceAsset: OWN + 'points_recurrent_presence',
    absenceAsset: OWN + 'points_recurrent_pseudoabsence',
    recurrent: true, paired: false
  },
  recurrent_access_matched: {
    presenceAsset: OWN + 'points_recurrent_matched_presence',
    absenceAsset: OWN + 'points_recurrent_matched_pseudoabsence',
    recurrent: true, paired: true
  }
};
if (INCLUDE_FULL_WINDOW_SENSITIVITY) {
  SCHEMES.primary_effort_full = {
    presenceAsset: OWN + 'points_full_presence',
    absenceAsset: OWN + 'points_full_pseudoabsence',
    recurrent: false, paired: false
  };
}

/** Replicate GEE_02's eligibility pre-filter on an uploaded point asset. */
function eligiblePoints(assetId, isRecurrentPresence) {
  var points = ee.FeatureCollection(assetId)
    .filterBounds(roi)
    .filter(ee.Filter.inList('esa', OUTDOOR_CLASSES));
  if (isRecurrentPresence) {
    points = points.filter(ee.Filter.and(
      ee.Filter.gte('n_user_days', 3),
      ee.Filter.gte('n_users', 2),
      ee.Filter.gte('n_dates', 2),
      ee.Filter.gte('span_days', 30)
    ));
  }
  return points;
}

function classCounts(fc) {
  return {
    p: fc.filter(ee.Filter.eq('presence', 1)).size(),
    a: fc.filter(ee.Filter.eq('presence', 0)).size()
  };
}

var auditRows = [];

Object.keys(SCHEMES).forEach(function (schemeName) {
  var scheme = SCHEMES[schemeName];
  var table = ee.FeatureCollection(
    OWN + 'SDM_Sampled_all_' + schemeName + '_outdoor_v13b')
    .filterBounds(roi);

  var uploadedPresence = eligiblePoints(scheme.presenceAsset, scheme.recurrent);
  var uploadedAbsence = eligiblePoints(scheme.absenceAsset, false);

  function complete(fc, props) {
    return fc.map(function (feature) {
      return feature.set('n_ok', feature.toDictionary(props).size());
    }).filter(ee.Filter.eq('n_ok', props.length));
  }
  var completeAll = complete(table, EMBEDDING_BANDS.concat(BASELINE_BANDS));
  var stages = [
    ['sampled_rows', table],
    ['complete_embeddings', complete(table, EMBEDDING_BANDS)],
    ['complete_baseline', complete(table, BASELINE_BANDS)],
    ['complete_ndvi', complete(table, ['ndvi'])],
    ['complete_all', completeAll]
  ];
  if (scheme.paired) {
    var pairCondition = ee.Filter.equals({
      leftField: 'match_pair_id', rightField: 'match_pair_id'});
    // matched schemes: GEE_03 requires complete embeddings (embedding model only)
    var completeEmb = complete(table, EMBEDDING_BANDS);
    var pSide = completeEmb.filter(ee.Filter.eq('presence', 1));
    var aSide = completeEmb.filter(ee.Filter.eq('presence', 0));
    var join = ee.Join.simple();
    stages.push(['complete_pairs',
      join.apply(pSide, aSide, pairCondition)
        .merge(join.apply(aSide, pSide, pairCondition))]);
  }

  auditRows.push(ee.Feature(null, {
    scheme: schemeName,
    stage: 'uploaded_eligible',
    paired_scheme: scheme.paired ? 1 : 0,
    n_presence: uploadedPresence.size(),
    n_absence: uploadedAbsence.size()
  }));
  stages.forEach(function (stage) {
    var counts = classCounts(stage[1]);
    auditRows.push(ee.Feature(null, {
      scheme: schemeName,
      stage: stage[0],
      paired_scheme: scheme.paired ? 1 : 0,
      n_presence: counts.p,
      n_absence: counts.a
    }));
  });

  // Rows whose embeddings are incomplete: where and what they are.
  var missing = table.map(function (feature) {
      return feature.set('n_ok', feature.toDictionary(EMBEDDING_BANDS).size());
    }).filter(ee.Filter.lt('n_ok', EMBEDDING_BANDS.length))
    .map(function (feature) {
      var coordinates = feature.geometry().coordinates();
      return ee.Feature(null, {
        scheme: schemeName,
        presence: feature.get('presence'),
        esa: feature.get('esa'),
        ndvi: feature.get('ndvi'),
        in_ordlo: feature.get('in_ordlo'),
        year: feature.get('year'),
        layer: feature.get('layer'),
        dist_infra_m: feature.get('dist_infra_m'),
        log_effort: feature.get('log_effort'),
        lon: coordinates.get(0),
        lat: coordinates.get(1)
      });
    });
  Export.table.toDrive({
    collection: missing,
    description: 'CompleteCase_MissingEmbeddings_' + schemeName + '_v13b2',
    folder: EXPORT_FOLDER,
    fileFormat: 'CSV',
    selectors: ['scheme', 'presence', 'esa', 'ndvi', 'in_ordlo', 'year',
                'layer', 'dist_infra_m', 'log_effort', 'lon', 'lat']
  });
});

Export.table.toDrive({
  collection: ee.FeatureCollection(auditRows),
  description: 'CompleteCase_Audit_v13b2',
  folder: EXPORT_FOLDER,
  fileFormat: 'CSV',
  selectors: ['scheme', 'stage', 'paired_scheme', 'n_presence', 'n_absence']
});

print('Created one summary task plus one missing-embeddings task per scheme.');
print('All are small table exports; start them all at once.');
