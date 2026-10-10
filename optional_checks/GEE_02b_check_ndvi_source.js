// GEE_02b -- ONE-OFF CHECK OF THE NDVI COLUMN IN EXISTING TRAINING TABLES.
// Read-only; creates no tasks. Written for the tables produced by the
// and 0% (recurrent) of rows equal to the MODIS value, median difference
// makes that explicit (MODIS band renamed 'ndvi_modis').
/**
 */

var ASSET_ROOT = 'projects/YOUR-PROJECT/assets';   // your Earth Engine project
// Every asset made by this pipeline (GEE_01, GEE_02, GEE_04 and the ZIP
// uploads of Python_A1-A3) is named <ASSET_ROOT>/<ASSET_PREFIX><name>. Use
// the same prefix in all GEE scripts and in support/folders.py; a new prefix
// gives a complete new set of assets, so nothing has to be deleted.
var ASSET_PREFIX = 'SUSTD_';
var OWN = ASSET_ROOT + '/' + ASSET_PREFIX;  // root + prefix of pipeline assets
var AOI_ASSET = OWN + 'aoi';
var EXPORT_CRS = 'EPSG:3035';
var EXPORT_TRANSFORM = [500, 0, 0, 0, -500, 0];
var SCHEMES = ['primary_effort', 'recurrent_effort'];
var TOLERANCE = 1e-4;

var roi = ee.FeatureCollection(AOI_ASSET).geometry();
var refProjection = ee.Projection(EXPORT_CRS, EXPORT_TRANSFORM);

// Identical to summerNdvi([2019, 2020, 2021]) in GEE_02.
var filteredNdvi = ee.ImageCollection('MODIS/061/MOD13Q1')
  .filterBounds(roi)
  .filterDate('2019-01-01', '2022-01-01')
  .filter(ee.Filter.calendarRange(5, 9, 'month'));
var modisProjection = ee.Image(filteredNdvi.first()).select('NDVI').projection();
var modisNdvi = filteredNdvi.map(function (image) {
    return image.select('NDVI').multiply(0.0001)
      .updateMask(image.select('SummaryQA').lte(1));
  }).median()
  .setDefaultProjection(modisProjection)
  .reduceResolution({reducer: ee.Reducer.mean(), bestEffort: false, maxPixels: 16})
  .reproject({crs: EXPORT_CRS, crsTransform: EXPORT_TRANSFORM})
  .rename('ndvi_modis_check').toFloat();

SCHEMES.forEach(function (schemeName) {
  var table = ee.FeatureCollection(
    OWN + 'SDM_Sampled_all_' + schemeName + '_outdoor_v13b')
    .filterBounds(roi)
    .filter(ee.Filter.notNull(['ndvi']))
    .select(['ndvi', 'presence']);
  var checked = modisNdvi.sampleRegions({
    collection: table,
    properties: ['ndvi', 'presence'],
    projection: refProjection,
    geometries: false,
    tileScale: 8
  }).map(function (feature) {
    var diff = ee.Number(feature.get('ndvi'))
      .subtract(ee.Number(feature.get('ndvi_modis_check'))).abs();
    return feature.set({abs_diff: diff, equal: diff.lte(TOLERANCE)});
  });
  var n = checked.size();
  var nEqual = checked.filter(ee.Filter.eq('equal', 1)).size();
  print(schemeName + ': rows compared', n);
  print(schemeName + ': share with table ndvi == MODIS ndvi',
        ee.Number(nEqual).divide(ee.Number(n).max(1)));
  print(schemeName + ': median |difference|',
        checked.reduceColumns(ee.Reducer.median(), ['abs_diff']).get('median'));
  print(schemeName + ': first 10 rows', checked.limit(10));
});
