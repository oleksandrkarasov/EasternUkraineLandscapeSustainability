/**
 * GEE 00: SUMMER-MEDIAN SENTINEL-2 NDVI, 2019-2021, 10 m
 *
 * Writes Summer_NDVI_Median_2019_2021.tif, the NDVI raster that
 * Python_A1_prepare_presences_and_backgrounds.py samples at the record
 * locations (point attribute 'ndvi'). This attribute is the NDVI predictor of
 * the environmental baseline and of the NDVI-only model.
 *
 * Recipe: Sentinel-2 surface reflectance (COPERNICUS/S2_SR_HARMONIZED),
 * June-August of 2019-2021, scenes with less than 20% cloudy pixels, clouds
 * and cirrus masked with the QA60 band, NDVI = (B8 - B4) / (B8 + B4), median
 * over all remaining observations of a pixel.
 *
 * Not needed to run the pipeline with the shipped synthetic point tables,
 * which already carry the 'ndvi' attribute.
 */

// ---- Configuration ---------------------------------------------------------
// Replace ASSET_ROOT with your own Earth Engine project. The study area
// ('aoi') is uploaded from data/upload_to_GEE/SUSTD_aoi.zip.
var ASSET_ROOT = 'projects/YOUR-PROJECT/assets';
var ASSET_PREFIX = 'SUSTD_';
var AOI_ASSET = ASSET_ROOT + '/' + ASSET_PREFIX + 'aoi';
var EXPORT_FOLDER = 'GEE_Exports_NDVI';
var SCALE_M = 10;            // Sentinel-2 resolution of bands B4 and B8
var START_YEAR = 2019;
var END_YEAR = 2021;
var START_MONTH = 6;         // June
var END_MONTH = 8;           // August
var MAX_CLOUDY_PERCENT = 20; // scene-level pre-filter

var aoi = ee.FeatureCollection(AOI_ASSET);

// Mask clouds and cirrus with the Sentinel-2 QA60 band (bits 10 and 11).
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}

function addNDVI(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}

var ndviCollection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filter(ee.Filter.calendarRange(START_YEAR, END_YEAR, 'year'))
    .filter(ee.Filter.calendarRange(START_MONTH, END_MONTH, 'month'))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUDY_PERCENT))
    .map(maskS2clouds)
    .map(addNDVI)
    .select('NDVI');

var ndviMedian = ndviCollection.median().clip(aoi);

Export.image.toDrive({
  image: ndviMedian,
  description: 'Summer_NDVI_Median_2019_2021',
  folder: EXPORT_FOLDER,
  scale: SCALE_M,
  region: aoi.geometry(),
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

print('Export ready in the Tasks tab.');
