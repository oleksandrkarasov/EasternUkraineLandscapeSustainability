/**
 * GEE 01: MATERIALISE THE ORIGINAL ANNUAL EMBEDDING BANDS AT 500 m
 *
 * This deliberately simple producer uses the 64 original Satellite Embedding
 * bands only. The source tiles
 * are mosaicked for each year and sampled directly on the fixed 500 m
 * EPSG:3035 export grid.
 *
 * How the 500 m values arise: Earth Engine serves a 500 m request from the
 * source image pyramid level closest to, but not coarser than, 500 m (320 m
 * for 10 m source data) and then resamples it to the 500 m grid by nearest
 * neighbour. With Earth Engine's default (mean) pyramiding, each 500 m value
 * is therefore a local average of 10 m embeddings rather than a single 10 m
 * pixel; the embeddings are designed to be aggregated in this way.
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
var EXPORT_CRS = 'EPSG:3035';  // ETRS89 / LAEA Europe; metres and equal-area.
var EXPORT_TRANSFORM = [500, 0, 0, 0, -500, 0];
var YEARS = [2019, 2020, 2021, 2022, 2023, 2024];

var aoi = ee.FeatureCollection(AOI_ASSET);
var roi = aoi.geometry();
var embeddingCollection =
  ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');

var EMBEDDING_BANDS = [];
for (var bandIndex = 0; bandIndex < 64; bandIndex++) {
  EMBEDDING_BANDS.push('A' + (bandIndex < 10 ? '0' : '') + bandIndex);
}

function originalAnnualEmbedding(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  return embeddingCollection
    .filterDate(start, start.advance(1, 'year'))
    .filterBounds(roi)
    .select(EMBEDDING_BANDS)
    .mosaic()
    .clip(roi)
    .toFloat()
    .set({
      year: year,
      source_dataset: 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL',
      predictor_representation: 'original_64_embedding_bands',
      output_resampling: 'nearest_neighbour_at_500m_export',
      spatial_summary_features: 'none',
      analysis_crs: EXPORT_CRS,
      analysis_transform: EXPORT_TRANSFORM.join(','),
      pipeline_version: 'v13b'
    });
}

YEARS.forEach(function (year) {
  var assetName = 'EmbeddingAnnual_' + year + '_EPSG3035_500m_v13b';
  Export.image.toAsset({
    image: originalAnnualEmbedding(year),
    description: assetName,
    assetId: OWN + assetName,
    region: roi,
    crs: EXPORT_CRS,
    crsTransform: EXPORT_TRANSFORM,
    maxPixels: 1e13,
    pyramidingPolicy: {'.default': 'sample'}
  });
});

print('Created six direct 64-band annual embedding tasks:', YEARS);
print('Wait for all six assets before running GEE 02.');
