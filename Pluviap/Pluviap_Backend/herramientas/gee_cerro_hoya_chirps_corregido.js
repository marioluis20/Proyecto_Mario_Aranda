// ============================================================
// PLUVIAP EXYNUS SAFE
// HISTÓRICO DE PRECIPITACIÓN - REGIÓN MONTAÑOSA DE CERRO HOYA
// Google Earth Engine / JavaScript
// Periodo: 2015-01-01 a 2025-12-31
// Definición operacional: ventana regional + elevación SRTM >= 500 m
// Fuente de lluvia: CHIRPS Daily
// ============================================================

var FECHA_INICIO = '2015-01-01';
var FECHA_FIN = '2026-01-01';
var ELEVACION_MINIMA = 500;

var regionCerroHoya = ee.Geometry.Rectangle([
  -80.90, 7.10, -80.50, 7.45
]);

var srtm = ee.Image('USGS/SRTMGL1_003').select('elevation');
var mascaraMontana = srtm.gte(ELEVACION_MINIMA).clip(regionCerroHoya);

Map.centerObject(regionCerroHoya, 10);
Map.addLayer(regionCerroHoya, {color: 'red'}, 'Región de estudio Cerro Hoya');
Map.addLayer(mascaraMontana.selfMask(), {palette: ['00FFFF']}, 'Cerro Hoya >=500 m');

var areaMontana = ee.Image.pixelArea()
  .updateMask(mascaraMontana)
  .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: regionCerroHoya,
    scale: 30,
    maxPixels: 1e10
  });
print('Área montañosa >=500 m (km2):', ee.Number(areaMontana.get('area')).divide(1000000));

var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate(FECHA_INICIO, FECHA_FIN)
  .select('precipitation')
  .sort('system:time_start');

var escalaChirps = ee.Image(chirps.first()).projection().nominalScale();
print('Número de días CHIRPS:', chirps.size());
print('Escala nominal CHIRPS (m):', escalaChirps);

var extraerDia = function(imagen) {
  imagen = ee.Image(imagen);
  var fecha = ee.Date(imagen.get('system:time_start'));
  var lluvia = imagen.updateMask(mascaraMontana).clip(regionCerroHoya);

  var media = lluvia.reduceRegion({
    reducer: ee.Reducer.mean(), geometry: regionCerroHoya,
    scale: escalaChirps, maxPixels: 1e9, bestEffort: false
  }).get('precipitation');

  var maximo = lluvia.reduceRegion({
    reducer: ee.Reducer.max(), geometry: regionCerroHoya,
    scale: escalaChirps, maxPixels: 1e9, bestEffort: false
  }).get('precipitation');

  var rP90 = lluvia.reduceRegion({
    reducer: ee.Reducer.percentile([90]), geometry: regionCerroHoya,
    scale: escalaChirps, maxPixels: 1e9, bestEffort: false
  });
  var p90 = ee.Algorithms.If(
    rP90.contains('precipitation'),
    rP90.get('precipitation'),
    rP90.get('precipitation_p90')
  );

  return ee.Feature(null, {
    fecha: fecha.format('YYYY-MM-dd'),
    fecha_ms: fecha.millis(),
    lluvia_cerro_hoya_media_mm: media,
    lluvia_cerro_hoya_max_mm: maximo,
    lluvia_cerro_hoya_p90_mm: p90
  });
};

var tablaDiaria = ee.FeatureCollection(chirps.map(extraerDia)).sort('fecha_ms');
print('Número de registros:', tablaDiaria.size());
print('Registros válidos MEDIA:', tablaDiaria.filter(ee.Filter.notNull(['lluvia_cerro_hoya_media_mm'])).size());
print('Registros válidos MAX:', tablaDiaria.filter(ee.Filter.notNull(['lluvia_cerro_hoya_max_mm'])).size());
print('Registros válidos P90:', tablaDiaria.filter(ee.Filter.notNull(['lluvia_cerro_hoya_p90_mm'])).size());
print('Primeros registros:', tablaDiaria.limit(10));

Export.table.toDrive({
  collection: tablaDiaria,
  description: 'PLUVIAP_Cerro_Hoya_CHIRPS_2015_2025',
  folder: 'PLUVIAP_GEE',
  fileNamePrefix: 'cerro_hoya_chirps_diario_2015_2025',
  fileFormat: 'CSV',
  selectors: [
    'fecha',
    'lluvia_cerro_hoya_media_mm',
    'lluvia_cerro_hoya_max_mm',
    'lluvia_cerro_hoya_p90_mm'
  ]
});
