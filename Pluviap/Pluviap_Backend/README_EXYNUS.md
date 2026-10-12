# PLUVIAP EXYNUS SAFE — Backend final con monitoreo de Cerro Hoya

El proyecto conserva exactamente el nombre **PLUVIAP EXYNUS SAFE** y el motor **EXYNUS v5.1-SAFE**. No se crea una nueva versión nominal del modelo.

## Resultado científico de Cerro Hoya

Se validó el CSV CHIRPS 2015–2025 de la región montañosa de Cerro Hoya (ventana `-80.90/-80.50, 7.10/7.45`, relieve >=500 m). El archivo contiene 4,018 días completos y valores válidos de precipitación media, máxima y P90.

La comparación A/B se realizó manteniendo los modelos base y la arquitectura EXYNUS, incorporando de forma experimental tres features sin fuga temporal:

- `lluvia_cerro_hoya_1d`
- `lluvia_cerro_hoya_3d`
- `lluvia_cerro_hoya_7d`

Las tres se construyeron desde el P90 diario CHIRPS y usan `shift(1)`, por lo que una predicción del día `t` solo usa lluvia cerrada hasta `t-1`.

### Decisión

**Cerro Hoya no se incorpora como feature predictiva activa en esta versión.**

Se mantiene como **monitoreo hidrometeorológico aguas arriba y variable explicativa**, porque:

- baseline reproducible seed 42: `Recall=1.000`, `F2=0.662651`, `FP=28`;
- variante con Cerro Hoya seed 42: `Recall=1.000`, `F2=0.647059`, `FP=30`;
- 10 corridas con purga de 72 h: `F2` medio A=0.535024 y B=0.531794;
- 10 corridas con purga de 72 h: `FP` medio A=28.6 y B=32.1;
- no hubo superioridad estadística después de Wilcoxon y corrección de Holm;
- las features de Cerro Hoya tuvieron importancia nula o casi nula dentro del meta-modelo;
- activar la regla SAFE predefinida de Cerro Hoya mantuvo `Recall=1.0`, pero elevó FP de 28 a 35 y redujo F2 a 0.611111.

Esto no significa que Cerro Hoya sea irrelevante hidrológicamente. La precipitación de la región montañosa fue descriptivamente mayor en los días positivos. El resultado indica que, con el dataset y arquitectura actuales, esa información es redundante frente a las variables de lluvia ya presentes y todavía no justifica modificar la decisión automática.

Los resultados completos están en `validacion_cerro_hoya/`.

## Qué cambia en el backend

- Se conserva el mismo modelo productivo y los 7 archivos `.pkl` originales, verificados por SHA-256.
- Se incorpora un módulo regional `cerro_hoya.py`.
- Open-Meteo se consulta sobre una malla reproducible de 25 puntos dentro de la región de estudio.
- Se conservan únicamente los puntos cuya elevación reportada es >=500 m.
- La lluvia operativa de Cerro Hoya se resume mediante **P90 espacial**, aproximando el criterio usado en el histórico CHIRPS.
- Se reconstruyen acumulados de 1, 3, 7, 15 y 30 días y pronósticos P90 de 24 h, 72 h y 7 días.
- NASA POWER se conserva como respaldo puntual y se identifica expresamente como no equivalente al P90 regional.
- Cerro Hoya se visualiza y explica, pero **no modifica la probabilidad, el soporte, la confianza ni la decisión final de EXYNUS SAFE**.
- `cerroHoyaDecisionActiva` permanece en `false`.
- Se restauraron exactamente los pesos y el máximo de soporte del baseline productivo (`MAX_SUPPORT_SCORE=13`).

## Campos de Cerro Hoya en `/current-prediction`

- `cerroHoyaDisponible`
- `lluviaCerroHoya1d`
- `lluviaCerroHoya3d`
- `lluviaCerroHoya7d`
- `lluviaCerroHoya15d`
- `lluviaCerroHoya30d`
- `pronosticoCerroHoya24h`
- `pronosticoCerroHoya72h`
- `pronosticoCerroHoya7d`
- `ratioCerroHoyaMariato3d`
- `respaldoCerroHoya`
- `cerroHoyaDecisionActiva`
- `fuenteCerroHoya`
- `cerroHoyaRol`

La aplicación Android puede utilizar estos campos en una vista técnica sin alterar la lógica de alertas.

## Endpoint de diagnóstico

`GET /cerro-hoya-debug`

Devuelve el estado del módulo, método de agregación, región, número de puntos de malla, puntos de montaña válidos, elevación mínima y fuente usada.

## Estructura

```text
PLUVIAP_EXYNUS_SAFE_Backend_Final/
├── main.py
├── exynus_engine.py
├── exynus_v5_1_safe.py
├── cerro_hoya.py
├── rain_map_v2.py
├── modelos_exynus/
├── herramientas/
│   ├── gee_cerro_hoya_chirps_corregido.js
│   └── preparar_dataset_cerro_hoya.py
├── validacion_cerro_hoya/
│   ├── DECISION.md
│   ├── cerro_hoya_chirps_diario_2015_2025.csv
│   ├── metricas_por_semilla.csv
│   ├── resumen_protocolos.csv
│   ├── estadistica_pareada.csv
│   ├── importancia_cerro_hoya.csv
│   ├── predicciones_2025_seed42_A_B.csv
│   ├── prueba_safe_cerro_hoya.csv
│   ├── auditoria_csv.json
│   ├── decision_cerro_hoya.json
│   └── modelo_experimental_cerro_hoya_seed42.pkl
├── pruebas/
│   └── test_regresion_cerro_hoya.py
├── requirements.txt
└── README_EXYNUS.md
```

## Endpoints principales

- `GET /health`
- `GET /exynus-health`
- `GET /current-prediction`
- `GET /current-prediction-exynus`
- `GET /cerro-hoya-debug`
- `POST /predict`
- `POST /predict-simple`
- `GET /rain-map-v2`
- `GET /weather-debug`
- `GET /sources-debug`
- `GET /routes-debug`

## Prueba local

```powershell
cd PLUVIAP_EXYNUS_SAFE_Backend_Final
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python pruebas/test_regresion_cerro_hoya.py
uvicorn main:app --reload
```

Después revisar:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/exynus-health
http://127.0.0.1:8000/cerro-hoya-debug
http://127.0.0.1:8000/current-prediction
```

## Nota metodológica

La región montañosa de Cerro Hoya es un **proxy hidrometeorológico de cabeceras**, no una delimitación formal de las subcuencas de Quebro, Pavo o Varadero. El siguiente nivel científico sería delimitar las subcuencas reales y obtener series prospectivas locales antes de reconsiderar su incorporación directa al meta-modelo.
