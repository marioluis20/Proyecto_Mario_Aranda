# Decisión técnica — Cerro Hoya en PLUVIAP EXYNUS SAFE

Se evaluó el histórico CHIRPS 2015–2025 de la región montañosa de Cerro Hoya (>=500 m) mediante tres features sin fuga temporal: acumulados P90 de 1, 3 y 7 días, siempre desplazados un día (`shift(1)`).

## Resultado

**Cerro Hoya NO se incorpora como feature predictiva activa del meta-modelo en esta versión.** Se conserva como monitoreo hidrometeorológico aguas arriba y variable explicativa.

Razones principales:

- Corrida reproducible seed 42: baseline `Recall=1.000`, `F2=0.662651`, `FP=28`; variante con Cerro Hoya `Recall=1.000`, `F2=0.647059`, `FP=30`.
- En 10 corridas con purga temporal de 72 h: `F2` medio A=0.535024 vs B=0.531794; `FP` medio A=28.6 vs B=32.1. No hubo mejora significativa.
- Las features Cerro Hoya tuvieron importancia nula o casi nula; `lluvia_cerro_hoya_1d` tuvo importancia 0 en 10/10 corridas.
- La regla SAFE predefinida (`3d>=50 mm` o `7d>=100 mm`) al activarse mantuvo Recall=1.0, pero aumentó FP de 28 a 35 y redujo F2 de 0.662651 a 0.611111.

La señal de Cerro Hoya sí presentó mayor lluvia media en los días positivos que en los negativos; por ello se conserva para visualización técnica, trazabilidad, estudio hidrológico y futuras validaciones prospectivas, pero sin alterar la decisión EXYNUS SAFE actual.
