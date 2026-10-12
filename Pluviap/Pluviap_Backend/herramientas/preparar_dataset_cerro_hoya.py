"""Prepara las features experimentales de Cerro Hoya sin fuga temporal.

La serie principal usa el P90 diario CHIRPS sobre la región montañosa >=500 m.
Para cada fila t se usa solo información cerrada hasta t-1 mediante shift(1).
"""
from pathlib import Path
import sys
import pandas as pd


def preparar(csv_cerro: str, csv_maestro: str, salida: str) -> None:
    cerro = pd.read_csv(csv_cerro)
    maestro = pd.read_csv(csv_maestro)
    cerro['fecha'] = pd.to_datetime(cerro['fecha'], errors='raise')
    maestro['fecha'] = pd.to_datetime(maestro['fecha'], errors='raise')
    required = [
        'lluvia_cerro_hoya_media_mm',
        'lluvia_cerro_hoya_max_mm',
        'lluvia_cerro_hoya_p90_mm',
    ]
    for col in required:
        cerro[col] = pd.to_numeric(cerro[col], errors='coerce')
        if cerro[col].isna().any():
            raise ValueError(f'{col} contiene nulos; no se realizará la unión.')
        if (cerro[col] < 0).any():
            raise ValueError(f'{col} contiene lluvia negativa.')
    if cerro['fecha'].duplicated().any():
        raise ValueError('El CSV Cerro Hoya contiene fechas duplicadas.')

    p90 = cerro['lluvia_cerro_hoya_p90_mm']
    cerro['lluvia_cerro_hoya_1d'] = p90.shift(1)
    cerro['lluvia_cerro_hoya_3d'] = p90.shift(1).rolling(3, min_periods=3).sum()
    cerro['lluvia_cerro_hoya_7d'] = p90.shift(1).rolling(7, min_periods=7).sum()

    features = cerro[[
        'fecha',
        'lluvia_cerro_hoya_1d',
        'lluvia_cerro_hoya_3d',
        'lluvia_cerro_hoya_7d',
    ]]
    merged = maestro.merge(features, on='fecha', how='left', validate='one_to_one')
    if len(merged) != len(maestro):
        raise RuntimeError('La unión modificó el número de filas del dataset maestro.')
    missing = merged[['lluvia_cerro_hoya_1d','lluvia_cerro_hoya_3d','lluvia_cerro_hoya_7d']].isna().sum()
    if int(missing.sum()) > 0:
        raise ValueError(f'Features Cerro Hoya faltantes tras la unión: {missing.to_dict()}')
    merged['fecha'] = merged['fecha'].dt.strftime('%Y-%m-%d')
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(salida, index=False)
    print(f'Dataset generado: {salida} ({len(merged)} filas)')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit('Uso: python preparar_dataset_cerro_hoya.py cerro.csv maestro.csv salida.csv')
    preparar(sys.argv[1], sys.argv[2], sys.argv[3])
