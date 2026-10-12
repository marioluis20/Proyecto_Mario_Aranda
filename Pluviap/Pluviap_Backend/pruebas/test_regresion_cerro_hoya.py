"""Pruebas offline mínimas para verificar que Cerro Hoya no rompe el baseline."""
from pathlib import Path
import hashlib
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'modelos_exynus'

EXPECTED = {
    'modelo_random_forest_mariato_v3.pkl': 'de38bc58b4870ea1f7bdf6d323f94ad33bc99f3f406b8c8c3e499667ef8b1217',
    'modelo_filtro_falsos_positivos_mariato_v3.pkl': '7a95c424b98abe71c30737f3d27bd2179cae7965ea74293ccd41bf4cb54cdf4d',
    'modelo_lightgbm_mariato_v3.pkl': 'ab47afdbd532a4f8915b49edc577eeb5fdb9ebc930cfd50ae7a3f8cd67c04b61',
    'configuracion_modelo_mariato_v3.pkl': 'effc3f1d535bc801fe8b18413f587593039aa3f5ec6164669498efd922433c3a',
    'modelo_xgboost_centinela_mariato_v3.pkl': '3f14c3ce5dda5c8f653c694f2f2323cc41d67d98e72b02a08e1e4b81ae631fdd',
    'exynus_v5_trs_meta.pkl': '1314fbec50162978b1469bb987653de53db85a155c55f168ef463cd776670c50',
    'exynus_v5_1_safe_package.pkl': '8f8489e902548da1a3046e60d08f8cd1e7d5622bd61bfe728aa2053afbb4795d',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    for name, expected in EXPECTED.items():
        actual = sha256(MODELS / name)
        assert actual == expected, f'Hash distinto en {name}: {actual}'
    safe = joblib.load(MODELS / 'exynus_v5_1_safe_package.pkl')
    assert safe['version'] == 'EXYNUS v5.1-SAFE'
    print('OK: 7/7 modelos baseline preservados y nombre PLUVIAP EXYNUS SAFE intacto.')


if __name__ == '__main__':
    main()
