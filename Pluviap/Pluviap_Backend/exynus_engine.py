"""Motor de inferencia EXYNUS v5.1-SAFE para PLUVIAP."""

# Los modelos se cargan una vez y se conservan a nivel de módulo.
# pylint: disable=global-statement

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
import threading
import warnings

import joblib
import numpy as np
import pandas as pd

from exynus_v5_1_safe import (
    MAX_SUPPORT_SCORE,
    SafePolicyConfig,
    apply_safe_policy,
)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier.*",
    category=UserWarning,
)

BASE_DIR = Path(__file__).resolve().parent
EXYNUS_MODEL_DIR = BASE_DIR / "modelos_exynus"

EXPECTED_MODEL_FILES = (
    "modelo_xgboost_centinela_mariato_v3.pkl",
    "modelo_random_forest_mariato_v3.pkl",
    "modelo_lightgbm_mariato_v3.pkl",
    "modelo_filtro_falsos_positivos_mariato_v3.pkl",
    "configuracion_modelo_mariato_v3.pkl",
    "exynus_v5_trs_meta.pkl",
    "exynus_v5_1_safe_package.pkl",
)

MODELOS_EXYNUS_CARGADOS = False
ERROR_CARGA_EXYNUS: str | None = None

MODEL_XGB = None
MODEL_RF = None
MODEL_LGBM = None
MODEL_FILTER = None
MODEL_EXYNUS = None
CONFIG_V3: dict[str, Any] = {}
TRS_PACKAGE: dict[str, Any] = {}
SAFE_PACKAGE: dict[str, Any] = {}
SAFE_CONFIG = SafePolicyConfig()
BASE_FEATURES: list[str] = []
META_FEATURES_V3: list[str] = []
EXYNUS_FEATURES: list[str] = []

_MODEL_LOAD_LOCK = threading.Lock()
_MODELOS_EXYNUS_CARGANDO = False


def _load_models() -> None:
    global MODELOS_EXYNUS_CARGADOS, ERROR_CARGA_EXYNUS
    global MODEL_XGB, MODEL_RF, MODEL_LGBM, MODEL_FILTER, MODEL_EXYNUS
    global CONFIG_V3, TRS_PACKAGE, SAFE_PACKAGE, SAFE_CONFIG
    global BASE_FEATURES, META_FEATURES_V3, EXYNUS_FEATURES

    try:
        missing = [
            name for name in EXPECTED_MODEL_FILES
            if not (EXYNUS_MODEL_DIR / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Faltan archivos EXYNUS: " + ", ".join(missing)
            )

        MODEL_XGB = joblib.load(
            EXYNUS_MODEL_DIR / "modelo_xgboost_centinela_mariato_v3.pkl"
        )
        MODEL_RF = joblib.load(
            EXYNUS_MODEL_DIR / "modelo_random_forest_mariato_v3.pkl"
        )
        MODEL_LGBM = joblib.load(
            EXYNUS_MODEL_DIR / "modelo_lightgbm_mariato_v3.pkl"
        )
        MODEL_FILTER = joblib.load(
            EXYNUS_MODEL_DIR / "modelo_filtro_falsos_positivos_mariato_v3.pkl"
        )
        CONFIG_V3 = joblib.load(
            EXYNUS_MODEL_DIR / "configuracion_modelo_mariato_v3.pkl"
        )
        TRS_PACKAGE = joblib.load(
            EXYNUS_MODEL_DIR / "exynus_v5_trs_meta.pkl"
        )
        SAFE_PACKAGE = joblib.load(
            EXYNUS_MODEL_DIR / "exynus_v5_1_safe_package.pkl"
        )

        base_package = SAFE_PACKAGE.get("base_model_package", {})
        MODEL_EXYNUS = base_package.get("model") or TRS_PACKAGE.get("model")
        if MODEL_EXYNUS is None:
            raise ValueError("El paquete EXYNUS no contiene el meta-modelo.")

        BASE_FEATURES = list(CONFIG_V3["features_base_v3"])
        META_FEATURES_V3 = list(CONFIG_V3["meta_features_v3"])
        exynus_config = (
            base_package.get("config")
            or TRS_PACKAGE.get("config")
            or {}
        )
        EXYNUS_FEATURES = list(exynus_config["features"])
        SAFE_CONFIG = SafePolicyConfig.from_packages(
            SAFE_PACKAGE, TRS_PACKAGE
        )

        MODELOS_EXYNUS_CARGADOS = True
        ERROR_CARGA_EXYNUS = None
        print(
            "EXYNUS v5.1-SAFE cargado correctamente desde:",
            EXYNUS_MODEL_DIR,
        )
    except Exception as exc:
        MODELOS_EXYNUS_CARGADOS = False
        ERROR_CARGA_EXYNUS = f"{type(exc).__name__}: {exc}"
        print("No se pudo cargar EXYNUS:", ERROR_CARGA_EXYNUS)




def cargar_modelos_exynus() -> None:
    """
    Carga EXYNUS una sola vez y de forma segura.

    Se ejecuta en segundo plano para que Render pueda abrir el puerto
    antes de cargar los archivos .pkl.
    """
    global _MODELOS_EXYNUS_CARGANDO

    if MODELOS_EXYNUS_CARGADOS:
        return

    with _MODEL_LOAD_LOCK:
        if MODELOS_EXYNUS_CARGADOS:
            return

        _MODELOS_EXYNUS_CARGANDO = True
        try:
            _load_models()
        finally:
            _MODELOS_EXYNUS_CARGANDO = False


def modelos_exynus_cargando() -> bool:
    """Indica si la carga de los modelos está en progreso."""
    return bool(_MODELOS_EXYNUS_CARGANDO)


def obtener_safe_config() -> SafePolicyConfig:
    """
    Devuelve la configuración SAFE vigente.

    Se usa mediante una función para que otros módulos reciban la
    configuración cargada y no una referencia previa al arranque.
    """
    return SAFE_CONFIG


def estado_modelos_exynus() -> dict[str, Any]:
    """Información segura de diagnóstico; nunca expone objetos ni secretos."""
    files = {
        name: (EXYNUS_MODEL_DIR / name).is_file()
        for name in EXPECTED_MODEL_FILES
    }
    return {
        "modelo": "EXYNUS v5.1-SAFE",
        "modelos_exynus_cargados": MODELOS_EXYNUS_CARGADOS,
        "modelos_exynus_cargando": modelos_exynus_cargando(),
        "ruta_modelos_exynus": str(EXYNUS_MODEL_DIR),
        "archivos": files,
        "error": ERROR_CARGA_EXYNUS,
        "endpoint_prueba": "/current-prediction",
        "endpoint_alias": "/current-prediction-exynus",
    }


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else float(default)
    except Exception:
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_float(value, default)))
    except Exception:
        return int(default)


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "sí", "si", "yes"}
    return bool(_int(value, int(default)))


def _safe_date(value: Any, default: datetime | None = None) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.Timestamp(default or datetime.now())
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_localize(None)
    return pd.Timestamp(parsed).normalize()


def _manual_daily_history(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Construye una secuencia coherente para los endpoints de prueba manual."""
    rain1 = max(_float(data.get("lluvia_1d")), 0.0)
    rain3 = max(_float(data.get("lluvia_3d"), rain1), rain1)
    rain7 = max(_float(data.get("lluvia_7d"), rain3), rain3)
    rain15 = max(_float(data.get("lluvia_15d"), rain7), rain7)
    rain30 = max(_float(data.get("lluvia_30d"), rain15), rain15)

    cerro_available = _bool(data.get("cerro_hoya_disponible"), False) or any(
        data.get(name) is not None
        for name in (
            "lluvia_cerro_hoya_1d",
            "lluvia_cerro_hoya_3d",
            "lluvia_cerro_hoya_7d",
        )
    )
    cerro1 = max(_float(data.get("lluvia_cerro_hoya_1d")), 0.0)
    cerro3 = max(_float(data.get("lluvia_cerro_hoya_3d"), cerro1), cerro1)
    cerro7 = max(_float(data.get("lluvia_cerro_hoya_7d"), cerro3), cerro3)

    daily = [0.0] * 40
    daily[-1] = rain1
    remainder = max(rain3 - rain1, 0.0)
    daily[-3:-1] = [remainder / 2.0] * 2
    remainder = max(rain7 - rain3, 0.0)
    daily[-7:-3] = [remainder / 4.0] * 4
    remainder = max(rain15 - rain7, 0.0)
    daily[-15:-7] = [remainder / 8.0] * 8
    remainder = max(rain30 - rain15, 0.0)
    daily[-30:-15] = [remainder / 15.0] * 15

    cerro_daily = [0.0] * 40
    cerro_daily[-1] = cerro1
    cerro_remainder = max(cerro3 - cerro1, 0.0)
    cerro_daily[-3:-1] = [cerro_remainder / 2.0] * 2
    cerro_remainder = max(cerro7 - cerro3, 0.0)
    cerro_daily[-7:-3] = [cerro_remainder / 4.0] * 4

    today = _safe_date(data.get("fecha", datetime.now()))
    pressure = _float(data.get("imhpa_presion_media_distrito_hpa"), 1007.0)
    ratio = min(max(_float(data.get("ratio_media_max_estacion"), 1.0), 0.0), 1.0)

    rows: list[dict[str, Any]] = []
    for offset, rain in enumerate(daily):
        date = today - pd.Timedelta(days=len(daily) - 1 - offset)
        rows.append(
            {
                "fecha": date.strftime("%Y-%m-%d"),
                "lluvia_diaria": float(rain),
                "lluvia_estacion_max": float(rain),
                "lluvia_estacion_media": float(rain * ratio),
                "ratio_media_max_estacion": ratio,
                "presion_media": pressure,
                "lluvia_cerro_hoya_diaria": float(cerro_daily[offset]),
                "cerro_hoya_dato_disponible": bool(cerro_available),
            }
        )
    return rows


def _history_frame(data: Mapping[str, Any]) -> pd.DataFrame:
    history = data.get("historico")
    if not isinstance(history, list) or not history:
        history = _manual_daily_history(data)

    frame = pd.DataFrame(history).copy()
    if "fecha" not in frame.columns:
        end = _safe_date(data.get("fecha", datetime.now()))
        frame["fecha"] = pd.date_range(
            end=end, periods=len(frame), freq="D"
        )
    frame["fecha"] = pd.to_datetime(frame["fecha"], errors="coerce")
    frame = frame.dropna(subset=["fecha"]).sort_values("fecha")
    frame = frame.drop_duplicates("fecha", keep="last").reset_index(drop=True)

    if frame.empty:
        raise ValueError("No hay registros históricos válidos para EXYNUS.")

    aliases = {
        "lluvia_diaria": (
            "lluvia_diaria",
            "precipitacion",
            "precipitacion_mm",
            "lluvia_dia",
        ),
        "lluvia_estacion_max": (
            "lluvia_estacion_max",
            "imhpa_lluvia_total_max_estacion_mm",
        ),
        "lluvia_estacion_media": (
            "lluvia_estacion_media",
            "imhpa_lluvia_total_media_estaciones_mm",
        ),
        "presion_media": (
            "presion_media",
            "imhpa_presion_media_distrito_hpa",
            "pressure_msl",
        ),
        "lluvia_cerro_hoya_diaria": (
            "lluvia_cerro_hoya_diaria",
            "lluvia_cerro_hoya_media_mm",
            "lluvia_cerro_hoya_p90_mm",
        ),
    }
    for target, candidates in aliases.items():
        if target not in frame.columns:
            source = next((c for c in candidates if c in frame.columns), None)
            frame[target] = frame[source] if source else np.nan

    for name in (
        "lluvia_diaria",
        "lluvia_estacion_max",
        "lluvia_estacion_media",
        "presion_media",
        "lluvia_cerro_hoya_diaria",
    ):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")

    frame["lluvia_diaria"] = frame["lluvia_diaria"].fillna(0.0).clip(lower=0.0)
    frame["lluvia_estacion_max"] = (
        frame["lluvia_estacion_max"]
        .fillna(frame["lluvia_diaria"])
        .clip(lower=0.0)
    )
    frame["lluvia_estacion_media"] = (
        frame["lluvia_estacion_media"]
        .fillna(frame["lluvia_diaria"])
        .clip(lower=0.0)
    )
    frame["presion_media"] = (
        frame["presion_media"].interpolate(limit_direction="both").fillna(1007.0)
    )
    frame["lluvia_cerro_hoya_diaria"] = (
        frame["lluvia_cerro_hoya_diaria"].fillna(0.0).clip(lower=0.0)
    )
    if "cerro_hoya_dato_disponible" in frame.columns:
        frame["cerro_hoya_dato_disponible"] = (
            frame["cerro_hoya_dato_disponible"]
            .fillna(False)
            .astype(bool)
        )
    else:
        frame["cerro_hoya_dato_disponible"] = False

    if "ratio_media_max_estacion" in frame.columns:
        ratio = pd.to_numeric(
            frame["ratio_media_max_estacion"], errors="coerce"
        )
    else:
        denominator = frame["lluvia_estacion_max"]
        ratio = np.where(
            denominator > 1e-9,
            frame["lluvia_estacion_media"] / denominator,
            1.0,
        )
        ratio = pd.Series(ratio, index=frame.index)
    frame["ratio_media_max_estacion"] = (
        pd.to_numeric(ratio, errors="coerce").fillna(1.0).clip(0.0, 1.0)
    )

    # EXYNUS necesita al menos 30 días de acumulación y 7 de persistencia.
    if len(frame) < 37:
        missing = 37 - len(frame)
        first = frame.iloc[0].copy()
        extra = []
        for day in range(missing, 0, -1):
            row = first.copy()
            row["fecha"] = frame.iloc[0]["fecha"] - pd.Timedelta(days=day)
            row["lluvia_diaria"] = 0.0
            row["lluvia_estacion_max"] = 0.0
            row["lluvia_estacion_media"] = 0.0
            row["ratio_media_max_estacion"] = 1.0
            row["lluvia_cerro_hoya_diaria"] = 0.0
            row["cerro_hoya_dato_disponible"] = False
            extra.append(row)
        frame = pd.concat([pd.DataFrame(extra), frame], ignore_index=True)

    return frame.reset_index(drop=True)


def _severity(value: float, percentiles: Mapping[str, Any]) -> int:
    if value >= _float(percentiles.get("p99"), np.inf):
        return 3
    if value >= _float(percentiles.get("p98"), np.inf):
        return 2
    if value >= _float(percentiles.get("p95"), np.inf):
        return 1
    return 0


def _reconstruct_features(
    data: Mapping[str, Any],
) -> pd.DataFrame:
    frame = _history_frame(data)
    rain = frame["lluvia_diaria"]

    frame["lluvia_1d"] = rain.rolling(1, min_periods=1).sum()
    frame["lluvia_3d"] = rain.rolling(3, min_periods=1).sum()
    frame["lluvia_7d"] = rain.rolling(7, min_periods=1).sum()
    frame["lluvia_15d"] = rain.rolling(15, min_periods=1).sum()
    frame["lluvia_30d"] = rain.rolling(30, min_periods=1).sum()

    cerro_rain = frame["lluvia_cerro_hoya_diaria"]
    frame["lluvia_cerro_hoya_1d"] = cerro_rain.rolling(1, min_periods=1).sum()
    frame["lluvia_cerro_hoya_3d"] = cerro_rain.rolling(3, min_periods=1).sum()
    frame["lluvia_cerro_hoya_7d"] = cerro_rain.rolling(7, min_periods=1).sum()
    frame["lluvia_cerro_hoya_15d"] = cerro_rain.rolling(15, min_periods=1).sum()
    frame["lluvia_cerro_hoya_30d"] = cerro_rain.rolling(30, min_periods=1).sum()
    frame["cerro_hoya_disponible"] = frame[
        "cerro_hoya_dato_disponible"
    ].astype(bool)

    # Solo las ventanas cortas del día actual incorporan el pronóstico; las
    # ventanas de 15 y 30 días permanecen como acumulados reales reconstruidos.
    last = frame.index[-1]
    frame.loc[last, "lluvia_1d"] = max(
        _float(frame.loc[last, "lluvia_1d"]),
        _float(data.get("lluvia_pronostico_24h")),
    )
    frame.loc[last, "lluvia_3d"] = max(
        _float(frame.loc[last, "lluvia_3d"]),
        _float(data.get("lluvia_pronostico_72h")),
    )
    frame.loc[last, "lluvia_7d"] = max(
        _float(frame.loc[last, "lluvia_7d"]),
        _float(data.get("lluvia_pronostico_7d")),
    )

    cerro_available = _bool(
        data.get("cerro_hoya_disponible"),
        bool(frame.loc[last, "cerro_hoya_disponible"]),
    )
    frame.loc[last, "cerro_hoya_disponible"] = cerro_available
    if cerro_available:
        frame.loc[last, "lluvia_cerro_hoya_1d"] = max(
            _float(frame.loc[last, "lluvia_cerro_hoya_1d"]),
            _float(data.get("lluvia_cerro_hoya_pronostico_24h")),
        )
        frame.loc[last, "lluvia_cerro_hoya_3d"] = max(
            _float(frame.loc[last, "lluvia_cerro_hoya_3d"]),
            _float(data.get("lluvia_cerro_hoya_pronostico_72h")),
        )
        frame.loc[last, "lluvia_cerro_hoya_7d"] = max(
            _float(frame.loc[last, "lluvia_cerro_hoya_7d"]),
            _float(data.get("lluvia_cerro_hoya_pronostico_7d")),
        )

    frame["ratio_cerro_hoya_mariato_3d"] = np.where(
        frame["lluvia_3d"] > 1e-6,
        frame["lluvia_cerro_hoya_3d"] / frame["lluvia_3d"],
        np.where(frame["lluvia_cerro_hoya_3d"] > 0.0, 99.0, 1.0),
    )
    frame["ratio_cerro_hoya_mariato_3d"] = pd.Series(
        frame["ratio_cerro_hoya_mariato_3d"], index=frame.index
    ).clip(lower=0.0, upper=99.0)

    # API proxy en milímetros, sin normalización 0–1.
    frame["api_proxy_lluvia"] = (
        0.50 * frame["lluvia_1d"]
        + 0.30 * frame["lluvia_3d"]
        + 0.20 * frame["lluvia_7d"]
    )

    if data.get("api_proxy_lluvia") is not None:
        frame.loc[last, "api_proxy_lluvia"] = _float(
            data.get("api_proxy_lluvia"),
            frame.loc[last, "api_proxy_lluvia"],
        )

    frame["mes"] = frame["fecha"].dt.month.astype(int)
    frame["mes_sin"] = np.sin(2.0 * np.pi * frame["mes"] / 12.0)
    frame["mes_cos"] = np.cos(2.0 * np.pi * frame["mes"] / 12.0)
    frame["temporada_lluviosa"] = frame["mes"].between(5, 11).astype(int)
    frame["pico_lluvioso_sep_nov"] = frame["mes"].between(9, 11).astype(int)

    eps = 1e-6
    frame["ratio_lluvia_1d_3d"] = (
        frame["lluvia_1d"] / (frame["lluvia_3d"] + eps)
    )
    frame["ratio_lluvia_3d_7d"] = (
        frame["lluvia_3d"] / (frame["lluvia_7d"] + eps)
    )
    frame["incremento_1d_vs_3d"] = (
        frame["lluvia_1d"] - frame["lluvia_3d"] / 3.0
    )
    frame["incremento_3d_vs_7d"] = (
        frame["lluvia_3d"] - frame["lluvia_7d"] * 3.0 / 7.0
    )

    anomaly = _float(data.get("anomalia_nino34"), 0.10)
    phase = str(data.get("fase_enso") or "Neutral")
    frame["anomalia_nino34"] = anomaly
    frame["fase_enso"] = phase

    frame["marea_alta_v3"] = 0
    frame["sistema_tropical_activo_v3"] = 0
    if "marea_alta" in frame.columns:
        frame["marea_alta_v3"] = (
            pd.to_numeric(frame["marea_alta"], errors="coerce")
            .fillna(0).astype(int).clip(0, 1)
        )
    if "sistema_tropical_activo" in frame.columns:
        frame["sistema_tropical_activo_v3"] = (
            pd.to_numeric(
                frame["sistema_tropical_activo"], errors="coerce"
            ).fillna(0).astype(int).clip(0, 1)
        )
    frame.loc[last, "marea_alta_v3"] = _int(data.get("marea_alta"), 0)
    frame.loc[last, "sistema_tropical_activo_v3"] = _int(
        data.get("sistema_tropical_activo"), 0
    )

    frame["lluvia1d_x_marea"] = (
        frame["lluvia_1d"] * frame["marea_alta_v3"]
    )
    frame["lluvia3d_x_marea"] = (
        frame["lluvia_3d"] * frame["marea_alta_v3"]
    )
    frame["lluvia7d_x_marea"] = (
        frame["lluvia_7d"] * frame["marea_alta_v3"]
    )

    percentiles = CONFIG_V3["percentiles_fisicos"]
    intensity: list[int] = []
    saturation: list[int] = []
    families: list[int] = []
    scores: list[int] = []
    for _, row in frame.iterrows():
        i = max(
            _severity(_float(row["lluvia_1d"]), percentiles["lluvia_1d"]),
            _severity(_float(row["lluvia_3d"]), percentiles["lluvia_3d"]),
        )
        s = max(
            _severity(_float(row["lluvia_7d"]), percentiles["lluvia_7d"]),
            _severity(_float(row["lluvia_15d"]), percentiles["lluvia_15d"]),
            _severity(_float(row["lluvia_30d"]), percentiles["lluvia_30d"]),
            _severity(
                _float(row["api_proxy_lluvia"]),
                percentiles["api_proxy_lluvia"],
            ),
        )
        tide = _int(row["marea_alta_v3"])
        tropical = _int(row["sistema_tropical_activo_v3"])
        family_count = int(i > 0) + int(s > 0) + tide + tropical
        score = i + s + tide + 2 * tropical
        intensity.append(i)
        saturation.append(s)
        families.append(family_count)
        scores.append(score)

    frame["intensidad"] = intensity
    frame["saturacion"] = saturation
    frame["familias"] = families
    frame["score"] = scores

    # Alias utilizados por el filtro V3.
    frame["score_fisico_agrupado"] = frame["score"]
    frame["familias_evidencia"] = frame["familias"]
    frame["intensidad_grupo"] = frame["intensidad"]
    frame["saturacion_grupo"] = frame["saturacion"]

    # Proxy de acumulación máxima por estación.
    if "imhpa_lluvia_30d_max_estacion_mm" not in frame.columns:
        frame["imhpa_lluvia_30d_max_estacion_mm"] = (
            frame["lluvia_estacion_max"].rolling(30, min_periods=1).sum()
        )
    else:
        frame["imhpa_lluvia_30d_max_estacion_mm"] = pd.to_numeric(
            frame["imhpa_lluvia_30d_max_estacion_mm"], errors="coerce"
        ).fillna(
            frame["lluvia_estacion_max"].rolling(30, min_periods=1).sum()
        )

    frame["imhpa_presion_media_distrito_hpa"] = frame["presion_media"]
    frame["d2_imhpa_lluvia_30d_max_estacion_mm"] = (
        frame["imhpa_lluvia_30d_max_estacion_mm"]
        .shift(2)
        .bfill()
        .fillna(0.0)
    )
    return frame


def _predict_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    base_x = result.reindex(columns=BASE_FEATURES)
    result["p_xgb"] = MODEL_XGB.predict_proba(base_x)[:, 1]
    result["p_rf"] = MODEL_RF.predict_proba(base_x)[:, 1]
    result["p_lgbm"] = MODEL_LGBM.predict_proba(base_x)[:, 1]
    result["p_core"] = (result["p_rf"] + result["p_lgbm"]) / 2.0

    meta_x = result.reindex(columns=META_FEATURES_V3)
    result["p_filtro_old"] = MODEL_FILTER.predict_proba(meta_x)[:, 1]

    # Variables temporales requeridas por el meta-modelo EXYNUS.
    result["roll7_max_p_filtro_old"] = (
        result["p_filtro_old"].rolling(7, min_periods=1).max()
    )
    result["roll3_max_p_filtro_old"] = (
        result["p_filtro_old"].rolling(3, min_periods=1).max()
    )
    result["roll7_mean_p_filtro_old"] = (
        result["p_filtro_old"].rolling(7, min_periods=1).mean()
    )
    result["roll3_mean_p_filtro_old"] = (
        result["p_filtro_old"].rolling(3, min_periods=1).mean()
    )
    result["roll7_mean_lluvia_30d"] = (
        result["lluvia_30d"].rolling(7, min_periods=1).mean()
    )
    result["roll3_mean_lluvia_30d"] = (
        result["lluvia_30d"].rolling(3, min_periods=1).mean()
    )
    result["roll3_max_lluvia_30d"] = (
        result["lluvia_30d"].rolling(3, min_periods=1).max()
    )
    result["roll3_mean_p_core"] = (
        result["p_core"].rolling(3, min_periods=1).mean()
    )

    exynus_x = result.reindex(columns=EXYNUS_FEATURES)
    result["prob_exynus"] = MODEL_EXYNUS.predict_proba(exynus_x)[:, 1]

    # Alias esperados por SAFE.
    result["marea"] = result["marea_alta_v3"].astype(bool)
    result["tropical"] = result["sistema_tropical_activo_v3"].astype(bool)
    return apply_safe_policy(result, SAFE_CONFIG)


def _clip01(value: Any) -> float:
    """Convierte un valor numérico a la escala cerrada de 0 a 1."""
    return float(np.clip(_float(value), 0.0, 1.0))


def _scale_above(
    value: Any,
    support_level: float,
    full_level: float,
) -> float:
    """
    Normaliza una variable física entre su nivel mínimo de respaldo
    y un nivel de evidencia intensa.

    Los valores inferiores al respaldo aportan 0. Los valores iguales
    o superiores al nivel intenso aportan 1.
    """
    numeric_value = max(_float(value), 0.0)
    support = float(support_level)
    full = float(full_level)

    if full <= support:
        return 1.0 if numeric_value >= support else 0.0

    return _clip01((numeric_value - support) / (full - support))


def _operational_risk_index(
    row: pd.Series,
) -> tuple[float, dict[str, Any]]:
    """
    Calcula el índice operativo compuesto de PLUVIAP en escala 0–1.

    La política SAFE selecciona la banda de riesgo. La probabilidad del
    meta-modelo y la evidencia física/temporal únicamente determinan la
    posición dentro de esa banda, por lo que no pueden contradecir la
    decisión final de EXYNUS Safe.

    Bandas:
        normal:      0.01–0.29
        vigilancia: 0.30–0.49
        moderado:    0.50–0.69
        alto:        0.70–0.89
        crítico:     0.90–1.00

    Este valor es un índice compuesto no calibrado; no representa una
    probabilidad estadística exacta de inundación.
    """
    probability = max(_float(row.get("prob_exynus")), 0.0)
    threshold = max(float(SAFE_CONFIG.decision_threshold), 1e-9)
    ratio = probability / threshold

    # Fuerza del modelo dentro y por encima de su umbral técnico.
    model_normal = _clip01(ratio)
    model_above = _clip01((ratio - 1.0) / 2.0)

    # Los niveles inferiores a los mínimos SAFE no elevan la evidencia.
    rain1_high = max(
        float(SAFE_CONFIG.support_rain1_min) * 2.5,
        float(SAFE_CONFIG.support_rain1_min) + 1.0,
    )
    rain1_score = _scale_above(
        row.get("lluvia_1d"),
        SAFE_CONFIG.support_rain1_min,
        rain1_high,
    )
    rain3_score = _scale_above(
        row.get("lluvia_3d"),
        SAFE_CONFIG.support_rain3_min,
        SAFE_CONFIG.critical_rain3_min,
    )
    cerro3_high = max(
        float(SAFE_CONFIG.support_cerro_hoya_3d_min) * 2.0,
        float(SAFE_CONFIG.support_cerro_hoya_3d_min) + 1.0,
    )
    cerro7_high = max(
        float(SAFE_CONFIG.support_cerro_hoya_7d_min) * 2.0,
        float(SAFE_CONFIG.support_cerro_hoya_7d_min) + 1.0,
    )
    cerro_score = max(
        _scale_above(
            row.get("lluvia_cerro_hoya_3d"),
            SAFE_CONFIG.support_cerro_hoya_3d_min,
            cerro3_high,
        ),
        _scale_above(
            row.get("lluvia_cerro_hoya_7d"),
            SAFE_CONFIG.support_cerro_hoya_7d_min,
            cerro7_high,
        ),
    ) if _bool(row.get("cerro_hoya_disponible")) else 0.0
    rain30_score = _scale_above(
        row.get("lluvia_30d"),
        SAFE_CONFIG.support_rain30_min,
        SAFE_CONFIG.critical_rain30_min,
    )
    api_high = max(
        float(SAFE_CONFIG.low_support_api_max),
        float(SAFE_CONFIG.support_api_min) + 1.0,
    )
    api_score = _scale_above(
        row.get("api_proxy_lluvia"),
        SAFE_CONFIG.support_api_min,
        api_high,
    )
    physical_score = _scale_above(
        row.get("score"),
        SAFE_CONFIG.support_score_min,
        SAFE_CONFIG.critical_score_min,
    )

    persistence_score = min(
        1.0,
        (0.40 if _bool(row.get("persistencia_3d")) else 0.0)
        + (0.60 if _bool(row.get("persistencia_7d")) else 0.0),
    )
    consensus_score = 1.0 if _bool(
        row.get("consenso_senales")
    ) else 0.0

    tide = _bool(row.get("marea_alta_v3"))
    tropical = _bool(row.get("sistema_tropical_activo_v3"))
    if tide and tropical:
        aggravants_score = 1.0
    elif tide or tropical:
        aggravants_score = 0.5
    else:
        aggravants_score = 0.0

    # Evidencia integrada de producción. Se conservan exactamente los pesos
    # del baseline validado de PLUVIAP EXYNUS SAFE. Cerro Hoya se calcula y
    # expone como diagnóstico aguas arriba, pero no altera el índice, la
    # confianza ni la decisión mientras no exista nueva evidencia prospectiva.
    evidence = _clip01(
        0.15 * rain1_score
        + 0.20 * rain3_score
        + 0.15 * rain30_score
        + 0.10 * api_score
        + 0.10 * physical_score
        + 0.15 * persistence_score
        + 0.10 * consensus_score
        + 0.05 * aggravants_score
    )

    state = str(row.get("estado_exynus", "normal"))
    confidence = str(row.get("confianza_exynus", "sin_alerta"))
    final_prediction = _bool(row.get("pred_exynus_safe"))
    critical = _bool(row.get("critical_override"))

    if critical:
        votes_score = _clip01(
            _int(row.get("critical_votes"))
            / max(int(SAFE_CONFIG.critical_votes_min), 1)
        )
        position = _clip01(0.50 * votes_score + 0.50 * evidence)
        value = 0.90 + 0.10 * position
        band = "critico"
        model_component = model_above

    elif final_prediction or state == "alerta_confirmada":
        position = _clip01(0.40 * model_above + 0.60 * evidence)
        model_component = model_above

        if confidence in {"alta", "muy_alta"}:
            value = 0.70 + 0.19 * position
            band = "alto"
        else:
            value = 0.50 + 0.19 * position
            band = "moderado"

    elif state in {
        "observacion",
        "observacion_bajo_consenso",
        "observacion_fuera_temporada",
    }:
        position = _clip01(0.50 * model_above + 0.50 * evidence)
        model_component = model_above

        if state in {
            "observacion_bajo_consenso",
            "observacion_fuera_temporada",
        }:
            # Vigilancia limitada por veto, bajo consenso o temporada.
            value = 0.30 + 0.09 * position
        else:
            value = 0.30 + 0.19 * position
        band = "vigilancia"

    else:
        position = _clip01(0.70 * model_normal + 0.30 * evidence)
        value = 0.01 + 0.28 * position
        band = "normal"
        model_component = model_normal

        if state == "recuperacion_residual":
            # La humedad remanente no se presenta como una nueva vigilancia.
            value = min(value, 0.19)

    index = round(float(np.clip(value, 0.01, 1.0)), 4)
    details: dict[str, Any] = {
        "banda": band,
        "fuerzaModelo": round(model_component, 4),
        "relacionUmbral": round(ratio, 6),
        "evidenciaFisica": round(evidence, 4),
        "posicionEnBanda": round(position, 4),
        "lluvia1d": round(rain1_score, 4),
        "lluvia3d": round(rain3_score, 4),
        "lluvia30d": round(rain30_score, 4),
        "saturacionApi": round(api_score, 4),
        "scoreFisico": round(physical_score, 4),
        "persistencia": round(persistence_score, 4),
        "consenso": round(consensus_score, 4),
        "agravantes": round(aggravants_score, 4),
        "cerroHoya": round(cerro_score, 4),
    }
    return index, details


def _trend(frame: pd.DataFrame) -> str:
    if len(frame) < 2:
        return "Estable"
    current = _float(frame.iloc[-1]["p_filtro_old"])
    previous = _float(frame.iloc[-2]["p_filtro_old"])
    if current > previous + 0.03:
        return "En aumento"
    if current < previous - 0.03:
        return "En descenso"
    if _bool(frame.iloc[-1].get("persistencia_7d")):
        return "Sostenida"
    return "Estable"


def _presentation(row: pd.Series) -> dict[str, Any]:
    """
    Convierte el estado técnico de EXYNUS en un contrato público estable.

    alerta:
        normal      -> sin señal confirmada
        vigilancia  -> señal observada, pero no confirma alerta
        moderado    -> alerta confirmada con confianza media
        alto        -> alerta confirmada con confianza alta
        critico     -> excepción crítica multiseñal
    """

    state = str(row.get("estado_exynus", "normal"))
    confidence = str(row.get("confianza_exynus", "sin_alerta"))

    if state == "alerta_critica_confirmada":
        return {
            "alerta": "critico",
            "nivel": 4,
            "title": "ALERTA CRÍTICA",
            "risk": "Crítico",
            "message": (
                "PLUVIAP confirma una condición crítica mediante la "
                "coincidencia de múltiples señales. Mantenga vigilancia "
                "y siga las indicaciones de las autoridades."
            ),
        }

    if state == "alerta_confirmada":
        if confidence in {"alta", "muy_alta"}:
            return {
                "alerta": "alto",
                "nivel": 3,
                "title": "RIESGO ALTO",
                "risk": "Alto",
                "message": (
                    "PLUVIAP confirma riesgo alto con persistencia temporal, "
                    "respaldo físico y consenso entre señales."
                ),
            }

        return {
            "alerta": "moderado",
            "nivel": 2,
            "title": "RIESGO MODERADO",
            "risk": "Moderado",
            "message": (
                "PLUVIAP confirma condiciones que requieren preparación "
                "preventiva y seguimiento cercano."
            ),
        }

    if state == "recuperacion_residual":
        return {
            "alerta": "normal",
            "nivel": 0,
            "title": "RECUPERACIÓN",
            "risk": "Bajo",
            "message": (
                "Persisten condiciones residuales de humedad, pero las "
                "señales recientes descienden y el modelo no confirma una "
                "nueva alerta."
            ),
        }

    if state == "observacion_bajo_consenso":
        return {
            "alerta": "vigilancia",
            "nivel": 1,
            "title": "VIGILANCIA",
            "risk": "Vigilancia",
            "message": (
                "PLUVIAP detectó una señal, pero no existe suficiente "
                "respaldo físico o consenso para confirmar una alerta."
            ),
        }

    if state == "observacion_fuera_temporada":
        return {
            "alerta": "vigilancia",
            "nivel": 1,
            "title": "VIGILANCIA",
            "risk": "Vigilancia",
            "message": (
                "PLUVIAP detectó una señal fuera de temporada, pero no "
                "alcanza los requisitos de la excepción crítica."
            ),
        }

    if state == "observacion":
        return {
            "alerta": "vigilancia",
            "nivel": 1,
            "title": "VIGILANCIA",
            "risk": "Vigilancia",
            "message": (
                "PLUVIAP detecta una señal cercana al umbral, pero "
                "la política SAFE todavía no confirma una alerta operativa."
            ),
        }

    return {
        "alerta": "normal",
        "nivel": 0,
        "title": "CONDICIONES NORMALES",
        "risk": "Bajo",
        "message": (
            "PLUVIAP no identifica condiciones suficientes para una alerta "
            "de inundación en las próximas 24 horas."
        ),
    }


_CONFIDENCE_TEXTS = {
    "muy_alta": "Muy alta",
    "alta": "Alta",
    "media": "Media",
    "baja": "Baja",
    "sin_alerta": "No aplica: sin alerta confirmada",
}


def _confidence_text(value: Any) -> str:
    code = str(value or "sin_alerta")
    return _CONFIDENCE_TEXTS.get(
        code,
        code.replace("_", " ").capitalize(),
    )


def _support_level(score: int) -> tuple[str, str]:
    score = max(0, min(int(score), MAX_SUPPORT_SCORE))
    ratio = score / MAX_SUPPORT_SCORE

    if ratio >= 0.90:
        return "muy_alto", "Muy alto"
    if ratio >= 0.70:
        return "alto", "Alto"
    if ratio >= 0.45:
        return "medio", "Medio"
    if ratio > 0:
        return "bajo", "Bajo"
    return "sin_soporte", "Sin soporte"


def predecir_exynus(datos_actuales: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecuta el motor completo EXYNUS v5.1-SAFE."""
    if not MODELOS_EXYNUS_CARGADOS:
        raise RuntimeError(
            "EXYNUS no está disponible: "
            + (ERROR_CARGA_EXYNUS or "error de carga desconocido")
        )
    if not isinstance(datos_actuales, Mapping):
        raise TypeError("predecir_exynus espera un diccionario.")

    reconstructed = _reconstruct_features(datos_actuales)
    predicted = _predict_probabilities(reconstructed)
    row = predicted.iloc[-1]

    raw_probability = _float(row["prob_exynus"])
    final_prediction = _bool(row["pred_exynus_safe"])
    critical = _bool(row["critical_override"])
    presentation = _presentation(row)
    sources = datos_actuales.get("_fuentes", {})
    if not isinstance(sources, Mapping):
        sources = {}

    tide_high = _int(row["marea_alta_v3"]) == 1
    rain3 = _float(row["lluvia_3d"])
    plug_effect = tide_high and rain3 >= SAFE_CONFIG.support_rain3_min

    warning_parts = []
    source_warning = sources.get("advertenciaDatos")
    if source_warning:
        warning_parts.append(str(source_warning))
    if len(reconstructed) < 40:
        warning_parts.append("La serie histórica disponible es corta.")
    warning = " ".join(warning_parts) or None

    risk_index, risk_details = _operational_risk_index(row)

    threshold = float(SAFE_CONFIG.decision_threshold)
    threshold_epsilon = float(SAFE_CONFIG.threshold_epsilon)
    effective_threshold = max(threshold - threshold_epsilon, 0.0)

    threshold_margin = raw_probability - threshold
    effective_threshold_margin = raw_probability - effective_threshold
    threshold_ratio = (
        raw_probability / threshold
        if threshold > 0
        else 0.0
    )

    threshold_exceeded_nominal = raw_probability >= threshold
    threshold_exceeded_effective = raw_probability >= effective_threshold

    support_score = max(
        0,
        min(_int(row["soporte_alerta"]), MAX_SUPPORT_SCORE),
    )
    support_code, support_text = _support_level(support_score)
    support_percentage = support_score / MAX_SUPPORT_SCORE

    confidence_code = str(row["confianza_exynus"])
    confidence_text = _confidence_text(confidence_code)

    reasons = list(row["razones_exynus"])
    result: Dict[str, Any] = {
        "version": "EXYNUS v5.1-SAFE",
        "modeloActivo": "EXYNUS v5.1-SAFE",
        "horizonteHoras": 72,
        "alertaFinal": presentation["alerta"],
        "nivelAlertaOrdinal": int(presentation["nivel"]),
        "estadoTitulo": presentation["title"],
        "mensaje": presentation["message"],
        "riesgoTexto": presentation["risk"],
        "tendencia": _trend(predicted),
        # Salida técnica real del meta-modelo.
        "probabilidadOperativa": round(raw_probability, 8),
        "probabilidadExynus": round(raw_probability, 8),

        # Índice compuesto de 0 a 1 para el medidor de Android.
        # SAFE fija la banda y los datos posicionan el valor dentro de ella.
        "indiceRiesgoOperativo": risk_index,
        "riesgoOperativoPorcentaje": round(risk_index * 100.0, 2),
        "bandaRiesgoOperativo": risk_details["banda"],
        "tipoIndiceRiesgoOperativo": "indice_compuesto_no_calibrado",
        "aporteModeloRiesgo": risk_details["fuerzaModelo"],
        "evidenciaFisicaRiesgo": risk_details["evidenciaFisica"],
        "componentesRiesgoOperativo": risk_details,

        "umbralExynus": round(threshold, 10),
        "umbralExynusEfectivo": round(effective_threshold, 10),
        "epsilonUmbralExynus": round(threshold_epsilon, 10),
        "umbralExynusSuperado": bool(threshold_exceeded_effective),
        "umbralExynusNominalSuperado": bool(
            threshold_exceeded_nominal
        ),
        "margenUmbralExynus": round(threshold_margin, 10),
        "margenUmbralExynusEfectivo": round(
            effective_threshold_margin,
            10,
        ),
        "relacionUmbralExynus": round(threshold_ratio, 6),

        "estadoExynus": str(row["estado_exynus"]),
        "prediccionExynusSafe": final_prediction,
        "alertaConfirmada": final_prediction,
        "metaModeloSuperaUmbral": _bool(row["pred_exynus_base"]),
        "safeAutorizaAlerta": final_prediction,

        "confianzaExynus": confidence_code,
        "confianzaExynusTexto": confidence_text,

        "soporteAlerta": support_score,
        "soporteAlertaMaximo": MAX_SUPPORT_SCORE,
        "soporteAlertaPorcentaje": round(
            support_percentage,
            4,
        ),
        "soporteAlertaNivel": support_code,
        "soporteAlertaTexto": support_text,
        "persistencia3d": _bool(row["persistencia_3d"]),
        "persistencia7d": _bool(row["persistencia_7d"]),
        "respaldoFisico": _bool(row["respaldo_fisico"]),
        "consensoSenales": _bool(row["consenso_senales"]),
        "lowSupportVeto": _bool(row["low_support_veto"]),
        "residualSuppression": _bool(row["residual_suppression"]),
        "criticalOverride": critical,
        "criticalVotes": _int(row["critical_votes"]),
        "notificacionCriticaPermitida": bool(
            critical
            or (
                final_prediction
                and not _bool(row["low_support_veto"])
                and not _bool(row["residual_suppression"])
            )
        ),
        "lluvia1d": round(_float(row["lluvia_1d"]), 2),
        "lluvia3d": round(rain3, 2),
        "lluvia7d": round(_float(row["lluvia_7d"]), 2),
        "lluvia15d": round(_float(row["lluvia_15d"]), 2),
        "lluvia30d": round(_float(row["lluvia_30d"]), 2),
        "apiProxy": round(_float(row["api_proxy_lluvia"]), 2),

        # Señal hidrológica aguas arriba. No cambia la probabilidad del
        # meta-modelo; actúa como evidencia adicional de la política SAFE.
        "cerroHoyaDisponible": _bool(row.get("cerro_hoya_disponible")),
        "lluviaCerroHoya1d": round(_float(row.get("lluvia_cerro_hoya_1d")), 2),
        "lluviaCerroHoya3d": round(_float(row.get("lluvia_cerro_hoya_3d")), 2),
        "lluviaCerroHoya7d": round(_float(row.get("lluvia_cerro_hoya_7d")), 2),
        "lluviaCerroHoya15d": round(_float(row.get("lluvia_cerro_hoya_15d")), 2),
        "lluviaCerroHoya30d": round(_float(row.get("lluvia_cerro_hoya_30d")), 2),
        "pronosticoCerroHoya24h": round(
            _float(datos_actuales.get("lluvia_cerro_hoya_pronostico_24h")), 2
        ),
        "pronosticoCerroHoya72h": round(
            _float(datos_actuales.get("lluvia_cerro_hoya_pronostico_72h")), 2
        ),
        "pronosticoCerroHoya7d": round(
            _float(datos_actuales.get("lluvia_cerro_hoya_pronostico_7d")), 2
        ),
        "ratioCerroHoyaMariato3d": round(
            _float(row.get("ratio_cerro_hoya_mariato_3d"), 1.0), 4
        ),
        "respaldoCerroHoya": _bool(row.get("respaldo_cerro_hoya")),
        "cerroHoyaDecisionActiva": bool(
            SAFE_CONFIG.cerro_hoya_decision_enabled
        ),
        "fuenteCerroHoya": sources.get(
            "fuenteCerroHoya", datos_actuales.get("fuente_cerro_hoya")
        ),
        "cerroHoyaRol": (
            "monitoreo_aguas_arriba_no_interviene_decision_exynus_safe"
        ),

        "faseEnso": str(row["fase_enso"]),
        "anomaliaNino34": round(_float(row["anomalia_nino34"]), 2),
        "mareaEstado": str(
            datos_actuales.get("marea_estado")
            or ("Alta" if tide_high else "Normal")
        ),
        "efectoTapon": bool(plug_effect),
        "sistemaTropicalActivo": _int(
            row["sistema_tropical_activo_v3"]
        ),
        "razonesExynus": reasons,
        "fuenteLluvia": sources.get(
            "fuenteLluvia", datos_actuales.get("fuente_lluvia")
        ),
        "consensoFuentes": sources.get("consensoFuentes"),
        "certezaDatos": sources.get("certezaDatos", "media"),
        "advertenciaDatos": warning,
        "fechaActualizacion": str(
            datos_actuales.get("fecha_actualizacion")
            or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ),

        # Diagnóstico técnico para la futura actualización Kotlin.
        "probabilidadXgbV3": round(_float(row["p_xgb"]), 8),
        "probabilidadRandomForestV3": round(_float(row["p_rf"]), 8),
        "probabilidadLightgbmV3": round(_float(row["p_lgbm"]), 8),
        "probabilidadCoreV3": round(_float(row["p_core"]), 8),
        "probabilidadFiltroV3": round(_float(row["p_filtro_old"]), 8),
        "scoreFisico": _int(row["score"]),
        "familiasEvidencia": _int(row["familias"]),
        "intensidadGrupo": _int(row["intensidad"]),
        "saturacionGrupo": _int(row["saturacion"]),
        "votosFisicos": _int(row["physical_votes"]),
        "votosConsenso": _int(row["consensus_votes"]),
        "votosDescenso": _int(row["votos_descenso"]),
        "temporadaLluviosa": _bool(row["season_ok"]),
        "ratioMediaMaxEstacion": round(
            _float(row["ratio_media_max_estacion"], 1.0), 4
        ),
        "presionMediaDistritoHpa": round(
            _float(row["imhpa_presion_media_distrito_hpa"], 1007.0), 2
        ),
        "vetoLowHydrology": _bool(row["veto_low_hydrology"]),
        "vetoLowModelConsensus": _bool(
            row["veto_low_model_consensus"]
        ),
        "vetoWeakPhysics": _bool(row["veto_weak_physics"]),
        "vetoSensorDisagreement": _bool(
            row["veto_sensor_disagreement"]
        ),
        "vetoIsolatedTemporalSpike": _bool(
            row["veto_isolated_temporal_spike"]
        ),

        # Alias útiles durante la migración del modelo Kotlin.
        "razonesFinales": reasons,
        "probabilidadFinal": round(raw_probability, 8),
        "indiceRiesgoFinal": risk_index,
    }
    return result


__all__ = [
    "EXYNUS_MODEL_DIR",
    "MODELOS_EXYNUS_CARGADOS",
    "ERROR_CARGA_EXYNUS",
    "SAFE_CONFIG",
    "MAX_SUPPORT_SCORE",
    "cargar_modelos_exynus",
    "modelos_exynus_cargando",
    "obtener_safe_config",
    "estado_modelos_exynus",
    "predecir_exynus",
]