from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional
from zoneinfo import ZoneInfo
import importlib
import os
import re
import threading

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

# Carga dinámica del motor local.
# Evita falsos errores de Pylint/Pylance al analizar atributos exportados
# por exynus_engine.py, sin cambiar el comportamiento en ejecución.
_exynus_engine = importlib.import_module("exynus_engine")

cargar_modelos_exynus: Callable[[], None] = getattr(
    _exynus_engine,
    "cargar_modelos_exynus",
)
estado_modelos_exynus: Callable[[], dict[str, Any]] = getattr(
    _exynus_engine,
    "estado_modelos_exynus",
)
obtener_safe_config: Callable[[], Any] = getattr(
    _exynus_engine,
    "obtener_safe_config",
)
predecir_exynus: Callable[
    [Dict[str, Any]],
    Dict[str, Any],
] = getattr(
    _exynus_engine,
    "predecir_exynus",
)
from rain_map_v2 import router as rain_map_v2_router
from cerro_hoya import (
    obtener_cerro_hoya_operativo,
    estado_cerro_hoya,
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LAT_MARIATO = 7.6500
LON_MARIATO = -81.0000
TIMEZONE_MARIATO = "America/Panama"
TZ_MARIATO = ZoneInfo(TIMEZONE_MARIATO)

OPEN_METEO_API_KEY = os.getenv("OPEN_METEO_API_KEY")
OPEN_METEO_FORECAST_URL = (
    "https://customer-api.open-meteo.com/v1/forecast"
    if OPEN_METEO_API_KEY
    else "https://api.open-meteo.com/v1/forecast"
)

MET_NORWAY_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
MET_NORWAY_USER_AGENT = (
    "PLUVIAP/2.0 contacto:marioluisluismario@gmail.com"
)

EXYNUS_STATIONS = [
    {"nombre": "Mariato", "lat": 7.6500, "lon": -81.0000},
    {"nombre": "Loma de Quebro", "lat": 7.5600, "lon": -80.9700},
    {"nombre": "Río Quebro", "lat": 7.5400, "lon": -80.9800},
    {"nombre": "Río Pavo", "lat": 7.4700, "lon": -80.9900},
    {"nombre": "Arenas", "lat": 7.6000, "lon": -80.9000},
    {"nombre": "El Cacao", "lat": 7.4300, "lon": -80.8800},
    {"nombre": "Palo Seco / costa", "lat": 7.6200, "lon": -81.0700},
    {"nombre": "Tebario", "lat": 7.7200, "lon": -80.9500},
]

CACHE_FUENTES: dict[str, dict[str, Any]] = {}


# ============================================================
# FASTAPI
# ============================================================

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Abre el servicio primero y carga EXYNUS en segundo plano.

    La función de carga usa un bloqueo interno y evita cargas duplicadas.
    """
    loader_thread = threading.Thread(
        target=cargar_modelos_exynus,
        name="carga-modelos-exynus",
        daemon=True,
    )
    loader_thread.start()
    yield


app = FastAPI(
    title="PLUVIAP API Predictiva",
    description=(
        "Backend de alerta temprana para Mariato con EXYNUS v5.1-SAFE."
    ),
    version="5.1.0",
    lifespan=lifespan,
)

# No se modifica rain_map_v2.py ni el contrato de /rain-map-v2.
app.include_router(rain_map_v2_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELOS DE ENTRADA MANUAL
# ============================================================

class PredictionRequest(BaseModel):
    features: Dict[str, Any]


class SimplePredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lluvia_1d": 27.39,
                "lluvia_3d": 69.0,
                "lluvia_7d": 142.4,
                "lluvia_15d": 146.35,
                "lluvia_30d": 285.42,
                "api_proxy_lluvia": 62.88,
                "fase_enso": "Neutral",
                "anomalia_nino34": 0.10,
                "marea_alta": 0,
                "sistema_tropical_activo": 0,
                "ratio_media_max_estacion": 0.76,
                "imhpa_presion_media_distrito_hpa": 1010.41,
                "cerro_hoya_disponible": True,
                "lluvia_cerro_hoya_1d": 18.0,
                "lluvia_cerro_hoya_3d": 62.0,
                "lluvia_cerro_hoya_7d": 128.0,
                "mes": 7,
            }
        }
    )

    lluvia_1d: Optional[float] = Field(default=12.4, ge=0.0)
    lluvia_3d: Optional[float] = Field(default=34.8, ge=0.0)
    lluvia_7d: Optional[float] = Field(default=68.2, ge=0.0)
    lluvia_15d: Optional[float] = Field(default=124.6, ge=0.0)
    lluvia_30d: Optional[float] = Field(default=210.5, ge=0.0)
    api_proxy_lluvia: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Índice de precipitación antecedente en milímetros. "
            "Si se omite, EXYNUS lo calcula."
        ),
    )
    fase_enso: Optional[str] = "Neutral"
    anomalia_nino34: Optional[float] = 0.10
    marea_alta: Optional[int] = Field(default=0, ge=0, le=1)
    sistema_tropical_activo: Optional[int] = Field(default=0, ge=0, le=1)
    ratio_media_max_estacion: Optional[float] = Field(
        default=1.0, ge=0.0, le=1.0
    )
    imhpa_presion_media_distrito_hpa: Optional[float] = Field(
        default=1007.0, ge=850.0, le=1100.0
    )
    cerro_hoya_disponible: bool = False
    lluvia_cerro_hoya_1d: Optional[float] = Field(default=None, ge=0.0)
    lluvia_cerro_hoya_3d: Optional[float] = Field(default=None, ge=0.0)
    lluvia_cerro_hoya_7d: Optional[float] = Field(default=None, ge=0.0)
    mes: Optional[int] = Field(default=None, ge=1, le=12)


class ExynusPredictionResponse(BaseModel):
    """Contrato público estable de la predicción EXYNUS."""

    model_config = ConfigDict(extra="allow")

    version: str
    modeloActivo: str
    horizonteHoras: int = Field(ge=1)
    alertaFinal: Literal[
        "normal", "vigilancia", "moderado", "alto", "critico"
    ]
    nivelAlertaOrdinal: int = Field(ge=0, le=4)
    estadoTitulo: str
    mensaje: str
    riesgoTexto: str
    tendencia: str

    probabilidadOperativa: float = Field(ge=0.0, le=1.0)
    probabilidadExynus: float = Field(ge=0.0, le=1.0)
    indiceRiesgoOperativo: float = Field(ge=0.01, le=1.0)
    riesgoOperativoPorcentaje: float = Field(ge=1.0, le=100.0)
    bandaRiesgoOperativo: Literal[
        "normal", "vigilancia", "moderado", "alto", "critico"
    ]
    tipoIndiceRiesgoOperativo: str
    aporteModeloRiesgo: float = Field(ge=0.0, le=1.0)
    evidenciaFisicaRiesgo: float = Field(ge=0.0, le=1.0)
    componentesRiesgoOperativo: Dict[str, Any]

    umbralExynus: float = Field(ge=0.0, le=1.0)
    umbralExynusEfectivo: float = Field(ge=0.0, le=1.0)
    epsilonUmbralExynus: float = Field(ge=0.0)
    umbralExynusSuperado: bool
    umbralExynusNominalSuperado: bool
    margenUmbralExynus: float
    margenUmbralExynusEfectivo: float
    relacionUmbralExynus: float = Field(ge=0.0)

    estadoExynus: Literal[
        "normal",
        "observacion",
        "observacion_bajo_consenso",
        "observacion_fuera_temporada",
        "recuperacion_residual",
        "alerta_confirmada",
        "alerta_critica_confirmada",
    ]
    prediccionExynusSafe: bool
    alertaConfirmada: bool
    metaModeloSuperaUmbral: bool
    safeAutorizaAlerta: bool

    confianzaExynus: Literal[
        "sin_alerta", "baja", "media", "alta", "muy_alta"
    ]
    confianzaExynusTexto: str

    soporteAlerta: int = Field(ge=0, le=13)
    soporteAlertaMaximo: int = Field(default=13, ge=1)
    soporteAlertaPorcentaje: float = Field(ge=0.0, le=1.0)
    soporteAlertaNivel: str
    soporteAlertaTexto: str

    persistencia3d: bool
    persistencia7d: bool
    respaldoFisico: bool
    consensoSenales: bool
    lowSupportVeto: bool
    residualSuppression: bool
    criticalOverride: bool
    criticalVotes: int = Field(ge=0)
    notificacionCriticaPermitida: bool

    lluvia1d: float = Field(ge=0.0)
    lluvia3d: float = Field(ge=0.0)
    lluvia7d: float = Field(ge=0.0)
    lluvia15d: float = Field(ge=0.0)
    lluvia30d: float = Field(ge=0.0)
    apiProxy: float = Field(ge=0.0)

    cerroHoyaDisponible: bool = False
    lluviaCerroHoya1d: float = Field(default=0.0, ge=0.0)
    lluviaCerroHoya3d: float = Field(default=0.0, ge=0.0)
    lluviaCerroHoya7d: float = Field(default=0.0, ge=0.0)
    lluviaCerroHoya15d: float = Field(default=0.0, ge=0.0)
    lluviaCerroHoya30d: float = Field(default=0.0, ge=0.0)
    pronosticoCerroHoya24h: float = Field(default=0.0, ge=0.0)
    pronosticoCerroHoya72h: float = Field(default=0.0, ge=0.0)
    pronosticoCerroHoya7d: float = Field(default=0.0, ge=0.0)
    ratioCerroHoyaMariato3d: float = Field(default=1.0, ge=0.0)
    respaldoCerroHoya: bool = False
    cerroHoyaDecisionActiva: bool = False
    fuenteCerroHoya: Optional[str] = None
    cerroHoyaRol: str = (
        "monitoreo_aguas_arriba_no_interviene_decision_exynus_safe"
    )

    faseEnso: str
    anomaliaNino34: float
    mareaEstado: str
    efectoTapon: bool
    sistemaTropicalActivo: int = Field(ge=0, le=1)
    razonesExynus: list[str]

    fuenteLluvia: Optional[str] = None
    consensoFuentes: Optional[str] = None
    certezaDatos: str
    advertenciaDatos: Optional[str] = None
    fechaActualizacion: str

    fuentes: Optional[Dict[str, Any]] = None
    nivelMarProxy: Optional[float] = None
    nombreSistemaTropical: Optional[str] = None
    comparacionFuentes: Optional[Dict[str, Any]] = None


# ============================================================
# UTILIDADES DE CACHÉ Y FECHAS
# ============================================================

def ahora_mariato() -> datetime:
    return datetime.now(TZ_MARIATO)


def obtener_con_cache(nombre: str, ttl_minutos: int, funcion):
    """Consulta una fuente, reutiliza caché vigente y conserva caché ante error."""
    now = ahora_mariato()
    item = CACHE_FUENTES.get(nombre)

    if item:
        age = now - item["fecha"]
        if age < timedelta(minutes=ttl_minutos):
            result = dict(item["resultado"])
            result["_cache"] = True
            result["_cache_estado"] = "cache_vigente"
            result["_cache_edad_minutos"] = round(
                age.total_seconds() / 60.0, 2
            )
            return result

    try:
        result = dict(funcion())
        CACHE_FUENTES[nombre] = {"fecha": now, "resultado": result}
        result["_cache"] = False
        result["_cache_estado"] = "consulta_nueva"
        result["_cache_edad_minutos"] = 0.0
        return result
    except Exception as exc:
        if item:
            age = now - item["fecha"]
            result = dict(item["resultado"])
            result["_cache"] = True
            result["_cache_estado"] = "cache_por_error_fuente"
            result["_cache_edad_minutos"] = round(
                age.total_seconds() / 60.0, 2
            )
            result["_cache_error"] = f"{type(exc).__name__}: {exc}"
            return result
        raise


def _parse_open_meteo_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TZ_MARIATO)
    return parsed.astimezone(TZ_MARIATO)


def _parse_met_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else float(default)
    except Exception:
        return float(default)


def resumir_historico_fuente(
    datos_fuente: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Calcula acumulados observados de 1, 3 y 7 días a partir del
    campo ``historico`` devuelto por Open-Meteo o NASA POWER.

    No usa los campos de pronóstico. NASA POWER se muestra como
    fuente histórica/observada y puede tener uno o más días de retraso.
    """
    vacio = {
        "lluvia1d": None,
        "lluvia3d": None,
        "lluvia7d": None,
        "ultimoDia": None,
        "diasValidos": 0,
    }

    if not isinstance(datos_fuente, Mapping):
        return dict(vacio)

    historico = datos_fuente.get("historico")
    if not isinstance(historico, list) or not historico:
        return dict(vacio)

    registros: list[tuple[datetime, float]] = []

    for item in historico:
        if not isinstance(item, Mapping):
            continue

        fecha_raw = item.get("fecha")
        lluvia_raw = item.get("lluvia_diaria")

        try:
            fecha = datetime.fromisoformat(str(fecha_raw))
            lluvia = float(lluvia_raw)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(lluvia) or lluvia < 0:
            continue

        registros.append((fecha, lluvia))

    if not registros:
        return dict(vacio)

    registros.sort(key=lambda par: par[0])
    valores = [lluvia for _, lluvia in registros]

    return {
        "lluvia1d": round(float(sum(valores[-1:])), 2),
        "lluvia3d": round(float(sum(valores[-3:])), 2),
        "lluvia7d": round(float(sum(valores[-7:])), 2),
        "ultimoDia": registros[-1][0].date().isoformat(),
        "diasValidos": len(registros),
    }


def _diferencia_fuentes(
    nasa_value: Any,
    open_meteo_value: Any,
) -> float | None:
    if nasa_value is None or open_meteo_value is None:
        return None

    try:
        return round(
            float(nasa_value) - float(open_meteo_value),
            2,
        )
    except (TypeError, ValueError):
        return None


def construir_comparacion_fuentes(
    fuentes: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Construye el contrato que consume la tarjeta de consenso Android."""
    fuentes = fuentes if isinstance(fuentes, Mapping) else {}
    met = fuentes.get("metNorway")
    met = met if isinstance(met, Mapping) else {}

    open_meteo = {
        "disponible": bool(
            fuentes.get("openMeteoDisponible", False)
        ),
        "lluvia1d": fuentes.get("openMeteoLluvia1d"),
        "lluvia3d": fuentes.get("openMeteoLluvia3d"),
        "lluvia7d": fuentes.get("openMeteoLluvia7d"),
        "ultimoDia": fuentes.get("openMeteoUltimoDia"),
    }

    nasa_power = {
        "disponible": bool(
            fuentes.get("nasaPowerDisponible", False)
        ),
        "lluvia1d": fuentes.get("nasaLluvia1d"),
        "lluvia3d": fuentes.get("nasaLluvia3d"),
        "lluvia7d": fuentes.get("nasaLluvia7d"),
        "ultimoDia": fuentes.get("nasaUltimoDia"),
    }

    met_norway = {
        "disponible": bool(met),
        "lluvia3h": met.get(
            "lluvia3h",
            met.get("lluvia_3h"),
        ),
        "lluvia6h": met.get(
            "lluvia6h",
            met.get("lluvia_6h"),
        ),
        "lluvia12h": met.get(
            "lluvia12h",
            met.get("lluvia_12h"),
        ),
    }

    return {
        "openMeteo": open_meteo,
        "nasaPower": nasa_power,
        "metNorway": met_norway,
        "diferencias": {
            "nasaMenosOpenMeteo1d": _diferencia_fuentes(
                nasa_power["lluvia1d"],
                open_meteo["lluvia1d"],
            ),
            "nasaMenosOpenMeteo3d": _diferencia_fuentes(
                nasa_power["lluvia3d"],
                open_meteo["lluvia3d"],
            ),
            "nasaMenosOpenMeteo7d": _diferencia_fuentes(
                nasa_power["lluvia7d"],
                open_meteo["lluvia7d"],
            ),
        },
        "periodosComparables": (
            open_meteo["ultimoDia"] is not None
            and open_meteo["ultimoDia"] == nasa_power["ultimoDia"]
        ),
    }


# ============================================================
# LLUVIA E HISTÓRICO MULTIESTACIÓN PARA EXYNUS
# ============================================================

def _station_hourly_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly", {}) if isinstance(payload, Mapping) else {}
    times = hourly.get("time", [])
    rain = hourly.get("precipitation", [])
    pressure = hourly.get("pressure_msl", [])

    if not times or not rain:
        raise ValueError("Open-Meteo no devolvió precipitación horaria.")

    if not pressure:
        pressure = [np.nan] * len(times)

    length = min(len(times), len(rain), len(pressure))
    frame = pd.DataFrame(
        {
            "fecha_hora": [
                _parse_open_meteo_time(value) for value in times[:length]
            ],
            "lluvia": [
                max(_safe_float(value), 0.0) for value in rain[:length]
            ],
            "presion": [
                _safe_float(value, np.nan) for value in pressure[:length]
            ],
        }
    )
    frame["fecha"] = frame["fecha_hora"].dt.date
    return frame


def _window_station_value(
    station_frames: list[pd.DataFrame],
    start: datetime,
    end: datetime,
) -> float:
    totals: list[float] = []
    for frame in station_frames:
        mask = (
            (frame["fecha_hora"] >= start)
            & (frame["fecha_hora"] < end)
        )
        totals.append(float(frame.loc[mask, "lluvia"].sum()))

    if not totals:
        return 0.0
    center = totals[0]
    district_mean = float(np.mean(totals))
    return max(center, district_mean)


def obtener_historico_open_meteo_exynus() -> dict[str, Any]:
    """Reconstruye 45 días y 7 días de pronóstico para las variables EXYNUS."""
    latitudes = ",".join(str(item["lat"]) for item in EXYNUS_STATIONS)
    longitudes = ",".join(str(item["lon"]) for item in EXYNUS_STATIONS)

    params: dict[str, Any] = {
        "latitude": latitudes,
        "longitude": longitudes,
        "hourly": "precipitation,pressure_msl",
        "past_days": 45,
        "forecast_days": 7,
        "timezone": TIMEZONE_MARIATO,
    }
    if OPEN_METEO_API_KEY:
        params["apikey"] = OPEN_METEO_API_KEY

    response = requests.get(
        OPEN_METEO_FORECAST_URL,
        params=params,
        timeout=35,
    )
    response.raise_for_status()
    payload = response.json()
    payloads = payload if isinstance(payload, list) else [payload]

    if len(payloads) != len(EXYNUS_STATIONS):
        raise ValueError(
            "Open-Meteo no devolvió todas las estaciones solicitadas."
        )

    station_frames = [_station_hourly_frame(item) for item in payloads]
    today = ahora_mariato().date()

    all_dates = sorted(
        {
            date
            for frame in station_frames
            for date in frame.loc[frame["fecha"] <= today, "fecha"].tolist()
        }
    )
    if len(all_dates) < 30:
        raise ValueError(
            "Open-Meteo no devolvió al menos 30 días para EXYNUS."
        )

    station_daily_rain: list[pd.Series] = []
    station_daily_pressure: list[pd.Series] = []
    for frame in station_frames:
        grouped = frame.groupby("fecha", as_index=True)
        station_daily_rain.append(grouped["lluvia"].sum())
        station_daily_pressure.append(grouped["presion"].mean())

    rain_matrix = pd.DataFrame(
        {
            station["nombre"]: series.reindex(all_dates).fillna(0.0)
            for station, series in zip(EXYNUS_STATIONS, station_daily_rain)
        },
        index=all_dates,
    )
    pressure_matrix = pd.DataFrame(
        {
            station["nombre"]: series.reindex(all_dates)
            for station, series in zip(
                EXYNUS_STATIONS, station_daily_pressure
            )
        },
        index=all_dates,
    )
    pressure_matrix = pressure_matrix.interpolate(
        limit_direction="both"
    ).fillna(1007.0)

    station_rolling_30 = rain_matrix.rolling(30, min_periods=1).sum()
    rows: list[dict[str, Any]] = []

    for date in all_dates:
        values = rain_matrix.loc[date]
        maximum = float(values.max())
        mean = float(values.mean())
        center = float(values.iloc[0])
        ratio = mean / maximum if maximum > 1e-9 else 1.0
        rows.append(
            {
                "fecha": date.isoformat(),
                "lluvia_diaria": round(max(center, mean), 4),
                "lluvia_estacion_max": round(maximum, 4),
                "lluvia_estacion_media": round(mean, 4),
                "ratio_media_max_estacion": round(
                    float(np.clip(ratio, 0.0, 1.0)), 6
                ),
                "presion_media": round(
                    float(pressure_matrix.loc[date].mean()), 4
                ),
                "imhpa_lluvia_30d_max_estacion_mm": round(
                    float(station_rolling_30.loc[date].max()), 4
                ),
            }
        )

    now = ahora_mariato()
    forecast_24h = _window_station_value(
        station_frames, now, now + timedelta(hours=24)
    )
    forecast_72h = _window_station_value(
        station_frames, now, now + timedelta(hours=72)
    )
    forecast_7d = _window_station_value(
        station_frames, now, now + timedelta(days=7)
    )

    return {
        "historico": rows,
        "lluvia_pronostico_24h": round(forecast_24h, 2),
        "lluvia_pronostico_72h": round(forecast_72h, 2),
        "lluvia_pronostico_7d": round(forecast_7d, 2),
        "fuente": "Open-Meteo multiestación",
        "estaciones_disponibles": len(station_frames),
        "dias_historicos": len(rows),
    }


def obtener_historico_nasa_power_exynus() -> dict[str, Any]:
    """Respaldo diario; no inventa pronóstico ni presión observada."""
    today = ahora_mariato().date()
    start = today - timedelta(days=50)
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "PRECTOTCORR,PS",
        "community": "AG",
        "longitude": LON_MARIATO,
        "latitude": LAT_MARIATO,
        "start": start.strftime("%Y%m%d"),
        "end": today.strftime("%Y%m%d"),
        "format": "JSON",
    }
    response = requests.get(url, params=params, timeout=35)
    response.raise_for_status()
    payload = response.json()
    parameters = (
        payload.get("properties", {})
        .get("parameter", {})
    )
    rain_values = parameters.get("PRECTOTCORR", {})
    pressure_values = parameters.get("PS", {})

    rows: list[dict[str, Any]] = []
    rolling: list[float] = []
    for key in sorted(rain_values):
        rain = _safe_float(rain_values.get(key), -999.0)
        if rain < 0:
            continue
        # NASA PS suele expresarse en kPa; se convierte a hPa.
        pressure = _safe_float(pressure_values.get(key), 100.7)
        if pressure < 200:
            pressure *= 10.0

        rolling.append(rain)
        rows.append(
            {
                "fecha": datetime.strptime(key, "%Y%m%d").date().isoformat(),
                "lluvia_diaria": round(rain, 4),
                "lluvia_estacion_max": round(rain, 4),
                "lluvia_estacion_media": round(rain, 4),
                "ratio_media_max_estacion": 1.0,
                "presion_media": round(pressure, 4),
                "imhpa_lluvia_30d_max_estacion_mm": round(
                    sum(rolling[-30:]), 4
                ),
            }
        )

    if len(rows) < 30:
        raise ValueError("NASA POWER no devolvió 30 días válidos.")

    return {
        "historico": rows,
        "lluvia_pronostico_24h": 0.0,
        "lluvia_pronostico_72h": 0.0,
        "lluvia_pronostico_7d": 0.0,
        "fuente": "NASA POWER",
        "estaciones_disponibles": 1,
        "dias_historicos": len(rows),
    }


def obtener_lluvia_met_norway_mariato() -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    response = requests.get(
        MET_NORWAY_URL,
        params={"lat": LAT_MARIATO, "lon": LON_MARIATO},
        headers={
            "User-Agent": MET_NORWAY_USER_AGENT,
            "Accept": "application/json",
        },
        timeout=25,
    )
    response.raise_for_status()
    timeseries = (
        response.json()
        .get("properties", {})
        .get("timeseries", [])
    )
    if not timeseries:
        raise ValueError("MET Norway no devolvió timeseries.")

    totals = {3: 0.0, 6: 0.0, 12: 0.0}
    for item in timeseries:
        try:
            timestamp = _parse_met_time(item["time"])
            next_hour = item.get("data", {}).get("next_1_hours", {})
            rain = _safe_float(
                next_hour.get("details", {}).get(
                    "precipitation_amount", 0.0
                )
            )
            for hours in totals:
                if now_utc <= timestamp < now_utc + timedelta(hours=hours):
                    totals[hours] += rain
        except Exception:
            continue

    return {
        "lluvia_3h": round(totals[3], 2),
        "lluvia_6h": round(totals[6], 2),
        "lluvia_12h": round(totals[12], 2),
        "fuente": "MET Norway",
    }


def obtener_lluvia_exynus_operativa() -> dict[str, Any]:
    errors: list[str] = []
    open_data = None
    nasa_data = None
    met_data = None

    try:
        open_data = obtener_con_cache(
            "historico_open_meteo_exynus",
            60,
            obtener_historico_open_meteo_exynus,
        )
    except Exception as exc:
        errors.append(f"open_meteo: {type(exc).__name__}: {exc}")

    try:
        nasa_data = obtener_con_cache(
            "historico_nasa_power_exynus",
            360,
            obtener_historico_nasa_power_exynus,
        )
    except Exception as exc:
        errors.append(f"nasa_power: {type(exc).__name__}: {exc}")

    try:
        met_data = obtener_con_cache(
            "met_norway_exynus",
            60,
            obtener_lluvia_met_norway_mariato,
        )
    except Exception as exc:
        errors.append(f"met_norway: {type(exc).__name__}: {exc}")

    selected = open_data or nasa_data
    if selected is None:
        raise RuntimeError(
            "No fue posible obtener un histórico de lluvia. "
            + " | ".join(errors)
        )

    resumen_open = resumir_historico_fuente(open_data)
    resumen_nasa = resumir_historico_fuente(nasa_data)

    source = selected["fuente"]
    certainty = "alta" if open_data else "media_baja"
    consensus = (
        "multiestacion_open_meteo"
        if open_data
        else "respaldo_nasa_sin_pronostico"
    )
    warnings: list[str] = []

    latest_ratio = _safe_float(
        selected["historico"][-1].get("ratio_media_max_estacion"),
        1.0,
    )
    if open_data and latest_ratio < 0.25:
        certainty = "media"
        consensus = "desacuerdo_espacial_entre_estaciones"
        warnings.append(
            "La lluvia presenta variación espacial importante entre zonas."
        )

    if met_data:
        if _safe_float(met_data.get("lluvia_12h")) >= 20:
            consensus += "_confirmado_met_norway"
        elif _safe_float(selected.get("lluvia_pronostico_24h")) >= 20:
            warnings.append(
                "Open-Meteo proyecta lluvia, pero MET Norway no la confirma "
                "con la misma intensidad."
            )
            if certainty == "alta":
                certainty = "media_alta"

    if open_data and nasa_data:
        open_1d = resumen_open.get("lluvia1d")
        nasa_1d = resumen_nasa.get("lluvia1d")

        if open_1d is not None and nasa_1d is not None:
            if abs(float(open_1d) - float(nasa_1d)) >= 25:
                warnings.append(
                    "Open-Meteo y NASA POWER difieren en la lluvia diaria."
                )
                if certainty == "alta":
                    certainty = "media"

        if (
            resumen_open.get("ultimoDia")
            and resumen_nasa.get("ultimoDia")
            and resumen_open["ultimoDia"] != resumen_nasa["ultimoDia"]
        ):
            warnings.append(
                "NASA POWER tiene una fecha de observación distinta a "
                "Open-Meteo; la comparación se muestra con su último día "
                "disponible."
            )

    if not open_data:
        warnings.append(
            "Se usa NASA POWER como respaldo; no hay pronóstico horario "
            "multiestación para EXYNUS."
        )

    return {
        **selected,
        "_fuentes": {
            "fuenteLluvia": source,
            "consensoFuentes": consensus,
            "certezaDatos": certainty,
            "advertenciaDatos": (
                " ".join(warnings) if warnings else None
            ),
            "erroresFuentes": errors,

            "metNorway": met_data,

            "openMeteoDisponible": open_data is not None,
            "openMeteoLluvia1d": resumen_open["lluvia1d"],
            "openMeteoLluvia3d": resumen_open["lluvia3d"],
            "openMeteoLluvia7d": resumen_open["lluvia7d"],
            "openMeteoUltimoDia": resumen_open["ultimoDia"],
            "openMeteoDiasValidos": resumen_open["diasValidos"],

            "nasaPowerDisponible": nasa_data is not None,
            "nasaLluvia1d": resumen_nasa["lluvia1d"],
            "nasaLluvia3d": resumen_nasa["lluvia3d"],
            "nasaLluvia7d": resumen_nasa["lluvia7d"],
            "nasaUltimoDia": resumen_nasa["ultimoDia"],
            "nasaDiasValidos": resumen_nasa["diasValidos"],
        },
    }


# ============================================================
# ENSO, MAREA Y SISTEMAS TROPICALES
# ============================================================

def _normalizar_region_enso(value: str) -> str:
    normalized = (
        str(value)
        .upper()
        .replace("NIÑO", "NINO")
        .replace(" ", "")
    )
    if normalized in {"NINO34", "NINO3.4"}:
        return "NINO3.4"
    return normalized


def obtener_enso_noaa() -> dict[str, Any]:
    """Obtiene Niño 3.4 verificando el orden real del encabezado NOAA."""
    url = "https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for"
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    lines = [
        line.strip()
        for line in response.text.splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError("NOAA CPC devolvió una respuesta vacía.")

    header_text = " ".join(lines[:6]).upper().replace("NIÑO", "NINO")
    matches = re.findall(r"NINO(?:1\+2|3\.?4|4|3)", header_text)
    region_order: list[str] = []
    for match in matches:
        normalized = _normalizar_region_enso(match)
        if normalized not in region_order:
            region_order.append(normalized)

    if "NINO3.4" not in region_order:
        raise ValueError(
            "No fue posible localizar la columna Niño 3.4 en NOAA CPC."
        )

    valid_rows: list[tuple[str, list[float]]] = []
    for line in lines:
        raw_numbers = re.findall(r"[-+]?\d+\.\d+", line)
        if len(raw_numbers) >= 8:
            valid_rows.append(
                (line, [float(value) for value in raw_numbers[:8]])
            )

    if not valid_rows:
        raise ValueError("NOAA CPC no devolvió una fila semanal válida.")

    raw_line, values = valid_rows[-1]
    pair_index = region_order.index("NINO3.4")
    sst_index = pair_index * 2
    anomaly_index = sst_index + 1
    if anomaly_index >= len(values):
        raise ValueError(
            "La fila NOAA no contiene el par SST/anomalía de Niño 3.4."
        )

    sst_nino34 = float(values[sst_index])
    anomaly = float(values[anomaly_index])
    if not 10.0 <= sst_nino34 <= 35.0:
        raise ValueError(
            f"SST Niño 3.4 fuera de rango físico: {sst_nino34}"
        )
    if not -5.0 <= anomaly <= 5.0:
        raise ValueError(
            f"Anomalía Niño 3.4 fuera de rango esperado: {anomaly}"
        )

    phase = (
        "El Niño"
        if anomaly >= 0.5
        else "La Niña"
        if anomaly <= -0.5
        else "Neutral"
    )
    return {
        "fase_enso": phase,
        "anomalia_nino34": round(anomaly, 2),
        "sst_nino34": round(sst_nino34, 2),
        "orden_regiones_enso": region_order,
        "fuente": "NOAA CPC",
        "linea_fuente": raw_line,
    }


def obtener_marea_open_meteo() -> dict[str, Any]:
    response = requests.get(
        "https://marine-api.open-meteo.com/v1/marine",
        params={
            "latitude": LAT_MARIATO,
            "longitude": LON_MARIATO,
            "hourly": "sea_level_height_msl",
            "past_days": 2,
            "forecast_days": 2,
            "timezone": TIMEZONE_MARIATO,
        },
        timeout=25,
    )
    response.raise_for_status()
    hourly = response.json().get("hourly", {})
    times = hourly.get("time", [])
    levels = hourly.get("sea_level_height_msl", [])
    if not times or not levels:
        raise ValueError("Open-Meteo Marine no devolvió nivel del mar.")

    parsed_times = [_parse_open_meteo_time(value) for value in times]
    numeric_levels = np.asarray(
        [_safe_float(value) for value in levels], dtype=float
    )
    now = ahora_mariato()
    current_index = min(
        range(len(parsed_times)),
        key=lambda index: abs(
            (parsed_times[index] - now).total_seconds()
        ),
    )
    current = float(numeric_levels[current_index])
    p75 = float(np.percentile(numeric_levels, 75))
    p90 = float(np.percentile(numeric_levels, 90))

    if current >= p90:
        state, high = "Alta", 1
    elif current >= p75:
        state, high = "Elevada", 1
    else:
        state, high = "Normal", 0

    return {
        "marea_alta": high,
        "marea_estado": state,
        "nivel_mar_proxy": round(current, 3),
        "fuente": "Open-Meteo Marine",
    }


def _coordinate(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group())
    if "S" in text or "W" in text:
        number = -abs(number)
    return number


def _recursive_value(value: Any, keys: set[str]) -> Any:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in keys:
                return child
        for child in value.values():
            found = _recursive_value(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _recursive_value(child, keys)
            if found is not None:
                return found
    return None


def obtener_sistema_tropical_nhc() -> dict[str, Any]:
    response = requests.get(
        "https://www.nhc.noaa.gov/CurrentStorms.json",
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()

    storms = []
    if isinstance(payload, Mapping):
        for key in ("activeStorms", "storms", "CurrentStorms"):
            value = payload.get(key)
            if isinstance(value, list):
                storms = value
                break

    relevant = []
    for storm in storms:
        latitude = _coordinate(
            _recursive_value(
                storm, {"lat", "latitude", "centerlat", "center_lat"}
            )
        )
        longitude = _coordinate(
            _recursive_value(
                storm,
                {"lon", "lng", "longitude", "centerlon", "center_lon"},
            )
        )
        name = _recursive_value(
            storm, {"name", "stormname", "storm_name", "systemname"}
        )
        if (
            latitude is not None
            and longitude is not None
            and 0 <= latitude <= 25
            and -110 <= longitude <= -55
        ):
            relevant.append(
                {"nombre": name, "lat": latitude, "lon": longitude}
            )

    return {
        "sistema_tropical_activo": int(bool(relevant)),
        "nombre_sistema_tropical": (
            relevant[0]["nombre"] if relevant else None
        ),
        "detalle": relevant or "sin_sistemas_cercanos",
        "fuente": "NOAA NHC",
    }


def integrar_historico_cerro_hoya(
    historico_base: list[dict[str, Any]],
    cerro_hoya: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Une por fecha la señal de Cerro Hoya al histórico operativo EXYNUS."""
    history = cerro_hoya.get("historico")
    if not isinstance(history, list) or not history:
        return [dict(row) for row in historico_base]

    by_date = {
        str(row.get("fecha")): row
        for row in history
        if isinstance(row, Mapping) and row.get("fecha")
    }
    merged: list[dict[str, Any]] = []
    for base_row in historico_base:
        row = dict(base_row)
        date = str(row.get("fecha"))
        upstream = by_date.get(date)
        if upstream is not None:
            row["lluvia_cerro_hoya_diaria"] = _safe_float(
                upstream.get("lluvia_cerro_hoya_diaria"), 0.0
            )
            row["cerro_hoya_dato_disponible"] = bool(
                upstream.get("cerro_hoya_dato_disponible", True)
            )
        else:
            row["lluvia_cerro_hoya_diaria"] = 0.0
            row["cerro_hoya_dato_disponible"] = False
        merged.append(row)
    return merged


def obtener_datos_actuales_exynus() -> dict[str, Any]:
    rain = obtener_lluvia_exynus_operativa()
    errors = list(rain["_fuentes"].get("erroresFuentes", []))

    cerro_hoya = obtener_cerro_hoya_operativo()
    errors.extend(cerro_hoya.get("erroresFuentes", []))

    historico_integrado = integrar_historico_cerro_hoya(
        rain["historico"], cerro_hoya
    )

    try:
        enso = obtener_con_cache("enso_noaa", 360, obtener_enso_noaa)
    except Exception as exc:
        errors.append(f"enso_noaa: {type(exc).__name__}: {exc}")
        enso = {
            "fase_enso": "Neutral",
            "anomalia_nino34": 0.10,
            "fuente": "respaldo_neutral",
        }

    try:
        tide = obtener_con_cache(
            "marea_open_meteo", 30, obtener_marea_open_meteo
        )
    except Exception as exc:
        errors.append(f"marea: {type(exc).__name__}: {exc}")
        tide = {
            "marea_alta": 0,
            "marea_estado": "Sin datos",
            "nivel_mar_proxy": None,
            "fuente": "respaldo_sin_marea",
        }

    try:
        tropical = obtener_con_cache(
            "ciclones_nhc", 30, obtener_sistema_tropical_nhc
        )
    except Exception as exc:
        errors.append(f"nhc: {type(exc).__name__}: {exc}")
        tropical = {
            "sistema_tropical_activo": 0,
            "nombre_sistema_tropical": None,
            "detalle": "error_consulta",
            "fuente": "respaldo_sin_ciclon",
        }

    sources = dict(rain["_fuentes"])
    sources["erroresFuentes"] = errors
    sources["fuenteEnso"] = enso.get("fuente")
    sources["fuenteMarea"] = tide.get("fuente")
    sources["fuenteCiclones"] = tropical.get("fuente")
    sources["fuenteCerroHoya"] = cerro_hoya.get("fuente")
    sources["cerroHoyaDisponible"] = bool(cerro_hoya.get("disponible"))
    sources["cerroHoyaReferencia"] = cerro_hoya.get("referencia")
    sources["cerroHoyaTipo"] = cerro_hoya.get("tipo")

    if errors and sources.get("certezaDatos") == "alta":
        sources["certezaDatos"] = "media_alta"

    return {
        "historico": historico_integrado,
        "lluvia_pronostico_24h": rain["lluvia_pronostico_24h"],
        "lluvia_pronostico_72h": rain["lluvia_pronostico_72h"],
        "lluvia_pronostico_7d": rain["lluvia_pronostico_7d"],
        "cerro_hoya_disponible": bool(cerro_hoya.get("disponible")),
        "lluvia_cerro_hoya_pronostico_24h": _safe_float(
            cerro_hoya.get("lluvia_pronostico_24h"), 0.0
        ),
        "lluvia_cerro_hoya_pronostico_72h": _safe_float(
            cerro_hoya.get("lluvia_pronostico_72h"), 0.0
        ),
        "lluvia_cerro_hoya_pronostico_7d": _safe_float(
            cerro_hoya.get("lluvia_pronostico_7d"), 0.0
        ),
        "fuente_cerro_hoya": cerro_hoya.get("fuente"),
        "fase_enso": enso["fase_enso"],
        "anomalia_nino34": enso["anomalia_nino34"],
        "marea_alta": tide["marea_alta"],
        "marea_estado": tide["marea_estado"],
        "nivel_mar_proxy": tide.get("nivel_mar_proxy"),
        "sistema_tropical_activo": tropical[
            "sistema_tropical_activo"
        ],
        "nombre_sistema_tropical": tropical.get(
            "nombre_sistema_tropical"
        ),
        "fecha_actualizacion": ahora_mariato().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "_fuentes": sources,
    }

MAX_SUPPORT_SCORE = 13


def completar_contrato_exynus(
    resultado: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Completa los campos públicos requeridos por ExynusPredictionResponse.

    Esta capa evita errores 500 cuando el motor EXYNUS desplegado
    todavía devuelve un contrato anterior.
    """

    respuesta = dict(resultado)
    safe_config = obtener_safe_config()

    probabilidad = float(
        respuesta.get(
            "probabilidadExynus",
            respuesta.get("probabilidadFinal", 0.0)
        ) or 0.0
    )

    umbral = float(
        respuesta.get(
            "umbralExynus",
            safe_config.decision_threshold
        ) or safe_config.decision_threshold
    )

    epsilon = float(
        getattr(safe_config, "threshold_epsilon", 1e-7) or 0.0
    )

    umbral_efectivo = max(umbral - epsilon, 0.0)

    supera_nominal = probabilidad >= umbral
    supera_efectivo = probabilidad >= umbral_efectivo

    margen_nominal = probabilidad - umbral
    margen_efectivo = probabilidad - umbral_efectivo

    relacion_umbral = (
        probabilidad / umbral
        if umbral > 0
        else 0.0
    )

    # Completa y normaliza el nuevo índice compuesto. Esta compatibilidad
    # permite que el contrato siga siendo válido durante el reemplazo de
    # archivos, incluso si una instancia conserva temporalmente el motor
    # anterior en memoria.
    indice_riesgo = float(np.clip(
        _safe_float(respuesta.get("indiceRiesgoOperativo"), 0.01),
        0.01,
        1.0,
    ))
    porcentaje_riesgo = float(np.clip(
        _safe_float(
            respuesta.get("riesgoOperativoPorcentaje"),
            indice_riesgo * 100.0,
        ),
        1.0,
        100.0,
    ))

    bandas_validas = {
        "normal", "vigilancia", "moderado", "alto", "critico"
    }
    banda_riesgo = str(
        respuesta.get(
            "bandaRiesgoOperativo",
            respuesta.get("alertaFinal", "normal"),
        )
    ).strip().casefold()
    if banda_riesgo not in bandas_validas:
        banda_riesgo = "normal"

    aporte_modelo = _safe_float(
        respuesta.get("aporteModeloRiesgo"),
        np.clip(relacion_umbral, 0.0, 1.0),
    )
    aporte_modelo = float(np.clip(aporte_modelo, 0.0, 1.0))

    evidencia_fisica = _safe_float(
        respuesta.get("evidenciaFisicaRiesgo"),
        0.0,
    )
    evidencia_fisica = float(np.clip(evidencia_fisica, 0.0, 1.0))

    componentes_riesgo = respuesta.get(
        "componentesRiesgoOperativo",
        {},
    )
    if not isinstance(componentes_riesgo, Mapping):
        componentes_riesgo = {}

    prediccion_safe = bool(
        respuesta.get("prediccionExynusSafe", False)
    )

    critical_override = bool(
        respuesta.get("criticalOverride", False)
    )

    low_support_veto = bool(
        respuesta.get("lowSupportVeto", False)
    )

    residual_suppression = bool(
        respuesta.get("residualSuppression", False)
    )

    soporte = int(
        respuesta.get("soporteAlerta", 0) or 0
    )

    soporte = max(
        0,
        min(soporte, MAX_SUPPORT_SCORE)
    )

    porcentaje_soporte = soporte / MAX_SUPPORT_SCORE

    if porcentaje_soporte >= 0.90:
        soporte_nivel = "muy_alto"
        soporte_texto = "Muy alto"
    elif porcentaje_soporte >= 0.70:
        soporte_nivel = "alto"
        soporte_texto = "Alto"
    elif porcentaje_soporte >= 0.45:
        soporte_nivel = "medio"
        soporte_texto = "Medio"
    elif porcentaje_soporte > 0:
        soporte_nivel = "bajo"
        soporte_texto = "Bajo"
    else:
        soporte_nivel = "sin_soporte"
        soporte_texto = "Sin soporte"

    confianza_codigo = str(
        respuesta.get("confianzaExynus", "sin_alerta")
    )

    textos_confianza = {
        "muy_alta": "Muy alta",
        "alta": "Alta",
        "media": "Media",
        "baja": "Baja",
        "sin_alerta": "No aplica: sin alerta confirmada",
    }

    confianza_texto = textos_confianza.get(
        confianza_codigo,
        confianza_codigo.replace("_", " ").capitalize()
    )

    respuesta["umbralExynus"] = round(umbral, 10)
    respuesta["umbralExynusEfectivo"] = round(
        umbral_efectivo,
        10
    )
    respuesta["epsilonUmbralExynus"] = round(
        epsilon,
        10
    )

    respuesta["umbralExynusSuperado"] = bool(
        supera_efectivo
    )
    respuesta["umbralExynusNominalSuperado"] = bool(
        supera_nominal
    )

    respuesta["margenUmbralExynus"] = round(
        margen_nominal,
        10
    )
    respuesta["margenUmbralExynusEfectivo"] = round(
        margen_efectivo,
        10
    )
    respuesta["relacionUmbralExynus"] = round(
        relacion_umbral,
        6
    )

    respuesta["indiceRiesgoOperativo"] = round(indice_riesgo, 4)
    respuesta["riesgoOperativoPorcentaje"] = round(
        porcentaje_riesgo,
        2,
    )
    respuesta["bandaRiesgoOperativo"] = banda_riesgo
    respuesta["tipoIndiceRiesgoOperativo"] = str(
        respuesta.get(
            "tipoIndiceRiesgoOperativo",
            "indice_compuesto_no_calibrado",
        )
    )
    respuesta["aporteModeloRiesgo"] = round(aporte_modelo, 4)
    respuesta["evidenciaFisicaRiesgo"] = round(
        evidencia_fisica,
        4,
    )
    respuesta["componentesRiesgoOperativo"] = dict(
        componentes_riesgo
    )

    respuesta["alertaConfirmada"] = prediccion_safe
    respuesta["metaModeloSuperaUmbral"] = bool(
        supera_efectivo
    )
    respuesta["safeAutorizaAlerta"] = prediccion_safe

    respuesta["confianzaExynusTexto"] = confianza_texto

    respuesta["soporteAlerta"] = soporte
    respuesta["soporteAlertaMaximo"] = MAX_SUPPORT_SCORE
    respuesta["soporteAlertaPorcentaje"] = round(
        porcentaje_soporte,
        4
    )
    respuesta["soporteAlertaNivel"] = soporte_nivel
    respuesta["soporteAlertaTexto"] = soporte_texto

    respuesta["notificacionCriticaPermitida"] = bool(
        critical_override
        or (
            prediccion_safe
            and not low_support_veto
            and not residual_suppression
        )
    )

    # Correcciones visibles del contrato actual.
    if respuesta.get("alertaFinal") == "normal":
        respuesta["estadoTitulo"] = "CONDICIONES NORMALES"
        respuesta["mensaje"] = (
            "PLUVIAP no identifica condiciones suficientes para "
            "una alerta de inundación en las próximas 24 horas."
        )

    return respuesta

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "mensaje": "PLUVIAP API funcionando",
        "modelo": "EXYNUS v5.1-SAFE",
        "version": "5.1.0",
        "endpointPrediccion": "/current-prediction",
        "endpointMapa": "/rain-map-v2",
    }


@app.get("/health")
def health():
    estado = estado_modelos_exynus()
    modelos_cargados = bool(
        estado.get("modelos_exynus_cargados", False)
    )
    return {
        "status": "ok" if modelos_cargados else "degraded",
        "modelo": "EXYNUS v5.1-SAFE",
        "modelosCargados": modelos_cargados,
        "modelosCargando": bool(
            estado.get("modelos_exynus_cargando", False)
        ),
        "errorModelo": estado.get("error"),
        "horaMariato": ahora_mariato().isoformat(),
    }


@app.get("/exynus-health")
def exynus_health():
    return estado_modelos_exynus()


@app.post("/predict", response_model=ExynusPredictionResponse)
def predict(request: PredictionRequest):
    try:
        return completar_contrato_exynus(
            predecir_exynus(request.features)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"No fue posible ejecutar EXYNUS: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/predict-simple", response_model=ExynusPredictionResponse)
def predict_simple(request: SimplePredictionRequest):
    values = request.model_dump()
    if values.get("mes") is None:
        values["mes"] = ahora_mariato().month
    values["_fuentes"] = {
        "fuenteLluvia": "entrada_manual",
        "consensoFuentes": "sin_comparacion_manual",
        "certezaDatos": "prueba",
        "advertenciaDatos": (
            "Predicción manual; no usa las fuentes meteorológicas operativas."
        ),
    }
    try:
        return completar_contrato_exynus(
            predecir_exynus(values)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"No fue posible ejecutar EXYNUS: {type(exc).__name__}: {exc}",
        ) from exc


@app.get(
    "/current-prediction",
    response_model=ExynusPredictionResponse,
)
def current_prediction():
    """Endpoint oficial consumido por Android: usa EXYNUS v5.1-SAFE."""
    try:
        data = obtener_datos_actuales_exynus()
        result = completar_contrato_exynus(
            predecir_exynus(data)
        )

        fuentes = data.get("_fuentes", {}) or {}
        result["fuentes"] = fuentes
        result["comparacionFuentes"] = construir_comparacion_fuentes(
            fuentes
        )
        result["nivelMarProxy"] = data.get("nivel_mar_proxy")
        result["nombreSistemaTropical"] = data.get(
            "nombre_sistema_tropical"
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "No fue posible construir la predicción EXYNUS: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc


@app.get(
    "/current-prediction-exynus",
    response_model=ExynusPredictionResponse,
)
def current_prediction_exynus():
    """Alias explícito durante pruebas de integración."""
    return current_prediction()


@app.get("/enso-debug")
def enso_debug():
    try:
        result = obtener_enso_noaa()
        return {
            "ok": True,
            "faseEnso": result["fase_enso"],
            "anomaliaNino34": result["anomalia_nino34"],
            "sstNino34": result.get("sst_nino34"),
            "ordenRegionesDetectado": result.get(
                "orden_regiones_enso"
            ),
            "fuente": result["fuente"],
            "lineaFuente": result.get("linea_fuente"),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


@app.get("/weather-debug")
def weather_debug():
    try:
        data = obtener_datos_actuales_exynus()
        return {
            "ok": True,
            "diasHistoricos": len(data["historico"]),
            "ultimoRegistro": data["historico"][-1],
            "pronostico24h": data["lluvia_pronostico_24h"],
            "pronostico72h": data["lluvia_pronostico_72h"],
            "pronostico7d": data["lluvia_pronostico_7d"],
            "cerroHoya": {
                "disponible": data.get("cerro_hoya_disponible"),
                "pronostico24h": data.get("lluvia_cerro_hoya_pronostico_24h"),
                "pronostico72h": data.get("lluvia_cerro_hoya_pronostico_72h"),
                "pronostico7d": data.get("lluvia_cerro_hoya_pronostico_7d"),
                "fuente": data.get("fuente_cerro_hoya"),
            },
            "fuentes": data["_fuentes"],
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


@app.get("/sources-debug")
def sources_debug():
    try:
        data = obtener_datos_actuales_exynus()
        return {
            "fuentes": data["_fuentes"],
            "faseEnso": data["fase_enso"],
            "anomaliaNino34": data["anomalia_nino34"],
            "mareaEstado": data["marea_estado"],
            "sistemaTropicalActivo": data[
                "sistema_tropical_activo"
            ],
            "cerroHoyaDisponible": data.get("cerro_hoya_disponible"),
            "fuenteCerroHoya": data.get("fuente_cerro_hoya"),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


@app.get("/nasa-power-debug")
def nasa_power_debug():
    """
    Verifica que NASA POWER entregue histórico válido y que sus
    acumulados de 1, 3 y 7 días puedan enviarse a Android.
    """
    try:
        data = obtener_con_cache(
            "historico_nasa_power_exynus",
            360,
            obtener_historico_nasa_power_exynus,
        )
        resumen = resumir_historico_fuente(data)
        historico = data.get("historico", [])

        return {
            "ok": all(
                resumen.get(clave) is not None
                for clave in ("lluvia1d", "lluvia3d", "lluvia7d")
            ),
            "fuente": data.get("fuente"),
            "diasHistoricos": len(historico),
            "resumen": resumen,
            "ultimos7Registros": historico[-7:],
            "cache": {
                "usado": data.get("_cache", False),
                "estado": data.get("_cache_estado"),
                "edadMinutos": data.get("_cache_edad_minutos"),
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


@app.get("/rain-consensus-debug")
def rain_consensus_debug():
    try:
        rain = obtener_lluvia_exynus_operativa()
        return {
            "fuentes": rain["_fuentes"],
            "ultimoRegistro": rain["historico"][-1],
            "pronostico24h": rain["lluvia_pronostico_24h"],
            "pronostico72h": rain["lluvia_pronostico_72h"],
            "pronostico7d": rain["lluvia_pronostico_7d"],
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


@app.get("/cerro-hoya-debug")
def cerro_hoya_debug():
    """Diagnóstico de la nueva señal aguas arriba sin exponer secretos."""
    try:
        data = obtener_cerro_hoya_operativo()
        history = data.get("historico", [])
        return {
            "ok": bool(data.get("disponible")),
            "configuracion": estado_cerro_hoya(),
            "fuente": data.get("fuente"),
            "diasHistoricos": len(history),
            "ultimoRegistro": history[-1] if history else None,
            "pronostico24h": data.get("lluvia_pronostico_24h"),
            "pronostico72h": data.get("lluvia_pronostico_72h"),
            "pronostico7d": data.get("lluvia_pronostico_7d"),
            "erroresFuentes": data.get("erroresFuentes", []),
            "cache": {
                "usado": data.get("_cache", False),
                "estado": data.get("_cache_estado"),
                "edadMinutos": data.get("_cache_edad_minutos"),
            },
            "nota": (
                "Cerro Hoya refuerza la política SAFE como evidencia aguas "
                "arriba; no modifica directamente la probabilidad del "
                "meta-modelo EXYNUS."
            ),
        }
    except Exception as exc:
        return {
            "ok": False,
            "configuracion": estado_cerro_hoya(),
            "error": f"{type(exc).__name__}: {exc}",
        }


@app.get("/routes-debug")
def routes_debug():
    return {
        "routes": sorted(
            {
                route.path
                for route in app.routes
                if hasattr(route, "path")
            }
        )
    }


@app.get("/open-meteo-config-debug")
def open_meteo_config_debug():
    return {
        "apiKeyConfigurada": bool(OPEN_METEO_API_KEY),
        "modo": "comercial" if OPEN_METEO_API_KEY else "gratuito",
        "endpoint": OPEN_METEO_FORECAST_URL,
        "claveExpuesta": False,
    }

@app.get("/render-health")
def render_health():
    """
    Health check de proceso para Render.

    Siempre devuelve HTTP 200 cuando FastAPI está atendiendo solicitudes.
    El estado detallado de los modelos se consulta en /exynus-health.
    """
    estado = estado_modelos_exynus()
    modelos_cargados = bool(
        estado.get("modelos_exynus_cargados", False)
    )
    modelos_cargando = bool(
        estado.get("modelos_exynus_cargando", False)
    )
    error_modelo = estado.get("error")

    if modelos_cargados:
        status = "ok"
    elif modelos_cargando:
        status = "starting"
    elif error_modelo:
        status = "degraded"
    else:
        status = "starting"

    return {
        "status": status,
        "servicioActivo": True,
        "modelo": "EXYNUS v5.1-SAFE",
        "modelosCargados": modelos_cargados,
        "modelosCargando": modelos_cargando,
        "errorModelo": error_modelo,
    }

