"""Monitoreo hidrometeorológico regional de Cerro Hoya para PLUVIAP EXYNUS SAFE.

El histórico usado para la comparación científica se construyó en Google Earth
Engine sobre una ventana regional del macizo de Cerro Hoya y restringiendo el
relieve a >= 500 m. La señal operativa de este módulo aproxima esa misma idea
consultando una malla reproducible de puntos con Open-Meteo, filtrando los
puntos cuya elevación reportada es >= 500 m y calculando el percentil 90 (P90)
de la precipitación diaria entre los puntos de montaña disponibles.

IMPORTANTE: la comparación A/B realizada con el histórico CHIRPS 2015-2025 no
mostró una mejora robusta al incorporar Cerro Hoya como feature predictiva y la
regla SAFE predefinida aumentó falsos positivos. Por ello, Cerro Hoya se conserva
como monitoreo aguas arriba y variable explicativa. NO modifica la decisión del
meta-modelo ni la política SAFE en producción.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo
import os
import threading

import numpy as np
import requests


TIMEZONE_MARIATO = "America/Panama"
TZ_MARIATO = ZoneInfo(TIMEZONE_MARIATO)

# Definición reproducible de la región de estudio utilizada para el proxy.
CERRO_HOYA_OESTE = -80.90
CERRO_HOYA_ESTE = -80.50
CERRO_HOYA_SUR = 7.10
CERRO_HOYA_NORTE = 7.45
CERRO_HOYA_ELEVACION_MIN_M = 500.0
CERRO_HOYA_GRID_N = 5
CERRO_HOYA_REFERENCIA = (
    "Región montañosa de Cerro Hoya: ventana -80.90/-80.50, 7.10/7.45; "
    "puntos operativos con elevación >=500 m; precipitación regional P90"
)


def _crear_malla() -> list[tuple[float, float]]:
    """Genera una malla de centros de celda para evitar depender de un único punto."""
    paso_lat = (CERRO_HOYA_NORTE - CERRO_HOYA_SUR) / CERRO_HOYA_GRID_N
    paso_lon = (CERRO_HOYA_ESTE - CERRO_HOYA_OESTE) / CERRO_HOYA_GRID_N
    puntos: list[tuple[float, float]] = []
    for i in range(CERRO_HOYA_GRID_N):
        lat = CERRO_HOYA_SUR + (i + 0.5) * paso_lat
        for j in range(CERRO_HOYA_GRID_N):
            lon = CERRO_HOYA_OESTE + (j + 0.5) * paso_lon
            puntos.append((round(lat, 5), round(lon, 5)))
    return puntos


CERRO_HOYA_GRID = _crear_malla()
CERRO_HOYA_LAT = (CERRO_HOYA_SUR + CERRO_HOYA_NORTE) / 2.0
CERRO_HOYA_LON = (CERRO_HOYA_OESTE + CERRO_HOYA_ESTE) / 2.0

OPEN_METEO_API_KEY = os.getenv("OPEN_METEO_API_KEY")
OPEN_METEO_FORECAST_URL = (
    "https://customer-api.open-meteo.com/v1/forecast"
    if OPEN_METEO_API_KEY
    else "https://api.open-meteo.com/v1/forecast"
)
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _ahora() -> datetime:
    return datetime.now(TZ_MARIATO)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if not np.isfinite(number):
            return float(default)
        return number
    except Exception:
        return float(default)


def _con_cache(nombre: str, ttl_minutos: int, funcion):
    ahora = _ahora()
    with _CACHE_LOCK:
        item = _CACHE.get(nombre)
        if item is not None:
            edad = ahora - item["fecha"]
            if edad < timedelta(minutes=ttl_minutos):
                result = dict(item["resultado"])
                result["_cache"] = True
                result["_cache_estado"] = "cache_vigente"
                result["_cache_edad_minutos"] = round(edad.total_seconds() / 60.0, 2)
                return result

    try:
        result = dict(funcion())
        with _CACHE_LOCK:
            _CACHE[nombre] = {"fecha": ahora, "resultado": dict(result)}
        result["_cache"] = False
        result["_cache_estado"] = "consulta_nueva"
        result["_cache_edad_minutos"] = 0.0
        return result
    except Exception as exc:
        with _CACHE_LOCK:
            item = _CACHE.get(nombre)
        if item is not None:
            edad = ahora - item["fecha"]
            result = dict(item["resultado"])
            result["_cache"] = True
            result["_cache_estado"] = "cache_por_error_fuente"
            result["_cache_edad_minutos"] = round(edad.total_seconds() / 60.0, 2)
            result["_cache_error"] = f"{type(exc).__name__}: {exc}"
            return result
        raise


def _parse_local(timestamp: Any) -> datetime:
    parsed = datetime.fromisoformat(str(timestamp))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TZ_MARIATO)
    else:
        parsed = parsed.astimezone(TZ_MARIATO)
    return parsed


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), 90))


def _acumulado_horizonte(
    times: list[datetime],
    rain: list[float],
    inicio: datetime,
    fin: datetime,
) -> float:
    return float(
        sum(value for timestamp, value in zip(times, rain) if inicio <= timestamp < fin)
    )


def obtener_cerro_hoya_open_meteo() -> dict[str, Any]:
    """Obtiene P90 regional de lluvia para puntos de Cerro Hoya >=500 m."""
    latitudes = ",".join(str(lat) for lat, _ in CERRO_HOYA_GRID)
    longitudes = ",".join(str(lon) for _, lon in CERRO_HOYA_GRID)
    params: dict[str, Any] = {
        "latitude": latitudes,
        "longitude": longitudes,
        "hourly": "precipitation",
        "past_days": 45,
        "forecast_days": 7,
        "timezone": TIMEZONE_MARIATO,
    }
    if OPEN_METEO_API_KEY:
        params["apikey"] = OPEN_METEO_API_KEY

    response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=35)
    response.raise_for_status()
    payload = response.json()
    payloads = payload if isinstance(payload, list) else [payload]
    if len(payloads) != len(CERRO_HOYA_GRID):
        raise ValueError("Open-Meteo devolvió una cantidad inesperada de puntos Cerro Hoya.")

    series_montana: list[dict[str, Any]] = []
    for (lat, lon), point in zip(CERRO_HOYA_GRID, payloads):
        elevation = _safe_float(point.get("elevation"), -999.0)
        if elevation < CERRO_HOYA_ELEVACION_MIN_M:
            continue
        hourly = point.get("hourly", {})
        raw_times = hourly.get("time", [])
        raw_rain = hourly.get("precipitation", [])
        if not raw_times or not raw_rain:
            continue
        times: list[datetime] = []
        rain: list[float] = []
        for timestamp, value in zip(raw_times, raw_rain):
            try:
                times.append(_parse_local(timestamp))
                rain.append(max(_safe_float(value), 0.0))
            except Exception:
                continue
        if len(times) < 24:
            continue
        daily: dict[str, float] = {}
        today = _ahora().date()
        for timestamp, value in zip(times, rain):
            if timestamp.date() <= today:
                key = timestamp.date().isoformat()
                daily[key] = daily.get(key, 0.0) + value
        series_montana.append(
            {
                "lat": lat,
                "lon": lon,
                "elevation": elevation,
                "times": times,
                "rain": rain,
                "daily": daily,
            }
        )

    if len(series_montana) < 3:
        raise ValueError(
            "Open-Meteo no devolvió suficientes puntos >=500 m para el P90 regional de Cerro Hoya."
        )

    all_dates = sorted(set().union(*(set(item["daily"]) for item in series_montana)))
    rows: list[dict[str, Any]] = []
    for key in all_dates:
        values = [item["daily"][key] for item in series_montana if key in item["daily"]]
        if not values:
            continue
        rows.append(
            {
                "fecha": key,
                "lluvia_cerro_hoya_diaria": round(_p90(values), 4),
                "cerro_hoya_dato_disponible": True,
                "puntos_montana": len(values),
            }
        )
    if len(rows) < 30:
        raise ValueError("Serie regional P90 insuficiente para Cerro Hoya.")

    now = _ahora()
    h24: list[float] = []
    h72: list[float] = []
    h7d: list[float] = []
    for item in series_montana:
        h24.append(_acumulado_horizonte(item["times"], item["rain"], now, now + timedelta(hours=24)))
        h72.append(_acumulado_horizonte(item["times"], item["rain"], now, now + timedelta(hours=72)))
        h7d.append(_acumulado_horizonte(item["times"], item["rain"], now, now + timedelta(days=7)))

    elevations = [item["elevation"] for item in series_montana]
    return {
        "disponible": True,
        "historico": rows,
        "lluvia_pronostico_24h": round(_p90(h24), 2),
        "lluvia_pronostico_72h": round(_p90(h72), 2),
        "lluvia_pronostico_7d": round(_p90(h7d), 2),
        "fuente": "Open-Meteo Cerro Hoya P90 regional",
        "lat": round(CERRO_HOYA_LAT, 5),
        "lon": round(CERRO_HOYA_LON, 5),
        "referencia": CERRO_HOYA_REFERENCIA,
        "dias_historicos": len(rows),
        "tipo": "proxy_regional_p90_montana_500m",
        "puntosMalla": len(CERRO_HOYA_GRID),
        "puntosMontanaValidos": len(series_montana),
        "elevacionMinimaM": CERRO_HOYA_ELEVACION_MIN_M,
        "elevacionMinPuntosM": round(min(elevations), 1),
        "elevacionMaxPuntosM": round(max(elevations), 1),
        "metodoAgregacion": "P90 espacial de precipitación acumulada",
        "participaDecisionExynus": False,
    }


def obtener_cerro_hoya_nasa_power() -> dict[str, Any]:
    """Respaldo puntual observado; se etiqueta como no equivalente al P90 regional."""
    today = _ahora().date()
    start = today - timedelta(days=50)
    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "longitude": CERRO_HOYA_LON,
        "latitude": CERRO_HOYA_LAT,
        "start": start.strftime("%Y%m%d"),
        "end": today.strftime("%Y%m%d"),
        "format": "JSON",
    }
    response = requests.get(NASA_POWER_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    values = payload.get("properties", {}).get("parameter", {}).get("PRECTOTCORR", {})
    if not values:
        raise ValueError("NASA POWER no devolvió lluvia para Cerro Hoya.")

    rows: list[dict[str, Any]] = []
    for key in sorted(values):
        value = _safe_float(values.get(key), -999.0)
        if value < 0:
            continue
        rows.append(
            {
                "fecha": datetime.strptime(key, "%Y%m%d").date().isoformat(),
                "lluvia_cerro_hoya_diaria": round(value, 4),
                "cerro_hoya_dato_disponible": True,
                "puntos_montana": 1,
            }
        )
    if len(rows) < 30:
        raise ValueError("NASA POWER no devolvió 30 días válidos para Cerro Hoya.")

    return {
        "disponible": True,
        "historico": rows,
        "lluvia_pronostico_24h": 0.0,
        "lluvia_pronostico_72h": 0.0,
        "lluvia_pronostico_7d": 0.0,
        "fuente": "NASA POWER Cerro Hoya respaldo puntual",
        "lat": round(CERRO_HOYA_LAT, 5),
        "lon": round(CERRO_HOYA_LON, 5),
        "referencia": CERRO_HOYA_REFERENCIA,
        "dias_historicos": len(rows),
        "tipo": "respaldo_puntual_no_equivalente_p90_regional",
        "puntosMalla": 1,
        "puntosMontanaValidos": 1,
        "elevacionMinimaM": CERRO_HOYA_ELEVACION_MIN_M,
        "metodoAgregacion": "Punto central; solo respaldo cuando falla Open-Meteo",
        "participaDecisionExynus": False,
    }


def obtener_cerro_hoya_operativo() -> dict[str, Any]:
    """Selecciona P90 regional Open-Meteo y usa NASA POWER solo como respaldo."""
    errors: list[str] = []
    try:
        result = _con_cache("cerro_hoya_open_meteo_regional", 60, obtener_cerro_hoya_open_meteo)
        result["erroresFuentes"] = errors
        return result
    except Exception as exc:
        errors.append(f"open_meteo_cerro_hoya: {type(exc).__name__}: {exc}")

    try:
        result = _con_cache("cerro_hoya_nasa_power", 360, obtener_cerro_hoya_nasa_power)
        result["erroresFuentes"] = errors
        return result
    except Exception as exc:
        errors.append(f"nasa_power_cerro_hoya: {type(exc).__name__}: {exc}")

    return {
        "disponible": False,
        "historico": [],
        "lluvia_pronostico_24h": 0.0,
        "lluvia_pronostico_72h": 0.0,
        "lluvia_pronostico_7d": 0.0,
        "fuente": "sin_fuente_cerro_hoya",
        "lat": round(CERRO_HOYA_LAT, 5),
        "lon": round(CERRO_HOYA_LON, 5),
        "referencia": CERRO_HOYA_REFERENCIA,
        "dias_historicos": 0,
        "tipo": "sin_datos",
        "puntosMalla": len(CERRO_HOYA_GRID),
        "puntosMontanaValidos": 0,
        "elevacionMinimaM": CERRO_HOYA_ELEVACION_MIN_M,
        "metodoAgregacion": "P90 regional no disponible",
        "participaDecisionExynus": False,
        "erroresFuentes": errors,
    }


def estado_cerro_hoya() -> dict[str, Any]:
    """Metadatos públicos del módulo sin exponer secretos."""
    return {
        "nombre": "PLUVIAP EXYNUS SAFE",
        "modulo": "Cerro Hoya",
        "lat": round(CERRO_HOYA_LAT, 5),
        "lon": round(CERRO_HOYA_LON, 5),
        "referencia": CERRO_HOYA_REFERENCIA,
        "openMeteoComercial": bool(OPEN_METEO_API_KEY),
        "region": {
            "oeste": CERRO_HOYA_OESTE,
            "este": CERRO_HOYA_ESTE,
            "sur": CERRO_HOYA_SUR,
            "norte": CERRO_HOYA_NORTE,
            "elevacionMinimaM": CERRO_HOYA_ELEVACION_MIN_M,
            "mallaPuntos": len(CERRO_HOYA_GRID),
        },
        "rol": (
            "monitoreo hidrometeorológico aguas arriba y variable explicativa; "
            "la comparación A/B no justificó incorporarlo como feature predictiva"
        ),
        "participaDecisionExynus": False,
    }
