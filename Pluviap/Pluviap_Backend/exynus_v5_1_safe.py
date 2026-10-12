"""Política temporal de seguridad para EXYNUS v5.1-SAFE.

Este módulo recibe las señales reconstruidas por ``exynus_engine`` y aplica
la política SAFE: persistencia, respaldo físico, consenso, veto de señales
aisladas, supresión residual y excepción crítica multiseñal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


# Escala pública del respaldo total de la política SAFE.
MAX_PHYSICAL_VOTES = 8
MAX_CONSENSUS_VOTES = 3
MAX_PERSISTENCE_VOTES = 2
MAX_SUPPORT_SCORE = (
    MAX_PHYSICAL_VOTES
    + MAX_CONSENSUS_VOTES
    + MAX_PERSISTENCE_VOTES
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _num(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(float(default))


def _bool(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(bool(default), index=frame.index, dtype=bool)
    series = frame[name]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(bool(default)).astype(bool)
    return pd.to_numeric(series, errors="coerce").fillna(int(default)).gt(0)


@dataclass(frozen=True)
class SafePolicyConfig:
    """Configuración efectiva de EXYNUS v5.1-SAFE."""

    version: str = "EXYNUS v5.1-SAFE"
    decision_threshold: float = 0.01002760510891676
    threshold_epsilon: float = 1e-7
    season_start_month: int = 5
    season_end_month: int = 11

    # Persistencia.
    persistence_3d_max_min: float = 0.10
    persistence_3d_mean_min: float = 0.05
    persistence_7d_mean_min: float = 0.03

    # Respaldo físico y consenso para la explicación operativa.
    support_rain1_min: float = 20.0
    support_rain3_min: float = 50.0
    support_rain30_min: float = 150.0
    support_api_min: float = 45.0
    support_score_min: int = 2
    support_families_min: int = 2

    # Respaldo aguas arriba de Cerro Hoya.
    # Se añade únicamente a la política SAFE; no altera la probabilidad
    # aprendida por el meta-modelo EXYNUS.
    support_cerro_hoya_3d_min: float = 50.0
    support_cerro_hoya_7d_min: float = 100.0
    # Mientras el histórico CHIRPS no esté validado/recalibrado, Cerro Hoya
    # aporta evidencia y visualización, pero no puede cambiar por sí mismo
    # un veto SAFE ni la decisión final.
    cerro_hoya_decision_enabled: bool = False

    consensus_filter_min: float = 0.10
    consensus_core_min: float = 0.01
    consensus_ratio_min: float = 0.45
    consensus_votes_min: int = 2

    # Excepción crítica exportada en el paquete v5.1.
    critical_prob_min: float = 0.80
    critical_votes_min: int = 5
    critical_p_filter_min: float = 0.60
    critical_score_min: float = 6.0
    critical_rain30_min: float = 400.0
    critical_rain3_min: float = 80.0
    critical_ratio_min: float = 0.45
    critical_p_core_min: float = 0.03
    critical_p_lgbm_min: float = 0.005

    # Veto de soporte bajo exportado en el paquete v5.1.
    low_support_rain30_max: float = 180.0
    low_support_api_max: float = 70.0
    low_support_rain3_max: float = 50.0
    low_support_p_core_max: float = 0.003
    low_support_p_lgbm_max: float = 0.002
    low_support_score_max: float = 2.0
    low_support_saturation_max: float = 0.0
    low_support_ratio_max: float = 0.25
    isolated_p_filter_min: float = 0.45
    isolated_roll3_mean_max: float = 0.22
    isolated_roll7_mean_max: float = 0.18

    # Supresión residual de EXYNUS v5.0-TRS, heredada por v5.1.
    residual_api_min: float = 68.38088
    residual_rain30_min: float = 287.129132
    residual_rain1_max: float = 4.943576
    residual_rain3_max: float = 41.126251
    residual_descending_votes_min: int = 2
    residual_score_max: int = 3
    residual_p_filter_max: float = 0.25

    @classmethod
    def from_packages(
        cls,
        safe_package: Any = None,
        trs_package: Any = None,
    ) -> "SafePolicyConfig":
        """Lee los valores reales de los paquetes serializados."""
        base = cls()
        safe = _as_mapping(safe_package)
        safe_cfg = _as_mapping(safe.get("safe_policy_config"))
        policy = _as_mapping(safe_cfg.get("policy"))

        trs = _as_mapping(trs_package)
        if "config" in trs:
            trs = _as_mapping(trs.get("config"))
        if not trs and safe:
            base_package = _as_mapping(safe.get("base_model_package"))
            trs = _as_mapping(base_package.get("config"))

        selection = _as_mapping(trs.get("selection"))
        residual = _as_mapping(trs.get("residual_config"))
        season = trs.get("season_months", [base.season_start_month, base.season_end_month])
        try:
            season_start, season_end = int(season[0]), int(season[1])
        except Exception:
            season_start, season_end = base.season_start_month, base.season_end_month

        def f(source: Mapping[str, Any], key: str, default: float) -> float:
            try:
                value = float(source.get(key, default))
                return value if np.isfinite(value) else float(default)
            except Exception:
                return float(default)

        def i(source: Mapping[str, Any], key: str, default: int) -> int:
            return int(round(f(source, key, default)))

        def b(source: Mapping[str, Any], key: str, default: bool) -> bool:
            value = source.get(key, default)
            if isinstance(value, str):
                return value.strip().casefold() in {"1", "true", "si", "sí", "yes"}
            return bool(value)

        return cls(
            version=str(policy.get("version", safe.get("version", base.version))),
            decision_threshold=f(
                {"v": safe_cfg.get("base_v5_threshold", selection.get("threshold", base.decision_threshold))},
                "v",
                base.decision_threshold,
            ),
            threshold_epsilon=f(policy, "threshold_epsilon", base.threshold_epsilon),
            season_start_month=i(policy, "season_start_month", season_start),
            season_end_month=i(policy, "season_end_month", season_end),
            critical_prob_min=f(policy, "critical_prob_min", base.critical_prob_min),
            critical_votes_min=i(policy, "critical_votes_min", base.critical_votes_min),
            critical_p_filter_min=f(policy, "critical_p_filter_min", base.critical_p_filter_min),
            critical_score_min=f(policy, "critical_score_min", base.critical_score_min),
            critical_rain30_min=f(policy, "critical_rain30_min", base.critical_rain30_min),
            critical_rain3_min=f(policy, "critical_rain3_min", base.critical_rain3_min),
            critical_ratio_min=f(policy, "critical_ratio_min", base.critical_ratio_min),
            critical_p_core_min=f(policy, "critical_p_core_min", base.critical_p_core_min),
            critical_p_lgbm_min=f(policy, "critical_p_lgbm_min", base.critical_p_lgbm_min),
            low_support_rain30_max=f(policy, "low_support_rain30_max", base.low_support_rain30_max),
            low_support_api_max=f(policy, "low_support_api_max", base.low_support_api_max),
            low_support_rain3_max=f(policy, "low_support_rain3_max", base.low_support_rain3_max),
            low_support_p_core_max=f(policy, "low_support_p_core_max", base.low_support_p_core_max),
            low_support_p_lgbm_max=f(policy, "low_support_p_lgbm_max", base.low_support_p_lgbm_max),
            low_support_score_max=f(policy, "low_support_score_max", base.low_support_score_max),
            low_support_saturation_max=f(
                policy, "low_support_saturation_max", base.low_support_saturation_max
            ),
            low_support_ratio_max=f(policy, "low_support_ratio_max", base.low_support_ratio_max),
            isolated_p_filter_min=f(policy, "isolated_p_filter_min", base.isolated_p_filter_min),
            isolated_roll3_mean_max=f(
                policy, "isolated_roll3_mean_max", base.isolated_roll3_mean_max
            ),
            isolated_roll7_mean_max=f(
                policy, "isolated_roll7_mean_max", base.isolated_roll7_mean_max
            ),
            residual_api_min=f(residual, "api_minimo", base.residual_api_min),
            residual_rain30_min=f(residual, "lluvia_30d_minima", base.residual_rain30_min),
            residual_rain1_max=f(residual, "lluvia_1d_maxima", base.residual_rain1_max),
            residual_rain3_max=f(residual, "lluvia_3d_maxima", base.residual_rain3_max),
            residual_descending_votes_min=i(
                residual, "votos_descenso_minimos", base.residual_descending_votes_min
            ),
            residual_score_max=i(residual, "score_maximo", base.residual_score_max),
            residual_p_filter_max=f(
                residual, "prob_filtro_maxima", base.residual_p_filter_max
            ),
            support_cerro_hoya_3d_min=f(
                policy,
                "support_cerro_hoya_3d_min",
                base.support_cerro_hoya_3d_min,
            ),
            support_cerro_hoya_7d_min=f(
                policy,
                "support_cerro_hoya_7d_min",
                base.support_cerro_hoya_7d_min,
            ),
            cerro_hoya_decision_enabled=b(
                policy,
                "cerro_hoya_decision_enabled",
                base.cerro_hoya_decision_enabled,
            ),
        )


def add_votes_descending(
    frame: pd.DataFrame,
    columns: Iterable[str] | None = None,
    output_column: str = "votos_descenso",
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Cuenta señales que descienden frente al día anterior."""
    result = frame.copy()
    if len(result) == 1 and output_column in result.columns:
        result[output_column] = (
            pd.to_numeric(result[output_column], errors="coerce").fillna(0).astype(int)
        )
        return result

    selected = list(
        columns
        or (
            "p_filtro_old",
            "p_core",
            "lluvia_1d",
            "lluvia_3d",
            "lluvia_30d",
            "api_proxy_lluvia",
            "score",
        )
    )
    votes = pd.Series(0, index=result.index, dtype=int)
    for name in selected:
        if name in result.columns:
            series = pd.to_numeric(result[name], errors="coerce")
            votes += series.diff().lt(-abs(float(tolerance))).fillna(False).astype(int)
    result[output_column] = votes.astype(int)
    return result


def _persistence_flags(
    frame: pd.DataFrame, config: SafePolicyConfig
) -> tuple[pd.Series, pd.Series]:
    p3 = (
        _num(frame, "roll3_max_p_filtro_old").ge(config.persistence_3d_max_min)
        & _num(frame, "roll3_mean_p_filtro_old").ge(config.persistence_3d_mean_min)
    )
    p7 = _num(frame, "roll7_mean_p_filtro_old").ge(config.persistence_7d_mean_min)
    return p3.astype(bool), p7.astype(bool)


def _cerro_hoya_support(
    frame: pd.DataFrame, config: SafePolicyConfig
) -> pd.Series:
    """Señal aguas arriba: lluvia relevante en Cerro Hoya.

    Esta evidencia nunca crea una alerta por sí sola. Refuerza el soporte
    operativo de una señal que ya proviene de EXYNUS. Solo cuando la
    configuración futura ``cerro_hoya_decision_enabled`` sea activada tras
    calibración podrá impedir un veto de hidrología baja o supresión residual.
    """
    available = _bool(frame, "cerro_hoya_disponible")
    rain3 = _num(frame, "lluvia_cerro_hoya_3d")
    rain7 = _num(frame, "lluvia_cerro_hoya_7d")
    return (
        available
        & (
            rain3.ge(config.support_cerro_hoya_3d_min)
            | rain7.ge(config.support_cerro_hoya_7d_min)
        )
    ).astype(bool)


def _physical_votes(frame: pd.DataFrame, config: SafePolicyConfig) -> pd.Series:
    votes = pd.Series(0, index=frame.index, dtype=int)
    votes += _num(frame, "lluvia_1d").ge(config.support_rain1_min).astype(int)
    votes += _num(frame, "lluvia_3d").ge(config.support_rain3_min).astype(int)
    votes += _num(frame, "lluvia_30d").ge(config.support_rain30_min).astype(int)
    votes += _num(frame, "api_proxy_lluvia").ge(config.support_api_min).astype(int)
    votes += _num(frame, "score").ge(config.support_score_min).astype(int)
    votes += _num(frame, "familias").ge(config.support_families_min).astype(int)
    votes += _bool(frame, "marea").astype(int)
    votes += _bool(frame, "tropical").astype(int)
    return votes.astype(int)


def _consensus_votes(frame: pd.DataFrame, config: SafePolicyConfig) -> pd.Series:
    votes = pd.Series(0, index=frame.index, dtype=int)
    votes += _num(frame, "p_filtro_old").ge(config.consensus_filter_min).astype(int)
    votes += _num(frame, "p_core").ge(config.consensus_core_min).astype(int)
    votes += _num(frame, "ratio_media_max_estacion", 1.0).ge(
        config.consensus_ratio_min
    ).astype(int)
    return votes.astype(int)


def residual_mask(frame: pd.DataFrame, config: SafePolicyConfig) -> pd.Series:
    """Detecta humedad residual descendente sin nueva lluvia intensa."""
    return (
        _num(frame, "api_proxy_lluvia").ge(config.residual_api_min)
        & _num(frame, "lluvia_30d").ge(config.residual_rain30_min)
        & _num(frame, "lluvia_1d").le(config.residual_rain1_max)
        & _num(frame, "lluvia_3d").le(config.residual_rain3_max)
        & _num(frame, "votos_descenso").ge(config.residual_descending_votes_min)
        & _num(frame, "score").le(config.residual_score_max)
        & _num(frame, "p_filtro_old").le(config.residual_p_filter_max)
        & ~(
            _bool(frame, "respaldo_cerro_hoya")
            & bool(config.cerro_hoya_decision_enabled)
        )
    ).astype(bool)


def _critical_votes(frame: pd.DataFrame, config: SafePolicyConfig) -> pd.Series:
    votes = pd.Series(0, index=frame.index, dtype=int)
    votes += _num(frame, "prob_exynus").ge(config.critical_prob_min).astype(int)
    votes += _num(frame, "p_filtro_old").ge(config.critical_p_filter_min).astype(int)
    votes += _num(frame, "score").ge(config.critical_score_min).astype(int)
    votes += _num(frame, "lluvia_30d").ge(config.critical_rain30_min).astype(int)
    votes += _num(frame, "lluvia_3d").ge(config.critical_rain3_min).astype(int)
    votes += _num(frame, "ratio_media_max_estacion", 1.0).ge(
        config.critical_ratio_min
    ).astype(int)
    votes += _num(frame, "p_core").ge(config.critical_p_core_min).astype(int)
    votes += _num(frame, "p_lgbm").ge(config.critical_p_lgbm_min).astype(int)
    return votes.astype(int)


def critical_override_mask(frame: pd.DataFrame, config: SafePolicyConfig) -> pd.Series:
    """Confirma condiciones críticas cuando coinciden múltiples señales."""
    votes = (
        _num(frame, "critical_votes").astype(int)
        if "critical_votes" in frame.columns
        else _critical_votes(frame, config)
    )
    return votes.ge(config.critical_votes_min).astype(bool)


def _low_support_components(
    frame: pd.DataFrame, config: SafePolicyConfig
) -> dict[str, pd.Series]:
    return {
        "veto_low_hydrology": (
            _num(frame, "lluvia_30d").le(config.low_support_rain30_max)
            & _num(frame, "api_proxy_lluvia").le(config.low_support_api_max)
            & _num(frame, "lluvia_3d").le(config.low_support_rain3_max)
            & ~(
                _bool(frame, "respaldo_cerro_hoya")
                & bool(config.cerro_hoya_decision_enabled)
            )
        ),
        "veto_low_model_consensus": (
            _num(frame, "p_core").le(config.low_support_p_core_max)
            & _num(frame, "p_lgbm").le(config.low_support_p_lgbm_max)
        ),
        "veto_weak_physics": (
            _num(frame, "score").le(config.low_support_score_max)
            & _num(frame, "saturacion").le(config.low_support_saturation_max)
        ),
        "veto_sensor_disagreement": _num(
            frame, "ratio_media_max_estacion", 1.0
        ).le(config.low_support_ratio_max),
        "veto_isolated_temporal_spike": (
            _num(frame, "p_filtro_old").ge(config.isolated_p_filter_min)
            & _num(frame, "roll3_mean_p_filtro_old").le(
                config.isolated_roll3_mean_max
            )
            & _num(frame, "roll7_mean_p_filtro_old").le(
                config.isolated_roll7_mean_max
            )
        ),
    }


def low_support_veto_mask(frame: pd.DataFrame, config: SafePolicyConfig) -> pd.Series:
    """Veta una señal base solo cuando convergen evidencias de soporte bajo."""
    base = _bool(frame, "pred_exynus_base")
    p3 = _bool(frame, "persistencia_3d")
    p7 = _bool(frame, "persistencia_7d")
    components = _low_support_components(frame, config)

    low_hydrology = components["veto_low_hydrology"]
    low_model = components["veto_low_model_consensus"]
    weak_physics = components["veto_weak_physics"]
    sensor_disagreement = components["veto_sensor_disagreement"]
    isolated = components["veto_isolated_temporal_spike"]

    # Un único indicio no basta para vetar. Esto conserva eventos con sensores
    # incompletos, pero bloquea picos secos, aislados o sin consenso.
    combined = (
        (low_hydrology & low_model & weak_physics)
        | (isolated & low_hydrology)
        | (sensor_disagreement & low_hydrology & weak_physics & ~p7)
    )
    return (base & combined).astype(bool)


def _reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if bool(row.get("pred_exynus_base", False)):
        reasons.append("meta_modelo_supera_umbral")
    if bool(row.get("persistencia_3d", False)):
        reasons.append("persistencia_temporal_3d")
    if bool(row.get("persistencia_7d", False)):
        reasons.append("persistencia_temporal_7d")
    if bool(row.get("respaldo_fisico", False)):
        reasons.append("respaldo_fisico")
    if bool(row.get("respaldo_cerro_hoya", False)):
        reasons.append("respaldo_aguas_arriba_cerro_hoya")
    if bool(row.get("consenso_senales", False)):
        reasons.append("consenso_de_senales")
    if bool(row.get("low_support_veto", False)):
        reasons.append("veto_soporte_insuficiente")
    if bool(row.get("residual_suppression", False)):
        reasons.append("supresion_humedad_residual_descendente")
    if bool(row.get("critical_override", False)):
        reasons.append("excepcion_critica_multisenal")
    if bool(row.get("pred_exynus_safe", False)):
        reasons.append("alerta_safe_confirmada")
    if not reasons:
        reasons.append("sin_senal_exynus_relevante")
    return reasons


def apply_safe_policy(
    frame: pd.DataFrame,
    config: SafePolicyConfig | None = None,
) -> pd.DataFrame:
    """Aplica la política completa y añade diagnósticos SAFE."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("apply_safe_policy espera un pandas.DataFrame.")
    if frame.empty:
        raise ValueError("apply_safe_policy no puede operar sobre una tabla vacía.")

    cfg = config or SafePolicyConfig()
    result = add_votes_descending(frame)

    result["pred_exynus_base"] = _num(result, "prob_exynus").ge(
        cfg.decision_threshold - cfg.threshold_epsilon
    )
    result["core_signal"] = result["pred_exynus_base"]

    month = _num(result, "mes", 1).astype(int)
    result["season_ok"] = month.between(
        cfg.season_start_month, cfg.season_end_month, inclusive="both"
    )

    p3, p7 = _persistence_flags(result, cfg)
    result["persistencia_3d"] = p3
    result["persistencia_7d"] = p7

    result["respaldo_cerro_hoya"] = _cerro_hoya_support(result, cfg)

    result["physical_votes"] = _physical_votes(result, cfg)
    result["consensus_votes"] = _consensus_votes(result, cfg)
    result["respaldo_fisico"] = result["physical_votes"].gt(0)
    result["consenso_senales"] = result["consensus_votes"].ge(
        cfg.consensus_votes_min
    )

    components = _low_support_components(result, cfg)
    for name, values in components.items():
        result[name] = values.astype(bool)

    result["critical_votes"] = _critical_votes(result, cfg)
    result["critical_override"] = critical_override_mask(result, cfg)
    result["low_support_veto"] = low_support_veto_mask(result, cfg)
    result["residual_suppression"] = residual_mask(result, cfg)

    # La señal normal se limita a la temporada. La excepción crítica puede
    # confirmar dentro o fuera de temporada.
    result["pred_exynus_safe"] = (
        result["pred_exynus_base"]
        & result["season_ok"]
        & ~result["low_support_veto"]
        & ~result["residual_suppression"]
    ) | result["critical_override"]

    result["soporte_alerta"] = (
        result["physical_votes"].clip(0, MAX_PHYSICAL_VOTES)
        + result["consensus_votes"].clip(0, MAX_CONSENSUS_VOTES)
        + result["persistencia_3d"].astype(int)
        + result["persistencia_7d"].astype(int)
    ).astype(int)

    result["confianza_exynus"] = np.select(
        [
            result["critical_override"],
            result["pred_exynus_safe"] & result["soporte_alerta"].ge(6),
            result["pred_exynus_safe"] & result["soporte_alerta"].ge(3),
            result["pred_exynus_base"],
        ],
        ["muy_alta", "alta", "media", "baja"],
        default="sin_alerta",
    )

    result["estado_exynus"] = np.select(
        [
            result["critical_override"],
            result["pred_exynus_safe"],
            result["residual_suppression"],
            result["low_support_veto"],
            result["pred_exynus_base"] & ~result["season_ok"],
            result["pred_exynus_base"],
        ],
        [
            "alerta_critica_confirmada",
            "alerta_confirmada",
            "recuperacion_residual",
            "observacion_bajo_consenso",
            "observacion_fuera_temporada",
            "observacion",
        ],
        default="normal",
    )
    result["razones_exynus"] = result.apply(_reasons, axis=1)
    return result


__all__ = [
    "SafePolicyConfig",
    "MAX_SUPPORT_SCORE",
    "add_votes_descending",
    "residual_mask",
    "critical_override_mask",
    "low_support_veto_mask",
    "apply_safe_policy",
]
