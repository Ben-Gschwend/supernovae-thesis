"""Physikalisch motivierte Cooling-Kurven für astronomix.

Die Schure-2009-Tabelle ist in astronomix bereits enthalten. Dieses Modul
verbindet sie mit unserer physikalischen Referenzskalierung, sodass Temperatur
und Cooling-Koeffizient konsistent in Codeeinheiten vorliegen.
"""

from __future__ import annotations

from astronomix._modules._cooling._cooling_tables import schure_cooling
from astronomix._modules._cooling.cooling_options import (
    EXPLICIT_COOLING,
    PIECEWISE_POWER_LAW,
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
)
from astronomix.units.unit_helpers import CodeUnits

from .physikalische_einheiten import PhysicalScaling


def build_schure2009_cie_cooling(
    scaling: PhysicalScaling,
    floor_temperature_K: float = 1.0e4,
):
    """Erzeuge die tabellierte Schure-2009-CIE-Cooling-Konfiguration.

    Die Tabelle basiert auf Schure et al. (2009),
    https://arxiv.org/abs/0909.5204. Der astronomix-Pfad für diese
    piecewise-power-law-Tabelle wird explizit integriert, weil die im Paket
    dokumentierte Townsend-Integration für diese Kurve derzeit nicht als
    zuverlässig markiert ist.
    """

    if floor_temperature_K <= 0.0:
        raise ValueError("Die Cooling-Floor-Temperatur muss positiv sein.")
    if floor_temperature_K < 1.0e4:
        raise ValueError(
            "Die Schure-2009-CIE-Kurve wird hier erst ab 1e4 K verwendet; "
            "setze die Cooling-Floor-Temperatur mindestens auf 1e4 K."
        )
    if abs(scaling.mean_molecular_weight - 0.61) > 0.06:
        raise ValueError(
            "Die Schure-2009-CIE-Kurve benötigt eine ionisierte "
            "Referenzskalierung mit mean_molecular_weight ungefähr 0.61."
        )

    code_mass_g = (
        scaling.density_unit_g_per_cm3 * scaling.length_unit_cm**3
    )
    code_units = CodeUnits.init_from_unit_params(
        scaling.length_unit_cm,
        code_mass_g,
        scaling.velocity_unit_cm_per_s,
    )
    curve_params = schure_cooling(code_units)
    floor_temperature_code = scaling.temperature_code_from_kelvin(
        floor_temperature_K
    )
    cooling_config = CoolingConfig(
        cooling=True,
        cooling_method=EXPLICIT_COOLING,
        cooling_curve_config=CoolingCurveConfig(
            cooling_curve_type=PIECEWISE_POWER_LAW,
        ),
    )
    cooling_params = CoolingParams(
        floor_temperature=floor_temperature_code,
        cooling_curve_params=curve_params,
    )
    return cooling_config, cooling_params
