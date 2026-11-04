"""Physikalische Referenzskalierung für die dimensionslosen Euler-Gleichungen.

Die astronomix-Zustände werden weiterhin in Codeeinheiten gespeichert. Dieses
Modul beschreibt nur, welche physikalischen Größen den Einheiten
entsprechen. Dadurch können Medium und Supernova konsistent auf ein
astronomisches Szenario zurückgeführt werden.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


PARSEC_CM = 3.0856775814913673e18
HYDROGEN_MASS_G = 1.6735575e-24
BOLTZMANN_ERG_PER_K = 1.380649e-16
SECONDS_PER_MYR = 1.0e6 * 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class PhysicalScaling:
    """Referenzgrößen für ein warmes interstellares Medium.

    ``mean_molecular_weight`` ist die mittlere Teilchenmasse in Einheiten der
    Wasserstoffmasse. ``mass_per_hydrogen_nucleus`` berücksichtigt zusätzlich
    die Masse von Helium und schweren Elementen pro Wasserstoffkern.
    """

    box_size_pc: float
    hydrogen_number_density_cm3: float
    temperature_K: float
    gamma: float = 5.0 / 3.0
    mean_molecular_weight: float = 1.27
    mass_per_hydrogen_nucleus: float = 1.4

    def __post_init__(self) -> None:
        if self.box_size_pc <= 0.0:
            raise ValueError("box_size_pc muss positiv sein.")
        if self.hydrogen_number_density_cm3 <= 0.0:
            raise ValueError("Die Wasserstoffdichte muss positiv sein.")
        if self.temperature_K <= 0.0:
            raise ValueError("Die Temperatur muss positiv sein.")
        if self.gamma <= 1.0:
            raise ValueError("gamma muss größer als 1 sein.")
        if self.mean_molecular_weight <= 0.0:
            raise ValueError("Die mittlere Molekülmasse muss positiv sein.")
        if self.mass_per_hydrogen_nucleus <= 0.0:
            raise ValueError("Die Masse pro Wasserstoffkern muss positiv sein.")

    @property
    def length_unit_cm(self) -> float:
        return self.box_size_pc * PARSEC_CM

    @property
    def density_unit_g_per_cm3(self) -> float:
        return (
            self.mass_per_hydrogen_nucleus
            * HYDROGEN_MASS_G
            * self.hydrogen_number_density_cm3
        )

    @property
    def pressure_unit_dyn_per_cm2(self) -> float:
        particle_number_density = self.density_unit_g_per_cm3 / (
            self.mean_molecular_weight * HYDROGEN_MASS_G
        )
        return particle_number_density * BOLTZMANN_ERG_PER_K * self.temperature_K

    @property
    def velocity_unit_cm_per_s(self) -> float:
        return math.sqrt(
            self.gamma
            * self.pressure_unit_dyn_per_cm2
            / self.density_unit_g_per_cm3
        )

    @property
    def time_unit_myr(self) -> float:
        return (
            self.length_unit_cm / self.velocity_unit_cm_per_s
        ) / SECONDS_PER_MYR

    @property
    def energy_unit_erg(self) -> float:
        return self.pressure_unit_dyn_per_cm2 * self.length_unit_cm**3

    def explosion_energy_code(self, energy_erg: float) -> float:
        if energy_erg <= 0.0:
            raise ValueError("Die physikalische Explosionsenergie muss positiv sein.")
        return energy_erg / self.energy_unit_erg

    def to_metadata(self) -> dict[str, float | str]:
        return {
            "unit_system": "dimensionless_euler_with_physical_reference_scaling",
            "box_size_pc": float(self.box_size_pc),
            "hydrogen_number_density_cm3": float(
                self.hydrogen_number_density_cm3
            ),
            "temperature_K": float(self.temperature_K),
            "gamma": float(self.gamma),
            "mean_molecular_weight": float(self.mean_molecular_weight),
            "mass_per_hydrogen_nucleus": float(self.mass_per_hydrogen_nucleus),
            "length_unit_cm": float(self.length_unit_cm),
            "density_unit_g_per_cm3": float(self.density_unit_g_per_cm3),
            "density_unit_kg_per_m3": float(self.density_unit_g_per_cm3 * 1000.0),
            "pressure_unit_dyn_per_cm2": float(self.pressure_unit_dyn_per_cm2),
            "pressure_unit_Pa": float(self.pressure_unit_dyn_per_cm2 * 0.1),
            "velocity_unit_km_per_s": float(self.velocity_unit_cm_per_s / 1.0e5),
            "time_unit_Myr": float(self.time_unit_myr),
            "energy_unit_erg": float(self.energy_unit_erg),
            "energy_unit_J": float(self.energy_unit_erg * 1.0e-7),
        }
