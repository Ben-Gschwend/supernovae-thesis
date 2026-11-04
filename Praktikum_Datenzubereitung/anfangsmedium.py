"""Erzeuge ein turbulentes Anfangsmedium für die spätere Supernova.

Dieses Skript bildet nur die erste Phase des geplanten Zweiphasen-Experiments
ab:

1. homogenes Medium initialisieren,
2. das Medium mit einer zeitlich korrelierten
   Ornstein-Uhlenbeck-Forcierung entwickeln,
3. den finalen turbulenten Zustand für die spätere Supernova speichern.

Die Supernova-Energie wird hier ausdrücklich noch nicht eingebracht. Das
gespeicherte ``primitive_state`` soll später als Ausgangszustand einer
zweiten Simulation dienen.

Die Defaultwerte sind vorläufige Pilotparameter und noch keine mit der
Betreuung abgestimmte wissenschaftliche Referenzkonfiguration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from astronomix import (
    CARTESIAN,
    FINITE_VOLUME,
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)
from astronomix._modules._cooling.cooling_options import (
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
    EXPLICIT_COOLING,
    SIMPLE_POWER_LAW,
    SimplePowerLawParams,
)
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    PERIODIC_BOUNDARY,
    SnapshotSettings,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Erzeuge ein turbulentes 3D-Anfangsmedium mit OU-Forcierung "
            "und speichere den finalen Zustand."
        )
    )
    parser.add_argument("--num-cells", type=int, default=128)
    parser.add_argument("--box-size", type=float, default=1.0)
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--num-snapshots", type=int, default=16)
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help=(
            "Alle vollständigen Zwischenzustände speichern, damit daraus "
            "ein zeitliches GIF erzeugt werden kann. Dies benötigt deutlich "
            "mehr Speicherplatz."
        ),
    )
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        help=(
            "Homogener Anfangsdruck des turbulenten Mediums. "
            "Vorläufiger Pilotwert; nicht mit dem bisherigen Sedov-p_amb "
            "von 1e-4 verwechseln."
        ),
    )
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--correlation-time",
        type=float,
        default=1.0,
        help="Zeitliche OU-Korrelation; ungefähr eine Eddy-Turnover-Zeit.",
    )
    parser.add_argument(
        "--forcing-wavenumber",
        type=float,
        default=4.0,
        help="Wellenzahl der OU-Forcierung in physikalischen Einheiten.",
    )
    parser.add_argument(
        "--forcing-amplitude",
        type=float,
        default=1.0,
        help="Beschleunigungsamplitude der OU-Forcierung.",
    )
    parser.add_argument(
        "--cooling",
        action="store_true",
        help=(
            "Aktiviere den expliziten dimensionslosen Power-Law-Cooling-Term. "
            "Ohne diese Option bleibt das Medium adiabatisch."
        ),
    )
    parser.add_argument(
        "--cooling-factor",
        type=float,
        default=1.0,
        help="Faktor Lambda_0 der dimensionslosen Cooling-Kurve.",
    )
    parser.add_argument(
        "--cooling-exponent",
        type=float,
        default=1.0,
        help="Exponent der dimensionslosen Cooling-Kurve.",
    )
    parser.add_argument(
        "--cooling-reference-temperature",
        type=float,
        default=1.0,
        help="Referenztemperatur T_ref in den Codeeinheiten.",
    )
    parser.add_argument(
        "--cooling-floor-temperature",
        type=float,
        default=1e-6,
        help="Untere Temperaturgrenze in den Codeeinheiten.",
    )
    parser.add_argument(
        "--protection-density-threshold",
        type=float,
        default=0.02,
        help="Dichteschwelle der astronomix-Vakuumsicherung.",
    )
    parser.add_argument(
        "--protection-max-velocity",
        type=float,
        default=50.0,
        help="Maximale Geschwindigkeit der astronomix-Vakuumsicherung.",
    )
    parser.add_argument(
        "--boundary",
        choices=("periodic", "open"),
        default="periodic",
        help="Randbedingung; periodisch ist der vorläufige Standard für Turbulenz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "Praktikum_Datenzubereitung/turbulent_medium_128cells.npz"
        ),
        help="Zieldatei für den finalen turbulenten Zustand.",
    )
    return parser.parse_args()


def _periodic_boundaries() -> BoundarySettings:
    periodic = BoundarySettings1D(
        left_boundary=PERIODIC_BOUNDARY,
        right_boundary=PERIODIC_BOUNDARY,
    )
    return BoundarySettings(x=periodic, y=periodic, z=periodic)


def _as_int_tuple(values) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(values).reshape(-1))


def _as_float_tuple(values) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values).reshape(-1))


def _next_available_path(output_path: Path) -> Path:
    """Verhindere das Überschreiben eines vorhandenen Mediums."""

    if not output_path.exists():
        return output_path
    counter = 1
    while True:
        candidate = output_path.with_name(
            f"{output_path.stem}_{counter}{output_path.suffix}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def build_simulation(args: argparse.Namespace):
    """Baue Konfiguration und homogenen Anfangszustand ohne Integration."""

    if args.num_cells < 8:
        raise ValueError("--num-cells muss mindestens 8 sein.")
    if args.box_size <= 0.0:
        raise ValueError("--box-size muss positiv sein.")
    if args.t_end <= 0.0:
        raise ValueError("--t-end muss positiv sein.")
    if args.num_snapshots < 2:
        raise ValueError("--num-snapshots muss mindestens 2 sein.")
    if args.density <= 0.0 or args.pressure <= 0.0:
        raise ValueError("Dichte und Druck müssen positiv sein.")
    if args.correlation_time <= 0.0:
        raise ValueError("--correlation-time muss positiv sein.")
    if args.forcing_wavenumber <= 0.0 or args.forcing_amplitude <= 0.0:
        raise ValueError("Forcierungswellenzahl und -amplitude müssen positiv sein.")
    if args.cooling_factor <= 0.0:
        raise ValueError("--cooling-factor muss positiv sein.")
    if args.cooling_reference_temperature <= 0.0:
        raise ValueError("--cooling-reference-temperature muss positiv sein.")
    if args.cooling_floor_temperature <= 0.0:
        raise ValueError("--cooling-floor-temperature muss positiv sein.")

    forcing_config = TurbulentForcingConfig(
        turbulent_forcing=True,
        ou_forcing=True,
        # Die Forcierung kann in einzelnen Zellen sehr kleine Dichten erzeugen.
        # Die astronomix-Sicherung ist deshalb standardmäßig aktiviert.
        vacuum_protection=True,
    )
    forcing_params = TurbulentForcingParams(
        protection_density_threshold=args.protection_density_threshold,
        protection_max_velocity=args.protection_max_velocity,
        # Bei OU-Forcierung ist forcing_amplitude maßgeblich; die
        # energy_injection_rate wird von diesem Zweig nicht verwendet.
        energy_injection_rate=2.0,
        correlation_time=args.correlation_time,
        forcing_wavenumber=args.forcing_wavenumber,
        forcing_amplitude=args.forcing_amplitude,
    )
    cooling_params = CoolingParams(
        floor_temperature=args.cooling_floor_temperature,
        cooling_curve_params=SimplePowerLawParams(
            factor=args.cooling_factor,
            exponent=args.cooling_exponent,
            reference_temperature=args.cooling_reference_temperature,
        ),
    )
    cooling_config = CoolingConfig(
        cooling=args.cooling,
        cooling_method=EXPLICIT_COOLING,
        cooling_curve_config=CoolingCurveConfig(
            cooling_curve_type=SIMPLE_POWER_LAW,
        ),
    )

    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=3,
        box_size=args.box_size,
        num_cells=args.num_cells,
        mhd=False,
        solver_mode=FINITE_VOLUME,
        progress_bar=True,
        random_seed=args.seed,
        boundary_settings=(
            _periodic_boundaries() if args.boundary == "periodic" else None
        ),
        turbulent_forcing_config=forcing_config,
        cooling_config=cooling_config,
        return_snapshots=True,
        num_snapshots=args.num_snapshots,
        snapshot_settings=SnapshotSettings(
            # Vollständige Zwischenzustände werden nur auf ausdrücklichen
            # Wunsch gespeichert, da sie bei 128^3 bereits sehr groß sind.
            return_states=args.save_snapshots,
            return_final_state=True,
            return_total_mass=True,
            return_total_energy=True,
            return_internal_energy=True,
            return_kinetic_energy=True,
        ),
    )
    params = SimulationParams(
        t_start=0.0,
        t_end=args.t_end,
        C_cfl=args.cfl,
        gamma=args.gamma,
        turbulent_forcing_params=forcing_params,
        cooling_params=cooling_params,
    )

    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    shape = helper_data.geometric_centers.shape[:-1]
    density = args.density * jnp.ones(shape)
    pressure = args.pressure * jnp.ones(shape)
    zero_velocity = jnp.zeros(shape)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zero_velocity,
        velocity_y=zero_velocity,
        velocity_z=zero_velocity,
        gas_pressure=pressure,
    )
    config = finalize_config(config, initial_state.shape)
    return config, params, registered_variables, initial_state


def save_turbulent_medium(
    output_path: Path,
    result,
    config,
    params,
    registered_variables,
    args: argparse.Namespace,
) -> None:
    """Speichere Endzustand, Diagnosen und reproduzierbare Metadaten."""

    if result.final_state is None:
        raise RuntimeError(
            "astronomix hat keinen final_state zurückgegeben. "
            "SnapshotSettings.return_final_state prüfen."
        )

    primitive_state = np.asarray(result.final_state)
    metadata = {
        "purpose": "turbulent_initial_medium_before_supernova",
        "forcing": "ornstein_uhlenbeck",
        "num_cells": _as_int_tuple(config.num_cells),
        "box_size": _as_float_tuple(config.box_size),
        "gamma": float(params.gamma),
        "t_start": float(params.t_start),
        "t_end": float(params.t_end),
        "cfl": float(params.C_cfl),
        "density_initial": float(args.density),
        "pressure_initial": float(args.pressure),
        "random_seed": int(args.seed),
        "correlation_time": float(args.correlation_time),
        "forcing_wavenumber": float(args.forcing_wavenumber),
        "forcing_amplitude": float(args.forcing_amplitude),
        "cooling": bool(args.cooling),
        "cooling_model": "simple_power_law",
        "cooling_method": "explicit",
        "cooling_factor": float(args.cooling_factor),
        "cooling_exponent": float(args.cooling_exponent),
        "cooling_reference_temperature": float(
            args.cooling_reference_temperature
        ),
        "cooling_floor_temperature": float(args.cooling_floor_temperature),
        "boundary": args.boundary,
        "vacuum_protection": True,
        "snapshots_saved": bool(args.save_snapshots),
        "state_layout": {
            "density_index": int(registered_variables.density_index),
            "pressure_index": int(registered_variables.pressure_index),
            "velocity_x_index": int(registered_variables.velocity_index.x),
            "velocity_y_index": int(registered_variables.velocity_index.y),
            "velocity_z_index": int(registered_variables.velocity_index.z),
        },
    }

    output_path = _next_available_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "primitive_state": primitive_state,
        "final_time": np.asarray(float(params.t_end)),
        "time_points": np.asarray(result.time_points),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for field in ("total_mass", "total_energy", "internal_energy", "kinetic_energy"):
        value = getattr(result, field, None)
        if value is not None:
            arrays[field] = np.asarray(value)

    states = getattr(result, "states", None)
    if states is not None:
        arrays["states"] = np.asarray(states)

    np.savez_compressed(output_path, **arrays)
    print(f"Gespeichertes turbulentes Anfangsmedium: {output_path}")
    print(f"Zustandsform: {primitive_state.shape}")
    print(f"Metadaten: {json.dumps(metadata, sort_keys=True)}")


def main() -> None:
    args = _parse_args()
    config, params, registered_variables, initial_state = build_simulation(args)

    print("Starte ausschließlich die Erzeugung des turbulenten Anfangsmediums.")
    print("Die Supernova-Energie wird in diesem Skript nicht eingebracht.")
    result = time_integration(
        initial_state,
        config,
        params,
        registered_variables,
    )
    save_turbulent_medium(
        args.output,
        result,
        config,
        params,
        registered_variables,
        args,
    )


if __name__ == "__main__":
    main()
