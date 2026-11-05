"""Simuliere eine Supernova im turbulenten Medium bei ``t=5``.

Das Skript ist die zweite Phase des Zweiphasen-Experiments:

1. den vollständigen turbulenten 3D-Zustand aus Phase 1 laden,
2. bei der absoluten Zeit ``t=5`` eine thermische Explosionsenergie im
   Zentrum einbringen,
3. die Hydrodynamik für ``Delta t=0.1`` ohne weitere OU-Forcierung integrieren,
4. zentrale z-Schnitte mit absoluten Zeitpunkten ``t=5...5.1`` speichern.

Die z-Schnitte dienen ausschließlich der Darstellung. Die Simulation selbst
verwendet weiterhin den vollständigen 3D-Zustand einschließlich Dichte,
Druck und aller drei Geschwindigkeitskomponenten.
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
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
)
from astronomix._modules._cooling.cooling_options import (
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
    EXPLICIT_COOLING,
    IMPLICIT_COOLING,
    SIMPLE_POWER_LAW,
    SimplePowerLawParams,
)
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    PERIODIC_BOUNDARY,
    SnapshotSettings,
)
from .physikalische_einheiten import PhysicalScaling
from .physikalische_cooling import build_schure2009_cie_cooling


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simuliere eine Supernova ab dem turbulenten Zustand bei t=5 "
            "bis zur absoluten Zeit t=5.1."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "Praktikum_Datenzubereitung/turbulent_medium_128cells_t5_kf25_gpu2.npz"
        ),
        help="NPZ-Datei mit dem finalen turbulenten 3D-Zustand.",
    )
    parser.add_argument(
        "--medium-time",
        type=float,
        default=None,
        help=(
            "Absolute Zeit des geladenen Mediums. Standardmäßig wird "
            "metadata.t_end verwendet, normalerweise t=5."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.1,
        help="Ausbreitungsdauer der Supernova; Standard: Delta t=0.1.",
    )
    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=60,
        help=(
            "Anzahl vollständiger 3D-Snapshots in Phase 2. "
            "60 entspricht der zeitlichen Auflösung des bisherigen "
            "supernova_3d_slices_128cells.gif."
        ),
    )
    energy_group = parser.add_mutually_exclusive_group()
    energy_group.add_argument(
        "--explosion-energy",
        type=float,
        default=None,
        help="Zusätzliche Explosionsenergie in Codeeinheiten; Standard: 1.",
    )
    energy_group.add_argument(
        "--physical-explosion-energy-erg",
        type=float,
        default=None,
        help=(
            "Zusätzliche Explosionsenergie in erg. Die zugehörige "
            "Codeenergie wird aus physical_scaling im Medium berechnet."
        ),
    )
    parser.add_argument(
        "--explosion-radius",
        type=float,
        default=0.01,
        help="Radius der zentralen zusätzlichen Explosionsregion.",
    )
    parser.add_argument(
        "--smoothing-cells",
        type=float,
        default=2.0,
        help="Glättungsbreite der Explosionsregion in Gitterzellen.",
    )
    parser.add_argument("--cfl", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument(
        "--boundary",
        choices=("metadata", "periodic", "open"),
        default="metadata",
        help="Randbedingung; metadata übernimmt die Medium-Konfiguration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "Praktikum_Datenzubereitung/supernovae_turbolenz_128cells.npz"
        ),
        help="Ausgabedatei für die zentralen Supernova-Schnittdaten.",
    )
    return parser.parse_args()


def _periodic_boundaries() -> BoundarySettings:
    periodic = BoundarySettings1D(
        left_boundary=PERIODIC_BOUNDARY,
        right_boundary=PERIODIC_BOUNDARY,
    )
    return BoundarySettings(x=periodic, y=periodic, z=periodic)


def _cooling_from_metadata(metadata: dict):
    """Erzeuge die Phase-2-Cooling-Konfiguration aus dem Medium-Metadatum."""

    if not bool(metadata.get("cooling", False)):
        return CoolingConfig(), CoolingParams(), False

    model = metadata.get("cooling_model", "simple_power_law")
    if model == "schure2009_cie":
        physical_scaling = _physical_scaling_from_metadata(metadata)
        if physical_scaling is None:
            raise ValueError(
                "Das Medium verwendet schure2009_cie, enthält aber keine "
                "physical_scaling-Metadaten."
            )
        if metadata.get("cooling_method", "explicit") != "explicit":
            raise ValueError(
                "schure2009_cie benötigt in Phase 2 die explizite "
                "Cooling-Integration."
            )
        cooling_config, cooling_params = build_schure2009_cie_cooling(
            physical_scaling,
            floor_temperature_K=float(
                metadata.get("cooling_floor_temperature_k", 1.0e4)
            ),
        )
        return cooling_config, cooling_params, True
    if model != "simple_power_law":
        raise ValueError(
            "Das Medium verwendet ein nicht unterstütztes Cooling-Modell: "
            f"{model!r}."
        )

    method_name = metadata.get("cooling_method", "explicit")
    if method_name == "implicit":
        method = IMPLICIT_COOLING
    elif method_name == "explicit":
        method = EXPLICIT_COOLING
    else:
        raise ValueError(
            "Unbekannte Cooling-Methode in den Medium-Metadaten: "
            f"{method_name!r}."
        )

    cooling_config = CoolingConfig(
        cooling=True,
        cooling_method=method,
        cooling_curve_config=CoolingCurveConfig(
            cooling_curve_type=SIMPLE_POWER_LAW,
        ),
    )
    cooling_params = CoolingParams(
        floor_temperature=float(
            metadata.get("cooling_floor_temperature", 1e-6)
        ),
        cooling_curve_params=SimplePowerLawParams(
            factor=float(metadata.get("cooling_factor", 1.0)),
            exponent=float(metadata.get("cooling_exponent", 1.0)),
            reference_temperature=float(
                metadata.get("cooling_reference_temperature", 1.0)
            ),
        ),
    )
    return cooling_config, cooling_params, True


def _next_available_path(output_path: Path) -> Path:
    """Schütze vorhandene Ergebnisdateien vor dem Überschreiben."""

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


def _load_medium(input_path: Path) -> tuple[np.ndarray, dict]:
    with np.load(input_path, allow_pickle=False) as data:
        required = {"primitive_state", "metadata_json"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Fehlende NPZ-Felder: {missing}")
        primitive_state = np.asarray(data["primitive_state"])
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    return primitive_state, metadata


def _physical_scaling_from_metadata(metadata: dict) -> PhysicalScaling | None:
    values = metadata.get("physical_scaling")
    if values is None:
        return None
    return PhysicalScaling(
        box_size_pc=float(values["box_size_pc"]),
        hydrogen_number_density_cm3=float(
            values["hydrogen_number_density_cm3"]
        ),
        temperature_K=float(values["temperature_K"]),
        gamma=float(values["gamma"]),
        mean_molecular_weight=float(values["mean_molecular_weight"]),
        mass_per_hydrogen_nucleus=float(values["mass_per_hydrogen_nucleus"]),
    )


def _build_phase_two(
    primitive_state: np.ndarray,
    metadata: dict,
    args: argparse.Namespace,
):
    if primitive_state.ndim != 4 or primitive_state.shape[0] < 5:
        raise ValueError(
            "primitive_state muss die Form (Variablen, Nx, Ny, Nz) besitzen "
            "und Dichte, drei Geschwindigkeiten sowie Druck enthalten."
        )
    if args.duration <= 0.0:
        raise ValueError("--duration muss positiv sein.")
    if args.num_snapshots < 2:
        raise ValueError("--num-snapshots muss mindestens 2 sein.")
    if (
        args.explosion_energy is not None
        and args.explosion_energy <= 0.0
    ):
        raise ValueError("--explosion-energy muss positiv sein.")
    if (
        args.physical_explosion_energy_erg is not None
        and args.physical_explosion_energy_erg <= 0.0
    ):
        raise ValueError("Die physikalische Explosionsenergie muss positiv sein.")
    if args.explosion_radius <= 0.0:
        raise ValueError("--explosion-radius muss positiv sein.")
    if args.smoothing_cells <= 0.0:
        raise ValueError("--smoothing-cells muss positiv sein.")

    box_size = float(np.asarray(metadata["box_size"]).reshape(-1)[0])
    nx, ny, nz = primitive_state.shape[1:]
    if not (nx == ny == nz):
        raise ValueError("Das Skript erwartet ein kubisches 3D-Gitter.")
    num_cells = nx
    gamma = float(args.gamma if args.gamma is not None else metadata["gamma"])
    physical_scaling = _physical_scaling_from_metadata(metadata)
    if args.physical_explosion_energy_erg is not None:
        if physical_scaling is None:
            raise ValueError(
                "Für --physical-explosion-energy-erg benötigt das Medium "
                "physical_scaling-Metadaten."
            )
        explosion_energy = physical_scaling.explosion_energy_code(
            args.physical_explosion_energy_erg
        )
    else:
        explosion_energy = float(
            args.explosion_energy if args.explosion_energy is not None else 1.0
        )

    medium_time = float(
        args.medium_time
        if args.medium_time is not None
        else metadata.get("t_end", 5.0)
    )

    if args.boundary == "open" or (
        args.boundary == "metadata" and metadata.get("boundary") == "open"
    ):
        boundary_settings = None
    else:
        boundary_settings = _periodic_boundaries()

    cooling_config, cooling_params, cooling_enabled = _cooling_from_metadata(
        metadata
    )

    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=3,
        box_size=box_size,
        num_cells=num_cells,
        mhd=False,
        solver_mode=FINITE_VOLUME,
        progress_bar=True,
        random_seed=int(metadata.get("random_seed", 42)),
        boundary_settings=boundary_settings,
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=False,
            ou_forcing=False,
            vacuum_protection=False,
        ),
        cooling_config=cooling_config,
        return_snapshots=True,
        num_snapshots=args.num_snapshots,
        snapshot_settings=SnapshotSettings(
            return_states=True,
            return_final_state=True,
            return_total_mass=True,
            return_total_energy=True,
            return_internal_energy=True,
            return_kinetic_energy=True,
        ),
    )

    registered_variables = get_registered_variables(config)
    saved_layout = metadata.get("state_layout", {})
    expected_layout = {
        "density_index": int(registered_variables.density_index),
        "pressure_index": int(registered_variables.pressure_index),
        "velocity_x_index": int(registered_variables.velocity_index.x),
        "velocity_y_index": int(registered_variables.velocity_index.y),
        "velocity_z_index": int(registered_variables.velocity_index.z),
    }
    if expected_layout != {
        key: int(value) for key, value in saved_layout.items()
    }:
        raise ValueError(
            "Die Variablenreihenfolge der Medium-NPZ-Datei passt nicht zur "
            f"aktuellen astronomix-Konfiguration: gespeichert={saved_layout}, "
            f"aktuell={expected_layout}."
        )

    helper_data = get_helper_data(config)
    coordinates = helper_data.geometric_centers
    center = jnp.array([box_size / 2.0] * 3)
    radius = jnp.linalg.norm(coordinates - center, axis=-1)
    smoothing_width = args.smoothing_cells * box_size / num_cells
    window = 0.5 * (
        1.0 - jnp.tanh((radius - args.explosion_radius) / smoothing_width)
    )
    cell_volume = (box_size / num_cells) ** 3
    delta_pressure = explosion_energy * (gamma - 1.0) / (
        cell_volume * jnp.sum(window)
    )

    state = jnp.asarray(primitive_state)
    density_index = registered_variables.density_index
    pressure_index = registered_variables.pressure_index
    if float(jnp.min(state[density_index])) <= 0.0:
        raise ValueError("Das turbulente Medium enthält nichtpositive Dichten.")
    if float(jnp.min(state[pressure_index])) <= 0.0:
        raise ValueError("Das turbulente Medium enthält nichtpositive Drücke.")

    state_with_explosion = state.at[pressure_index].add(delta_pressure * window)
    injected_energy = float(
        jnp.sum(delta_pressure * window / (gamma - 1.0)) * cell_volume
    )
    config = finalize_config(config, state_with_explosion.shape)
    cfl = float(args.cfl if args.cfl is not None else metadata.get("cfl", 0.4))
    params = SimulationParams(
        t_start=0.0,
        t_end=args.duration,
        C_cfl=cfl,
        gamma=gamma,
        cooling_params=cooling_params,
    )
    return (
        config,
        params,
        registered_variables,
        state_with_explosion,
        medium_time,
        injected_energy,
        cooling_enabled,
        cooling_config,
        cooling_params,
    )


def _save_central_slices(
    snapshots,
    registered_variables,
    metadata: dict,
    output_path: Path,
    medium_time: float,
    duration: float,
    injected_energy: float,
    cooling_enabled: bool,
) -> Path:
    states = np.asarray(snapshots.states)
    local_time_points = np.asarray(snapshots.time_points, dtype=float)
    physical_scaling = _physical_scaling_from_metadata(metadata)
    if states.ndim != 5:
        raise ValueError(
            "Die Supernova-Snapshots müssen die Form "
            f"(Frames, Variablen, Nx, Ny, Nz) besitzen; gefunden: {states.shape}."
        )
    if len(local_time_points) != states.shape[0]:
        raise ValueError("Anzahl der Zeitpunkte und Zustände stimmt nicht überein.")
    if not np.all(np.isfinite(states)):
        raise RuntimeError(
            "Die Supernova-Snapshots enthalten nicht-endliche Werte. "
            "Die Phase 2 ist numerisch ungültig und wird nicht gespeichert."
        )
    if (
        not np.all(np.isfinite(local_time_points))
        or np.any(np.diff(local_time_points) <= 0.0)
    ):
        raise RuntimeError(
            "Die lokalen Supernova-Zeitpunkte sind ungültig oder nicht "
            "streng ansteigend."
        )

    nx, ny, nz = states.shape[2:]
    if not (nx == ny == nz):
        raise ValueError("Für die Schnittdarstellung wird ein kubisches Gitter erwartet.")
    z_index = nz // 2
    density_index = int(registered_variables.density_index)
    pressure_index = int(registered_variables.pressure_index)
    velocity_x_index = int(registered_variables.velocity_index.x)
    velocity_y_index = int(registered_variables.velocity_index.y)
    velocity_z_index = int(registered_variables.velocity_index.z)

    output_path = _next_available_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_out = {
        "phase": "supernova_in_turbulent_medium",
        "times_are_absolute": True,
        "medium_end_time": float(medium_time),
        "phase_end_time": float(medium_time + duration),
        "z_index": int(z_index),
        "box_size": metadata["box_size"],
        "state_layout": {
            "density_index": density_index,
            "velocity_x_index": velocity_x_index,
            "velocity_y_index": velocity_y_index,
            "velocity_z_index": velocity_z_index,
            "pressure_index": pressure_index,
        },
        "injected_energy": float(injected_energy),
        "injected_energy_code": float(injected_energy),
        "injected_energy_erg": (
            float(injected_energy * physical_scaling.energy_unit_erg)
            if physical_scaling is not None
            else None
        ),
        "physical_scaling": metadata.get("physical_scaling"),
        "cooling": bool(cooling_enabled),
        "cooling_model": metadata.get("cooling_model") if cooling_enabled else None,
        "cooling_method": metadata.get("cooling_method") if cooling_enabled else None,
        "cooling_factor": metadata.get("cooling_factor") if cooling_enabled else None,
        "cooling_exponent": (
            metadata.get("cooling_exponent") if cooling_enabled else None
        ),
        "cooling_reference_temperature": (
            metadata.get("cooling_reference_temperature")
            if cooling_enabled
            else None
        ),
        "cooling_floor_temperature": (
            metadata.get("cooling_floor_temperature")
            if cooling_enabled
            else None
        ),
        "cooling_floor_temperature_k": (
            metadata.get("cooling_floor_temperature_k")
            if cooling_enabled
            else None
        ),
        "cooling_table_source": (
            metadata.get("cooling_table_source") if cooling_enabled else None
        ),
    }
    absolute_time_points = medium_time + local_time_points
    np.savez_compressed(
        output_path,
        time_points=absolute_time_points,
        local_time_points=local_time_points,
        density=states[:, density_index, :, :, z_index],
        velocity_x=states[:, velocity_x_index, :, :, z_index],
        velocity_y=states[:, velocity_y_index, :, :, z_index],
        velocity_z=states[:, velocity_z_index, :, :, z_index],
        pressure=states[:, pressure_index, :, :, z_index],
        metadata_json=json.dumps(metadata_out, sort_keys=True),
    )
    print(f"Gespeicherte Supernova-Schnittdaten: {output_path}")
    print(f"Absolute Zeit: {absolute_time_points[0]:.6f} ... {absolute_time_points[-1]:.6f}")
    print(f"Verwendete zentrale z-Ebene: z_index={z_index}")
    return output_path


def main() -> None:
    args = _parse_args()
    primitive_state, metadata = _load_medium(args.input)
    (
        config,
        params,
        registered_variables,
        state_with_explosion,
        medium_time,
        injected_energy,
        cooling_enabled,
        cooling_config,
        cooling_params,
    ) = _build_phase_two(primitive_state, metadata, args)

    print(f"Geladenes turbulentes Medium: {args.input}")
    print(f"Supernova-Startzeit: t={medium_time:g}")
    print(f"Supernova-Endzeit: t={medium_time + args.duration:g}")
    print(f"OU-Forcierung in Phase 2: {config.turbulent_forcing_config.turbulent_forcing}")
    print(f"Cooling in Phase 2: {cooling_enabled}")
    if cooling_enabled:
        print(
            "Cooling-Parameter in Phase 2: "
            f"Methode={metadata.get('cooling_method')}, "
            f"Faktor={metadata.get('cooling_factor')}"
        )
    print(f"Zusätzliche Explosionsenergie in Codeeinheiten: {injected_energy:.8g}")
    physical_scaling = _physical_scaling_from_metadata(metadata)
    if physical_scaling is not None:
        print(
            "Zusätzliche Explosionsenergie physikalisch: "
            f"{injected_energy * physical_scaling.energy_unit_erg:.6e} erg"
        )
    print("Starte die Supernova-Phase im vollständigen 3D-Zustand.")

    snapshots = time_integration(
        state_with_explosion,
        config,
        params,
        registered_variables,
    )
    _save_central_slices(
        snapshots,
        registered_variables,
        metadata,
        args.output,
        medium_time,
        args.duration,
        injected_energy,
        cooling_enabled,
    )


if __name__ == "__main__":
    main()
