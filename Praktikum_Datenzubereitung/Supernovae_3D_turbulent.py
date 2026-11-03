"""Simuliere eine Supernova in einem zuvor erzeugten turbulenten Medium.

Dieses Skript bildet die zweite Phase des Experiments ab:

1. den finalen Zustand aus ``anfangsmedium.py`` bei beispielsweise ``t=5``
   laden,
2. eine zusätzliche thermische Explosionsenergie im Boxzentrum einbringen,
3. die Hydrodynamik ohne erneute OU-Forcierung weiterentwickeln,
4. radiale Profile von Dichte, Geschwindigkeitsbetrag und Druck als GIF und
   NPZ speichern.

Die vorhandene Dichte und die vorhandenen Geschwindigkeitskomponenten des
turbulenten Mediums bleiben erhalten. Nur der Druck wird um den normierten
Explosionsbeitrag erhöht.

Die Defaultwerte für Explosionsradius und Energie sind vorläufig. Der Radius
``0.01`` entspricht dem bisherigen idealisierten 128er-Sedov-Test, erzeugt
aber eine sehr konzentrierte Explosion und muss für die wissenschaftliche
Konfiguration noch mit der Betreuung abgestimmt werden.
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
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    PERIODIC_BOUNDARY,
    SnapshotSettings,
)

from Praktikum_Datenzubereitung.Supernova_plot_3D import (
    save_3d_radial_profiles_gif,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simuliere eine Supernova im gespeicherten turbulenten Medium."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "Praktikum_Datenzubereitung/turbulent_medium_128cells_t5_kf25_gpu2.npz"
        ),
        help="NPZ-Datei mit dem finalen turbulenten Zustand.",
    )
    parser.add_argument("--duration", type=float, default=0.1)
    parser.add_argument("--num-snapshots", type=int, default=60)
    parser.add_argument("--explosion-energy", type=float, default=1.0)
    parser.add_argument(
        "--explosion-radius",
        type=float,
        default=0.01,
        help="Radius der zusätzlichen thermischen Explosionsregion.",
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
        "--num-bins",
        type=int,
        default=128,
        help="Anzahl radialer Bins für die Profilanalyse.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "Praktikum_Datenzubereitung/supernova_3d_turbulent_128cells.gif"
        ),
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=Path(
            "Praktikum_Datenzubereitung/supernova_3d_turbulent_128cells.npz"
        ),
        help="Ausgabedatei für die numerischen radialen Profile.",
    )
    parser.add_argument(
        "--slice-output",
        type=Path,
        default=Path(
            "Praktikum_Datenzubereitung/supernova_3d_turbulent_128cells_slices.npz"
        ),
        help=(
            "Ausgabedatei für zentrale z-Schnitte der Supernova-Phase; "
            "wird für die gemeinsame Medium/Supernova-Animation benötigt."
        ),
    )
    return parser.parse_args()


def _periodic_boundaries() -> BoundarySettings:
    periodic = BoundarySettings1D(
        left_boundary=PERIODIC_BOUNDARY,
        right_boundary=PERIODIC_BOUNDARY,
    )
    return BoundarySettings(x=periodic, y=periodic, z=periodic)


def _load_medium(input_path: Path):
    with np.load(input_path, allow_pickle=False) as data:
        required = {"primitive_state", "metadata_json"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Fehlende NPZ-Felder: {missing}")
        primitive_state = np.asarray(data["primitive_state"])
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    return primitive_state, metadata


def _next_available_path(output_path: Path) -> Path:
    """Verhindere das Überschreiben einer vorhandenen Schnittdatei."""

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


def _save_supernova_slice_data(
    snapshots,
    registered_variables,
    metadata: dict,
    output_path: Path,
) -> Path:
    """Speichere zentrale Schnittdaten für die kombinierte Animation."""

    states = np.asarray(snapshots.states)
    time_points = np.asarray(snapshots.time_points)
    if states.ndim != 5:
        raise ValueError(
            "Die Supernova-Snapshots müssen die Form "
            f"(Frames, Variablen, Nx, Ny, Nz) besitzen; gefunden: {states.shape}."
        )
    if len(time_points) != states.shape[0]:
        raise ValueError("Anzahl der Supernova-Zeitpunkte und Zustände stimmt nicht überein.")

    nx, ny, nz = states.shape[2:]
    if not (nx == ny == nz):
        raise ValueError("Für die Schnittdarstellung wird ein kubisches Gitter erwartet.")
    z_index = nz // 2
    density_index = int(registered_variables.density_index)
    pressure_index = int(registered_variables.pressure_index)
    velocity_x_index = int(registered_variables.velocity_index.x)
    velocity_y_index = int(registered_variables.velocity_index.y)
    velocity_z_index = int(registered_variables.velocity_index.z)

    slice_metadata = {
        "phase": "supernova_in_turbulent_medium",
        "medium_end_time": float(metadata.get("t_end", 0.0)),
        "z_index": z_index,
        "box_size": metadata["box_size"],
        "state_layout": {
            "density_index": density_index,
            "velocity_x_index": velocity_x_index,
            "velocity_y_index": velocity_y_index,
            "velocity_z_index": velocity_z_index,
            "pressure_index": pressure_index,
        },
    }
    output_path = _next_available_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        time_points=time_points,
        density=states[:, density_index, :, :, z_index],
        velocity_x=states[:, velocity_x_index, :, :, z_index],
        velocity_y=states[:, velocity_y_index, :, :, z_index],
        velocity_z=states[:, velocity_z_index, :, :, z_index],
        pressure=states[:, pressure_index, :, :, z_index],
        metadata_json=json.dumps(slice_metadata),
    )
    print(f"Gespeicherte Supernova-Schnittdaten: {output_path}")
    print(f"Verwendete z-Ebene: z_index={z_index}")
    return output_path


def _build_state_with_explosion(
    primitive_state: np.ndarray,
    metadata: dict,
    args: argparse.Namespace,
):
    if primitive_state.ndim != 4 or primitive_state.shape[0] < 5:
        raise ValueError(
            "primitive_state muss die Form (Variablen, Nx, Ny, Nz) besitzen "
            "und Dichte, drei Geschwindigkeiten sowie Druck enthalten."
        )
    if args.explosion_energy <= 0.0:
        raise ValueError("--explosion-energy muss positiv sein.")
    if args.explosion_radius <= 0.0:
        raise ValueError("--explosion-radius muss positiv sein.")
    if args.smoothing_cells <= 0.0:
        raise ValueError("--smoothing-cells muss positiv sein.")

    box_size = float(metadata["box_size"][0])
    nx, ny, nz = primitive_state.shape[1:]
    if not (nx == ny == nz):
        raise ValueError("Das aktuelle Skript erwartet ein kubisches 3D-Gitter.")
    num_cells = nx
    gamma = float(args.gamma if args.gamma is not None else metadata["gamma"])

    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=3,
        box_size=box_size,
        num_cells=num_cells,
        mhd=False,
        solver_mode=FINITE_VOLUME,
        progress_bar=True,
        random_seed=int(metadata.get("random_seed", 42)),
        boundary_settings=_periodic_boundaries(),
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=False,
            ou_forcing=False,
            vacuum_protection=False,
        ),
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
    if args.boundary == "open":
        config = config._replace(boundary_settings=None)
    elif args.boundary == "metadata" and metadata.get("boundary") == "open":
        config = config._replace(boundary_settings=None)

    registered_variables = get_registered_variables(config)
    layout = metadata["state_layout"]
    expected_layout = {
        "density_index": int(registered_variables.density_index),
        "pressure_index": int(registered_variables.pressure_index),
        "velocity_x_index": int(registered_variables.velocity_index.x),
        "velocity_y_index": int(registered_variables.velocity_index.y),
        "velocity_z_index": int(registered_variables.velocity_index.z),
    }
    if expected_layout != {key: int(value) for key, value in layout.items()}:
        raise ValueError(
            "Die Variablenreihenfolge der NPZ-Datei passt nicht zur aktuellen "
            f"astronomix-Konfiguration: gespeichert={layout}, "
            f"aktuell={expected_layout}."
        )

    helper_data = get_helper_data(config)
    coordinates = helper_data.geometric_centers
    center = jnp.array([box_size / 2.0, box_size / 2.0, box_size / 2.0])
    radius = jnp.linalg.norm(coordinates - center, axis=-1)
    smoothing_width = args.smoothing_cells * box_size / num_cells
    window = 0.5 * (
        1.0 - jnp.tanh((radius - args.explosion_radius) / smoothing_width)
    )
    cell_volume = (box_size / num_cells) ** 3
    delta_pressure = args.explosion_energy * (gamma - 1.0) / (
        cell_volume * jnp.sum(window)
    )

    state = jnp.asarray(primitive_state)
    pressure_index = registered_variables.pressure_index
    density_index = registered_variables.density_index
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
    )
    return (
        config,
        params,
        registered_variables,
        state_with_explosion,
        injected_energy,
        float(jnp.min(state[pressure_index])),
        float(jnp.max(state[pressure_index])),
        float(jnp.min(state_with_explosion[pressure_index])),
        float(jnp.max(state_with_explosion[pressure_index])),
    )


def main() -> None:
    args = _parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration muss positiv sein.")
    if args.num_snapshots < 2:
        raise ValueError("--num-snapshots muss mindestens 2 sein.")

    primitive_state, metadata = _load_medium(args.input)
    (
        config,
        params,
        registered_variables,
        state_with_explosion,
        injected_energy,
        pressure_min_before,
        pressure_max_before,
        pressure_min_after,
        pressure_max_after,
    ) = _build_state_with_explosion(primitive_state, metadata, args)

    print(f"Geladenes turbulentes Medium: {args.input}")
    print(f"Medium-Endzeit vor Explosion: {metadata.get('t_end', 'unbekannt')}")
    print(f"OU-Forcierung in Phase 2: {config.turbulent_forcing_config.turbulent_forcing}")
    print(f"Normierte zusätzliche Explosionsenergie: {injected_energy:.8g}")
    print(
        "Druckbereich vor/nach Explosion: "
        f"[{pressure_min_before:.6g}, {pressure_max_before:.6g}] -> "
        f"[{pressure_min_after:.6g}, {pressure_max_after:.6g}]"
    )
    print("Starte die Supernova-Phase ohne erneute Medium-Initialisierung.")

    snapshots = time_integration(
        state_with_explosion,
        config,
        params,
        registered_variables,
    )
    _save_supernova_slice_data(
        snapshots,
        registered_variables,
        metadata,
        args.slice_output,
    )
    save_3d_radial_profiles_gif(
        snapshots,
        registered_variables,
        box_size=config.box_size,
        output_path=args.output,
        data_output_path=args.data_output,
        num_bins=args.num_bins,
        interval=100,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
