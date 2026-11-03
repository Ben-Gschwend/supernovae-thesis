"""Erzeuge eine gemeinsame z-Schnitt-Animation von Medium und Supernova.

Die Animation kombiniert:

* die gespeicherten vollständigen 3D-Snapshots des turbulenten Mediums von
  ``t=0`` bis ``t=5``;
* die zentralen z-Schnitte der Supernova-Phase von ``t=5`` bis ``t=5.1``.

Der Druck wird logarithmisch dargestellt. Standardmäßig wird die Druckskala
aus dem turbulenten Medium bestimmt, damit die Hintergrundturbulenz auch nach
dem Einsetzen der Supernova sichtbar bleibt. Sehr hohe Supernova-Drücke
sättigen dabei am oberen Rand der Farbskala.

Auch die Dichte kann logarithmisch dargestellt werden. Standardmäßig werden
dabei die positiven Dichtewerte des turbulenten Mediums als Referenz für die
Farbgrenzen verwendet, damit Dichteschwankungen im Hintergrund sichtbar
bleiben.

Mit ``log-two-sided`` kann die Dichteskala alternativ über niedrige und hohe
Dichten gemeinsam aufgeteilt werden. Sie wird im logarithmischen Dichtewert
um die mittlere Mediumdichte zentriert; die beiden Randbereiche erhalten mehr
Farbauflösung, während mittlere Dichten komprimiert werden.

Zusätzlich kann der logarithmisch normierte Dichtekontrast mit einer
Potenztransformation verstärkt werden. Ein kleinerer Exponent als ``1`` hebt
Werte nahe der unteren Farbgrenze stärker hervor; dies ersetzt eine nicht
sinnvolle doppelte Anwendung von ``log``.

Die Medium-Phase wird bis ``t=4.98`` normal abgespielt. Ab ``t=4.98`` werden
die Frames standardmäßig viermal wiederholt, sodass der Übergang bei ``t=5``
und die Ausbreitung der Supernova bis ``t=5.1`` deutlich langsamer erscheint.
Das Skript startet keine Simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm, Normalize


def _next_available_path(output_path: Path) -> Path:
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


def _read_json(data, key: str) -> dict:
    if key not in data.files:
        raise ValueError(f"Die NPZ-Datei enthält kein Feld {key!r}.")
    return json.loads(str(np.asarray(data[key]).item()))


def _load_medium(input_path: Path):
    with np.load(input_path, allow_pickle=False) as data:
        required = {"states", "time_points", "metadata_json"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                f"Fehlende Medium-Felder: {missing}. "
                "Das Medium muss mit --save-snapshots erzeugt worden sein."
            )
        states = np.asarray(data["states"])
        time_points = np.asarray(data["time_points"], dtype=float)
        metadata = _read_json(data, "metadata_json")
    return states, time_points, metadata


def _load_supernova(input_path: Path):
    with np.load(input_path, allow_pickle=False) as data:
        required = {
            "time_points",
            "density",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "pressure",
            "metadata_json",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Fehlende Supernova-Felder: {missing}")
        fields = {
            key: np.asarray(data[key])
            for key in (
                "time_points",
                "density",
                "velocity_x",
                "velocity_y",
                "velocity_z",
                "pressure",
            )
        }
        fields["time_points"] = fields["time_points"].astype(float)
        metadata = _read_json(data, "metadata_json")
    return fields, metadata


def _validate_medium(states, time_points, metadata):
    if states.ndim != 5:
        raise ValueError(
            "Die Medium-Snapshots müssen die Form "
            f"(Frames, Variablen, Nx, Ny, Nz) besitzen; gefunden: {states.shape}."
        )
    if len(time_points) != states.shape[0]:
        raise ValueError("Medium-Zeitpunkte und Medium-Snapshots passen nicht zusammen.")
    if states.shape[2] != states.shape[3] or states.shape[2] != states.shape[4]:
        raise ValueError("Für die gemeinsame Darstellung wird ein kubisches Medium erwartet.")
    layout = metadata.get("state_layout")
    required_layout = {
        "density_index",
        "velocity_x_index",
        "velocity_y_index",
        "velocity_z_index",
        "pressure_index",
    }
    if layout is None or not required_layout.issubset(layout):
        raise ValueError("Die Medium-Metadaten enthalten kein vollständiges state_layout.")


def _validate_supernova(fields):
    number_of_frames = len(fields["time_points"])
    shapes = {
        key: fields[key].shape
        for key in ("density", "velocity_x", "velocity_y", "velocity_z", "pressure")
    }
    if any(len(shape) != 3 for shape in shapes.values()):
        raise ValueError(
            "Die Supernova-Schnittfelder müssen die Form (Frames, Nx, Ny) besitzen."
        )
    if any(shape[0] != number_of_frames for shape in shapes.values()):
        raise ValueError("Supernova-Zeitpunkte und Supernova-Schnittfelder passen nicht zusammen.")
    if len({shape[1:] for shape in shapes.values()}) != 1:
        raise ValueError("Die Supernova-Schnittfelder besitzen unterschiedliche Gitterformen.")


def _box_size(metadata: dict) -> float:
    values = np.asarray(metadata["box_size"], dtype=float).reshape(-1)
    if values.size == 0 or not np.allclose(values, values[0]):
        raise ValueError("Die Darstellung erwartet eine kubische Box.")
    return float(values[0])


def save_supernovae_turbolenz_gif(
    medium_input: str | Path,
    supernova_input: str | Path,
    output_path: str | Path,
    *,
    slow_start: float = 4.98,
    slowdown_factor: int = 2,
    quiver_stride: int = 8,
    fps: int = 10,
    pressure_scale: str = "combined",
    pressure_low_percentile: float = 0.0,
    pressure_high_percentile: float = 100.0,
    density_scale: str = "log-two-sided",
    density_low_percentile: float = 1.0,
    density_high_percentile: float = 99.0,
    density_contrast: float = 1.0,
    density_tail_exponent: float = 0.35,
    density_center: float | None = None,
) -> Path:
    """Speichere die gemeinsame Medium-/Supernova-Schnittanimation."""

    if slowdown_factor < 1:
        raise ValueError("slowdown_factor muss mindestens 1 sein.")
    if quiver_stride < 1:
        raise ValueError("quiver_stride muss mindestens 1 sein.")
    if fps < 1:
        raise ValueError("fps muss mindestens 1 sein.")
    if pressure_scale not in {"medium", "combined"}:
        raise ValueError("pressure_scale muss 'medium' oder 'combined' sein.")
    if not 0.0 <= pressure_low_percentile < pressure_high_percentile <= 100.0:
        raise ValueError(
            "Die Druck-Perzentile müssen 0 <= niedrig < hoch <= 100 erfüllen."
        )
    if not 0.0 <= density_low_percentile < density_high_percentile <= 100.0:
        raise ValueError(
            "Die Dichte-Perzentile müssen 0 <= niedrig < hoch <= 100 erfüllen."
        )
    if density_scale not in {
        "linear",
        "log-medium",
        "log-combined",
        "log-two-sided",
    }:
        raise ValueError(
            "density_scale muss 'linear', 'log-medium', 'log-combined' "
            "oder 'log-two-sided' sein."
        )
    if density_contrast <= 0.0:
        raise ValueError("density_contrast muss positiv sein.")
    if density_tail_exponent <= 0.0:
        raise ValueError("density_tail_exponent muss positiv sein.")

    states, medium_times, medium_metadata = _load_medium(Path(medium_input))
    supernova, supernova_metadata = _load_supernova(Path(supernova_input))
    _validate_medium(states, medium_times, medium_metadata)
    _validate_supernova(supernova)

    layout = medium_metadata["state_layout"]
    medium_density = states[:, int(layout["density_index"]), :, :, states.shape[4] // 2]
    medium_vx = states[:, int(layout["velocity_x_index"]), :, :, states.shape[4] // 2]
    medium_vy = states[:, int(layout["velocity_y_index"]), :, :, states.shape[4] // 2]
    medium_vz = states[:, int(layout["velocity_z_index"]), :, :, states.shape[4] // 2]
    medium_pressure = states[:, int(layout["pressure_index"]), :, :, states.shape[4] // 2]

    density = np.concatenate((medium_density, supernova["density"]), axis=0)
    velocity_x = np.concatenate((medium_vx, supernova["velocity_x"]), axis=0)
    velocity_y = np.concatenate((medium_vy, supernova["velocity_y"]), axis=0)
    velocity_z = np.concatenate((medium_vz, supernova["velocity_z"]), axis=0)
    pressure = np.concatenate((medium_pressure, supernova["pressure"]), axis=0)
    time_points = np.concatenate((medium_times, supernova["time_points"]), axis=0)
    phase_names = np.concatenate(
        (
            np.full(len(medium_times), "Turbulentes Medium", dtype=object),
            np.full(len(supernova["time_points"]), "Supernova im turbulenten Medium", dtype=object),
        )
    )

    if np.any(np.diff(time_points) < -1e-9):
        raise ValueError("Die kombinierten Zeitpunkte sind nicht monoton sortiert.")
    if not np.isclose(supernova["time_points"][0], medium_times[-1], atol=1e-5):
        raise ValueError(
            "Die Supernova muss beim letzten Medium-Zeitpunkt starten: "
            f"Medium={medium_times[-1]}, Supernova={supernova['time_points'][0]}"
        )

    speed = np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)
    finite_density = density[np.isfinite(density)]
    finite_speed = speed[np.isfinite(speed)]
    positive_pressure = pressure[np.isfinite(pressure) & (pressure > 0.0)]
    if not finite_density.size or not finite_speed.size or not positive_pressure.size:
        raise ValueError("Die kombinierten Felder enthalten keine gültigen Werte.")

    if density_scale == "linear":
        density_reference = finite_density
    elif density_scale == "log-medium":
        density_reference = medium_density[
            np.isfinite(medium_density) & (medium_density > 0.0)
        ]
    else:
        density_reference = finite_density[finite_density > 0.0]
    if density_reference.size == 0:
        raise ValueError("Es gibt keine gültigen positiven Dichtewerte für die Skala.")
    density_min = float(np.percentile(density_reference, density_low_percentile))
    density_max = float(np.percentile(density_reference, density_high_percentile))
    if density_center is None:
        medium_positive_density = medium_density[
            np.isfinite(medium_density) & (medium_density > 0.0)
        ]
        density_center_value = float(np.median(medium_positive_density))
    else:
        density_center_value = float(density_center)
    if density_center_value <= 0.0:
        raise ValueError("density_center muss positiv sein.")
    if density_scale == "log-two-sided" and not (
        density_min < density_center_value < density_max
    ):
        raise ValueError(
            "density_center muss bei log-two-sided zwischen den Dichtegrenzen liegen."
        )
    speed_min, speed_max = float(np.min(finite_speed)), float(np.max(finite_speed))
    if pressure_scale == "medium":
        pressure_reference = medium_pressure[
            np.isfinite(medium_pressure) & (medium_pressure > 0.0)
        ]
        if pressure_reference.size == 0:
            raise ValueError("Das turbulente Medium enthält keine positiven Drücke.")
    else:
        pressure_reference = positive_pressure
    pressure_min = float(np.percentile(pressure_reference, pressure_low_percentile))
    pressure_max = float(np.percentile(pressure_reference, pressure_high_percentile))
    if density_min == density_max:
        density_max = density_min + 1.0
    if speed_min == speed_max:
        speed_max = speed_min + 1.0
    if pressure_min == pressure_max:
        pressure_max = pressure_min * 10.0

    frames_y, frames_x = density.shape[1:]
    box_size = _box_size(medium_metadata)
    x = np.linspace(-box_size / 2.0, box_size / 2.0, frames_x)
    y = np.linspace(-box_size / 2.0, box_size / 2.0, frames_y)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Xq = X[::quiver_stride, ::quiver_stride]
    Yq = Y[::quiver_stride, ::quiver_stride]

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    density_colorbar_ticks = None
    density_colorbar_labels = None
    if density_scale == "linear":
        density_for_display = density
        density_norm = Normalize(density_min, density_max)
    elif density_scale in {"log-medium", "log-combined"}:
        log_density_min = np.log10(density_min)
        log_density_range = np.log10(density_max) - log_density_min
        if log_density_range <= 0.0:
            log_density_range = 1.0

        def transform_density(values):
            safe_values = np.maximum(values, density_min)
            normalized = np.clip(
                (np.log10(safe_values) - log_density_min) / log_density_range,
                0.0,
                1.0,
            )
            return normalized**density_contrast

        density_for_display = transform_density(density)
        density_norm = Normalize(0.0, 1.0)
        density_tick_values = np.geomspace(density_min, density_max, num=4)
        density_colorbar_ticks = transform_density(density_tick_values)
        density_colorbar_labels = [f"{value:.3g}" for value in density_tick_values]
    else:
        log_density_min = np.log10(density_min)
        log_density_center = np.log10(density_center_value)
        log_density_max = np.log10(density_max)
        lower_log_range = log_density_center - log_density_min
        upper_log_range = log_density_max - log_density_center

        def transform_density(values):
            safe_values = np.maximum(values, density_min)
            log_values = np.log10(safe_values)
            lower = log_values <= log_density_center
            normalized = np.empty_like(log_values, dtype=float)
            normalized[lower] = 0.5 * np.clip(
                (log_values[lower] - log_density_min) / lower_log_range,
                0.0,
                1.0,
            ) ** density_tail_exponent
            normalized[~lower] = 0.5 + 0.5 * np.clip(
                (log_values[~lower] - log_density_center) / upper_log_range,
                0.0,
                1.0,
            ) ** density_tail_exponent
            return normalized

        density_for_display = transform_density(density)
        density_norm = Normalize(0.0, 1.0)
        lower_ticks = np.geomspace(density_min, density_center_value, num=3)
        upper_ticks = np.geomspace(density_center_value, density_max, num=3)[1:]
        density_tick_values = np.concatenate((lower_ticks, upper_ticks))
        density_colorbar_ticks = transform_density(density_tick_values)
        density_colorbar_labels = [f"{value:.3g}" for value in density_tick_values]
    if density_scale == "linear":
        density_title = r"Dichte $\rho$"
    elif density_scale == "log-two-sided":
        density_title = r"Dichte $\rho$ (logarithmisch, Ränder verstärkt)"
    else:
        density_title = r"Dichte $\rho$ (logarithmisch)"
    fields = (
        (density_for_display, density_title, density_norm, "viridis"),
        (speed, r"Geschwindigkeitsbetrag $|v|$", Normalize(speed_min, speed_max), "magma"),
        (
            pressure,
            r"Druck $p$ (logarithmisch)",
            LogNorm(pressure_min, pressure_max, clip=True),
            "plasma",
        ),
    )
    images = []
    for axis, (values, title, norm, cmap) in zip(axes, fields):
        image = axis.imshow(
            values[0].T,
            origin="lower",
            extent=(-box_size / 2.0, box_size / 2.0, -box_size / 2.0, box_size / 2.0),
            aspect="equal",
            norm=norm,
            cmap=cmap,
            interpolation="nearest",
        )
        colorbar = figure.colorbar(image, ax=axis, pad=0.02)
        if density_colorbar_ticks is not None and axis is axes[0]:
            colorbar.set_ticks(density_colorbar_ticks)
            colorbar.set_ticklabels(density_colorbar_labels)
        axis.set_title(title)
        axis.set_xlabel("x' = x - L/2")
        axis.set_ylabel("y' = y - L/2")
        images.append(image)

    quiver = axes[1].quiver(
        Xq,
        Yq,
        velocity_x[0].T[::quiver_stride, ::quiver_stride],
        velocity_y[0].T[::quiver_stride, ::quiver_stride],
        color="white",
        angles="xy",
        scale_units="xy",
        scale=max(float(np.nanmax(speed[0])) / 0.12, 1e-12),
        width=0.0025,
    )

    repeat_counts = np.where(time_points >= slow_start, slowdown_factor, 1)
    # Bei t=5 liegen zwei unterschiedliche Zustände direkt hintereinander:
    # der letzte Zustand vor der Explosion und der Zustand nach dem Einsetzen
    # der Explosionsenergie. Den Vorher-Zustand nur einmal halten, damit die
    # physikalisch abrupte Änderung sichtbar bleibt, aber nicht unnötig lange
    # als scheinbarer Stillstand erscheint.
    same_time = np.isclose(np.diff(time_points), 0.0, atol=1e-8)
    phase_change = phase_names[:-1] != phase_names[1:]
    repeat_counts[:-1][same_time & phase_change] = 1
    expanded_frames = np.repeat(np.arange(len(time_points)), repeat_counts)

    def update(expanded_frame: int):
        frame = int(expanded_frames[expanded_frame])
        for image, (values, _, _, _) in zip(images, fields):
            image.set_data(values[frame].T)
        vx = velocity_x[frame].T
        vy = velocity_y[frame].T
        quiver.set_UVC(
            vx[::quiver_stride, ::quiver_stride],
            vy[::quiver_stride, ::quiver_stride],
        )
        current_max_speed = max(float(np.nanmax(speed[frame])), 1e-12)
        quiver.scale = max(current_max_speed / 0.12, 1e-12)
        figure.suptitle(
            f"{phase_names[frame]} | z-Schnitt | "
            f"t = {time_points[frame]:.5f}"
        )
        return (*images, quiver)

    movie = FuncAnimation(
        figure,
        update,
        frames=len(expanded_frames),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    output_path = _next_available_path(Path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    print(f"Gespeicherte kombinierte Animation: {output_path}")
    print(f"Normale Darstellung bis t={slow_start:g}")
    print(f"Zeitlupenfaktor ab t={slow_start:g}: {slowdown_factor}")
    print(f"Druckskalierung: logarithmisch, Referenz={pressure_scale}")
    print(
        "Druck-Perzentile: "
        f"{pressure_low_percentile:g}...{pressure_high_percentile:g}"
    )
    print(f"Dichteskalierung: {density_scale}")
    if density_scale != "linear":
        print(
            "Dichte-Perzentile: "
            f"{density_low_percentile:g}...{density_high_percentile:g}"
        )
        print(f"Dichte-Kontrastexponent: {density_contrast}")
    if density_scale == "log-two-sided":
        print(
            f"Dichtezentrum: {density_center_value:.6g}; "
            f"Randexponent: {density_tail_exponent:.6g}"
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plotten der Turbulenz und der Supernova bis t=5.1."
    )
    parser.add_argument(
        "--medium-input",
        type=Path,
        default=Path(
            "Praktikum_Datenzubereitung/turbulent_medium_128cells_t5_kf25_gpu2.npz"
        ),
        help=(
            "Medium-NPZ mit vollständigen Snapshots aus derselben kf25-"
            "Turbulenzsimulation wie der Supernova-Input."
        ),
    )
    parser.add_argument(
        "--supernova-input",
        type=Path,
        default=Path(
            "Praktikum_Datenzubereitung/supernovae_turbolenz_128cells.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "Praktikum_Datenzubereitung/supernovae_turbolenz_128cells.gif"
        ),
    )
    parser.add_argument("--slow-start", type=float, default=4.98)
    parser.add_argument("--slowdown-factor", type=int, default=2)
    parser.add_argument("--quiver-stride", type=int, default=8)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--pressure-scale",
        choices=("medium", "combined"),
        default="combined",
        help=(
            "Druck-Farbgrenzen aus dem turbulenten Medium oder aus allen "
            "Frames bestimmen; beide Varianten sind logarithmisch."
        ),
    )
    parser.add_argument(
        "--pressure-low-percentile",
        type=float,
        default=0.0,
        help="Untere Perzentilgrenze der logarithmischen Druckskala.",
    )
    parser.add_argument(
        "--pressure-high-percentile",
        type=float,
        default=100.0,
        help=(
            "Obere Perzentilgrenze der logarithmischen Druckskala. "
            "Standardmäßig wird wieder das vollständige Medium-Min/Max verwendet."
        ),
    )
    parser.add_argument(
        "--density-scale",
        choices=("linear", "log-medium", "log-combined", "log-two-sided"),
        default="log-two-sided",
        help=(
            "Dichteskala: linear oder logarithmisch mit Farbgrenzen aus "
            "dem Medium, aus allen Frames oder zweiseitig randverstärkt."
        ),
    )
    parser.add_argument(
        "--density-contrast",
        type=float,
        default=1.0,
        help=(
            "Kontrastexponent nach der logarithmischen Dichtenormierung. "
            "1.0 entspricht einer unveränderten logarithmischen Skala."
        ),
    )
    parser.add_argument(
        "--density-low-percentile",
        type=float,
        default=1.0,
        help="Untere Perzentilgrenze der logarithmischen Dichteskala.",
    )
    parser.add_argument(
        "--density-high-percentile",
        type=float,
        default=99.0,
        help="Obere Perzentilgrenze der logarithmischen Dichteskala.",
    )
    parser.add_argument(
        "--density-tail-exponent",
        type=float,
        default=0.35,
        help=(
            "Exponent der zweiseitigen Dichteskala. Kleinere Werte geben "
            "den niedrigen und hohen Dichten mehr Farbauflösung."
        ),
    )
    parser.add_argument(
        "--density-center",
        type=float,
        default=None,
        help=(
            "Zentrum der zweiseitigen Dichteskala. Standard ist die "
            "Mediane der Mediumdichte."
        ),
    )
    args = parser.parse_args()
    save_supernovae_turbolenz_gif(
        args.medium_input,
        args.supernova_input,
        args.output,
        slow_start=args.slow_start,
        slowdown_factor=args.slowdown_factor,
        quiver_stride=args.quiver_stride,
        fps=args.fps,
        pressure_scale=args.pressure_scale,
        pressure_low_percentile=args.pressure_low_percentile,
        pressure_high_percentile=args.pressure_high_percentile,
        density_scale=args.density_scale,
        density_low_percentile=args.density_low_percentile,
        density_high_percentile=args.density_high_percentile,
        density_contrast=args.density_contrast,
        density_tail_exponent=args.density_tail_exponent,
        density_center=args.density_center,
    )


if __name__ == "__main__":
    main()
