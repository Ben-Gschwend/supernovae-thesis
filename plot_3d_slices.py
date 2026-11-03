"""Animierter z-Schnitt durch gespeicherte 3D-Sedov-Daten.

Die Eingabe wird von ``save_3d_radial_profiles_gif`` erzeugt und enthält die
vollständigen 3D-Felder ``cell_density``, ``cell_velocity`` und
``cell_pressure``. Die Standardauswertung verwendet die zentrale Ebene
``z' = 0`` relativ zum Boxzentrum. Bei einer Box von 0 bis 1 entspricht das
der physikalischen Ebene ``z = 0.5``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm, Normalize


def _next_available_path(output_path: Path) -> Path:
    """Verhindere, dass eine bestehende Ergebnisdatei überschrieben wird."""

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


def _finite_range(values: np.ndarray, name: str) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"{name} enthält keine endlichen Werte.")

    value_min = float(np.min(finite))
    value_max = float(np.max(finite))
    if value_min == value_max:
        value_max = value_min + 1.0
    return value_min, value_max


def save_3d_slice_gif(
    input_path: str | Path,
    output_path: str | Path,
    *,
    box_size: float = 1.0,
    z_index: int | None = None,
    fps: int = 10,
    interval: int = 100,
) -> Path:
    """Erzeuge ein GIF mit Dichte-, Geschwindigkeits- und Druck-Schnitt.

    Parameters
    ----------
    input_path:
        NPZ-Datei aus der 3D-Profil-Auswertung.
    output_path:
        Ziel-GIF. Existierende Dateien werden automatisch nummeriert.
    box_size:
        Kantenlänge der kubischen Simulationsbox.
    z_index:
        Gitterindex der Schnittfläche. Standardmäßig wird die zentrale Ebene
        ``nz // 2`` verwendet, also ``z' = 0`` relativ zum Boxzentrum.
    fps, interval:
        Bildrate und Anzeigeintervall der Animation.
    """

    input_path = Path(input_path)
    output_path = _next_available_path(Path(output_path))

    with np.load(input_path, allow_pickle=False) as data:
        required = {
            "cell_density",
            "cell_velocity",
            "cell_pressure",
            "time_points",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(
                "Die NPZ-Datei enthält nicht die benötigten Felder: "
                + ", ".join(missing)
            )

        density = np.asarray(data["cell_density"])
        velocity = np.asarray(data["cell_velocity"])
        pressure = np.asarray(data["cell_pressure"])
        time_points = np.asarray(data["time_points"])

    for name, values in (
        ("cell_density", density),
        ("cell_velocity", velocity),
        ("cell_pressure", pressure),
    ):
        if values.ndim != 4:
            raise ValueError(
                f"{name} muss die Form (Frames, Nx, Ny, Nz) besitzen; "
                f"gefunden wurde {values.shape}."
            )

    if not (density.shape == velocity.shape == pressure.shape):
        raise ValueError("Dichte, Geschwindigkeit und Druck haben unterschiedliche Formen.")
    if len(time_points) != density.shape[0]:
        raise ValueError("Anzahl der Zeitpunkte und Frames stimmt nicht überein.")
    if box_size <= 0.0:
        raise ValueError("box_size muss positiv sein.")

    number_of_frames, nx, ny, nz = density.shape
    if z_index is None:
        z_index = nz // 2
    if not 0 <= z_index < nz:
        raise ValueError(f"z_index muss zwischen 0 und {nz - 1} liegen.")

    z_coordinate = (z_index + 0.5) * box_size / nz
    z_relative = z_coordinate - box_size / 2.0

    density_min, density_max = _finite_range(density, "Dichte")
    velocity_min, velocity_max = _finite_range(velocity, "Geschwindigkeit")
    pressure_positive = pressure[np.isfinite(pressure) & (pressure > 0.0)]
    if pressure_positive.size == 0:
        raise ValueError("Der Druck enthält keine positiven Werte für eine Log-Skala.")
    pressure_min = float(np.min(pressure_positive))
    pressure_max = float(np.max(pressure_positive))
    if pressure_min == pressure_max:
        pressure_max = pressure_min * 10.0

    x_min = -box_size / 2.0
    x_max = box_size / 2.0
    y_min = -box_size / 2.0
    y_max = box_size / 2.0
    extent = (x_min, x_max, y_min, y_max)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fields = (
        (density, r"Dichte $\rho$", Normalize(density_min, density_max), "viridis"),
        (
            velocity,
            r"Geschwindigkeitsbetrag $|v|$",
            Normalize(velocity_min, velocity_max),
            "magma",
        ),
        (pressure, r"Druck $p$", LogNorm(pressure_min, pressure_max), "plasma"),
    )

    images = []
    colorbars = []
    for axis, (values, title, norm, cmap) in zip(axes, fields):
        initial_slice = values[0, :, :, z_index].T
        image = axis.imshow(
            initial_slice,
            origin="lower",
            extent=extent,
            aspect="equal",
            norm=norm,
            cmap=cmap,
            interpolation="nearest",
        )
        images.append(image)
        colorbars.append(figure.colorbar(image, ax=axis, pad=0.02))
        axis.set_title(title)
        axis.set_xlabel("x' = x - 0.5")
        axis.set_ylabel("y' = y - 0.5")

    def update(frame: int):
        for image, (values, _, _, _) in zip(images, fields):
            image.set_data(values[frame, :, :, z_index].T)
        figure.suptitle(
            "3D-Sedov-Schnitt: z' = 0 "
            f"(physikalisch z = {z_coordinate:.6f}), "
            f"t = {time_points[frame]:.6f}"
        )
        return tuple(images)

    movie = FuncAnimation(
        figure,
        update,
        frames=number_of_frames,
        interval=interval,
        blit=False,
        repeat=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    print(f"Gespeichertes 3D-Slice-GIF: {output_path}")
    print(
        f"Verwendete Ebene: z_index={z_index}, "
        f"physikalisch z={z_coordinate:.6f}, relativ z'={z_relative:.6f}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Erzeuge drei animierte z-Schnittflächen aus einer 3D-NPZ-Datei."
    )
    parser.add_argument("--input", required=True, type=Path, help="Eingabe-NPZ")
    parser.add_argument("--output", required=True, type=Path, help="Ausgabe-GIF")
    parser.add_argument("--box-size", type=float, default=1.0)
    parser.add_argument(
        "--z-index",
        type=int,
        default=None,
        help="Gitterindex der z-Ebene; Standard ist die zentrale Ebene.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--interval", type=int, default=100)
    args = parser.parse_args()

    save_3d_slice_gif(
        args.input,
        args.output,
        box_size=args.box_size,
        z_index=args.z_index,
        fps=args.fps,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()
