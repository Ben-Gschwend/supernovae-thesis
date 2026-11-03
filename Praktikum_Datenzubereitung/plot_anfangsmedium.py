"""Erzeuge ein GIF aus den gespeicherten turbulenten Medium-Snapshots.

Die Eingabe wird von ``anfangsmedium.py --save-snapshots`` erzeugt. Dargestellt
wird ein zentraler z-Schnitt mit:

* Dichte,
* Geschwindigkeitsbetrag,
* Druck auf logarithmischer Farbskala,
* Geschwindigkeitsvektoren im Geschwindigkeits-Panel.

Das Skript startet keine Simulation. Es liest nur die gespeicherte NPZ-Datei.
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
    """Verhindere das Überschreiben eines vorhandenen GIFs."""

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


def _load_snapshots(input_path: Path):
    with np.load(input_path, allow_pickle=False) as data:
        if "states" not in data.files:
            raise ValueError(
                "Die NPZ-Datei enthält keine Zwischenzustände. "
                "Das Medium muss mit anfangsmedium.py --save-snapshots "
                "neu erzeugt werden."
            )
        required = {"states", "time_points", "metadata_json"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Fehlende NPZ-Felder: {missing}")
        states = np.asarray(data["states"])
        time_points = np.asarray(data["time_points"])
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    return states, time_points, metadata


def save_anfangsmedium_gif(
    input_path: str | Path,
    output_path: str | Path,
    *,
    z_index: int | None = None,
    quiver_stride: int = 8,
    fps: int = 10,
):
    """Speichere die zeitliche Entwicklung des turbulenten Mediums als GIF."""

    input_path = Path(input_path)
    output_path = _next_available_path(Path(output_path))
    if quiver_stride < 1:
        raise ValueError("quiver_stride muss mindestens 1 sein.")
    if fps < 1:
        raise ValueError("fps muss mindestens 1 sein.")

    states, time_points, metadata = _load_snapshots(input_path)
    if states.ndim != 5:
        raise ValueError(
            "states muss die Form (Frames, Variablen, Nx, Ny, Nz) besitzen; "
            f"gefunden wurde {states.shape}."
        )
    if len(time_points) != states.shape[0]:
        raise ValueError("Anzahl der Zeitpunkte und Zustände stimmt nicht überein.")

    nx, ny, nz = states.shape[2:]
    if z_index is None:
        z_index = nz // 2
    if not 0 <= z_index < nz:
        raise ValueError(f"z_index muss zwischen 0 und {nz - 1} liegen.")

    layout = metadata["state_layout"]
    density = states[:, int(layout["density_index"])]
    pressure = states[:, int(layout["pressure_index"])]
    velocity_x = states[:, int(layout["velocity_x_index"])]
    velocity_y = states[:, int(layout["velocity_y_index"])]
    velocity_z = states[:, int(layout["velocity_z_index"])]
    speed = np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)

    density_min = float(np.nanmin(density))
    density_max = float(np.nanmax(density))
    speed_min = float(np.nanmin(speed))
    speed_max = float(np.nanmax(speed))
    positive_pressure = pressure[np.isfinite(pressure) & (pressure > 0.0)]
    if positive_pressure.size == 0:
        raise ValueError("Der Druck enthält keine positiven Werte.")
    pressure_min = float(np.min(positive_pressure))
    pressure_max = float(np.max(positive_pressure))
    if density_min == density_max:
        density_max = density_min + 1.0
    if speed_min == speed_max:
        speed_max = speed_min + 1.0
    if pressure_min == pressure_max:
        pressure_max = pressure_min * 10.0

    box_size = float(metadata["box_size"][0])
    x = np.linspace(-box_size / 2.0, box_size / 2.0, nx)
    y = np.linspace(-box_size / 2.0, box_size / 2.0, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Xq = X[::quiver_stride, ::quiver_stride]
    Yq = Y[::quiver_stride, ::quiver_stride]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
        constrained_layout=True,
    )
    fields = (
        (density, r"Dichte $\rho$", Normalize(density_min, density_max), "viridis"),
        (
            speed,
            r"Geschwindigkeitsbetrag $|v|$",
            Normalize(speed_min, speed_max),
            "magma",
        ),
        (pressure, r"Druck $p$", LogNorm(pressure_min, pressure_max), "plasma"),
    )

    images = []
    for axis, (values, title, norm, cmap) in zip(axes, fields):
        image = axis.imshow(
            values[0, :, :, z_index].T,
            origin="lower",
            extent=(-box_size / 2.0, box_size / 2.0, -box_size / 2.0, box_size / 2.0),
            aspect="equal",
            norm=norm,
            cmap=cmap,
            interpolation="nearest",
        )
        figure.colorbar(image, ax=axis, pad=0.02)
        axis.set_title(title)
        axis.set_xlabel("x' = x - L/2")
        axis.set_ylabel("y' = y - L/2")
        images.append(image)

    velocity_x_initial = velocity_x[0, :, :, z_index].T
    velocity_y_initial = velocity_y[0, :, :, z_index].T
    quiver_scale = max(speed_max / 0.12, 1e-12)
    quiver = axes[1].quiver(
        Xq,
        Yq,
        velocity_x_initial[::quiver_stride, ::quiver_stride],
        velocity_y_initial[::quiver_stride, ::quiver_stride],
        color="white",
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
        width=0.0025,
    )

    def update(frame: int):
        for image, (values, _, _, _) in zip(images, fields):
            image.set_data(values[frame, :, :, z_index].T)

        vx = velocity_x[frame, :, :, z_index].T
        vy = velocity_y[frame, :, :, z_index].T
        quiver.set_UVC(
            vx[::quiver_stride, ::quiver_stride],
            vy[::quiver_stride, ::quiver_stride],
        )
        figure.suptitle(
            "Turbulentes Anfangsmedium: "
            f"z-Schnitt, t = {time_points[frame]:.6f}"
        )
        return (*images, quiver)

    movie = FuncAnimation(
        figure,
        update,
        frames=states.shape[0],
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    print(f"Gespeichertes turbulentes Medium-GIF: {output_path}")
    print(f"Verwendete z-Ebene: z_index={z_index}")
    print(f"Geschwindigkeitsvektoren: jeder {quiver_stride}. Punkt")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Erzeuge ein GIF aus den turbulenten Medium-Snapshots."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "Praktikum_Datenzubereitung/turbulent_medium_128cells.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "Praktikum_Datenzubereitung/turbulent_medium_slices_128cells.gif"
        ),
    )
    parser.add_argument("--z-index", type=int, default=None)
    parser.add_argument("--quiver-stride", type=int, default=8)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    save_anfangsmedium_gif(
        args.input,
        args.output,
        z_index=args.z_index,
        quiver_stride=args.quiver_stride,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
