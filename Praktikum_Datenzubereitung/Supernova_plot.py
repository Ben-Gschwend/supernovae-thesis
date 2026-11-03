"""Animation der gespeicherten astronomix-Snapshots.

Dieses Modul startet keine Simulation. Es erwartet ein SnapshotData-Objekt,
das von astronomix mit ``return_snapshots=True`` und
``SnapshotSettings(return_states=True)`` erzeugt wurde. 2D-Snapshots werden
direkt dargestellt; bei 3D-Snapshots wird die zentrale z-Schnittfläche gezeigt.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def _next_available_path(output_path: Path) -> Path:
    """Gib einen noch nicht belegten Ausgabepfad zurück."""

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


def animate_snapshots(
    snapshots,
    registered_variables,
    field: str = "pressure",
    output_path: str | Path | None = None,
    interval: int = 100,
):
    """Erzeuge eine Animation eines Feldes aus astronomix-Snapshots.

    Parameters
    ----------
    snapshots:
        Ergebnis von ``time_integration`` bei aktivierten Snapshots.
    registered_variables:
        Ergebnis von ``get_registered_variables(config)``.
    field:
        Zu animierendes Feld: ``"density"`` oder ``"pressure"``.
        Bei 3D-Snapshots wird die zentrale z-Schnittfläche dargestellt.
    output_path:
        Optionaler Dateiname, z. B. ``"supernova_pressure.gif"``.
        Existiert der Dateiname bereits, wird automatisch ein freier Name mit
        fortlaufender Nummer verwendet, z. B. ``"supernova_pressure_1.gif"``.
        Ohne Dateinamen wird die Animation nur angezeigt.
    interval:
        Zeit zwischen zwei Bildern in Millisekunden.
    """

    if field == "density":
        field_index = registered_variables.density_index
        label = "Dichte"
    elif field == "pressure":
        field_index = registered_variables.pressure_index
        label = "Druck"
    else:
        raise ValueError("field muss 'density' oder 'pressure' sein.")

    states = np.asarray(snapshots.states)
    time_points = np.asarray(snapshots.time_points)
    field_values = states[:, field_index, ...]

    if field_values.ndim == 4:
        # 3D-Feld: Für die 2D-Animation die zentrale z-Ebene verwenden.
        z_index = field_values.shape[-1] // 2
        field_values = field_values[:, :, :, z_index]
        slice_description = f", z-Schnitt {z_index}"
    elif field_values.ndim == 3:
        # 2D-Feld: direkt animieren.
        slice_description = ""
    else:
        raise ValueError(
            "Diese Funktion erwartet 2D- oder 3D-Snapshots. Nach Auswahl "
            "des Feldes muss die Form (Anzahl_Snapshots, Nx, Ny) oder "
            "(Anzahl_Snapshots, Nx, Ny, Nz) vorliegen."
        )

    if len(time_points) != len(field_values):
        raise ValueError(
            "Anzahl der Zeitpunkte und Anzahl der Feld-Snapshots stimmen nicht überein."
        )

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(
        field_values[0].T,
        origin="lower",
        cmap="viridis",
        vmin=np.nanmin(field_values),
        vmax=np.nanmax(field_values),
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(label)

    ax.set_xlabel("x-Zelle")
    ax.set_ylabel("y-Zelle")

    def update(frame: int):
        image.set_array(field_values[frame].T)
        ax.set_title(
            f"{label}{slice_description}, t = {time_points[frame]:.4f}"
        )
        return (image,)

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(field_values),
        interval=interval,
        # Der Titel wird in jedem Frame aktualisiert. Mit blit=True würde er
        # ohne Rückgabe eines eigenen Text-Artists häufig bei t=0 bleiben.
        blit=False,
    )

    if output_path is not None:
        output_path = _next_available_path(Path(output_path))
        print(f"Speichere Animation nach: {output_path}")
        if output_path.suffix.lower() == ".gif":
            movie.save(output_path, writer=animation.PillowWriter(fps=1000 / interval))
        elif output_path.suffix.lower() == ".mp4":
            movie.save(output_path, writer=animation.FFMpegWriter(fps=1000 / interval))
        else:
            raise ValueError("output_path muss auf .gif oder .mp4 enden.")

    plt.show()
    return movie
