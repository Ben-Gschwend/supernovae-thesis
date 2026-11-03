"""Plotte Geschwindigkeitsstatistiken des turbulenten 3D-Mediums.

Die Eingabe wird von ``anfangsmedium.py --save-snapshots`` erzeugt. Für jeden
gespeicherten Zeitpunkt werden über alle Zellen berechnet:

* ``mean_speed``: mittlerer Betrag ``<|v|>``;
* ``rms_speed``: RMS-Geschwindigkeit ``sqrt(<|v|^2>)``;
* ``mean_vector_speed``: Betrag der gemittelten Geschwindigkeit
  ``|<v>|``;
* die drei Komponenten des gemittelten Geschwindigkeitsvektors.

``<|v|>`` beschreibt die typische Bewegung einzelner Zellen. ``|<v>``
beschreibt dagegen einen gemeinsamen Nettofluss. In einem turbulenten Medium
kann die erste Größe groß und die zweite trotzdem nahe null sein.

Das Skript startet keine Simulation und benötigt keine JAX- oder GPU-Nutzung.
Es erzeugt eine PNG-Grafik und eine CSV-Datei mit den berechneten Werten.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _next_available_path(output_path: Path) -> Path:
    """Verhindere das Überschreiben einer vorhandenen Grafik."""

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


def _load_velocity_data(input_path: Path):
    with np.load(input_path, allow_pickle=False) as data:
        required = {"states", "time_points", "metadata_json"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                f"Fehlende NPZ-Felder: {missing}. "
                "Wurde das Medium mit --save-snapshots erzeugt?"
            )
        states = np.asarray(data["states"])
        time_points = np.asarray(data["time_points"], dtype=float)
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))

    if states.ndim != 5:
        raise ValueError(
            "states muss die Form (Frames, Variablen, Nx, Ny, Nz) haben; "
            f"gefunden wurde {states.shape}."
        )
    if len(time_points) != states.shape[0]:
        raise ValueError("time_points und states enthalten unterschiedlich viele Frames.")

    layout = metadata["state_layout"]
    velocity_x = states[:, int(layout["velocity_x_index"])]
    velocity_y = states[:, int(layout["velocity_y_index"])]
    velocity_z = states[:, int(layout["velocity_z_index"])]
    return time_points, velocity_x, velocity_y, velocity_z, metadata


def calculate_velocity_statistics(input_path: str | Path):
    """Berechne Geschwindigkeitsstatistiken für alle gespeicherten Zeitpunkte."""

    time_points, velocity_x, velocity_y, velocity_z, metadata = _load_velocity_data(
        Path(input_path)
    )
    speed_squared = velocity_x**2 + velocity_y**2 + velocity_z**2
    speed = np.sqrt(speed_squared)

    mean_vx = np.nanmean(velocity_x, axis=(1, 2, 3))
    mean_vy = np.nanmean(velocity_y, axis=(1, 2, 3))
    mean_vz = np.nanmean(velocity_z, axis=(1, 2, 3))
    mean_speed = np.nanmean(speed, axis=(1, 2, 3))
    rms_speed = np.sqrt(np.nanmean(speed_squared, axis=(1, 2, 3)))
    mean_vector_speed = np.sqrt(mean_vx**2 + mean_vy**2 + mean_vz**2)

    statistics = {
        "time": time_points,
        "mean_speed": mean_speed,
        "rms_speed": rms_speed,
        "mean_vx": mean_vx,
        "mean_vy": mean_vy,
        "mean_vz": mean_vz,
        "mean_vector_speed": mean_vector_speed,
    }
    return statistics, metadata


def save_velocity_statistics_plot(
    input_path: str | Path,
    output_path: str | Path,
):
    """Speichere Grafik und CSV-Datei der Geschwindigkeitsstatistiken."""

    input_path = Path(input_path)
    output_path = _next_available_path(Path(output_path))
    statistics, metadata = calculate_velocity_statistics(input_path)

    time = statistics["time"]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(
        time,
        statistics["mean_speed"],
        color="tab:blue",
        linewidth=2.0,
        label=r"Mittlerer Betrag $\langle |v| \rangle$",
    )
    axes[0].plot(
        time,
        statistics["rms_speed"],
        color="tab:orange",
        linewidth=2.0,
        label=r"RMS-Geschwindigkeit $u_{\mathrm{rms}}$",
    )
    axes[0].set_ylabel("Geschwindigkeit")
    axes[0].set_title("Typische Geschwindigkeit im 3D-Medium")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        time,
        statistics["mean_vector_speed"],
        color="black",
        linewidth=2.0,
        label=r"Nettofluss $|\langle \vec v \rangle|$",
    )
    for component, color, label in (
        ("mean_vx", "tab:red", r"$\langle v_x \rangle$"),
        ("mean_vy", "tab:green", r"$\langle v_y \rangle$"),
        ("mean_vz", "tab:purple", r"$\langle v_z \rangle$"),
    ):
        axes[1].plot(
            time,
            statistics[component],
            color=color,
            linewidth=1.3,
            label=label,
        )
    axes[1].set_xlabel("Zeit")
    axes[1].set_ylabel("Geschwindigkeit")
    axes[1].set_title("Gemittelter Vektor und möglicher Nettofluss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2)

    seed = metadata.get("random_seed", "unbekannt")
    kf = metadata.get("forcing_wavenumber", "unbekannt")
    amplitude = metadata.get("forcing_amplitude", "unbekannt")
    figure.suptitle(
        "Geschwindigkeitsstatistik des turbulenten Mediums "
        f"(Seed={seed}, k_f={kf}, Amplitude={amplitude})"
    )
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    csv_path = output_path.with_suffix(".csv")
    fieldnames = list(statistics)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(time)):
            writer.writerow(
                {name: float(values[index]) for name, values in statistics.items()}
            )

    print(f"Gespeicherte Geschwindigkeitsgrafik: {output_path}")
    print(f"Gespeicherte Statistikdaten: {csv_path}")
    return output_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plotte mittlere Geschwindigkeiten des turbulenten Mediums."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ-Datei mit gespeicherten Medium-Snapshots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "Praktikum_Datenzubereitung/turbulent_velocity_statistics.png"
        ),
    )
    args = parser.parse_args()
    save_velocity_statistics_plot(args.input, args.output)


if __name__ == "__main__":
    main()
