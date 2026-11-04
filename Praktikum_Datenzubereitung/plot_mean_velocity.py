"""Plotte die typische Geschwindigkeit relativ zur Schallgeschwindigkeit.

Die Eingabe wird von ``anfangsmedium.py --save-snapshots`` erzeugt. Für jeden
gespeicherten Zeitpunkt werden über alle Zellen berechnet:

Für jeden gespeicherten Zeitpunkt werden berechnet:

* ``mean_speed``: typische Geschwindigkeit ``<|v|>``;
* ``mean_sound_speed``: mittlere lokale Schallgeschwindigkeit
  ``<sqrt(gamma*p/rho)>``;
* ``speed_to_sound_ratio``: Quotient ``<|v|>/<c_s>``.

Der letzte Wert ist eine charakteristische Mach-Zahl des Mediums. Er ist nicht
identisch mit dem Mittelwert der lokalen Mach-Zahlen ``<|v|/c_s>``.

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


DATA_DIR = Path(__file__).resolve().parent


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
    density = states[:, int(layout["density_index"])]
    pressure = states[:, int(layout["pressure_index"])]
    velocity_x = states[:, int(layout["velocity_x_index"])]
    velocity_y = states[:, int(layout["velocity_y_index"])]
    velocity_z = states[:, int(layout["velocity_z_index"])]
    return density, pressure, velocity_x, velocity_y, velocity_z, time_points, metadata


def calculate_velocity_statistics(input_path: str | Path):
    """Berechne Geschwindigkeitsstatistiken für alle gespeicherten Zeitpunkte."""

    (
        density,
        pressure,
        velocity_x,
        velocity_y,
        velocity_z,
        time_points,
        metadata,
    ) = _load_velocity_data(
        Path(input_path)
    )
    speed_squared = velocity_x**2 + velocity_y**2 + velocity_z**2
    speed = np.sqrt(speed_squared)

    gamma = float(metadata.get("gamma", 5.0 / 3.0))
    valid = (
        np.isfinite(density)
        & np.isfinite(pressure)
        & np.isfinite(speed)
        & (density > 0.0)
        & (pressure > 0.0)
    )
    valid_speed = np.where(valid, speed, np.nan)
    sound_speed = np.full_like(pressure, np.nan, dtype=float)
    sound_speed[valid] = np.sqrt(
        gamma * pressure[valid] / density[valid]
    )

    mean_speed = np.nanmean(valid_speed, axis=(1, 2, 3))
    mean_sound_speed = np.nanmean(sound_speed, axis=(1, 2, 3))
    speed_to_sound_ratio = mean_speed / mean_sound_speed

    statistics = {
        "time": time_points,
        "mean_speed": mean_speed,
        "mean_sound_speed": mean_sound_speed,
        "speed_to_sound_ratio": speed_to_sound_ratio,
    }
    metadata = dict(metadata)
    metadata["gamma_used_for_sound_speed"] = gamma
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
        label=r"Typische Geschwindigkeit $\langle |v| \rangle$",
    )
    axes[0].plot(
        time,
        statistics["mean_sound_speed"],
        color="tab:orange",
        linewidth=2.0,
        label=r"Mittlere Schallgeschwindigkeit $\langle c_s \rangle$",
    )
    axes[0].set_ylabel("Geschwindigkeit")
    axes[0].set_title("Typische Geschwindigkeit und Schallgeschwindigkeit")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        time,
        statistics["speed_to_sound_ratio"],
        color="tab:blue",
        linewidth=2.0,
        label=r"$\langle |v| \rangle / \langle c_s \rangle$",
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Mach 1")
    axes[1].set_xlabel("Zeit")
    axes[1].set_ylabel(r"$\langle |v| \rangle / \langle c_s \rangle$")
    axes[1].set_title("Charakteristischer Geschwindigkeitsquotient")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    seed = metadata.get("random_seed", "unbekannt")
    kf = metadata.get("forcing_wavenumber", "unbekannt")
    amplitude = metadata.get("forcing_amplitude", "unbekannt")
    gamma = metadata.get("gamma_used_for_sound_speed", metadata.get("gamma", "unbekannt"))
    figure.suptitle(
        "Charakteristische Mach-Zahl des turbulenten Mediums "
        f"(Seed={seed}, k_f={kf}, Amplitude={amplitude}, gamma={gamma})"
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
        default=DATA_DIR / "turbulent_velocity_statistics.png",
    )
    args = parser.parse_args()
    save_velocity_statistics_plot(args.input, args.output)


if __name__ == "__main__":
    main()
