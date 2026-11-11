"""Kalibriere die erreichte Machzahl gegen OU-Forcing-Amplitude und n_H.

Das Skript startet ausschließlich Mediumläufe (keine Supernovae). Es
übernimmt die Medium-Defaults aus einer Trainingsdaten-JSON, überschreibt
Forcing-Amplitude und physikalische Wasserstoffdichte und misst den finalen
Zustand bei t_end und speichert alle Medium-Snapshots sowie eine gemeinsame
CSV mit den finalen Machzahlmessungen. Es erzeugt keine Plots. Vor den
Mediumläufen wählt ``autocvd`` genau eine freie GPU aus. Ist keine frei,
wartet das Skript standardmäßig unbegrenzt und prüft alle 30 Sekunden. Die
Auswahl wird an die seriell gestarteten Mediumprozesse weitergegeben.

Beispiel für einen groben Pilotlauf:

    JAX_PLATFORMS=cuda,cpu python -u -m Praktikum_Datenzubereitung.calibrate_mach_forcing \\
      --config Praktikum_Datenzubereitung/training_dataset_config.example.json \\
      --forcing-amplitudes 3 6 9 --densities 0.1 1 100 --continue-on-error

Die drei Dichten sind zunächst Pilotpunkte, keine vollständige Abdeckung des
späteren Dichtesweeps. Das primäre Ergebnis ist derselbe charakteristische
Quotient wie in plot_mean_velocity.py: <|v|>/<c_s>. Zusätzlich werden der
Mittelwert der lokalen Machzahlen und der Zellanteil mit M>1 protokolliert.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDIUM_MODULE = "Praktikum_Datenzubereitung.anfangsmedium"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Führe Medium-only-Pilotläufe aus und messe die erreichte "
            "Machzahl bei t_end."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "Praktikum_Datenzubereitung/training_dataset_config.example.json"
        ),
        help="JSON mit den zu kalibrierenden medium_defaults.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python-Interpreter für anfangsmedium.py.",
    )
    parser.add_argument(
        "--forcing-amplitudes",
        nargs="+",
        type=float,
        default=[3.0, 6.0, 9.0],
        metavar="A",
        help="Zu testende OU-Forcing-Amplituden (Pilot-Default: 3 6 9).",
    )
    parser.add_argument(
        "--densities",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 100.0],
        metavar="N_H",
        help=(
            "Zu testende physikalische mittlere Wasserstoffdichten in cm^-3 "
            "(Pilot-Default: 0.1 1 100)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Praktikum_Datenzubereitung/mach_calibration"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Befehle anzeigen, ohne Mediumläufe zu starten.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Weitere Kombinationen auch nach Prozessfehlern ausführen.",
    )
    parser.add_argument(
        "--gpu-wait-timeout",
        type=int,
        default=None,
        metavar="SEKUNDEN",
        help="Maximale Wartezeit auf eine freie GPU; Standard: unbegrenzt.",
    )
    parser.add_argument(
        "--gpu-poll-interval",
        type=int,
        default=30,
        metavar="SEKUNDEN",
        help="Abfrageintervall für autocvd; Standard: 30 Sekunden.",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not isinstance(
        config.get("medium_defaults"), dict
    ):
        raise ValueError("Die JSON benötigt ein Objekt 'medium_defaults'.")
    return config


def _cli_options(options: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            result.append(flag)
        elif isinstance(value, (str, int, float)):
            result.extend((flag, str(value)))
        else:
            raise TypeError(f"Nicht unterstützter Wert für {key}: {value!r}")
    return result


def _tag(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def _validate_values(values: list[float], label: str) -> None:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{label} müssen endliche, positive Werte sein.")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} müssen eindeutig sein.")


def _measure_final_state(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        if not {"primitive_state", "metadata_json", "states"}.issubset(data.files):
            raise ValueError(
                "NPZ benötigt primitive_state, metadata_json und gespeicherte states."
            )
        state = np.asarray(data["primitive_state"], dtype=np.float64)
        metadata_text = np.asarray(data["metadata_json"]).item()
        if isinstance(metadata_text, bytes):
            metadata_text = metadata_text.decode("utf-8")
        metadata = json.loads(str(metadata_text))
        times = np.asarray(data["time_points"], dtype=np.float64)

    if not metadata.get("snapshots_saved", False):
        raise ValueError("Metadaten markieren snapshots_saved nicht als true.")

    if state.ndim != 4:
        raise ValueError(f"Unerwartete Zustandsform: {state.shape}.")
    layout = metadata.get("state_layout", {})
    try:
        rho = state[int(layout["density_index"])]
        pressure = state[int(layout["pressure_index"])]
        vx = state[int(layout["velocity_x_index"])]
        vy = state[int(layout["velocity_y_index"])]
        vz = state[int(layout["velocity_z_index"])]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("state_layout ist unvollständig oder ungültig.") from error

    speed = np.sqrt(vx * vx + vy * vy + vz * vz)
    valid_cells = (
        np.isfinite(rho)
        & np.isfinite(pressure)
        & np.isfinite(speed)
        & (rho > 0.0)
        & (pressure > 0.0)
    )
    total_cells = int(rho.size)
    valid_count = int(valid_cells.sum())
    if valid_count != total_cells:
        raise ValueError(
            f"Finalzustand hat nur {valid_count}/{total_cells} endliche Zellen "
            "mit positiver Dichte und positivem Druck."
        )
    if metadata.get("valid") is False:
        raise ValueError("Medium-Metadaten markieren den Lauf als valid=false.")
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("time_points sind nicht endlich oder unvollständig.")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("time_points sind nicht streng ansteigend.")
    target_end = float(metadata["t_end"])
    if not math.isclose(float(times[-1]), target_end, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(
            f"Lauf erreichte t={times[-1]:.8g}, erwartet wurde t_end={target_end:.8g}."
        )

    gamma = float(metadata["gamma"])
    sound_speed = np.sqrt(gamma * pressure / rho)
    local_mach = speed / sound_speed
    mean_speed = float(speed.mean())
    mean_sound_speed = float(sound_speed.mean())
    physical_scaling = metadata.get("physical_scaling") or {}
    return {
        "forcing_amplitude": float(metadata["forcing_amplitude"]),
        "hydrogen_number_density_cm3": float(
            physical_scaling["hydrogen_number_density_cm3"]
        ),
        "characteristic_mach_mean_speed_over_mean_sound_speed": (
            mean_speed / mean_sound_speed
        ),
        "mean_local_mach": float(local_mach.mean()),
        "fraction_cells_local_mach_gt_1": float((local_mach > 1.0).mean()),
        "mean_speed_code_units": mean_speed,
        "mean_sound_speed_code_units": mean_sound_speed,
        "final_time": float(times[-1]),
        "valid_cells": valid_count,
        "total_cells": total_cells,
        "valid": True,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(command: list[str], log_path: Path, dry_run: bool) -> int:
    print("\n" + " ".join(subprocess.list2cmdline([part]) for part in command))
    if dry_run:
        return 0
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        return process.wait()


def main() -> int:
    args = _parse_args()
    config_path = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    config = _load_config(config_path)
    medium_defaults = dict(config["medium_defaults"])
    _validate_values(args.forcing_amplitudes, "Forcing-Amplituden")
    _validate_values(args.densities, "Dichten")
    if args.gpu_poll_interval < 1:
        raise ValueError("--gpu-poll-interval muss mindestens 1 Sekunde sein.")
    if args.gpu_wait_timeout is not None and args.gpu_wait_timeout < 1:
        raise ValueError("--gpu-wait-timeout muss mindestens 1 Sekunde sein.")

    combinations = [
        (density, amplitude)
        for density in args.densities
        for amplitude in args.forcing_amplitudes
    ]
    print(f"Geplante Medium-Pilotläufe: {len(combinations)}")
    manifest_path = output_dir / "mach_calibration_manifest.json"
    csv_path = output_dir / "mach_calibration.csv"
    manifest: dict[str, Any] = {
        "config": str(config_path),
        "python": str(args.python),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "mean(|v|) / mean(sqrt(gamma*p/rho)) at t_end",
        "save_snapshots": True,
        "num_snapshots": medium_defaults.get("num_snapshots"),
        "experiments": [],
    }
    rows: list[dict[str, Any]] = []
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists() or csv_path.exists():
            raise FileExistsError(
                f"Kalibrierungsmanifest oder CSV existiert bereits in {output_dir}."
            )
        collisions = []
        for density, amplitude in combinations:
            run_id = f"nh{_tag(density)}_amp{_tag(amplitude)}"
            for candidate in (
                output_dir / f"medium_{run_id}.npz",
                output_dir / f"medium_{run_id}.log",
            ):
                if candidate.exists():
                    collisions.append(str(candidate))
        if collisions:
            raise FileExistsError(
                "Vorhandene Kalibrierungsausgaben werden nicht überschrieben: "
                + ", ".join(collisions)
            )

        try:
            from autocvd import autocvd
        except ImportError as error:
            raise RuntimeError(
                "autocvd fehlt im aktiven Python-Environment. Im pde-Environment "
                "installieren mit: python -m pip install autocvd==0.2.1"
            ) from error

        print(
            "Warte mit autocvd auf genau eine freie GPU; "
            f"Abfrageintervall: {args.gpu_poll_interval} s."
        )
        selected_gpu_ids = autocvd(
            num_gpus=1,
            timeout=args.gpu_wait_timeout,
            interval=args.gpu_poll_interval,
            set_env=True,
        )
        manifest["selected_physical_gpu_ids"] = selected_gpu_ids
        manifest["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        manifest["cuda_device_order"] = os.environ.get("CUDA_DEVICE_ORDER")
        print(
            "autocvd hat GPU "
            + ", ".join(str(gpu_id) for gpu_id in selected_gpu_ids)
            + " ausgewählt; die Mediumläufe starten jetzt seriell."
        )

    any_failure = False
    for index, (density, amplitude) in enumerate(combinations, start=1):
        run_id = f"nh{_tag(density)}_amp{_tag(amplitude)}"
        output_path = output_dir / f"medium_{run_id}.npz"
        log_path = output_dir / f"medium_{run_id}.log"
        if not args.dry_run and (output_path.exists() or log_path.exists()):
            raise FileExistsError(
                f"Ausgabe existiert bereits; nichts wird überschrieben: {output_path}"
            )
        options = dict(medium_defaults)
        options["physical_hydrogen_number_density"] = density
        options["forcing_amplitude"] = amplitude
        # Für die spätere zeitliche Velocity-Auswertung müssen die vollständigen
        # Zustände gespeichert werden; time_points allein reichen nicht aus.
        options["save_snapshots"] = True
        # Rohdaten eines numerisch ungültigen Laufs für die Diagnose behalten;
        # solche Dateien werden unten nicht als Messpunkte akzeptiert.
        options["save_invalid"] = True
        options["output"] = str(output_path)
        command = [
            str(args.python),
            "-u",
            "-m",
            MEDIUM_MODULE,
            *_cli_options(options),
        ]
        experiment: dict[str, Any] = {
            "id": run_id,
            "density_cm3": density,
            "forcing_amplitude": amplitude,
            "medium_output": str(output_path),
            "log": str(log_path),
            "command": command,
            "status": "planned" if args.dry_run else "running",
        }
        manifest["experiments"].append(experiment)
        if not args.dry_run:
            _write_json(manifest_path, manifest)
        return_code = _run(command, log_path, args.dry_run)
        experiment["returncode"] = None if args.dry_run else return_code
        if args.dry_run:
            experiment["status"] = "planned"
        elif return_code != 0:
            experiment["status"] = "simulation_failed"
            experiment["diagnostic"] = f"Medium-Prozess Exit-Code {return_code}."
            any_failure = True
        else:
            try:
                measurements = _measure_final_state(output_path)
                experiment["status"] = "complete"
                experiment["measurements"] = measurements
                rows.append({"run_id": run_id, **measurements})
                _write_csv(csv_path, rows)
            except Exception as error:
                experiment["status"] = "invalid"
                experiment["diagnostic"] = str(error)
                any_failure = True
        if not args.dry_run:
            _write_json(manifest_path, manifest)
        if any_failure and not args.continue_on_error:
            break

    if not args.dry_run:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)
        print(f"Manifest: {manifest_path}")
        if csv_path.exists():
            print(f"Messwerte: {csv_path}")
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
