"""Führe mehrere Medium-/Supernova-Läufe seriell als Trainingsdatensatz aus.

Das Skript startet bewusst keine parallelen Jobs. Pro Experiment wird zuerst
das turbulente Medium erzeugt und danach die Supernova im gespeicherten Medium
simuliert. Alle Befehle, Logs, Rückgabecodes und NPZ-Pfade werden in einem
Manifest dokumentiert.

GPU-Auswahl erfolgt weiterhin ausschließlich über die Umgebung, zum Beispiel:

    CUDA_VISIBLE_DEVICES=2 python -u \
        -m Praktikum_Datenzubereitung.run_training_dataset \
        --config Praktikum_Datenzubereitung/training_dataset_config.example.json

Vor einem echten Lauf sollte zunächst ``--dry-run`` verwendet werden.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDIUM_MODULE = "Praktikum_Datenzubereitung.anfangsmedium"
SUPERNOVA_MODULE = "Praktikum_Datenzubereitung.supernovae_turbolenz"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Erzeuge einen Trainingsdatensatz durch serielle Medium- und "
            "Supernova-Simulationen."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python-Interpreter für die untergeordneten Simulationsmodule.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur Befehle ausgeben; keine Simulation starten.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Nach einem fehlgeschlagenen Experiment mit dem nächsten fortfahren.",
    )
    parser.add_argument(
        "--allow-invalid-medium",
        action="store_true",
        help="Supernova auch starten, wenn das Medium als ungültig markiert ist.",
    )
    parser.add_argument(
        "--sweep-energy-e51",
        nargs="+",
        type=float,
        metavar="E51",
        help=(
            "Erzeuge statt der config-runs einen Sweep über diese Faktoren "
            "von 10^51 erg, z. B. --sweep-energy-e51 0.5 1 2."
        ),
    )
    parser.add_argument(
        "--sweep-nh-range",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "STEP"),
        help=(
            "Sweep der mittleren Wasserstoffzahlendichte in cm^-3, "
            "z. B. --sweep-nh-range 0.1 100 5. Der STOP-Wert wird immer "
            "aufgenommen, auch wenn dafür der letzte Schritt kürzer ist."
        ),
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Die Konfigurationsdatei muss ein JSON-Objekt enthalten.")
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Die Konfiguration benötigt eine nichtleere Liste 'runs'.")
    return config


def _cli_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def _options_to_cli(options: dict[str, Any]) -> list[str]:
    arguments: list[str] = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        flag = _cli_flag(key)
        if value is True:
            arguments.append(flag)
        elif isinstance(value, (str, int, float)):
            arguments.extend([flag, str(value)])
        else:
            raise TypeError(
                f"Option '{key}' hat einen nicht unterstützten Werttyp: "
                f"{type(value).__name__}."
            )
    return arguments


def _merged_options(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(defaults)
    result.update(overrides)
    return result


def _decimal_range(start: float, stop: float, step: float) -> list[float]:
    """Erzeuge eine inklusive Dezimalreihe ohne kumulative Float-Fehler."""

    if not all(math.isfinite(value) for value in (start, stop, step)):
        raise ValueError("Sweep-Bereich muss aus endlichen Zahlen bestehen.")
    first, last, increment = map(lambda value: Decimal(str(value)), (start, stop, step))
    if increment <= 0 or last < first:
        raise ValueError("Sweep-Bereich benötigt STOP >= START und STEP > 0.")
    values: list[Decimal] = []
    current = first
    while current <= last:
        values.append(current)
        current += increment
    if values[-1] != last:
        values.append(last)
    return [float(value) for value in values]


def _number_tag(value: float) -> str:
    return format(Decimal(str(value)).normalize(), "f").replace("-", "m").replace(".", "p")


def _sweep_runs(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Ersetze bei angefordertem Sweep die JSON-runs durch das Kreuzprodukt."""

    if args.sweep_energy_e51 is None and args.sweep_nh_range is None:
        return config["runs"]

    base_medium = dict(config.get("medium_defaults", {}))
    base_supernova = dict(config.get("supernova_defaults", {}))
    base_run = config["runs"][0] if config["runs"] else {}
    base_medium.update(base_run.get("medium", {}))
    base_supernova.update(base_run.get("supernova", {}))

    energies = args.sweep_energy_e51
    if energies is None:
        energy_erg = float(base_supernova.get("physical_explosion_energy_erg", 1.0e51))
        energies = [energy_erg / 1.0e51]
    if any(not math.isfinite(value) or value <= 0.0 for value in energies):
        raise ValueError("Alle Supernovaenergien müssen positiv sein.")
    if len(set(energies)) != len(energies):
        raise ValueError("Die Energiewerte im Sweep müssen eindeutig sein.")

    if args.sweep_nh_range is None:
        densities = [float(base_medium.get("physical_hydrogen_number_density", 1.0))]
    else:
        densities = _decimal_range(*args.sweep_nh_range)
    if any(not math.isfinite(value) or value <= 0.0 for value in densities):
        raise ValueError("Alle Wasserstoffzahlendichten müssen positiv sein.")

    generated: list[dict[str, Any]] = []
    for density in densities:
        for energy in energies:
            generated.append(
                {
                    "id": f"nh{_number_tag(density)}_e{_number_tag(energy)}e51",
                    "medium": {
                        **base_medium,
                        "physical_hydrogen_number_density": density,
                    },
                    "supernova": {
                        **base_supernova,
                        "physical_explosion_energy_erg": energy * 1.0e51,
                    },
                }
            )
    return generated


def _relative_or_absolute(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _experiment_paths(output_dir: Path, run_id: str, run: dict[str, Any]) -> dict[str, Path]:
    medium_name = run.get("medium_output", f"turbulent_medium_{run_id}.npz")
    supernova_name = run.get("supernova_output", f"supernova_{run_id}.npz")
    log_name = run.get("log_output", f"training_{run_id}.log")
    return {
        "medium": output_dir / str(medium_name),
        "supernova": output_dir / str(supernova_name),
        "log": output_dir / str(log_name),
    }


def _metadata_from_npz(path: Path) -> dict[str, Any]:
    """Lese astronomix-Metadaten aus dem tatsächlich verwendeten NPZ-Feld."""

    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        metadata_key = next(
            (key for key in ("metadata_json", "metadata") if key in data), None
        )
        if metadata_key is None:
            raise ValueError("NPZ enthält weder 'metadata_json' noch 'metadata'.")
        value = data[metadata_key]
        if getattr(value, "shape", ()) == ():
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
    raise ValueError("NPZ-Metadaten liegen nicht als JSON-Text vor.")


def _medium_is_usable(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "Mediumdatei fehlt nach erfolgreichem Prozess."
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(path)
            if "primitive_state" not in data:
                return False, "Medium-NPZ enthält kein primitive_state-Feld."
            state = data["primitive_state"]
    except Exception as error:  # pragma: no cover - Diagnosepfad
        return False, f"Metadaten konnten nicht gelesen werden: {error}"
    if metadata.get("valid") is not True:
        return False, "Medium hat kein explizites valid=true in metadata_json."
    if state.ndim != 4 or state.shape[0] < 5:
        return False, f"Unerwartete primitive_state-Form: {state.shape}."
    if not np.isfinite(state).all():
        return False, "Medium enthält nicht-endliche Zustandswerte."
    if np.any(state[0] <= 0.0) or np.any(state[4] <= 0.0):
        return False, "Medium enthält nichtpositive Dichte oder nichtpositiven Druck."
    return True, "Medium-Metadaten, Endlichkeit und positive Dichte/Drücke bestätigt."


def _supernova_is_usable(path: Path) -> tuple[bool, str]:
    """Prüfe NPZ-Metadaten und gespeicherte Schnitte vor Trainingsfreigabe."""
    if not path.exists():
        return False, "Supernova-NPZ fehlt trotz Exit-Code 0."
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(path)
            required = {"time_points", "density", "pressure"}
            missing = sorted(required.difference(data.files))
            if missing:
                return False, f"Supernova-NPZ fehlen Felder: {missing}."
            times = data["time_points"]
            density = data["density"]
            pressure = data["pressure"]
            if metadata.get("valid") is not True:
                diagnostics = metadata.get("validity_diagnostics", {})
                return False, (
                    "Supernova-Metadaten melden nicht valid=true: "
                    f"{diagnostics or metadata.get('valid', 'Feld fehlt')}."
                )
            if not (np.isfinite(times).all() and np.isfinite(density).all()
                    and np.isfinite(pressure).all()):
                return False, "Supernova-Schnitte oder Zeitpunkte sind nicht endlich."
            if np.any(density <= 0.0) or np.any(pressure <= 0.0):
                return False, "Supernova-Schnitte enthalten nichtpositive Dichte oder Drücke."
            if times.ndim != 1 or times.size < 2 or not np.all(np.diff(times) > 0.0):
                return False, "Supernova-Zeitpunkte sind nicht streng ansteigend."
    except Exception as error:  # pragma: no cover - Diagnosepfad
        return False, f"Supernova-NPZ konnte nicht validiert werden: {error}"
    return True, "Supernova-Metadaten, Schnitte und Zeiten bestanden die Prüfung."


def _format_command(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def _run_command(command: list[str], log_path: Path, dry_run: bool, label: str) -> int:
    print(f"\n[{label}] {_format_command(command)}")
    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n\n[{datetime.now(timezone.utc).isoformat()}] {label}\n")
        log_handle.write(_format_command(command) + "\n")
        log_handle.flush()
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


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    args = _parse_args()
    config_path = _relative_or_absolute(args.config)
    config = _load_config(config_path)
    output_dir = _relative_or_absolute(
        config.get("output_dir", "Praktikum_Datenzubereitung/training_data")
    )
    manifest_path = output_dir / str(config.get("manifest", "manifest.json"))

    medium_defaults = dict(config.get("medium_defaults", {}))
    supernova_defaults = dict(config.get("supernova_defaults", {}))
    manifest: dict[str, Any] = {
        "config": str(config_path),
        "project_root": str(PROJECT_ROOT),
        "python": str(args.python),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiments": [],
    }

    runs = _sweep_runs(args, config)
    print(f"Geplante Experimente: {len(runs)}")
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{index - 1}] muss ein JSON-Objekt sein.")
        run_id = str(run.get("id", f"run_{index:04d}"))
        if not SAFE_ID.fullmatch(run_id):
            raise ValueError(
                f"Ungültige Experiment-ID '{run_id}'. Erlaubt sind Buchstaben, "
                "Ziffern, Punkt, Unterstrich und Bindestrich."
            )

        paths = _experiment_paths(output_dir, run_id, run)
        if not args.dry_run:
            existing = [
                str(paths[key])
                for key in ("medium", "supernova")
                if paths[key].exists()
            ]
            if existing:
                raise FileExistsError(
                    "Abbruch, um vorhandene Simulationsergebnisse nicht zu "
                    f"überschreiben: {', '.join(existing)}"
                )
        paths["medium"].parent.mkdir(parents=True, exist_ok=True)
        medium_options = _merged_options(medium_defaults, run.get("medium", {}))
        supernova_options = _merged_options(supernova_defaults, run.get("supernova", {}))
        medium_options["output"] = str(paths["medium"])
        supernova_options["input"] = str(paths["medium"])
        supernova_options["output"] = str(paths["supernova"])

        medium_command = [
            str(args.python),
            "-u",
            "-m",
            MEDIUM_MODULE,
            *_options_to_cli(medium_options),
        ]
        supernova_command = [
            str(args.python),
            "-u",
            "-m",
            SUPERNOVA_MODULE,
            *_options_to_cli(supernova_options),
        ]

        record: dict[str, Any] = {
            "id": run_id,
            "index": index,
            "medium_output": str(paths["medium"]),
            "supernova_output": str(paths["supernova"]),
            "medium_command": medium_command,
            "supernova_command": supernova_command,
            "status": "planned" if args.dry_run else "running",
        }
        manifest["experiments"].append(record)
        _write_manifest(manifest_path, manifest)

        medium_code = _run_command(
            medium_command,
            paths["log"],
            args.dry_run,
            f"{run_id}: Phase 1 Medium",
        )
        if args.dry_run:
            record["medium_returncode"] = None
            record["medium_validation"] = "Dry-Run; Medium wurde nicht erzeugt."
            record["status"] = "planned"
            medium_ok = True
        elif medium_code != 0:
            record["medium_returncode"] = medium_code
            record["status"] = "medium_failed"
            _write_manifest(manifest_path, manifest)
            if not args.continue_on_error:
                return medium_code
            continue
        else:
            record["medium_returncode"] = medium_code
            medium_ok, medium_message = _medium_is_usable(paths["medium"])
            record["medium_validation"] = medium_message
        if not medium_ok and not args.allow_invalid_medium:
            record["status"] = "medium_invalid_supernova_skipped"
            _write_manifest(manifest_path, manifest)
            if not args.continue_on_error:
                return 2
            continue

        supernova_code = _run_command(
            supernova_command,
            paths["log"],
            args.dry_run,
            f"{run_id}: Phase 2 Supernova",
        )
        record["supernova_returncode"] = supernova_code
        if args.dry_run:
            record["supernova_validation"] = "Dry-Run; Supernova wurde nicht erzeugt."
            record["status"] = "planned"
        elif supernova_code != 0:
            record["status"] = "supernova_failed"
        else:
            supernova_ok, supernova_message = _supernova_is_usable(paths["supernova"])
            record["supernova_validation"] = supernova_message
            if not supernova_ok:
                record["status"] = "supernova_invalid"
            elif not medium_ok:
                record["status"] = "complete_from_invalid_medium"
            else:
                record["status"] = "complete"
        _write_manifest(manifest_path, manifest)
        if record["status"] in {"supernova_failed", "supernova_invalid"}:
            if not args.continue_on_error:
                return supernova_code if supernova_code != 0 else 3

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)
    print(f"\nManifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
