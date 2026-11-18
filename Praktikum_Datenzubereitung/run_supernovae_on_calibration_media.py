"""Run Supernova phase 2 serially on valid Mach-calibration media.

The input media are reused as-is. ``autocvd`` selects one free physical GPU,
then all runs execute serially on that GPU. Supernova outputs are the central
2D slices written by ``supernovae_turbolenz.py``; the integration remains 3D.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_training_dataset import _medium_is_usable, _options_to_cli, _supernova_is_usable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = Path(
    "Praktikum_Datenzubereitung/mach_calibration_snapshots_compgpu13_retry3/"
    "mach_calibration_manifest.json"
)
DEFAULT_CONFIG = Path(
    "Praktikum_Datenzubereitung/training_dataset_config.example.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "Praktikum_Datenzubereitung/mach_calibration_snapshots_compgpu13_retry3/"
    "supernova_phase2_10valid"
)
SUPERNOVA_MODULE = "Praktikum_Datenzubereitung.supernovae_turbolenz"


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simuliere seriell Supernovae in den gültigen Medien eines "
            "Machzahl-Kalibrierungslaufs."
        )
    )
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gpu-poll-interval",
        type=int,
        default=30,
        help="Sekunden zwischen autocvd-Abfragen; Standard: 30.",
    )
    parser.add_argument(
        "--gpu-wait-timeout",
        type=int,
        default=None,
        help="Maximale Wartezeit auf eine freie GPU; Standard: unbegrenzt.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON-Datei enthält kein Objekt: {path}")
    return value


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _run_one(command: list[str], log_path: Path) -> int:
    print(f"\nBefehl: {_format_command(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "cuda,cpu"),
                "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
                    "XLA_PYTHON_CLIENT_PREALLOCATE", "false"
                ),
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
        return process.wait()


def main() -> int:
    args = _parse_args()
    if args.gpu_poll_interval < 1:
        raise ValueError("--gpu-poll-interval muss mindestens 1 Sekunde sein.")
    if args.gpu_wait_timeout is not None and args.gpu_wait_timeout < 1:
        raise ValueError("--gpu-wait-timeout muss mindestens 1 Sekunde sein.")

    source_path = _project_path(args.source_manifest)
    config_path = _project_path(args.config)
    output_dir = _project_path(args.output_dir)
    source_manifest = _read_json(source_path)
    config = _read_json(config_path)
    experiments = source_manifest.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("Quellmanifest enthält keine experiments-Liste.")

    valid_media = [item for item in experiments if item.get("status") == "complete"]
    if len(valid_media) != 10:
        raise ValueError(
            f"Erwartete genau 10 gültige Medien, gefunden: {len(valid_media)}."
        )

    supernova_defaults = config.get("supernova_defaults")
    if not isinstance(supernova_defaults, dict) or not supernova_defaults:
        raise ValueError("Konfiguration benötigt nichtleere supernova_defaults.")

    runs: list[dict[str, Any]] = []
    for item in valid_media:
        run_id = str(item.get("id", ""))
        medium_path = _project_path(item.get("medium_output", ""))
        if not run_id or not medium_path.is_file():
            raise FileNotFoundError(
                f"Gültiges Medium fehlt für {run_id or '<ohne id>'}: {medium_path}"
            )
        usable, diagnostic = _medium_is_usable(medium_path)
        if not usable:
            raise ValueError(f"Medium {run_id} besteht Eingabeprüfung nicht: {diagnostic}")

        output_path = output_dir / f"supernova_{run_id}.npz"
        log_path = output_dir / f"supernova_{run_id}.log"
        command_options = dict(supernova_defaults)
        command_options["input"] = str(medium_path)
        command_options["output"] = str(output_path)
        command = [
            str(args.python),
            "-u",
            "-m",
            SUPERNOVA_MODULE,
            *_options_to_cli(command_options),
        ]
        runs.append(
            {
                "id": run_id,
                "medium_input": str(medium_path),
                "medium_validation": diagnostic,
                "supernova_output": str(output_path),
                "log": str(log_path),
                "command": command,
                "status": "planned" if args.dry_run else "pending",
            }
        )

    manifest_path = output_dir / "supernova_batch_manifest.json"
    collisions = [str(path) for path in [manifest_path] if path.exists()]
    for run in runs:
        for key in ("supernova_output", "log"):
            path = Path(run[key])
            if path.exists():
                collisions.append(str(path))
    if collisions:
        raise FileExistsError(
            "Vorhandene Supernova-Ausgaben werden nicht überschrieben: "
            + ", ".join(collisions)
        )

    print(f"Geplante Supernova-Läufe: {len(runs)} (seriell)")
    if args.dry_run:
        for run in runs:
            print(f"\n[{run['id']}] {_format_command(run['command'])}")
        return 0

    try:
        from autocvd import autocvd
    except ImportError as error:
        raise RuntimeError(
            "autocvd fehlt im aktiven Environment; benötigt wird autocvd==0.2.1."
        ) from error

    print(
        "Warte mit autocvd auf genau eine freie GPU; "
        f"Abfrageintervall: {args.gpu_poll_interval} s.",
        flush=True,
    )
    selected_gpu_ids = autocvd(
        num_gpus=1,
        timeout=args.gpu_wait_timeout,
        interval=args.gpu_poll_interval,
        set_env=True,
    )
    print(f"autocvd hat GPU {selected_gpu_ids} ausgewählt.", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source_manifest": str(source_path),
        "config": str(config_path),
        "selected_physical_gpu_ids": selected_gpu_ids,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "serial",
        "experiments": runs,
    }
    _write_manifest(manifest_path, manifest)

    any_invalid = False
    for index, run in enumerate(runs, start=1):
        print(f"\n=== Lauf {index}/{len(runs)}: {run['id']} ===", flush=True)
        run["status"] = "running"
        run["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        return_code = _run_one(run["command"], Path(run["log"]))
        run["returncode"] = return_code
        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if return_code != 0:
            run["status"] = "simulation_failed"
            run["diagnostic"] = f"Supernova-Prozess Exit-Code {return_code}."
            any_invalid = True
        else:
            usable, diagnostic = _supernova_is_usable(Path(run["supernova_output"]))
            run["status"] = "complete" if usable else "invalid"
            run["validation"] = diagnostic
            any_invalid |= not usable
        _write_manifest(manifest_path, manifest)

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)
    print(f"\nManifest: {manifest_path}")
    return 1 if any_invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
