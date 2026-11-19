"""Repeat the three invalid n_H=100 TurbGen runs at a lower CFL number."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from autocvd import autocvd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/export/home/bgschwen/micromamba/envs/pde/bin/python")
OUTPUT_DIR = ROOT / "Praktikum_Datenzubereitung/mach_turbgen_solenoidal_implicit_cfl0p1_t2_20261119_retry3"
OVERLAY = Path("/tmp/fd_riemann_rerun_overlay")
ASTRONOMIX = Path("/tmp/astronomix-fd-riemann")
DRIVING_VELOCITIES = (2.0, 2.5, 3.0)


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_id(velocity: float) -> str:
    return f"nh100p0_dv{velocity:.1f}".replace(".", "p")


def command_for(velocity: float, output: Path) -> list[str]:
    return [
        str(PYTHON), "-u", "-m", "Praktikum_Datenzubereitung.anfangsmedium",
        "--num-cells", "128", "--box-size", "1", "--t-end", "2",
        "--num-snapshots", "33", "--save-snapshots", "--save-invalid",
        "--density", "1", "--pressure", "1", "--gamma", "1.6666666666666667",
        "--seed", "42", "--cfl", "0.1",
        "--physical-box-size-pc", "100",
        "--physical-hydrogen-number-density", "100",
        "--physical-temperature-k", "10000",
        "--physical-mean-molecular-weight", "0.61",
        "--physical-mass-per-hydrogen-nucleus", "1.4",
        "--forcing-mode", "turbgen", "--driving-velocity", str(velocity),
        "--forcing-k-min", "2", "--forcing-k-max", "6",
        "--forcing-k-driving", "0.6", "--forcing-amplitude-factor", "1",
        "--forcing-solenoidal-weight", "1", "--forcing-steps-per-turnover", "10",
        "--fd-riemann", "--boundary", "periodic",
        "--cooling", "--cooling-model", "schure2009_cie",
        "--cooling-method", "implicit", "--cooling-floor-temperature-k", "10000",
        "--output", str(output),
    ]


def validate_and_measure(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        states = data["states"]
        times = np.asarray(data["time_points"], dtype=np.float64)

        if states.ndim != 5:
            raise ValueError(f"Unerwartete Snapshot-Form: {states.shape}")
        layout = metadata["state_layout"]
        density_index = int(layout["density_index"])
        pressure_index = int(layout["pressure_index"])
        vx_index = int(layout["velocity_x_index"])
        vy_index = int(layout["velocity_y_index"])
        vz_index = int(layout["velocity_z_index"])
        snapshot_mach_means = []
        for snapshot in states:
            rho = snapshot[density_index]
            pressure = snapshot[pressure_index]
            vx, vy, vz = snapshot[vx_index], snapshot[vy_index], snapshot[vz_index]
            if not (np.isfinite(rho).all() and np.isfinite(pressure).all()
                    and np.isfinite(vx).all() and np.isfinite(vy).all()
                    and np.isfinite(vz).all()):
                raise ValueError("Snapshots enthalten nicht-endliche Werte.")
            if np.any(rho <= 0) or np.any(pressure <= 0):
                raise ValueError("Snapshots enthalten nicht-positive Dichte oder Druck.")
            speed = np.sqrt(vx * vx + vy * vy + vz * vz)
            sound_speed = np.sqrt(float(metadata["gamma"]) * pressure / rho)
            snapshot_mach_means.append(float(np.mean(speed / sound_speed)))

    if metadata.get("valid") is not True:
        raise ValueError("Metadaten markieren den Lauf als ungueltig.")
    if times.ndim != 1 or len(times) != len(snapshot_mach_means):
        raise ValueError("Zeitpunkte und Snapshotzahl passen nicht zusammen.")
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Zeitpunkte sind nicht endlich und streng monoton.")
    if not np.isclose(times[-1], 2.0, rtol=1e-6, atol=1e-8):
        raise ValueError(f"Endzeit ist {times[-1]}, erwartet war 2.0.")
    window = (times >= 1.5) & (times <= 2.0)
    if not np.any(window):
        raise ValueError("Keine Snapshots im Auswertungsfenster [1.5, 2.0].")
    return {
        "mach_end_mean_local": snapshot_mach_means[-1],
        "mach_mean_local_t1p5_t2": float(np.mean(np.asarray(snapshot_mach_means)[window])),
        "final_time": float(times[-1]),
        "valid": True,
    }


def main() -> int:
    manifest_path = OUTPUT_DIR / "batch_manifest.json"
    csv_path = OUTPUT_DIR / "mach_calibration.csv"
    planned = []
    for velocity in DRIVING_VELOCITIES:
        name = run_id(velocity)
        planned.append({
            "id": name,
            "driving_velocity": velocity,
            "output": str(OUTPUT_DIR / f"{name}.npz"),
            "log": str(OUTPUT_DIR / f"{name}.log"),
            "status": "pending",
        })
    collisions = [p for run in planned for p in (run["output"], run["log"])
                  if Path(p).exists()]
    if OUTPUT_DIR.exists() or manifest_path.exists() or csv_path.exists() or collisions:
        raise FileExistsError(f"Ausgabeordner oder Ergebnis existiert bereits: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sweep": "mach_turbgen_solenoidal_implicit_cfl1_t2_20261119",
        "gpu_selection": "autocvd selects one currently free GPU",
        "gpu_ids": None,
        "num_cells": [128, 128, 128],
        "hydrogen_number_density_cm3": 100.0,
        "cfl": 0.1,
        "t_start": 0.0,
        "t_end": 2.0,
        "num_snapshots": 33,
        "gamma": 5.0 / 3.0,
        "seed": 42,
        "forcing": "TurbGen solenoidal",
        "solver": "FD-Riemann",
        "cooling": "Schure 2009 CIE implicit",
        "runs": planned,
        "finished": False,
    }
    write_manifest(manifest_path, manifest)

    selected_gpu_ids = autocvd(num_gpus=1, timeout=None, interval=30, set_env=True)
    manifest["gpu_ids"] = selected_gpu_ids
    write_manifest(manifest_path, manifest)
    environment = os.environ.copy()
    extra_paths = [str(OVERLAY), str(ROOT), str(ASTRONOMIX)]
    if environment.get("PYTHONPATH"):
        extra_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(extra_paths)
    environment["JAX_PLATFORMS"] = "cuda,cpu"
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    print(f"autocvd selected GPU(s): {selected_gpu_ids}", flush=True)

    rows = []
    any_invalid = False
    for run in planned:
        print(f"Starting {run['id']} on GPU(s) {selected_gpu_ids}", flush=True)
        run["status"] = "running"
        write_manifest(manifest_path, manifest)
        log_path = Path(run["log"])
        with log_path.open("x", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command_for(run["driving_velocity"], Path(run["output"])),
                cwd=OVERLAY,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()
            run["returncode"] = process.wait()
        output_path = Path(run["output"])
        if run["returncode"] != 0 or not output_path.is_file():
            run["status"] = "simulation_failed"
            any_invalid = True
        else:
            try:
                run["measurements"] = validate_and_measure(output_path)
                run["status"] = "complete"
                rows.append({"id": run["id"], "driving_velocity": run["driving_velocity"],
                             "cfl": 0.1, **run["measurements"]})
            except Exception as error:
                run["status"] = "invalid"
                run["diagnostic"] = str(error)
                any_invalid = True
        write_manifest(manifest_path, manifest)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "id", "driving_velocity", "cfl", "mach_end_mean_local",
                "mach_mean_local_t1p5_t2", "final_time", "valid",
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"{run['id']}: {run['status']}", flush=True)

    manifest["finished"] = True
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest_path, manifest)
    print(f"Sweep done. Manifest: {manifest_path}; CSV: {csv_path}", flush=True)
    return 1 if any_invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
