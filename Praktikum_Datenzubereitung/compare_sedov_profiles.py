"""Vergleiche radiale 1D- und 3D-Sedov-Profile bei einem gemeinsamen Zeitpunkt."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _load_profile(path: Path, target_time: float):
    if not path.exists():
        candidates = sorted(
            path.parent.glob(f"{path.stem}_*.npz"),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            path = candidates[0]
            print(f"Verwende zuletzt erzeugte Profildatei: {path}")
        else:
            raise FileNotFoundError(
                f"Profildatei nicht gefunden: {path}. "
                "Führe beide Simulationen nach der Profilspeicherung erneut aus."
            )

    with np.load(path) as data:
        required = {"radius", "density", "velocity", "pressure", "time_points"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} fehlen die Felder: {sorted(missing)}")

        radius = np.asarray(data["radius"], dtype=float)
        time_points = np.asarray(data["time_points"], dtype=float)
        frame = int(np.argmin(np.abs(time_points - target_time)))
        return {
            "radius": radius,
            "density": np.asarray(data["density"][frame], dtype=float),
            "velocity": np.asarray(data["velocity"][frame], dtype=float),
            "pressure": np.asarray(data["pressure"][frame], dtype=float),
            "time": float(time_points[frame]),
        }


def _load_profile_series(path: Path):
    """Lade alle gespeicherten Zeitpunkte einer Profildatei."""

    if not path.exists():
        candidates = sorted(
            path.parent.glob(f"{path.stem}_*.npz"),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            path = candidates[0]
            print(f"Verwende zuletzt erzeugte Profildatei: {path}")
        else:
            raise FileNotFoundError(
                f"Profildatei nicht gefunden: {path}. "
                "Führe beide Simulationen nach der Profilspeicherung erneut aus."
            )

    with np.load(path) as data:
        required = {"radius", "density", "velocity", "pressure", "time_points"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} fehlen die Felder: {sorted(missing)}")
        result = {
            "radius": np.asarray(data["radius"], dtype=float),
            "density": np.asarray(data["density"], dtype=float),
            "velocity": np.asarray(data["velocity"], dtype=float),
            "pressure": np.asarray(data["pressure"], dtype=float),
            "time_points": np.asarray(data["time_points"], dtype=float),
        }
        axis_fields = {
            "axis_x",
            "axis_radius",
            "axis_density",
            "axis_velocity",
            "axis_pressure",
        }
        if axis_fields.issubset(data.files):
            result.update(
                {
                    "axis_x": np.asarray(data["axis_x"], dtype=float),
                    "axis_radius": np.asarray(data["axis_radius"], dtype=float),
                    "axis_density": np.asarray(data["axis_density"], dtype=float),
                    "axis_velocity": np.asarray(data["axis_velocity"], dtype=float),
                    "axis_pressure": np.asarray(data["axis_pressure"], dtype=float),
                }
            )
        cell_fields = {
            "cell_radius",
            "cell_density",
            "cell_velocity",
            "cell_pressure",
        }
        if cell_fields.issubset(data.files):
            result.update(
                {
                    "cell_radius": np.asarray(data["cell_radius"], dtype=float),
                    "cell_density": np.asarray(data["cell_density"], dtype=float),
                    "cell_velocity": np.asarray(data["cell_velocity"], dtype=float),
                    "cell_pressure": np.asarray(data["cell_pressure"], dtype=float),
                }
            )
        return result


def _common_profiles(one_d, three_d):
    radius_1d = one_d["radius"]
    radius_3d = three_d["radius"]
    if radius_1d.ndim != 1 or radius_3d.ndim != 1:
        raise ValueError("Die Radien müssen eindimensionale Arrays sein.")

    common_max = min(float(np.max(radius_1d)), float(np.max(radius_3d)))
    common_mask = radius_3d <= common_max
    fields_1d = {}
    fields_3d = {}
    valid_common = common_mask.copy()
    valid_1d_by_name = {}
    for name in ("density", "velocity", "pressure"):
        values_1d = one_d[name]
        values_3d = three_d[name]
        valid_1d = np.isfinite(radius_1d) & np.isfinite(values_1d)
        valid_3d = common_mask & np.isfinite(values_3d)
        if name == "pressure":
            valid_1d &= values_1d > 0.0
            valid_3d &= values_3d > 0.0

        if np.count_nonzero(valid_1d) < 2 or not np.any(valid_3d):
            raise ValueError(f"Keine ausreichenden gemeinsamen Werte für {name}.")
        valid_common &= valid_3d
        valid_1d_by_name[name] = valid_1d

    common_radius = radius_3d[valid_common]
    for name in ("density", "velocity", "pressure"):
        values_1d = one_d[name]
        values_3d = three_d[name]
        fields_1d[name] = np.interp(
            common_radius,
            radius_1d[valid_1d_by_name[name]],
            values_1d[valid_1d_by_name[name]],
        )
        fields_3d[name] = values_3d[valid_common]

    return common_radius, fields_1d, fields_3d


def compare_profiles(
    one_d_path: Path,
    three_d_path: Path,
    *,
    target_time: float = 0.1,
    output_path: Path = Path(
        "Praktikum_Datenzubereitung/sedov_1d_3d_comparison_t010.png"
    ),
):
    one_d = _load_profile(one_d_path, target_time)
    three_d = _load_profile(three_d_path, target_time)
    radius, fields_1d, fields_3d = _common_profiles(one_d, three_d)

    if abs(one_d["time"] - three_d["time"]) > 1e-6:
        print(
            "Warnung: Die nächstliegenden Zeitpunkte unterscheiden sich: "
            f"1D={one_d['time']:.8f}, 3D={three_d['time']:.8f}."
        )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    specifications = (
        ("density", "Dichte", r"$\rho$"),
        ("velocity", "Geschwindigkeit", r"$|v|$"),
        ("pressure", "Druck", "p"),
    )

    for axis, (name, title, ylabel) in zip(axes, specifications):
        values_1d = fields_1d[name]
        values_3d = fields_3d[name]
        axis.plot(radius, values_1d, color="red", linewidth=2.0, label="1D")
        axis.plot(radius, values_3d, color="blue", linewidth=2.0, label="3D")
        axis.set_title(title)
        axis.set_xlabel("Radius r")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)

        finite_values = np.concatenate((values_1d, values_3d))
        if name == "pressure":
            axis.set_yscale("log")
            positive_values = finite_values[finite_values > 0.0]
            axis.set_ylim(
                float(np.min(positive_values)),
                1.15 * float(np.max(positive_values)),
            )
        else:
            axis.set_ylim(0.0, max(1.05 * float(np.max(finite_values)), 1e-12))
        axis.legend()

    figure.suptitle(
        "Sedov-Profilvergleich: "
        f"1D t={one_d['time']:.4f}, 3D t={three_d['time']:.4f}"
    )
    figure.tight_layout()

    output_path = _next_available_path(Path(output_path))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Gespeicherter 1D/3D-Profilvergleich: {output_path}")
    return output_path


def save_comparison_gif(
    one_d_path: Path,
    three_d_path: Path,
    *,
    output_path: Path = Path(
        "Praktikum_Datenzubereitung/sedov_1d_3d_comparison.gif"
    ),
    interval: int = 100,
    fps: int = 10,
):
    """Erzeuge ein animiertes 1D/3D-Profilvergleichs-GIF."""

    one_d = _load_profile_series(one_d_path)
    three_d = _load_profile_series(three_d_path)

    if one_d["density"].shape[0] != one_d["time_points"].size:
        raise ValueError("Die 1D-Profildaten haben inkonsistente Zeitdimensionen.")
    if three_d["density"].shape[0] != three_d["time_points"].size:
        raise ValueError("Die 3D-Profildaten haben inkonsistente Zeitdimensionen.")

    frame_data = []
    for index, time in enumerate(one_d["time_points"]):
        three_d_index = int(np.argmin(np.abs(three_d["time_points"] - time)))
        one_d_frame = {
            "radius": one_d["radius"],
            "density": one_d["density"][index],
            "velocity": one_d["velocity"][index],
            "pressure": one_d["pressure"][index],
        }
        three_d_frame = {
            "radius": three_d["radius"],
            "density": three_d["density"][three_d_index],
            "velocity": three_d["velocity"][three_d_index],
            "pressure": three_d["pressure"][three_d_index],
        }
        radius, profiles_1d, profiles_3d = _common_profiles(one_d_frame, three_d_frame)
        frame_data.append(
            (float(time), radius, profiles_1d, profiles_3d)
        )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    specifications = (
        ("density", "Dichte", r"$\rho$"),
        ("velocity", "Geschwindigkeit", r"$|v|$"),
        ("pressure", "Druck", "p"),
    )
    lines = {}
    for axis, (name, title, ylabel) in zip(axes, specifications):
        line_1d, = axis.plot([], [], color="red", linewidth=2.0, label="1D")
        line_3d, = axis.plot([], [], color="blue", linewidth=2.0, label="3D")
        lines[name] = (line_1d, line_3d)
        axis.set_title(title)
        axis.set_xlabel("Radius r")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend()
        if name == "pressure":
            axis.set_yscale("log")

    def update(frame: int):
        time, radius, profiles_1d, profiles_3d = frame_data[frame]
        for axis, (name, _, _) in zip(axes, specifications):
            values_1d = profiles_1d[name]
            values_3d = profiles_3d[name]
            line_1d, line_3d = lines[name]
            line_1d.set_data(radius, values_1d)
            line_3d.set_data(radius, values_3d)
            axis.set_xlim(0.0, float(np.max(radius)))
            finite_values = np.concatenate((values_1d, values_3d))
            finite_values = finite_values[np.isfinite(finite_values)]
            if name == "pressure":
                positive_values = finite_values[finite_values > 0.0]
                axis.set_ylim(
                    float(np.min(positive_values)),
                    1.15 * float(np.max(positive_values)),
                )
            else:
                axis.set_ylim(0.0, max(1.05 * float(np.max(finite_values)), 1e-12))
        figure.suptitle(f"1D/3D-Sedov-Profilvergleich, t = {time:.4f}")
        return tuple(line for pair in lines.values() for line in pair)

    movie = FuncAnimation(
        figure,
        update,
        frames=len(frame_data),
        interval=interval,
        blit=False,
        repeat=True,
    )

    output_path = _next_available_path(Path(output_path))
    movie.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    print(f"Gespeichertes animiertes 1D/3D-Profil-GIF: {output_path}")
    return output_path


def save_axis_comparison_gif(
    one_d_path: Path,
    three_d_path: Path,
    *,
    output_path: Path = Path(
        "Praktikum_Datenzubereitung/sedov_1d_3d_axis_comparison.gif"
    ),
    interval: int = 100,
    fps: int = 10,
):
    """Vergleiche 1D-Kurven mit Scatterpunkten auf der zentralen 3D-x-Achse."""

    one_d = _load_profile_series(one_d_path)
    three_d = _load_profile_series(three_d_path)
    required_axis_fields = {
        "axis_radius",
        "axis_density",
        "axis_velocity",
        "axis_pressure",
    }
    if not required_axis_fields.issubset(three_d):
        raise ValueError(
            "Die 3D-Profildatei enthält noch keine zentralen Achsendaten. "
            "Führe den 3D-Lauf nach der aktuellen Plotteränderung erneut aus."
        )

    frame_data = []
    for index, time in enumerate(one_d["time_points"]):
        three_d_index = int(np.argmin(np.abs(three_d["time_points"] - time)))
        radius_1d = one_d["radius"]
        radius_3d = three_d["axis_radius"]
        frame_1d = {
            "density": one_d["density"][index],
            "velocity": one_d["velocity"][index],
            "pressure": one_d["pressure"][index],
        }
        frame_3d = {
            "density": three_d["axis_density"][three_d_index],
            "velocity": three_d["axis_velocity"][three_d_index],
            "pressure": three_d["axis_pressure"][three_d_index],
        }
        frame_data.append((float(time), radius_1d, radius_3d, frame_1d, frame_3d))

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    specifications = (
        ("density", "Dichte", r"$\rho$"),
        ("velocity", "Geschwindigkeit", r"$|v|$"),
        ("pressure", "Druck", "p"),
    )
    lines = {}
    scatters = {}
    for axis, (name, title, ylabel) in zip(axes, specifications):
        line, = axis.plot([], [], color="red", linewidth=2.0, label="1D")
        scatter = axis.scatter([], [], color="blue", s=18, alpha=0.8, label="3D")
        lines[name] = line
        scatters[name] = scatter
        axis.set_title(title)
        axis.set_xlabel("Radius r")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend()
        if name == "pressure":
            axis.set_yscale("log")

    def update(frame: int):
        time, radius_1d, radius_3d, frame_1d, frame_3d = frame_data[frame]
        common_max = min(float(np.max(radius_1d)), float(np.max(radius_3d)))
        for axis, (name, _, _) in zip(axes, specifications):
            values_1d = frame_1d[name]
            values_3d = frame_3d[name]
            valid_1d = np.isfinite(radius_1d) & np.isfinite(values_1d)
            valid_3d = (radius_3d <= common_max) & np.isfinite(values_3d)
            if name == "pressure":
                valid_1d &= values_1d > 0.0
                valid_3d &= values_3d > 0.0

            lines[name].set_data(radius_1d[valid_1d], values_1d[valid_1d])
            scatters[name].set_offsets(
                np.column_stack((radius_3d[valid_3d], values_3d[valid_3d]))
            )
            axis.set_xlim(0.0, common_max)

            all_values = np.concatenate(
                (values_1d[valid_1d], values_3d[valid_3d])
            )
            if name == "pressure":
                axis.set_ylim(
                    float(np.min(all_values)),
                    1.15 * float(np.max(all_values)),
                )
            else:
                axis.set_ylim(0.0, max(1.05 * float(np.max(all_values)), 1e-12))
        figure.suptitle(f"1D/3D-Achsenvergleich, t = {time:.4f}")
        return tuple(lines.values()) + tuple(scatters.values())

    movie = FuncAnimation(
        figure,
        update,
        frames=len(frame_data),
        interval=interval,
        blit=False,
        repeat=True,
    )

    output_path = _next_available_path(Path(output_path))
    movie.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    print(f"Gespeichertes 1D/3D-Achsen-GIF: {output_path}")
    return output_path


def save_cell_scatter_comparison_gif(
    one_d_path: Path,
    three_d_path: Path,
    *,
    analytic_path: Path = Path(
        "Praktikum_Datenzubereitung/sedov_analytic_profiles.npz"
    ),
    output_path: Path = Path(
        "Praktikum_Datenzubereitung/sedov_1d_3d_cell_scatter_comparison.gif"
    ),
    interval: int = 100,
    fps: int = 10,
):
    """Vergleiche 1D, analytische Sedov-Profile und einzelne 3D-Zellwerte."""

    one_d = _load_profile_series(one_d_path)
    three_d = _load_profile_series(three_d_path)
    analytic = _load_profile_series(analytic_path)
    required_cell_fields = {
        "cell_radius",
        "cell_density",
        "cell_velocity",
        "cell_pressure",
    }
    if not required_cell_fields.issubset(three_d):
        raise ValueError(
            "Die 3D-Profildatei enthält noch keine vollständigen Zellwerte. "
            "Führe den 3D-Lauf nach der aktuellen Plotteränderung erneut aus."
        )

    radius_3d = three_d["cell_radius"].ravel()
    radius_3d_mean = three_d["radius"]
    radius_analytic = analytic["radius"]
    frame_data = []
    for index, time in enumerate(one_d["time_points"]):
        three_d_index = int(np.argmin(np.abs(three_d["time_points"] - time)))
        analytic_index = int(np.argmin(np.abs(analytic["time_points"] - time)))
        frame_data.append(
            (
                float(time),
                one_d["radius"],
                {
                    "density": one_d["density"][index],
                    "velocity": one_d["velocity"][index],
                    "pressure": one_d["pressure"][index],
                },
                {
                    "density": three_d["cell_density"][three_d_index].ravel(),
                    "velocity": three_d["cell_velocity"][three_d_index].ravel(),
                    "pressure": three_d["cell_pressure"][three_d_index].ravel(),
                },
                {
                    "density": three_d["density"][three_d_index],
                    "velocity": three_d["velocity"][three_d_index],
                    "pressure": three_d["pressure"][three_d_index],
                },
                {
                    "density": analytic["density"][analytic_index],
                    "velocity": analytic["velocity"][analytic_index],
                    "pressure": analytic["pressure"][analytic_index],
                },
            )
        )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    specifications = (
        ("density", "Dichte", r"$\rho$"),
        ("velocity", "Geschwindigkeit", r"$|v|$"),
        ("pressure", "Druck", "p"),
    )
    lines = {}
    mean_lines = {}
    analytic_lines = {}
    scatters = {}
    for axis, (name, title, ylabel) in zip(axes, specifications):
        line, = axis.plot([], [], color="red", linewidth=2.0, label="1D")
        mean_line, = axis.plot(
            [], [], color="blue", linewidth=2.0, label="3D-Mittelwert"
        )
        analytic_line, = axis.plot(
            [], [], color="green", linewidth=2.0, linestyle="--",
            label="Sedov analytisch",
        )
        scatter = axis.scatter(
            [], [], color="gray", s=2, alpha=0.22, linewidths=0, label="3D-Zellen"
        )
        lines[name] = line
        mean_lines[name] = mean_line
        analytic_lines[name] = analytic_line
        scatters[name] = scatter
        axis.set_title(title)
        axis.set_xlabel("Radius r")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend()
        if name == "pressure":
            axis.set_yscale("log")

    def update(frame: int):
        (
            time,
            radius_1d,
            values_1d,
            values_3d,
            means_3d,
            values_analytic,
        ) = frame_data[frame]
        common_max = min(
            float(np.max(radius_1d)),
            float(np.max(radius_3d)),
            float(np.max(radius_analytic)),
        )
        for axis, (name, _, _) in zip(axes, specifications):
            one_d_values = values_1d[name]
            three_d_values = values_3d[name]
            analytic_values = values_analytic[name]
            valid_1d = np.isfinite(radius_1d) & np.isfinite(one_d_values)
            valid_3d = (radius_3d <= common_max) & np.isfinite(three_d_values)
            valid_analytic = (
                (time > 0.0)
                & (radius_analytic <= common_max)
                & np.isfinite(radius_analytic)
                & np.isfinite(analytic_values)
            )
            if name == "pressure":
                valid_1d &= one_d_values > 0.0
                valid_3d &= three_d_values > 0.0
                valid_analytic &= analytic_values > 0.0

            lines[name].set_data(radius_1d[valid_1d], one_d_values[valid_1d])
            mean_values = means_3d[name]
            valid_mean = (
                (radius_3d_mean <= common_max)
                & np.isfinite(radius_3d_mean)
                & np.isfinite(mean_values)
            )
            if name == "pressure":
                valid_mean &= mean_values > 0.0
            mean_lines[name].set_data(
                radius_3d_mean[valid_mean], mean_values[valid_mean]
            )
            analytic_lines[name].set_data(
                radius_analytic[valid_analytic], analytic_values[valid_analytic]
            )
            scatters[name].set_offsets(
                np.column_stack((radius_3d[valid_3d], three_d_values[valid_3d]))
            )
            axis.set_xlim(0.0, common_max)

            all_values = np.concatenate(
                (
                    one_d_values[valid_1d],
                    three_d_values[valid_3d],
                    mean_values[valid_mean],
                    analytic_values[valid_analytic],
                )
            )
            if name == "pressure":
                axis.set_ylim(
                    float(np.min(all_values)),
                    1.15 * float(np.max(all_values)),
                )
            else:
                axis.set_ylim(0.0, max(1.05 * float(np.max(all_values)), 1e-12))
        figure.suptitle(f"1D/3D-Zellvergleich, t = {time:.4f}")
        return (
            tuple(lines.values())
            + tuple(mean_lines.values())
            + tuple(analytic_lines.values())
            + tuple(scatters.values())
        )

    movie = FuncAnimation(
        figure,
        update,
        frames=len(frame_data),
        interval=interval,
        blit=False,
        repeat=True,
    )

    output_path = _next_available_path(Path(output_path))
    movie.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    print(f"Gespeichertes 1D/3D-Zell-Scatter-GIF: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--one-d",
        type=Path,
        default=Path("Praktikum_Datenzubereitung/supernova_1d_profiles_64cells.npz"),
    )
    parser.add_argument(
        "--three-d",
        type=Path,
        default=Path("Praktikum_Datenzubereitung/supernova_3d_radial_profiles_64cells.npz"),
    )
    parser.add_argument(
        "--analytic",
        type=Path,
        default=Path("Praktikum_Datenzubereitung/sedov_analytic_profiles.npz"),
    )
    parser.add_argument("--time", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Praktikum_Datenzubereitung/sedov_1d_3d_cell_scatter_comparison.gif"),
    )
    args = parser.parse_args()
    save_cell_scatter_comparison_gif(
        args.one_d,
        args.three_d,
        analytic_path=args.analytic,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
