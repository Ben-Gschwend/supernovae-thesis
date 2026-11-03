"""Sedov-Taylor-Referenzprofile für den sphärischen 3D-Blastwave-Test.

Die Implementierung gilt für die klassische starke Sedov-Taylor-Lösung in
einem homogenen Medium und für ``gamma=5/3``. Sie erzeugt keine numerische
Hydrodynamik-Simulation, sondern wertet die selbstähnliche analytische
Referenzlösung zu vorgegebenen Zeiten aus.

Die parametrische Darstellung der Profile folgt den Gleichungen (7.28)--
(7.30) in:

    Gordon I. Ogilvie, *Astrophysical fluid dynamics*, Lecture Notes,
    Abschnitt 7.4--7.6.

    https://api.repository.cam.ac.uk/server/api/core/bitstreams/
    c50602cc-91b5-4eb6-9a72-467954d597f2/content

Für die spätere Plotdarstellung werden außerhalb der analytischen
Schockfront die Umgebungswerte ``rho_0``, ``velocity=0`` und
``ambient_pressure`` verwendet. Der ideale starke-Sedov-Anteil selbst nimmt
einen vernachlässigbaren Umgebungsdruck an.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


GAMMA = 5.0 / 3.0


def _next_available_path(path: Path) -> Path:
    """Verhindere, dass eine vorhandene Referenzdatei überschrieben wird."""

    if not path.exists():
        return path
    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Kein freier Ausgabename für {path} gefunden.")


def _parametric_profiles(num_profile_points: int = 20000):
    """Erzeuge die dimensionslosen Sedov-Profile für gamma=5/3.

    Der Parameter ``v`` ist nicht die physikalische Geschwindigkeit. Es gilt
    ``u_tilde = v * xi`` und damit physikalisch ``u = shock_speed*u_tilde``.
    Die zurückgegebenen Profile sind nach der dimensionslosen Position
    ``xi=r/R`` sortiert und erfüllen ``0 <= xi <= 1``.
    """

    if num_profile_points < 100:
        raise ValueError("num_profile_points muss mindestens 100 sein.")

    v = np.linspace(3.0 / 5.0, 3.0 / 4.0, num_profile_points)

    xi = (
        (4.0 * v / 3.0) ** (-2.0 / 5.0)
        * (20.0 * v / 3.0 - 4.0) ** (2.0 / 13.0)
        * (2.5 - 2.0 * v) ** (-82.0 / 195.0)
    )
    density_tilde = (
        4.0
        * (4.0 - 4.0 * v) ** (-6.0)
        * (20.0 * v / 3.0 - 4.0) ** (9.0 / 13.0)
        * (2.5 - 2.0 * v) ** (82.0 / 13.0)
    )
    pressure_tilde = (
        3.0
        / 4.0
        * (4.0 * v / 3.0) ** (6.0 / 5.0)
        * (4.0 - 4.0 * v) ** (-5.0)
        * (2.5 - 2.0 * v) ** (82.0 / 15.0)
    )
    velocity_tilde = v * xi

    # Die Formeln sind bereits monoton in xi, die Sortierung schützt aber
    # die Interpolation gegen Rundungsfehler an den Grenzpunkten.
    order = np.argsort(xi)
    return (
        xi[order],
        density_tilde[order],
        velocity_tilde[order],
        pressure_tilde[order],
    )


def sedov_energy_constant(
    *,
    gamma: float = GAMMA,
    num_profile_points: int = 20000,
) -> float:
    """Bestimme die Sedov-Konstante beta aus der Gesamtenergie.

    Für die gewählte Normierung gilt

    ``R(t) = beta * (E*t**2/rho_0)**(1/5)``.
    """

    if not np.isclose(gamma, GAMMA):
        raise ValueError(
            "Die geschlossene parametrische Implementierung ist derzeit "
            "nur für gamma=5/3 verfügbar."
        )

    xi, density_tilde, velocity_tilde, pressure_tilde = _parametric_profiles(
        num_profile_points
    )
    energy_integrand = (
        0.5 * density_tilde * velocity_tilde**2
        + pressure_tilde / (gamma - 1.0)
    ) * xi**2
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    dimensionless_energy = float(integrate(energy_integrand, xi))
    beta = (1.0 / ((16.0 * np.pi / 25.0) * dimensionless_energy)) ** (1.0 / 5.0)
    return beta


def sedov_profiles_at_time(
    radius,
    time: float,
    *,
    explosion_energy: float = 1.0,
    ambient_density: float = 1.0,
    ambient_pressure: float = 1.0e-4,
    gamma: float = GAMMA,
    num_profile_points: int = 20000,
):
    """Berechne rho, radial velocity und Druck zu einer Zeit.

    Parameters
    ----------
    radius:
        Nichtnegative Radien, an denen die Lösung ausgewertet wird.
    time:
        Muss strikt positiv sein. Die ideale Punkt-Explosion ist bei ``t=0``
        singulär und wird deshalb nicht ausgewertet.
    explosion_energy, ambient_density, ambient_pressure:
        Physikalische Normierung, passend zu den numerischen Testskripten.

    Returns
    -------
    dict
        ``radius``, ``density``, ``velocity``, ``pressure``, ``shock_radius``
        und ``shock_speed``.
    """

    if time <= 0.0:
        raise ValueError("Die analytische Sedov-Lösung ist nur für time > 0 definiert.")
    if explosion_energy <= 0.0 or ambient_density <= 0.0:
        raise ValueError("explosion_energy und ambient_density müssen positiv sein.")
    if ambient_pressure < 0.0:
        raise ValueError("ambient_pressure darf nicht negativ sein.")

    radius = np.asarray(radius, dtype=float)
    if np.any(~np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("radius muss aus endlichen, nichtnegativen Werten bestehen.")

    xi_profile, density_tilde, velocity_tilde, pressure_tilde = (
        _parametric_profiles(num_profile_points)
    )
    beta = sedov_energy_constant(
        gamma=gamma,
        num_profile_points=num_profile_points,
    )
    shock_radius = beta * (explosion_energy * time**2 / ambient_density) ** (1.0 / 5.0)
    shock_speed = 2.0 * shock_radius / (5.0 * time)

    inside = radius <= shock_radius
    xi = np.clip(radius / shock_radius, 0.0, 1.0)
    density = np.full(radius.shape, ambient_density, dtype=float)
    velocity = np.zeros(radius.shape, dtype=float)
    pressure = np.full(radius.shape, ambient_pressure, dtype=float)
    density[inside] = ambient_density * np.interp(xi[inside], xi_profile, density_tilde)
    velocity[inside] = shock_speed * np.interp(
        xi[inside], xi_profile, velocity_tilde
    )
    pressure[inside] = ambient_density * shock_speed**2 * np.interp(
        xi[inside], xi_profile, pressure_tilde
    )

    return {
        "radius": radius,
        "density": density,
        "velocity": velocity,
        "pressure": pressure,
        "shock_radius": shock_radius,
        "shock_speed": shock_speed,
        "beta": beta,
    }


def generate_analytic_profiles(
    *,
    t_start: float = 1.0e-4,
    t_end: float = 0.1,
    num_snapshots: int = 60,
    max_radius: float = 1.0,
    num_radii: int = 2048,
    explosion_energy: float = 1.0,
    ambient_density: float = 1.0,
    ambient_pressure: float = 1.0e-4,
    gamma: float = GAMMA,
):
    """Erzeuge eine zeitliche Serie analytischer Sedov-Profile."""

    if t_start <= 0.0 or t_end <= t_start:
        raise ValueError("Es muss 0 < t_start < t_end gelten.")
    if num_snapshots < 1 or num_radii < 2:
        raise ValueError("num_snapshots muss >=1 und num_radii muss >=2 sein.")
    if max_radius <= 0.0:
        raise ValueError("max_radius muss positiv sein.")

    time_points = np.linspace(t_start, t_end, num_snapshots)
    radius = np.linspace(0.0, max_radius, num_radii)
    density = np.empty((num_snapshots, num_radii), dtype=float)
    velocity = np.empty_like(density)
    pressure = np.empty_like(density)
    shock_radii = np.empty(num_snapshots, dtype=float)
    shock_speeds = np.empty(num_snapshots, dtype=float)

    for index, time in enumerate(time_points):
        profile = sedov_profiles_at_time(
            radius,
            float(time),
            explosion_energy=explosion_energy,
            ambient_density=ambient_density,
            ambient_pressure=ambient_pressure,
            gamma=gamma,
        )
        density[index] = profile["density"]
        velocity[index] = profile["velocity"]
        pressure[index] = profile["pressure"]
        shock_radii[index] = profile["shock_radius"]
        shock_speeds[index] = profile["shock_speed"]

    return {
        "radius": radius,
        "density": density,
        "velocity": velocity,
        "pressure": pressure,
        "time_points": time_points,
        "shock_radius": shock_radii,
        "shock_speed": shock_speeds,
        "beta": sedov_energy_constant(gamma=gamma),
        "explosion_energy": float(explosion_energy),
        "ambient_density": float(ambient_density),
        "ambient_pressure": float(ambient_pressure),
        "gamma": float(gamma),
    }


def save_analytic_profiles(
    output_path: str | Path = "Praktikum_Datenzubereitung/sedov_analytic_profiles.npz",
    **kwargs,
) -> Path:
    """Erzeuge und speichere die analytischen Sedov-Profile als NPZ-Datei."""

    output_path = _next_available_path(Path(output_path))
    profiles = generate_analytic_profiles(**kwargs)
    np.savez_compressed(output_path, **profiles)
    print(f"Gespeicherte analytische Sedov-Profile: {output_path}")
    print(f"Sedov-Konstante beta: {profiles['beta']:.6f}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Praktikum_Datenzubereitung/sedov_analytic_profiles.npz"),
    )
    parser.add_argument("--t-start", type=float, default=1.0e-4)
    parser.add_argument("--t-end", type=float, default=0.1)
    parser.add_argument("--num-snapshots", type=int, default=60)
    parser.add_argument("--max-radius", type=float, default=1.0)
    parser.add_argument("--num-radii", type=int, default=2048)
    args = parser.parse_args()
    save_analytic_profiles(
        args.output,
        t_start=args.t_start,
        t_end=args.t_end,
        num_snapshots=args.num_snapshots,
        max_radius=args.max_radius,
        num_radii=args.num_radii,
    )


if __name__ == "__main__":
    main()
