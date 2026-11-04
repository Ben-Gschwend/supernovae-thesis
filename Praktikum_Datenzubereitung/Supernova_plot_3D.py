"""Interaktive 3D-Darstellung der astronomix-Snapshots.

Es werden nur Zellen mit einem Druck oberhalb eines vorgegebenen
Hintergrunddrucks als transparente 3D-Punkte dargestellt. Zusätzlich wird
eine zentrale z-Schnittfläche des Druckfeldes angezeigt. Das Matplotlib-
3D-Fenster kann mit der Maus gedreht, verschoben und gezoomt werden.
"""

from __future__ import annotations

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors, colormaps
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent


def animate_interactive_3d(
    snapshots,
    registered_variables,
    *,
    box_size: float | tuple[float, float, float] = 1.0,
    background_pressure: float = 0.1,
    pressure_margin: float = 0.0,
    cell_stride: int = 2,
    interval: int = 100,
    show: bool = True,
):
    """Zeige 3D-Druck-Snapshots interaktiv.

    Zellen mit ``pressure > background_pressure + pressure_margin`` werden
    als transparente Punkte geplottet. Die zentrale z-Ebene wird als farbige
    Fläche eingeblendet. Die Ansicht bleibt während der Animation drehbar.

    Parameters
    ----------
    snapshots:
        Ergebnis von ``time_integration`` mit gespeicherten Zuständen.
    registered_variables:
        Registrierte astronomix-Variablen.
    box_size:
        Kantenlänge des kartesischen Simulationsgebiets als Skalar oder als
        drei Kantenlängen ``(box_x, box_y, box_z)``.
    background_pressure:
        Druck des Außenmediums im Anfangszustand.
    pressure_margin:
        Zusätzlicher Abstand zur Druckschwelle.
    cell_stride:
        Nur jede ``cell_stride``-te Zelle für die 3D-Punkte verwenden.
        Die zentrale Schnittfläche wird ebenfalls ausgedünnt.
    interval:
        Zeit zwischen zwei Animationsframes in Millisekunden.
    show:
        Bei ``True`` das interaktive Matplotlib-Fenster anzeigen.

    Returns
    -------
    tuple
        ``(movie, figure)`` für weitere interaktive Verwendung.
    """

    if cell_stride < 1:
        raise ValueError("cell_stride muss mindestens 1 sein.")

    states = np.asarray(snapshots.states)
    time_points = np.asarray(snapshots.time_points)
    field_values = states[:, registered_variables.pressure_index, ...]

    if field_values.ndim != 4:
        raise ValueError(
            "Die 3D-Darstellung erwartet Druck-Snapshots mit der Form "
            "(Anzahl_Snapshots, Nx, Ny, Nz)."
        )
    if len(time_points) != len(field_values):
        raise ValueError(
            "Anzahl der Zeitpunkte und Druck-Snapshots stimmen nicht überein."
        )

    number_of_frames, nx, ny, nz = field_values.shape
    if min(nx, ny, nz) < 2:
        raise ValueError("Das 3D-Feld muss in jeder Raumrichtung mindestens 2 Zellen haben.")

    box_sizes = np.asarray(box_size, dtype=float)
    if box_sizes.ndim == 0:
        box_x = box_y = box_z = float(box_sizes)
    elif box_sizes.shape == (3,):
        box_x, box_y, box_z = (float(value) for value in box_sizes)
    else:
        raise ValueError("box_size muss ein Skalar oder ein Vektor mit drei Werten sein.")

    x_full = (np.arange(nx) + 0.5) * box_x / nx
    y_full = (np.arange(ny) + 0.5) * box_y / ny
    z_full = (np.arange(nz) + 0.5) * box_z / nz

    x = x_full[::cell_stride]
    y = y_full[::cell_stride]
    z = z_full[::cell_stride]
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")

    z_index = nz // 2
    x_slice, y_slice = np.meshgrid(x, y, indexing="ij")
    z_slice = np.full_like(x_slice, z_full[z_index], dtype=float)

    finite_values = field_values[np.isfinite(field_values)]
    if finite_values.size == 0:
        raise ValueError("Die Druck-Snapshots enthalten keine endlichen Werte.")
    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if vmin == vmax:
        vmax = vmin + 1.0

    cmap = colormaps["viridis"]
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    threshold = background_pressure + pressure_margin

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.set_xlim(0.0, box_x)
    axis.set_ylim(0.0, box_y)
    axis.set_zlim(0.0, box_z)
    axis.set_box_aspect((box_x, box_y, box_z))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")

    colorbar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar_map.set_array(finite_values)
    colorbar = figure.colorbar(colorbar_map, ax=axis, pad=0.1, shrink=0.75)
    colorbar.set_label("Druck")

    artists = {"points": None, "slice": None}

    def draw_frame(frame: int):
        if artists["points"] is not None:
            artists["points"].remove()
        if artists["slice"] is not None:
            artists["slice"].remove()

        pressure = field_values[frame]
        pressure_sampled = pressure[::cell_stride, ::cell_stride, ::cell_stride]
        finite_mask = np.isfinite(pressure_sampled)
        point_mask = finite_mask & (pressure_sampled > threshold)

        point_colors = cmap(norm(pressure_sampled[point_mask]))
        artists["points"] = axis.scatter(
            x_grid[point_mask],
            y_grid[point_mask],
            z_grid[point_mask],
            c=point_colors,
            s=8,
            alpha=0.28,
            depthshade=False,
            linewidths=0,
        )

        slice_pressure = pressure[::cell_stride, ::cell_stride, z_index]
        slice_colors = cmap(norm(np.nan_to_num(slice_pressure, nan=vmin)))
        artists["slice"] = axis.plot_surface(
            x_slice,
            y_slice,
            z_slice,
            facecolors=slice_colors,
            rstride=1,
            cstride=1,
            shade=False,
            alpha=0.62,
            linewidth=0,
        )

        axis.set_title(
            "Druck > "
            f"{threshold:g}; z-Schnitt {z_index}; "
            f"t = {time_points[frame]:.4f}"
        )

    draw_frame(0)
    movie = animation.FuncAnimation(
        figure,
        draw_frame,
        frames=number_of_frames,
        interval=interval,
        blit=False,
        repeat=True,
    )

    if show:
        plt.show()

    return movie, figure


def save_interactive_3d_html(
    snapshots,
    registered_variables,
    *,
    output_path: str | Path = DATA_DIR / "supernova_3d.html",
    box_size: float | tuple[float, float, float] = 1.0,
    background_pressure: float = 0.1,
    initial_inside_pressure: float = 1.0,
    pressure_margin: float = 0.0,
    contrast_exponent: float = 0.3,
    cell_stride: int = 2,
):
    """Speichere eine drehbare 3D-Darstellung als selbstständige HTML-Datei.

    Die HTML-Datei kann auf dem lokalen Rechner in einem Browser geöffnet
    werden. Sie enthält einen Zeitschieber und Play/Pause-Steuerung. Plotly
    wird erst beim Aufruf importiert, damit die Matplotlib-Funktion weiterhin
    unabhängig davon importierbar bleibt. Die Darstellung verwendet ein
    volumetrisches Druckfeld statt Mittelpunkt-Punkten. Der Außendruck ist
    die transparente Untergrenze; der größte tatsächlich vorkommende Druck
    definiert die obere Farbgrenze. Die Drucknormierung ist logarithmisch.
    """

    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "Für die HTML-Darstellung fehlt Plotly. Installiere es mit "
            "`python -m pip install plotly`."
        ) from error

    if cell_stride < 1:
        raise ValueError("cell_stride muss mindestens 1 sein.")
    if contrast_exponent <= 0.0:
        raise ValueError("contrast_exponent muss positiv sein.")

    states = np.asarray(snapshots.states)
    time_points = np.asarray(snapshots.time_points)
    field_values = states[:, registered_variables.pressure_index, ...]

    if field_values.ndim != 4:
        raise ValueError(
            "Die HTML-3D-Darstellung erwartet Druck-Snapshots mit der Form "
            "(Anzahl_Snapshots, Nx, Ny, Nz)."
        )
    if len(time_points) != len(field_values):
        raise ValueError(
            "Anzahl der Zeitpunkte und Druck-Snapshots stimmen nicht überein."
        )

    number_of_frames, nx, ny, nz = field_values.shape
    box_sizes = np.asarray(box_size, dtype=float)
    if box_sizes.ndim == 0:
        box_x = box_y = box_z = float(box_sizes)
    elif box_sizes.shape == (3,):
        box_x, box_y, box_z = (float(value) for value in box_sizes)
    else:
        raise ValueError("box_size muss ein Skalar oder ein Vektor mit drei Werten sein.")

    x_full = (np.arange(nx) + 0.5) * box_x / nx
    y_full = (np.arange(ny) + 0.5) * box_y / ny
    z_full = (np.arange(nz) + 0.5) * box_z / nz
    x = x_full[::cell_stride]
    y = y_full[::cell_stride]
    z = z_full[::cell_stride]
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")

    z_index = nz // 2
    z_slice = np.full((len(x), len(y)), z_full[z_index], dtype=float)
    threshold = background_pressure + pressure_margin
    if initial_inside_pressure <= background_pressure:
        raise ValueError(
            "initial_inside_pressure muss größer als background_pressure sein."
        )

    finite_values = field_values[np.isfinite(field_values)]
    if finite_values.size == 0:
        raise ValueError("Die Druck-Snapshots enthalten keine endlichen Werte.")
    color_min_pressure = float(background_pressure)
    if color_min_pressure <= 0.0:
        raise ValueError(
            "background_pressure muss für die logarithmische Skalierung positiv sein."
        )
    color_max_pressure = max(
        float(initial_inside_pressure),
        float(np.max(finite_values)),
    )
    if color_max_pressure <= color_min_pressure:
        color_max_pressure = color_min_pressure * 10.0

    log_min_pressure = np.log10(color_min_pressure)
    log_max_pressure = np.log10(color_max_pressure)
    log_pressure_range = log_max_pressure - log_min_pressure

    def normalize_pressure(pressure):
        """Normiere den Druck logarithmisch und kontrastverstärkt auf [0, 1]."""

        safe_pressure = np.maximum(pressure, color_min_pressure)
        normalized_log = np.clip(
            (np.log10(safe_pressure) - log_min_pressure) / log_pressure_range,
            0.0,
            1.0,
        )
        return normalized_log**contrast_exponent

    threshold_normalized = normalize_pressure(threshold)
    tick_pressures = np.geomspace(
        color_min_pressure,
        color_max_pressure,
        num=3,
    )
    tick_values = normalize_pressure(tick_pressures)

    def frame_data(pressure):
        pressure_sampled = pressure[::cell_stride, ::cell_stride, ::cell_stride]
        pressure_sampled = np.nan_to_num(
            pressure_sampled,
            nan=color_min_pressure,
        )
        normalized_sampled = normalize_pressure(pressure_sampled)
        normalized_sampled[pressure_sampled <= threshold] = 0.0

        pressure_volume = go.Volume(
            x=x_grid.ravel(),
            y=y_grid.ravel(),
            z=z_grid.ravel(),
            value=normalized_sampled.ravel(),
            name="Druckvolumen",
            isomin=threshold_normalized,
            isomax=1.0,
            surface_count=18,
            opacity=0.18,
            opacityscale=[
                [0.00, 0.00],
                [0.01, 0.03],
                [0.05, 0.10],
                [0.25, 0.28],
                [0.60, 0.58],
                [1.00, 0.85],
            ],
            colorscale="Viridis",
            cmin=0.0,
            cmax=1.0,
            colorbar=dict(
                title="Druck (log)",
                tickvals=tick_values.tolist(),
                ticktext=[f"{value:g}" for value in tick_pressures],
            ),
            caps=dict(x_show=False, y_show=False, z_show=False),
            hovertemplate="logarithmisch normierter Druck=%{value:.3f}<extra></extra>",
        )

        slice_values = normalize_pressure(
            np.nan_to_num(
                pressure[::cell_stride, ::cell_stride, z_index],
                nan=color_min_pressure,
            )
        )
        slice_surface = go.Surface(
            x=x,
            y=y,
            z=z_slice,
            surfacecolor=slice_values,
            colorscale="Viridis",
            cmin=0.0,
            cmax=1.0,
            opacity=0.62,
            showscale=False,
            name="z-Schnitt",
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>logarithmisch normierter Druck=%{surfacecolor:.3f}<extra></extra>",
        )
        return [pressure_volume, slice_surface]

    frames = [
        go.Frame(
            data=frame_data(pressure),
            name=str(index),
            layout=go.Layout(
                title_text=(
                    f"Druckvolumen (log, Kontrast {contrast_exponent:g}) "
                    f"{color_min_pressure:g} bis "
                    f"{color_max_pressure:g}; z-Schnitt {z_index}; "
                    f"t = {time_points[index]:.4f}"
                )
            ),
        )
        for index, pressure in enumerate(field_values)
    ]

    figure = go.Figure(data=frames[0].data, frames=frames)
    figure.update_layout(
        title=frames[0].layout.title.text,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            xaxis=dict(range=[0.0, box_x]),
            yaxis=dict(range=[0.0, box_y]),
            zaxis=dict(range=[0.0, box_z]),
            aspectmode="manual",
            aspectratio=dict(x=box_x, y=box_y, z=box_z),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=1.12,
                buttons=[
                    dict(
                        label="▶ Start",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=100, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Snapshot: "),
                pad=dict(t=50),
                steps=[
                    dict(
                        label=f"{time:.4f}",
                        method="animate",
                        args=[
                            [str(index)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for index, time in enumerate(time_points)
                ],
            )
        ],
    )

    output_path = Path(output_path)
    if output_path.exists():
        counter = 1
        while True:
            candidate = output_path.with_name(
                f"{output_path.stem}_{counter}{output_path.suffix}"
            )
            if not candidate.exists():
                output_path = candidate
                break
            counter += 1

    figure.write_html(output_path, include_plotlyjs=True, auto_play=False)
    print(f"Gespeicherte interaktive 3D-Ansicht: {output_path}")
    return figure


def save_3d_radial_profiles_gif(
    snapshots,
    registered_variables,
    *,
    box_size: float | tuple[float, float, float] = 1.0,
    output_path: str | Path = DATA_DIR / "supernova_3d_radial_profiles.gif",
    data_output_path: str | None = None,
    num_bins: int = 128,
    interval: int = 100,
    fps: int = 10,
    headroom: float = 1.05,
    pressure_headroom: float = 1.15,
):
    """Speichere radiale 3D-Profile von Dichte, Geschwindigkeit und Druck.

    Die kartesischen Zellen werden nach ihrem Abstand zum Boxzentrum gebinnt.
    Pro radialem Bin wird der Mittelwert der jeweiligen Größe dargestellt.
    """

    from matplotlib.animation import FuncAnimation, PillowWriter

    if num_bins < 2:
        raise ValueError("num_bins muss mindestens 2 sein.")
    if headroom <= 1.0 or pressure_headroom <= 1.0:
        raise ValueError("headroom und pressure_headroom müssen größer als 1 sein.")

    states = np.asarray(snapshots.states)
    time_points = np.asarray(snapshots.time_points)
    if states.ndim != 5:
        raise ValueError(
            "Die radiale 3D-Auswertung erwartet die Form "
            "(Snapshots, Variablen, Nx, Ny, Nz)."
        )
    if len(time_points) != states.shape[0]:
        raise ValueError("Zeitpunkte und Feld-Snapshots haben unterschiedliche Längen.")

    _, nx, ny, nz = states.shape[0], *states.shape[2:]
    box_sizes = np.asarray(box_size, dtype=float)
    if box_sizes.ndim == 0:
        box_x = box_y = box_z = float(box_sizes)
    elif box_sizes.shape == (3,):
        box_x, box_y, box_z = (float(value) for value in box_sizes)
    else:
        raise ValueError("box_size muss ein Skalar oder ein Vektor mit drei Werten sein.")

    velocity_indices = np.asarray(registered_variables.velocity_index, dtype=int).reshape(-1)
    if velocity_indices.size != 3:
        raise ValueError("Die radiale 3D-Auswertung erwartet drei Geschwindigkeitskomponenten.")

    x = (np.arange(nx) + 0.5) * box_x / nx
    y = (np.arange(ny) + 0.5) * box_y / ny
    z = (np.arange(nz) + 0.5) * box_z / nz
    radius = np.sqrt(
        (x[:, None, None] - box_x / 2.0) ** 2
        + (y[None, :, None] - box_y / 2.0) ** 2
        + (z[None, None, :] - box_z / 2.0) ** 2
    )
    radius_flat = radius.ravel()
    radius_max = float(np.max(radius_flat))
    bin_edges = np.linspace(0.0, radius_max, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_indices = np.clip(np.digitize(radius_flat, bin_edges) - 1, 0, num_bins - 1)
    counts = np.bincount(bin_indices, minlength=num_bins)
    valid_bins = counts > 0

    density_index = registered_variables.density_index
    pressure_index = registered_variables.pressure_index
    density = states[:, density_index, ...]
    velocity = np.sqrt(np.sum(states[:, velocity_indices, ...] ** 2, axis=1))
    pressure = states[:, pressure_index, ...]

    # Zentrale x-Achse durch das Explosionszentrum: y und z liegen möglichst
    # nahe bei der jeweiligen Boxmitte, x läuft über alle Zellen.
    axis_y_index = int(np.argmin(np.abs(y - box_y / 2.0)))
    axis_z_index = int(np.argmin(np.abs(z - box_z / 2.0)))
    axis_x = x.copy()
    axis_radius = np.abs(axis_x - box_x / 2.0)
    axis_order = np.argsort(axis_radius)
    axis_x = axis_x[axis_order]
    axis_radius = axis_radius[axis_order]
    axis_density = density[:, :, axis_y_index, axis_z_index][:, axis_order]
    axis_velocity = velocity[:, :, axis_y_index, axis_z_index][:, axis_order]
    axis_pressure = pressure[:, :, axis_y_index, axis_z_index][:, axis_order]

    def radial_mean(values):
        result = np.full((values.shape[0], num_bins), np.nan, dtype=float)
        flat_values = values.reshape(values.shape[0], -1)
        for frame in range(values.shape[0]):
            finite = np.isfinite(flat_values[frame])
            sums = np.bincount(
                bin_indices[finite],
                weights=flat_values[frame, finite],
                minlength=num_bins,
            )
            result[frame, valid_bins] = sums[valid_bins] / counts[valid_bins]
        return result

    profiles = {
        "Dichte": radial_mean(density),
        "Geschwindigkeit": radial_mean(velocity),
        "Druck": radial_mean(pressure),
    }
    colors_by_name = {
        "Dichte": "#2ca02c",
        "Geschwindigkeit": "#d62728",
        "Druck": "#1f77b4",
    }
    labels = {"Dichte": r"$\rho$", "Geschwindigkeit": r"$|v|$", "Druck": "p"}

    bounds = {}
    for name, values in profiles.items():
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError(f"{name} enthält keine endlichen radialen Mittelwerte.")
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        if name == "Druck" and y_min <= 0.0:
            raise ValueError("Eine logarithmische Druckachse benötigt positive Werte.")
        if y_min == y_max:
            y_max = y_min + 1.0
        bounds[name] = (y_min, y_max)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    lines = []
    for axis, name in zip(axes, profiles):
        line, = axis.plot([], [], color=colors_by_name[name], linewidth=2.0)
        lines.append(line)
        y_min, y_max = bounds[name]
        axis.set_xlim(0.0, radius_max)
        axis.set_ylim(y_min, y_max)
        if name == "Druck":
            axis.set_yscale("log")
        axis.set_title(name)
        axis.set_xlabel("Radius r")
        axis.set_ylabel(labels[name])
        axis.grid(True, alpha=0.25)

    def update(frame: int):
        for axis, line, (name, values) in zip(axes, lines, profiles.items()):
            current_values = values[frame]
            line.set_data(bin_centers, current_values)
            finite_current = current_values[np.isfinite(current_values)]
            if finite_current.size == 0:
                continue
            current_max = float(np.max(finite_current))
            if name == "Druck":
                current_positive = finite_current[finite_current > 0.0]
                current_min = float(np.min(current_positive))
                axis.set_ylim(current_min, pressure_headroom * current_max)
            else:
                axis.set_ylim(0.0, max(headroom * current_max, 1e-12))
        figure.suptitle(f"3D-Sedov-Profile, t = {time_points[frame]:.4f}")
        return tuple(lines)

    movie = FuncAnimation(
        figure,
        update,
        frames=states.shape[0],
        interval=interval,
        blit=False,
        repeat=True,
    )

    output_path = Path(output_path)
    if output_path.exists():
        counter = 1
        while True:
            candidate = output_path.with_name(
                f"{output_path.stem}_{counter}{output_path.suffix}"
            )
            if not candidate.exists():
                output_path = candidate
                break
            counter += 1

    if data_output_path is None:
        data_output_path = output_path.with_suffix(".npz")
    else:
        data_output_path = Path(data_output_path)
        if data_output_path.exists():
            counter = 1
            while True:
                candidate = data_output_path.with_name(
                    f"{data_output_path.stem}_{counter}{data_output_path.suffix}"
                )
                if not candidate.exists():
                    data_output_path = candidate
                    break
                counter += 1

    np.savez_compressed(
        data_output_path,
        radius=bin_centers,
        density=profiles["Dichte"],
        velocity=profiles["Geschwindigkeit"],
        pressure=profiles["Druck"],
        time_points=time_points,
        axis_x=axis_x,
        axis_radius=axis_radius,
        axis_density=axis_density,
        axis_velocity=axis_velocity,
        axis_pressure=axis_pressure,
        cell_radius=radius,
        cell_density=density,
        cell_velocity=velocity,
        cell_pressure=pressure,
    )
    print(f"Gespeicherte 3D-Profildaten: {data_output_path}")
    movie.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    print(f"Gespeichertes 3D-Profil-GIF: {output_path}")
    return output_path
