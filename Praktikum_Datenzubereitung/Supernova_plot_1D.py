"""Interaktive HTML-Darstellung radialer 1D-Snapshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent


def _next_available_path(output_path: Path) -> Path:
    """Gib einen freien Ausgabepfad zurück."""

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


def save_interactive_1d_html(
    snapshots,
    registered_variables,
    radius,
    *,
    field: str = "pressure",
    output_path: str | Path = DATA_DIR / "supernova_1d.html",
    title: str | None = None,
):
    """Speichere eine interaktive radiale 1D-Animation als HTML-Datei."""

    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "Für die HTML-Darstellung fehlt Plotly. Installiere es mit "
            "`python -m pip install plotly`."
        ) from error

    if field == "pressure":
        field_index = registered_variables.pressure_index
        field_label = "Druck"
    elif field == "density":
        field_index = registered_variables.density_index
        field_label = "Dichte"
    else:
        raise ValueError("field muss 'pressure' oder 'density' sein.")

    states = np.asarray(snapshots.states)
    time_points = np.asarray(snapshots.time_points)
    radius = np.asarray(radius)
    field_values = states[:, field_index, ...]

    if field_values.ndim != 2:
        raise ValueError(
            "Die 1D-Darstellung erwartet die Form "
            "(Anzahl_Snapshots, Anzahl_Radialzellen)."
        )
    if radius.ndim != 1 or radius.shape[0] != field_values.shape[1]:
        raise ValueError("radius muss ein 1D-Array mit einer Koordinate pro Zelle sein.")
    if len(time_points) != len(field_values):
        raise ValueError("Zeitpunkte und Feld-Snapshots haben unterschiedliche Längen.")

    finite_values = field_values[np.isfinite(field_values)]
    if finite_values.size == 0:
        raise ValueError("Die Snapshots enthalten keine endlichen Werte.")

    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    if y_min == y_max:
        y_max = y_min + 1.0

    def make_trace(frame: int):
        values = np.nan_to_num(field_values[frame], nan=y_min)
        return go.Scatter(
            x=radius,
            y=values,
            mode="lines",
            line=dict(color="#1f77b4", width=3),
            name=field_label,
            hovertemplate=(
                "r=%{x:.5f}<br>"
                f"{field_label}=%{{y:.5f}}<extra></extra>"
            ),
        )

    frames = [
        go.Frame(
            data=[make_trace(index)],
            name=str(index),
            layout=dict(
                title_text=f"{field_label} über Radius, t = {time_points[index]:.4f}"
            ),
        )
        for index in range(len(field_values))
    ]

    figure = go.Figure(data=frames[0].data, frames=frames)
    figure.update_layout(
        title=title or frames[0].layout.title.text,
        xaxis=dict(
            title="Radius r",
            range=[float(np.min(radius)), float(np.max(radius))],
        ),
        yaxis=dict(
            title=field_label,
            range=[y_min, y_max],
            type="log" if field == "pressure" else "linear",
        ),
        template="plotly_white",
        hovermode="x unified",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=1.15,
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

    output_path = _next_available_path(Path(output_path))
    figure.write_html(output_path, include_plotlyjs=True, auto_play=False)
    print(f"Gespeicherte interaktive 1D-Ansicht: {output_path}")
    return figure


def save_1d_gif(
    snapshots,
    registered_variables,
    radius,
    *,
    field: str = "pressure",
    output_path: str | Path = DATA_DIR / "supernova_1d.gif",
    interval: int = 100,
    fps: int = 10,
):
    """Speichere eine radiale 1D-Feldentwicklung als GIF."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if field == "pressure":
        field_index = registered_variables.pressure_index
        field_label = "Druck"
    elif field == "density":
        field_index = registered_variables.density_index
        field_label = "Dichte"
    else:
        raise ValueError("field muss 'pressure' oder 'density' sein.")

    states = np.asarray(snapshots.states)
    time_points = np.asarray(snapshots.time_points)
    radius = np.asarray(radius)
    field_values = states[:, field_index, ...]

    if field_values.ndim != 2:
        raise ValueError(
            "Die 1D-GIF-Darstellung erwartet die Form "
            "(Anzahl_Snapshots, Anzahl_Radialzellen)."
        )
    if radius.ndim != 1 or radius.shape[0] != field_values.shape[1]:
        raise ValueError("radius muss ein 1D-Array mit einer Koordinate pro Zelle sein.")
    if len(time_points) != len(field_values):
        raise ValueError("Zeitpunkte und Feld-Snapshots haben unterschiedliche Längen.")

    finite_values = field_values[np.isfinite(field_values)]
    if finite_values.size == 0:
        raise ValueError("Die Snapshots enthalten keine endlichen Werte.")

    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    if y_min == y_max:
        y_max = y_min + 1.0

    figure, axis = plt.subplots(figsize=(8, 5))
    line, = axis.plot([], [], color="#1f77b4", linewidth=2.5)
    axis.set_xlim(float(np.min(radius)), float(np.max(radius)))
    axis.set_ylim(y_min, y_max)
    if field == "pressure":
        if y_min <= 0:
            raise ValueError("Eine logarithmische Druckachse benötigt positive Werte.")
        axis.set_yscale("log")
    axis.set_xlabel("Radius r")
    axis.set_ylabel(field_label)
    axis.grid(True, alpha=0.25)

    def update(frame: int):
        values = np.nan_to_num(field_values[frame], nan=y_min)
        line.set_data(radius, values)
        axis.set_title(f"{field_label} über Radius, t = {time_points[frame]:.4f}")
        return (line,)

    movie = FuncAnimation(
        figure,
        update,
        frames=len(field_values),
        interval=interval,
        blit=False,
        repeat=True,
    )

    output_path = _next_available_path(Path(output_path))
    movie.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    print(f"Gespeichertes 1D-GIF: {output_path}")
    return output_path


def save_1d_profiles_gif(
    snapshots,
    registered_variables,
    radius,
    *,
    output_path: str | Path = DATA_DIR / "supernova_1d_profiles.gif",
    data_output_path: str | None = None,
    interval: int = 100,
    fps: int = 10,
    headroom: float = 1.05,
    pressure_headroom: float = 1.15,
):
    """Speichere Dichte, Geschwindigkeitsbetrag und Druck gemeinsam als GIF."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    states = np.asarray(snapshots.states)
    time_points = np.asarray(snapshots.time_points)
    radius = np.asarray(radius)

    if headroom <= 1.0 or pressure_headroom <= 1.0:
        raise ValueError("headroom und pressure_headroom müssen größer als 1 sein.")

    if states.ndim != 3:
        raise ValueError(
            "Die Profil-GIF-Darstellung erwartet die Form "
            "(Snapshots, Variablen, Radialzellen)."
        )
    if radius.ndim != 1 or radius.shape[0] != states.shape[-1]:
        raise ValueError("radius muss ein 1D-Array mit einer Koordinate pro Zelle sein.")
    if len(time_points) != states.shape[0]:
        raise ValueError("Zeitpunkte und Feld-Snapshots haben unterschiedliche Längen.")

    velocity_index = registered_variables.velocity_index
    if not np.isscalar(velocity_index):
        raise ValueError("Die Profilansicht erwartet eine eindimensionale Geschwindigkeit.")

    profiles = {
        "Dichte": states[:, registered_variables.density_index, :],
        "Geschwindigkeit": np.abs(states[:, int(velocity_index), :]),
        "Druck": states[:, registered_variables.pressure_index, :],
    }
    colors = {
        "Dichte": "#2ca02c",
        "Geschwindigkeit": "#d62728",
        "Druck": "#1f77b4",
    }
    labels = {"Dichte": r"$\rho$", "Geschwindigkeit": r"$|v|$", "Druck": "p"}

    bounds = {}
    for name, values in profiles.items():
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError(f"{name} enthält keine endlichen Werte.")
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        if name == "Druck" and y_min <= 0:
            raise ValueError("Eine logarithmische Druckachse benötigt positive Werte.")
        if y_min == y_max:
            y_max = y_min + 1.0
        bounds[name] = (y_min, y_max)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    lines = []
    for axis, name in zip(axes, profiles):
        line, = axis.plot([], [], color=colors[name], linewidth=2.0)
        lines.append(line)
        y_min, y_max = bounds[name]
        axis.set_xlim(float(np.min(radius)), float(np.max(radius)))
        axis.set_ylim(y_min, y_max)
        if name == "Druck":
            axis.set_yscale("log")
        axis.set_title(name)
        axis.set_xlabel("Radius r")
        axis.set_ylabel(labels[name])
        axis.grid(True, alpha=0.25)

    def update(frame: int):
        for axis, line, (name, values) in zip(axes, lines, profiles.items()):
            current_values = np.nan_to_num(values[frame], nan=bounds[name][0])
            line.set_data(radius, current_values)

            current_max = float(np.max(current_values))
            if name == "Druck":
                positive_values = current_values[current_values > 0.0]
                current_min = float(np.min(positive_values))
                if current_max <= current_min:
                    current_max = current_min * headroom
                axis.set_ylim(current_min, pressure_headroom * current_max)
            else:
                if current_max <= 0.0:
                    current_max = 1.0
                axis.set_ylim(0.0, headroom * current_max)
        figure.suptitle(f"Sedov-Profile, t = {time_points[frame]:.4f}")
        return tuple(lines)

    movie = FuncAnimation(
        figure,
        update,
        frames=states.shape[0],
        interval=interval,
        blit=False,
        repeat=True,
    )

    output_path = _next_available_path(Path(output_path))
    if data_output_path is None:
        data_output_path = output_path.with_suffix(".npz")
    else:
        data_output_path = _next_available_path(Path(data_output_path))
    np.savez_compressed(
        data_output_path,
        radius=radius,
        density=profiles["Dichte"],
        velocity=profiles["Geschwindigkeit"],
        pressure=profiles["Druck"],
        time_points=time_points,
    )
    print(f"Gespeicherte 1D-Profildaten: {data_output_path}")
    movie.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    print(f"Gespeichertes 1D-Profil-GIF: {output_path}")
    return output_path
