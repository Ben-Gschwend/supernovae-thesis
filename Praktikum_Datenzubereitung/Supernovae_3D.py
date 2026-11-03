"""3D-Sedov-Simulation mit astronomix.

Die Anfangsbedingungen entsprechen dem derzeit verwendeten 1D-Sedov-Test.
Die Paperauflösung von 256^3 Zellen ist als späterer Vergleichslauf vorgesehen;
dieses Skript verwendet zunächst die vorhandene 3D-Pilotauflösung.
"""

# TODO: Imports und SimulationConfig gemeinsam ausarbeiten.
import jax.numpy as jnp
from astronomix import (
    SimulationConfig, SimulationParams,
    get_helper_data, finalize_config,
    get_registered_variables, construct_primitive_state,
    time_integration,
    CARTESIAN,
    FINITE_VOLUME,
)
from astronomix.option_classes.simulation_config import SnapshotSettings

# the SimulationConfig holds static 
# configuration parameters
config = SimulationConfig(
    geometry=CARTESIAN,
    dimensionality=3,
    box_size = 1.0,
    num_cells = 128,
    mhd=False,
    solver_mode=FINITE_VOLUME,
    progress_bar=True,

    return_snapshots=True,
    # Weniger gespeicherte Zeitpunkte halten die interaktive HTML-Datei klein.
    num_snapshots=60,
    snapshot_settings=SnapshotSettings(
    return_states=True),
)

# the SimulationParams can be changed
# without causing re-compilation
params = SimulationParams(
      t_start=0.0,
      t_end=0.1,
      C_cfl=0.4,
      gamma=5 / 3,
  )

# the variable registry allows for the principled
# access of simulation variables from the state array
registered_variables = get_registered_variables(config)

# next we set up the initial state using the helper data
helper_data = get_helper_data(config)
coordinates = helper_data.geometric_centers
center = jnp.array([config.box_size / 2, config.box_size / 2, config.box_size / 2])
r = jnp.linalg.norm(coordinates - center, axis=-1)

# Sedov-Anfangszustand: homogenes Medium und rein thermische Explosion.
rho = jnp.ones_like(r)
ux = jnp.zeros_like(r)
uy = jnp.zeros_like(r)
uz = jnp.zeros_like(r)
p_amb = 1e-4
E_expl = 1.0
r_expl = 0.01
delta = 2.0 * config.box_size / config.num_cells
w = 0.5 * (1.0 - jnp.tanh((r - r_expl) / delta))

# Für das kartesische Gitter haben alle Zellen dasselbe Volumen.
cell_volume = (config.box_size / config.num_cells) ** 3
delta_p = E_expl * (params.gamma - 1.0) / (cell_volume * jnp.sum(w))
p = p_amb + delta_p * w
initial_inside_pressure = float(p_amb + delta_p)

# get initial state
initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = ux,
    velocity_y = uy,
    velocity_z = uz,
    gas_pressure = p,
)

# finalize and check the config
config = finalize_config(config, initial_state.shape)


from Praktikum_Datenzubereitung.Supernova_plot_3D import save_3d_radial_profiles_gif

# now we run the simulation
snapshots = time_integration(
    initial_state,
    config,
    params,
    registered_variables,
)

save_3d_radial_profiles_gif(
    snapshots,
    registered_variables,
    box_size=config.box_size,
    output_path="Praktikum_Datenzubereitung/supernova_3d_radial_profiles_128cells.gif",
    num_bins=128,
    interval=100,
)
