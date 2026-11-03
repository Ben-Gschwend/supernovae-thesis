"""Radiale 1D-Supernova-Simulation mit astronomix."""

import jax.numpy as jnp
from astronomix import (
    SimulationConfig, SimulationParams,
    get_helper_data, finalize_config,
    get_registered_variables, construct_primitive_state,
    time_integration,
    SPHERICAL,
    FINITE_VOLUME,
)
from astronomix.option_classes.simulation_config import SnapshotSettings

# the SimulationConfig holds static 
# configuration parameters
config = SimulationConfig(
    geometry=SPHERICAL,
    dimensionality=1,
    box_size = 1.0,
    # Erster höher aufgelöster 1D-Vergleich gegenüber dem bisherigen Lauf.
    num_cells = 128,
    mhd=False,
    solver_mode=FINITE_VOLUME,
    progress_bar=True,

    return_snapshots=True,
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
r = helper_data.geometric_centers
cell_volumes = helper_data.cell_volumes
rho = jnp.ones_like(r)
u = jnp.zeros_like(r)
p_amb = 1e-4
E_expl = 1.0
r_expl = 0.01
delta = 2.0 * config.box_size / config.num_cells
#w = jnp.where(r < r_expl, 1.0, 0.0)
w = 0.5 * (1.0 - jnp.tanh((r - r_expl) / delta))
delta_p = (E_expl * (params.gamma - 1.0) / jnp.sum(cell_volumes * w))
p = p_amb + delta_p * w

# get initial state
initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    # In 1D ist die einzige Geschwindigkeitskomponente die radiale x-Komponente.
    velocity_x = u,
    gas_pressure = p
)

# finalize and check the config
config = finalize_config(config, initial_state.shape)


from Praktikum_Datenzubereitung.Supernova_plot_1D import (
    save_1d_profiles_gif,
)

# now we run the simulation
snapshots = time_integration(
    initial_state,
    config,
    params,
    registered_variables,
)


save_1d_profiles_gif(
    snapshots,
    registered_variables,
    radius=r,
    output_path="Praktikum_Datenzubereitung/supernova_1d_profiles_128cells.gif",
    interval=100,
)

 
