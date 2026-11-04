"""Erste vereinfachte 2D-Supernova-Simulation mit astronomix.

Dieses Skript wird schrittweise selbst entwickelt. Der Simulationscode ist
noch nicht implementiert.
"""

# TODO: Imports und SimulationConfig gemeinsam ausarbeiten.
from pathlib import Path

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
    dimensionality=2,
    box_size = 1.0,
    num_cells = 100,
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
      t_end=0.2,
      C_cfl=0.4,
      gamma=5 / 3,
  )

# the variable registry allows for the principled
# access of simulation variables from the state array
registered_variables = get_registered_variables(config)

# next we set up the initial state using the helper data
helper_data = get_helper_data(config)
coordinates = helper_data.geometric_centers
center = jnp.array([config.box_size / 2, config.box_size / 2,])
r = jnp.linalg.norm(coordinates - center, axis=-1)
explosion_radius = 0.1
rho = jnp.where(r < explosion_radius, 1.0, 0.1)
ux = jnp.zeros_like(r)
uy = jnp.zeros_like(r)
p = jnp.where(r < explosion_radius, 1.0, 0.1)

# get initial state
initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = ux,
    velocity_y = uy,
    gas_pressure = p,
)

# finalize and check the config
config = finalize_config(config, initial_state.shape)


try:
    from Praktikum_Datenzubereitung.Supernova_plot import animate_snapshots
except ModuleNotFoundError:
    # Kompatibilität mit dem dokumentierten Direktstart aus diesem Ordner.
    from Supernova_plot import animate_snapshots

# now we run the simulation
snapshots = time_integration(
    initial_state,
    config,
    params,
    registered_variables,
)

animate_snapshots(
    snapshots,
    registered_variables,
    field="pressure",
    output_path=Path(__file__).resolve().parent / "supernova_pressure.gif",
    interval=100,
)

