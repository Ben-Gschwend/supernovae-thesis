"""Kleine CPU-Funktionstests; keine hydrodynamische Simulation.

Prüft Grenzen der aktuell installierten astronomix-Implementierung.
Die synthetischen Flüsse sind ein Gegenbeispiel für die Schutzgarantie,
keine Rekonstruktion des konkreten Supernova-Abbruchs.
"""
import os

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import jax.numpy as jnp
import numpy as np
from astronomix import SimulationConfig, SimulationParams, get_registered_variables, FINITE_DIFFERENCE
from astronomix.option_classes.simulation_config import PositivityConfig
from astronomix._finite_difference._interface_fluxes._flux_blending import _blend_interface_flux
from astronomix._integrators._explicit_rk import ssprk4


def main():
    print('Geräte:', jax.devices())
    config = SimulationConfig(dimensionality=3, num_cells=4, mhd=False,
                              solver_mode=FINITE_DIFFERENCE,
                              positivity_config=PositivityConfig(preserving_flux=True))
    variables = get_registered_variables(config)
    params = SimulationParams()
    q = jnp.zeros((5, 4, 4, 4)).at[variables.density_index].set(1.0)
    q = q.at[variables.energy_index].set(1.5)
    changes = []
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = 4
        f = jnp.zeros_like(q).at[variables.density_index].set(
            jnp.broadcast_to(12.0 * jnp.arange(4).reshape(shape), (4, 4, 4)))
        limited = _blend_interface_flux(f, q, axis, 0.05, params, config, variables)
        delta = -0.05 * (limited - jnp.roll(limited, 1, axis=axis + 1))
        changes.append(delta)
        print('Achse', axis, 'Einzelupdate rho:', float((q + delta)[0, 1, 1, 1]))
    print('Summe aller Achsen rho:', float((q + sum(changes))[0, 1, 1, 1]))

    # Ereignisfolge mit ausgeschaltetem JIT: Hook-Vertrag vor jeder RHS?
    events = []
    def pre(u):
        events.append('Schutz')
        return u
    def rhs(u, dt):
        events.append('RHS')
        return jnp.zeros_like(u)
    with jax.disable_jit():
        ssprk4(jnp.array(1.0), 0.1, rhs=rhs, pre_stage=pre)
    print('RK-Aufrufe:', events)
    print('float32: (1000 + 1.5e-14) - 1000 =',
          np.float32(np.float32(1000) + np.float32(1.5e-14)) - np.float32(1000))


if __name__ == '__main__':
    main()
