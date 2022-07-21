"""
Contains the function that analyzes the simulation results,
after the simulation was run.
"""
import numpy as np

# This must include all quantities calculated by this script, except from f
# These parameters are not used by libEnsemble, but they provide additional
# information / diagnostic for the user
# The third parameter is the shape of the corresponding array
analyzed_quantities = [
    ('energy_std', float, (1,)),
    # Final average energy, in MeV.
    ('energy_avg', float, (1,)),
    # Final beam charge.
    ('charge', float, (1,)),
    # Final beam emittance.
    ('emittance', float, (1,)),
]


def get_emittance(ts, t):
    import numpy as np
    w, x, ux = ts.get_particle(['w', 'x', 'ux'], t=t)
    x2 = np.average(x**2, weights=w)
    u2 = np.average(ux**2, weights=w)
    xu = np.average(x*ux, weights=w)
    return np.sqrt( x2 * u2 - xu**2 )


def analyze_simulation( simulation_directory, libE_output ):

    import os
    from openpmd_viewer.addons import LpaDiagnostics

    # Define/calculate the objective function 'f'
    # as well as the diagnostic quantities listed in `analyzed_quantities` above
    ts = LpaDiagnostics( os.path.join(simulation_directory, 'diag') )
    t0 = 4.e-11 # Time, in the boosted-frame, at which we calculate beam properties
    
    charge_i = ts.get_charge( t=0 )
    emittance_i = get_emittance( ts, t=0 )
    charge_f = ts.get_charge( t=t0 )
    emittance_f = get_emittance( ts, t=t0 )
    energy_avg, energy_std = ts.get_mean_gamma( t=t0 )

    # Here: Build a quantity to minimize (f) that encompasses
    # emittance AND charge loss 1% charge loss has the
    # same impact as doubling the initial emittance.
    # we minimize f!
    libE_output['f'] = np.log( emittance_f + emittance_i*(1.-charge_f/charge_i)*100 )
    libE_output['energy_std'] = energy_std
    libE_output['energy_avg'] = energy_avg
    libE_output['charge'] = charge_f
    libE_output['emittance'] = emittance_f

    return libE_output
