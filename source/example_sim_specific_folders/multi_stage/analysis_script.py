"""
Contains the function that analyzes the simulation results,
after the simulation was run.
"""
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


def analyze_simulation( simulation_directory, libE_output ):

    import os
    from openpmd_viewer.addons import LpaDiagnostics

    # Define/calculate the objective function 'f'
    # as well as the diagnostic quantities listed in `analyzed_quantities` above
    ts = LpaDiagnostics( os.path.join(simulation_directory, 'lab_diags/hdf5') )

    select = {'x':[-100.e-6,100.e-6], 'y':[-100.e-6, 100.e-6]}
    charge_i = ts.get_charge( iteration=ts.iterations[0], select=select )
    emittance_i = ts.get_emittance( iteration=ts.iterations[0],
                                    select=select )[0]
    charge_f = ts.get_charge( iteration=ts.iterations[-1],
                              select=select )
    emittance_f = ts.get_emittance( iteration=ts.iterations[-1],
                                    select=select )[0]
    energy_avg, energy_std = ts.get_mean_gamma(
        iteration=ts.iterations[-1], select=select )

    # Here: Build a quantity to minimize (f) that encompasses
    # emittance AND charge loss 1% charge loss has the
    # same impact as doubling the initial emittance.
    # we minimize f!
    libE_output['f'] = emittance_f + emittance_i*(1.-charge_f/charge_i)*100
    libE_output['energy_std'] = energy_std
    libE_output['energy_avg'] = energy_avg
    libE_output['charge'] = charge_f
    libE_output['emittance'] = emittance_f

    return libE_output
