"""
Contains the function that analyzes the simulation results,
after the simulation was run.
"""
# This must include all quantities calculated by this script, except from f
# These parameters are not used by libEnsemble, but they provide additional
# information / diagnostic for the user
# The third parameter is the shape of the corresponding array
analyzed_quantities = [
    ('energy_med', float, (1,)),
    # Final average energy, in MeV.
    ('energy_mad', float, (1,)),
    # Final beam charge.
    ('charge', float, (1,)),
]

import numpy as np
from scipy.constants import e

def weighted_median(data, weights):
    """
    Compute the weighted quantile of a 1D numpy array.
    Parameters
    ----------
    data : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.
    Returns
    -------
    quantile_1D : float
        The output value.
    """
    quantile=.5
    # Check the data
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    nd = data.ndim
    if nd != 1:
        raise TypeError("data must be a one dimensional array")
    ndw = weights.ndim
    if ndw != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if ((quantile > 1.) or (quantile < 0.)):
        raise ValueError("quantile must have a value between 0. and 1.")
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    # TODO: Check that the weights do not sum zero
    #assert Sn != 0, "The sum of the weights must not be zero"
    Pn = (Sn-0.5*sorted_weights)/Sn[-1]
    # Get the value of the weighted median
    return np.interp(quantile, Pn, sorted_data)

def weighted_mad(x, w):
    med = weighted_median(x,w)
    mad = weighted_median(np.abs(x-med), w)
    return med, mad

def analyze_simulation( simulation_directory, libE_output ):

    import os
    from openpmd_viewer.addons import LpaDiagnostics

    # Define/calculate the objective function 'f'
    # as well as the diagnostic quantities listed in `analyzed_quantities` above
    d = LpaDiagnostics( os.path.join(simulation_directory, 'diags/hdf5') )

    uz,w = d.get_particle(['uz','w'], iteration=1,
                select={'uz':[10,None],'x':[-15e-6,15e-6],'y':[-15e-6,15e-6]})
    q = w.sum()*e*1e12
    if len(w) < 2: # Need at least two particles to calculate the energy spread
        libE_output['f'] = 0
    else:
        med, mad = weighted_mad(uz/2, w)
        libE_output['f'] = -np.sqrt(q)*med/mad/100
        libE_output['charge'] = q
        libE_output['energy_med'] = med
        libE_output['energy_mad'] = mad

    return libE_output
