import os
import numpy as np
import visualpic as vp


def analyze_simulation(simulation_directory, output_params):

    # Load simulation data.
    dc = vp.DataContainer(
        'openpmd',
        os.path.join(simulation_directory, 'diags/hdf5')
    )
    dc.load_data()

    # Get bunch charge and longitudinal momentum.
    bunch = dc.get_species('bunch')
    ts = bunch.timesteps
    bunch_data = bunch.get_data(ts[-1], ['pz', 'q'])
    pz = bunch_data['pz'][0]
    q = bunch_data['q'][0]

    # Filter out any charge below ~50 MeV.
    pz_filter = np.where(pz > 100)
    pz = pz[pz_filter]
    q = q[pz_filter]

    # Calculate relevant quantities.
    q_tot = np.abs(np.sum(q)) * 1e12  # pC
    q_ref = 10  # pC
    med, mad = weighted_mad(pz * 0.511, q)
    mad_rel = mad/med
    med *= 1e-3  # GeV
    mad_rel_ref = 1e-2

    # Calculate value of objective function.
    f = np.log(med * q_tot / q_ref / (mad_rel / mad_rel_ref))

    # Fill output dictionary.
    output_params['f'] = -f
    output_params['charge'] = q_tot
    output_params['energy_med'] = med
    output_params['energy_mad'] = mad

    # For convenience, save value of objective to text file.
    np.savetxt('f.txt', np.array([f]))

    return output_params


def weighted_mad(x, w):
    med = weighted_median(x, w)
    mad = weighted_median(np.abs(x-med), w)
    return med, mad


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
    quantile = .5
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
    # assert Sn != 0, "The sum of the weights must not be zero"
    Pn = (Sn-0.5*sorted_weights)/Sn[-1]
    # Get the value of the weighted median
    return np.interp(quantile, Pn, sorted_data)
