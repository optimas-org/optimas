"""Optimization of an LPA with APOSMM/IBCDFO and Wake-T."""

import sys
import pickle
import numpy as np
import jax
from gest_api.vocs import VOCS

jax.config.update("jax_enable_x64", True)

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "ibcdfo_pounders"

from libensemble.gen_classes import APOSMM

from optimas.generators import ExternalGenerator
from optimas.evaluators import TemplateEvaluator
from optimas.explorations import Exploration

from analysis_script import analyze_simulation

try:
    from ibcdfo import run_pounders  # noqa: F401
except ModuleNotFoundError:
    sys.exit("Please 'pip install ibcdfo'")
try:
    from minqsw import minqsw  # noqa: F401
except ModuleNotFoundError:
    sys.exit(
        "Ensure https://github.com/POptUS/minq has been cloned and that minq/py/minq5/ is on the PYTHONPATH"
    )


def compute_f_from_bins(F, q_ref=200, energy_spread_ref=5e-3):
    """Compute objective from binned particle data."""
    n = (F.size - 1) // 2
    q_tot = F[0]
    bin_gammas = F[1 : n + 1]
    bin_nparts = F[n + 1 :]
    epsilon = 1e-23
    mean_gamma = jax.numpy.average(bin_gammas, weights=bin_nparts)
    std_gamma = jax.numpy.sqrt(
        jax.numpy.average((bin_gammas - mean_gamma) ** 2, weights=bin_nparts)
    )
    energy_spread = std_gamma / mean_gamma
    f = (
        -1
        * (mean_gamma + epsilon)
        * (q_tot + epsilon)
        / q_ref
        / ((energy_spread + epsilon) / energy_spread_ref)
    )
    return f


def hfun(z):
    """Compute objective function from observables."""
    return compute_f_from_bins(z)


@jax.jit
def hfun_d(z, zd):
    """Compute first derivative using JAX automatic differentiation."""
    return jax.jvp(hfun, (z,), (zd,))[1]


@jax.jit
def hfun_dd(z, zd, zdt, zdd):
    """Compute second derivative using JAX automatic differentiation."""
    _, resdd = jax.jvp(hfun_d, (z, zd), (zdt, zdd))
    return resdd


def G_combine(Cres, Gres):
    """Combine gradient models for structure-exploiting optimization."""
    n, m = Gres.shape
    G = np.array([hfun_d(Cres, Gres[i, :]) for i in range(n)])
    return G


def H_combine(Cres, Gres, Hres):
    """Combine Hessian models for structure-exploiting optimization."""
    n, _, m = Hres.shape
    H = np.array(
        [
            [
                hfun_dd(Cres, Gres[i, :], Gres[j, :], Hres[i, j, :])
                for j in range(n)
            ]
            for i in range(n)
        ]
    )
    return H


def combinemodels_jax(Cres, Gres, Hres):
    """Combine separate surrogate models into objective model."""
    return G_combine(Cres, Gres), H_combine(Cres, Gres, Hres)


# Number of simulation batches, their size, and the maximum number of simulations
n_batches = 10
batch_size = 4

initial_sample = batch_size  # *4
max_evals = n_batches * batch_size + initial_sample
nworkers = batch_size


# Create varying parameters and objectives.
mcr = 1e-2  # minimal current ratio
# back current ratio: sum with front equals to 1 and they get doubled and multiplied with the average current

# Single source of truth for variable definitions
vars_std = {
    "beam_i_r2": [mcr, 1.0 - mcr],
    "beam_z_i_2": [-20.0, 20.0],  # µm
    "beam_length": [1.0, 20.0],  # µm
    "beam_i_r2_on_cube": [0, 1.0],
    "beam_z_i_2_on_cube": [0, 1.0],
    "beam_length_on_cube": [0, 1.0],
}
n = 3

# start for bin results
bin_start = 4
# number of bins
nbins = 10

# Build observables set with all parameters
observables_set = {
    "mean_gamma",  # arithmetic mean
    "std_gamma",  # standard deviation
    "charge",  # Track charge to see if we lost any particles
}

# Add bin results to observables
for i in range(nbins):
    observables_set.add(f"bin_gammas_{i+1}")  # average gammas per bin

for i in range(10):
    observables_set.add(f"bin_nparts_{i+1}")  # number of particles per bin

vocs = VOCS(
    variables=vars_std,
    objectives={"f": "MINIMIZE"},
    observables=observables_set,
)

bounds = np.array(vocs.bounds)
LB = bounds[:n, 0]  # Lower bounds
UB = bounds[:n, 1]  # Upper bounds

pts_in_unit_cube = 0.5 * np.ones((1, 3))
pts_in_original_domain = pts_in_unit_cube * (UB - LB) + LB

np.random.seed(1234)
rand_dir = np.random.normal(size=3)
rand_dir = rand_dir / np.linalg.norm(rand_dir)

fvec_vars = (
    ["charge"]
    + [f"bin_gammas_{i+1}" for i in range(nbins)]
    + [f"bin_nparts_{i+1}" for i in range(nbins)]
)
variables_mapping = {
    "x": ["beam_i_r2", "beam_z_i_2", "beam_length"],
    "x_on_cube": [
        "beam_i_r2_on_cube",
        "beam_z_i_2_on_cube",
        "beam_length_on_cube",
    ],
    "fvec": fvec_vars,
}

aposmm = APOSMM(
    vocs=vocs,
    variables_mapping=variables_mapping,
    initial_sample_size=initial_sample,
    sample_points=pts_in_original_domain,
    localopt_method="ibcdfo_pounders",
    run_max_eval=20 * (n + 1),
    max_active_runs=batch_size,
    stop_after_k_runs=1,
    dist_to_bound_multiple=0.05,
    rk_const=1e-4,
    hfun=hfun,
    components=len(fvec_vars),
    combinemodels=combinemodels_jax,
    ftol_rel=1e-8,
)

# Create generator.
gen = ExternalGenerator(
    ext_gen=aposmm,
    vocs=vocs,
    save_model=True,
)

# Create evaluator.
ev = TemplateEvaluator(
    sim_template="template_simulation_script.py",
    analysis_func=analyze_simulation,
    sim_files=["bunch_utils.py"],
)

# Create exploration.
exp = Exploration(
    generator=gen,
    evaluator=ev,
    max_evals=max_evals,
    sim_workers=nworkers,
    run_async=True,
    exploration_dir_path="./exploration_0",
)


if __name__ == "__main__":
    # Run exploration
    exp.run()

    if exp.is_manager:
        aposmm.finalize()

        # Obtain APOSMM history which includes local minima information
        aposmm_hist, persis_info, _ = aposmm.export(user_fields=True)
        np.save("aposmm_hist.npy", aposmm_hist)
        pickle.dump(persis_info, open("aposmm_persis_info.pickle", "wb"))

        # Check sampling followed by optimization runs
        assert not np.any(aposmm_hist["local_pt"][:initial_sample])
        assert np.all(aposmm_hist["local_pt"][initial_sample:])

        # Check local_min field present and count number of local minima
        assert (
            "local_min" in aposmm_hist.dtype.names
        ), "local_min field not found in history array"
        n_local_minima = np.sum(aposmm_hist["local_min"])
        print(
            f"\nFound {n_local_minima} local minima in the optimization history."
        )
