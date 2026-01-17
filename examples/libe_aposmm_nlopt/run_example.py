"""Optimization of an LPA with APOSMM/nlopt and Wake-T."""

import pickle
import numpy as np

# from multiprocessing import set_start_method

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"

from libensemble.gen_classes import APOSMM
from optimas.generators import ExternalGenerator

from gest_api.vocs import VOCS
from optimas.evaluators import TemplateEvaluator
from optimas.explorations import Exploration

from analysis_script import analyze_simulation


# Number of simulation batches, their size, and the maximum number of simulations
n_batches = 10
batch_size = 4

initial_sample = batch_size  # *4
max_evals = n_batches * batch_size + initial_sample
nworkers = batch_size


# Create varying parameters and objectives.
mcr = 1e-2  # minimal current ratio

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

# Build observables set
observables_set = {
    "mean_gamma",
    "std_gamma",
    "charge",
}

vocs = VOCS(
    variables=vars_std,
    objectives={"f": "MINIMIZE"},
    observables=observables_set,
)

variables_mapping = {
    "x": ["beam_i_r2", "beam_z_i_2", "beam_length"],
    "x_on_cube": [
        "beam_i_r2_on_cube",
        "beam_z_i_2_on_cube",
        "beam_length_on_cube",
    ],
}

aposmm = APOSMM(
    vocs=vocs,
    variables_mapping=variables_mapping,
    initial_sample_size=initial_sample,
    sample_points=np.atleast_2d(0.1 * (np.arange(n) + 1)),
    localopt_method="LN_BOBYQA",
    rk_const=1e-4,
    run_max_eval=100 * (n + 1),
    max_active_runs=batch_size,
    dist_to_bound_multiple=0.5,
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
