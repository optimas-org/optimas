"""
Optimization of an LPA with APOSMM/nlopt and Wake-T.

Export options can also be tested.
"""

import numpy as np

# from multiprocessing import set_start_method

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"

from libensemble.gen_classes import APOSMM
from optimas.generators import ExternalGenerator
from libensemble.tools import add_unique_random_streams

from gest_api.vocs import VOCS
from optimas.evaluators import TemplateEvaluator
from optimas.explorations import Exploration

from analysis_script import analyze_simulation

check_map_v_unmapped = True
if check_map_v_unmapped:
    from check_map_v_unmapped import check_mapped_vs_unmapped


# Number of simulation batches, their size, and the maximum number of simulations
n_batches = 10  # 8
batch_size = 4  # 24

initial_sample = batch_size  # *4
max_evals = n_batches * batch_size  # + initial_sample
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

# Create VOCS object
# start for bin results
bin_start = 4
# number of bins for structure-exploiting optimization (note this is nlopt so not using)
nbins = 10

# Build observables set with all parameters
observables_set = {
    "mean_gamma",  # arithmetic mean
    "std_gamma",  # standard deviation
    "charge",  # Track charge to see if we lost any particles
}

# Note: This nlopt example does not use these fields but this tests the setup.
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

for obs in vocs.observables:
    print(obs)


# Set up APOSMM generator
persis_info = add_unique_random_streams({}, 5)[
    1
]  # SH Dont need the 5.Better to have APOSMM defaults.
persis_info["nworkers"] = (
    nworkers  # SH - not taking account of gen_on_manager in APOSMM
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
    persis_info=persis_info,
    initial_sample_size=initial_sample,
    sample_points=np.atleast_2d(0.1 * (np.arange(n) + 1)),
    localopt_method="LN_BOBYQA",
    rk_const=1e-4,  #  0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
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
    run_async=False,  # SH - also try with True
    exploration_dir_path="./exploration_0",
)


if __name__ == "__main__":
    # set_start_method("spawn")

    # Test running export when no data (optional test)
    empty_result = aposmm.export()
    assert empty_result == (
        None,
        None,
        None,
    ), f"Expected (None, None, None) but got {empty_result}"

    # Run exploration
    exp.run()

    if exp.is_manager:
        aposmm.finalize()

        # Get data in gen format and in user format
        H, _, _ = aposmm.export()
        H_unmapped, _, _ = aposmm.export(user_fields=True)

        # H_dicts, _, _ = aposmm.export(as_dicts=True)
        # H_dicts, _, _ = aposmm.export(as_dicts=True, user_fields=True)
        # print(f"\n\nH_dicts: {H_dicts}")

        # Check data consistency if enabled
        if check_map_v_unmapped:
            check_mapped_vs_unmapped(H, H_unmapped, print_rows=True)

        # Check sampling followed by optimization runs
        assert not np.any(H_unmapped["local_pt"][:initial_sample])
        assert np.all(H_unmapped["local_pt"][initial_sample:])

        np.save("H_final.npy", H)
        np.save("H_final_unmapped.npy", H_unmapped)
