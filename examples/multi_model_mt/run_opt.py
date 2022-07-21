from libensemble.tools import parse_args
from libe_opt.ensemble_runner import run_ensemble

from varying_parameters import varying_parameters
from analysis_script import analyze_simulation, analyzed_quantities
from mt_parameters import mt_parameters


# Ensemble parameters.
gen_type = 'bo'
backend = 'ax'
n_batches = 10
sim_max = (
    (mt_parameters['n_opt_lofi'] + mt_parameters['n_opt_hifi']) * n_batches
    + mt_parameters['n_init_lofi'] + mt_parameters['n_init_hifi']
)
run_async = False
nworkers, is_master, libE_specs, _ = parse_args()


# Files to copy to each simulation folder.
libE_specs['sim_dir_copy_files'] = [
    'bunch_utils.py',
    'custom_fld_diags.py',
    'custom_ptcl_diags.py'
]


# Run optimization.
run_ensemble(
    nworkers, sim_max, is_master, gen_type, bo_backend=backend,
    analyzed_params=analyzed_quantities, var_params=varying_parameters,
    analysis_func=analyze_simulation, mt_params=mt_parameters,
    libE_specs=libE_specs, run_async=run_async)
