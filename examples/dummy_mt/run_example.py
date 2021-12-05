from libensemble.tools import parse_args
from libe_opt.ensemble_runner import run_ensemble

from varying_parameters import varying_parameters
from analysis_script import analyze_simulation, analyzed_quantities
from mt_parameters import mt_parameters


gen_type = 'bo'
backend = 'ax'
sim_max = 25
run_async = False
nworkers, is_master, libE_specs, _ = parse_args()

run_ensemble(
    nworkers, sim_max, is_master, gen_type,
    analyzed_params=analyzed_quantities, var_params=varying_parameters,
    analysis_func=analyze_simulation, mt_params=mt_parameters,
    libE_specs=libE_specs, run_async=run_async, bo_backend=backend)
