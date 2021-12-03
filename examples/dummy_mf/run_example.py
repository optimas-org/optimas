from libensemble.tools import parse_args
from libe_opt.ensemble_runner import run_ensemble

from varying_parameters import varying_parameters
from analysis_script import analyze_simulation, analyzed_quantities
from mf_parameters import mf_parameters


gen_type = 'bo'
backend = 'df'
sim_max = 10
run_async = True
nworkers, is_master, libE_specs, _ = parse_args()

run_ensemble(
    nworkers, sim_max, is_master, gen_type,
    analyzed_params=analyzed_quantities, var_params=varying_parameters,
    mf_params=mf_parameters, analysis_func=analyze_simulation,
    libE_specs=libE_specs, run_async=run_async, bo_backend=backend)
