from libensemble.tools import parse_args
from libe_opt.ensemble_runner import run_ensemble

from varying_parameters import varying_parameters, ax_client
from analysis_script import analyze_simulation, analyzed_quantities


gen_type = 'ax'
sim_max = 10
run_async = True
nworkers, is_master, libE_specs, _ = parse_args()

run_ensemble(
    nworkers, sim_max, is_master, gen_type,
    analyzed_params=analyzed_quantities, var_params=varying_parameters,
    analysis_func=analyze_simulation, libE_specs=libE_specs,
    run_async=run_async, ax_client=ax_client
)

ax_client.save_to_json_file('ax_client.json')
