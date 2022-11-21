import numpy as np
from libe_opt.search_methods import BayesianOptimization

from analysis_script import analyze_simulation, analyzed_quantities

var_names = ['x0', 'x1']
var_lb = np.array([0., 0.])
var_ub = np.array([15., 15.])

bo = BayesianOptimization(
    var_names=var_names,
    var_lb=var_lb,
    var_ub=var_ub,
    sim_workers=4,
    sim_template='template_simulation_script.py',
    analysis_func=analyze_simulation,
    analyzed_params=analyzed_quantities,
    sim_number=10,
    run_async=True
)

bo.run()
