import numpy as np
from libe_opt.search_methods import GridSearch

from analysis_script import analyze_simulation

var_names = ['x0', 'x1']
var_lb = np.array([0., 0.])
var_ub = np.array([15., 15.])
var_steps = np.array([5, 7])

gs = GridSearch(
    var_names=var_names,
    var_lb=var_lb,
    var_ub=var_ub,
    var_steps=var_steps,
    sim_workers=4,
    sim_template='template_simulation_script.py',
    analysis_func=analyze_simulation,
    run_async=True
)

gs.run()
