import numpy as np
from libe_opt.search_methods import MultifidelityBayesianOptimization
from analysis_script import analyze_simulation

var_names = ['x0', 'x1']
var_lb = np.array([0., 0.])
var_ub = np.array([15., 15.])


mfbo = MultifidelityBayesianOptimization(
    var_names=var_names,
    var_lb=var_lb,
    var_ub=var_ub,
    sim_workers=4,
    sim_template='template_simulation_script.py',
    analysis_func=analyze_simulation,
    sim_number=10,
    fidel_name='resolution',
    fidel_lb=1.,
    fidel_ub=8.,
    fidel_cost_intercept=2.
)

mfbo.run()
