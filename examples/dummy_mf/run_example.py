from libe_opt.core import VaryingParameter, Objective
from libe_opt.generators import AxMultiFidelityGenerator
from libe_opt.evaluators import TemplateEvaluator
from libe_opt.explorations import Exploration


def analyze_simulation(simulation_directory, output_params):
    """Function that analyzes the simulation output and fills in the
    dictionary of output parameters."""
    # Read back result from file
    with open('result.txt') as f:
        result = float(f.read())
    # Fill in output parameters.
    output_params['f'] = result
    return output_params


# Create varying parameters (including fidelity) and objectives.
var_1 = VaryingParameter('x0', 0., 15.)
var_2 = VaryingParameter('x1', 0., 15.)
res = VaryingParameter('resolution', 1., 8., is_fidelity=True,
                       fidelity_target_value=8.)
obj = Objective('f', minimize=True)


# Create generator.
gen = AxMultiFidelityGenerator(
    varying_parameters=[var_1, var_2, res],
    objectives=[obj],
    n_init=4,
    fidel_cost_intercept=2.
)


# Create evaluator.
ev = TemplateEvaluator(
    sim_template='template_simulation_script.py',
    analysis_func=analyze_simulation
)


# Create exploration.
exp = Exploration(
    generator=gen,
    evaluator=ev,
    max_evals=10,
    sim_workers=2,
    run_async=True
)


# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == '__main__':
    exp.run()
