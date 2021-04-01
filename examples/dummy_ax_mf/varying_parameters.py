from collections import OrderedDict
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models

# List the names of the varying parameters, and their range of value
# The names must be the same as those used in 'template_fbpic_script.py'
# Should be specified as float if parameter is continuous
varying_parameters = OrderedDict({
    'x0': [0., 15.],
    'x1': [0., 15.],
    'resolution': [1., 8.]
})

parameters = list()
for key, value in varying_parameters.items():
    parameters.append(
        {
            'name': key,
            "type": "range",
            "bounds": value
        }
    )
# Specify fidelity
parameters[-1]['is_fidelity'] = True
parameters[-1]['target_value'] = 8.

# Specify generation strategy ( 5 runs quasi-random, then mf-knowledge gradient
gs = GenerationStrategy(
steps=[
    GenerationStep(model=Models.SOBOL,num_trials = 5),
    GenerationStep(model=Models.GPKG, num_trials=-1,
                   model_kwargs={'cost_intercept': 2.})
])

ax_client = AxClient(generation_strategy=gs)
ax_client.create_experiment(
    parameters= parameters,
    objective_name="f",
    minimize=True,
)