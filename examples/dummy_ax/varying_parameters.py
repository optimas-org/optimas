from collections import OrderedDict
from ax.service.ax_client import AxClient

# List the names of the varying parameters, and their range of value
# The names must be the same as those used in 'template_fbpic_script.py'
# Should be specified as float if parameter is continuous
varying_parameters = OrderedDict({
    'x0': [0., 15.],
    'x1': [0., 15.],
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

ax_client = AxClient()
ax_client.create_experiment(
    parameters= parameters,
    objective_name="f",
    minimize=True,
)