from collections import OrderedDict
from ax.service.ax_client import AxClient

# List the names of the varying parameters, and their range of value
# The names must be the same as those used in 'template_fbpic_script.py'
# Should be specified as float if parameter is continuous
varying_parameters = OrderedDict({
    'x0': [0., 15.],
    'x1': [0., 15.],
})
