from collections import OrderedDict

# List the names of the varying parameters, and their range of value
# The names must be the same as those used in 'template_fbpic_script.py'
varying_parameters = OrderedDict({
    'beam_i_1': [0.1, 10],  # kA
    'beam_i_2': [0.1, 10],  # kA
    'beam_z_i_2': [-10, 10],  # µm
    'beam_length': [1, 20]  # µm
})