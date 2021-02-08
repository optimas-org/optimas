from collections import OrderedDict

# List the names of the varying parameters, and their range of value
# The names must be the same as those used in 'template_fbpic_script.py'
varying_parameters = OrderedDict({
    'laser_scale': [0.7, 1.05],
    'z_foc': [3, 7.5],
    'mult': [0.6, 0.8],
    'plasma_scale': [0.1, 1.5]
})
