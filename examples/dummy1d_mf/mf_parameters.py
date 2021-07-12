""""
Contains the parameters for multi-fidelity optimization.
"""

mf_parameters = {
    'name': 'resolution',
    'range': [1, 2],
    'discrete': False,
    'cost_func': lambda z: z[0]**2
}
