""""
Contains the parameters for multi-fidelity optimization.
"""
mf_parameters = {
    'name': 'resolution',
    'range': [2., 4.],
    'discrete': True,
    'cost_func': lambda z: z[0][0]**3
}
