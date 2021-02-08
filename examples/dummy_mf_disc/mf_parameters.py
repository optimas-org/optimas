""""
Contains the parameters for multi-fidelity optimization.
"""

mf_parameters = {
    'name': 'resolution',
    'range': [1, 2, 4, 8],
    'discrete': True,
    'cost_func': lambda z: int(z[0][0])**2
}
