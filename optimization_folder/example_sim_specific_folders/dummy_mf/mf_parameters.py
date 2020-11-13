""""
Contains the parameters for multi-fidelity optimization.

"""

fidelity_name = 'resolution'


fidelity_range = [1, 8]


disctete_fidelity = False


cost_func = lambda z: z[0]**2


mf_parameters = {
    'name': fidelity_name,
    'range': fidelity_range,
    'discrete': disctete_fidelity,
    'cost_func': cost_func
}
