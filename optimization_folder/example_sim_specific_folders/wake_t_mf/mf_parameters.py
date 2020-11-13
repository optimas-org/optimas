""""
Contains the parameters for multi-fidelity optimization.

"""

fidelity_name = 'model'


fidelity_range = ["wake-t", "fbpic"]


disctete_fidelity = True


def cost_func(fidel_list):
    fidel = fidel_list[0][0]
    if fidel == fidelity_range[0]:
        return 1
    elif fidel == fidelity_range[1]:
        return 300


mf_parameters = {
    'name': fidelity_name,
    'range': fidelity_range,
    'discrete': disctete_fidelity,
    'cost_func': cost_func
}
