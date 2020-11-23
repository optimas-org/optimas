""""
Contains the parameters for multi-fidelity optimization.
"""
# Computational cost: depends on which model is used
def cost_func(fidel_list):
    fidel = fidel_list[0][0]
    if fidel == "wake-t":
        return 1
    elif fidel == "fbpic":
        return 300

mf_parameters = {
    'name': 'model',
    'range': ["wake-t", "fbpic"],
    'discrete': True,
    'cost_func': cost_func
}
