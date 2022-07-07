""""
Contains the parameters for multi-fidelity optimization.
"""

mt_parameters = {
    'name_hifi': 'fbpic',
    'name_lofi': 'wake-t',
    'n_init_hifi': 4,
    'n_init_lofi': 20,
    'n_opt_hifi': 4,
    'n_opt_lofi': 20,    
    'extra_args_lofi': '-np 1 --bind-to none',
    'extra_args_hifi': '-np 1 --bind-to none'
}