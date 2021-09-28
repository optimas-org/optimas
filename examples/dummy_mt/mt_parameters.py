""""
Contains the parameters for multi-fidelity optimization.
"""

mt_parameters = {
    'name_hifi': 'expensive_model',
    'name_lofi': 'cheap_model',
    'n_init_hifi': 3,
    'n_init_lofi': 10,
    'n_opt_hifi': 1,
    'n_opt_lofi': 10,
    'extra_args_lofi': '',
    'extra_args_hifi': ''
}
