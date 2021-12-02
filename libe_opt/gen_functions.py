from libensemble.gen_funcs.persistent_uniform_sampling import (
    persistent_uniform)
from .persistent_gp import (
    persistent_gp_gen_f, persistent_gp_mf_gen_f, persistent_gp_mf_disc_gen_f,
    persistent_gp_mt_ax_gen_f)
from .persistent_evenly_spaced import persistent_evenly_spaced

gen_functions = {
    'random': persistent_uniform,
    'evenly_spaced': persistent_evenly_spaced,
    'bo': persistent_gp_gen_f,
    'bo_mf': persistent_gp_mf_gen_f,
    'bo_mf_disc': persistent_gp_mf_disc_gen_f,
    'bo_mt': persistent_gp_mt_ax_gen_f
}

# Try to import `aposmm` if avaiable (heavier dependencies)
try:
    import libensemble.gen_funcs as libe_genf
    libe_genf.rc.aposmm_optimizers = 'nlopt'
    from libensemble.gen_funcs.persistent_aposmm import aposmm
    gen_functions['aposmm'] = aposmm
except ImportError:
    pass

def get_generator_function(gen_type):
    if gen_type in gen_functions:
        return gen_functions[gen_type]
    elif gen_type == 'aposmm':
        raise RuntimeError(
        """You are trying to use APOSMM, but that it could not be imported. \n
       Please make sure that the following lines work in your Python environment:

       import libensemble.gen_funcs as libe_genf
       libe_genf.rc.aposmm_optimizers = 'nlopt'
       from libensemble.gen_funcs.persistent_aposmm import aposmm
       """)
    else:
        raise ValueError(
            "Generator type '{}' not recognized.".format(gen_type))
