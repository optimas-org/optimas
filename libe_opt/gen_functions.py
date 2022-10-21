from .persistent_gp import (
    persistent_gp_gen_f, persistent_gp_mf_gen_f, persistent_gp_mf_disc_gen_f,
    persistent_gp_ax_gen_f, persistent_gp_mt_ax_gen_f)
from .persistent_sampling import (
    persistent_normal, persistent_uniform)


basic_functions = {
    'random': persistent_uniform,
    'normal': persistent_normal
}


dragonfly_functions = {
    'bo': persistent_gp_gen_f,
    'bo_mf': persistent_gp_mf_gen_f,
    'bo_mf_disc': persistent_gp_mf_disc_gen_f,
    'bo_mt': None
}


ax_functions = {
    'bo': persistent_gp_ax_gen_f,
    'bo_mf': persistent_gp_ax_gen_f,
    'bo_mf_disc': None,
    'bo_mt': persistent_gp_mt_ax_gen_f
}


# Try to import `aposmm` if avaiable (heavier dependencies)
try:
    import libensemble.gen_funcs as libe_genf
    libe_genf.rc.aposmm_optimizers = 'nlopt'
    from libensemble.gen_funcs.persistent_aposmm import aposmm
    basic_functions['aposmm'] = aposmm
except ImportError:
    pass


def get_generator_function(
        gen_type, bo_backend='df', use_mf=False, discrete_mf=False,
        use_mt=False):
    if gen_type in basic_functions:
        return basic_functions[gen_type]
    elif gen_type == 'aposmm':
        raise RuntimeError(
        """You are trying to use APOSMM, but that it could not be imported. \n
       Please make sure that the following lines work in your Python environment:

       import libensemble.gen_funcs as libe_genf
       libe_genf.rc.aposmm_optimizers = 'nlopt'
       from libensemble.gen_funcs.persistent_aposmm import aposmm
       """)
    elif gen_type == 'bo':
        gen_code = 'bo'
        if use_mf:
            gen_code += '_mf'
            if discrete_mf:
                gen_code += '_disc'
        elif use_mt:
            gen_code += '_mt'
        if bo_backend == 'df':
            return dragonfly_functions[gen_code]
        if bo_backend == 'ax':
            return ax_functions[gen_code]
    else:
        raise ValueError(
            "Generator type '{}' not recognized.".format(gen_type))
