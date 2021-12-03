from libensemble.gen_funcs.persistent_uniform_sampling import (
    persistent_uniform)

gen_functions = {
    'random': persistent_uniform,
}

# Try to import `aposmm` if avaiable (heavier dependencies)
try:
    import libensemble.gen_funcs as libe_genf
    libe_genf.rc.aposmm_optimizers = 'nlopt'
    from libensemble.gen_funcs.persistent_aposmm import aposmm
    gen_functions['aposmm'] = aposmm
except ImportError:
    pass

# Try to import Dragonfly's Bayesian optimizers
try:
    from .dragonfly_gen_funcs import \
       persistent_gp_gen_f, persistent_gp_mf_gen_f, persistent_gp_mf_disc_gen_f
    gen_functions['bo'] = persistent_gp_gen_f
    gen_functions['bo_mf'] = persistent_gp_mf_gen_f
    gen_functions['bo_mf_disc'] = persistent_gp_mf_disc_gen_f
except ImportError:
    pass

# Try to import Ax's Multi-task optimizer
try:
    from .ax_gen_funcs import persistent_gp_mt_ax_gen_f
    gen_functions['bo_mt'] = persistent_gp_mt_ax_gen_f
except ImportError:
    pass

def get_generator_function(gen_type):
    if gen_type in gen_functions:
        return gen_functions[gen_type]
    elif gen_type == 'aposmm':
        raise RuntimeError(
        """You are trying to use APOSMM, but it could not be imported. \n
        Please make sure that the following lines work in your Python environment:

        import libensemble.gen_funcs as libe_genf
        libe_genf.rc.aposmm_optimizers = 'nlopt'
        from libensemble.gen_funcs.persistent_aposmm import aposmm
        """)
    elif gen_type in ['bo', 'bo_mf', 'bo_mf_disc']:
        raise RuntimeError(
        """You are trying to use dragonfly's Bayesian optimization, but it
        could not be imported. \n  Please make sure that the following lines
        work in your Python environment:

        import dragonfly

        If not, this can be installed with `pip install dragonfly`
        """)
    elif gen_type == 'bo_mt':
        raise RuntimeError(
        """You are trying to use Ax's multi-task optimization, but it
        could not be imported. \n Please make sure that the following lines
        work in your Python environment:

        import ax
        import pandas

        If not, this can be installed with `pip install ax-platform pandas`
        """)
    else:
        raise ValueError(
            "Generator type '{}' not recognized.".format(gen_type))
