from libensemble.gen_funcs.persistent_uniform_sampling import (
    persistent_uniform)
from libensemble.gen_funcs.persistent_gp import (
    persistent_gp_gen_f, persistent_gp_mf_gen_f, persistent_gp_mf_disc_gen_f)
import libensemble.gen_funcs as libe_genf
libe_genf.rc.aposmm_optimizers = 'nlopt'
from libensemble.gen_funcs.persistent_aposmm import aposmm


gen_functions = {
    'random': persistent_uniform,
    'bo': persistent_gp_gen_f,
    'bo_mf': persistent_gp_mf_gen_f,
    'bo_mf_disc': persistent_gp_mf_disc_gen_f,
    'aposmm': aposmm
}


def get_generator_function(gen_type):
    if gen_type in gen_functions:
        return gen_functions[gen_type]
    else:
        raise ValueError(
            "Generator type '{}' not recognized.".format(gen_type))
