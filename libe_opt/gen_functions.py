
def get_generator_function(gen_type):
    if gen_type == 'aposmm':
        import libensemble.gen_funcs
        libensemble.gen_funcs.rc.aposmm_optimizers = 'nlopt'
        from libensemble.gen_funcs.persistent_aposmm import aposmm
        return aposmm
    elif gen_type == 'random':
        from libensemble.gen_funcs.persistent_uniform_sampling import \
            persistent_uniform
        return persistent_uniform
    elif gen_type in 'bo':
        from libensemble.gen_funcs.persistent_gp import persistent_gp_gen_f
        return persistent_gp_gen_f
    elif gen_type in 'bo_mf':
        from libensemble.gen_funcs.persistent_gp import persistent_gp_mf_gen_f
        return persistent_gp_mf_gen_f
    elif gen_type in 'bo_mf_disc':
        from libensemble.gen_funcs.persistent_gp import persistent_gp_mf_disc_gen_f
        return persistent_gp_mf_disc_gen_f
    else:
        raise ValueError(
            "Generator type '{}' not recognized.".format(gen_type))
