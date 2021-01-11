from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc


alloc_functions = {
    'only_persistent': only_persistent_gens,
    'persistent_aposmm': persistent_aposmm_alloc
}


def get_alloc_function(alloc_type):
    if alloc_type in alloc_functions:
        return alloc_functions[alloc_type]
    else:
        raise ValueError(
            "Allocation type '{}' not recognized.".format(alloc_type))
