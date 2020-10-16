"""
This file defines the `gen_f` and `alloc_f` for Bayesian optimization with
a Gaussian process.

- The `gen_f` is called once by a dedicated worker and only returns at the end
  of the whole libEnsemble run.
- The `alloc_f` is called by the manager. It receives asks for parameters
  to try, from the `gen_f` and in turns sends the results of simulations.

Thus the `gen_f` and `alloc_f` are closely linked, since they communicate
with each other.
"""

import numpy as np
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg

# import dragonfly Gaussian Process functions
from dragonfly.exd.domains import EuclideanDomain
from dragonfly.exd.experiment_caller import EuclideanFunctionCaller
from dragonfly.opt.gp_bandit import EuclideanGPBandit
from argparse import Namespace

def persistent_gp_gen_f( H, persis_info, gen_specs, libE_info ):
    """
    Create a Gaussian Process model, update it as new simulation results
    are available, and generate inputs for the next simulations.

    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    # Extract bounds of the parameter space, and batch size
    ub_list = gen_specs['user']['ub']
    lb_list = gen_specs['user']['lb']
    batch_size = gen_specs['user']['gen_batch_size']

    # Initialize the dragonfly GP optimizer
    domain = EuclideanDomain( [ [l,u] for l,u in zip(lb_list, ub_list) ] )
    func_caller = EuclideanFunctionCaller(None, domain)
    opt = EuclideanGPBandit( func_caller, ask_tell_mode=True,
          options=Namespace(acq='ts', build_new_model_every=batch_size) )
    opt.initialise()

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:

        # TODO: Periodically re-fit the hyperparameters of the GP

        # Ask the optimizer to generate `batch_size` new points
        # Store this information in the format expected by libE
        H_o = np.zeros(batch_size, dtype=gen_specs['out'])
        for i in range(batch_size):
            x = opt.ask()
            H_o['x'][i] = x

        # Send data and get results from finished simulation
        # Is this call blocking? Does it only continue once the
        # the manager calls `gen_work`?
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)
        if calc_in is not None:
            # Update the GP with latest simulation results
            for i in range(batch_size):
                x = calc_in['x'][i]
                y = calc_in['f'][i]
                opt.tell([ (x, -y) ])
            # Update hyperparameters
            opt._build_new_model()

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
