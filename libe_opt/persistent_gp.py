"""
This file defines the `gen_f` for Bayesian optimization with a Gaussian process.

The `gen_f` is called once by a dedicated worker and only returns at the end
of the whole libEnsemble run.

This `gen_f` is meant to be used with the `alloc_f` function
`only_persistent_gens`
"""
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg

# import dragonfly Gaussian Process functions
from dragonfly.exd.domains import EuclideanDomain
from dragonfly.exd.experiment_caller import (EuclideanFunctionCaller,
                                             CPFunctionCaller)
from dragonfly.opt.gp_bandit import EuclideanGPBandit, CPGPBandit
from dragonfly.exd.cp_domain_utils import load_config
from argparse import Namespace
from ax import Metric, Runner
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.parameter import RangeParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.core.optimization_config import OptimizationConfig
from ax.core.objective import Objective
from ax.modelbridge.factory import get_sobol, get_MTGP
from ax.core.observation import ObservationFeatures


def persistent_gp_gen_f(H, persis_info, gen_specs, libE_info):
    """
    Create a Gaussian Process model, update it as new simulation results
    are available, and generate inputs for the next simulations.

    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    # Extract bounds of the parameter space, and batch size
    ub_list = gen_specs['user']['ub']
    lb_list = gen_specs['user']['lb']

    # Number of points to generate initially
    number_of_gen_points = gen_specs['user']['gen_batch_size']

    # Initialize the dragonfly GP optimizer
    domain = EuclideanDomain([[lo, up] for lo, up in zip(lb_list, ub_list)])
    func_caller = EuclideanFunctionCaller(None, domain)
    opt = EuclideanGPBandit(func_caller, ask_tell_mode=True,
                            options=Namespace(
                                acq='ts', build_new_model_every=number_of_gen_points,
                                init_capital=number_of_gen_points))
    opt.initialise()

    # Initialize folder to log the model
    if not os.path.exists('model_history'):
        os.mkdir('model_history')

    # If there is any past history, feed it to the GP
    if len(H) > 0:
        for i in range(len(H)):
            x = H['x'][i]
            y = H['f'][i]
            opt.tell([(x, -y)])
        # Update hyperparameters
        opt._build_new_model()

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    model_iteration = -1
    while tag not in [STOP_TAG, PERSIS_STOP]:
        model_iteration += 1

        # Ask the optimizer to generate `batch_size` new points
        # Store this information in the format expected by libE
        H_o = np.zeros(number_of_gen_points, dtype=gen_specs['out'])
        for i in range(number_of_gen_points):
            x = opt.ask()
            H_o['x'][i] = x

        # Log the parameters of the model
        with open('model_history/model_%05d.txt' %model_iteration, 'w') as f:
            f.write( opt.gp.__str__() + "\n" )

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)
        if calc_in is not None:
            # Check how many simulations have returned
            n = len(calc_in['f'])
            # Update the GP with latest simulation results
            for i in range(n):
                x = calc_in['x'][i]
                y = calc_in['f'][i]
                opt.tell([(x, -y)])
            # Update hyperparameters
            opt._build_new_model()
            # Set the number of points to generate to that number:
            number_of_gen_points = n
        else:
            number_of_gen_points = 0

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


def persistent_gp_mf_gen_f(H, persis_info, gen_specs, libE_info):
    """
    Create a Gaussian Process model, for multi-fidelity optimization,
    and update it as new simulation results are available, and generate
    inputs for the next simulations.

    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    # Extract bounds of the parameter space, and batch size
    ub_list = gen_specs['user']['ub']
    lb_list = gen_specs['user']['lb']

    # Fidelity range.
    fidel_range = gen_specs['user']['range']

    # Get fidelity cost function.
    cost_func = gen_specs['user']['cost_func']

    # Number of points to generate initially
    number_of_gen_points = gen_specs['user']['gen_batch_size']

    # Initialize the dragonfly GP optimizer
    domain = EuclideanDomain([[lo, up] for lo, up in zip(lb_list, ub_list)])
    fidel_space = EuclideanDomain([fidel_range])
    func_caller = EuclideanFunctionCaller(None,
                                          raw_domain=domain,
                                          raw_fidel_space=fidel_space,
                                          fidel_cost_func=cost_func,
                                          raw_fidel_to_opt=fidel_range[-1])
    opt = EuclideanGPBandit(func_caller,
                            ask_tell_mode=True,
                            is_mf=True,
                            options=Namespace(acq='ts',
                                              build_new_model_every=number_of_gen_points,
                                              init_capital=number_of_gen_points))
    opt.initialise()

    # Initialize folder to log the model
    if not os.path.exists('model_history'):
        os.mkdir('model_history')

    # If there is any past history, feed it to the GP
    if len(H) > 0:
        for i in range(len(H)):
            x = H['x'][i]
            z = H['z'][i]
            y = H['f'][i]
            opt.tell([([z], x, -y)])
        # Update hyperparameters
        opt._build_new_model()

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    model_iteration = -1
    while tag not in [STOP_TAG, PERSIS_STOP]:
        model_iteration += 1

        # Ask the optimizer to generate `batch_size` new points
        # Store this information in the format expected by libE
        H_o = np.zeros(number_of_gen_points, dtype=gen_specs['out'])
        for i in range(number_of_gen_points):
            z, input_vector = opt.ask()
            H_o['x'][i] = input_vector
            H_o['z'][i] = z[0]

        # Log the parameters of the model
        with open('model_history/model_%05d.txt' %model_iteration, 'w') as f:
            f.write( opt.gp.__str__() + "\n" )

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)
        if calc_in is not None:
            # Check how many simulations have returned
            n = len(calc_in['f'])
            # Update the GP with latest simulation results
            for i in range(n):
                x = calc_in['x'][i]
                z = calc_in['z'][i]
                y = calc_in['f'][i]
                opt.tell([([z], x, -y)])
            # Update hyperparameters
            opt._build_new_model()
            # Set the number of points to generate to that number:
            number_of_gen_points = n
        else:
            number_of_gen_points = 0

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


def persistent_gp_mf_disc_gen_f(H, persis_info, gen_specs, libE_info):
    """
    Create a Gaussian Process model, for multi-fidelity optimization
    in a discrete fidelity space, and update it as new simulation results are
    available, and generate inputs for the next simulations.

    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    # Extract bounds of the parameter space, and batch size
    ub_list = gen_specs['user']['ub']
    lb_list = gen_specs['user']['lb']

    # Multifidelity settings.
    cost_func = gen_specs['user']['cost_func']
    # discrete_fidel = gen_specs['user']['discrete']
    fidel_range = gen_specs['user']['range']

    # Number of points to generate initially.
    number_of_gen_points = gen_specs['user']['gen_batch_size']

    # Create configuration dictionary from which Dragongly will
    # automatically generate the necessary domains and orderings.
    config_params = {}
    config_params['domain'] = []
    for ub, lb in zip(ub_list, lb_list):
        domain_dict = {
            'max': ub,
            'min': lb,
            'type': 'float'
        }
        config_params['domain'].append(domain_dict)
    config_params['fidel_space'] = [{
        'type': 'discrete',
        'items': fidel_range
    }]
    config_params['fidel_to_opt'] = [fidel_range[-1]]
    config = load_config(config_params)

    # Initialize the dragonfly GP optimizer.
    func_caller = CPFunctionCaller(
        None,
        domain=config.domain,
        domain_orderings=config.domain_orderings,
        fidel_space=config.fidel_space,
        fidel_cost_func=cost_func,
        fidel_to_opt=config.fidel_to_opt,
        fidel_space_orderings=config.fidel_space_orderings)
    opt = CPGPBandit(
        func_caller, ask_tell_mode=True, is_mf=True,
        options=Namespace(
            acq='ts', build_new_model_every=number_of_gen_points,
            init_capital=number_of_gen_points))
    opt.initialise()

    # Initialize folder to log the model
    if not os.path.exists('model_history'):
        os.mkdir('model_history')

    # If there is any past history, feed it to the GP
    if len(H) > 0:
        for i in range(len(H)):
            x = H['x'][i]
            z = H['z'][i]
            y = H['f'][i]
            opt.tell([([z], x, -y)])
        # Update hyperparameters
        opt._build_new_model()

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    model_iteration = -1
    while tag not in [STOP_TAG, PERSIS_STOP]:
        model_iteration += 1

        # Ask the optimizer to generate `batch_size` new points
        # Store this information in the format expected by libE
        H_o = np.zeros(number_of_gen_points, dtype=gen_specs['out'])
        for i in range(number_of_gen_points):
            z, input_vector = opt.ask()
            H_o['x'][i] = input_vector
            H_o['z'][i] = z[0]

        # Log the parameters of the model
        with open('model_history/model_%05d.txt' %model_iteration, 'w') as f:
            f.write( opt.gp.__str__() + "\n" )

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)
        if calc_in is not None:
            # Check how many simulations have returned
            n = len(calc_in['f'])
            # Update the GP with latest simulation results
            for i in range(n):
                x = calc_in['x'][i]
                z = calc_in['z'][i]
                y = calc_in['f'][i]
                opt.tell([([z], x, -y)])
            # Update hyperparameters
            opt._build_new_model()
            # Set the number of points to generate to that number:
            number_of_gen_points = n
        else:
            number_of_gen_points = 0


    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


def persistent_gp_mt_ax_gen_f(H, persis_info, gen_specs, libE_info):
    """
    Create a Gaussian Process model for multi-task optimization
    and update it as new simulation results are
    available, and generate inputs for the next simulations.

    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    # Extract bounds of the parameter space, and batch size
    ub_list = gen_specs['user']['ub']
    lb_list = gen_specs['user']['lb']

    # Number of points to generate intially and during optimization.
    n_init_hifi = gen_specs['user']['n_init_hifi']
    n_init_lofi = gen_specs['user']['n_init_lofi']
    n_opt_hifi = gen_specs['user']['n_opt_hifi']
    n_opt_lofi = gen_specs['user']['n_opt_lofi']

    # Create search space.
    parameters = []
    for i, (ub, lb) in enumerate(zip(ub_list, lb_list)):
        parameters.append(
            RangeParameter(
                name='x{}'.format(i),
                parameter_type=ParameterType.FLOAT,
                lower=float(lb),
                upper=float(ub))
        )
    search_space=SearchSpace(parameters=parameters)

    # Create metrics.
    online_objective = AxMetric(
        name='online_metric',
        lower_is_better=True
    )
    offline_objective = AxMetric(
        name='offline_metric',
        lower_is_better=True
    )

    # Create optimization config.
    opt_config = OptimizationConfig(
        objective=Objective(online_objective, minimize=True))

    # Create experiment.
    exp = MultiTypeExperiment(
            name="mt_exp",
            search_space=search_space,
            default_trial_type="online",
            default_runner=AxRunner(libE_info, gen_specs),
            optimization_config=opt_config,
        )
    exp.add_trial_type("offline", AxRunner(libE_info, gen_specs))
    exp.add_tracking_metric(
        metric=offline_objective,
        trial_type="offline",
        canonical_name="online_metric")

    # If there is any past history, feed it to the GP
    # if len(H) > 0:
    #     for i in range(len(H)):
    #         x = H['x'][i]
    #         z = H['z'][i]
    #         y = H['f'][i]
    #         opt.tell([([z], x, -y)])
    #     # Update hyperparameters
    #     opt._build_new_model()

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    model_iteration = 0
    online_trials = []
    while tag not in [STOP_TAG, PERSIS_STOP]:

        if model_iteration == 0:
            # Initialize with sobol sample.
            for model, n_gen in zip(['online', 'offline'], [n_init_hifi, n_init_lofi]):
                s = get_sobol(exp.search_space, scramble=False)
                gr = s.gen(n_gen)
                trial = exp.new_batch_trial(trial_type=model, generator_run=gr)
                trial.run()
                trial.mark_completed()
                tag = trial.run_metadata['tag']
                if tag in [STOP_TAG, PERSIS_STOP]:
                    break
                if model == 'online':
                    online_trials.append(trial.index)

        else:
            # Run multi-task BO.

            # Fit the MTGP.
            m = get_MTGP(
                experiment=exp,
                data=exp.fetch_data(),
                search_space=exp.search_space,
            )

            # Find the best points for the online task.
            gr = m.gen(
                n=n_opt_lofi,
                optimization_config=exp.optimization_config,
                fixed_features=ObservationFeatures(
                    parameters={}, trial_index=online_trials[-1]),
            )

            # But launch them offline.
            tr = exp.new_batch_trial(trial_type="offline", generator_run=gr)
            tr.run()
            tr.mark_completed()
            tag = tr.run_metadata['tag']
            if tag in [STOP_TAG, PERSIS_STOP]:
                break
            
            # Update the model.
            m = get_MTGP(
                experiment=exp,
                data=exp.fetch_data(),
                search_space=exp.search_space,
            )
            
            # Select max-utility points from the offline batch to generate an online batch
            gr = max_utility_from_GP(
                n=n_opt_hifi,
                m=m,
                gr=gr
            )
            tr = exp.new_batch_trial(trial_type="online", generator_run=gr)
            tr.run()
            tr.mark_completed()
            tag = tr.run_metadata['tag']
            if tag in [STOP_TAG, PERSIS_STOP]:
                break
            online_trials.append(tr.index)

        # Make dummy H_o. Is it needed?
        H_o = np.zeros(1, dtype=gen_specs['out'])

        model_iteration += 1


    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


class AxRunner(Runner):
    def __init__(self, libE_info, gen_specs):
        self.libE_info = libE_info
        self.gen_specs = gen_specs
        super().__init__()

    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        task = trial.trial_type
        number_of_gen_points = len(trial.arms)
        H_o = np.zeros(number_of_gen_points, dtype=self.gen_specs['out'])
        
        for i, (arm_name, arm) in enumerate(trial.arms_by_name.items()):
            # fill H_o
            params = arm.parameters
            n_param = len(params)
            param_array = np.zeros(n_param)
            for j in range(n_param):
                param_array[j] = params['x{}'.format(j)]
            H_o['x'][i] = param_array
            H_o['task'][i] = task

        tag, Work, calc_in = sendrecv_mgr_worker_msg(self.libE_info['comm'], H_o)
        
        trial_metadata['tag'] = tag
        for i, (arm_name, arm) in enumerate(trial.arms_by_name.items()):
            # fill metadata
            params = arm.parameters
            trial_metadata[arm_name] = {
                "arm_name": arm_name,
                "trial_index": trial.index,
                "f": calc_in['f'][i] if calc_in is not None else None                
            }
        return trial_metadata


class AxMetric(Metric):
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                "mean": trial.run_metadata[arm_name]['f'],
                "sem": 0.0,
            })
        return Data(df=pd.DataFrame.from_records(records))


def max_utility_from_GP(n, m, gr):
    """
    Online batches are constructed by selecting the maximum utility points
    from the offline batch, after updating the model with the offline results.
    This function selects the max utility points according to the MTGP
    predictions.
    """
    obsf = []
    for arm in gr.arms:
        params = deepcopy(arm.parameters)
        params['trial_type'] = 'online'
        obsf.append(ObservationFeatures(parameters=params))
    # Make predictions
    f, cov = m.predict(obsf)
    # Compute expected utility
    u = -np.array(f['online_metric']) 
    best_arm_indx = np.flip(np.argsort(u))[:n]
    gr_new = GeneratorRun(
        arms = [
            gr.arms[i] for i in best_arm_indx
        ],
        weights = [1.] * n,
    )
    return gr_new
