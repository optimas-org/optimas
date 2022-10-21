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
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.resources.resources import Resources

# import dragonfly Gaussian Process functions
from dragonfly.exd.domains import EuclideanDomain
from dragonfly.exd.experiment_caller import (EuclideanFunctionCaller,
                                             CPFunctionCaller)
from dragonfly.opt.gp_bandit import EuclideanGPBandit, CPGPBandit
from dragonfly.exd.cp_domain_utils import load_config
from argparse import Namespace
import torch
from ax import Metric, Runner
from ax.runners import SyntheticRunner
from ax.storage.json_store.save import save_experiment
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.parameter import RangeParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.core.optimization_config import OptimizationConfig
from ax.core.objective import Objective
from ax.modelbridge.factory import get_sobol
from ax.core.observation import ObservationFeatures
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import (
    GenerationStep, GenerationStrategy)
from ax.modelbridge.registry import Models

# Imports and definitions needed for custom `get_MTGP`
from ax.core.experiment import Experiment
from typing import Optional
from ax.modelbridge.transforms.convert_metric_names import (
    tconfig_from_mt_experiment
)
from ax.modelbridge.registry import (
    MT_MTGP_trans,
    ST_MTGP_trans,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch.botorch import BotorchModel
DEFAULT_TORCH_DEVICE = torch.device("cpu")


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
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

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
            H_o['resource_sets'][i] = 1

        # Log the parameters of the model
        with open('model_history/model_%05d.txt' %model_iteration, 'w') as f:
            f.write( opt.gp.__str__() + "\n" )

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = ps.send_recv(H_o)
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
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Fidelity range.
    fidel_range = gen_specs['user']['mf_params']['range']

    # Get fidelity cost function.
    cost_func = gen_specs['user']['mf_params']['cost_func']

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
            H_o['resource_sets'][i] = 1

        # Log the parameters of the model
        with open('model_history/model_%05d.txt' %model_iteration, 'w') as f:
            f.write( opt.gp.__str__() + "\n" )

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = ps.send_recv(H_o)
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
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Multifidelity settings.
    cost_func = gen_specs['user']['mf_params']['cost_func']
    # discrete_fidel = gen_specs['user']['discrete']
    fidel_range = gen_specs['user']['mf_params']['range']

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
            H_o['resource_sets'][i] = 1

        # Log the parameters of the model
        with open('model_history/model_%05d.txt' %model_iteration, 'w') as f:
            f.write( opt.gp.__str__() + "\n" )

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = ps.send_recv(H_o)
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


def persistent_gp_ax_gen_f(H, persis_info, gen_specs, libE_info):
    """
    Create a Gaussian Process model, update it as new simulation results
    are available, and generate inputs for the next simulations.
    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    # Create Ax client.
    if gen_specs['user']['client'] is not None:
        ax_client = gen_specs['user']['client']
    else:
        # Extract bounds of the parameter space
        names_list = gen_specs['user']['params']
        ub_list = gen_specs['user']['ub']
        lb_list = gen_specs['user']['lb']

        # Create parameter list.
        # The use of `.item()` converts from numpy types to native Python
        # types. This is needed becase Ax seems to support only native types.
        parameters = list()
        for name, lb, ub in zip(names_list, lb_list, ub_list):
            parameters.append(
                {
                    'name': name,
                    'type': 'range',
                    'bounds': [lb.item(), ub.item()]
                }
            )

        use_mf = False
        if 'mf_params' in gen_specs['user']:
            use_mf = True
            mf_params = gen_specs['user']['mf_params']
            fidel_name = mf_params['name']
            parameters.append(
                {
                    'name': mf_params['name'],
                    'type': 'range',
                    'bounds': mf_params['range'],
                    'is_fidelity': True,
                    'target_value': mf_params['range'][-1]
                }
            )

        # Batch size
        batch_size = gen_specs['user']['gen_batch_size']

        # Make generation strategy:
        steps = []

        # If there is no past history,
        # adds Sobol initialization with `batch_size` random trials:
        if len(H) == 0:
            steps.append(GenerationStep(model=Models.SOBOL, num_trials=batch_size))

        # If CUDA is available, run BO loop on the GPU.
        if gen_specs['user']['use_cuda'] and torch.cuda.is_available():
            torch_device = 'cuda'
            resources = Resources.resources.worker_resources
            resources.set_env_to_slots('CUDA_VISIBLE_DEVICES')
        else:
            torch_device = 'cpu'

        # continue indefinitely with GPEI (of GPKG for multifidelity).
        if use_mf:
            steps.append(
                GenerationStep(
                    model=Models.GPKG,
                    num_trials=-1,
                    model_kwargs={
                        'cost_intercept': mf_params['cost_intercept'],
                        'torch_dtype': torch.double,
                        'torch_device': torch.device(torch_device)
                    }
                )
            )
        else:
            steps.append(
                GenerationStep(
                    model=Models.GPEI,
                    num_trials=-1,
                    model_kwargs = {
                        'torch_dtype': torch.double,
                        'torch_device': torch.device(torch_device)
                    }
                )
            )

        gs = GenerationStrategy(steps)

        # Create client and experiment.
        ax_client = AxClient(generation_strategy=gs)
        ax_client.create_experiment(
            parameters=parameters,
            objective_name="f",
            minimize=True
        )

    # Metric name.
    metric_name = list(ax_client.experiment.metrics.keys())[0]

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Number of points to generate initially.
    number_of_gen_points = gen_specs['user']['gen_batch_size']

    # If there is any past history, feed it to the GP
    if len(H) > 0:
        names_list = gen_specs['user']['params']
        params = dict.fromkeys(names_list)

        for i in range(len(H)):
            for j, name in enumerate(names_list):
                params[name] = H['x'][i][j]

            if 'mf_params' in gen_specs['user']:
                fidel_name = gen_specs['user']['mf_params']['name']
                params[fidel_name] = H['z'][i]

            _, trial_id = ax_client.attach_trial(params)
            ax_client.complete_trial(trial_id, {metric_name: (H['f'][i], np.nan)})

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    model_iteration = 0
    while tag not in [STOP_TAG, PERSIS_STOP]:

        # Ask the optimizer to generate `batch_size` new points
        # Store this information in the format expected by libE
        H_o = np.zeros(number_of_gen_points, dtype=gen_specs['out'])
        for i in range(number_of_gen_points):
            parameters, _ = ax_client.get_next_trial()
            if use_mf:
                H_o['z'][i] = parameters.pop(fidel_name)
            H_o['x'][i] = list(parameters.values())
            H_o['resource_sets'][i] = 1

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = ps.send_recv(H_o)
        if calc_in is not None:
            # Check how many simulations have returned
            n = len(calc_in['f'])
            # Update the GP with latest simulation results
            for i in range(n):
                trial_index = int(calc_in['sim_id'][i])
                y = calc_in['f'][i]
                # Register trial with unknown SEM
                ax_client.complete_trial(trial_index, {metric_name: (y, np.nan)})
            # Set the number of points to generate to that number:
            number_of_gen_points = n
        else:
            number_of_gen_points = 0

        # Save current model.
        if model_iteration == 0:
            # Initialize folder to log the model.
            if not os.path.exists('model_history'):
                os.mkdir('model_history')
        ax_client.save_to_json_file(
            'model_history/ax_client_%05d.json' % model_iteration)

        # Increase iteration counter.
        model_iteration += 1

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

    # Get task names.
    hifi_task = gen_specs['user']['mt_params']['name_hifi']
    lofi_task = gen_specs['user']['mt_params']['name_lofi']

    # Number of points to generate initially and during optimization.
    n_init_hifi = gen_specs['user']['mt_params']['n_init_hifi']
    n_init_lofi = gen_specs['user']['mt_params']['n_init_lofi']
    n_opt_hifi = gen_specs['user']['mt_params']['n_opt_hifi']
    n_opt_lofi = gen_specs['user']['mt_params']['n_opt_lofi']

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
    hifi_objective = AxMetric(
        name='hifi_metric',
        lower_is_better=True
    )
    lofi_objective = AxMetric(
        name='lofi_metric',
        lower_is_better=True
    )

    # Create optimization config.
    opt_config = OptimizationConfig(
        objective=Objective(hifi_objective, minimize=True))

    # Create runner.
    ax_runner = AxRunner(libE_info, gen_specs)

    # Create experiment.
    exp = MultiTypeExperiment(
            name="mt_exp",
            search_space=search_space,
            default_trial_type=hifi_task,
            default_runner=ax_runner,
            optimization_config=opt_config,
        )

    exp.add_trial_type(lofi_task, ax_runner)
    exp.add_tracking_metric(
        metric=lofi_objective,
        trial_type=lofi_task,
        canonical_name='hifi_metric')

    # TODO: Implement reading past history (by reading saved experiment or
    # libEnsemble hystory file).

    # If CUDA is available, run BO loop on the GPU.
    if gen_specs['user']['use_cuda'] and torch.cuda.is_available():
        torch_device = 'cuda'
        resources = Resources.resources.worker_resources
        resources.set_env_to_slots('CUDA_VISIBLE_DEVICES')
    else:
        torch_device = 'cpu'

    # Batch limit for the initialization of the aquisition function optimization.
    init_batch_limit = 1000

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    model_iteration = 0
    hifi_trials = []
    while tag not in [STOP_TAG, PERSIS_STOP]:

        if model_iteration == 0:
            # Initialize with sobol sample.
            for model, n_gen in zip([hifi_task, lofi_task], [n_init_hifi, n_init_lofi]):
                s = get_sobol(exp.search_space, scramble=True)
                gr = s.gen(n_gen)
                trial = exp.new_batch_trial(trial_type=model, generator_run=gr)
                trial.run()
                trial.mark_completed()
                tag = trial.run_metadata['tag']
                if tag in [STOP_TAG, PERSIS_STOP]:
                    break
                if model == hifi_task:
                    hifi_trials.append(trial.index)

        else:
            # Run multi-task BO.

            # 1) Fit the MTGP.
            m = get_MTGP(
                experiment=exp,
                data=exp.fetch_data(),
                search_space=exp.search_space,
                dtype=torch.double,
                device=torch.device(torch_device)
            )

            # 2) Find the best points for the high fidelity task.

            # Too large initialization batches can lead to out-of-memory errors.
            # The loop below tries to generate the next points to evaluate
            # using `init_batch_limit`. If an RuntimeError is raised during
            # generation (namely an out-of-memory error), `init_batch_limit`
            # is divided by two and a new attempt is made. This is repeated
            # until the generation runs successfully.            
            generator_success = True
            while True:
                try:
                    # Try to generate the new points.
                    gr = m.gen(
                        n=n_opt_lofi,
                        optimization_config=exp.optimization_config,
                        fixed_features=ObservationFeatures(
                            parameters={}, trial_index=hifi_trials[-1]),
                        model_gen_options={
                            'optimizer_kwargs': {
                                'init_batch_limit': init_batch_limit
                            }
                        }
                    )
                    # When successful, break loop.
                    break
                except RuntimeError as e:
                    # Print exception.
                    print('RuntimeError: {}'.format(e), flush=True)
                    # Divide batch size by 2.
                    init_batch_limit //= 2
                    print('Retrying with `init_batch_limit={}`'.format(
                        init_batch_limit), flush=True)
                finally:
                    # If all attempts have failed (even for batch size of 1),
                    # mark generation as failed and break loop.
                    if init_batch_limit == 0:
                        generator_success = False
                        break
            # If generation failed, stop optimization.
            if not generator_success:
                tag = STOP_TAG
                break

            # 3) But launch them at low fidelity.
            tr = exp.new_batch_trial(trial_type=lofi_task, generator_run=gr)
            tr.run()
            tr.mark_completed()
            tag = tr.run_metadata['tag']
            if tag in [STOP_TAG, PERSIS_STOP]:
                break

            # 4) Update the model.
            m = get_MTGP(
                experiment=exp,
                data=exp.fetch_data(),
                search_space=exp.search_space,
                dtype=torch.double,
                device=torch.device(torch_device)
            )

            # 5) Select max-utility points from the low fidelity batch to generate a high fidelity batch.
            gr = max_utility_from_GP(
                n=n_opt_hifi,
                m=m,
                gr=gr,
                hifi_task=hifi_task
            )
            tr = exp.new_batch_trial(trial_type=hifi_task, generator_run=gr)
            tr.run()
            tr.mark_completed()
            tag = tr.run_metadata['tag']
            if tag in [STOP_TAG, PERSIS_STOP]:
                break
            hifi_trials.append(tr.index)

        if model_iteration == 0:
            # Initialize folder to log the model.
            if not os.path.exists('model_history'):
                os.mkdir('model_history')
            # Register metric and runner in order to be able to save to json.
            _, encoder_registry, decoder_registry = register_metric(AxMetric)
            _, encoder_registry, decoder_registry = register_runner(
                AxRunner,
                encoder_registry=encoder_registry,
                decoder_registry=decoder_registry
            )

        # Save current experiment.
        # Saving the experiment to a json file currently requires a bit of
        # trickery. The `AxRunner` cannot be serialized into a json file
        # due to the `libE_info` and `gen_specs` attributes. This also prevents
        # the experiment from being saved to file. In order to overcome this,
        # all instances of the `AxRunner` are replaced by a `SyntheticRunner`
        # before saving. Afterwards, the `AxRunner` is reasigned again to both
        # high- and low-fidelity tasks in order to allow the optimization to
        # continue.
        for i, trial in exp.trials.items():
            trial._runner = SyntheticRunner()
        exp.update_runner(lofi_task, SyntheticRunner())
        exp.update_runner(hifi_task, SyntheticRunner())
        save_experiment(
            exp,
            'model_history/experiment_%05d.json' % model_iteration,
            encoder_registry=encoder_registry
        )
        exp.update_runner(lofi_task, ax_runner)
        exp.update_runner(hifi_task, ax_runner)

        # Increase iteration counter.
        model_iteration += 1

    return [], persis_info, FINISHED_PERSISTENT_GEN_TAG


class AxRunner(Runner):
    """ Custom runner in charge of executing the trials using libEnsemble. """

    def __init__(self, libE_info, gen_specs):
        self.libE_info = libE_info
        self.gen_specs = gen_specs
        self.ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
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
            H_o['resource_sets'][i] = 1
            H_o['task'][i] = task

        tag, Work, calc_in = self.ps.send_recv(H_o)

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
    """ Custom metric to be optimized during the experiment. """

    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                "mean": trial.run_metadata[arm_name]['f'],
                "sem": np.nan,
            })
        return Data(df=pd.DataFrame.from_records(records))


def max_utility_from_GP(n, m, gr, hifi_task):
    """
    High fidelity batches are constructed by selecting the maximum utility points
    from the low fidelity batch, after updating the model with the low fidelity results.
    This function selects the max utility points according to the MTGP
    predictions.
    """
    obsf = []
    for arm in gr.arms:
        params = deepcopy(arm.parameters)
        params['trial_type'] = hifi_task
        obsf.append(ObservationFeatures(parameters=params))
    # Make predictions
    f, cov = m.predict(obsf)
    # Compute expected utility
    u = -np.array(f['hifi_metric'])
    best_arm_indx = np.flip(np.argsort(u))[:n]
    gr_new = GeneratorRun(
        arms = [
            gr.arms[i] for i in best_arm_indx
        ],
        weights = [1.] * n,
    )
    return gr_new


def get_MTGP(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    dtype: torch.dtype = torch.double,
    device: torch.device = DEFAULT_TORCH_DEVICE,
    trial_index: Optional[int] = None,
) -> TorchModelBridge:
    """Instantiates a Multi-task Gaussian Process (MTGP) model that generates
    points with EI.
    If the input experiment is a MultiTypeExperiment then a
    Multi-type Multi-task GP model will be instantiated.
    Otherwise, the model will be a Single-type Multi-task GP.

    This method is a custom version of `get_MTGP` in `ax.modelbridge.factory`
    that exposes a `device` parameter in order to be able to run on the GPU.
    See https://github.com/facebook/Ax/issues/928 for details.

    Implemented by S. Jalas.
    """

    if isinstance(experiment, MultiTypeExperiment):
        trial_index_to_type = {
            t.index: t.trial_type for t in experiment.trials.values()
        }
        transforms = MT_MTGP_trans
        transform_configs = {
            "TrialAsTask": {"trial_level_map": {"trial_type": trial_index_to_type}},
            "ConvertMetricNames": tconfig_from_mt_experiment(experiment),
        }
    else:
        # Set transforms for a Single-type MTGP model.
        transforms = ST_MTGP_trans
        transform_configs = None

    # Choose the status quo features for the experiment from the selected trial.
    # If trial_index is None, we will look for a status quo from the last
    # experiment trial to use as a status quo for the experiment.
    if trial_index is None:
        trial_index = len(experiment.trials) - 1
    elif trial_index >= len(experiment.trials):
        raise ValueError("trial_index is bigger than the number of experiment trials")

    status_quo = experiment.trials[trial_index].status_quo
    if status_quo is None:
        status_quo_features = None
    else:
        status_quo_features = ObservationFeatures(
            parameters=status_quo.parameters,
            trial_index=trial_index,
        )

    return TorchModelBridge(
        experiment=experiment,
        search_space=search_space or experiment.search_space,
        data=data,
        model=BotorchModel(),
        transforms=transforms,
        transform_configs=transform_configs,
        torch_dtype=torch.double,
        torch_device=device,
        status_quo_features=status_quo_features,
    )
