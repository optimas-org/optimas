"""Contains the definition of the base Ax generator."""

from typing import List, Optional
import logging

import torch

from optimas.core import Objective, TrialParameter, VaryingParameter, Parameter
from optimas.generators.base import Generator


# Disable Ax loggers to get cleaner output. In principle, setting
# `verbose_logging=False` in the `AxClient` should already avoid most of the
# logs, but this does not work when using 'spawn' multiprocessing.
for logger in logging.root.manager.loggerDict:
    if logger.startswith("ax.") or logger == "ax":
        logging.getLogger(logger).setLevel(logging.ERROR)


class AxGenerator(Generator):
    """Base class for all Ax generators.

    Parameters
    ----------
    varying_parameters : list of VaryingParameter
        List of input parameters to vary.
    objectives : list of Objective
        List of optimization objectives.
    analyzed_parameters : list of Parameter, optional
        List of parameters to analyze at each trial, but which are not
        optimization objectives. By default ``None``.
    use_cuda : bool, optional
        Whether to allow the generator to run on a CUDA GPU. By default
        ``False``.
    gpu_id : int, optional
        The ID of the GPU in which to run the generator. By default, ``0``.
    dedicated_resources : bool, optional
        Whether to allocated dedicated resources (e.g., the GPU) for the
        generator. These resources will not be available to the
        simulation workers. By default, ``False``.
    save_model : bool, optional
        Whether to save the optimization model (e.g., the surrogate model) to
        disk. By default ``False``.
    model_save_period : int, optional
        Periodicity, in number of evaluated Trials, with which to save the
        model to disk. By default, ``5``.
    model_history_dir : str, optional
        Name of the directory in which the model will be saved. By default,
        ``'model_history'``.
    custom_trial_parameters : list of TrialParameter
        For some generators, it might be necessary to attach additional
        parameters to the trials. If so, they can be given here as a list.
        By default, ``None``.
    allow_fixed_parameters : bool, optional
        Whether the generator supports ``VaryingParameter``s whose value
        has been fixed. By default, False.
    allow_updating_parameters : list of TrialParameter
        Whether the generator supports updating the ``VaryingParameter``s.
        If so, the `_update_parameter` method must be implemented.
        By default, False.

    """

    def __init__(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        analyzed_parameters: Optional[List[Parameter]] = None,
        use_cuda: Optional[bool] = False,
        gpu_id: Optional[int] = 0,
        dedicated_resources: Optional[bool] = False,
        save_model: Optional[bool] = False,
        model_save_period: Optional[int] = 5,
        model_history_dir: Optional[str] = "model_history",
        custom_trial_parameters: Optional[TrialParameter] = None,
        allow_fixed_parameters: Optional[bool] = False,
        allow_updating_parameters: Optional[bool] = False,
    ) -> None:
        super().__init__(
            varying_parameters=varying_parameters,
            objectives=objectives,
            analyzed_parameters=analyzed_parameters,
            use_cuda=use_cuda,
            gpu_id=gpu_id,
            dedicated_resources=dedicated_resources,
            save_model=save_model,
            model_save_period=model_save_period,
            model_history_dir=model_history_dir,
            custom_trial_parameters=custom_trial_parameters,
            allow_fixed_parameters=allow_fixed_parameters,
            allow_updating_parameters=allow_updating_parameters,
        )
        self._determine_torch_device()

    def _determine_torch_device(self) -> None:
        """Determine whether to run the generator on GPU (CUDA) or CPU."""
        # Use CUDA if available.
        if self.use_cuda and torch.cuda.is_available():
            self.torch_device = "cuda"
        else:
            self.torch_device = "cpu"
