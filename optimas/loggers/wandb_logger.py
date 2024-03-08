"""This module defines the class for logging to Weights and Biases."""

from __future__ import annotations
import pathlib
from typing import TYPE_CHECKING, Optional, Callable, Dict

from matplotlib.figure import Figure
import wandb

from .base import Logger

if TYPE_CHECKING:
    from optimas.core import Trial
    from optimas.generators.base import Generator
    from optimas.explorations import Exploration


class WandBLogger(Logger):
    r"""Weights and Biases logger class.

    Parameters
    ----------
    api_key : str
        The API key used to log into Weight and Biases.
    project : str
        Project name.
    run : str, optional
        Run name. If not given, a random name will be assigned by W&B.
    run_id : str, optional
        A unique ID for this run, used for resuming. It must
        be unique in the project, and if you delete a run you can't reuse
        the ID. Use the ``run`` field for a short descriptive name, or
        `config` (passed in the ``init_kwargs``)
        for saving hyperparameters to compare across runs. The ID cannot
        contain the following special characters: ``/\#?%:``.
        See the `W&B guide to resuming runs <https://docs.wandb.com/guides/runs/resuming>`_.
    data_types : Dict, optional
        A dictionary of the shape
        ``{"name": {"type": DataType, "type_kwargs": {}}``,
        where ``name`` is the
        name of a varying parameter, objective or other analyzed parameter,
        ``DataType`` is a W&B `DataType <https://docs.wandb.ai/ref/python/data-types/>`_
        and ``type_kwargs`` can include additional arguments to pass to the
        data type.
        If provided, the given parameters will be converted to the specified
        data types when logging.
    custom_logs : Callable, optional
        A user-defined function for creating custom logs. This function must
        be of the shape `custom_logs(trial, generator)`, where ``trial`` is
        the most recently evaluated trial and ``generator`` is the currently
        active generator. The function must return a dictionary with the
        appropriate shape to that it can be given to `wandb.log`.
    login_kwargs : Dict, optional
        Additional arguments to pass to ``wandb.login``.
    init_kwargs : Dict, optional
        Additional arguments to pass to ``wandb.init``.
    """
    def __init__(
        self,
        api_key: str,
        project: str,
        run: Optional[str] = None,
        run_id: Optional[str] = None,
        data_types: Optional[Dict] = None,
        custom_logs: Optional[Callable] = None,
        login_kwargs: Optional[Dict] = None,
        init_kwargs: Optional[Dict] = None,
    ) -> None:
        self._api_key = api_key
        self._project = project
        self._run_name = run
        self._run_id = run_id
        self._data_types = {} if data_types is None else data_types
        self._user_function = custom_logs
        self._login_kwargs = {} if login_kwargs is None else login_kwargs
        self._init_kwargs = {} if init_kwargs is None else init_kwargs
        self._run = None
        self._dir = None

    def initialize(self, exploration: Exploration):
        """Initialize the W&B logger.

        This method logs into WandB and created a new run using the output
        directory if the exploration.

        Parameters
        ----------
        exploration : Exploration
            The exploration instance to which the logger was attached.
        """
        # Create dir if it doesn't exist.
        # We need to do this because the logger is typically initialized
        # before the exploration runs and, thus, before the exploration dir
        # has been created.
        dir = exploration.exploration_dir_path
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        self._dir = dir

        # Login and initialize run.
        wandb.login(key=self._api_key, **self._login_kwargs)
        if self._run is None:
            self._run = wandb.init(
                project=self._project,
                name=self._run_name,
                resume=True,
                id=self._run_id,
                dir=self._dir,
                **self._init_kwargs,
            )
            if self._run_id is None:
                self._run_id = self._run.id

    def log_trial(self, trial: Trial, generator: Generator):
        """Log a trial.

        This method is called every time an evaluated trial is given back
        to the generator.

        Parameters
        ----------
        trial : Trial
            The last trial that has been evaluated.
        generator : Generator
            The currently active generator.
        """
        # Get and process trial data.
        logs = trial.data
        for key in list(logs.keys()):
            # Apply user-provided wandb types.
            if key in self._data_types:
                logs[key] = self._data_types[key]["type"](
                    logs[key], **self._data_types[key]["type_kwargs"]
                )
            # By default, convert matplotlib figures to images.
            elif isinstance(logs[key], Figure):
                logs[key] = wandb.Image(logs[key])
            # By default, only log scalars.
            elif hasattr(logs[key], "__len__"):
                del logs[key]

        # Organize in sections.
        for par in generator.varying_parameters:
            if par.name in logs:
                logs[f"Varying parameters/{par.name}"] = logs.pop(par.name)
        for par in generator.objectives:
            if par.name in logs:
                logs[f"Objectives/{par.name}"] = logs.pop(par.name)
        for par in generator.analyzed_parameters:
            if par.name in logs:
                logs[f"Analyzed parameters/{par.name}"] = logs.pop(par.name)

        # Add custom user-defined logs.
        if self._user_function is not None:
            custom_logs = self._user_function(trial, generator)
            logs = {**logs, **custom_logs}

        # Log data.
        self._run.log(logs)

    def finish(self):
        """Finish logging.

        Call this method to finish the current run on W&B.
        """
        self._run.finish()
