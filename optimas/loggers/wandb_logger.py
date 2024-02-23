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
    def __init__(
        self,
        api_key: str,
        project: str,
        run_name: str,
        run_id: Optional[str] = None,
        data_types: Optional[Dict] = None,
        user_function: Optional[Callable] = None,
        login_kwargs: Optional[Dict] = None,
        init_kwargs: Optional[Dict] = None,
    ) -> None:
        self._api_key = api_key
        self._project = project
        self._run_name = run_name
        self._run_id = run_id
        self._data_types = {} if data_types is None else data_types
        self._user_function = user_function
        self._login_kwargs = {} if login_kwargs is None else login_kwargs
        self._init_kwargs = {} if init_kwargs is None else init_kwargs
        self._run = None
        self._dir = None

    def initialize(self, exploration: Exploration):
        # Create dir if it doesn't exist.
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
        self._run.finish()
