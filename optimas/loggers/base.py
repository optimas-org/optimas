"""This module defines the base Logger class."""

from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod


if TYPE_CHECKING:
    from optimas.core import Trial
    from optimas.generators.base import Generator
    from optimas.explorations import Exploration


class Logger(ABC):
    """Base class for all loggers."""

    def initialize(self, exploration: Exploration):
        """Initialize logger.

        This method is called in `Exploration.__init__`.

        Parameters
        ----------
        exploration : Exploration
            The exploration instance to which the logger was attached.
        """
        pass

    @abstractmethod
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
        pass

    def finish(self):
        """Finish logging.

        This method is meant to be called then the exploration is finished.
        """
        pass
