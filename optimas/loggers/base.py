from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from optimas.core import Trial
    from optimas.generators.base import Generator
    from optimas.explorations import Exploration


class Logger:

    def initialize(self, exploration: Exploration):
        pass

    def log_trial(self, trial: Trial, generator: Generator):
        pass

    def log_custom_metrics(self, last_trial: Trial, generator: Generator):
        pass

    def finish(self):
        pass
