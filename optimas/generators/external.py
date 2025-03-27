"""Contains the definition of an external generator."""

from .base import Generator


class ExternalGenerator(Generator):
    """Supports a generator in the CAMPA generator standard."""

    def __init__(
        self,
        ext_gen,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.gen = ext_gen

    def suggest(self, n_trials):
        """Request the next set of points to evaluate."""
        return self.gen.suggest(n_trials)

    def ingest(self, trials):
        """Send the results of evaluations to the generator."""
        self.gen.ingest(trials)
        
