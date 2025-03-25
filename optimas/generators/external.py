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

    def ask(self, n_trials):
        """Request the next set of points to evaluate."""
        return self.gen.ask(n_trials)

    def tell(self, trials):
        """Send the results of evaluations to the generator."""
        self.gen.tell(trials)
