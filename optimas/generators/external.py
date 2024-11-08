from .base import Generator


class ExternalGenerator(Generator):
    """Supports a generator in the CAMPA generator standard"""

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
        return self.gen.ask(n_trials)

    def tell(self, trials):
        self.gen.tell(trials)
