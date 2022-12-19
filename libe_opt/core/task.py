from .base import NamedBase


class Task(NamedBase):
    def __init__(self, name, n_init, n_opt):
        super().__init__(name)
        self._n_init = n_init
        self._n_opt = n_opt

    @property
    def n_init(self):
        return self._n_init

    @property
    def n_opt(self):
        return self._n_opt
