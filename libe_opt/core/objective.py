from .base import NamedBase


class Objective(NamedBase):
    def __init__(self, name='f', minimize=True):
        super().__init__(name)
        self._minimize = minimize

    @property
    def minimize(self):
        return self._minimize
