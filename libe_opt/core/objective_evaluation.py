class ObjectiveEvaluation():
    def __init__(self, objective, value, sem=None):
        self._objective = objective
        self._value = value
        self._sem = sem

    @property
    def objective(self):
        return self._objective

    @property
    def value(self):
        return self._value

    @property
    def sem(self):
        return self._sem
