class Evaluation():
    def __init__(self, parameter, value, sem=None):
        self._parameter = parameter
        self._value = value
        self._sem = sem

    @property
    def parameter(self):
        return self._parameter

    @property
    def value(self):
        return self._value

    @property
    def sem(self):
        return self._sem
