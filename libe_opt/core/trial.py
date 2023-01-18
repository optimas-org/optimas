import numpy as np


class Trial():
    def __init__(
            self, varying_parameters, objectives, analyzed_parameters=None,
            parameter_values=None, evaluations=None, index=None,
            custom_parameters=None):

        self._varying_parameters = varying_parameters
        self._objectives = objectives
        self._analyzed_parameters = (
            [] if analyzed_parameters is None else analyzed_parameters)
        self._parameter_values = (
            [] if parameter_values is None else parameter_values)
        evaluations = [] if evaluations is None else evaluations
        self._index = index
        self._custom_parameters = (
            [] if custom_parameters is None else custom_parameters)

        for param in self._custom_parameters:
            setattr(self, param.name, None)

        self._mapped_evaluations = {}
        for par in self._objectives + self._analyzed_parameters:
            self._mapped_evaluations[par.name] = None
        for ev in evaluations:
            self._mapped_evaluations[ev.parameter.name] = ev

    @property
    def varying_parameters(self):
        return self._varying_parameters

    @property
    def objectives(self):
        return self._objectives

    @property
    def analyzed_parameters(self):
        return self._analyzed_parameters

    @property
    def parameter_values(self):
        return self._parameter_values

    @parameter_values.setter
    def parameter_values(self, values):
        assert len(values) == len(self._varying_parameters)
        values = np.array(values) if isinstance(values, list) else values
        self._parameter_values = values

    @property
    def objective_evaluations(self):
        return [self._mapped_evaluations[obj.name] for obj in self._objectives]

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def custom_parameters(self):
        return self._custom_parameters

    def complete_evaluation(self, evaluation):
        evaluated_parameter = evaluation.parameter.name
        assert evaluated_parameter in self._mapped_evaluations
        if self._mapped_evaluations[evaluated_parameter] is None:
            self._mapped_evaluations[evaluated_parameter] = evaluation

    def parameters_as_dict(self):
        params = {}
        for var, val in zip(self._varying_parameters, self._parameter_values):
            params[var.name] = val
        return params

    def objectives_as_dict(self):
        params = {}
        for obj in self._objectives:
            ev = self._mapped_evaluations[obj.name]
            params[obj.name] = (ev.value, ev.sem)
        return params

    def analyzed_parameters_as_dict(self):
        params = {}
        for par in self._analyzed_parameters:
            ev = self._mapped_evaluations[par.name]
            params[par.name] = (ev.value, ev.sem)
        return params

    def completed(self):
        for par, ev in self._mapped_evaluations.items():
            if ev is None:
                return False
        return True
