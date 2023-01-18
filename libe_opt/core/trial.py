import numpy as np


class Trial():
    def __init__(
            self, varying_parameters, objectives, parameter_values=None,
            objective_evaluations=None, index=None, custom_parameters=None):

        self._varying_parameters = varying_parameters
        self._objectives = objectives
        self._parameter_values = (
            [] if parameter_values is None else parameter_values)
        self._objective_evaluations = (
            [] if objective_evaluations is None else objective_evaluations)
        self._index = index
        self._custom_parameters = (
            [] if custom_parameters is None else custom_parameters)

        for param in self._custom_parameters:
            setattr(self, param.name, None)

    @property
    def varying_parameters(self):
        return self._varying_parameters

    @property
    def objectives(self):
        return self._objectives

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
        return self._objective_evaluations

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def custom_parameters(self):
        return self._custom_parameters

    def complete_evaluation(self, objective_evaluation):
        self._objective_evaluations.append(objective_evaluation)

    def parameters_as_dict(self):
        params = {}
        for var, val in zip(self._varying_parameters, self._parameter_values):
            params[var.name] = val
        return params

    def objectives_as_dict(self):
        params = {}
        for obj, oe in zip(self._objectives, self._objective_evaluations):
            params[obj.name] = (oe.value, oe.sem)
        return params

    def completed(self):
        evaluated_objectives = [
            oe.objective.name for oe in self._objective_evaluations]
        objectives = [obj.name for obj in self._objectives]
        return set(evaluated_objectives) == set(objectives)
