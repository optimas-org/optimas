class Objective():
    def __init__(self, name='f', minimize=True):
        self.name = name
        self.minimize = minimize


class Variable():
    def __init__(self, name, lower_bound, upper_bound, is_fidelity=False, target_value=None):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_fidelity = is_fidelity
        self.target_value = target_value


class Evaluation():
    def __init__(self, objectives, values, noises=None):
        self.objectives = objectives
        self.values = values
        self.noises = noises


class ObjectiveEvaluation():
    def __init__(self, objective, value, sem=None):
        self.objective = objective
        self.value = value
        self.sem = sem


class Trial():
    def __init__(
            self, variables, objectives, variable_values=[],
            objective_evaluations=[], index=None, **kwargs):
        
        self.variables = variables
        self.variable_values = variable_values
        self.objectives = objectives
        self.objective_evaluations = objective_evaluations
        self.index = index

        for k, v in kwargs.items():
            setattr(self, k, v)

    def complete_evaluation(self, objective_evaluation):
        self.objective_evaluations.append(objective_evaluation)


# class Trial2():
#     def __init__(self, variables, variable_values, objectives, objective_evaluations=None, **kwargs):
        
#         self.variables = variables
#         self.variable_values = variable_values
#         self.objectives = objectives
#         self.objective_evaluations = objective_evaluations

#         # variable_names = [var.name for var in variables]
#         # self.candidate = {zip(variable_names, variable_values)}

#         # if objective_values is None:
#         # self.result = {}
#         # for objective in objectives:
#         #     self.result[objective.name] = None
#         # else:
#         #     variable_names = [var.name for var in variables]
#         #     self.result =
#         for k, v in kwargs.items():
#             setattr(self, k, v)

#     def complete_evaluation(self, objective, value, noise=None):
#         self.result[objective.name] = (value, noise)
