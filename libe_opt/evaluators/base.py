class Evaluator:
    def __init__(self, sim_function, analyzed_parameters=None, n_gpus=1):
        self.sim_function = sim_function
        self._analyzed_parameters = (
            [] if analyzed_parameters is None else analyzed_parameters)
        self._n_gpus = n_gpus
        self._initialized = False

    def get_sim_specs(self, varying_parameters, objectives):
        if not self._initialized:
            raise RuntimeError(
                'Evaluator must be initialized before generating sim_specs')
        sim_specs = {
            # Function whose output is being minimized.
            'sim_f': self.sim_function,
            # Name of input for sim_f, that LibEnsemble is allowed to modify.
            # May be a 1D array.
            'in': [var.name for var in varying_parameters],
            'out': (
                [(obj.name, float) for obj in objectives]
                # f is the single float output that LibEnsemble minimizes.
                + [(par.name, par.type) for par in self._analyzed_parameters]
                # input parameters
                + [(var.name, float) for var in varying_parameters]
            ),
            'user': {
                'n_gpus': self._n_gpus,
            }
        }
        return sim_specs

    def get_libe_specs(self):
        libE_specs = {}
        return libE_specs

    def initialize(self):
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def _initialize(self):
        pass
