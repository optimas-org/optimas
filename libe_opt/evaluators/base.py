class Evaluator:
    def __init__(self, analyzed_params=None, n_gpus=1):
        self.analyzed_params = [] if analyzed_params is None else analyzed_params
        self.n_gpus = n_gpus
        self._initialized= False

    def initialize(self):
        if not self._initialized:
            self._initialize()
            self._initialized = True        
        
    def _initialize(self):
        pass
