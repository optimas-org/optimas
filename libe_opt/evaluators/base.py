class Evaluator:
    def __init__(self):
        self._initialized= False

    def initialize(self):
        if not self._initialized:
            self._initialize()
            self._initialized = True        
        
    def _initialize(self):
        pass
