import torch

from libe_opt.generators.base import Generator


class AxGenerator(Generator):
    def __init__(self, variables, objectives, use_cuda=False,
                 custom_trial_metadata=None):
        super().__init__(variables, objectives, use_cuda=use_cuda,
                         custom_trial_metadata=custom_trial_metadata)
        self._determine_torch_device()

    def _determine_torch_device(self):
        # Use CUDA if available.
        if self.use_cuda and torch.cuda.is_available():
            self.torch_device = 'cuda'
        else:
            self.torch_device = 'cpu'
