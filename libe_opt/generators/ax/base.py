import torch

from libe_opt.generators.base import Generator


class AxGenerator(Generator):
    def __init__(self, varying_parameters, objectives, use_cuda=False,
                 save_model=False, model_save_period=5,
                 model_history_dir='model_history',
                 custom_trial_parameters=None):
        super().__init__(varying_parameters, objectives, use_cuda=use_cuda,
                         save_model=save_model,
                         model_save_period=model_save_period,
                         model_history_dir=model_history_dir,
                         custom_trial_parameters=custom_trial_parameters)
        self._determine_torch_device()

    def _determine_torch_device(self):
        # Use CUDA if available.
        if self.use_cuda and torch.cuda.is_available():
            self.torch_device = 'cuda'
        else:
            self.torch_device = 'cpu'
