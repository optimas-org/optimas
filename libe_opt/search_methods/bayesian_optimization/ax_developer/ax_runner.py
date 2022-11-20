import numpy as np
from ax import Runner
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.message_numbers import EVAL_GEN_TAG


class AxRunner(Runner):
    """ Custom runner in charge of executing the trials using libEnsemble. """

    def __init__(self, libE_info, gen_specs):
        self.libE_info = libE_info
        self.gen_specs = gen_specs
        self.ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
        super().__init__()

    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        task = trial.trial_type
        number_of_gen_points = len(trial.arms)
        H_o = np.zeros(number_of_gen_points, dtype=self.gen_specs['out'])

        for i, (arm_name, arm) in enumerate(trial.arms_by_name.items()):
            # fill H_o
            params = arm.parameters
            n_param = len(params)
            param_array = np.zeros(n_param)
            for j in range(n_param):
                param_array[j] = params['x{}'.format(j)]
            H_o['x'][i] = param_array
            H_o['resource_sets'][i] = 1
            H_o['task'][i] = task

        tag, Work, calc_in = self.ps.send_recv(H_o)

        trial_metadata['tag'] = tag
        for i, (arm_name, arm) in enumerate(trial.arms_by_name.items()):
            # fill metadata
            params = arm.parameters
            trial_metadata[arm_name] = {
                "arm_name": arm_name,
                "trial_index": trial.index,
                "f": calc_in['f'][i] if calc_in is not None else None
            }
        return trial_metadata
