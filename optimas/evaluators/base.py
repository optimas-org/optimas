"""Contains the definition of the base Evaluator class."""

from typing import Callable, Optional, Dict, List

from optimas.core import VaryingParameter, Objective, Parameter


class Evaluator:
    """Base class for all evaluators.

    Parameters
    ----------
    sim_function : callable
        The simulation function (as defined in libEnsemble) to be used for
        carrying out the evaluations.
    n_procs : int, optional
        The number of processes that will be used for each evaluation. By
        default, ``n_procs=1`` if ``n_gpus`` is not given. Otherwise, the
        default behavior is to match the number of processes to the number
        of GPUs, i.e., ``n_procs=n_gpus``.
    n_gpus : int, optional
        The number of GPUs that will be made available for each evaluation. By
        default, 0.
    """
    def __init__(
        self,
        sim_function: Callable,
        n_procs: Optional[int] = None,
        n_gpus: Optional[int] = None
    ) -> None:
        self.sim_function = sim_function
        # If no resources are specified, use 1 CPU an 0 GPUs.
        if n_procs is None and n_gpus is None:
            n_procs = 1
            n_gpus = 0
        # If `n_gpus` is given without specifying `n_procs`, match processes
        # to GPUs.
        elif n_procs is None:
            n_procs = n_gpus
        # If `n_procs` is given without specifying `n_gpus`, do not use GPUs.
        elif n_gpus is None:
            n_gpus = 0
        self._n_procs = n_procs
        self._n_gpus = n_gpus
        self._initialized = False

    def get_sim_specs(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        analyzed_parameters: List[Parameter],
    ) -> Dict:
        """Get a dictionary with the ``sim_specs`` as expected
        by ``libEnsemble``
        """
        # Only generate sim_specs if evaluator has been initialized.
        if not self._initialized:
            raise RuntimeError(
                'Evaluator must be initialized before generating sim_specs')

        # Create sim_specs.
        sim_specs = {
            # Function whose output is being minimized.
            'sim_f': self.sim_function,
            # Name of input for sim_f, that LibEnsemble is allowed to modify.
            # May be a 1D array.
            'in': [var.name for var in varying_parameters],
            'out': (
                [(obj.name, float) for obj in objectives]
                # f is the single float output that LibEnsemble minimizes.
                + [(par.name, par.dtype) for par in analyzed_parameters]
                # input parameters
                + [(var.name, float) for var in varying_parameters]
            ),
            'user': {
                'n_procs': self._n_procs,
                'n_gpus': self._n_gpus,
            }
        }
        return sim_specs

    def get_libe_specs(self) -> Dict:
        """Get a dictionary with the ``libE_specs`` as expected
        by ``libEnsemble``
        """
        libE_specs = {}
        return libE_specs

    def get_run_params(self) -> Dict:
        """Return run parameters for this evaluator."""
        run_params = {
            'num_procs': self._n_procs,
            'num_gpus': self._n_gpus
        }
        return run_params

    def initialize(self) -> None:
        """Initialize the evaluator."""
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def _initialize(self) -> None:
        """Method that executes the code required to initialize the evaluator.
        Has to be implemented by the subclasses.
        """
        pass
