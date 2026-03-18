"""Contains the definition of the multi-fidelity Ax generator."""

from typing import List, Optional, Dict

from botorch.acquisition.knowledge_gradient import (
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.input_constructors import (
    ACQF_INPUT_CONSTRUCTOR_REGISTRY,
)
from ax.generation_strategy.generation_node import GenerationStep
from ax.adapter.registry import Generators

from .base import AxServiceGenerator
from gest_api.vocs import VOCS

# Workaround for BoTorch bug: X_pending is not in the allowed variable
# kwargs for construct_inputs_qMFKG, but Ax always passes it. KG-based
# acquisition functions handle pending points via fantasization, so
# X_pending can be safely ignored.
_original_constructor = ACQF_INPUT_CONSTRUCTOR_REGISTRY[
    qMultiFidelityKnowledgeGradient
]


def _patched_constructor(*args, **kwargs):
    kwargs.pop("X_pending", None)
    return _original_constructor(*args, **kwargs)


ACQF_INPUT_CONSTRUCTOR_REGISTRY[qMultiFidelityKnowledgeGradient] = (
    _patched_constructor
)


class AxMultiFidelityGenerator(AxServiceGenerator):
    """Multifidelity Bayesian optimization using the Ax service API.

    Parameters
    ----------
    vocs : VOCS
        VOCS object defining variables, objectives, constraints, and observables.
        One of the variables should be a fidelity parameter.

    n_init : int, optional
        Number of evaluations to perform during the initialization phase using
        Sobol sampling. If external data is attached to the exploration, the
        number of initialization evaluations will be reduced by the same
        amount, unless `enforce_n_init=True`. By default, ``4``.
    enforce_n_init : bool, optional
        Whether to enforce the generation of `n_init` Sobol trials, even if
        external data is supplied. By default, ``False``.
    abandon_failed_trials : bool, optional
        Whether failed trials should be abandoned (i.e., not suggested again).
        By default, ``True``.
    fit_out_of_design : bool, optional
        Whether to fit the surrogate model taking into account evaluations
        outside of the range of the varying parameters. This can be useful
        if the range of parameter has been reduced during the optimization.
        By default, False.
    fidel_cost_intercept : float, optional
        The cost intercept for the affine cost of the form
        `cost_intercept + n`, where `n` is the number of generated points.
        Used for the knowledge gradient acquisition function. By default, 1.
    use_cuda : bool, optional
        Whether to allow the generator to run on a CUDA GPU. By default
        ``False``.
    gpu_id : int, optional
        The ID of the GPU in which to run the generator. By default, ``0``.
    dedicated_resources : bool, optional
        Whether to allocated dedicated resources (e.g., the GPU) for the
        generator. These resources will not be available to the
        simulation workers. By default, ``False``.
    save_model : bool, optional
        Whether to save the optimization model (in this case, the Ax client) to
        disk. By default ``True``.
    model_save_period : int, optional
        Periodicity, in number of evaluated Trials, with which to save the
        model to disk. By default, ``5``.
    model_history_dir : str, optional
        Name of the directory in which the model will be saved. By default,
        ``'model_history'``.

    """

    def __init__(
        self,
        vocs: VOCS,
        n_init: Optional[int] = 4,
        enforce_n_init: Optional[bool] = False,
        abandon_failed_trials: Optional[bool] = True,
        fit_out_of_design: Optional[bool] = False,
        fidel_cost_intercept: Optional[float] = 1.0,
        use_cuda: Optional[bool] = False,
        gpu_id: Optional[int] = 0,
        dedicated_resources: Optional[bool] = False,
        save_model: Optional[bool] = True,
        model_save_period: Optional[int] = 5,
        model_history_dir: Optional[str] = "model_history",
    ) -> None:
        self.fidel_cost_intercept = fidel_cost_intercept
        super().__init__(
            vocs=vocs,
            n_init=n_init,
            enforce_n_init=enforce_n_init,
            abandon_failed_trials=abandon_failed_trials,
            fit_out_of_design=fit_out_of_design,
            use_cuda=use_cuda,
            gpu_id=gpu_id,
            dedicated_resources=dedicated_resources,
            save_model=save_model,
            model_save_period=model_save_period,
            model_history_dir=model_history_dir,
        )

    def _create_generation_steps(
        self, bo_model_kwargs: Dict
    ) -> List[GenerationStep]:
        """Create generation steps for multifidelity optimization."""
        # Add acquisition function to generator kwargs.
        bo_model_kwargs["botorch_acqf_class"] = qMultiFidelityKnowledgeGradient

        # Make generation strategy.
        steps = []

        # Add Sobol initialization with `n_init` random trials.
        steps.append(self._create_sobol_step())

        # Continue indefinitely with GPKG.
        bo_model_kwargs["botorch_acqf_options"] = {
            "cost_intercept": self.fidel_cost_intercept,
        }
        steps.append(
            GenerationStep(
                generator=Generators.BOTORCH_MODULAR,
                num_trials=-1,
                generator_kwargs=bo_model_kwargs,
            ),
        )

        return steps

    def set_fidelity_param(
        self,
        var_name: str,
        fidelity_target_value: float = None,
    ) -> None:
        """Set a parameter as the fidelity parameter for multi-fidelity optimization.

        Parameters
        ----------
        var_name : str
            Name of the variable to set as fidelity parameter.
        fidelity_target_value : float, optional
            The target fidelity value for optimization.
        """
        var = None
        for vp in self._varying_parameters:
            if vp.name == var_name:
                var = vp
                break

        if var is None:
            raise ValueError(
                f"Variable '{var_name}' not found in varying parameters"
            )

        var.is_fidelity = True
        if fidelity_target_value is not None:
            var.fidelity_target_value = fidelity_target_value

        # Update the Ax client
        self._update_parameter(var)
