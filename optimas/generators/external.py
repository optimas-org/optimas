"""Contains the definition of ExternalGenerator.

This module provides a wrapper that integrates third-party generators
implementing the ``gest-api`` generator standard
(https://github.com/campa-consortium/gest-api) into Optimas.
"""

from .base import Generator


class ExternalGenerator(Generator):
    """Wrap a third-party generator that follows the ``gest-api`` standard.

    https://github.com/campa-consortium/gest-api

    Any external generator that implements this interface can be used inside
    optimas by wrapping it in ``ExternalGenerator``.

    Known libraries containing generators compatible with this interface include
    `Xopt <https://github.com/xopt-org/Xopt>`_ and `libEnsemble
    <https://github.com/Libensemble/libensemble>`_.

    Parameters
    ----------
    ext_gen : object
        An object implementing/sub-classing the ``gest-api`` generator interface. The
        external generator should be fully configured (including any initial
        data ingested) before being passed here. The external library itself
        must be installed separately.
    **kwargs
        Additional keyword arguments forwarded to the base
        :class:`~optimas.generators.Generator` (e.g., ``vocs``,
        ``save_model``).

    Examples
    --------
    Using a generic ``gest-api``-compatible generator:

    .. code-block:: python

        from optimas.generators import ExternalGenerator
        from gest_api.vocs import VOCS
        from some_library import SomeGenerator

        vocs = VOCS(
            variables={"x1": [0.0, 1.0], "x2": [0.0, 10.0]},
            objectives={"y1": "MINIMIZE"},
        )

        ext_gen = SomeGenerator(vocs=vocs)
        gen = ExternalGenerator(ext_gen=ext_gen, vocs=vocs)

    Using an `Xopt <https://github.com/xopt-org/Xopt>`_ generator:

    .. code-block:: python

        from optimas.generators import ExternalGenerator
        from optimas.evaluators import FunctionEvaluator
        from optimas.explorations import Exploration
        from gest_api.vocs import VOCS
        from xopt.generators.bayesian.expected_improvement import (
            ExpectedImprovementGenerator,
        )

        vocs = VOCS(
            variables={"x1": [0.0, 1.0], "x2": [0.0, 10.0]},
            objectives={"y1": "MINIMIZE"},
        )

        # Create and (optionally) pre-seed the external generator.
        ext_gen = ExpectedImprovementGenerator(vocs=vocs)
        ext_gen.ingest([{"x1": 0.5, "x2": 5.0, "y1": 5.0}])

        # Wrap it for use with optimas.
        gen = ExternalGenerator(ext_gen=ext_gen, vocs=vocs)

        ev = FunctionEvaluator(function=my_function)
        exp = Exploration(generator=gen, evaluator=ev, max_evals=20, sim_workers=4)
        exp.run()
    """

    def __init__(
        self,
        ext_gen,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.gen = ext_gen

    def suggest(self, n_trials):
        """Request the next set of points to evaluate.

        Delegates to the wrapped generator's ``suggest`` method.
        """
        return self.gen.suggest(n_trials)

    def ingest(self, trials):
        """Send the results of evaluations to the generator.

        Delegates to the wrapped generator's ``ingest`` method.
        """
        self.gen.ingest(trials)
