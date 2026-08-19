"""Contains the definition of ExternalGenerator.

This module provides a wrapper that integrates third-party generators
implementing the ``gest-api`` generator standard
(https://github.com/campa-consortium/gest-api) into Optimas.
"""

import warnings

from gest_api.generator import Generator as StandardGenerator

from .base import Generator


class _ExternalGeneratorAdapter(Generator):
    """Adapt a ``gest-api`` generator to the Optimas generator protocol.

    The adapter supplies Optimas trial bookkeeping while delegating the
    standardized ``suggest`` and ``ingest`` operations to the wrapped object.
    """

    def __init__(
        self,
        ext_gen,
        vocs=None,
        **kwargs,
    ):
        if not isinstance(ext_gen, StandardGenerator):
            raise TypeError(
                "ext_gen must implement the gest-api Generator interface."
            )
        if vocs is None:
            try:
                vocs = ext_gen.vocs
            except AttributeError as exc:
                raise ValueError(
                    "The external generator must expose a `vocs` attribute, "
                    "or `vocs` must be provided explicitly."
                ) from exc
        super().__init__(vocs=vocs, **kwargs)
        self.gen = ext_gen

    def suggest(self, n_trials):
        """Request the next set of points to evaluate."""
        return self.gen.suggest(n_trials)

    def ingest(self, trials):
        """Send the results of evaluations to the generator."""
        self.gen.ingest(trials)


def adapt_generator(generator, **kwargs):
    """Adapt a gest-api generator to the Optimas generator protocol."""
    if isinstance(generator, Generator):
        return generator
    if isinstance(generator, StandardGenerator):
        return _ExternalGeneratorAdapter(generator, **kwargs)
    raise TypeError(
        "generator must be an Optimas Generator or implement the gest-api "
        "Generator interface."
    )


class ExternalGenerator(_ExternalGeneratorAdapter):
    """Deprecated compatibility wrapper for a gest-api generator.

    Pass a gest-api generator directly to :class:`~optimas.explorations.Exploration`
    instead. This class remains available for compatibility and supports the
    Optimas-specific generator options accepted by the previous wrapper.
    """

    def __init__(self, ext_gen, **kwargs):
        warnings.warn(
            "ExternalGenerator is deprecated. Pass the gest-api generator "
            "directly to Exploration instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(ext_gen=ext_gen, **kwargs)
