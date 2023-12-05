"""Contains the definition of the ``Evaluation`` class."""

from typing import Optional

from .parameter import Parameter


class Evaluation:
    """Class used to store the evaluation of a parameter.

    The evaluation consists of the observed value and the observation noise.

    Parameters
    ----------
    parameter : Parameter
        The parameter that has been evaluated.
    value : float
        The observed value of the evaluation.
    sem : float, optional
        The observation noise of the evaluation.
    """

    def __init__(
        self, parameter: Parameter, value: float, sem: Optional[float] = None
    ) -> None:
        self._parameter = parameter
        self._value = value
        self._sem = sem

    @property
    def parameter(self) -> Parameter:
        """Get the evaluated parameter."""
        return self._parameter

    @property
    def value(self) -> float:
        """Get the evaluation value."""
        return self._value

    @property
    def sem(self) -> float:
        """Get the evaluation noise."""
        return self._sem
