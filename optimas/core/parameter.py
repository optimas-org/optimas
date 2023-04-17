"""Contains the definition of the various optimization parameters."""

from typing import Optional

import numpy as np

from .base import NamedBase


class Parameter(NamedBase):
    """Base class for all optimization parameters.

    Parameters
    ----------
    name : str
        Name of the parameter.
    dtype : np.dtype
        The data type of the parameter.
    """
    def __init__(
        self,
        name: str,
        dtype: Optional[np.dtype] = float
    ):
        super().__init__(name)
        self._dtype = dtype

    @property
    def dtype(self) -> np.dtype:
        """Get parameter data type."""
        return self._dtype


class VaryingParameter(Parameter):
    """Defines an input parameter to be varied during optimization.

    Parameters
    ----------
    name : str
        The name of the parameter.
    lower_bound, upper_bound : float
        Lower and upper bounds of the range in which the parameter can vary.
    is_fidelity : bool, optional
        Indicates whether the parameter is a fidelity. Only needed for
        multifidelity optimization.
    fidelity_target_value : float, optional
        The target value of the fidelity. Only needed for multifidelity
        optimization.
    default_value : float, optional
        Default value of the parameter when it is not being varied. Only needed
        for some generators.
    """
    def __init__(
        self,
        name: str,
        lower_bound: float,
        upper_bound: float,
        is_fidelity: Optional[bool] = False,
        fidelity_target_value: Optional[float] = None,
        default_value: Optional[float] = None,
        dtype: Optional[np.dtype] = float
    ) -> None:
        super().__init__(name, dtype)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._is_fidelity = is_fidelity
        self._fidelity_target_value = fidelity_target_value
        self._default_value = default_value

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    @property
    def upper_bound(self) -> float:
        return self._upper_bound

    @property
    def is_fidelity(self) -> bool:
        return self._is_fidelity

    @property
    def fidelity_target_value(self) -> float:
        return self._fidelity_target_value

    @property
    def default_value(self) -> float:
        return self._default_value


class TrialParameter(Parameter):
    """Defines a parameter that can be attached to a trial.

    Parameters
    ----------
    name : str
        Name of the parameter.
    save_name : str
        Name under which the parameter should be saved to the history array. If
        not given, the parameter ``name`` will be used.
    dtype : np.dtype
        The data type of the parameter.
    """
    def __init__(
        self,
        name: str,
        save_name: Optional[str] = None,
        dtype: Optional[np.dtype] = float
    ) -> None:
        super().__init__(name, dtype=dtype)
        self._save_name = name if save_name is None else save_name

    @property
    def save_name(self) -> str:
        return self._save_name


class Objective(Parameter):
    """Defines an optimization objective.

    Parameters
    ----------
    name : str, optional
        Name of the objective. By default ``'f'``.
    minimize : bool, optional
        Indicates whether the objective should be minimized or,
        otherwise, maximized. By default, ``True``.
    """
    def __init__(
        self,
        name: Optional[str] = 'f',
        minimize: Optional[bool] = True
    ) -> None:
        super().__init__(name)
        self._minimize = minimize

    @property
    def minimize(self) -> bool:
        return self._minimize
