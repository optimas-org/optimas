"""Contains the definition of the various optimization parameters."""

from typing import Optional

import numpy as np
from pydantic.dataclasses import dataclass

from .base import NamedBase


@dataclass
class Parameter(NamedBase):
    """Base class for all optimization parameters.

    Parameters
    ----------
    name : str
        Name of the parameter.
    dtype : np.dtype
        The data type of the parameter.
    """
    dtype: Optional[np.dtype] = float


@dataclass
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
    lower_bound: float
    upper_bound: float
    is_fidelity: Optional[bool] = False
    fidelity_target_value: Optional[float] = None
    default_value: Optional[float] = None


@dataclass
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
    save_name: Optional[str] = None


@dataclass
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
    name: Optional[str] = 'f',
    minimize: Optional[bool] = True
