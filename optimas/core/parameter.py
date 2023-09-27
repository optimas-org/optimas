"""Contains the definition of the various optimization parameters."""

from typing import Optional, Any
import json

from pydantic import BaseModel, Extra, validator
import numpy as np


def json_dumps_dtype(v, *, default):
    """Add support for dumping numpy dtype to json."""
    for key, value in v.items():
        if key == 'dtype':
            v[key] = np.dtype(value).descr
    return json.dumps(v)


class Parameter(BaseModel):
    """Base class for all optimization parameters.

    Parameters
    ----------
    name : str
        Name of the parameter.
    dtype : data-type
        The data type of the parameter. Any object that can be converted to a
        numpy dtype.
    """
    name: str
    dtype: Optional[Any]

    def __init__(
        self,
        name: str,
        dtype: Optional[Any] = float,
        **kwargs
    ) -> None:
        super().__init__(name=name, dtype=dtype, **kwargs)

    @validator("dtype", pre=True)
    def check_valid_out(cls, v):
        try:
            _ = np.dtype(v)
        except TypeError:
            raise ValueError(f"Unable to coerce '{v}' into a NumPy dtype.")
        else:
            return v

    class Config:
        extra = Extra.ignore
        json_dumps = json_dumps_dtype


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

    def __init__(
        self,
        name: str,
        lower_bound: float,
        upper_bound: float,
        is_fidelity: Optional[bool] = False,
        fidelity_target_value: Optional[float] = None,
        default_value: Optional[float] = None,
        dtype: Optional[Any] = float
    ) -> None:
        super().__init__(
            name=name,
            dtype=dtype,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            is_fidelity=is_fidelity,
            fidelity_target_value=fidelity_target_value,
            default_value=default_value
        )


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

    def __init__(
        self,
        name: str,
        save_name: Optional[str] = None,
        dtype: Optional[Any] = float
    ) -> None:
        super().__init__(name=name, save_name=save_name, dtype=dtype)
        self.save_name = name if save_name is None else save_name


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

    def __init__(
        self,
        name: Optional[str] = 'f',
        minimize: Optional[bool] = True
    ) -> None:
        super().__init__(name=name, minimize=minimize)
