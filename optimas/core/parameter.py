"""Contains the definition of the various optimization parameters.

All parameters are Pydantic BaseModels, but include the __init__ definition
to allow for positional arguments (otherwise the BaseModels only take
keyword arguments). This is needed for backward compatibility.
"""

from typing import Optional, Any

from pydantic import BaseModel, field_serializer, validator, PrivateAttr
import numpy as np


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
        self, name: str, dtype: Optional[Any] = float, **kwargs
    ) -> None:
        super().__init__(name=name, dtype=dtype, **kwargs)

    @validator("dtype", pre=True)
    def _check_valid_dtype(cls, v):
        """Check that the given dtype can be converted to a numpy dtype."""
        try:
            # For dtypes that were serialized with `json_dumps_dtype`.
            if isinstance(v, list):
                v = v[0][1]
            # Check that the given dtype can be converted to a numpy dtype.
            _ = np.dtype(v)
        except TypeError:
            raise ValueError(f"Unable to coerce '{v}' into a NumPy dtype.")
        else:
            return v

    @field_serializer("dtype")
    def _serialize_dtype(self, value, _info):
        """Add support for dumping numpy dtype to json."""
        return np.dtype(value).descr


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
    dtype : data-type
        The data type of the parameter. Any object that can be converted to a
        numpy dtype.
    """

    lower_bound: float
    upper_bound: float
    is_fidelity: Optional[bool] = False
    fidelity_target_value: Optional[float] = None
    default_value: Optional[float] = None
    _is_fixed: bool = PrivateAttr(False)

    def __init__(
        self,
        name: str,
        lower_bound: float,
        upper_bound: float,
        is_fidelity: Optional[bool] = False,
        fidelity_target_value: Optional[float] = None,
        default_value: Optional[float] = None,
        dtype: Optional[Any] = float,
    ) -> None:
        super().__init__(
            name=name,
            dtype=dtype,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            is_fidelity=is_fidelity,
            fidelity_target_value=fidelity_target_value,
            default_value=default_value,
        )

    @property
    def is_fixed(self) -> bool:
        """Get whether the parameter is fixed to a certain value."""
        return self._is_fixed

    def update_range(self, lower_bound: float, upper_bound: float) -> None:
        """Update range of the parameter.

        Parameters
        ----------
        lower_bound, upper_bound : float
            Lower and upper bounds of the range in which the parameter can vary.
        """
        self._check_range(lower_bound, upper_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def fix_value(self, value: float) -> None:
        """Fix the value of the parameter.

        The value must be within the range of the parameter.

        Parameters
        ----------
        value : float
            The value to which the parameter will be fixed.
        """
        if value < self.lower_bound or value > self.upper_bound:
            raise ValueError(
                f"The value {value} is outside of the range of parameter "
                f"{self.name}: [{self.lower_bound},{self.upper_bound}]"
            )
        self.default_value = value
        self._is_fixed = True

    def free_value(self) -> None:
        """Free the value of the parameter."""
        self._is_fixed = False

    def _check_range(self, lower_bound, upper_bound):
        # TODO, turn this into a pydantic validator.
        if upper_bound <= lower_bound:
            raise ValueError(
                "Inconsistent range bounds. "
                f"Upper bound ({upper_bound}) < lower bound ({lower_bound})."
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
    dtype : data-type
        The data type of the parameter.
    """

    save_name: Optional[str] = None

    def __init__(
        self,
        name: str,
        save_name: Optional[str] = None,
        dtype: Optional[Any] = float,
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
    dtype : data-type
        The data type of the parameter. Any object that can be converted to a
        numpy dtype.
    """

    minimize: Optional[bool] = True

    def __init__(
        self,
        name: Optional[str] = "f",
        minimize: Optional[bool] = True,
        dtype: Optional[Any] = float,
    ) -> None:
        super().__init__(name=name, minimize=minimize, dtype=dtype)
