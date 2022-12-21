from .base import NamedBase


class Parameter(NamedBase):
    def __init__(self, name, dtype=float):
        super().__init__(name)
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype


class VaryingParameter(Parameter):
    def __init__(self, name, lower_bound, upper_bound, is_fidelity=False,
                 fidelity_target_value=None, default_value=None, dtype=float):
        super().__init__(name, dtype)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._is_fidelity = is_fidelity
        self._fidelity_target_value = fidelity_target_value
        self._default_value = default_value

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def is_fidelity(self):
        return self._is_fidelity

    @property
    def fidelity_target_value(self):
        return self._fidelity_target_value

    @property
    def default_value(self):
        return self._default_value

class TrialParameter(Parameter):
    def __init__(self, name, save_name=None, dtype=float):
        super().__init__(name, dtype=dtype)
        self._save_name = name if save_name is None else save_name

    @property
    def save_name(self):
        return self._save_name
