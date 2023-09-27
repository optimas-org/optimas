"""
This module defines a base class for all classes that have a name attribute.
Examples of these are the different optimization parameters and tasks.
"""
import json

from pydantic import BaseModel, Extra
import numpy as np


def json_dumps_dtype(v, *, default):
    """Add support for dumping numpy dtype to json."""
    for key, value in v.items():
        if key == 'dtype':
            v[key] = np.dtype(value).descr
    return json.dumps(v)


class NamedBase(BaseModel):
    """Base class for all classes with a ``name`` attribute.

    Parameters
    ----------
    name : str
        The name to assign.
    """
    name: str

    def __init__(
            self,
            name: str,
            **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
    
    class Config:
        extra = Extra.ignore
        json_dumps = json_dumps_dtype
