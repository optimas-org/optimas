"""
This module defines a base class for all classes that have a name attribute.
Examples of these are the different optimization parameters and tasks.
"""

from pydantic import BaseModel


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
