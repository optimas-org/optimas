"""
This module defines a base class for all classes that have a name attribute.
Examples of these are the different optimization parameters and tasks.
"""

from pydantic.dataclasses import dataclass


@dataclass
class NamedBase():
    """Base class for all classes with a ``name`` attribute.

    Parameters
    ----------
    name : str
        The name to assign.
    """
    name: str

