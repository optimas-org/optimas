"""
This module defines a base class for all classes that have a name attribute.
Examples of these are the different optimization parameters and tasks.
"""
import json

from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder


@dataclass
class NamedBase():
    """Base class for all classes with a ``name`` attribute.

    Parameters
    ----------
    name : str
        The name to assign.
    """
    name: str

    def json(self):
        return json.dumps(self, indent=4, default=pydantic_encoder)
