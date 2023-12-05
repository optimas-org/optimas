"""Defines a base class for all parameters that have a name."""


class NamedBase:
    """Base class for all classes with a ``name`` attribute.

    Parameters
    ----------
    name : str
        The name to assign.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Get name."""
        return self._name
