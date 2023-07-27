from typing import Any


def update_object(
    object_old: Any,
    object_new: Any
) -> None:
    """Update the attributes of an object with those from a newer one.

    This method is intended to be used with objects of the same type and that
    have a `__dict__` attribute.

    Parameters
    ----------
    object_old : Any
        The object to be updated.
    object_new : Any
        The object from which to get the updated attributes.
    """
    assert isinstance(object_new, type(object_old)), "Object types don't match"
    for key, value in vars(object_new).items():
        setattr(object_old, key, value)
