"""Contains the definition of dummy generator that raises an import error."""


class AxImportErrorDummyGenerator(object):
    """Class that raises an error when instantiated, telling the user to install ax-platform.

    This class replaces all other Ax-based classes,
    when Ax is not installed
    """

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "You need to install ax-platform, in order "
            "to use Ax-based generators in optimas.\n"
            "e.g. with `pip install ax-platform >= 0.3.4`"
        )
