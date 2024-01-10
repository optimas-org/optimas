# Import Ax generators
try:
    from .ax.service.single_fidelity import AxSingleFidelityGenerator
    from .ax.service.multi_fidelity import AxMultiFidelityGenerator
    from .ax.service.ax_client import AxClientGenerator
    from .ax.developer.multitask import AxMultitaskGenerator
except ImportError as e:
    if e.__str__() == "No module named 'ax'":
        # Replace generators by dummy generators that will
        # raise an error only if the user tries to instantiate them
        # and tell them to install ax-platform
        from .ax.import_error_dummy_generator import AxImportErrorDummyGenerator

        AxSingleFidelityGenerator = AxImportErrorDummyGenerator
        AxMultiFidelityGenerator = AxImportErrorDummyGenerator
        AxClientGenerator = AxImportErrorDummyGenerator
        AxMultitaskGenerator = AxImportErrorDummyGenerator
    else:
        raise (e)

# Import optimas native generators
from .grid_sampling import GridSamplingGenerator
from .line_sampling import LineSamplingGenerator
from .random_sampling import RandomSamplingGenerator


__all__ = [
    "AxSingleFidelityGenerator",
    "AxMultiFidelityGenerator",
    "AxMultitaskGenerator",
    "AxClientGenerator",
    "GridSamplingGenerator",
    "LineSamplingGenerator",
    "RandomSamplingGenerator",
]
