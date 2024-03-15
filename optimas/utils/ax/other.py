"""Contains the definition of various utilities for using Ax."""

from typing import List, Dict

import numpy as np
from ax.service.utils.instantiation import ObjectiveProperties

from optimas.core import VaryingParameter, Objective


def convert_optimas_to_ax_parameters(
    varying_parameters: List[VaryingParameter],
) -> List[Dict]:
    """Create list of Ax parameters from optimas varying parameters."""
    parameters = []
    for var in varying_parameters:
        # Determine parameter type.
        value_dtype = np.dtype(var.dtype)
        if value_dtype.kind == "f":
            value_type = "float"
        elif value_dtype.kind == "i":
            value_type = "int"
        else:
            raise ValueError(
                "Ax range parameter can only be of type 'float'ot 'int', "
                f"not {var.dtype}."
            )
        # Create parameter dict and append to list.
        parameters.append(
            {
                "name": var.name,
                "type": "range",
                "bounds": [var.lower_bound, var.upper_bound],
                "is_fidelity": var.is_fidelity,
                "target_value": var.fidelity_target_value,
                "value_type": value_type,
            }
        )
    return parameters


def convert_optimas_to_ax_objectives(
    objectives: List[Objective],
) -> Dict[str, ObjectiveProperties]:
    """Create list of Ax objectives from optimas objectives."""
    ax_objectives = {}
    for obj in objectives:
        ax_objectives[obj.name] = ObjectiveProperties(minimize=obj.minimize)
    return ax_objectives
